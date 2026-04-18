use rx888_stream::fx3;
use rx888_stream::rx888;

use std::{
    fs::File,
    path::PathBuf,
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};

use bytemuck::cast_slice;
use clap::Parser;
use rusb::{Context, UsbContext};
use rusb_async::TransferPool;
use rustfft::{FftPlanner, num_complex::Complex};
use rayon::prelude::*;

use rx888::{
    rx888_send_argument, rx888_send_command, rx888_send_command_u64, ArgumentList, FX3Command,
    GPIOPin,
};

const FX3_VID: u16 = 0x04b4;
const FX3_BOOTLOADER_PID: u16 = 0x00f3;
const FX3_FIRMWARE_PID_1: u16 = 0x00f1;
const FX3_FIRMWARE_PID_2: u16 = 0x3ddc;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    firmware: PathBuf,

    #[arg(short, long, default_value_t = 127500000)]
    center_freq: u64,

    #[arg(short, long, default_value_t = 32000000)]
    sample_rate: u32,

    #[arg(short, long, default_value_t = 60)]
    gain: u8,

    #[arg(short, long, default_value_t = -20.0)]
    threshold_db: f32,
}

fn main() {
    let args = Cli::parse();
    let context = Context::new().expect("Could not create USB context");

    // 1. Load Firmware
    println!("[*] Loading firmware: {:?}", args.firmware);
    // 1. Reset to Bootloader if in firmware mode
    for pid in [FX3_FIRMWARE_PID_1, FX3_FIRMWARE_PID_2] {
        if let Some(handle) = context.open_device_with_vid_pid(FX3_VID, pid) {
            println!("[*] Found RX888 in firmware mode (PID {:04x}), resetting...", pid);
            if pid == FX3_FIRMWARE_PID_2 {
                // libsddc (3ddc) firmware reset command
                let _ = handle.write_control(
                    0x40, // Vendor Out
                    0x01, // REGOP
                    0,    // Register 0 (RESET)
                    0,    // Value
                    &0u32.to_le_bytes(),
                    Duration::from_secs(1)
                );
            } else {
                // rx888_stream (00f1) firmware reset command
                let _ = rx888_send_command(&handle, FX3Command::RESETFX3, 0);
            }
            thread::sleep(Duration::from_millis(2000));
            break;
        }
    }

    let handle = open_device_with_timeout(&context, FX3_VID, FX3_BOOTLOADER_PID, Duration::from_secs(5))
        .expect("Could not find RX888 bootloader (00f3). Try re-plugging the device.");
    
    let mut fw_file = File::open(&args.firmware).expect("Could not open firmware");
    fx3::fx3_load_ram(handle, &mut fw_file).expect("Firmware load failed");
    thread::sleep(Duration::from_millis(1000));

    // 2. Open Device after loading
    let handle = open_device_with_timeout(&context, FX3_VID, FX3_FIRMWARE_PID_1, Duration::from_secs(5))
        .expect("Could not find RX888 after firmware load (00f1)");
    
    handle.claim_interface(0).expect("Could not claim USB interface");

    // 3. Configure Hardware for Airband
    println!("[*] Initializing VHF Tuner at {} MHz...", args.center_freq as f64 / 1e6);
    
    let gpio = GPIOPin::VHF_EN as u32 | GPIOPin::PGA_EN as u32;
    rx888_send_command(&handle, FX3Command::TUNERINIT, 0).unwrap();
    rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, args.center_freq).unwrap();
    rx888_send_command(&handle, FX3Command::GPIOFX3, gpio).unwrap();
    
    // Set Gains (LNA=20, VGA=12 are good defaults for airband)
    rx888_send_argument(&handle, ArgumentList::R82XX_ATTENUATOR, 20).unwrap();
    rx888_send_argument(&handle, ArgumentList::R82XX_VGA, 12).unwrap();
    rx888_send_argument(&handle, ArgumentList::AD8340_VGA, (args.gain | 0x80) as u16).unwrap();

    // Start Streaming
    rx888_send_command(&handle, FX3Command::STARTADC, args.sample_rate).unwrap();
    rx888_send_command(&handle, FX3Command::STARTFX3, 0).unwrap();

    let handle = Arc::new(handle);
    let mut transfer_pool = TransferPool::new(handle.clone()).unwrap();
    for _ in 0..32 {
        transfer_pool.submit_bulk(0x81, Vec::with_capacity(131072)).unwrap();
    }

    // 4. Processing Loop (FFT)
    let fft_size = 4096;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::SeqCst)).ok();

    println!("[*] Scanning 118-137 MHz... Press Ctrl+C to stop.");

    while running.load(Ordering::SeqCst) {
        let data = transfer_pool.poll(Duration::from_secs(1)).expect("USB Timeout");
        let samples: &[i16] = cast_slice(&data);
        
        // Process in chunks of fft_size (complex IQ)
        // Note: RX888 outputs real samples or interleaved IQ depending on mode.
        // In VHF mode it outputs interleaved IQ.
        let mut buffer: Vec<Complex<f32>> = samples.chunks_exact(2)
            .take(fft_size)
            .map(|iq| Complex::new(iq[0] as f32 / 32768.0, iq[1] as f32 / 32768.0))
            .collect();

        if buffer.len() == fft_size {
            fft.process(&mut buffer);

            // Calculate power and find peaks
            let sample_rate = args.sample_rate as f64;
            let center_freq = args.center_freq as f64;

            buffer.par_iter().enumerate().for_each(|(i, bin)| {
                let power = bin.norm_sqr();
                let db = 10.0 * power.log10();
                
                if db > args.threshold_db {
                    // Map bin to frequency
                    let freq_offset = if i < fft_size / 2 {
                        (i as f64 / fft_size as f64) * sample_rate
                    } else {
                        ((i as f64 - fft_size as f64) / fft_size as f64) * sample_rate
                    };
                    
                    let target_freq = center_freq + freq_offset;
                    
                    // Filter for airband range
                    if target_freq >= 118.0e6 && target_freq <= 137.0e6 {
                        // Only print every ~100ms or so per frequency (simplification)
                        println!("[DETECT] {:.3} MHz | Power: {:.1} dB", target_freq / 1e6, db);
                    }
                }
            });
        }

        transfer_pool.submit_bulk(0x81, data).unwrap();
    }

    println!("[*] Stopping...");
    rx888_send_command(handle.as_ref(), FX3Command::STOPFX3, 0).ok();
}

fn open_device_with_timeout(context: &Context, vid: u16, pid: u16, timeout: Duration) -> Option<rusb::DeviceHandle<Context>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Some(h) = context.open_device_with_vid_pid(vid, pid) { return Some(h); }
        thread::sleep(Duration::from_millis(50));
    }
    None
}
