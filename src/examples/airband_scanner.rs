use rx888_stream::fx3;
use rx888_stream::rx888;

use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    path::PathBuf,
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};

use bytemuck::{cast_slice, cast_slice_mut};
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

    /// Start frequency in MHz
    #[arg(short = 's', long, default_value_t = 118.0)]
    start_mhz: f64,

    /// End frequency in MHz
    #[arg(short = 'e', long, default_value_t = 128.0)]
    end_mhz: f64,

    #[arg(short, long, default_value_t = 32000000)]
    sample_rate: u32,

    #[arg(short, long, default_value_t = 60)]
    gain: u8,

    /// Squelch threshold in dBFS (e.g. -90.0). Only channels above this will appear.
    #[arg(short = 't', long, default_value_t = -92.0, allow_hyphen_values = true)]
    threshold: f32,

    #[arg(short, long, default_value_t = false)]
    randomize: bool,
}

struct ChannelState {
    accumulator: f32,
    count: u32,
    last_db: f32,
}

fn power_to_dots(db: f32) -> String {
    let min_db = -100.0;
    let max_db = -25.0;
    let width = 50;
    
    if db < min_db { return "...".to_string(); }
    
    let normalized = ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
    let dot_count = (normalized * width as f32) as usize;
    ".".repeat(dot_count.max(3))
}

fn main() {
    let args = Cli::parse();
    
    let span = args.end_mhz - args.start_mhz;
    if span <= 0.0 || span > 10.0 {
        eprintln!("Error: Scan range must be between 0 and 10.0 MHz (Requested: {:.1} MHz)", span);
        std::process::exit(1);
    }

    let context = Context::new().expect("Could not create USB context");

    println!("[*] Initializing RX888 Multi-Channel Monitor...");
    for pid in [FX3_FIRMWARE_PID_1, FX3_FIRMWARE_PID_2] {
        if let Some(handle) = context.open_device_with_vid_pid(FX3_VID, pid) {
            let _ = handle.write_control(0x40, 0x01, 0, 0, &0u32.to_le_bytes(), Duration::from_secs(1));
            let _ = rx888_send_command(&handle, FX3Command::RESETFX3, 0);
            thread::sleep(Duration::from_millis(1500));
        }
    }

    let handle = open_device_with_timeout(&context, FX3_VID, FX3_BOOTLOADER_PID, Duration::from_secs(5))
        .expect("Could not find bootloader.");
    
    let mut fw_file = File::open(&args.firmware).expect("Could not open firmware");
    fx3::fx3_load_ram(handle, &mut fw_file).expect("Firmware load failed");
    thread::sleep(Duration::from_millis(1000));

    let handle = open_device_with_timeout(&context, FX3_VID, FX3_FIRMWARE_PID_1, Duration::from_secs(5))
        .expect("Could not find RX888 after firmware load");
    
    handle.claim_interface(0).unwrap();

    let center_freq_hz = ((args.start_mhz + args.end_mhz) / 2.0 * 1e6) as u64;
    let mut channel_map: BTreeMap<u64, ChannelState> = BTreeMap::new();
    let mut curr_mhz = args.start_mhz;
    while curr_mhz <= args.end_mhz {
        channel_map.insert((curr_mhz * 1e6) as u64, ChannelState {
            accumulator: 0.0,
            count: 0,
            last_db: -110.0,
        });
        curr_mhz += 0.025; 
    }

    rx888_send_command(&handle, FX3Command::TUNERINIT, 0).unwrap();
    rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, center_freq_hz).unwrap();
    
    let mut gpio = GPIOPin::VHF_EN as u32 | GPIOPin::PGA_EN as u32;
    if args.randomize { gpio |= GPIOPin::RANDO as u32; }
    rx888_send_command(&handle, FX3Command::GPIOFX3, gpio).unwrap();
    
    rx888_send_argument(&handle, ArgumentList::R82XX_ATTENUATOR, 20).unwrap();
    rx888_send_argument(&handle, ArgumentList::R82XX_VGA, 12).unwrap();
    rx888_send_argument(&handle, ArgumentList::AD8340_VGA, (args.gain | 0x80) as u16).unwrap();

    rx888_send_command(&handle, FX3Command::STARTADC, args.sample_rate).unwrap();
    rx888_send_command(&handle, FX3Command::STARTFX3, 0).unwrap();

    let handle = Arc::new(handle);
    let mut transfer_pool = TransferPool::new(handle.clone()).unwrap();
    for _ in 0..32 {
        transfer_pool.submit_bulk(0x81, Vec::with_capacity(131072)).unwrap();
    }

    let fft_size = 4096;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::SeqCst)).ok();

    let mut last_ui_update = Instant::now();
    let sample_rate = args.sample_rate as f64;
    let fft_norm_factor = (fft_size as f32).powi(2);

    while running.load(Ordering::SeqCst) {
        let mut data = transfer_pool.poll(Duration::from_secs(1)).expect("USB Timeout");
        
        if args.randomize {
            let d_u16: &mut [u16] = cast_slice_mut(&mut data);
            for x in d_u16 { *x ^= 0xFFFE * (*x & 0x1); }
        }

        let samples: &[i16] = cast_slice(&data);

        for chunk in samples.chunks_exact(fft_size * 2) {
            let mut buf: Vec<Complex<f32>> = chunk.chunks_exact(2)
                .map(|iq| Complex::new(iq[0] as f32 / 32768.0, iq[1] as f32 / 32768.0))
                .collect();
            fft.process(&mut buf);

            for (freq_hz, state) in channel_map.iter_mut() {
                let offset = *freq_hz as f64 - center_freq_hz as f64;
                let bin_idx = if offset >= 0.0 {
                    (offset / sample_rate * fft_size as f64) as usize
                } else {
                    ((offset + sample_rate) / sample_rate * fft_size as f64) as usize
                };
                if bin_idx < fft_size {
                    state.accumulator += buf[bin_idx].norm_sqr() / fft_norm_factor;
                    state.count += 1;
                }
            }
        }

        if last_ui_update.elapsed() >= Duration::from_millis(200) {
            print!("\x1B[2J\x1B[H"); 
            println!("=== RX888 Real-time Airband Monitor ({:.3} - {:.3} MHz) ===", args.start_mhz, args.end_mhz);
            println!("Time: {} | Channels: {} | Gain: {} | Squelch: {:.1}", 
                chrono::Local::now().format("%H:%M:%S"), channel_map.len(), args.gain, args.threshold);
            println!("{:-<110}", "");

            let mut active_count = 0;
            for (freq_hz, state) in channel_map.iter_mut() {
                let avg_power = if state.count > 0 { state.accumulator / state.count as f32 } else { state.last_db.powf(10.0/10.0) };
                let db = 10.0 * avg_power.log10();
                state.last_db = db;

                if db > args.threshold {
                    active_count += 1;
                    // Aligned dots followed by the dB value
                    println!("{:>8.3} MHz: {:<52} {:>5.1} dB", 
                        *freq_hz as f64 / 1e6, 
                        power_to_dots(db), 
                        db);
                }

                state.accumulator = 0.0;
                state.count = 0;
            }

            if active_count == 0 {
                println!("\n(Searching... All channels below threshold)");
            }
            std::io::stdout().flush().unwrap();
            last_ui_update = Instant::now();
        }

        transfer_pool.submit_bulk(0x81, data).unwrap();
    }
    
    rx888_send_command(handle.as_ref(), FX3Command::STOPFX3, 0).ok();
}

fn open_device_with_timeout(context: &Context, vid: u16, pid: u16, timeout: Duration) -> Option<rusb::DeviceHandle<Context>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Some(h) = context.open_device_with_vid_pid(vid, pid) { return Some(h); }
        thread::sleep(Duration::from_millis(100));
    }
    None
}
