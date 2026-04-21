use rx888_stream::fx3;
use rx888_stream::rx888;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};
use bytemuck::cast_slice;
use clap::Parser;
use rusb::{Context, UsbContext};
use rustfft::{FftPlanner, num_complex::Complex};

use rx888::{
    rx888_send_argument, rx888_send_command, rx888_send_command_u64, ArgumentList, FX3Command,
    GPIOPin,
};

#[derive(Parser, Clone)]
struct Cli {
    /// Firmware file to load
    #[arg(short, long, default_value = "SDDC_FX3.img")] 
    firmware: PathBuf,

    /// Start frequency in MHz
    #[arg(short = 's', long, default_value_t = 118.0)] 
    start_mhz: f64,

    /// End frequency in MHz
    #[arg(short = 'e', long, default_value_t = 128.0)] 
    end_mhz: f64,

    /// ADC sample rate in Hz
    #[arg(long, default_value_t = 32000000)] 
    sample_rate: u32,

    /// VGA gain setting 0-127
    #[arg(short, long, default_value_t = 30)] 
    gain: u8,

    /// Tuner LNA gain 0-29
    #[arg(long, default_value_t = 29)] 
    vhf_lna: u16,

    /// Tuner VGA gain 0-15
    #[arg(long, default_value_t = 10)] 
    vhf_vga: u16,

    /// Attenuator setting 0-63
    #[arg(short, long, default_value_t = 0)] 
    attenuation: u16,

    /// IF frequency in MHz (typical for RX888 is 10.2 MHz)
    #[arg(short = 'i', long, default_value_t = 10.2)] 
    if_mhz: f64,
}

fn open_device_with_timeout(ctx: &Context, vid: u16, pid: u16, timeout: Duration) -> Option<rusb::DeviceHandle<Context>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Some(h) = ctx.open_device_with_vid_pid(vid, pid) {
            return Some(h);
        }
        thread::sleep(Duration::from_millis(100));
    }
    None
}

fn main() {
    let args = Cli::parse();

    // 1. Validation
    if args.end_mhz - args.start_mhz > 10.0 {
        eprintln!("Error: Frequency span exceeds 10MHz ({} MHz provided)", args.end_mhz - args.start_mhz);
        std::process::exit(1);
    }
    if args.start_mhz >= args.end_mhz {
        eprintln!("Error: Start frequency must be less than end frequency");
        std::process::exit(1);
    }

    let context = Context::new().expect("Could not create USB context");

    // 2. Device initialization
    // First, check if already in firmware mode and reset if requested/needed
    // For simplicity, we'll try to find the bootloader first. If not found, check if firmware is running and reset it.
    if context.open_device_with_vid_pid(0x04b4, 0x00f3).is_none() {
        if let Some(h) = context.open_device_with_vid_pid(0x04b4, 0x00f1) {
            println!("Resetting active device to bootloader...");
            let _ = rx888_send_command(&h, FX3Command::RESETFX3, 0);
            thread::sleep(Duration::from_millis(1500));
        }
    }

    let b_handle = open_device_with_timeout(&context, 0x04b4, 0x00f3, Duration::from_secs(5))
        .expect("Could not find RX888 bootloader (VID:04b4 PID:00f3)");

    println!("Loading firmware {:?}...", args.firmware);
    let mut fw_file = File::open(&args.firmware).expect("Could not open firmware file");
    fx3::fx3_load_ram(b_handle, &mut fw_file).expect("Failed to load firmware");
    thread::sleep(Duration::from_millis(1500));

    let handle = open_device_with_timeout(&context, 0x04b4, 0x00f1, Duration::from_secs(5))
        .expect("Device did not reappear after firmware load");
    
    if handle.kernel_driver_active(0).unwrap_or(false) {
        let _ = handle.detach_kernel_driver(0);
    }
    handle.claim_interface(0).expect("Could not claim USB interface");

    // 3. HW Configuration
    let tuner_center_mhz = (args.start_mhz + args.end_mhz) / 2.0;
    let tuner_center_hz = (tuner_center_mhz * 1e6) as u64;

    println!("Tuning to center frequency {} MHz...", tuner_center_mhz);
    rx888_send_command(&handle, FX3Command::TUNERSTDBY, 0).ok();
    thread::sleep(Duration::from_millis(100));
    rx888_send_command(&handle, FX3Command::TUNERINIT, 0).ok();
    thread::sleep(Duration::from_millis(100));
    rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, tuner_center_hz).ok();
    thread::sleep(Duration::from_millis(100));

    // Enable VHF
    let gpio = GPIOPin::VHF_EN as u32;
    rx888_send_command(&handle, FX3Command::GPIOFX3, gpio).ok();

    rx888_send_argument(&handle, ArgumentList::R82XX_ATTENUATOR, args.vhf_lna).ok();
    rx888_send_argument(&handle, ArgumentList::R82XX_VGA, args.vhf_vga).ok();
    rx888_send_argument(&handle, ArgumentList::DAT31_ATT, args.attenuation).ok();
    rx888_send_argument(&handle, ArgumentList::AD8340_VGA, (args.gain as u16) | 0x80).ok();

    rx888_send_command(&handle, FX3Command::STARTADC, args.sample_rate).ok();
    rx888_send_command(&handle, FX3Command::STARTFX3, 0).ok();

    // 4. Processing Loop Setup
    let fft_size = 4096; 
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    
    // Blackman-Harris window
    let window: Vec<f32> = (0..fft_size).map(|i| {
        let a0 = 0.35875;
        let a1 = 0.48829;
        let a2 = 0.14128;
        let a3 = 0.01168;
        let t = 2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32;
        a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos()
    }).collect();

    let handle_arc = Arc::new(handle);
    let mut transfer_pool = rusb_async::TransferPool::new(handle_arc.clone()).unwrap();
    let packet_size = 131072;
    for _ in 0..32 {
        transfer_pool.submit_bulk(0x81, Vec::with_capacity(packet_size)).unwrap();
    }

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::SeqCst)).ok();

    let mut channels: BTreeMap<u64, (f32, u32)> = BTreeMap::new();
    let mut curr_mhz = args.start_mhz;
    while curr_mhz <= args.end_mhz + 0.001 { 
        channels.insert((curr_mhz * 1000.0).round() as u64, (0.0, 0));
        curr_mhz += 0.025;
    }

    println!("Scanning {} channels every 25kHz...", channels.len());
    let mut last_display = Instant::now();
    let if_hz = args.if_mhz * 1e6;
    let fs = args.sample_rate as f32;

    while running.load(Ordering::SeqCst) {
        let mut data = transfer_pool.poll(Duration::from_secs(1)).expect("USB Timeout");
        let samples: &[i16] = cast_slice(&data);

        for chunk in samples.chunks_exact(fft_size) {
            let mut buf: Vec<Complex<f32>> = chunk.iter().enumerate()
                .map(|(i, &s)| Complex::new((s as f32 / 32768.0) * window[i], 0.0))
                .collect();
            
            fft.process(&mut buf);

            for (freq_khz, (power_acc, count)) in channels.iter_mut() {
                let freq_mhz = *freq_khz as f64 / 1000.0;
                let rel_hz = (freq_mhz - tuner_center_mhz) * 1e6;
                let target_hz = if_hz + rel_hz;
                
                let bin = (target_hz.abs() / (fs as f64) * (fft_size as f64)) as usize;
                if bin < fft_size {
                    *power_acc += buf[bin].norm_sqr();
                    *count += 1;
                }
            }
        }

        if last_display.elapsed() >= Duration::from_millis(400) {
            print!("\x1B[2J\x1B[H"); // Clear and home
            println!(" RX888 Airband Scanner | Center: {:.3} MHz | Span: {:.1} MHz", tuner_center_mhz, args.end_mhz - args.start_mhz);
            println!("{:-<75}", "");

            for (freq_khz, (power_acc, count)) in channels.iter_mut() {
                let avg_power = if *count > 0 { *power_acc / *count as f32 } else { 1e-12 };
                let db = 10.0 * avg_power.log10() - 25.0; // Adjusted calibration
                
                let mhz = *freq_khz as f64 / 1000.0;
                let dots_count = (((db + 95.0) * 0.6).max(0.0).min(40.0)) as usize;
                let dots = ".".repeat(dots_count);
                
                println!(" {:>7.3} MHz: {:<42} {:>6.1} dB", mhz, dots, db);
                
                *power_acc = 0.0;
                *count = 0;
            }
            std::io::stdout().flush().ok();
            last_display = Instant::now();
        }

        transfer_pool.submit_bulk(0x81, data).unwrap();
    }

    println!("\nStopping...");
    rx888_send_command(handle_arc.as_ref(), FX3Command::STARTADC, 10000000).ok();
    rx888_send_command(handle_arc.as_ref(), FX3Command::STOPFX3, 0).ok();
}
