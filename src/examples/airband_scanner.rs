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

    #[arg(short = 's', long, default_value_t = 118.0)]
    start_mhz: f64,

    #[arg(short = 'e', long, default_value_t = 120.0)]
    end_mhz: f64,

    #[arg(short, long, default_value_t = 32000000)]
    sample_rate: u32,

    #[arg(short, long, default_value_t = 30)]
    gain: u8,

    #[arg(long, default_value_t = 10)]
    vhf_lna: u16,

    #[arg(long, default_value_t = 5)]
    vhf_vga: u16,

    #[arg(short, long, default_value_t = 20)]
    attenuation: u16,

    #[arg(short = 't', long, default_value_t = -95.0, allow_hyphen_values = true)]
    threshold: f32,

    #[arg(short = 'i', long, default_value_t = 5.0)]
    if_mhz: f64,

    #[arg(short, long, default_value_t = false)]
    randomize: bool,
}

struct ChannelState {
    accumulator: f32,
    count: u32,
}

fn power_to_dots(db: f32) -> String {
    let min_db = -115.0;
    let max_db = -20.0;
    let width = 45;
    if db < min_db { return "...".to_string(); }
    let normalized = ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
    let dot_count = (normalized * width as f32) as usize;
    ".".repeat(dot_count.max(3))
}

fn main() {
    let args = Cli::parse();
    let context = Context::new().expect("USB context failed");
    
    // Auto-Reset
    for pid in [FX3_FIRMWARE_PID_1, FX3_FIRMWARE_PID_2] {
        if let Some(handle) = context.open_device_with_vid_pid(FX3_VID, pid) {
            let _ = rx888_send_command(&handle, FX3Command::RESETFX3, 0);
            thread::sleep(Duration::from_millis(1500));
        }
    }

    let handle = open_device_with_timeout(&context, FX3_VID, FX3_BOOTLOADER_PID, Duration::from_secs(5)).expect("Bootloader not found");
    let mut fw_file = File::open(&args.firmware).unwrap();
    fx3::fx3_load_ram(handle, &mut fw_file).unwrap();
    thread::sleep(Duration::from_millis(1000));
    let handle = open_device_with_timeout(&context, FX3_VID, FX3_FIRMWARE_PID_1, Duration::from_secs(5)).unwrap();
    handle.claim_interface(0).unwrap();

    let if_offset_hz = args.if_mhz * 1e6;
    let center_freq_hz = ((args.start_mhz + args.end_mhz) / 2.0 * 1e6) as u64;
    
    let mut channel_map: BTreeMap<u64, ChannelState> = BTreeMap::new();
    let mut curr_mhz = args.start_mhz;
    while curr_mhz <= args.end_mhz {
        channel_map.insert((curr_mhz * 1e6) as u64, ChannelState { accumulator: 0.0, count: 0 });
        curr_mhz += 0.025; 
    }

    rx888_send_command(&handle, FX3Command::TUNERSTDBY, 0).ok();
    
    let mut gpio = GPIOPin::VHF_EN as u32 | GPIOPin::PGA_EN as u32;
    if args.randomize { gpio |= GPIOPin::RANDO as u32; }
    rx888_send_command(&handle, FX3Command::GPIOFX3, gpio).unwrap();

    rx888_send_command(&handle, FX3Command::TUNERINIT, 0).unwrap();
    rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, center_freq_hz).unwrap();
    
    rx888_send_argument(&handle, ArgumentList::R82XX_ATTENUATOR, args.vhf_lna).unwrap();
    rx888_send_argument(&handle, ArgumentList::R82XX_VGA, args.vhf_vga).unwrap();
    rx888_send_argument(&handle, ArgumentList::DAT31_ATT, args.attenuation).unwrap();
    rx888_send_argument(&handle, ArgumentList::AD8340_VGA, (args.gain as u16) | 0x80).unwrap();

    rx888_send_command(&handle, FX3Command::STARTADC, args.sample_rate).unwrap();
    rx888_send_command(&handle, FX3Command::STARTFX3, 0).unwrap();

    let handle = Arc::new(handle);
    let mut transfer_pool = TransferPool::new(handle.clone()).unwrap();
    for _ in 0..32 { transfer_pool.submit_bulk(0x81, Vec::with_capacity(131072)).unwrap(); }

    let fft_size = 4096;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let window: Vec<f32> = (0..fft_size).map(|i| {
        let a0 = 0.35875; let a1 = 0.48829; let a2 = 0.14128; let a3 = 0.01168;
        let t = (2.0 * std::f32::consts::PI * i as f32) / (fft_size - 1) as f32;
        a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos()
    }).collect();

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::SeqCst)).ok();

    let mut last_ui_update = Instant::now();
    let sample_rate = args.sample_rate as f64;
    let fft_norm_factor = (fft_size as f32).powi(2) * 0.15;
    
    // To track where the IF hump actually is
    let mut wide_accumulator = vec![0.0f32; fft_size / 2];
    let mut wide_count = 0;

    while running.load(Ordering::SeqCst) {
        let mut data = transfer_pool.poll(Duration::from_secs(1)).expect("USB Timeout");
        if args.randomize {
            let d_u16: &mut [u16] = cast_slice_mut(&mut data);
            for x in d_u16 { *x ^= 0xFFFE * (*x & 0x1); }
        }

        let samples: &[i16] = cast_slice(&data);

        for chunk in samples.chunks_exact(fft_size) {
            let mut buf: Vec<Complex<f32>> = chunk.iter().enumerate()
                .map(|(i, &s)| Complex::new((s as f32 / 32768.0) * window[i], 0.0))
                .collect();
            fft.process(&mut buf);

            wide_count += 1;
            for i in 0..(fft_size / 2) {
                let p = buf[i].norm_sqr() / fft_norm_factor;
                wide_accumulator[i] += p;
            }

            for (freq_hz, state) in channel_map.iter_mut() {
                let rel_offset = *freq_hz as f64 - center_freq_hz as f64;
                let target_freq_in_baseband = (if_offset_hz + rel_offset).abs();
                let bin_idx = (target_freq_in_baseband / (sample_rate / 2.0) * (fft_size as f64 / 2.0)) as usize;
                if bin_idx < fft_size / 2 {
                    state.accumulator += buf[bin_idx].norm_sqr() / fft_norm_factor;
                    state.count += 1;
                }
            }
        }

        if last_ui_update.elapsed() >= Duration::from_millis(200) {
            print!("\x1B[2J\x1B[H"); 
            println!("=== RX888 Diagnostic Monitor ({:.3} - {:.3} MHz) ===", args.start_mhz, args.end_mhz);
            
            // Find Peak in wide spectrum to auto-detect IF location
            let mut peaks: Vec<(usize, f32)> = wide_accumulator.iter().enumerate()
                .skip(20) // skip DC spike
                .map(|(i, &p)| (i, 10.0 * (p / wide_count as f32).log10()))
                .collect();
            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            print!("Detected Peaks: ");
            for (i, db) in peaks.iter().take(3) {
                let freq = (*i as f64 / (fft_size as f64 / 2.0)) * (sample_rate / 2.0);
                print!("| {:.2} MHz ({:.1} dB) ", freq / 1e6, db);
            }
            println!("|");
            
            println!("Gains: LNA {} | VGA {} | PGA {} | IF: {:.1} MHz", args.vhf_lna, args.vhf_vga, args.gain, args.if_mhz);
            println!("{:-<110}", "");

            let mut active_count = 0;
            for (freq_hz, state) in channel_map.iter_mut() {
                let db = if state.count > 0 { 10.0 * (state.accumulator / state.count as f32).log10() } else { -120.0 };
                if db > args.threshold {
                    active_count += 1;
                    println!("{:>8.3} MHz: {:<48} {:>6.1} dB", *freq_hz as f64 / 1e6, power_to_dots(db), db);
                }
                state.accumulator = 0.0; state.count = 0;
            }
            if active_count == 0 { println!("\n(Scanning... Look at 'Detected Peaks' while someone is talking!)"); }
            std::io::stdout().flush().unwrap();
            last_ui_update = Instant::now();
            wide_accumulator.fill(0.0); wide_count = 0;
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
