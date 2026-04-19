use rx888_stream::fx3;
use rx888_stream::rx888;

use std::{
    collections::{BTreeMap, VecDeque},
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

    #[arg(short = 'e', long, default_value_t = 137.0)]
    end_mhz: f64,

    #[arg(short = 'n', long, default_value_t = 5)]
    interval: u64,

    #[arg(short, long, default_value_t = 32000000)]
    sample_rate: u32,

    #[arg(short, long, default_value_t = 60)]
    gain: u8,

    #[arg(short = 't', long, default_value_t = -95.0, allow_hyphen_values = true)]
    threshold: f32,

    /// Enable de-randomization (Try this if you see flat -40dB noise)
    #[arg(short, long, default_value_t = false)]
    randomize: bool,
}

struct ChannelData {
    history: VecDeque<f32>,
    accumulator: f32,
    count: u32,
}

fn main() {
    let args = Cli::parse();
    let context = Context::new().expect("Could not create USB context");

    println!("[*] Initializing RX888 Dashboard (Randomizer: {})...", args.randomize);
    for pid in [FX3_FIRMWARE_PID_1, FX3_FIRMWARE_PID_2] {
        if let Some(handle) = context.open_device_with_vid_pid(FX3_VID, pid) {
            let _ = handle.write_control(0x40, 0x01, 0, 0, &0u32.to_le_bytes(), Duration::from_secs(1));
            let _ = rx888_send_command(&handle, FX3Command::RESETFX3, 0);
            thread::sleep(Duration::from_millis(1500));
        }
    }

    let handle = open_device_with_timeout(&context, FX3_VID, FX3_BOOTLOADER_PID, Duration::from_secs(5))
        .expect("Could not find RX888 bootloader.");
    
    let mut fw_file = File::open(&args.firmware).expect("Could not open firmware");
    fx3::fx3_load_ram(handle, &mut fw_file).expect("Firmware load failed");
    thread::sleep(Duration::from_millis(1000));

    let handle = open_device_with_timeout(&context, FX3_VID, FX3_FIRMWARE_PID_1, Duration::from_secs(5))
        .expect("Could not find RX888 after firmware load");
    
    handle.claim_interface(0).unwrap();

    let mut channel_map: BTreeMap<u64, ChannelData> = BTreeMap::new();
    let mut curr = (args.start_mhz * 1000.0) as u64;
    while curr <= (args.end_mhz * 1000.0) as u64 {
        channel_map.insert(curr * 1000, ChannelData {
            history: VecDeque::with_capacity(10),
            accumulator: 0.0,
            count: 0,
        });
        curr += 25; 
    }

    let b_width = args.end_mhz - args.start_mhz;
    let mut centers_hz = Vec::new();
    if b_width <= 10.0 {
        centers_hz.push(((args.start_mhz + args.end_mhz) / 2.0 * 1e6) as u64);
    } else {
        let mut c = args.start_mhz + 5.0;
        while c < args.end_mhz + 5.0 {
            centers_hz.push((c * 1e6) as u64);
            c += 10.0;
        }
    }

    rx888_send_command(&handle, FX3Command::TUNERINIT, 0).unwrap();
    rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, centers_hz[0]).unwrap();
    
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

    let mut last_report = Instant::now();
    let mut last_hop = Instant::now();
    let mut current_center_idx = 0;
    let sample_rate = args.sample_rate as f64;
    let fft_norm_factor = (fft_size as f32).powi(2);

    while running.load(Ordering::SeqCst) {
        if centers_hz.len() > 1 && last_hop.elapsed() >= Duration::from_millis(500) {
            current_center_idx = (current_center_idx + 1) % centers_hz.len();
            let _ = rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, centers_hz[current_center_idx]);
            last_hop = Instant::now();
            thread::sleep(Duration::from_millis(30));
        }

        let active_center = centers_hz[current_center_idx] as f64;
        let mut data = transfer_pool.poll(Duration::from_secs(1)).expect("USB Timeout");
        
        // De-randomize if requested
        if args.randomize {
            let data_u16: &mut [u16] = cast_slice_mut(&mut data);
            for i in 0..data_u16.len() {
                data_u16[i] ^= 0xFFFE * (data_u16[i] & 0x1);
            }
        }

        let samples: &[u16] = cast_slice(&data);

        for chunk in samples.chunks_exact(fft_size * 2) {
            let mut buffer: Vec<Complex<f32>> = chunk.chunks_exact(2)
                .map(|iq| Complex::new(
                    (iq[0] as f32 - 32768.0) / 32768.0, 
                    (iq[1] as f32 - 32768.0) / 32768.0
                ))
                .collect();
            fft.process(&mut buffer);

            for (freq_hz, data) in channel_map.iter_mut() {
                let offset = *freq_hz as f64 - active_center;
                if offset.abs() <= 5.0e6 {
                    let bin_idx = if offset >= 0.0 {
                        (offset / sample_rate * fft_size as f64) as usize
                    } else {
                        ((offset + sample_rate) / sample_rate * fft_size as f64) as usize
                    };
                    if bin_idx < fft_size {
                        data.accumulator += buffer[bin_idx].norm_sqr() / fft_norm_factor;
                        data.count += 1;
                    }
                }
            }
        }

        if last_report.elapsed().as_secs() >= args.interval {
            print!("\x1B[2J\x1B[H"); 
            println!("=== RX888 Dashboard ({:.1} - {:.1} MHz) ===", args.start_mhz, args.end_mhz);
            println!("Time: {} | Interval: {}s | Tuner Center: {:.1} MHz", 
                chrono::Local::now().format("%H:%M:%S"), args.interval, active_center / 1e6);
            println!("Scale: dBFS | Threshold: {:.1} | Randomizer: {}", args.threshold, args.randomize);
            println!("{:-<110}", "");

            let mut active_count = 0;
            for (freq_hz, data) in channel_map.iter_mut() {
                let avg_power = if data.count > 0 { data.accumulator / data.count as f32 } else { 1e-12 };
                let db = 10.0 * avg_power.log10();

                data.history.push_front(db);
                if data.history.len() > 10 { data.history.pop_back(); }

                if db > args.threshold {
                    active_count += 1;
                    let hist: String = data.history.iter()
                        .map(|v| format!("{:>6.1}", v))
                        .collect::<Vec<_>>().join(" ");
                    println!("[DETECT] {:>8.3} MHz: {}", *freq_hz as f64 / 1e6, hist);
                }
                data.accumulator = 0.0;
                data.count = 0;
            }

            if active_count == 0 {
                println!("\n(Scanning... No signals above {:.1} dBFS)", args.threshold);
            } else {
                println!("\nActive Channels: {}", active_count);
            }
            std::io::stdout().flush().unwrap();
            last_report = Instant::now();
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
