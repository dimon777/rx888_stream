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
use rustfft::{FftPlanner, num_complex::Complex};
use rayon::prelude::*;

use rx888::{
    rx888_send_argument, rx888_send_command, rx888_send_command_u64, ArgumentList, FX3Command,
    GPIOPin,
};

#[derive(Parser, Clone)]
struct Cli {
    #[arg(short, long)] firmware: PathBuf,
    #[arg(short = 's', long, default_value_t = 133.0)] start_mhz: f64,
    #[arg(short = 'e', long, default_value_t = 133.5)] end_mhz: f64,
    #[arg(short, long, default_value_t = 32000000)] sample_rate: u32,
    #[arg(short, long, default_value_t = 40)] gain: u8,
    #[arg(long, default_value_t = 25)] vhf_lna: u16,
    #[arg(long, default_value_t = 12)] vhf_vga: u16,
    #[arg(short, long, default_value_t = 10)] attenuation: u16,
    #[arg(short = 't', long, default_value_t = -95.0, allow_hyphen_values = true)] threshold: f32,
    #[arg(short = 'i', long, default_value_t = 3.57)] if_mhz: f64,
}

fn power_to_char(db: f32) -> char {
    if db > -25.0 { '#' }
    else if db > -40.0 { '=' }
    else if db > -55.0 { '-' }
    else if db > -80.0 { '.' }
    else { ' ' }
}

fn main() {
    let args = Cli::parse();
    let context = Context::new().expect("USB failed");
    
    // Recovery reset
    for pid in [0x00f1, 0x3ddc] {
        if let Some(h) = context.open_device_with_vid_pid(0x04b4, pid) {
            let _ = rx888_send_command(&h, FX3Command::RESETFX3, 0);
            thread::sleep(Duration::from_millis(1500));
        }
    }

    let b_handle = open_device_with_timeout(&context, 0x04b4, 0x00f3, Duration::from_secs(5)).expect("No bootloader");
    fx3::fx3_load_ram(b_handle, &mut File::open(&args.firmware).unwrap()).unwrap();
    thread::sleep(Duration::from_millis(1500));
    let handle = open_device_with_timeout(&context, 0x04b4, 0x00f1, Duration::from_secs(5)).unwrap();
    handle.claim_interface(0).unwrap();

    let tuner_freq_hz = (args.start_mhz * 1e6) as u64; // Tune directly to start freq to minimize offset
    let mut channel_map: BTreeMap<u64, (f32, u32)> = BTreeMap::new();
    let mut curr_mhz = args.start_mhz;
    while curr_mhz <= args.end_mhz {
        channel_map.insert((curr_mhz * 1e6) as u64, (0.0, 0));
        curr_mhz += 0.025; 
    }

    // --- HARDWARE LOCK SEQUENCE ---
    rx888_send_command(&handle, FX3Command::TUNERSTDBY, 0).ok();
    thread::sleep(Duration::from_millis(200));
    rx888_send_command(&handle, FX3Command::STARTADC, args.sample_rate).ok(); // Set Clock Early
    thread::sleep(Duration::from_millis(200));
    rx888_send_command(&handle, FX3Command::TUNERINIT, 0).ok();
    rx888_send_command_u64(&handle, FX3Command::TUNERTUNE, tuner_freq_hz).ok();
    
    let gpio = (GPIOPin::VHF_EN as u32) | (1 << 5); // VHF_EN + SHDWN
    rx888_send_command(&handle, FX3Command::GPIOFX3, gpio).ok();

    rx888_send_argument(&handle, ArgumentList::R82XX_ATTENUATOR, args.vhf_lna).ok();
    rx888_send_argument(&handle, ArgumentList::R82XX_VGA, args.vhf_vga).ok();
    rx888_send_argument(&handle, ArgumentList::DAT31_ATT, args.attenuation).ok();
    rx888_send_argument(&handle, ArgumentList::AD8340_VGA, (args.gain as u16) | 0x80).ok();

    rx888_send_command(&handle, FX3Command::STARTADC, args.sample_rate).ok();
    rx888_send_command(&handle, FX3Command::STARTFX3, 0).ok();

    let handle_arc = Arc::new(handle);
    let mut transfer_pool = rusb_async::TransferPool::new(handle_arc.clone()).unwrap();
    for _ in 0..128 { transfer_pool.submit_bulk(0x81, Vec::with_capacity(16384)).unwrap(); }

    let fft_size = 4096;
    let mut fft = FftPlanner::new().plan_fft_forward(fft_size);
    let window: Vec<f32> = (0..fft_size).map(|i| {
        let t = (2.0 * std::f32::consts::PI * i as f32) / (fft_size - 1) as f32;
        0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
    }).collect();

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::SeqCst)).ok();

    let mut last_ui_update = Instant::now();
    let sample_rate = args.sample_rate as f64;
    let fft_norm = (fft_size as f32).powi(2) * 0.15;
    let mut wide_acc = vec![0.0f32; fft_size / 2];
    let mut wide_cnt = 0;
    let mut current_if_hz = args.if_mhz * 1e6;
    let mut auto_locked = false;

    while running.load(Ordering::SeqCst) {
        let mut data = transfer_pool.poll(Duration::from_secs(1)).expect("USB Timeout");
        // De-randomize (Match main tool)
        let d_u16: &mut [u16] = cast_slice_mut(&mut data);
        for x in d_u16 { *x ^= 0xFFFE * (*x & 0x1); }
        let samples: &[i16] = cast_slice(&data);

        for chunk in samples.chunks_exact(fft_size) {
            let mut buf: Vec<Complex<f32>> = chunk.iter().enumerate()
                .map(|(i, &s)| Complex::new((s as f32 / 32768.0) * window[i], 0.0)).collect();
            fft.process(&mut buf);
            wide_cnt += 1;
            for (i, p) in wide_acc.iter_mut().enumerate() { *p += buf[i].norm_sqr() / fft_norm; }
            if auto_locked {
                for (freq_hz, (acc, cnt)) in channel_map.iter_mut() {
                    let rel = ( *freq_hz as f64 - tuner_freq_hz as f64).abs();
                    let bin = ((current_if_hz + rel) / (sample_rate / 2.0) * (fft_size as f64 / 2.0)) as usize;
                    if bin < fft_size / 2 { *acc += buf[bin].norm_sqr() / fft_norm; *cnt += 1; }
                }
            }
        }

        if last_ui_update.elapsed() >= Duration::from_millis(250) {
            let mut pks: Vec<(usize, f32)> = wide_acc.iter().enumerate().skip(60)
                .map(|(i, &p)| (i, 10.0 * (p / wide_cnt as f32).log10())).collect();
            pks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            if !auto_locked && start_time.elapsed().as_secs() >= 3 {
                if let Some(p) = pks.get(0) { current_if_hz = (p.0 as f64 / (fft_size as f64 / 2.0)) * (sample_rate / 2.0); auto_locked = true; }
            }
            
            print!("\x1B[2J\x1B[H");
            println!("=== RX888 Final Locked Monitor ({:.3} - {:.3} MHz) ===", args.start_mhz, args.end_mhz);
            print!("Waterfall: [");
            for i in 0..64 {
                let mut max_db: f32 = -120.0;
                let bs = i * (fft_size/2) / 64; let be = (i+1) * (fft_size/2) / 64;
                for b in bs..be { if b < wide_acc.len() { max_db = max_db.max(10.0 * (wide_acc[b] / (wide_cnt as f32)).log10()); } }
                print!("{}", power_to_char(max_db));
            }
            println!("]");
            
            let strongest_f = (pks[0].0 as f64 / (fft_size as f64 / 2.0)) * (sample_rate / 2.0);
            println!("Status: {} | Tuner Lock: {:.3} MHz | Strongest Peak: {:.3} MHz ({:.1} dB)", 
                if auto_locked { "LOCKED" } else { "TUNING" }, tuner_freq_hz as f64 / 1e6, strongest_f / 1e6, pks[0].1);
            println!("{:-<105}", "");
            if auto_locked {
                for (f_hz, (acc, cnt)) in channel_map.iter_mut() {
                    let db = if *cnt > 0 { 10.0 * (*acc / *cnt as f32).log10() } else { -120.0 };
                    if db > args.threshold {
                        println!("{:>8.3} MHz: {:<40} {:>5.1} dB", *f_hz as f64 / 1e6, ".".repeat(((db+110.0)*0.5).max(1.0).min(40.0) as usize), db);
                    }
                    *acc = 0.0; *cnt = 0;
                }
            }
            std::io::stdout().flush().ok();
            last_ui_update = Instant::now();
            if !auto_locked { wide_acc.fill(0.0); wide_cnt = 0; }
        }
        transfer_pool.submit_bulk(0x81, data).unwrap();
    }
    rx888_send_command(handle_arc.as_ref(), FX3Command::STOPFX3, 0).ok();
}

fn open_device_with_timeout(ctx: &rusb::Context, vid: u16, pid: u16, timeout: Duration) -> Option<rusb::DeviceHandle<rusb::Context>> {
    let s = Instant::now();
    while s.elapsed() < timeout { if let Some(h) = ctx.open_device_with_vid_pid(vid, pid) { return Some(h); } thread::sleep(Duration::from_millis(100)); }
    None
}
