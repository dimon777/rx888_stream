//! Airband Scanner for RX888 SDR
//!
//! Continuously scans frequencies and displays signal strength using terminal escape sequences.

mod fx3;
mod rx888;

use clap::Parser;
use rusb::{Context, UsbContext};
use rusb_async::TransferPool;
use rx888::{
    rx888_send_argument, rx888_send_command, rx888_send_command_u64, ArgumentList, FX3Command,
    GPIOPin,
};
use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
    thread,
    time::Duration,
};
use std::sync::atomic::{AtomicBool, Ordering};

const FX3_VID: u16 = 0x04b4;
const FX3_BOOTLOADER_PID: u16 = 0x00f3;
const FX3_FIRMWARE_PID: u16 = 0x00f1;

/// Scanner channel width: 25 kHz = 0.025 MHz
const CHANNEL_WIDTH_HZ: u64 = 25_000;

/// Maximum scan range: 10 MHz
const MAX_SCAN_RANGE_HZ: u64 = 10_000_000;

/// Number of samples to average per channel
const SAMPLES_PER_CHANNEL: usize = 131072;

const PACKET_SIZE: usize = 131072;
const NUM_TRANSFERS: usize = 32;

/// Airband scanner command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about = "Airband Scanner for RX888 SDR")]
struct Args {
    /// Firmware file (SDDC_FX3.img)
    #[arg(short, long)]
    firmware: PathBuf,

    /// Start frequency in MHz (e.g., 118.0)
    #[arg(short = 's', long)]
    start_freq: f64,

    /// End frequency in MHz (e.g., 137.0)
    #[arg(short = 'e', long)]
    end_freq: f64,

    /// VGA gain setting 0-127
    #[arg(short, long, default_value_t = 40)]
    gain: u8,

    /// Attenuator setting 0-63
    #[arg(short, long, default_value_t = 0)]
    attenuation: u8,

    /// Enable dithering
    #[arg(long, default_value_t = false)]
    dither: bool,

    /// Measurement interval in milliseconds
    #[arg(long, default_value_t = 100)]
    interval_ms: u64,

    /// Packet size for USB transfers
    #[arg(long, default_value_t = 131072)]
    packet_size: usize,

    /// Number of USB transfers
    #[arg(long, default_value_t = 32)]
    num_transfers: usize,
}

/// Represents a frequency scan result
struct ScanResult {
    frequency_mhz: f64,
    signal_dbm: f64,
    max_signal: f64,
}

impl ScanResult {
    /// Create bars representing signal strength
    /// More dots = higher signal
    fn signal_bars(&self) -> String {
        // Map dBm to bar count (0-40 bars)
        // Typical range: -90 dBm (noise floor) to -30 dBm (strong signal)
        let normalized = ((self.signal_dbm + 90.0) / 60.0).clamp(0.0, 1.0);
        let bars = (normalized * 40.0) as usize;
        ".".repeat(bars)
    }
}

/// Convert raw ADC samples to dB relative to full scale
fn samples_to_db(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return -100.0;
    }

    // Calculate RMS (Root Mean Square)
    let sum_squares: f64 = samples.iter()
        .map(|&s| {
            let normalized = s as f64 / 32768.0;
            normalized * normalized
        })
        .sum();

    let rms = (sum_squares / samples.len() as f64).sqrt();

    // Convert to dBFS (dB relative to full scale)
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -100.0
    }
}

/// Clear current line and move cursor to beginning using ANSI escape codes
fn clear_line() {
    print!("\r\x1B[K");
}

/// Move cursor up N lines
fn cursor_up(lines: u32) {
    print!("\x1B[{}A", lines);
}

/// Move cursor down N lines
fn cursor_down(lines: u32) {
    print!("\x1B[{}B", lines);
}

/// Flush stdout immediately
fn flush_output() {
    io::stdout().flush().ok();
}

/// Open device with VID/PID with timeout
fn open_device_with_vid_pid_timeout(
    context: &Context,
    vid: u16,
    pid: u16,
    timeout: Duration,
) -> Option<rusb::DeviceHandle<Context>> {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if let Some(handle) = context.open_device_with_vid_pid(vid, pid) {
            return Some(handle);
        }
        thread::sleep(Duration::from_millis(100));
    }
    None
}

/// Initialize RX888 device with firmware and settings
fn initialize_device(
    context: &Context,
    firmware_path: &PathBuf,
) -> rusb::DeviceHandle<Context> {
    // Check if device is in firmware mode, if so reset to bootloader
    if let Some(handle) = context.open_device_with_vid_pid(FX3_VID, FX3_FIRMWARE_PID) {
        rx888_send_command(&handle, FX3Command::RESETFX3, 0)
            .expect("Could not reset FX3 to bootloader mode");
        eprintln!("Reset device to bootloader mode");
    }

    // Wait and open bootloader
    let handle = open_device_with_vid_pid_timeout(
        context,
        FX3_VID,
        FX3_BOOTLOADER_PID,
        Duration::from_secs(5),
    )
    .expect("Could not find or open bootloader");

    // Load firmware
    let mut file = File::open(firmware_path).expect("Could not open firmware file");
    fx3::fx3_load_ram(handle, &mut file).expect("Could not load firmware");

    eprintln!("Firmware loaded, waiting for device...");
    thread::sleep(Duration::from_millis(1500));

    // Open firmware device
    let handle = open_device_with_vid_pid_timeout(
        context,
        FX3_VID,
        FX3_FIRMWARE_PID,
        Duration::from_secs(5),
    )
    .expect("Could not find or open device with firmware");

    // Detach kernel driver if active
    if handle.kernel_driver_active(0).unwrap_or(false) {
        handle
            .detach_kernel_driver(0)
            .expect("Could not detach kernel driver");
    }
    handle.claim_interface(0).expect("Could not claim interface");

    handle
}

/// Configure RX888 for scanning
fn configure_device(handle: &rusb::DeviceHandle<Context>, args: &Args) {
    let mut gpio: u32 = 0;

    if args.dither {
        gpio |= GPIOPin::DITH as u32;
    }

    eprintln!("Setting GPIO: {}", gpio);
    rx888_send_command(handle, FX3Command::GPIOFX3, gpio).expect("Could not set GPIO");

    // Set attenuator
    rx888_send_argument(handle, ArgumentList::DAT31_ATT, args.attenuation)
        .expect("Could not set ATT");

    // Set VGA gain
    rx888_send_argument(handle, ArgumentList::AD8340_VGA, args.gain as u16)
        .expect("Could not set VGA");

    // Initialize tuner for HF mode (direct sampling)
    rx888_send_command(handle, FX3Command::TUNERSTDBY, 0)
        .expect("Could not set tuner standby");

    // Start ADC at 50 MHz
    rx888_send_command(handle, FX3Command::STARTADC, 50_000_000)
        .expect("Could not start ADC");

    // Start FX3 streaming
    rx888_send_command(handle, FX3Command::STARTFX3, 0)
        .expect("Could not start FX3");
}

/// Tune to a specific frequency
fn tune_to_frequency(handle: &rusb::DeviceHandle<Context>, frequency_hz: u64) {
    rx888_send_command_u64(handle, FX3Command::TUNERTUNE, frequency_hz)
        .expect("Could not tune to frequency");
    // Small delay for tuning
    thread::sleep(Duration::from_millis(10));
}

/// Scan a single frequency channel
fn scan_channel(
    handle: &Arc<rusb::DeviceHandle<Context>>,
    frequency_hz: u64,
    packet_size: usize,
    num_transfers: usize,
    interval_ms: u64,
) -> ScanResult {
    // Tune to frequency
    tune_to_frequency(handle, frequency_hz);

    // Short settling time
    thread::sleep(Duration::from_millis(interval_ms));

    // Create transfer pool
    let pool = TransferPool::new(handle.clone())
        .expect("Could not create transfer pool");

    // Submit initial transfers
    while pool.pending() < num_transfers {
        pool.submit_bulk(0x81, Vec::with_capacity(packet_size))
            .expect("Could not submit transfer");
    }

    let timeout = Duration::from_secs(1);
    let mut all_samples: Vec<i16> = Vec::with_capacity(SAMPLES_PER_CHANNEL);
    let mut max_amplitude: f64 = 0.0;

    // Collect samples for this channel
    while all_samples.len() < SAMPLES_PER_CHANNEL {
        if let Ok(mut data) = pool.poll(timeout) {
            // Data is raw u16 samples from ADC
            let samples_u16: &[u16] = bytemuck::cast_slice(&data);

            // Convert to i16 (centered around 0)
            for &sample in samples_u16 {
                let signed_sample = (sample as i16).wrapping_sub(32768);
                let normalized = (signed_sample as f64).abs() / 32768.0;
                if normalized > max_amplitude {
                    max_amplitude = normalized;
                }
                all_samples.push(signed_sample);
            }

            // Resubmit transfer
            pool.submit_bulk(0x81, data)
                .expect("Failed to resubmit transfer");
        }
    }

    // Cancel remaining transfers
    pool.cancel_all();

    // Calculate signal strength
    let db_fs = samples_to_db(&all_samples);

    // Rough calibration: assume noise floor around -90 dBFS
    // and convert to approximate dBm (actual calibration would require known reference)
    let approx_dbm = db_fs + 30.0;  // Typical SDR noise floor offset

    let frequency_mhz = frequency_hz as f64 / 1_000_000.0;

    ScanResult {
        frequency_mhz,
        signal_dbm: approx_dbm,
        max_signal: 20.0 * max_amplitude.log10().max(-100.0),
    }
}

/// Display scan results with escape sequences
fn display_results(results: &[ScanResult], current_idx: usize) {
    // Move cursor up to overwrite previous results
    if !results.is_empty() {
        cursor_up(results.len() as u32);
    }

    for (idx, result) in results.iter().enumerate() {
        clear_line();
        let bars = result.signal_bars();
        let freq_str = format!("{:>9.3} MHz: {:<40}", result.frequency_mhz, bars);
        print!("{}", freq_str);
        println!("  {:>7.1f} dB", result.signal_dbm);
    }
    flush_output();
}

/// Main scanning loop
fn run_scanner(context: &Context, args: &Args) {
    // Calculate scan parameters
    let start_hz = (args.start_freq * 1_000_000.0) as u64;
    let end_hz = (args.end_freq * 1_000_000.0) as u64;

    // Validate range
    let range_hz = end_hz.saturating_sub(start_hz);
    if range_hz > MAX_SCAN_RANGE_HZ {
        eprintln!("Error: Frequency range ({} MHz) exceeds maximum of {} MHz",
            range_hz as f64 / 1_000_000.0,
            MAX_SCAN_RANGE_HZ as f64 / 1_000_000.0);
        std::process::exit(1);
    }

    if start_hz >= end_hz {
        eprintln!("Error: Start frequency must be less than end frequency");
        std::process::exit(1);
    }

    let num_channels = (range_hz / CHANNEL_WIDTH_HZ + 1) as usize;
    eprintln!("Scanning {} channels from {:.3} MHz to {:.3} MHz",
        num_channels, args.start_freq, args.end_freq);
    eprintln!("Channel width: {} kHz, Range: {:.3} MHz",
        CHANNEL_WIDTH_HZ / 1000, range_hz as f64 / 1_000_000.0);

    // Initialize device
    let mut handle = initialize_device(context, &args.firmware);
    configure_device(&handle, args);

    let handle = Arc::new(handle);

    // Handle Ctrl-C gracefully
    let terminate = Arc::new(AtomicBool::new(false));
    {
        let terminate = terminate.clone();
        ctrlc::set_handler(move || {
            terminate.store(true, Ordering::Relaxed);
        }).ok();
    }

    // Results buffer for display
    let mut results: Vec<ScanResult> = Vec::with_capacity(num_channels);
    let mut current_channel: u64 = start_hz;

    // Terminal setup - hide cursor
    print!("\x1B[?25l");
    flush_output();

    eprintln!("\nStarting scan... (Press Ctrl+C to stop)\n");

    while !terminate.load(Ordering::Relaxed) {
        results.clear();

        // Scan all channels in range
        let mut freq = start_hz;
        while freq <= end_hz {
            if terminate.load(Ordering::Relaxed) {
                break;
            }

            let result = scan_channel(
                &handle,
                freq,
                args.packet_size,
                args.num_transfers,
                args.interval_ms,
            );

            results.push(result);
            freq += CHANNEL_WIDTH_HZ;
        }

        // Display results
        display_results(&results, 0);

        // Small pause between scans
        thread::sleep(Duration::from_millis(50));
    }

    // Cleanup - show cursor
    print!("\x1B[?25h");
    cursor_down(1);
    println!("\nScan stopped.");
}

fn main() {
    let args = Args::parse();

    let context = Context::new().expect("Could not create USB context");

    // Run scanner
    run_scanner(&context, &args);
}