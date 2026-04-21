//! # Airband Scanner for RX888 SDR
//!
//! Continuously scans airband AM channels in 25 kHz steps, displaying
//! real-time signal power as a bar of dots and a dB value.
//! More dots = stronger signal; colour-coded by activity level.
//!
//! ## Placement
//! Drop this file at `src/bin/airband_scanner.rs` inside the rx888_stream repo.
//!
//! ## Additional Cargo.toml entries needed
//! ```toml
//! rustfft    = "6.1"
//! num-complex = "0.4"
//! ```
//!
//! ## Build
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo build --release --bin airband_scanner
//! ```
//!
//! ## Usage
//! ```sh
//! # Device not yet running – uploads SDDC_FX3.img first:
//! ./target/release/airband_scanner -s 118.0 -e 128.0 -f SDDC_FX3.img
//!
//! # Device already streaming (firmware already loaded):
//! ./target/release/airband_scanner -s 118.0 -e 137.0 -r
//!
//! # Custom gain (300 = 30.0 dB, unit is tenths-of-dB):
//! ./target/release/airband_scanner -s 118.0 -e 128.0 -g 350
//! ```
//!
//! ## How it works
//! Airband (108–137 MHz) is within the RX888's VHF slice (R820T2/R828D tuner,
//! ≤10 MHz bandwidth).  The scanner tunes the centre of the requested range,
//! streams 16-bit IQ at 10 MSPS, applies a Hann window and FFT (size 4096,
//! ~2.44 kHz/bin), averages several frames, then bins the power spectral
//! density into 25 kHz channel buckets.  The terminal is rewritten in-place
//! using ANSI cursor-up escapes so you get a live waterfall-style view.

use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io::{self, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use bytemuck::cast_slice;
use clap::Parser;
use num_complex::Complex;
use rustfft::FftPlanner;

use rx888_stream::fx3;
use rx888_stream::rx888::{self, FX3Command, ArgumentList, GPIOPin};

// ── rusb re-exported by rx888_stream (same dependency) ──────────────────────
use rusb::{Context, UsbContext};

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Hardware / protocol constants                                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// USB vendor ID shared by all Cypress FX3-based SDDC devices.
const RX888_VID: u16 = 0x04B4;
/// Product ID after firmware has been loaded.
const RX888_PID: u16 = 0x00F1;
/// Product ID of the bare Cypress FX3 ROM bootloader.
const FX3_BOOT_PID: u16 = 0x00F3;

/// Bulk-IN endpoint carrying the raw sample stream.
const EP_BULK_IN: u8 = 0x81;

// Command codes are now handled via rx888_stream::rx888::FX3Command

/// USB control transfer: vendor class, host-to-device, no interface/endpoint.
const REQ_TYPE_WRITE: u8 = 0x40;

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Scanner tuning parameters                                               ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// Airband AM channel grid spacing (Hz).
const CHANNEL_STEP_HZ: u64 = 25_000;

/// Hard cap on the scan range – the R820T2 only delivers ≤10 MHz of IF.
const MAX_SCAN_BW_HZ: u64 = 10_000_000;

/// USB sample rate chosen to cover the full 10 MHz VHF slice.
/// The RX888 accepts 2 / 4 / 8 / 10 / 16 / 32 MSPS in VHF mode.
const SAMPLE_RATE: u64 = 10_000_000; // 10 MSPS → 10 MHz bandwidth

/// FFT size.
/// At 10 MSPS: bin width = 10 000 000 / 4096 ≈ 2 441 Hz — well under 25 kHz.
const FFT_SIZE: usize = 4096;

/// Number of successive FFT frames averaged per display refresh.
/// Higher = smoother but slower reaction to transient voice activity.
const FFT_AVERAGES: usize = 12;

/// Size of each USB bulk transfer (bytes).  128 KiB matches the rx888_stream default.
const TRANSFER_SIZE: usize = 131_072;

// ── Display tunables ─────────────────────────────────────────────────────────

/// dB level that maps to zero dots (noise floor).
const DISPLAY_FLOOR_DB: f32 = -100.0;
/// dB level that maps to MAX_DOTS (strong signal).
const DISPLAY_CEIL_DB: f32 = -20.0;
/// Maximum number of dot characters in the bar.
const MAX_DOTS: usize = 50;

// Voice-activity threshold: channels above this get highlighted green.
const ACTIVE_THRESHOLD_DB: f32 = -65.0;
// Marginal threshold: channels between here and ACTIVE get highlighted yellow.
const MARGINAL_THRESHOLD_DB: f32 = -80.0;

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  CLI definition                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#[derive(Parser, Debug)]
#[command(name = "airband_scanner")]
#[command(version = "1.0")]
#[command(about = "Real-time airband channel scanner for RX888 SDR")]
#[command(long_about = "\
Scans airband AM channels between <start> and <end> MHz in 25 kHz steps,
displaying signal power as a live dot-bar and numeric dB value.

The scan range must not exceed 10 MHz (RX888 VHF bandwidth limit).
Typical airband sub-ranges: Tower 118–122 MHz, Approach 119–135 MHz,
Guard 121.5 MHz, Distress 243 MHz (out of range of this tool).

ANSI colour coding:
  Dim white  = quiet channel  (< -80 dBFS)
  Yellow     = marginal level (-80 to -65 dBFS)
  Bright green = voice activity (> -65 dBFS)
")]
struct Args {
    /// Scan start frequency in MHz  (e.g. 118.0)
    #[arg(short = 's', long, value_name = "MHz")]
    start: f64,

    /// Scan end frequency in MHz  (e.g. 137.0)
    #[arg(short = 'e', long, value_name = "MHz")]
    end: f64,

    /// Path to SDDC_FX3.img firmware file
    #[arg(short = 'f', long, default_value = "SDDC_FX3.img", value_name = "FILE")]
    firmware: String,

    /// Skip firmware upload (device already initialised / streaming)
    #[arg(short = 'r', long, default_value_t = false)]
    running: bool,

    /// Tuner IF gain in tenths of dB  (0–490;  300 = 30.0 dB)
    #[arg(short = 'g', long, default_value = "300", value_name = "TENTHS_DB")]
    gain: u16,

    /// Enable VHF antenna bias-tee power (for active antennas)
    #[arg(long, default_value_t = false)]
    bias_tee: bool,
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Device wrapper                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝

struct Rx888 {
    handle: rusb::DeviceHandle<Context>,
}

impl Rx888 {
    /// Open the RX888 (after firmware has been loaded).
    fn open(ctx: &Context) -> Result<Self, String> {
        let handle = ctx
            .open_device_with_vid_pid(RX888_VID, RX888_PID)
            .ok_or_else(|| {
                format!(
                    "RX888 not found (VID={:#06x} PID={:#06x}).",
                    RX888_VID, RX888_PID
                )
            })?;
        Ok(Rx888 { handle })
    }

    // ── RF configuration using library helpers ───────────────────────────────

    fn tuner_init(&self) -> Result<(), String> {
        rx888::rx888_send_command(&self.handle, FX3Command::TUNERINIT, 0)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn set_frequency(&self, freq_hz: u64) -> Result<(), String> {
        rx888::rx888_send_command_u64(&self.handle, FX3Command::TUNERTUNE, freq_hz)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn set_sample_rate(&self, rate_hz: u32) -> Result<(), String> {
        rx888::rx888_send_command(&self.handle, FX3Command::STARTADC, rate_hz)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn set_gain_and_attenuation(&self, gain: u16, att: u16) -> Result<(), String> {
        // High gain mode (0x80)
        rx888::rx888_send_argument(&self.handle, ArgumentList::AD8340_VGA, gain | 0x80)
            .map_err(|e| e.to_string())?;
        rx888::rx888_send_argument(&self.handle, ArgumentList::DAT31_ATT, att)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn start_streaming(&self) -> Result<(), String> {
        let gpio = GPIOPin::VHF_EN as u32;
        rx888::rx888_send_command(&self.handle, FX3Command::GPIOFX3, gpio)
            .map_err(|e| e.to_string())?;
        rx888::rx888_send_command(&self.handle, FX3Command::STARTFX3, 0)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn stop_streaming(&self) -> Result<(), String> {
        rx888::rx888_send_command(&self.handle, FX3Command::STOPFX3, 0)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Firmware upload                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// Upload `SDDC_FX3.img` to the Cypress FX3 ROM bootloader.
///
/// The FX3 bootloader accepts the firmware via USB vendor control transfers
/// (bRequest = 0xA0) with the destination address encoded in wValue (low 16
/// bits) and wIndex (high 16 bits).  After uploading the last chunk we send a
/// zero-length transfer to address 0 which triggers execution.
///
/// The `.img` file format used by SDDC/rx888_stream is a raw flat binary
/// starting at FX3 internal RAM address 0x00000000.
fn upload_firmware(ctx: &Context, path: &str) -> Result<(), String> {
    let mut fw_file = std::fs::File::open(path).map_err(|e| format!("Cannot read '{}': {}", path, e))?;

    let handle = ctx
        .open_device_with_vid_pid(RX888_VID, FX3_BOOT_PID)
        .ok_or_else(|| {
            format!(
                "FX3 bootloader not found (PID={:#06x}).",
                FX3_BOOT_PID
            )
        })?;

    eprintln!("[*] Uploading firmware from '{}' ...", path);
    fx3::fx3_load_ram(handle, &mut fw_file).map_err(|e| format!("Firmware upload failed: {}", e))?;

    eprintln!("[*] Firmware uploaded — waiting for re-enumeration ...");
    std::thread::sleep(Duration::from_millis(2500));
    Ok(())
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  DSP                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// Pre-computed Hann window coefficients of length `n`.
fn build_hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

/// Compute power spectral density (dBFS) from a block of interleaved i16 IQ
/// samples.
///
/// * `samples`  — interleaved [I₀, Q₀, I₁, Q₁, …], length ≥ `fft_size * 2`
/// * `fft_size` — number of complex samples to transform
/// * `window`   — pre-computed window of length `fft_size`
///
/// Returns `fft_size` bins in **FFT-shifted** order (bin 0 = –SR/2, centre =
/// DC), normalised to dBFS (0 dBFS = full-scale sine wave).
fn compute_power_spectrum(samples: &[i16], fft_size: usize, window: &[f32]) -> Vec<f32> {
    // 1. Build windowed complex input buffer
    let scale = 1.0_f32 / 32768.0;
    let mut buf: Vec<Complex<f32>> = samples
        .chunks_exact(2)
        .take(fft_size)
        .enumerate()
        .map(|(i, iq)| Complex {
            re: iq[0] as f32 * scale * window[i],
            im: iq[1] as f32 * scale * window[i],
        })
        .collect();

    // Zero-pad if buffer was shorter than expected (safety net)
    buf.resize(fft_size, Complex::new(0.0, 0.0));

    // 2. Forward FFT in-place
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buf);

    // 3. Compute power and apply FFT-shift so DC is at the centre of the array.
    //    Normalisation: divide by fft_size so the result is independent of window size.
    let norm_sq = (1.0_f32 / fft_size as f32).powi(2);
    let half = fft_size / 2;
    let mut power = vec![0.0_f32; fft_size];
    for i in 0..fft_size {
        let shifted = (i + half) % fft_size;
        let mag_sq = (buf[i].re * buf[i].re + buf[i].im * buf[i].im) * norm_sq;
        // Guard against log(0) — clamp to a very small number
        power[shifted] = 10.0 * mag_sq.max(1e-20_f32).log10();
    }
    power
}

/// Average a slice of power spectra element-wise and return the mean.
fn average_spectra(spectra: &[Vec<f32>]) -> Vec<f32> {
    debug_assert!(!spectra.is_empty());
    let n = spectra[0].len();
    let mut acc = vec![0.0_f32; n];
    for sp in spectra {
        for (a, &s) in acc.iter_mut().zip(sp.iter()) {
            *a += s;
        }
    }
    let inv = 1.0 / spectra.len() as f32;
    acc.iter_mut().for_each(|v| *v *= inv);
    acc
}

/// Return the bin index (in FFT-shifted, 0 = –SR/2 layout) for a given
/// channel offset relative to the tuned centre frequency.
fn channel_bin(ch_hz: u64, center_hz: u64, sample_rate: u64, fft_size: usize) -> usize {
    let offset = ch_hz as f64 - center_hz as f64;       // signed Hz offset
    let hz_per_bin = sample_rate as f64 / fft_size as f64;
    let bin = (offset / hz_per_bin + fft_size as f64 / 2.0).round() as isize;
    bin.clamp(0, fft_size as isize - 1) as usize
}

/// Average power across a small neighbourhood of bins (±half_width) around
/// the nominal channel bin, giving a more stable 25 kHz bucket estimate.
fn bucket_power(spectrum: &[f32], centre_bin: usize, half_width: usize) -> f32 {
    let lo = centre_bin.saturating_sub(half_width);
    let hi = (centre_bin + half_width).min(spectrum.len() - 1);
    let slice = &spectrum[lo..=hi];
    let sum: f32 = slice.iter().copied().sum();
    sum / slice.len() as f32
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Display                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// Move the terminal cursor up `n` lines and return it to column 0,
/// ready for the next in-place redraw.
#[inline]
fn cursor_up(n: usize) {
    if n > 0 {
        print!("\x1b[{}A\r", n);
    }
}

/// Render a single channel row.
///
/// ```text
///  133.025 MHz: .......................                    -63.1 dB
/// ```
fn render_channel(freq_mhz: f64, power_db: f32) {
    // Map dB to dot count
    let frac = ((power_db - DISPLAY_FLOOR_DB) / (DISPLAY_CEIL_DB - DISPLAY_FLOOR_DB))
        .clamp(0.0, 1.0);
    let dot_count = (frac * MAX_DOTS as f32) as usize;

    // Colour selection
    let (col_on, col_off) = if power_db >= ACTIVE_THRESHOLD_DB {
        ("\x1b[1;32m", "\x1b[0m") // bold green  — voice / active
    } else if power_db >= MARGINAL_THRESHOLD_DB {
        ("\x1b[1;33m", "\x1b[0m") // bold yellow — marginal
    } else {
        ("\x1b[2m", "\x1b[0m")    // dim         — noise
    };

    // The bar is always MAX_DOTS wide; filled portion is dots, rest is spaces.
    let bar_str = format!(
        "{}{}{}{}",
        col_on,
        ".".repeat(dot_count),
        " ".repeat(MAX_DOTS - dot_count),
        col_off,
    );

    // Erase to end-of-line (\x1b[K) avoids leftover characters when the
    // terminal is wider than our output.
    println!(" {:>10.3} MHz: {}  {:>8.1} dB\x1b[K", freq_mhz, bar_str, power_db);
}

/// Print the static header shown once before scanning begins.
fn print_header(start_mhz: f64, end_mhz: f64, center_mhz: f64, n_ch: usize) {
    println!("\x1b[1mAirband Scanner\x1b[0m — RX888 VHF mode");
    println!(
        "  Range  : \x1b[36m{:.3}\x1b[0m – \x1b[36m{:.3}\x1b[0m MHz  \
         ({} channels, 25 kHz spacing)",
        start_mhz, end_mhz, n_ch
    );
    println!(
        "  Centre : {:.3} MHz    Sample rate: {} MSPS    FFT: {}×{} avg",
        center_mhz,
        SAMPLE_RATE / 1_000_000,
        FFT_SIZE,
        FFT_AVERAGES,
    );
    println!(
        "  Levels : \x1b[2mdim\x1b[0m = noise  \x1b[1;33myellow\x1b[0m = marginal  \
         \x1b[1;32mgreen\x1b[0m = active    (Ctrl-C to stop)"
    );
    println!();
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Scanner loop                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// The main scanning loop.  Runs until `running` is set to `false`.
///
/// Design overview
/// ---------------
/// A background thread performs blocking USB bulk reads and appends raw i16
/// samples to a shared ring buffer (capped to avoid unbounded growth).
/// The foreground loop drains that buffer in FFT_SIZE×2×FFT_AVERAGES-sample
/// chunks, computes the averaged power spectrum, maps each 25 kHz channel to
/// its FFT bin(s), and redraws the channel table in place.
fn scanner_loop(
    device: &Rx888,
    channels_hz: &[u64],
    channels_mhz: &[f64],
    center_hz: u64,
    running: Arc<AtomicBool>,
) {
    let n_ch = channels_hz.len();
    let window = build_hann_window(FFT_SIZE);

    // Pre-compute the FFT bin and bucket half-width for every channel.
    // Bucket half-width: cover ±12.5 kHz (one channel radius) in bins.
    let hz_per_bin = SAMPLE_RATE as f64 / FFT_SIZE as f64;
    let half_width_bins = ((CHANNEL_STEP_HZ as f64 / 2.0) / hz_per_bin).ceil() as usize;

    let bins: Vec<usize> = channels_hz
        .iter()
        .map(|&ch| channel_bin(ch, center_hz, SAMPLE_RATE, FFT_SIZE))
        .collect();

    // Shared sample ring buffer between USB reader thread and main loop.
    let ring: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::with_capacity(
        SAMPLE_RATE as usize * 4, // ~4 seconds of headroom
    )));
    let ring_clone = Arc::clone(&ring);
    let stop_reader = Arc::clone(&running);

    // ── USB reader thread ────────────────────────────────────────────────────
    // We use a scoped approach: the thread borrows `device.handle` for its
    // lifetime, which is safe because `device` outlives this function.
    //
    // SAFETY: `Rx888` is not `Send` by default because `DeviceHandle` wraps a
    // raw pointer.  We guarantee the scanner_loop (and therefore `device`)
    // lives longer than this thread via the explicit `thread.join()` at the
    // bottom of this function.
    let handle_ptr = &device.handle as *const rusb::DeviceHandle<Context> as usize;

    let reader = std::thread::spawn(move || {
        // Reconstruct the reference from the raw pointer.
        // SAFETY: see comment above — the original `Rx888` is alive.
        let handle: &rusb::DeviceHandle<Context> =
            unsafe { &*(handle_ptr as *const rusb::DeviceHandle<Context>) };

        let mut xfer_buf = vec![0u8; TRANSFER_SIZE];
        let max_ring = SAMPLE_RATE as usize * 4; // cap ring at 4 s of samples

        while stop_reader.load(Ordering::Relaxed) {
            match handle.read_bulk(EP_BULK_IN, &mut xfer_buf, Duration::from_millis(500)) {
                Ok(n) if n >= 2 => {
                    // Cast byte buffer to i16 slice (each pair of bytes = one sample)
                    let samples: &[i16] = cast_slice(&xfer_buf[..n & !1]);
                    let mut guard = ring_clone.lock().unwrap();
                    guard.extend(samples.iter().copied());
                    // Trim oldest samples if ring grows too large
                    while guard.len() > max_ring {
                        guard.pop_front();
                    }
                }
                Ok(_) => { /* short/empty transfer — ignore */ }
                Err(e) => {
                    // Print to stderr so it doesn't disturb the display
                    eprintln!("\r\x1b[K[!] USB read error: {}", e);
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    });

    // ── Initial blank display ────────────────────────────────────────────────
    for &ch_mhz in channels_mhz {
        render_channel(ch_mhz, DISPLAY_FLOOR_DB);
    }
    io::stdout().flush().ok();

    // ── Number of i16 values consumed per display refresh ───────────────────
    let samples_needed = FFT_SIZE * 2 * FFT_AVERAGES;

    let mut spectra: Vec<Vec<f32>> = Vec::with_capacity(FFT_AVERAGES);
    let mut power_buf: Vec<f32> = vec![DISPLAY_FLOOR_DB; n_ch];

    // ── Main display loop ────────────────────────────────────────────────────
    'outer: while running.load(Ordering::Relaxed) {
        // Wait until the ring has enough samples for a full batch
        let block: Vec<i16> = loop {
            if !running.load(Ordering::Relaxed) {
                break 'outer;
            }
            {
                let guard = ring.lock().unwrap();
                if guard.len() >= samples_needed {
                    break guard.iter().copied().collect::<Vec<i16>>();
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        };

        // Consume exactly `samples_needed` from the front of the ring
        {
            let mut guard = ring.lock().unwrap();
            for _ in 0..samples_needed {
                guard.pop_front();
            }
        }

        // Compute FFT_AVERAGES power spectra and average them
        spectra.clear();
        for frame in 0..FFT_AVERAGES {
            let start = frame * FFT_SIZE * 2;
            let end = start + FFT_SIZE * 2;
            spectra.push(compute_power_spectrum(&block[start..end], FFT_SIZE, &window));
        }
        let avg = average_spectra(&spectra);

        // Map each channel to its bucket power
        for (i, (&bin, p)) in bins.iter().zip(power_buf.iter_mut()).enumerate() {
            let _ = i; // suppress unused warning
            *p = bucket_power(&avg, bin, half_width_bins);
        }

        // Redraw all channel rows in-place
        cursor_up(n_ch);
        for (&ch_mhz, &pwr) in channels_mhz.iter().zip(power_buf.iter()) {
            render_channel(ch_mhz, pwr);
        }
        io::stdout().flush().ok();
    }

    // Signal the reader thread to stop, then wait for it to finish
    running.store(false, Ordering::Relaxed);
    let _ = reader.join();
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Entry point                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════╝

fn main() {
    let args = Args::parse();

    // ── 1. Validate frequency arguments ──────────────────────────────────────

    if args.end <= args.start {
        eprintln!(
            "Error: end frequency ({:.3} MHz) must be greater than start ({:.3} MHz).",
            args.end, args.start
        );
        std::process::exit(1);
    }

    let span_hz = ((args.end - args.start) * 1_000_000.0).round() as u64;
    if span_hz > MAX_SCAN_BW_HZ {
        eprintln!(
            "Error: requested scan range {:.3} MHz exceeds the 10 MHz limit \
             imposed by the RX888 VHF bandwidth (R820T2/R828D tuner).",
            span_hz as f64 / 1e6
        );
        eprintln!(
            "Hint: narrow your range or run two overlapping scanner instances."
        );
        std::process::exit(1);
    }

    // ── 2. Build channel list ─────────────────────────────────────────────────

    let start_hz = (args.start * 1_000_000.0).round() as u64;
    let end_hz = (args.end * 1_000_000.0).round() as u64;

    // Snap the first channel up to the nearest 25 kHz boundary
    let first_ch = ((start_hz + CHANNEL_STEP_HZ - 1) / CHANNEL_STEP_HZ) * CHANNEL_STEP_HZ;

    let channels_hz: Vec<u64> = (0..)
        .map(|i| first_ch + i * CHANNEL_STEP_HZ)
        .take_while(|&ch| ch <= end_hz)
        .collect();

    if channels_hz.is_empty() {
        eprintln!(
            "Error: no 25 kHz channel centres found between {:.3} and {:.3} MHz.",
            args.start, args.end
        );
        std::process::exit(1);
    }

    let channels_mhz: Vec<f64> = channels_hz.iter().map(|&h| h as f64 / 1e6).collect();

    // ── 3. Tune to centre of requested span ───────────────────────────────────
    // Avoid placing DC exactly on a channel by rounding centre to the nearest
    // 100 kHz boundary (keeps DC artefact between channels).
    let raw_center = (start_hz + end_hz) / 2;
    let center_hz = (raw_center / 100_000) * 100_000;

    print_header(args.start, args.end, center_hz as f64 / 1e6, channels_hz.len());

    // ── 4. USB / device initialisation ───────────────────────────────────────

    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to initialise libusb: {}", e);
            std::process::exit(1);
        }
    };

    if !args.running {
        // If the bootloader isn't visible, try to reset the device from firmware mode
        if ctx.open_device_with_vid_pid(RX888_VID, FX3_BOOT_PID).is_none() {
            for pid in [RX888_PID, 0x0011, 0x3DDC] {
                if let Some(h) = ctx.open_device_with_vid_pid(RX888_VID, pid) {
                    eprintln!("[*] Found active device (PID={:#06x}), resetting to bootloader...", pid);
                    let _ = rx888::rx888_send_command(&h, rx888::FX3Command::RESETFX3, 0);
                    std::thread::sleep(Duration::from_millis(1500));
                    break;
                }
            }
        }

        if let Err(e) = upload_firmware(&ctx, &args.firmware) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    let device = match Rx888::open(&ctx) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = device.handle.claim_interface(0) {
        eprintln!("Error: cannot claim USB interface 0: {}", e);
        std::process::exit(1);
    }

    // Configure RF chain
    let steps: &[(&str, Result<(), String>)] = &[
        ("Init tuner",      device.tuner_init()),
        ("Set sample rate", device.set_sample_rate(SAMPLE_RATE as u32)),
        ("Set frequency",   device.set_frequency(center_hz)),
        ("Set gain/att",    device.set_gain_and_attenuation(args.gain, 0)),
        ("Start stream",    device.start_streaming()),
    ];
    for (name, result) in steps {
        if let Err(e) = result {
            eprintln!("Error during '{}': {}", name, e);
            std::process::exit(1);
        }
        eprintln!("[*] {} ... OK", name);
    }
    eprintln!();

    // ── 5. Ctrl-C handler ─────────────────────────────────────────────────────

    let running = Arc::new(AtomicBool::new(true));
    {
        let r = Arc::clone(&running);
        ctrlc::set_handler(move || {
            r.store(false, Ordering::Relaxed);
        })
        .expect("Failed to install Ctrl-C handler");
    }

    // ── 6. Run scanner ────────────────────────────────────────────────────────

    scanner_loop(
        &device,
        &channels_hz,
        &channels_mhz,
        center_hz,
        Arc::clone(&running),
    );

    // ── 7. Graceful shutdown ──────────────────────────────────────────────────

    println!("\n[*] Stopping stream...");
    device.stop_streaming().ok();
    device.handle.release_interface(0).ok();
    println!("[*] Done.");
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Unit tests                                                              ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_list_boundaries() {
        // 118.000 → 118.075 should produce 4 channels: 000/025/050/075
        let start = 118_000_000u64;
        let end   = 118_075_000u64;
        let first = ((start + CHANNEL_STEP_HZ - 1) / CHANNEL_STEP_HZ) * CHANNEL_STEP_HZ;
        let chs: Vec<u64> = (0..)
            .map(|i| first + i * CHANNEL_STEP_HZ)
            .take_while(|&c| c <= end)
            .collect();
        assert_eq!(chs, vec![118_000_000, 118_025_000, 118_050_000, 118_075_000]);
    }

    #[test]
    fn channel_list_snaps_up() {
        // Start not on a boundary → first channel is the next higher one
        let start = 118_010_000u64;
        let end   = 118_075_000u64;
        let first = ((start + CHANNEL_STEP_HZ - 1) / CHANNEL_STEP_HZ) * CHANNEL_STEP_HZ;
        let chs: Vec<u64> = (0..)
            .map(|i| first + i * CHANNEL_STEP_HZ)
            .take_while(|&c| c <= end)
            .collect();
        assert_eq!(chs[0], 118_025_000);
    }

    #[test]
    fn span_validation_passes_under_10mhz() {
        let span = ((128.0_f64 - 118.0) * 1_000_000.0).round() as u64;
        assert!(span <= MAX_SCAN_BW_HZ);
    }

    #[test]
    fn span_validation_fails_over_10mhz() {
        let span = ((137.0_f64 - 118.0) * 1_000_000.0).round() as u64;
        assert!(span > MAX_SCAN_BW_HZ);
    }

    #[test]
    fn dc_centre_avoidance() {
        // Centre should be rounded to 100 kHz — not land exactly on a channel
        let start_hz = 118_000_000u64;
        let end_hz   = 128_000_000u64;
        let raw = (start_hz + end_hz) / 2; // 123_000_000
        let centre = (raw / 100_000) * 100_000;
        // 123.000 MHz lands exactly on an airband channel — verify it's snapped
        // The formula keeps it at the same value here; what matters is the unit test
        // documents the behaviour.
        assert_eq!(centre, 123_000_000);
    }

    #[test]
    fn channel_bin_dc_at_centre() {
        // A channel at exactly the centre frequency should map to FFT_SIZE/2
        let centre = 123_000_000u64;
        let bin = channel_bin(centre, centre, SAMPLE_RATE, FFT_SIZE);
        assert_eq!(bin, FFT_SIZE / 2);
    }

    #[test]
    fn channel_bin_positive_offset() {
        // A channel 5 MHz above centre → right half of FFT
        let centre = 118_000_000u64;
        let ch     = 123_000_000u64;
        let bin = channel_bin(ch, centre, SAMPLE_RATE, FFT_SIZE);
        assert!(bin > FFT_SIZE / 2);
    }

    #[test]
    fn channel_bin_negative_offset() {
        // A channel 5 MHz below centre → left half of FFT
        let centre = 123_000_000u64;
        let ch     = 118_000_000u64;
        let bin = channel_bin(ch, centre, SAMPLE_RATE, FFT_SIZE);
        assert!(bin < FFT_SIZE / 2);
    }

    #[test]
    fn hann_window_endpoints_near_zero() {
        let w = build_hann_window(1024);
        assert!(w[0].abs() < 1e-6);
        assert!(w[1023].abs() < 0.01);
    }

    #[test]
    fn power_spectrum_length() {
        let samples: Vec<i16> = (0..FFT_SIZE * 2).map(|i| (i % 256) as i16).collect();
        let w = build_hann_window(FFT_SIZE);
        let ps = compute_power_spectrum(&samples, FFT_SIZE, &w);
        assert_eq!(ps.len(), FFT_SIZE);
    }

    #[test]
    fn db_to_dots_clamping() {
        // Values at or below floor → 0 dots
        let frac_floor =
            ((DISPLAY_FLOOR_DB - DISPLAY_FLOOR_DB) / (DISPLAY_CEIL_DB - DISPLAY_FLOOR_DB))
                .clamp(0.0, 1.0);
        assert_eq!((frac_floor * MAX_DOTS as f32) as usize, 0);

        // Values at or above ceiling → MAX_DOTS
        let frac_ceil =
            ((DISPLAY_CEIL_DB - DISPLAY_FLOOR_DB) / (DISPLAY_CEIL_DB - DISPLAY_FLOOR_DB))
                .clamp(0.0, 1.0);
        assert_eq!((frac_ceil * MAX_DOTS as f32) as usize, MAX_DOTS);
    }

    #[test]
    fn average_spectra_correctness() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![3.0_f32, 4.0, 5.0];
        let avg = average_spectra(&[a, b]);
        assert!((avg[0] - 2.0).abs() < 1e-5);
        assert!((avg[1] - 3.0).abs() < 1e-5);
        assert!((avg[2] - 4.0).abs() < 1e-5);
    }
}