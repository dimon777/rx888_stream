// airband_scanner.rs - Fixed version
// Place this in src/bin/airband_scanner.rs or examples/airband_scanner.rs

use clap::Parser;
use rx888_stream::{Device, DeviceConfig, StreamConfig, DataFormat};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use rustfft::{FftPlanner, num_complex::Complex};

const CHANNEL_SPACING_MHZ: f64 = 0.025; // 25 kHz spacing for airband
const FFT_SIZE: usize = 16384; // For good frequency resolution
const SAMPLE_RATE_HZ: f64 = 32_000_000.0; // RX888 native rate
const AVERAGING_SECONDS: f64 = 0.1; // Fast update for voice activity
const MAX_BANDWIDTH_MHZ: f64 = 10.0; // Limit scanning range

#[derive(Parser, Debug)]
#[command(author, version, about = "RX888 Airband Scanner", long_about = None)]
struct Args {
    /// Start frequency in MHz (e.g., 133.0)
    #[arg(short)]
    s: f64,

    /// End frequency in MHz (e.g., 134.0)
    #[arg(short)]
    e: f64,
}

/// Represents a single scanning channel
struct Channel {
    frequency_mhz: f64,
    power_dbfs: f64, // Current power in dBFS
}

/// Scanner that manages continuous spectrum analysis
struct Scanner {
    device: Device,
    channels: Vec<Channel>,
    center_freq_hz: u64,
    running: Arc<AtomicBool>,
}

impl Scanner {
    /// Initialize the scanner with RX888 hardware
    fn new(start_mhz: f64, end_mhz: f64) -> Result<Self, Box<dyn std::error::Error>> {
        // Validate frequency range
        let bandwidth = end_mhz - start_mhz;
        if bandwidth > MAX_BANDWIDTH_MHZ {
            return Err(format!(
                "Frequency range {:.3} MHz exceeds maximum allowed {:.3} MHz",
                bandwidth,
                MAX_BANDWIDTH_MHZ
            ).into());
        }
        if bandwidth <= 0.0 {
            return Err("End frequency must be greater than start frequency".into());
        }

        // Create channel list with 25 kHz spacing
        let mut channels = Vec::new();
        let mut freq = start_mhz;
        while freq <= end_mhz + 1e-9 {
            channels.push(Channel {
                frequency_mhz: freq,
                power_dbfs: -120.0, // Initial noise floor
            });
            freq += CHANNEL_SPACING_MHZ;
        }

        let center_freq_hz = ((start_mhz + end_mhz) / 2.0 * 1_000_000.0) as u64;
        
        println!("Initializing RX888 with {} channels...", channels.len());
        println!("Center frequency: {:.3} MHz", center_freq_hz as f64 / 1_000_000.0);

        // Configure and open device
        let config = DeviceConfig {
            sample_rate: SAMPLE_RATE_HZ as u32,
            center_frequency: center_freq_hz,
            ..Default::default()
        };

        let device = Device::open(config)?;

        Ok(Self {
            device,
            channels,
            center_freq_hz,
            running: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Start continuous scanning
    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let running = self.running.clone();

        // Set up Ctrl-C handler for graceful exit
        ctrlc::set_handler(move || {
            println!("\nShutting down scanner...");
            running.store(false, Ordering::SeqCst);
        })?;

        // Configure stream
        let stream_config = StreamConfig {
            format: DataFormat::Float32,
            buffer_size: FFT_SIZE,
            ..Default::default()
        };

        // Start streaming
        let mut stream = self.device.start_stream(stream_config)?;

        println!("\nAirband Scanner Active");
        println!("Press Ctrl-C to stop\n");

        let mut last_update = Instant::now();
        let samples_per_update = (SAMPLE_RATE_HZ * AVERAGING_SECONDS) as usize;

        // Main scanning loop
        while self.running.load(Ordering::SeqCst) {
            // Read samples for power measurement
            let mut buffer = vec![0.0f32; samples_per_update];
            stream.read(&mut buffer)?;

            // Update channel powers using FFT
            self.update_channel_powers(&buffer)?;

            // Refresh display if enough time has passed
            if last_update.elapsed() >= Duration::from_secs_f64(AVERAGING_SECONDS) {
                self.display_channels()?;
                last_update = Instant::now();
            }
        }

        println!("Scanner stopped.");
        Ok(())
    }

    /// Update power measurements for all channels using FFT processing
    fn update_channel_powers(&mut self, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        // Prepare FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        // Convert samples to complex and apply window
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(FFT_SIZE)
            .enumerate()
            .map(|(i, &s)| {
                // Apply Hann window
                let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos());
                Complex::new(s * window, 0.0)
            })
            .collect();

        // Perform FFT
        fft.process(&mut buffer);

        // Calculate power for each channel
        let bin_resolution = SAMPLE_RATE_HZ / FFT_SIZE as f64;

        for channel in &mut self.channels {
            // Find FFT bin corresponding to channel frequency
            let freq_offset = (channel.frequency_mhz * 1_000_000.0) - (self.center_freq_hz as f64);
            let bin_index = ((freq_offset / bin_resolution).round() as isize).rem_euclid(FFT_SIZE as isize) as usize;

            // Calculate power in dBFS
            let power_linear = buffer[bin_index].norm_sqr();
            channel.power_dbfs = if power_linear > 0.0 {
                let power_f32 = 10.0 * (power_linear / FFT_SIZE as f32).log10();
                power_f32 as f64
            } else {
                -120.0 // Noise floor
            };
        }

        Ok(())
    }

    /// Display channel powers with visual bars using terminal escape sequences
    fn display_channels(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Move cursor to home position and clear screen
        print!("\x1B[2J\x1B[H");

        // Print header
        println!("RX888 Airband Scanner - {:.3} MHz to {:.3} MHz",
                 self.channels.first().map(|c| c.frequency_mhz).unwrap_or(0.0),
                 self.channels.last().map(|c| c.frequency_mhz).unwrap_or(0.0));
        println!("{:=<60}", "");

        // Display each channel
        for channel in &self.channels {
            // Map dBFS to bar length (-120 dB = 0 dots, -30 dB = 45 dots)
            let normalized_power = ((channel.power_dbfs + 120.0) / 90.0).clamp(0.0, 1.0);
            let bar_length = (normalized_power * 45.0) as usize;
            let bar = ".".repeat(bar_length);

            println!(
                "{:8.3} MHz: {:.<45} {:>6.1} dB",
                channel.frequency_mhz,
                bar,
                channel.power_dbfs
            );
        }

        io::stdout().flush()?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Validate frequencies are within airband (optional)
    if args.s < 118.0 || args.e > 137.0 {
        eprintln!("Warning: Frequencies outside typical airband (118-137 MHz)");
    }

    let mut scanner = Scanner::new(args.s, args.e)?;
    scanner.run()?;

    Ok(())
}