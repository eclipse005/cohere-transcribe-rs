//! CLI entry point: `native-transcribe transcribe --model-dir --audio --json`.
//!
//! Output schema matches the candle baseline `cohere-transcribe-rs` so
//! `tests/bench_baseline.ps1 -CudaBin <this binary>` works unchanged.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::transcribe::{Device, TranscribeReport, Transcriber};

#[derive(Debug, Parser)]
#[command(
    name = "native-transcribe",
    about = "Hand-written cudarc + cuBLAS + NVRTC Cohere ASR engine (sm_61+)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Transcribe one WAV file.
    Transcribe {
        /// Directory containing model.safetensors + tokenizer/config files.
        #[arg(long, default_value = "../models/cohere-transcribe")]
        model_dir: PathBuf,
        /// WAV file to transcribe.
        #[arg(long)]
        audio: PathBuf,
        /// Inference device: "cuda" (default, tuned), "cpu" (pure-Rust f32), or "auto".
        #[arg(long, default_value = "auto")]
        device: String,
        /// CUDA device ordinal (0 on a single-GPU host).
        #[arg(long, default_value_t = 0)]
        device_ordinal: usize,
        /// Decoder batch size (data-parallel chunks per decode batch).
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// Write full TranscribeReport JSON instead of plain transcript text.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Compute precision: "f16" (default, exact) or "int8" (DP4A, faster, slight accuracy loss).
        /// INT8 is CUDA-only; ignored on CPU.
        #[arg(long, default_value = "f16")]
        precision: String,
        /// Optional output file; same format as console (text by default, JSON with --json).
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Transcribe {
            model_dir,
            audio,
            device,
            device_ordinal: _,
            batch_size: _,
            json,
            precision,
            output,
        } => {
            let device = Device::parse(&device)?;
            let int8 = precision == "int8";
            let transcriber = Transcriber::load_with_device(&model_dir, device, int8)?;
            let report: TranscribeReport = transcriber.transcribe_file(&audio)?;
            let out = if json {
                serde_json::to_string_pretty(&report)?
            } else {
                report.text.clone()
            };
            if let Some(output) = output {
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&output, &out)?;
            }
            println!("{out}");
        }
    }
    Ok(())
}
