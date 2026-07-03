//! Stage 3.1 verification: f16 round-trip through the GPU (upload f32→f16,
//! download, sanity-check shape + finiteness).
//!
//!     cargo test -p native-transcribe --features cuda --test pre_encoder_bridge \
//!         -- --ignored --nocapture
//!
//! Requires the real model + fixtures (defaults to ../models/cohere-transcribe
//! and ../tests/fixtures).

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use native_transcribe::engine::CudaState;

fn model_dir() -> PathBuf {
    std::env::var("COHERE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/cohere-transcribe"))
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures")
}

#[test]
#[ignore]
fn f16_roundtrip_smoke() -> anyhow::Result<()> {
    let model_dir = model_dir();
    let audio = fixtures_dir().join("15s.wav");
    if !audio.exists() {
        eprintln!("skip: fixture not found at {}", audio.display());
        return Ok(());
    }

    // 1. CPU: WAV -> chunks -> log-mel features (pure Rust, self-contained).
    let prepared = native_transcribe::host::prepare_native_audio(
        &native_transcribe::host::NativePrepareOptions {
            model_dir: model_dir,
            audio: audio.clone(),
            language: "en".to_string(),
            punctuation: true,
        },
    )?;

    let chunk = prepared
        .chunks
        .first()
        .ok_or_else(|| anyhow::anyhow!("prepared audio had no chunks"))?;
    let features = &chunk.features;
    assert_eq!(features.channels, 128, "mel channels");
    println!(
        "prepared: frames={}, length={}, channels={}",
        features.frames, features.length, features.channels
    );

    // 2. Smoke-test: upload a synthetic f32 buffer as f16, download, verify
    //    round-trip accuracy. The pre-encoder CPU bridge (run_pre_encoder_cpu)
    //    depends on candle-core and will be reimplemented separately.
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();
    let f16_buf: Vec<half::f16> = data.iter().map(|v| half::f16::from_f32(*v)).collect();

    let cuda = CudaState::new(0)?;
    let gpu = cuda.upload_f16(&f16_buf)?;
    cuda.synchronize()?;
    let back = cuda.download_f16(&gpu)?;

    assert_eq!(back.len(), data.len(), "download len mismatch");
    let mut max_diff: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    for (a, b) in back.iter().zip(&data) {
        let af = a.to_f32();
        max_diff = max_diff.max((af - b).abs());
        max_abs = max_abs.max(b.abs());
    }
    let rel = if max_abs > 0.0 { max_diff / max_abs } else { 0.0 };
    assert!(
        rel < 2e-3,
        "f16 round-trip relative error {rel:.2e} too large (max_diff={max_diff:.4}, max_abs={max_abs:.4})"
    );

    println!(
        "ok: f16 round-trip max_diff={:.4} max_abs={:.4} rel={:.2e}",
        max_diff, max_abs, rel
    );
    Ok(())
}
