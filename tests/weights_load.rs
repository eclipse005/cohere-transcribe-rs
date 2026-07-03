//! Stage 2 verification: load the real Cohere model and sanity-check the
//! weight shapes/recounts after the host-side rewrites + f16 upload.
//!
//! Gated behind `#[ignore]` + an env-gated path so it only runs when invoked
//! explicitly against a real model dir:
//!
//!     cargo test -p native-transcribe --features cuda --test weights_load \
//!         -- --ignored --nocapture
//!
//! Requires `COHERE_MODEL_DIR` env var (defaults to ../models/cohere-transcribe).

#![cfg(feature = "cuda")]

use native_transcribe::engine::CudaState;
use native_transcribe::weights_gpu::ModelWeights;

fn model_dir() -> std::path::PathBuf {
    std::env::var("COHERE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/cohere-transcribe")
        })
}

#[test]
#[ignore]
fn loads_full_model_with_expected_shapes() -> anyhow::Result<()> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip: model not found at {}", dir.display());
        return Ok(());
    }

    let cuda = CudaState::new(0)?;
    let m = ModelWeights::load(&dir, &cuda, false)?;

    // Top-level structural checks.
    assert_eq!(m.encoder.layers.len(), 48, "expected 48 encoder layers");
    assert_eq!(m.decoder.layers.len(), 8, "expected 8 decoder layers");

    // Encoder layer fused QKV: rows = 3*d_model = 3*1280 = 3840, cols = 1280.
    let qkv = &m.encoder.layers[0].qkv_w_t;
    assert_eq!(qkv.rows, 3 * 1280, "qkv rows");
    assert_eq!(qkv.cols, 1280, "qkv cols");
    assert_eq!(qkv.data.len(), 3 * 1280 * 1280, "qkv data len");

    // Encoder -> decoder projection: rows = 1024, cols = 1280.
    assert_eq!(m.encoder.enc_proj_w_t.rows, 1024);
    assert_eq!(m.encoder.enc_proj_w_t.cols, 1280);

    // Token embedding: rows = vocab = 16384, cols = hidden = 1024.
    assert_eq!(m.decoder.token_embedding.rows, 16384, "vocab");
    assert_eq!(m.decoder.token_embedding.cols, 1024, "hidden");

    // LM head weight-transposed: rows = vocab = 16384, cols = 1024.
    assert_eq!(m.decoder.lm_w_t.rows, 16384);
    assert_eq!(m.decoder.lm_w_t.cols, 1024);

    // Packed depthwise conv params: [channels=1280, 10].
    assert_eq!(m.encoder.layers[0].cdw_params.0.len(), 1280 * 10, "cdw_params packed len");

    // Pre-encoder out projection weight-transposed: rows = d_model = 1280.
    assert_eq!(m.pre_encoder.out_w_t.rows, 1280, "pre-encode out rows");

    println!(
        "ok: 48 enc layers, 8 dec layers, qkv {}x{}, enc_proj 1024x1280, vocab 16384x1024",
        qkv.rows, qkv.cols
    );
    Ok(())
}
