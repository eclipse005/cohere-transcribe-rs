//! Regression test: `DecoderLayer::forward_prefill` (the cached prefill path,
//! used by the real decoder hot path) must produce the same per-position
//! hidden states as `DecoderLayer::forward` (the non-cached path, which the
//! existing parity tests pin against a CPU reference).
//!
//! Why this test exists: `forward_prefill`'s self-attention used to apply the
//! head_dim^-0.5 scale twice — once in the `attention_qk` GEMM `alpha` and
//! again inside `causal_softmax` — collapsing the softmax and corrupting the
//! prefill (the first generated token diverged, and greedy decoding dropped
//! whole sentences). The non-cached `forward` scales once, so the two paths
//! diverged. After the fix (`alpha = 1.0` in the prefill Q@K^T GEMM) they
//! agree again. This test fails (max rel diff ~1.0) on the buggy build and
//! passes (< 2e-2, f16 rounding) on the fixed build.
//!
//! Requires a real model (for real weights). Run with:
//!     cargo test -p native-transcribe --features cuda \
//!         --test decoder_prefill_parity -- --ignored --nocapture

#![cfg(feature = "cuda")]

use native_transcribe::decoder::{Decoder, DecoderKvCache};
use native_transcribe::weights_gpu::ModelWeights;
use native_transcribe::backend::{Backend, CpuBackend};

const DEC_DIM: usize = 1024;

fn model_dir() -> std::path::PathBuf {
    std::env::var("COHERE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("models")
                .join("cohere-transcribe")
        })
}

fn make_random(seed: u64, n: usize) -> Vec<f32> {
    // Same LCG as tests/decoder_layer.rs — deterministic, zero-mean input.
    let mut state = seed;
    (0..n).map(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 37) as f32 / (u32::MAX as f32) * 2.0 - 1.0
    })
    .collect()
}

#[test]
#[ignore]
fn prefill_matches_non_cached_forward() -> anyhow::Result<()> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip: model not found at {}", dir.display());
        return Ok(());
    }

    let backend = CpuBackend::new();
    let weights = ModelWeights::load(&dir, &backend, false)?;
    let decoder = Decoder::new(&weights.decoder);

    // Small prompt + short encoder states (keeps the CPU run fast). Any sizes
    // work; we just need seq >= 2 to exercise the causal masking / scaling.
    let seq = 5usize;
    let enc_tokens = 7usize;
    let max_seq = 16usize;

    let x_f32 = make_random(7, seq * DEC_DIM);
    let enc_f32 = make_random(11, enc_tokens * DEC_DIM);
    let x = backend.upload_f16(&x_f32.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>())?;
    let enc = backend.upload_f16(&enc_f32.iter().map(|v| half::f16::from_f32(*v)).collect::<Vec<_>>())?;

    // --- Reference: non-cached forward (pinned by decoder_layer.rs parity) ---
    // Run every layer; we compare the final decoder hidden state.
    let mut ref_out = decoder.layers[0]
        .forward(&backend, &x, &enc, seq, enc_tokens)?;
    for li in 1..decoder.layers.len() {
        ref_out = decoder.layers[li]
            .forward(&backend, &ref_out, &enc, seq, enc_tokens)?;
    }

    // --- Hot path: cached prefill (the one the real transcriber uses) ---
    let mut cache = DecoderKvCache::new(&backend, decoder.layers.len(), max_seq, enc_tokens)?;
    decoder.build_cross_kv_cache(&backend, &mut cache, &enc)?;

    let mut pf_out = decoder.layers[0].forward_prefill(
        &backend, &x,
        &mut cache.self_k[0], &mut cache.self_v[0],
        &cache.cross_k[0], &cache.cross_v[0],
        seq, max_seq, enc_tokens,
    )?;
    for li in 1..decoder.layers.len() {
        pf_out = decoder.layers[li].forward_prefill(
            &backend, &pf_out,
            &mut cache.self_k[li], &mut cache.self_v[li],
            &cache.cross_k[li], &cache.cross_v[li],
            seq, max_seq, enc_tokens,
        )?;
    }

    // Compare. The prefill and non-cached paths use identical weights, inputs,
    // and (f16-storage / f32-accumulate) arithmetic, so on a correct build they
    // are bit-identical here (rel = 0.0). The buggy (double-scaled) build
    // diverges to rel ≈ 1.7e-2 on this input — well above the threshold. The
    // 5e-3 cut cleanly separates the two (≈3× headroom below the observed bug,
    // ≫ any rounding noise, which is zero on the fixed build).
    let ref_v: Vec<f32> = backend.download_f16(&ref_out)?.iter().map(|h| h.to_f32()).collect();
    let pf_v: Vec<f32> = backend.download_f16(&pf_out)?.iter().map(|h| h.to_f32()).collect();

    let max_diff = ref_v.iter().zip(&pf_v).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let max_val = ref_v.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1.0);
    let rel = max_diff / max_val;

    println!("prefill vs forward: max_diff={max_diff:.5} max_val={max_val:.3} rel={rel:.6}");
    assert!(
        rel < 5e-3,
        "prefill path diverges from non-cached forward: rel={rel:.6} \
         (likely a self-attention scaling regression in forward_prefill)",
    );
    Ok(())
}
