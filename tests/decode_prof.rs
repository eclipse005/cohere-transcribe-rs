//! Profile the decoder hot loop: per-op cost breakdown + allocation overhead.
//! Runs the actual decode_step path to find where time goes.
#![cfg(feature = "cuda")]
use std::time::Instant;
use half::f16;
use native_transcribe::backend::{Backend, CpuBackendF16};
use native_transcribe::decoder::{Decoder, DecoderKvCache};
use native_transcribe::weights_gpu::ModelWeights;

#[test]
fn decode_step_profile() -> anyhow::Result<()> {
    let model_dir = std::env::var("COHERE_MODEL_DIR").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("../models/cohere-transcribe"));
    if !model_dir.exists() { return Ok(()); }

    let b = CpuBackendF16::new();
    let weights = ModelWeights::load(&model_dir, &b, false)?;
    let decoder = Decoder::new(&weights.decoder);

    let enc_tokens = 187usize;
    let enc_states = b.alloc_uninit(enc_tokens * 1024)?;
    let max_seq = 256usize;
    let mut cache = DecoderKvCache::new(&b, decoder.layers.len(), max_seq, enc_tokens)?;
    decoder.build_cross_kv_cache(&b, &mut cache, &enc_states)?;

    // Warm + measure decode_step_cached (the hot path) for 50 steps.
    let prompt_ids = [4i32, 1, 2, 5, 6, 7, 8, 9, 10];
    let _ = decoder.prefill(&b, &mut cache, &prompt_ids)?;
    let mut x = decoder.embed_one(&b, 16, prompt_ids.len())?;

    // Measure a single decode_step (all 8 layers).
    let iters = 50;
    let t0 = Instant::now();
    for step in 0..iters {
        let pos = prompt_ids.len() + step;
        let tok = decoder.decode_step_cached(&b, &mut cache, &x, pos)?;
        x = decoder.embed_one(&b, tok, pos + 1)?;
    }
    let total = t0.elapsed().as_secs_f64() / iters as f64;
    eprintln!("decode_step_cached (8 layers): {:.3} ms/step", total * 1e3);
    eprintln!("  → projected generate for 220 steps: {:.1} ms", total * 220.0 * 1e3);

    // Now measure just the M=1 GEMV (linear) cost vs everything else.
    // A decode step has: 6 linears (self_qkv, self_out, cross_q, cross_out, ffn_in, ffn_out) × 8 layers
    // = 48 GEMVs. Plus attention + layernorm + elementwise.
    let w = &weights.decoder.layers[0].self_qkv_w_t;
    let xone = b.alloc_uninit(1024)?;
    let t1 = Instant::now();
    for _ in 0..(iters * 48) {
        let _ = b.linear(&xone, 1, w)?;
    }
    let gemv = t1.elapsed().as_secs_f64() / (iters * 48) as f64;
    eprintln!("single M=1 GEMV (self_qkv 1024->3072): {:.4} ms", gemv * 1e3);
    eprintln!("  → 48 GEMVs/step projected: {:.3} ms", gemv * 48.0 * 1e3);

    Ok(())
}
