//! Smoke test for the CPU backend: load the model on CPU and run the pre-encoder
//! + a few encoder layers + a decoder prefill, checking no panics and that
//! dimensions flow through. Slower than the CUDA path (pure-Rust f32) but
//! verifies the generic forward code compiles and runs end-to-end on CPU.

#![cfg(feature = "cuda")]

use half::f16;

use native_transcribe::backend::{Backend, CpuBackend};
use native_transcribe::decoder::{Decoder, DecoderKvCache};
use native_transcribe::encoder::Encoder;
use native_transcribe::weights_gpu::ModelWeights;

/// Fast (no model) correctness check of the gemm-backed `linear` /
/// `attention_qk` / `attention_av` against naive scalar references. Pins the
/// gemm-crate quirks we work around: its β=0 path is broken (we emulate via
/// pre-zeroed dst + β=1) and its α argument is ignored (we post-scale qk).
#[test]
fn cpu_gemm_correctness() -> anyhow::Result<()> {
    let b = CpuBackend::new();
    let f = |x: f32| f16::from_f32(x);

    // linear: y[m,n] = x[m,k] @ W[n,k]^T
    let (m, n, k) = (3usize, 4usize, 5usize);
    let x_f16: Vec<f16> = (0..(m * k)).map(|i| f((i as f32) * 0.1 - 0.3)).collect();
    let x = b.upload_f16(&x_f16)?;
    let w_f16: Vec<f16> = (0..(n * k)).map(|i| f((i as f32) * 0.05 - 0.1)).collect();
    let w = b.upload_weight(&w_f16, n, k)?;
    let y = b.linear(&x, m, &w)?;
    let mut expect = vec![0f32; m * n];
    for i in 0..m { for j in 0..n {
        let mut acc = 0f32;
        for ki in 0..k { acc += x_f16[i * k + ki].to_f32() * w_f16[j * k + ki].to_f32(); }
        expect[i * n + j] = acc;
    }}
    let d = y.iter().zip(&expect).map(|(hv, e)| (hv - e).abs()).fold(0f32, f32::max);
    assert!(d < 1e-2, "linear mismatch {d}");

    // attention_qk with α != 1 (decoder scale): post-scale path must apply it.
    let (heads, mq, k_seq, d) = (2usize, 3usize, 4usize, 5usize);
    let q_f16: Vec<f16> = (0..(heads * mq * d)).map(|i| f((i as f32) * 0.03 - 0.4)).collect();
    let k_f16: Vec<f16> = (0..(heads * k_seq * d)).map(|i| f((i as f32) * 0.02 - 0.3)).collect();
    let q = b.upload_f16(&q_f16)?;
    let kbuf = b.upload_f16(&k_f16)?;
    let alpha = 0.5f32;
    let s = b.attention_qk(&q, &kbuf, heads, mq, k_seq, d, k_seq, alpha)?;
    let mut es = vec![0f32; heads * mq * k_seq];
    for h in 0..heads { for i in 0..mq { for nn in 0..k_seq {
        let mut acc = 0f32;
        for di in 0..d { acc += q_f16[h * mq * d + i * d + di].to_f32() * k_f16[h * k_seq * d + nn * d + di].to_f32(); }
        es[h * mq * k_seq + i * k_seq + nn] = alpha * acc;
    }}}
    let dq = s.iter().zip(&es).map(|(hv, e)| (hv - e).abs()).fold(0f32, f32::max);
    assert!(dq < 1e-1, "attention_qk (α=0.5) mismatch {dq}");

    // attention_av: out[h,i,d] = Σ_n a[h,i,n]*v[h,n,d]
    let a_f16: Vec<f16> = (0..(heads * mq * k_seq)).map(|i| f((i as f32) * 0.1)).collect();
    let v_f16: Vec<f16> = (0..(heads * k_seq * d)).map(|i| f((i as f32) * 0.04)).collect();
    let a = b.upload_f16(&a_f16)?;
    let v = b.upload_f16(&v_f16)?;
    let o = b.attention_av(&a, &v, heads, mq, k_seq, d, k_seq)?;
    let mut eo = vec![0f32; heads * mq * d];
    for h in 0..heads { for i in 0..mq { for di in 0..d {
        let mut acc = 0f32;
        for nn in 0..k_seq { acc += a_f16[h * mq * k_seq + i * k_seq + nn].to_f32() * v_f16[h * k_seq * d + nn * d + di].to_f32(); }
        eo[h * mq * d + i * d + di] = acc;
    }}}
    let da = o.iter().zip(&eo).map(|(hv, e)| (hv - e).abs()).fold(0f32, f32::max);
    assert!(da < 1e-1, "attention_av mismatch {da}");
    Ok(())
}

#[test]
fn cpu_pre_encoder_and_encoder_smoke() -> anyhow::Result<()> {
    let model_dir = std::env::var("COHERE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("models")
                .join("cohere-transcribe")
        });
    if !model_dir.exists() {
        eprintln!("skipping: model dir {:?} not found", model_dir);
        return Ok(());
    }

    let backend = CpuBackend::new();
    let weights = ModelWeights::load(&model_dir, &backend, false)?;

    // Fake mel features: [128 bins, frames] row-major. Small chunk.
    let frames = 200usize;
    let mel: Vec<f32> = (0..(128 * frames)).map(|i| (i as f32 * 0.0001).sin()).collect();

    let (x, tokens) = weights.pre_encoder.forward(&backend, &mel, frames)?;
    eprintln!("pre-encoder: tokens={tokens}, x len={}", <CpuBackend as Backend>::buf_len(&x));

    let pos = Encoder::generate_position_encoding(&backend, tokens)?;
    eprintln!("pos len={}", <CpuBackend as Backend>::buf_len(&pos));

    // Run just the first 2 encoder layers (full 48 is slow on CPU).
    let encoder = Encoder::new(&weights.encoder);
    let mut out = encoder.layers[0].forward(&backend, &x, &pos, tokens)?;
    eprintln!("layer0 out len={}", <CpuBackend as Backend>::buf_len(&out));
    out = encoder.layers[1].forward(&backend, &out, &pos, tokens)?;
    eprintln!("layer1 out len={}", <CpuBackend as Backend>::buf_len(&out));

    Ok(())
}

/// Decoder prefill on CPU — this is where the OOB was seen. Uses a tiny
/// fake encoder output (enc_tokens=4) and a 3-token prompt to exercise
/// split_qkv_batch_scatter + the attention/cross-attn/FFN paths.
#[test]
fn cpu_decoder_prefill_smoke() -> anyhow::Result<()> {
    let model_dir = std::env::var("COHERE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("models")
                .join("cohere-transcribe")
        });
    if !model_dir.exists() {
        eprintln!("skipping: model dir {:?} not found", model_dir);
        return Ok(());
    }

    let backend = CpuBackend::new();
    let weights = ModelWeights::load(&model_dir, &backend, false)?;
    let decoder = Decoder::new(&weights.decoder);

    let enc_tokens = 4usize;
    let enc_states = backend.alloc_uninit(enc_tokens * 1024)?;
    let max_seq = 32usize;
    let mut cache = DecoderKvCache::new(&backend, decoder.layers.len(), max_seq, enc_tokens)?;
    decoder.build_cross_kv_cache(&backend, &mut cache, &enc_states)?;

    let prompt_ids = [4i32, 1, 2]; // 3 tokens
    let tok = decoder.prefill(&backend, &mut cache, &prompt_ids)?;
    eprintln!("prefill predicted token: {tok}");
    Ok(())
}
