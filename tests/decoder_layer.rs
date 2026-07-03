//! 对拍 (parity) test: compare native GPU decoder layer against CPU reference.
//!
//! Requires a real model. Run with:
//!     cargo test -p native-transcribe --features cuda --test decoder_layer \
//!         -- --ignored --nocapture

#![cfg(feature = "cuda")]

use half::f16;
use native_transcribe::engine::CudaState;
use native_transcribe::decoder::DecoderLayer;
use native_transcribe::weights_gpu::ModelWeights;

const DEC_DIM: usize = 1024;
const DEC_HEADS: usize = 8;
const DEC_HEAD_DIM: usize = DEC_DIM / DEC_HEADS;
const DEC_FFN: usize = 4096;
const EPS: f32 = 1e-5;

fn model_dir() -> std::path::PathBuf {
    std::env::var("COHERE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/cohere-transcribe")
        })
}

// CPU reference
fn layer_norm_cpu(x: &[f32], w: &[f32], b: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + EPS).sqrt();
        for d in 0..dim {
            out[r * dim + d] = ((row[d] - mean) * inv_std) * w[d] + b[d];
        }
    }
    out
}

fn linear_cpu(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let m = x.len() / in_dim;
    let mut y = vec![0f32; m * out_dim];
    for i in 0..m {
        for o in 0..out_dim {
            let mut acc = 0f32;
            for k in 0..in_dim {
                acc += x[i * in_dim + k] * w[o * in_dim + k];
            }
            y[i * out_dim + o] = acc;
        }
    }
    y
}

fn linear_bias_cpu(x: &[f32], w: &[f32], bias: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let m = x.len() / in_dim;
    let mut y = vec![0f32; m * out_dim];
    for i in 0..m {
        for o in 0..out_dim {
            let mut acc = bias[o];
            for k in 0..in_dim {
                acc += x[i * in_dim + k] * w[o * in_dim + k];
            }
            y[i * out_dim + o] = acc;
        }
    }
    y
}

fn bias_residual_cpu(out: &[f32], bias: &[f32], residual: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            y[r * cols + c] = out[r * cols + c] + bias[c] + residual[r * cols + c];
        }
    }
    y
}

fn softmax_last_dim_cpu(x: &[f32], rows_ab: usize, dim: usize, causal: bool) -> Vec<f32> {
    let mut out = vec![0f32; rows_ab * dim];
    for r in 0..rows_ab {
        let q = r % (rows_ab / DEC_HEADS); // recover query position
        let row = &x[r * dim..(r + 1) * dim];
        let mut mx = f32::NEG_INFINITY;
        for d in 0..dim {
            if !causal || d <= q {
                mx = mx.max(row[d]);
            }
        }
        let mut sum = 0f32;
        for d in 0..dim {
            if !causal || d <= q {
                let e = (row[d] - mx).exp();
                out[r * dim + d] = e;
                sum += e;
            } else {
                out[r * dim + d] = 0.0;
            }
        }
        let inv = 1.0 / sum;
        for d in 0..dim {
            out[r * dim + d] *= inv;
        }
    }
    out
}

fn attention_cpu(
    input: &[f32], context: &[f32],
    q_w: &[f32], q_b: &[f32],
    k_w: &[f32], k_b: &[f32],
    v_w: &[f32], v_b: &[f32],
    out_w: &[f32], out_b: &[f32],
    residual: &[f32],
    causal: bool,
    tokens: usize, ctx_tokens: usize,
) -> Vec<f32> {
    // Q, K, V projections with bias
    let q = linear_bias_cpu(input, q_w, q_b, DEC_DIM, DEC_DIM);
    let k = linear_bias_cpu(context, k_w, k_b, DEC_DIM, DEC_DIM);
    let v = linear_bias_cpu(context, v_w, v_b, DEC_DIM, DEC_DIM);

    // Reshape to [heads, seq, head_dim]
    let to_heads = |data: &[f32], seq: usize| -> Vec<f32> {
        let mut out = vec![0f32; DEC_HEADS * seq * DEC_HEAD_DIM];
        for h in 0..DEC_HEADS {
            for s in 0..seq {
                for d in 0..DEC_HEAD_DIM {
                    out[(h * seq + s) * DEC_HEAD_DIM + d] = data[s * DEC_DIM + h * DEC_HEAD_DIM + d];
                }
            }
        }
        out
    };

    let q_h = to_heads(&q, tokens);
    let k_h = to_heads(&k, ctx_tokens);
    let v_h = to_heads(&v, ctx_tokens);

    // Scores: Q @ K^T
    let scale = (DEC_HEAD_DIM as f32).powf(-0.5);
    let mut scores = vec![0f32; DEC_HEADS * tokens * ctx_tokens];
    for h in 0..DEC_HEADS {
        for i in 0..tokens {
            for j in 0..ctx_tokens {
                let mut acc = 0f32;
                for d in 0..DEC_HEAD_DIM {
                    acc += q_h[(h * tokens + i) * DEC_HEAD_DIM + d]
                        * k_h[(h * ctx_tokens + j) * DEC_HEAD_DIM + d];
                }
                scores[(h * tokens + i) * ctx_tokens + j] = acc * scale;
            }
        }
    }

    // Softmax (causal or not)
    let attn = softmax_last_dim_cpu(&scores, DEC_HEADS * tokens, ctx_tokens, causal);

    // Attend: attn @ V
    let mut ctx = vec![0f32; DEC_HEADS * tokens * DEC_HEAD_DIM];
    for h in 0..DEC_HEADS {
        for i in 0..tokens {
            for d in 0..DEC_HEAD_DIM {
                let mut acc = 0f32;
                for j in 0..ctx_tokens {
                    acc += attn[(h * tokens + i) * ctx_tokens + j]
                        * v_h[(h * ctx_tokens + j) * DEC_HEAD_DIM + d];
                }
                ctx[(h * tokens + i) * DEC_HEAD_DIM + d] = acc;
            }
        }
    }

    // Merge heads: [heads, tokens, head_dim] → [tokens, DEC_DIM]
    let mut merged = vec![0f32; tokens * DEC_DIM];
    for h in 0..DEC_HEADS {
        for t in 0..tokens {
            for d in 0..DEC_HEAD_DIM {
                merged[t * DEC_DIM + h * DEC_HEAD_DIM + d] = ctx[(h * tokens + t) * DEC_HEAD_DIM + d];
            }
        }
    }

    // Output projection
    let out_proj = linear_cpu(&merged, out_w, DEC_DIM, DEC_DIM);
    bias_residual_cpu(&out_proj, out_b, residual, tokens, DEC_DIM)
}

fn decoder_layer_cpu(
    x: &[f32], encoder_states: &[f32],
    w: &DecoderLayerWeightsCpu,
    tokens: usize, enc_tokens: usize,
) -> Vec<f32> {
    // 1. Self-attention (causal)
    let normed = layer_norm_cpu(x, &w.norm1_w, &w.norm1_b, tokens, DEC_DIM);
    let x = attention_cpu(
        &normed, &normed,
        &w.self_q_w, &w.self_q_b,
        &w.self_k_w, &w.self_k_b,
        &w.self_v_w, &w.self_v_b,
        &w.self_out_w, &w.self_out_b,
        x, true, tokens, tokens,
    );

    // 2. Cross-attention
    let normed = layer_norm_cpu(&x, &w.norm2_w, &w.norm2_b, tokens, DEC_DIM);
    let x = attention_cpu(
        &normed, encoder_states,
        &w.cross_q_w, &w.cross_q_b,
        &w.cross_k_w, &w.cross_k_b,
        &w.cross_v_w, &w.cross_v_b,
        &w.cross_out_w, &w.cross_out_b,
        &x, false, tokens, enc_tokens,
    );

    // 3. FFN (ReLU)
    let normed = layer_norm_cpu(&x, &w.norm3_w, &w.norm3_b, tokens, DEC_DIM);
    let hidden = linear_bias_cpu(&normed, &w.ffn_in_w, &w.ffn_in_b, DEC_FFN, DEC_DIM);
    let hidden_relu: Vec<f32> = hidden.iter().map(|v| v.max(0.0)).collect();
    let out = linear_cpu(&hidden_relu, &w.ffn_out_w, DEC_DIM, DEC_FFN);
    bias_residual_cpu(&out, &w.ffn_out_b, &x, tokens, DEC_DIM)
}

struct DecoderLayerWeightsCpu {
    norm1_w: Vec<f32>, norm1_b: Vec<f32>,
    self_q_w: Vec<f32>, self_q_b: Vec<f32>,
    self_k_w: Vec<f32>, self_k_b: Vec<f32>,
    self_v_w: Vec<f32>, self_v_b: Vec<f32>,
    self_out_w: Vec<f32>, self_out_b: Vec<f32>,
    norm2_w: Vec<f32>, norm2_b: Vec<f32>,
    cross_q_w: Vec<f32>, cross_q_b: Vec<f32>,
    cross_k_w: Vec<f32>, cross_k_b: Vec<f32>,
    cross_v_w: Vec<f32>, cross_v_b: Vec<f32>,
    cross_out_w: Vec<f32>, cross_out_b: Vec<f32>,
    norm3_w: Vec<f32>, norm3_b: Vec<f32>,
    ffn_in_w: Vec<f32>, ffn_in_b: Vec<f32>,
    ffn_out_w: Vec<f32>, ffn_out_b: Vec<f32>,
}

fn make_random(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 37) as f32 / (u32::MAX as f32) * 2.0 - 1.0
    }).collect()
}

fn to_f16(v: &[f32]) -> Vec<f16> { v.iter().map(|x| f16::from_f32(*x)).collect() }

#[test]
#[ignore]
fn decoder_layer_parity() -> anyhow::Result<()> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip: model not found at {}", dir.display());
        return Ok(());
    }

    let cuda = CudaState::new(0)?;
    let model = ModelWeights::load(&dir, &cuda, false)?;
    let w = &model.decoder.layers[0];

    let tokens = 4usize;
    let enc_tokens = 6usize;
    let seed = 42u64;

    let x_f32 = make_random(seed, tokens * DEC_DIM);
    let enc_f32 = make_random(seed + 1, enc_tokens * DEC_DIM);

    // Download weights for CPU
    let d = |s: &cudarc::driver::safe::CudaSlice<f16>| -> Vec<f32> {
        cuda.download_f16(s).unwrap().iter().map(|h| h.to_f32()).collect()
    };
    let dw = |wt: &native_transcribe::tensor::GpuWeight| -> Vec<f32> {
        cuda.download_f16(&wt.data).unwrap().iter().map(|h| h.to_f32()).collect()
    };

    let cpu_w = DecoderLayerWeightsCpu {
        norm1_w: d(&w.norm1_w.0), norm1_b: d(&w.norm1_b.0),
        self_q_w: dw(&w.self_q_w_t), self_q_b: d(&w.self_q_b.0),
        self_k_w: dw(&w.self_k_w_t), self_k_b: d(&w.self_k_b.0),
        self_v_w: dw(&w.self_v_w_t), self_v_b: d(&w.self_v_b.0),
        self_out_w: dw(&w.self_out_w_t), self_out_b: d(&w.self_out_b.0),
        norm2_w: d(&w.norm2_w.0), norm2_b: d(&w.norm2_b.0),
        cross_q_w: dw(&w.cross_q_w_t), cross_q_b: d(&w.cross_q_b.0),
        cross_k_w: dw(&w.cross_k_w_t), cross_k_b: d(&w.cross_k_b.0),
        cross_v_w: dw(&w.cross_v_w_t), cross_v_b: d(&w.cross_v_b.0),
        cross_out_w: dw(&w.cross_out_w_t), cross_out_b: d(&w.cross_out_b.0),
        norm3_w: d(&w.norm3_w.0), norm3_b: d(&w.norm3_b.0),
        ffn_in_w: dw(&w.ffn_in_w_t), ffn_in_b: d(&w.ffn_in_b.0),
        ffn_out_w: dw(&w.ffn_out_w_t), ffn_out_b: d(&w.ffn_out_b.0),
    };

    // CPU
    let cpu_out = decoder_layer_cpu(&x_f32, &enc_f32, &cpu_w, tokens, enc_tokens);

    // GPU
    let x_gpu = cuda.upload_f16(&to_f16(&x_f32))?;
    let enc_gpu = cuda.upload_f16(&to_f16(&enc_f32))?;
    let layer = DecoderLayer { w: &model.decoder.layers[0] };
    let gpu_out = layer.forward(&cuda, &x_gpu, &enc_gpu, tokens, enc_tokens)?;
    cuda.synchronize()?;
    let gpu_f32: Vec<f32> = cuda.download_f16(&gpu_out)?.iter().map(|h| h.to_f32()).collect();

    let max_diff = cpu_out.iter().zip(&gpu_f32).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let max_val = cpu_out.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1.0);
    let rel = max_diff / max_val;

    println!("decoder parity: max_diff={max_diff:.6} max_val={max_val:.3} rel={rel:.6}");
    assert!(rel < 1e-2, "decoder layer mismatch: rel={rel:.6}");

    Ok(())
}