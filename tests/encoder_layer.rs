//! 对拍 (parity) test: compare native GPU encoder layer against CPU reference.
//!
//! Requires a real model (env `COHERE_MODEL_DIR` or `../models/cohere-transcribe`).
//! Run with:
//!
//!     cargo test -p native-transcribe --features cuda --test encoder_layer \
//!         -- --ignored --nocapture
//!
//! The CPU reference mirrors candle's `ResidentEncoderLayer::forward` CPU path
//! using f32 arithmetic. The GPU output (f16) is compared elementwise against
//! the CPU reference with a max-diff tolerance of ~5e-2 (f16 precision).

#![cfg(feature = "cuda")]

use half::f16;
use native_transcribe::engine::CudaState;
use native_transcribe::encoder::EncoderLayer;
use native_transcribe::weights_gpu::{EncoderLayerWeights, ModelWeights};

fn model_dir() -> std::path::PathBuf {
    std::env::var("COHERE_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/cohere-transcribe")
        })
}

// ============================================================================
// CPU reference (mirrors candle ResidentEncoderLayer::forward CPU path, f32)
// ============================================================================

const D_MODEL: usize = 1280;
const HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / HEADS;
const FFN_EXPAND: usize = 5120;
const EPS: f32 = 1e-5;

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

/// y = x @ w_t  (w_t is [out, in] weight-transposed, i.e. stored [in, out] row-major)
/// y = x @ w^T  where w is [out, in] row-major (matching GPU layout).
fn linear_cpu(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let m = x.len() / in_dim;
    let mut y = vec![0f32; m * out_dim];
    for i in 0..m {
        for o in 0..out_dim {
            let mut acc = 0f32;
            for k in 0..in_dim {
                // w is [out, in] row-major: element (o, k) = o * in_dim + k
                acc += x[i * in_dim + k] * w[o * in_dim + k];
            }
            y[i * out_dim + o] = acc;
        }
    }
    y
}

fn silu_bias_cpu(x: &[f32], bias: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let v = x[r * cols + c] + bias[c];
            let s = 1.0 / (1.0 + (-v).exp());
            out[r * cols + c] = v * s;
        }
    }
    out
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

fn softmax_last_dim_cpu(x: &[f32], rows_ab: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows_ab * dim];
    for r in 0..rows_ab {
        let row = &x[r * dim..(r + 1) * dim];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - mx).exp()).sum();
        let inv_sum = 1.0 / sum;
        for d in 0..dim {
            out[r * dim + d] = (row[d] - mx).exp() * inv_sum;
        }
    }
    out
}

fn rel_shift_cpu(bd: &[f32], heads: usize, q_len: usize, pos_len: usize) -> Vec<f32> {
    let mut out = vec![0f32; heads * q_len * pos_len];
    for h in 0..heads {
        for i_dst in 0..q_len {
            for j in 0..pos_len {
                let flat2 = i_dst * pos_len + j;
                let rp = flat2 / q_len;
                let c = flat2 % q_len;
                let r = rp + 1;
                let flat = r * q_len + c;
                let i_src = flat / (pos_len + 1);
                let slot = flat % (pos_len + 1);
                let val = if slot == 0 {
                    0.0f32
                } else {
                    bd[(h * q_len + i_src) * pos_len + (slot - 1)]
                };
                out[(h * q_len + i_dst) * pos_len + j] = val;
            }
        }
    }
    out
}

fn glu_depthwise_conv_cpu(
    x: &[f32],       // [tokens, 2*C]
    bias: &[f32],     // [2*C]
    cdw_params: &[f32], // [C, 10] packed
    tokens: usize,
    channels: usize,
) -> Vec<f32> {
    let c2 = channels * 2;
    let mut gated = vec![0f32; tokens * channels];
    for t in 0..tokens {
        for c in 0..channels {
            let left = x[t * c2 + c] + bias[c];
            let right = x[t * c2 + channels + c] + bias[channels + c];
            let sig = 1.0 / (1.0 + (-right).exp());
            gated[t * channels + c] = left * sig;
        }
    }
    let mut out = vec![0f32; tokens * channels];
    for t in 0..tokens {
        for c in 0..channels {
            let mut acc = cdw_params[c * 10 + 9]; // bias
            for k in 0..9 {
                let src_t = t as isize + k as isize - 4;
                if src_t >= 0 && src_t < tokens as isize {
                    let g = gated[src_t as usize * channels + c];
                    acc += g * cdw_params[c * 10 + k];
                }
            }
            let silu = acc * (1.0 / (1.0 + (-acc).exp()));
            out[t * channels + c] = silu;
        }
    }
    out
}

/// CPU reference for a single Conformer encoder layer.
fn encoder_layer_cpu(
    x: &[f32],           // [tokens, D_MODEL]
    pos: &[f32],         // [tokens, D_MODEL]
    tokens: usize,
    w: &CpuLayerWeights,
) -> Vec<f32> {
    // FFN1
    let normed = layer_norm_cpu(x, &w.ffn1_norm_w, &w.ffn1_norm_b, tokens, D_MODEL);
    let h = linear_cpu(&normed, &w.ffn1_l1_w_t, D_MODEL, FFN_EXPAND);
    let h_silu = silu_bias_cpu(&h, &w.ffn1_l1_b, tokens, FFN_EXPAND);
    let out = linear_cpu(&h_silu, &w.ffn1_l2_w_t, FFN_EXPAND, D_MODEL);
    let x = bias_residual_cpu(&out, &w.ffn1_l2_b, x, tokens, D_MODEL);

    // Self-attention
    let normed = layer_norm_cpu(&x, &w.att_norm_w, &w.att_norm_b, tokens, D_MODEL);
    let qkv = linear_cpu(&normed, &w.qkv_w_t, D_MODEL, 3 * D_MODEL);

    // Split qkv (with qkv bias)
    let mut q = vec![0f32; tokens * D_MODEL];
    let mut k = vec![0f32; tokens * D_MODEL];
    let mut v = vec![0f32; tokens * D_MODEL];
    for t in 0..tokens {
        let base = t * 3 * D_MODEL;
        for i in 0..D_MODEL {
            q[t * D_MODEL + i] = qkv[base + i] + w.qkv_b[i];
            k[t * D_MODEL + i] = qkv[base + D_MODEL + i] + w.qkv_b[D_MODEL + i];
            v[t * D_MODEL + i] = qkv[base + 2 * D_MODEL + i] + w.qkv_b[2 * D_MODEL + i];
        }
    }

    // Add pos biases
    let mut q_u = q.clone();
    let mut q_v = q.clone();
    for t in 0..tokens {
        for i in 0..D_MODEL {
            q_u[t * D_MODEL + i] += w.pos_bias_u[i];
            q_v[t * D_MODEL + i] += w.pos_bias_v[i];
        }
    }

    // pos_proj
    let pos_proj = linear_cpu(pos, &w.p_w_t, D_MODEL, D_MODEL);

    // Reshape to [heads, tokens, head_dim]
    let to_heads = |data: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; HEADS * tokens * HEAD_DIM];
        for h in 0..HEADS {
            for t in 0..tokens {
                for d in 0..HEAD_DIM {
                    out[(h * tokens + t) * HEAD_DIM + d] = data[t * D_MODEL + h * HEAD_DIM + d];
                }
            }
        }
        out
    };

    let q_u_h = to_heads(&q_u);
    let q_v_h = to_heads(&q_v);
    let k_h = to_heads(&k);
    let v_h = to_heads(&v);
    let p_h = to_heads(&pos_proj);

    // Attention scores (per head)
    let mut ac = vec![0f32; HEADS * tokens * tokens];
    let mut bd = vec![0f32; HEADS * tokens * tokens];
    for h in 0..HEADS {
        for i in 0..tokens {
            for j in 0..tokens {
                let mut ac_val = 0f32;
                let mut bd_val = 0f32;
                for d in 0..HEAD_DIM {
                    ac_val += q_u_h[(h * tokens + i) * HEAD_DIM + d]
                        * k_h[(h * tokens + j) * HEAD_DIM + d];
                    bd_val += q_v_h[(h * tokens + i) * HEAD_DIM + d]
                        * p_h[(h * tokens + j) * HEAD_DIM + d];
                }
                ac[(h * tokens + i) * tokens + j] = ac_val;
                bd[(h * tokens + i) * tokens + j] = bd_val;
            }
        }
    }

    let shifted = rel_shift_cpu(&bd, HEADS, tokens, tokens);
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let mut scores = vec![0f32; HEADS * tokens * tokens];
    for i in 0..scores.len() {
        scores[i] = (ac[i] + shifted[i]) * scale;
    }
    let attn = softmax_last_dim_cpu(&scores, HEADS * tokens, tokens);

    // Attend
    let mut ctx = vec![0f32; HEADS * tokens * HEAD_DIM];
    for h in 0..HEADS {
        for i in 0..tokens {
            for d in 0..HEAD_DIM {
                let mut acc = 0f32;
                for j in 0..tokens {
                    acc += attn[(h * tokens + i) * tokens + j]
                        * v_h[(h * tokens + j) * HEAD_DIM + d];
                }
                ctx[(h * tokens + i) * HEAD_DIM + d] = acc;
            }
        }
    }

    // Merge heads
    let mut merged = vec![0f32; tokens * D_MODEL];
    for h in 0..HEADS {
        for t in 0..tokens {
            for d in 0..HEAD_DIM {
                merged[t * D_MODEL + h * HEAD_DIM + d] = ctx[(h * tokens + t) * HEAD_DIM + d];
            }
        }
    }

    let attn_out = linear_cpu(&merged, &w.out_w_t, D_MODEL, D_MODEL);
    let x = bias_residual_cpu(&attn_out, &w.out_b, &x, tokens, D_MODEL);

    // Conv module
    let normed = layer_norm_cpu(&x, &w.conv_norm_w, &w.conv_norm_b, tokens, D_MODEL);
    let conv_in = linear_cpu(&normed, &w.cpw1_w_t, D_MODEL, 2 * D_MODEL);
    let conv_mid = glu_depthwise_conv_cpu(&conv_in, &w.cpw1_b, &w.cdw_params, tokens, D_MODEL);
    let conv_out = linear_cpu(&conv_mid, &w.cpw2_w_t, D_MODEL, D_MODEL);
    let x = bias_residual_cpu(&conv_out, &w.cpw2_b, &x, tokens, D_MODEL);

    // FFN2
    let normed = layer_norm_cpu(&x, &w.ffn2_norm_w, &w.ffn2_norm_b, tokens, D_MODEL);
    let h = linear_cpu(&normed, &w.ffn2_l1_w_t, D_MODEL, FFN_EXPAND);
    let h_silu = silu_bias_cpu(&h, &w.ffn2_l1_b, tokens, FFN_EXPAND);
    let out = linear_cpu(&h_silu, &w.ffn2_l2_w_t, FFN_EXPAND, D_MODEL);
    let x = bias_residual_cpu(&out, &w.ffn2_l2_b, &x, tokens, D_MODEL);

    // Final LN
    layer_norm_cpu(&x, &w.out_norm_w, &w.out_norm_b, tokens, D_MODEL)
}

// ============================================================================
// CPU-side mirror of the GPU weights (f32, for reference computation)
// ============================================================================

struct CpuLayerWeights {
    ffn1_norm_w: Vec<f32>,
    ffn1_norm_b: Vec<f32>,
    ffn1_l1_w_t: Vec<f32>,
    ffn1_l1_b: Vec<f32>,
    ffn1_l2_w_t: Vec<f32>,
    ffn1_l2_b: Vec<f32>,
    att_norm_w: Vec<f32>,
    att_norm_b: Vec<f32>,
    qkv_w_t: Vec<f32>,
    qkv_b: Vec<f32>,
    pos_bias_u: Vec<f32>,
    pos_bias_v: Vec<f32>,
    p_w_t: Vec<f32>,
    out_w_t: Vec<f32>,
    out_b: Vec<f32>,
    conv_norm_w: Vec<f32>,
    conv_norm_b: Vec<f32>,
    cpw1_w_t: Vec<f32>,
    cpw1_b: Vec<f32>,
    cdw_params: Vec<f32>,
    cpw2_w_t: Vec<f32>,
    cpw2_b: Vec<f32>,
    ffn2_norm_w: Vec<f32>,
    ffn2_norm_b: Vec<f32>,
    ffn2_l1_w_t: Vec<f32>,
    ffn2_l1_b: Vec<f32>,
    ffn2_l2_w_t: Vec<f32>,
    ffn2_l2_b: Vec<f32>,
    out_norm_w: Vec<f32>,
    out_norm_b: Vec<f32>,
}

impl CpuLayerWeights {
    fn from_gpu(cuda: &CudaState, w: &EncoderLayerWeights) -> anyhow::Result<Self> {
        let d = |s: &cudarc::driver::safe::CudaSlice<f16>| -> anyhow::Result<Vec<f32>> {
            let v = cuda.download_f16(s)?;
            Ok(v.iter().map(|h| h.to_f32()).collect())
        };
        let dw = |wt: &native_transcribe::tensor::GpuWeight| -> anyhow::Result<Vec<f32>> {
            let v = cuda.download_f16(&wt.data)?;
            Ok(v.iter().map(|h| h.to_f32()).collect())
        };
        Ok(Self {
            ffn1_norm_w: d(&w.ffn1_norm_w.0)?,
            ffn1_norm_b: d(&w.ffn1_norm_b.0)?,
            ffn1_l1_w_t: dw(&w.ffn1_l1_w_t)?,
            ffn1_l1_b: d(&w.ffn1_l1_b.0)?,
            ffn1_l2_w_t: dw(&w.ffn1_l2_w_t)?,
            ffn1_l2_b: d(&w.ffn1_l2_b.0)?,
            att_norm_w: d(&w.att_norm_w.0)?,
            att_norm_b: d(&w.att_norm_b.0)?,
            qkv_w_t: dw(&w.qkv_w_t)?,
            qkv_b: d(&w.qkv_b.0)?,
            pos_bias_u: d(&w.pos_bias_u.0)?,
            pos_bias_v: d(&w.pos_bias_v.0)?,
            p_w_t: dw(&w.p_w_t)?,
            out_w_t: dw(&w.out_w_t)?,
            out_b: d(&w.out_b.0)?,
            conv_norm_w: d(&w.conv_norm_w.0)?,
            conv_norm_b: d(&w.conv_norm_b.0)?,
            cpw1_w_t: dw(&w.cpw1_w_t)?,
            cpw1_b: d(&w.cpw1_b.0)?,
            cdw_params: d(&w.cdw_params.0)?,
            cpw2_w_t: dw(&w.cpw2_w_t)?,
            cpw2_b: d(&w.cpw2_b.0)?,
            ffn2_norm_w: d(&w.ffn2_norm_w.0)?,
            ffn2_norm_b: d(&w.ffn2_norm_b.0)?,
            ffn2_l1_w_t: dw(&w.ffn2_l1_w_t)?,
            ffn2_l1_b: d(&w.ffn2_l1_b.0)?,
            ffn2_l2_w_t: dw(&w.ffn2_l2_w_t)?,
            ffn2_l2_b: d(&w.ffn2_l2_b.0)?,
            out_norm_w: d(&w.out_norm_w.0)?,
            out_norm_b: d(&w.out_norm_b.0)?,
        })
    }
}

// ============================================================================
// Test
// ============================================================================

#[test]
#[ignore]
fn encoder_layer_parity_small() -> anyhow::Result<()> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip: model not found at {}", dir.display());
        return Ok(());
    }

    let cuda = CudaState::new(0)?;
    let model = ModelWeights::load(&dir, &cuda, false)?;

    // Use first encoder layer only.
    let cpu_w = CpuLayerWeights::from_gpu(&cuda, &model.encoder.layers[0])?;

    // Small input: 4 tokens, 1280 dims. Random but deterministic.
    let tokens = 4usize;
    let seed = 42u64;
    let x_f32: Vec<f32> = (0..tokens * D_MODEL)
        .map(|i| {
            let x = (seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
                .wrapping_add(i as u64))
            .wrapping_mul(0x2545F4914F6CDD1D)
                >> 37;
            (x as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let pos_f32: Vec<f32> = (0..tokens * D_MODEL)
        .map(|i| {
            let x = (seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
                .wrapping_add((i + tokens * D_MODEL) as u64))
            .wrapping_mul(0x2545F4914F6CDD1D)
                >> 37;
            (x as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    // CPU reference
    let cpu_out = encoder_layer_cpu(&x_f32, &pos_f32, tokens, &cpu_w);

    // GPU forward
    let x_f16: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let pos_f16: Vec<f16> = pos_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let x_gpu = cuda.upload_f16(&x_f16)?;
    let pos_gpu = cuda.upload_f16(&pos_f16)?;

    let layer = EncoderLayer { w: &model.encoder.layers[0] };
    let gpu_out = layer.forward(&cuda, &x_gpu, &pos_gpu, tokens)?;
    cuda.synchronize()?;
    let gpu_out_f16 = cuda.download_f16(&gpu_out)?;
    let gpu_out_f32: Vec<f32> = gpu_out_f16.iter().map(|h| h.to_f32()).collect();

    assert_eq!(gpu_out_f32.len(), cpu_out.len());

    let mut max_diff = 0f32;
    let mut sum_diff = 0f32;
    let mut max_idx = 0usize;
    for i in 0..cpu_out.len() {
        let diff = (gpu_out_f32[i] - cpu_out[i]).abs();
        sum_diff += diff;
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    let avg_diff = sum_diff / cpu_out.len() as f32;

    println!(
        "parity: tokens={tokens} max_diff={max_diff:.6} avg_diff={avg_diff:.6} max_idx={max_idx}",
    );
    println!(
        "  cpu[{max_idx}]={:.6} gpu[{max_idx}]={:.6}",
        cpu_out[max_idx], gpu_out_f32[max_idx]
    );

    let max_val = cpu_out.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1.0);
    let rel_diff = max_diff / max_val;
    println!("  max_val={max_val:.1} rel_diff={rel_diff:.6}");

    // f16 precision across 5 sub-blocks (FFN1→Attn→Conv→FFN2→LN) with nonlinear
    // ops (softmax, SiLU, GLU) and residual connections accumulates ~0.7% error.
    // Individual operations are <0.06% each (verified by isolate_error test).
    assert!(
        rel_diff < 1e-2,
        "encoder layer parity failure: max_diff={max_diff:.6} rel_diff={rel_diff:.6} (threshold 1e-2)"
    );

    Ok(())
}