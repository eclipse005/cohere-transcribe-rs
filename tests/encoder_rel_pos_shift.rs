//! Correctness test for the fused encoder relative-position attention shift
//! (`fused_attn_scores_softmax`), the rank-3 (batch=1) Conformer path.
//!
//! Why this test exists: that op used to compute the rel-shift source index
//! with a reshape-trick formula that didn't match candle's
//! `FusedAttentionScoresShifted` (which uses the simple column index
//! `pos_col = k_len - 1 - q + k`), AND it was fed a truncated `bd`
//! (`[heads, tokens, tokens]` instead of `[heads, tokens, pos_len]`). Both
//! bugs corrupted the encoder's relative-position attention and caused
//! per-token greedy drift / word substitutions. This test pins the corrected
//! behaviour against a direct re-implementation of candle's formula, on both
//! the CPU and CUDA backends.

#![cfg(feature = "cuda")]

use half::f16;
use native_transcribe::backend::{Backend, CpuBackend};
use native_transcribe::engine::CudaState;

/// Candle's FusedAttentionScoresShifted (src/app.rs:3182-3189), re-implemented
/// as a plain reference. ac: [heads, q, k]; bd: [heads, q, pos_len] with
/// pos_len = 2*k-1 and q == k. Returns the pre-softmax shifted+scaled scores
/// [heads, q, k] (NOT softmaxed) so the test isolates the shift/add/scale
/// exactly, independent of the softmax implementation.
fn candle_shifted_scores(
    ac: &[f32],
    bd: &[f32],
    heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q_len, k_len);
    let pos_len = 2 * k_len - 1;
    let mut out = vec![0f32; heads * q_len * k_len];
    for h in 0..heads {
        for q in 0..q_len {
            for k in 0..k_len {
                let pos_col = k_len - 1 - q + k;
                let ac_v = ac[h * q_len * k_len + q * k_len + k];
                let bd_v = bd[h * q_len * pos_len + q * pos_len + pos_col];
                out[h * q_len * k_len + q * k_len + k] = (ac_v + bd_v) * scale;
            }
        }
    }
    out
}

/// Softmax over the last dim (matches the backend op).
fn softmax_last_dim(x: &[f32], heads: usize, q: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; heads * q * k];
    for h in 0..heads {
        for r in 0..q {
            let base = h * q * k + r * k;
            let mx = x[base..base + k].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for j in 0..k {
                let e = (x[base + j] - mx).exp();
                out[base + j] = e;
                s += e;
            }
            let inv = 1.0 / s;
            for j in 0..k {
                out[base + j] *= inv;
            }
        }
    }
    out
}

fn random(seed: u64, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 37) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f16buf(b: &CpuBackend, v: &[f32]) -> anyhow::Result<<CpuBackend as Backend>::Buf> {
    b.upload_f16(&v.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())
}

#[test]
fn fused_shift_matches_candle_formula_cpu() -> anyhow::Result<()> {
    let b = CpuBackend::new();
    let heads = 3usize;
    let q_len = 5usize; // tokens
    let k_len = q_len;
    let pos_len = 2 * k_len - 1; // 9
    let scale = (160f32).powf(-0.5); // encoder head_dim=160

    // Distinct, structured values so a wrong index is obvious.
    let ac_f32: Vec<f32> = (0..heads * q_len * k_len).map(|i| (i as f32) * 0.01).collect();
    let bd_f32: Vec<f32> = (0..heads * q_len * pos_len).map(|i| (i as f32) * 0.1 - 1.0).collect();

    let ac = f16buf(&b, &ac_f32)?;
    let bd = f16buf(&b, &bd_f32)?;
    let out = b.fused_attn_scores_softmax(&ac, &bd, heads, q_len, k_len, scale)?;
    let out_f32: Vec<f32> = b.download_f16(&out)?.iter().map(|h| h.to_f32()).collect();

    // Reference: candle shifted scores → softmax.
    let shifted = candle_shifted_scores(&ac_f32, &bd_f32, heads, q_len, k_len, scale);
    let expect = softmax_last_dim(&shifted, heads, q_len, k_len);

    let max_diff = out_f32
        .iter()
        .zip(&expect)
        .map(|(a, e)| (a - e).abs())
        .fold(0f32, f32::max);
    println!("cpu fused-shift: max_diff={max_diff:.6}");
    assert!(
        max_diff < 1e-2,
        "fused_attn_scores_softmax (cpu) diverges from candle column formula: max_diff={max_diff:.6}",
    );
    Ok(())
}

#[test]
fn fused_shift_matches_candle_formula_cuda() -> anyhow::Result<()> {
    let cuda = CudaState::new(0)?;
    let heads = 4usize;
    let q_len = 7usize; // tokens
    let k_len = q_len;
    let pos_len = 2 * k_len - 1; // 13
    let scale = (160f32).powf(-0.5);

    let ac_f32 = random(101, heads * q_len * k_len);
    let bd_f32 = random(202, heads * q_len * pos_len);

    let ac_h: Vec<f16> = ac_f32.iter().map(|x| f16::from_f32(*x)).collect();
    let bd_h: Vec<f16> = bd_f32.iter().map(|x| f16::from_f32(*x)).collect();
    let ac = cuda.upload_f16(&ac_h)?;
    let bd = cuda.upload_f16(&bd_h)?;
    let out = cuda.fused_attn_scores_softmax(&ac, &bd, heads, q_len, k_len, scale)?;
    cuda.synchronize()?;
    let out_f32: Vec<f32> = cuda.download_f16(&out)?.iter().map(|h| h.to_f32()).collect();

    let shifted = candle_shifted_scores(&ac_f32, &bd_f32, heads, q_len, k_len, scale);
    let expect = softmax_last_dim(&shifted, heads, q_len, k_len);

    let max_diff = out_f32
        .iter()
        .zip(&expect)
        .map(|(a, e)| (a - e).abs())
        .fold(0f32, f32::max);
    println!("cuda fused-shift: max_diff={max_diff:.6}");
    assert!(
        max_diff < 2e-2,
        "fused_attn_scores_softmax (cuda) diverges from candle column formula: max_diff={max_diff:.6}",
    );
    Ok(())
}

/// Edge: q_len=1 (single-token decode-like shape). pos_len = 1, the single
/// column index is (k_len-1-q+k) = 0. Guards against off-by-one at the
/// smallest size.
#[test]
fn fused_shift_q1_matches_candle_formula() -> anyhow::Result<()> {
    let b = CpuBackend::new();
    let (heads, q_len, k_len) = (2usize, 1usize, 1usize);
    let scale = 0.125f32;
    let ac_f32 = vec![0.5f32, -0.5];
    let bd_f32 = vec![0.25f32, -0.25];
    let ac = f16buf(&b, &ac_f32)?;
    let bd = f16buf(&b, &bd_f32)?;
    let out = b.fused_attn_scores_softmax(&ac, &bd, heads, q_len, k_len, scale)?;
    let out_f32: Vec<f32> = b.download_f16(&out)?.iter().map(|h| h.to_f32()).collect();
    let shifted = candle_shifted_scores(&ac_f32, &bd_f32, heads, q_len, k_len, scale);
    let expect = softmax_last_dim(&shifted, heads, q_len, k_len);
    let max_diff = out_f32
        .iter()
        .zip(&expect)
        .map(|(a, e)| (a - e).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-3, "q_len=1 fused-shift diverges: {max_diff}");
    Ok(())
}
