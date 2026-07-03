//! Stage 1 verification: CudaState GEMM correctness against a CPU reference.
//!
//! Builds only under the `cuda` feature. Uploads small f16 tensors, runs the
//! cuBLAS `linear_gpu` (y = x @ W^T) and `attention_qk` (batched Q@K^T), and
//! checks max-abs-diff vs a naive CPU f32 implementation. hgemm on sm_61
//! accumulates in fp32, so we tolerate ~1e-2 (f16 storage truncation).

#![cfg(feature = "cuda")]

use half::f16;
use native_transcribe::engine::CudaState;
use native_transcribe::tensor::{CpuTensor, GpuTensor, GpuWeight};

fn f(x: f32) -> f16 {
    f16::from_f32(x)
}

/// Naive CPU reference: y[m,n] = sum_k x[m,k] * W[n,k]  (i.e. x @ W^T).
fn cpu_linear(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w[ni * k + ki];
            }
            y[mi * n + ni] = acc;
        }
    }
    y
}

/// Naive CPU reference for batched Q@K^T: s[b,h,m,n] = sum_d Q[b,h,m,d]*K[b,h,n,d].
fn cpu_qk(q: &[f32], k: &[f32], b: usize, h: usize, m: usize, n: usize, d: usize) -> Vec<f32> {
    let mut s = vec![0f32; b * h * m * n];
    for bi in 0..b {
        for hi in 0..h {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0f32;
                    for di in 0..d {
                        let qv = q[((bi * h + hi) * m + mi) * d + di];
                        let kv = k[((bi * h + hi) * n + ni) * d + di];
                        acc += qv * kv;
                    }
                    s[((bi * h + hi) * m + mi) * n + ni] = acc;
                }
            }
        }
    }
    s
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn linear_gpu_matches_cpu_reference() -> anyhow::Result<()> {
    // x: [2, 3], W: [4, 3] (N=4 out, K=3 in), y: [2, 4]
    let (m, k, n) = (2usize, 3usize, 4usize);
    let x_f32: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_f32: Vec<f32> = (0..(n * k)).map(|i| (i as f32) * 0.05 - 0.1).collect();
    let expected = cpu_linear(&x_f32, &w_f32, m, k, n);

    let cuda = CudaState::new(0)?;
    let x_gpu = cuda.upload_tensor(&CpuTensor::new(x_f32.iter().copied().map(f).collect(), vec![m, k]))?;
    let w_data: Vec<f16> = w_f32.iter().copied().map(f).collect();
    let w_gpu = GpuWeight::new(cuda.upload_f16(&w_data)?, n, k);

    let y_gpu: GpuTensor = cuda.linear_gpu(&x_gpu, &w_gpu)?;
    cuda.synchronize()?;
    let y_f16 = cuda.download_f16(&y_gpu.data)?;
    let y_f32: Vec<f32> = y_f16.iter().map(|h| h.to_f32()).collect();

    let diff = max_abs_diff(&y_f32, &expected);
    assert!(
        diff < 1e-2,
        "linear_gpu max-abs-diff {diff:.4} exceeds 1e-2 (f16 storage tolerance)"
    );
    Ok(())
}

#[test]
fn attention_qk_matches_cpu_reference() -> anyhow::Result<()> {
    // Q: [1, 2, 3, 4], K: [1, 2, 5, 4]  → scores [1, 2, 3, 5]
    let (b, h, m, d, n) = (1usize, 2usize, 3usize, 4usize, 5usize);
    let q_f32: Vec<f32> = (0..(b * h * m * d)).map(|i| (i as f32) * 0.03 - 0.4).collect();
    let k_f32: Vec<f32> = (0..(b * h * n * d)).map(|i| (i as f32) * 0.02 - 0.3).collect();
    let expected = cpu_qk(&q_f32, &k_f32, b, h, m, n, d);

    let cuda = CudaState::new(0)?;
    let q_gpu = cuda.upload_tensor(&CpuTensor::new(q_f32.iter().copied().map(f).collect(), vec![b, h, m, d]))?;
    let k_gpu = cuda.upload_tensor(&CpuTensor::new(k_f32.iter().copied().map(f).collect(), vec![b, h, n, d]))?;

    let s_gpu = cuda.attention_qk(&q_gpu, &k_gpu)?;
    cuda.synchronize()?;
    let s_f16 = cuda.download_f16(&s_gpu.data)?;
    let s_f32: Vec<f32> = s_f16.iter().map(|h| h.to_f32()).collect();

    let diff = max_abs_diff(&s_f32, &expected);
    assert!(
        diff < 1e-1,
        "attention_qk max-abs-diff {diff:.4} exceeds 1e-1 (f16 storage, larger accum)"
    );
    Ok(())
}
