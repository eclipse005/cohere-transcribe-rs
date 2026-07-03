//! CPU backend: mixed-precision implementation of `Backend`.
//!
//! **Activation storage is f32** (`Vec<f32>`) — elementwise/fusion ops run
//! native f32 with no per-op conversion, and encoder M>1 GEMMs hit the fast f32
//! path directly. **Weights are cached in BOTH f32 and f16**, and GEMM dispatch
//! picks the format per op:
//! - M>1 (encoder/FFN/conv, compute-bound): f32 GEMM on the cached f32 weight
//!   (~3× faster than f16 — the `gemm` f16 microkernel unpacks f16→f32 each call).
//! - M=1 (decoder GEMV, bandwidth-bound): f16 GEMV on the cached f16 weight,
//!   converting the small `x` to f16 per call (~1.6× faster — half the weight bytes).
//!
//! The trait boundary still speaks f16 (`upload_f16`/`download_f16`/weights) to
//! stay interface-compatible with CUDA; conversion happens only at that boundary.
//! More precise than the CUDA f16-storage path, which fixes word substitutions.

use anyhow::Result;
use half::f16;

use crate::backend::{Backend, Int8Weight};

/// CPU weight `[rows, cols]` row-major, cached in f32 (encoder GEMMs) and f16
/// (decoder GEMVs). Both built once at upload.
#[derive(Debug, Clone)]
pub struct CpuWeight {
    pub data: Vec<f32>,
    pub data_f16: Vec<f16>,
    pub rows: usize,
    pub cols: usize,
}

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { Self }
}
impl Default for CpuBackend {
    fn default() -> Self { Self::new() }
}

/// `y[m,n] = x[m,k] @ W[n,k]^T` (β=0) f32. For M>1 (encoder/FFN/conv).
fn gemm_linear_f32(x: &[f32], m: usize, n: usize, k: usize, w: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    unsafe {
        gemm::gemm(m, n, k, y.as_mut_ptr(), 1, n as isize, true,
            x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
            1.0f32, 1.0f32, false, false, false, gemm::Parallelism::Rayon(0));
    }
    y
}

/// M=1 GEMV on the cached f16 weight. Converts small `x` (k elts) to f16; `w_f16`
/// is pre-cached. Returns f32. For the decoder (bandwidth-bound).
fn gemv_f16(x: &[f32], n: usize, k: usize, w_f16: &[f16]) -> Vec<f32> {
    let x16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
    let mut y16 = vec![f16::ZERO; n];
    unsafe {
        gemm::gemm(1, n, k, y16.as_mut_ptr(), 1, n as isize, true,
            x16.as_ptr(), 1, k as isize, w_f16.as_ptr(), k as isize, 1,
            f16::ONE, f16::ONE, false, false, false, gemm::Parallelism::None);
    }
    y16.iter().map(|h| h.to_f32()).collect()
}

impl Backend for CpuBackend {
    type Buf = Vec<f32>;
    type Weight = CpuWeight;
    type IBuf = Vec<i8>;

    fn name(&self) -> &str { "cpu" }
    fn int8_enabled(&self) -> bool { false }

    fn alloc_uninit(&self, n: usize) -> Result<Self::Buf> { Ok(vec![0f32; n]) }
    fn upload_f16(&self, data: &[f16]) -> Result<Self::Buf> {
        Ok(data.iter().map(|h| h.to_f32()).collect())
    }
    fn download_f16(&self, b: &Self::Buf) -> Result<Vec<f16>> {
        Ok(b.iter().map(|x| f16::from_f32(*x)).collect())
    }
    fn synchronize(&self) -> Result<()> { Ok(()) }

    fn upload_weight(&self, data: &[f16], rows: usize, cols: usize) -> Result<Self::Weight> {
        Ok(CpuWeight {
            data: data.iter().map(|h| h.to_f32()).collect(),
            data_f16: data.to_vec(),
            rows, cols,
        })
    }
    fn upload_int8_weight(&self, data: &[i8], wt_inv: &[f16], rows: usize, cols: usize)
        -> Result<Int8Weight<Self>> {
        Ok(Int8Weight { data: data.to_vec(), wt_inv: wt_inv.iter().map(|h| h.to_f32()).collect(), rows, cols })
    }

    fn linear(&self, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<Self::Buf> {
        let k = w.cols; let n = w.rows;
        if m == 1 { Ok(gemv_f16(x, n, k, &w.data_f16)) }
        else { Ok(gemm_linear_f32(x, m, n, k, &w.data)) }
    }

    fn linear_accum(&self, y: &mut Self::Buf, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<()> {
        let fresh = if m == 1 { gemv_f16(x, w.rows, w.cols, &w.data_f16) }
                    else { gemm_linear_f32(x, m, w.rows, w.cols, &w.data) };
        for i in 0..y.len() { y[i] += fresh[i]; }
        Ok(())
    }

    fn attention_qk(&self, q: &Self::Buf, k: &Self::Buf, heads: usize, m: usize,
        k_seq: usize, d: usize, k_stride: usize, alpha: f32) -> Result<Self::Buf> {
        if m > 1 {
            let mut s = vec![0f32; heads * m * k_seq];
            for h in 0..heads {
                let qh = &q[h*m*d..h*m*d+m*d];
                let kh = &k[h*k_stride*d..h*k_stride*d+k_seq*d];
                let sh = &mut s[h*m*k_seq..h*m*k_seq+m*k_seq];
                unsafe { gemm::gemm(m, k_seq, d, sh.as_mut_ptr(), 1, k_seq as isize, true,
                    qh.as_ptr(), 1, d as isize, kh.as_ptr(), d as isize, 1,
                    1.0f32, 1.0f32, false, false, false, gemm::Parallelism::None); }
            }
            if alpha != 1.0 { for v in s.iter_mut() { *v *= alpha; } }
            Ok(s)
        } else {
            let q16: Vec<f16> = q.iter().map(|x| f16::from_f32(*x)).collect();
            let k16: Vec<f16> = k.iter().map(|x| f16::from_f32(*x)).collect();
            let mut s = vec![f16::ZERO; heads * m * k_seq];
            for h in 0..heads {
                let qh = &q16[h*m*d..h*m*d+m*d];
                let kh = &k16[h*k_stride*d..h*k_stride*d+k_seq*d];
                let sh = &mut s[h*m*k_seq..h*m*k_seq+m*k_seq];
                unsafe { gemm::gemm(m, k_seq, d, sh.as_mut_ptr(), 1, k_seq as isize, true,
                    qh.as_ptr(), 1, d as isize, kh.as_ptr(), d as isize, 1,
                    f16::ONE, f16::ONE, false, false, false, gemm::Parallelism::None); }
            }
            if alpha != 1.0 { for v in s.iter_mut() { *v = f16::from_f32(v.to_f32() * alpha); } }
            Ok(s.iter().map(|h| h.to_f32()).collect())
        }
    }

    fn attention_av(&self, a: &Self::Buf, v: &Self::Buf, heads: usize, m: usize,
        k_seq: usize, d: usize, v_stride: usize) -> Result<Self::Buf> {
        if m > 1 {
            let mut out = vec![0f32; heads * m * d];
            for h in 0..heads {
                let ah = &a[h*m*k_seq..h*m*k_seq+m*k_seq];
                let vh = &v[h*v_stride*d..h*v_stride*d+k_seq*d];
                let oh = &mut out[h*m*d..h*m*d+m*d];
                unsafe { gemm::gemm(m, d, k_seq, oh.as_mut_ptr(), 1, d as isize, true,
                    ah.as_ptr(), 1, k_seq as isize, vh.as_ptr(), 1, d as isize,
                    1.0f32, 1.0f32, false, false, false, gemm::Parallelism::None); }
            }
            Ok(out)
        } else {
            let a16: Vec<f16> = a.iter().map(|x| f16::from_f32(*x)).collect();
            let v16: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
            let mut out = vec![f16::ZERO; heads * m * d];
            for h in 0..heads {
                let ah = &a16[h*m*k_seq..h*m*k_seq+m*k_seq];
                let vh = &v16[h*v_stride*d..h*v_stride*d+k_seq*d];
                let oh = &mut out[h*m*d..h*m*d+m*d];
                unsafe { gemm::gemm(m, d, k_seq, oh.as_mut_ptr(), 1, d as isize, true,
                    ah.as_ptr(), 1, k_seq as isize, vh.as_ptr(), 1, d as isize,
                    f16::ONE, f16::ONE, false, false, false, gemm::Parallelism::None); }
            }
            Ok(out.iter().map(|h| h.to_f32()).collect())
        }
    }

    fn linear_int8(&self, _x: &Self::Buf, _w: &Int8Weight<Self>, _m: usize) -> Result<Self::Buf> {
        anyhow::bail!("CPU INT8 path not yet implemented")
    }

    // ---- elementwise / fusion kernels (native f32) -------------------------

    fn layer_norm(&self, x: &Self::Buf, w: &Self::Buf, b: &Self::Buf, rows: usize, dim: usize, eps: f32) -> Result<Self::Buf> {
        let mut out = vec![0f32; rows * dim];
        for r in 0..rows {
            let xr = &x[r*dim..(r+1)*dim];
            let mean: f32 = xr.iter().sum::<f32>() / dim as f32;
            let var: f32 = xr.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..dim {
                out[r*dim+c] = (xr[c] - mean) * inv_std * w[c] + b[c];
            }
        }
        Ok(out)
    }

    fn silu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; n];
        for i in 0..n {
            let v = x[i] + bias[i % cols];
            out[i] = v * (1.0 / (1.0 + (-v).exp()));
        }
        Ok(out)
    }

    fn add_bias_inplace(&self, x: &mut Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<()> {
        for i in 0..n { x[i] += bias[i % cols]; }
        Ok(())
    }

    fn bias_residual(&self, out: &Self::Buf, bias: &Self::Buf, residual: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        let mut y = vec![0f32; n];
        for i in 0..n { y[i] = out[i] + bias[i % cols] + residual[i]; }
        Ok(y)
    }

    fn softmax_last_dim(&self, x: &Self::Buf, rows: usize, dim: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; rows * dim];
        for r in 0..rows {
            let xr = &x[r*dim..(r+1)*dim];
            let mx = xr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for c in 0..dim { let e = (xr[c]-mx).exp(); out[r*dim+c] = e; s += e; }
            let inv = 1.0 / s;
            for c in 0..dim { out[r*dim+c] *= inv; }
        }
        Ok(out)
    }

    fn scale_inplace(&self, x: &mut Self::Buf, n: usize, scale: f32) -> Result<()> {
        for i in 0..n { x[i] *= scale; }
        Ok(())
    }

    fn add(&self, a: &Self::Buf, b: &Self::Buf, n: usize) -> Result<Self::Buf> {
        let mut y = vec![0f32; n];
        for i in 0..n { y[i] = a[i] + b[i]; }
        Ok(y)
    }

    fn glu_depthwise_conv(&self, x: &Self::Buf, bias: &Self::Buf, cdw_params: &Self::Buf, tokens: usize, channels: usize) -> Result<Self::Buf> {
        let c2 = channels * 2;
        let mut gated = vec![0f32; tokens * channels];
        for t in 0..tokens {
            for c in 0..channels {
                let left = x[t*c2+c] + bias[c];
                let right = x[t*c2+channels+c] + bias[channels+c];
                gated[t*channels+c] = left * (1.0/(1.0+(-right).exp()));
            }
        }
        let mut y = vec![0f32; tokens * channels];
        for t in 0..tokens {
            for c in 0..channels {
                let mut acc = cdw_params[c*10+9];
                for kk in 0i32..9 {
                    let src_t = t as isize + kk as isize - 4;
                    if src_t >= 0 && (src_t as usize) < tokens {
                        acc += gated[(src_t as usize)*channels+c] * cdw_params[c*10+kk as usize];
                    }
                }
                y[t*channels+c] = acc * (1.0/(1.0+(-acc).exp()));
            }
        }
        Ok(y)
    }

    fn split_qkv_heads_bias(&self, qkv: &Self::Buf, pos_bias_u: &Self::Buf, pos_bias_v: &Self::Buf, tokens: usize, heads: usize, head_dim: usize) -> Result<(Self::Buf, Self::Buf, Self::Buf, Self::Buf)> {
        let d = heads * head_dim;
        let n = heads * tokens * head_dim;
        let mut qu = vec![0f32; n];
        let mut qv = vec![0f32; n];
        let mut k = vec![0f32; n];
        let mut v = vec![0f32; n];
        for h in 0..heads {
            for t in 0..tokens {
                for dd in 0..head_dim {
                    let out_idx = h*tokens*head_dim + t*head_dim + dd;
                    let q_base = t*3*d + h*head_dim + dd;
                    qu[out_idx] = qkv[q_base] + pos_bias_u[h*head_dim+dd];
                    qv[out_idx] = qkv[q_base] + pos_bias_v[h*head_dim+dd];
                    k[out_idx] = qkv[q_base+d];
                    v[out_idx] = qkv[q_base+2*d];
                }
            }
        }
        Ok((qu, qv, k, v))
    }

    fn merge_heads(&self, inp: &Self::Buf, tokens: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads * head_dim;
        let mut out = vec![0f32; tokens * d];
        for h in 0..heads { for t in 0..tokens { for dd in 0..head_dim {
            out[t*d+h*head_dim+dd] = inp[h*tokens*head_dim+t*head_dim+dd];
        }}}
        Ok(out)
    }

    fn split_to_heads(&self, inp: &Self::Buf, tokens: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads * head_dim;
        let mut out = vec![0f32; heads*tokens*head_dim];
        for h in 0..heads { for t in 0..tokens { for dd in 0..head_dim {
            out[h*tokens*head_dim+t*head_dim+dd] = inp[t*d+h*head_dim+dd];
        }}}
        Ok(out)
    }

    fn causal_softmax(&self, x: &Self::Buf, heads: usize, tokens: usize, scale: f32) -> Result<Self::Buf> {
        let mut out = vec![0f32; heads*tokens*tokens];
        for h in 0..heads {
            for i in 0..tokens {
                let base = h*tokens*tokens + i*tokens;
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i { let v = x[base+j]*scale; if v > mx { mx = v; } }
                let mut s = 0f32;
                for j in 0..=i { let e = (x[base+j]*scale-mx).exp(); out[base+j] = e; s += e; }
                let inv = 1.0/s;
                for j in 0..=i { out[base+j] *= inv; }
            }
        }
        Ok(out)
    }

    fn relu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; n];
        for i in 0..n { let v = x[i]+bias[i%cols]; out[i] = if v > 0.0 { v } else { 0.0 }; }
        Ok(out)
    }

    fn merge_heads_single(&self, inp: &Self::Buf, stride: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; heads*head_dim];
        for h in 0..heads { for i in 0..head_dim { out[h*head_dim+i] = inp[h*stride*head_dim+i]; } }
        Ok(out)
    }

    fn split_qkv_step_cached(&self, qkv: &Self::Buf, bias: &Self::Buf, k_cache: &mut Self::Buf, v_cache: &mut Self::Buf, pos: usize, max_seq: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads * head_dim;
        let mut q_out = vec![0f32; heads*head_dim];
        for h in 0..heads {
            for dd in 0..head_dim {
                let qkv_off = h*head_dim+dd;
                let cache_off = h*max_seq*head_dim + pos*head_dim + dd;
                q_out[h*head_dim+dd] = qkv[qkv_off] + bias[qkv_off];
                k_cache[cache_off] = qkv[d+qkv_off] + bias[d+qkv_off];
                v_cache[cache_off] = qkv[2*d+qkv_off] + bias[2*d+qkv_off];
            }
        }
        Ok(q_out)
    }

    fn split_qkv_batch_scatter(&self, qkv: &Self::Buf, bias: &Self::Buf, k_cache: &mut Self::Buf, v_cache: &mut Self::Buf, seq: usize, max_seq: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads * head_dim;
        let mut q_out = vec![0f32; heads*seq*head_dim];
        for h in 0..heads {
            for t in 0..seq {
                for dd in 0..head_dim {
                    let row = t*3*d;
                    let q_off = row + h*head_dim + dd;
                    let cache_off = h*max_seq*head_dim + t*head_dim + dd;
                    let out_idx = h*seq*head_dim + t*head_dim + dd;
                    q_out[out_idx] = qkv[q_off] + bias[h*head_dim+dd];
                    k_cache[cache_off] = qkv[q_off+d] + bias[d+h*head_dim+dd];
                    v_cache[cache_off] = qkv[q_off+2*d] + bias[2*d+h*head_dim+dd];
                }
            }
        }
        Ok(q_out)
    }

    fn fused_attn_scores_softmax(&self, ac: &Self::Buf, bd: &Self::Buf, heads: usize, q_len: usize, k_len: usize, scale: f32) -> Result<Self::Buf> {
        // ac: [heads, q_len, k_len]; bd: [heads, q_len, pos_len], pos_len = 2*k_len-1,
        // and q_len == k_len (Conformer self-attention). Output: [heads, q_len, k_len].
        // Matches candle's FusedAttentionScoresShifted (src/app.rs:3182):
        //   out[h,q,k] = (ac[h,q,k] + bd[h, q, (k_len-1-q+k)]) * scale   then softmax.
        // The relative-position column (k_len-1-q+k) is in [0, pos_len-1] for
        // every q in [0,q_len) and k in [0,k_len), so every read is in bounds.
        let pos_len = 2 * k_len - 1;
        debug_assert_eq!(q_len, k_len, "fused_attn_scores_softmax expects q_len==k_len");
        let mut out = vec![0f32; heads*q_len*k_len];
        for h in 0..heads {
            for q in 0..q_len {
                let ac_row = h*q_len*k_len + q*k_len;
                let bd_row = h*q_len*pos_len + q*pos_len;
                // shifted scores + scale
                let mut row = vec![0f32; k_len];
                for k in 0..k_len {
                    let pos_col = k_len - 1 - q + k;
                    row[k] = (ac[ac_row + k] + bd[bd_row + pos_col]) * scale;
                }
                // softmax over last dim
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0f32;
                for k in 0..k_len { let e = (row[k]-mx).exp(); out[ac_row + k] = e; s += e; }
                let inv = 1.0/s;
                for k in 0..k_len { out[ac_row + k] *= inv; }
            }
        }
        Ok(out)
    }

    // ---- embedding ----------------------------------------------------------

    fn embed_gather_add(&self, token_emb: &Self::Buf, pos_emb: &Self::Buf, tok: usize, pos: usize, dim: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; dim];
        for d in 0..dim { out[d] = token_emb[tok*dim+d] + pos_emb[pos*dim+d]; }
        Ok(out)
    }

    fn embed_batch(&self, token_emb: &Self::Buf, pos_emb: &Self::Buf, ids: &[i32], seq: usize, dim: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; seq*dim];
        for (s, &id) in ids.iter().enumerate() {
            let tok = id as usize;
            for d in 0..dim { out[s*dim+d] = token_emb[tok*dim+d] + pos_emb[s*dim+d]; }
        }
        Ok(out)
    }

    // ---- pre-encoder conv stack --------------------------------------------

    fn conv2d3x3_s2_relu(&self, inp: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, n: usize, cin: usize, cout: usize, h: usize, ww: usize) -> Result<(Self::Buf, usize, usize)> {
        let hout = (h+1)/2; let wout = (ww+1)/2;
        let mut out = vec![0f32; n*cout*hout*wout];
        for nn in 0..n { for oc in 0..cout { for oh in 0..hout { for ow in 0..wout {
            let mut acc = bias[oc];
            let ih_base = oh as isize*2-1; let iw_base = ow as isize*2-1;
            for ic in 0..cin { for kh in 0usize..3 {
                let ih = ih_base+kh as isize; if ih<0||ih>=h as isize { continue; }
                for kw in 0usize..3 {
                    let iw = iw_base+kw as isize; if iw<0||iw>=ww as isize { continue; }
                    let inv = inp[((nn*cin+ic) as usize)*h*ww + (ih as usize)*ww + (iw as usize)];
                    acc += inv * w[oc*cin*9 + ic*9 + kh*3 + kw];
                }
            }}
            let idx = ((nn*cout+oc) as usize)*hout*wout + oh*wout + ow;
            out[idx] = if acc > 0.0 { acc } else { 0.0 };
        }}}}
        Ok((out, hout, wout))
    }

    fn depthwise_conv2d3x3_s2(&self, inp: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, n: usize, c: usize, h: usize, ww: usize) -> Result<(Self::Buf, usize, usize)> {
        let hout = (h+1)/2; let wout = (ww+1)/2;
        let mut out = vec![0f32; n*c*hout*wout];
        for nn in 0..n { for cc in 0..c { for oh in 0..hout { for ow in 0..wout {
            let mut acc = bias[cc];
            let ih_base = oh as isize*2-1; let iw_base = ow as isize*2-1;
            let in_base = ((nn*c+cc) as usize)*h*ww;
            for kh in 0usize..3 {
                let ih = ih_base+kh as isize; if ih<0||ih>=h as isize { continue; }
                for kw in 0usize..3 {
                    let iw = iw_base+kw as isize; if iw<0||iw>=ww as isize { continue; }
                    acc += inp[in_base+(ih as usize)*ww+(iw as usize)] * w[cc*9+kh*3+kw];
                }
            }
            let idx = ((nn*c+cc) as usize)*hout*wout + oh*wout + ow;
            out[idx] = acc;
        }}}}
        Ok((out, hout, wout))
    }

    fn pointwise_conv_relu(&self, inp: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, n: usize, cin: usize, cout: usize, h: usize, ww: usize) -> Result<Self::Buf> {
        // Pointwise 1×1 conv = a GEMM over channels at each spatial location.
        // inp is NCHW [n, cin, h, w]; reshape to NHWC [n, h*w, cin] for the GEMM,
        // run out = inp_nhwc @ w^T + bias → [n, h*w, cout], ReLU, back to NCHW.
        let spatial = h * ww;
        let mut out = vec![0f32; n * cout * spatial];
        for nn in 0..n {
            // Gather NCHW [cin, h*w] → NHWC [h*w, cin] (transpose).
            let in_base = nn * cin * spatial;
            let mut in_nhwc = vec![0f32; spatial * cin];
            for hp in 0..h {
                for wp in 0..ww {
                    let s = hp * ww + wp;
                    for ic in 0..cin {
                        in_nhwc[s * cin + ic] = inp[in_base + ic * spatial + s];
                    }
                }
            }
            // GEMM: out_nhwc[spatial, cout] = in_nhwc[spatial, cin] @ w[cout, cin]^T.
            // Add bias by initializing dst with bias broadcast, then β=1.
            let mut out_nhwc = vec![0f32; spatial * cout];
            unsafe {
                gemm::gemm(spatial, cout, cin,
                    out_nhwc.as_mut_ptr(), 1, cout as isize, true,
                    in_nhwc.as_ptr(), 1, cin as isize,
                    w.as_ptr(), cin as isize, 1,
                    1.0f32, 1.0f32, false, false, false, gemm::Parallelism::Rayon(0));
            }
            // ReLU + bias add + NHWC → NCHW.
            let out_base = nn * cout * spatial;
            for hp in 0..h {
                for wp in 0..ww {
                    let s = hp * ww + wp;
                    for oc in 0..cout {
                        let v = out_nhwc[s * cout + oc] + bias[oc];
                        out[out_base + oc * spatial + s] = if v > 0.0 { v } else { 0.0 };
                    }
                }
            }
        }
        Ok(out)
    }

    fn nchw_to_tokens(&self, inp: &Self::Buf, c: usize, t: usize, f: usize) -> Result<Self::Buf> {
        let mut out = vec![0f32; t*c*f];
        for tt in 0..t { for cc in 0..c { for ff in 0..f {
            out[tt*c*f+cc*f+ff] = inp[cc*t*f+tt*f+ff];
        }}}
        Ok(out)
    }

    // ---- token selection ----------------------------------------------------

    fn argmax(&self, x: &Self::Buf, offset: usize, n: usize) -> Result<i32> {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0i32;
        for i in 0..n {
            if x[offset+i] > max_val { max_val = x[offset+i]; max_idx = i as i32; }
        }
        Ok(max_idx)
    }

    fn buf_len(b: &Self::Buf) -> usize { b.len() }
    fn weight_data(&self, w: &Self::Weight) -> Self::Buf { w.data.clone() }
}
