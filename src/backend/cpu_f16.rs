//! CPU backend variant with **f16 activation storage**, used exclusively for the
//! decoder. The decoder runs every projection as an M=1 GEMV (memory-bandwidth-
//! bound), where f16 storage is ~1.6× faster than f32 (half the weight + KV
//! traffic) AND needs no per-op activation conversion. Pairing this with the
//! f32 `CpuBackend` (used for the encoder's compute-bound M>1 GEMMs) is what
//! lets the native CPU path beat candle on both halves.
//!
//! All ops run f16 storage / f32 accumulate via the `gemm` crate, matching the
//! CUDA path's numerics.

use anyhow::Result;
use half::f16;

use crate::backend::{Backend, Int8Weight};

#[derive(Debug, Clone)]
pub struct CpuWeightF16 {
    pub data: Vec<f16>,
    pub rows: usize,
    pub cols: usize,
}

pub struct CpuBackendF16;

impl CpuBackendF16 {
    pub fn new() -> Self { Self }
}
impl Default for CpuBackendF16 {
    fn default() -> Self { Self::new() }
}

impl Backend for CpuBackendF16 {
    type Buf = Vec<f16>;
    type Weight = CpuWeightF16;
    type IBuf = Vec<i8>;

    fn name(&self) -> &str { "cpu(f16)" }
    fn int8_enabled(&self) -> bool { false }

    fn alloc_uninit(&self, n: usize) -> Result<Self::Buf> { Ok(vec![f16::ZERO; n]) }
    fn upload_f16(&self, data: &[f16]) -> Result<Self::Buf> { Ok(data.to_vec()) }
    fn download_f16(&self, b: &Self::Buf) -> Result<Vec<f16>> { Ok(b.clone()) }
    fn synchronize(&self) -> Result<()> { Ok(()) }

    fn upload_weight(&self, data: &[f16], rows: usize, cols: usize) -> Result<Self::Weight> {
        Ok(CpuWeightF16 { data: data.to_vec(), rows, cols })
    }
    fn upload_int8_weight(&self, data: &[i8], wt_inv: &[f16], rows: usize, cols: usize)
        -> Result<Int8Weight<Self>> {
        Ok(Int8Weight { data: data.to_vec(), wt_inv: wt_inv.to_vec(), rows, cols })
    }

    fn linear(&self, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<Self::Buf> {
        let k = w.cols; let n = w.rows;
        let mut y = vec![f16::ZERO; m * n];
        unsafe {
            gemm::gemm(m, n, k, y.as_mut_ptr(), 1, n as isize, true,
                x.as_ptr(), 1, k as isize, w.data.as_ptr(), k as isize, 1,
                f16::ONE, f16::ONE, false, false, false, gemm::Parallelism::None);
        }
        Ok(y)
    }

    fn linear_accum(&self, y: &mut Self::Buf, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<()> {
        let k = w.cols; let n = w.rows;
        let fresh = self.linear(x, m, w)?;
        for i in 0..y.len() { y[i] = f16::from_f32(y[i].to_f32() + fresh[i].to_f32()); }
        let _ = (k, n); Ok(())
    }

    fn attention_qk(&self, q: &Self::Buf, k: &Self::Buf, heads: usize, m: usize,
        k_seq: usize, d: usize, k_stride: usize, alpha: f32) -> Result<Self::Buf> {
        let mut s = vec![f16::ZERO; heads * m * k_seq];
        for h in 0..heads {
            let qh = &q[h*m*d..h*m*d+m*d];
            let kh = &k[h*k_stride*d..h*k_stride*d+k_seq*d];
            let sh = &mut s[h*m*k_seq..h*m*k_seq+m*k_seq];
            unsafe { gemm::gemm(m, k_seq, d, sh.as_mut_ptr(), 1, k_seq as isize, true,
                qh.as_ptr(), 1, d as isize, kh.as_ptr(), d as isize, 1,
                f16::ONE, f16::ONE, false, false, false, gemm::Parallelism::None); }
        }
        if alpha != 1.0 { for v in s.iter_mut() { *v = f16::from_f32(v.to_f32() * alpha); } }
        Ok(s)
    }

    fn attention_av(&self, a: &Self::Buf, v: &Self::Buf, heads: usize, m: usize,
        k_seq: usize, d: usize, v_stride: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; heads * m * d];
        for h in 0..heads {
            let ah = &a[h*m*k_seq..h*m*k_seq+m*k_seq];
            let vh = &v[h*v_stride*d..h*v_stride*d+k_seq*d];
            let oh = &mut out[h*m*d..h*m*d+m*d];
            unsafe { gemm::gemm(m, d, k_seq, oh.as_mut_ptr(), 1, d as isize, true,
                ah.as_ptr(), 1, k_seq as isize, vh.as_ptr(), 1, d as isize,
                f16::ONE, f16::ONE, false, false, false, gemm::Parallelism::None); }
        }
        Ok(out)
    }

    fn linear_int8(&self, _x: &Self::Buf, _w: &Int8Weight<Self>, _m: usize) -> Result<Self::Buf> {
        anyhow::bail!("CPU INT8 path not yet implemented")
    }

    fn layer_norm(&self, x: &Self::Buf, w: &Self::Buf, b: &Self::Buf, rows: usize, dim: usize, eps: f32) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; rows * dim];
        for r in 0..rows {
            let xr: Vec<f32> = x[r*dim..(r+1)*dim].iter().map(|h| h.to_f32()).collect();
            let mean: f32 = xr.iter().sum::<f32>() / dim as f32;
            let var: f32 = xr.iter().map(|v| (v-mean).powi(2)).sum::<f32>() / dim as f32;
            let inv = 1.0/(var+eps).sqrt();
            for c in 0..dim { out[r*dim+c] = f16::from_f32((xr[c]-mean)*inv*w[c].to_f32()+b[c].to_f32()); }
        }
        Ok(out)
    }

    fn silu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; n];
        for i in 0..n { let v = x[i].to_f32()+bias[i%cols].to_f32(); out[i] = f16::from_f32(v*(1.0/(1.0+(-v).exp()))); }
        Ok(out)
    }

    fn add_bias_inplace(&self, x: &mut Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<()> {
        for i in 0..n { x[i] = f16::from_f32(x[i].to_f32()+bias[i%cols].to_f32()); }
        Ok(())
    }

    fn bias_residual(&self, out: &Self::Buf, bias: &Self::Buf, residual: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        let mut y = vec![f16::ZERO; n];
        for i in 0..n { y[i] = f16::from_f32(out[i].to_f32()+bias[i%cols].to_f32()+residual[i].to_f32()); }
        Ok(y)
    }

    fn softmax_last_dim(&self, x: &Self::Buf, rows: usize, dim: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; rows*dim];
        for r in 0..rows {
            let xr: Vec<f32> = x[r*dim..(r+1)*dim].iter().map(|h| h.to_f32()).collect();
            let mx = xr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            let mut e = vec![0f32; dim];
            for c in 0..dim { e[c] = (xr[c]-mx).exp(); s += e[c]; }
            let inv = 1.0/s;
            for c in 0..dim { out[r*dim+c] = f16::from_f32(e[c]*inv); }
        }
        Ok(out)
    }

    fn scale_inplace(&self, x: &mut Self::Buf, n: usize, scale: f32) -> Result<()> {
        for i in 0..n { x[i] = f16::from_f32(x[i].to_f32()*scale); }
        Ok(())
    }

    fn add(&self, a: &Self::Buf, b: &Self::Buf, n: usize) -> Result<Self::Buf> {
        let mut y = vec![f16::ZERO; n];
        for i in 0..n { y[i] = f16::from_f32(a[i].to_f32()+b[i].to_f32()); }
        Ok(y)
    }

    fn glu_depthwise_conv(&self, _x: &Self::Buf, _bias: &Self::Buf, _cdw: &Self::Buf, _toks: usize, _ch: usize) -> Result<Self::Buf> {
        anyhow::bail!("glu_depthwise_conv not used on the f16 decoder backend")
    }

    fn split_qkv_heads_bias(&self, _qkv: &Self::Buf, _pu: &Self::Buf, _pv: &Self::Buf, _toks: usize, _h: usize, _hd: usize) -> Result<(Self::Buf,Self::Buf,Self::Buf,Self::Buf)> {
        anyhow::bail!("split_qkv_heads_bias not used on the f16 decoder backend")
    }

    fn merge_heads(&self, inp: &Self::Buf, tokens: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads*head_dim;
        let mut out = vec![f16::ZERO; tokens*d];
        for h in 0..heads { for t in 0..tokens { for dd in 0..head_dim {
            out[t*d+h*head_dim+dd] = inp[h*tokens*head_dim+t*head_dim+dd];
        }}}
        Ok(out)
    }

    fn split_to_heads(&self, inp: &Self::Buf, tokens: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads*head_dim;
        let mut out = vec![f16::ZERO; heads*tokens*head_dim];
        for h in 0..heads { for t in 0..tokens { for dd in 0..head_dim {
            out[h*tokens*head_dim+t*head_dim+dd] = inp[t*d+h*head_dim+dd];
        }}}
        Ok(out)
    }

    fn causal_softmax(&self, x: &Self::Buf, heads: usize, tokens: usize, scale: f32) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; heads*tokens*tokens];
        for h in 0..heads { for i in 0..tokens {
            let base = h*tokens*tokens+i*tokens;
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i { let v = x[base+j].to_f32()*scale; if v>mx { mx=v; } }
            let mut s = 0f32; let mut e = vec![0f32; tokens];
            for j in 0..=i { e[j] = (x[base+j].to_f32()*scale-mx).exp(); s += e[j]; }
            let inv = 1.0/s;
            for j in 0..=i { out[base+j] = f16::from_f32(e[j]*inv); }
        }}
        Ok(out)
    }

    fn relu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; n];
        for i in 0..n { let v = x[i].to_f32()+bias[i%cols].to_f32(); out[i] = f16::from_f32(if v>0.0 {v} else {0.0}); }
        Ok(out)
    }

    fn merge_heads_single(&self, inp: &Self::Buf, stride: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; heads*head_dim];
        for h in 0..heads { for i in 0..head_dim { out[h*head_dim+i] = inp[h*stride*head_dim+i]; } }
        Ok(out)
    }

    fn split_qkv_step_cached(&self, qkv: &Self::Buf, bias: &Self::Buf, k_cache: &mut Self::Buf, v_cache: &mut Self::Buf, pos: usize, max_seq: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads*head_dim;
        let mut q_out = vec![f16::ZERO; heads*head_dim];
        for h in 0..heads { for dd in 0..head_dim {
            let qkv_off = h*head_dim+dd;
            let cache_off = h*max_seq*head_dim+pos*head_dim+dd;
            q_out[h*head_dim+dd] = f16::from_f32(qkv[qkv_off].to_f32()+bias[qkv_off].to_f32());
            k_cache[cache_off] = f16::from_f32(qkv[d+qkv_off].to_f32()+bias[d+qkv_off].to_f32());
            v_cache[cache_off] = f16::from_f32(qkv[2*d+qkv_off].to_f32()+bias[2*d+qkv_off].to_f32());
        }}
        Ok(q_out)
    }

    fn split_qkv_batch_scatter(&self, qkv: &Self::Buf, bias: &Self::Buf, k_cache: &mut Self::Buf, v_cache: &mut Self::Buf, seq: usize, max_seq: usize, heads: usize, head_dim: usize) -> Result<Self::Buf> {
        let d = heads*head_dim;
        let mut q_out = vec![f16::ZERO; heads*seq*head_dim];
        for h in 0..heads { for t in 0..seq { for dd in 0..head_dim {
            let row = t*3*d;
            let q_off = row+h*head_dim+dd;
            let cache_off = h*max_seq*head_dim+t*head_dim+dd;
            let out_idx = h*seq*head_dim+t*head_dim+dd;
            q_out[out_idx] = f16::from_f32(qkv[q_off].to_f32()+bias[h*head_dim+dd].to_f32());
            k_cache[cache_off] = f16::from_f32(qkv[q_off+d].to_f32()+bias[d+h*head_dim+dd].to_f32());
            v_cache[cache_off] = f16::from_f32(qkv[q_off+2*d].to_f32()+bias[2*d+h*head_dim+dd].to_f32());
        }}}
        Ok(q_out)
    }

    fn fused_attn_scores_softmax(&self, _ac: &Self::Buf, _bd: &Self::Buf, _h: usize, _ql: usize, _kl: usize, _s: f32) -> Result<Self::Buf> {
        anyhow::bail!("fused_attn_scores_softmax (encoder rel-pos) not used on the f16 decoder backend")
    }

    fn embed_gather_add(&self, token_emb: &Self::Buf, pos_emb: &Self::Buf, tok: usize, pos: usize, dim: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; dim];
        for d in 0..dim { out[d] = f16::from_f32(token_emb[tok*dim+d].to_f32()+pos_emb[pos*dim+d].to_f32()); }
        Ok(out)
    }

    fn embed_batch(&self, token_emb: &Self::Buf, pos_emb: &Self::Buf, ids: &[i32], seq: usize, dim: usize) -> Result<Self::Buf> {
        let mut out = vec![f16::ZERO; seq*dim];
        for (s, &id) in ids.iter().enumerate() {
            let tok = id as usize;
            for d in 0..dim { out[s*dim+d] = f16::from_f32(token_emb[tok*dim+d].to_f32()+pos_emb[s*dim+d].to_f32()); }
        }
        Ok(out)
    }

    fn conv2d3x3_s2_relu(&self, _inp: &Self::Buf, _w: &Self::Buf, _bias: &Self::Buf, _n: usize, _cin: usize, _cout: usize, _h: usize, _ww: usize) -> Result<(Self::Buf,usize,usize)> {
        anyhow::bail!("conv not used on the f16 decoder backend")
    }
    fn depthwise_conv2d3x3_s2(&self, _inp: &Self::Buf, _w: &Self::Buf, _bias: &Self::Buf, _n: usize, _c: usize, _h: usize, _ww: usize) -> Result<(Self::Buf,usize,usize)> {
        anyhow::bail!("conv not used on the f16 decoder backend")
    }
    fn pointwise_conv_relu(&self, _inp: &Self::Buf, _w: &Self::Buf, _bias: &Self::Buf, _n: usize, _cin: usize, _cout: usize, _h: usize, _ww: usize) -> Result<Self::Buf> {
        anyhow::bail!("conv not used on the f16 decoder backend")
    }
    fn nchw_to_tokens(&self, _inp: &Self::Buf, _c: usize, _t: usize, _f: usize) -> Result<Self::Buf> {
        anyhow::bail!("nchw_to_tokens not used on the f16 decoder backend")
    }

    fn argmax(&self, x: &Self::Buf, offset: usize, n: usize) -> Result<i32> {
        let mut mx = f32::NEG_INFINITY; let mut idx = 0i32;
        for i in 0..n { let v = x[offset+i].to_f32(); if v>mx { mx=v; idx=i as i32; } }
        Ok(idx)
    }

    fn buf_len(b: &Self::Buf) -> usize { b.len() }
    fn weight_data(&self, w: &Self::Weight) -> Self::Buf { w.data.clone() }
}
