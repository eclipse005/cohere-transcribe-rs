//! CUDA backend: implements `Backend` by delegating to the hand-tuned
//! `CudaState` engine (cuBLAS + NVRTC kernels). This is a thin adapter — every
//! method maps 1:1 to an existing `CudaState` method so the optimized CUDA path
//! is untouched.

#![cfg(feature = "cuda")]

use anyhow::Result;
use cudarc::driver::safe::CudaSlice;
use half::f16;

use crate::backend::{Backend, Int8Weight};
use crate::engine::CudaState;
use crate::tensor::GpuWeight;

/// The CUDA backend. Owns the `CudaState` (context, stream, cuBLAS handle,
/// kernel registry, cuBLAS workspace).
pub struct CudaBackend {
    pub state: CudaState,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self> {
        Ok(Self { state: CudaState::new(ordinal)? })
    }
}

impl Backend for CudaBackend {
    type Buf = CudaSlice<f16>;
    type Weight = GpuWeight;
    type IBuf = CudaSlice<i8>;

    fn name(&self) -> &str {
        "cuda:0"
    }

    fn int8_enabled(&self) -> bool {
        // sm_61 DP4A is available; whether it's *used* depends on Int8Weight
        // presence in the loaded weights.
        true
    }

    // ---- memory primitives --------------------------------------------------

    fn alloc_uninit(&self, n: usize) -> Result<Self::Buf> {
        self.state.alloc_uninit_f16(n)
    }
    fn upload_f16(&self, data: &[f16]) -> Result<Self::Buf> {
        self.state.upload_f16(data)
    }
    fn download_f16(&self, b: &Self::Buf) -> Result<Vec<f16>> {
        self.state.download_f16(b)
    }
    fn synchronize(&self) -> Result<()> {
        self.state.synchronize()
    }

    // ---- weight construction ------------------------------------------------

    fn upload_weight(&self, data: &[f16], rows: usize, cols: usize) -> Result<Self::Weight> {
        let d = self.state.upload_f16(data)?;
        Ok(GpuWeight::new(d, rows, cols))
    }

    fn upload_int8_weight(
        &self,
        data: &[i8],
        wt_inv: &[f16],
        rows: usize,
        cols: usize,
    ) -> Result<Int8Weight<Self>> {
        Ok(Int8Weight {
            data: self.state.upload_i8(data)?,
            wt_inv: self.state.upload_f16(wt_inv)?,
            rows,
            cols,
        })
    }

    // ---- GEMM ---------------------------------------------------------------

    fn linear(&self, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<Self::Buf> {
        self.state.linear_gpu_raw(x, m, w)
    }

    fn linear_accum(
        &self,
        y: &mut Self::Buf,
        x: &Self::Buf,
        m: usize,
        w: &Self::Weight,
    ) -> Result<()> {
        // The existing linear_gpu_accum takes GpuTensor; build a transient one.
        // NOTE: kept for trait completeness; not on the hot path today.
        use crate::tensor::GpuTensor;
        let n = w.rows;
        let k = w.cols;
        let xt = GpuTensor::new(x.clone(), vec![m, k]);
        let mut yt = GpuTensor::new(y.clone(), vec![m, n]);
        self.state.linear_gpu_accum(&mut yt, &xt, w)?;
        *y = yt.data;
        Ok(())
    }

    fn attention_qk(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        heads: usize,
        m: usize,
        k_seq: usize,
        d: usize,
        k_stride: usize,
        alpha: f32,
    ) -> Result<Self::Buf> {
        self.state.attention_qk_cached(q, k, heads, m, k_seq, d, k_stride, alpha)
    }

    fn attention_av(
        &self,
        a: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        m: usize,
        k_seq: usize,
        d: usize,
        v_stride: usize,
    ) -> Result<Self::Buf> {
        self.state.attention_av_cached(a, v, heads, m, k_seq, d, v_stride)
    }

    fn linear_int8(&self, x: &Self::Buf, w: &Int8Weight<Self>, m: usize) -> Result<Self::Buf> {
        // Wrap the CUDA Int8Weight fields into the engine's view. The engine's
        // linear_int8_f16 takes the engine's own Int8Weight type, but the math
        // is identical — reconstruct the call from the quant steps directly so
        // we don't depend on the engine's Int8Weight struct shape.
        let in_dim = w.cols;
        let out_dim = w.rows;
        let mx = self.state.max_abs_reduce(x, m * in_dim)?;
        let x_i8 = self.state.quantize_f16_i8(x, m * in_dim, &mx)?;
        let mut c_i32 = self.state.alloc_uninit_i32(m * out_dim)?;
        self.state.linear_gpu_int8(&x_i8, &w.data, &mut c_i32, m, in_dim, out_dim)?;
        self.state.dequant_i32_f16(&c_i32, &w.wt_inv, &mx, m * out_dim, out_dim)
    }

    // ---- elementwise / fusion kernels --------------------------------------

    fn layer_norm(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        b: &Self::Buf,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Self::Buf> {
        self.state.layer_norm(x, w, b, rows, dim, eps)
    }

    fn silu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        self.state.silu_bias(x, bias, n, cols)
    }

    fn add_bias_inplace(
        &self,
        x: &mut Self::Buf,
        bias: &Self::Buf,
        n: usize,
        cols: usize,
    ) -> Result<()> {
        self.state.add_bias_inplace(x, bias, n, cols)
    }

    fn bias_residual(
        &self,
        out: &Self::Buf,
        bias: &Self::Buf,
        residual: &Self::Buf,
        n: usize,
        cols: usize,
    ) -> Result<Self::Buf> {
        self.state.bias_residual(out, bias, residual, n, cols)
    }

    fn softmax_last_dim(&self, x: &Self::Buf, rows: usize, dim: usize) -> Result<Self::Buf> {
        self.state.softmax_last_dim(x, rows, dim)
    }

    fn scale_inplace(&self, x: &mut Self::Buf, n: usize, scale: f32) -> Result<()> {
        self.state.scale_inplace(x, n, scale)
    }

    fn add(&self, a: &Self::Buf, b: &Self::Buf, n: usize) -> Result<Self::Buf> {
        self.state.add(a, b, n)
    }

    fn glu_depthwise_conv(
        &self,
        x: &Self::Buf,
        bias: &Self::Buf,
        cdw_params: &Self::Buf,
        tokens: usize,
        channels: usize,
    ) -> Result<Self::Buf> {
        self.state.glu_depthwise_conv(x, bias, cdw_params, tokens, channels)
    }

    fn split_qkv_heads_bias(
        &self,
        qkv: &Self::Buf,
        pos_bias_u: &Self::Buf,
        pos_bias_v: &Self::Buf,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(Self::Buf, Self::Buf, Self::Buf, Self::Buf)> {
        self.state.split_qkv_heads_bias(qkv, pos_bias_u, pos_bias_v, tokens, heads, head_dim)
    }

    fn merge_heads(
        &self,
        inp: &Self::Buf,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self::Buf> {
        self.state.merge_heads(inp, tokens, heads, head_dim)
    }

    fn split_to_heads(
        &self,
        inp: &Self::Buf,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self::Buf> {
        self.state.split_to_heads(inp, tokens, heads, head_dim)
    }

    fn causal_softmax(&self, x: &Self::Buf, heads: usize, tokens: usize, scale: f32)
        -> Result<Self::Buf> {
        self.state.causal_softmax(x, heads, tokens, scale)
    }

    fn relu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf> {
        self.state.relu_bias(x, bias, n, cols)
    }

    fn merge_heads_single(
        &self,
        inp: &Self::Buf,
        stride: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self::Buf> {
        self.state.merge_heads_single(inp, stride, heads, head_dim)
    }

    fn split_qkv_step_cached(
        &self,
        qkv: &Self::Buf,
        bias: &Self::Buf,
        k_cache: &mut Self::Buf,
        v_cache: &mut Self::Buf,
        pos: usize,
        max_seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self::Buf> {
        self.state.split_qkv_step_cached(qkv, bias, k_cache, v_cache, pos, max_seq, heads, head_dim)
    }

    fn split_qkv_batch_scatter(
        &self,
        qkv: &Self::Buf,
        bias: &Self::Buf,
        k_cache: &mut Self::Buf,
        v_cache: &mut Self::Buf,
        seq: usize,
        max_seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self::Buf> {
        self.state.split_qkv_batch_scatter(qkv, bias, k_cache, v_cache, seq, max_seq, heads, head_dim)
    }

    fn fused_attn_scores_softmax(
        &self,
        ac: &Self::Buf,
        bd: &Self::Buf,
        heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
    ) -> Result<Self::Buf> {
        self.state.fused_attn_scores_softmax(ac, bd, heads, q_len, k_len, scale)
    }

    // ---- embedding ----------------------------------------------------------

    fn embed_gather_add(
        &self,
        token_emb: &Self::Buf,
        pos_emb: &Self::Buf,
        tok: usize,
        pos: usize,
        dim: usize,
    ) -> Result<Self::Buf> {
        self.state.embed_gather_add(token_emb, pos_emb, tok, pos, dim)
    }

    fn embed_batch(
        &self,
        token_emb: &Self::Buf,
        pos_emb: &Self::Buf,
        ids: &[i32],
        seq: usize,
        dim: usize,
    ) -> Result<Self::Buf> {
        let ids_gpu = self.state.upload_i32(ids)?;
        self.state.embed_batch(token_emb, pos_emb, &ids_gpu, seq, dim)
    }

    // ---- pre-encoder conv stack --------------------------------------------

    fn conv2d3x3_s2_relu(
        &self,
        inp: &Self::Buf,
        w: &Self::Buf,
        bias: &Self::Buf,
        n: usize,
        cin: usize,
        cout: usize,
        h: usize,
        ww: usize,
    ) -> Result<(Self::Buf, usize, usize)> {
        self.state.conv2d3x3_s2_relu(inp, w, bias, n, cin, cout, h, ww)
    }

    fn depthwise_conv2d3x3_s2(
        &self,
        inp: &Self::Buf,
        w: &Self::Buf,
        bias: &Self::Buf,
        n: usize,
        c: usize,
        h: usize,
        ww: usize,
    ) -> Result<(Self::Buf, usize, usize)> {
        self.state.depthwise_conv2d3x3_s2(inp, w, bias, n, c, h, ww)
    }

    fn pointwise_conv_relu(
        &self,
        inp: &Self::Buf,
        w: &Self::Buf,
        bias: &Self::Buf,
        n: usize,
        cin: usize,
        cout: usize,
        h: usize,
        ww: usize,
    ) -> Result<Self::Buf> {
        self.state.pointwise_conv_relu(inp, w, bias, n, cin, cout, h, ww)
    }

    fn nchw_to_tokens(&self, inp: &Self::Buf, c: usize, t: usize, f: usize) -> Result<Self::Buf> {
        self.state.nchw_to_tokens(inp, c, t, f)
    }

    // ---- token selection ----------------------------------------------------

    fn argmax(&self, x: &Self::Buf, offset: usize, n: usize) -> Result<i32> {
        let arg = self.state.argmax(x, offset, n)?;
        let v = self.state.download_i32(&arg)?;
        Ok(v[0])
    }

    fn buf_len(b: &Self::Buf) -> usize {
        b.len()
    }

    fn weight_data(&self, w: &Self::Weight) -> Self::Buf {
        w.data.clone()
    }
}
