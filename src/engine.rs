//! `CudaState` — the long-lived CUDA context, stream, persistent cuBLAS handle,
//! and NVRTC kernel registry. Plus the cuBLAS wrapper GEMMs that are the
//! backbone of the engine.
//!
//! Design (mirrors qwen3-asr `CudarcEngine`, tuned for Cohere shapes):
//!   - ONE cuBLAS handle, held for the process lifetime, with
//!     `CUBLAS_TENSOR_OP_MATH` set (no-op on Pascal, engages TCs on Ampere+).
//!   - `alloc_uninit_*` for any output that is fully overwritten (β=0 GEMM,
//!     fused kernels writing all elements) — skips the `cudaMemset` the
//!     Pascal driver throttles.
//!   - `linear_gpu`        : y = x @ W^T               (β=0)
//!   - `linear_gpu_accum`  : y = y + x @ W^T           (β=1 — folds a residual
//!                              add into the GEMM, removing one launch)
//!   - `attention_qk`      : batched Q @ K^T  (strided batched GEMM)
//!   - `attention_av`      : batched A @ V     (strided batched GEMM)

use std::sync::Arc;

use anyhow::Context;
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys;
use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;

use crate::kernels::CudaKernels;
use crate::tensor::{CpuTensor, GpuTensor, GpuWeight};

pub struct CudaState {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub k: CudaKernels,
    /// cuBLAS device workspace, kept alive for the handle's lifetime.
    _blas_ws: Option<CudaSlice<u8>>,
}

// SAFETY: cudarc's safe handles are not auto-Sync, but the CUDA context is
// single-threaded by construction (all launches go through `stream`, which is
// serialized). We only mutate GPU memory via the stream. Marking Send+Sync
// lets us store the engine behind an `Arc` / `Mutex`.
unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl CudaState {
    pub fn new(ordinal: usize) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(ordinal).context("creating CUDA context")?;
        Self::new_with_ctx(&ctx)
    }

    pub fn new_with_ctx(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .context("creating cuBLAS handle")?;

        // Enable Tensor-Op math mode — no-op on Pascal, ensures TC usage on
        // Ampere+ without code changes.
        unsafe {
            sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
        }

        // Give cuBLAS a dedicated device workspace (16 MB) so it can pick
        // better kernels — especially helpful for the many small (M=1) GEMMs
        // that dominate the decoder's autoregressive loop. Held for the handle
        // lifetime (cuBLAS references the pointer until reset).
        let blas_ws = stream.alloc_zeros::<u8>(16 * 1024 * 1024).ok();
        if let Some(ref ws) = blas_ws {
            use cudarc::driver::safe::DevicePtr;
            unsafe {
                let (ptr, _guard) = ws.device_ptr(&stream);
                let _ = sys::cublasSetWorkspace_v2(
                    *blas.handle(),
                    ptr as *mut std::ffi::c_void,
                    16 * 1024 * 1024,
                );
            }
        }

        let k = CudaKernels::load_all(ctx).context("loading NVRTC kernels")?;

        Ok(Self {
            ctx: ctx.clone(),
            stream,
            blas,
            k,
            _blas_ws: blas_ws,
        })
    }

    // ---- memory primitives -------------------------------------------------

    pub fn upload_f16(&self, data: &[f16]) -> anyhow::Result<CudaSlice<f16>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn upload_i32(&self, data: &[i32]) -> anyhow::Result<CudaSlice<i32>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn upload_i8(&self, data: &[i8]) -> anyhow::Result<CudaSlice<i8>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn upload_i64(&self, data: &[i64]) -> anyhow::Result<CudaSlice<i64>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn alloc_zeros_f16(&self, n: usize) -> anyhow::Result<CudaSlice<f16>> {
        Ok(self.stream.alloc_zeros::<f16>(n)?)
    }

    pub fn alloc_zeros_i32(&self, n: usize) -> anyhow::Result<CudaSlice<i32>> {
        Ok(self.stream.alloc_zeros::<i32>(n)?)
    }

    /// Allocate uninitialized f16 — caller MUST ensure every byte is written
    /// before read. Saves one `cudaMemset_async` vs `alloc_zeros_f16` for
    /// cuBLAS/kernel outputs that are fully overwritten (β=0 GEMM, fused
    /// kernels writing all of `out`, etc.).
    pub fn alloc_uninit_f16(&self, n: usize) -> anyhow::Result<CudaSlice<f16>> {
        // SAFETY: callers only read regions they have fully written.
        Ok(unsafe { self.stream.alloc::<f16>(n)? })
    }

    /// Allocate uninitialized i32 — same semantics as `alloc_uninit_f16`.
    pub fn alloc_uninit_i32(&self, n: usize) -> anyhow::Result<CudaSlice<i32>> {
        Ok(unsafe { self.stream.alloc::<i32>(n)? })
    }

    /// Allocate uninitialized i8 (INT8 weight/activation buffers).
    pub fn alloc_uninit_i8(&self, n: usize) -> anyhow::Result<CudaSlice<i8>> {
        Ok(unsafe { self.stream.alloc::<i8>(n)? })
    }

    pub fn download_f16(&self, slice: &CudaSlice<f16>) -> anyhow::Result<Vec<f16>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }

    pub fn download_i32(&self, slice: &CudaSlice<i32>) -> anyhow::Result<Vec<i32>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }

    /// Allocate a pinned 1-element i32 scratch buffer for async D2H of a single
    /// argmax result. Pinned memory enables true async host transfer and avoids
    /// a per-call host heap allocation. Reading via `as_ptr()` synchronizes.
    pub fn pinned_i32(&self, n: usize) -> anyhow::Result<PinnedHostSlice<i32>> {
        Ok(unsafe { self.ctx.alloc_pinned::<i32>(n)? })
    }

    /// D2H into pinned host memory. `dst.as_ptr()` synchronizes then returns
    /// the value — used for the decoded token at the end of each decode step.
    pub fn download_i32_into_pinned(
        &self,
        src: &CudaSlice<i32>,
        dst: &mut PinnedHostSlice<i32>,
    ) -> anyhow::Result<()> {
        Ok(self.stream.memcpy_dtoh(src, dst)?)
    }

    pub fn upload_tensor(&self, t: &CpuTensor) -> anyhow::Result<GpuTensor> {
        let d = self.upload_f16(&t.data)?;
        Ok(GpuTensor::new(d, t.shape.clone()))
    }

    pub fn download_tensor(&self, t: &GpuTensor) -> anyhow::Result<CpuTensor> {
        let d = self.download_f16(&t.data)?;
        Ok(CpuTensor::new(d, t.shape.clone()))
    }

    pub fn synchronize(&self) -> anyhow::Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    // ---- cuBLAS wrappers ---------------------------------------------------

    /// y = x @ W^T  (x: [..., K], W: [N, K], y: [..., N]). β=0.
    pub fn linear_gpu(&self, x: &GpuTensor, w: &GpuWeight) -> anyhow::Result<GpuTensor> {
        let nd = x.ndim();
        let m: usize = x.shape()[..nd - 1].iter().product();
        let k = x.shape()[nd - 1];
        let n = w.rows;
        assert_eq!(
            k, w.cols,
            "linear_gpu K mismatch: x last={} vs W cols={}",
            k, w.cols
        );
        let mut y = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: f16::from_f32(1.0),
                    lda: k as i32,
                    ldb: k as i32,
                    beta: f16::from_f32(0.0),
                    ldc: n as i32,
                },
                &w.data,
                &x.data,
                &mut y,
            )?;
        }
        let mut out_shape = x.shape().to_vec();
        out_shape[nd - 1] = n;
        Ok(GpuTensor::new(y, out_shape))
    }

    /// Like `linear_gpu` but takes a raw `&CudaSlice` + explicit dims (rows, in_dim).
    /// Returns a raw `CudaSlice` of shape [rows, N]. Avoids GpuTensor ownership
    /// dance for borrowed slices (decoder attention hot path).
    pub fn linear_gpu_raw(
        &self,
        x: &CudaSlice<f16>,
        rows: usize,
        w: &GpuWeight,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let k = w.cols;
        let n = w.rows;
        let m = rows;
        assert_eq!(
            k, w.cols,
            "linear_gpu_raw K mismatch: W cols={}",
            w.cols
        );
        let mut y = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: f16::from_f32(1.0),
                    lda: k as i32,
                    ldb: k as i32,
                    beta: f16::from_f32(0.0),
                    ldc: n as i32,
                },
                &w.data,
                x,
                &mut y,
            )?;
        }
        Ok(y)
    }

    /// y = y + x @ W^T  — cuBLAS GEMM with β=1, in-place accumulation on `y`.
    /// Fuses a residual add into a linear projection (saves an add launch).
    pub fn linear_gpu_accum(
        &self,
        y: &mut GpuTensor,
        x: &GpuTensor,
        w: &GpuWeight,
    ) -> anyhow::Result<()> {
        let nd = x.ndim();
        let m: usize = x.shape()[..nd - 1].iter().product();
        let k = x.shape()[nd - 1];
        let n = w.rows;
        assert_eq!(k, w.cols, "linear_gpu_accum K mismatch");
        assert_eq!(y.numel(), m * n, "linear_gpu_accum y size mismatch");
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: f16::from_f32(1.0),
                    lda: k as i32,
                    ldb: k as i32,
                    beta: f16::from_f32(1.0),
                    ldc: n as i32,
                },
                &w.data,
                &x.data,
                &mut y.data,
            )?;
        }
        Ok(())
    }

    /// scores = Q @ K^T  (Q: [b,h,m,d], K: [b,h,n,d] → [b,h,m,n]). β=0.
    pub fn attention_qk(&self, q: &GpuTensor, k: &GpuTensor) -> anyhow::Result<GpuTensor> {
        let (b, h, m, d) = match q.shape() {
            [b, h, m, d] => (*b, *h, *m, *d),
            s => anyhow::bail!("attention_qk: Q must be 4-D [b,h,m,d], got {s:?}"),
        };
        let n = match k.shape() {
            [kb, kh, kn, kd] => {
                assert_eq!(*kb, b, "Q/K batch mismatch");
                assert_eq!(*kh, h, "Q/K heads mismatch");
                assert_eq!(*kd, d, "Q/K head_dim mismatch");
                *kn
            }
            s => anyhow::bail!("attention_qk: K must be 4-D [b,h,n,d], got {s:?}"),
        };
        let mut s = self.alloc_uninit_f16(b * h * m * n)?;
        let batch = (b * h) as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: d as i32,
                        alpha: f16::from_f32(1.0),
                        lda: d as i32,
                        ldb: d as i32,
                        beta: f16::from_f32(0.0),
                        ldc: n as i32,
                    },
                    batch_size: batch,
                    stride_a: (n * d) as i64,
                    stride_b: (m * d) as i64,
                    stride_c: (m * n) as i64,
                },
                &k.data,
                &q.data,
                &mut s,
            )?;
        }
        Ok(GpuTensor::new(s, vec![b, h, m, n]))
    }

    /// out = A @ V  (A: [b,h,m,n], V: [b,h,n,d] → [b,h,m,d]). β=0.
    pub fn attention_av(&self, a: &GpuTensor, v: &GpuTensor) -> anyhow::Result<GpuTensor> {
        let (b, h, m, n) = match a.shape() {
            [b, h, m, n] => (*b, *h, *m, *n),
            s => anyhow::bail!("attention_av: A must be 4-D [b,h,m,n], got {s:?}"),
        };
        let d = match v.shape() {
            [vb, vh, vn, vd] => {
                assert_eq!(*vb, b, "A/V batch mismatch");
                assert_eq!(*vh, h, "A/V heads mismatch");
                assert_eq!(*vn, n, "A/V n mismatch");
                *vd
            }
            s => anyhow::bail!("attention_av: V must be 4-D [b,h,n,d], got {s:?}"),
        };
        let mut out = self.alloc_uninit_f16(b * h * m * d)?;
        let batch = (b * h) as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: d as i32,
                        n: m as i32,
                        k: n as i32,
                        alpha: f16::from_f32(1.0),
                        lda: d as i32,
                        ldb: n as i32,
                        beta: f16::from_f32(0.0),
                        ldc: d as i32,
                    },
                    batch_size: batch,
                    stride_a: (n * d) as i64,
                    stride_b: (m * n) as i64,
                    stride_c: (m * d) as i64,
                },
                &v.data,
                &a.data,
                &mut out,
            )?;
        }
        Ok(GpuTensor::new(out, vec![b, h, m, d]))
    }

    /// Cached Q@K^T for incremental decoding. Q: [heads, q_seq, d] contiguous,
    /// K: [heads, max_seq, d] buffer with only first `k_seq` positions valid.
    /// `k_stride_seq` = max_seq (the buffer's allocated seq dim). Returns [heads, q_seq, k_seq].
    /// `alpha` folds in the head_dim^-0.5 scale (avoids a separate scale_inplace launch).
    pub fn attention_qk_cached(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        heads: usize,
        q_seq: usize,
        k_seq: usize,
        d: usize,
        k_stride_seq: usize,
        alpha: f32,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let m = q_seq;
        let n = k_seq;
        let mut s = self.alloc_uninit_f16(heads * m * n)?;
        let batch = heads as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: d as i32,
                        alpha: f16::from_f32(alpha),
                        lda: d as i32,
                        ldb: d as i32,
                        beta: f16::from_f32(0.0),
                        ldc: n as i32,
                    },
                    batch_size: batch,
                    stride_a: (k_stride_seq * d) as i64,
                    stride_b: (m * d) as i64,
                    stride_c: (m * n) as i64,
                },
                k,
                q,
                &mut s,
            )?;
        }
        Ok(s)
    }

    /// Cached A@V for incremental decoding. A: [heads, q_seq, k_seq] contiguous,
    /// V: [heads, max_seq, d] buffer with first `k_seq` positions valid.
    /// Returns [heads, q_seq, d].
    pub fn attention_av_cached(
        &self,
        a: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        heads: usize,
        q_seq: usize,
        k_seq: usize,
        d: usize,
        v_stride_seq: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let m = q_seq;
        let n = k_seq;
        let mut out = self.alloc_uninit_f16(heads * m * d)?;
        let batch = heads as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: d as i32,
                        n: m as i32,
                        k: n as i32,
                        alpha: f16::from_f32(1.0),
                        lda: d as i32,
                        ldb: n as i32,
                        beta: f16::from_f32(0.0),
                        ldc: d as i32,
                    },
                    batch_size: batch,
                    stride_a: (v_stride_seq * d) as i64,
                    stride_b: (m * n) as i64,
                    stride_c: (m * d) as i64,
                },
                v,
                a,
                &mut out,
            )?;
        }
        Ok(out)
    }

    // ---- kernel launch helpers (Stage 3.2 naive elementwise) --------------

    /// ReLU in place: x = relu(x). `n` = numel.
    pub fn relu_inplace(&self, x: &mut CudaSlice<f16>, n: usize) -> anyhow::Result<()> {
        let cfg = LaunchConfig::for_num_elems((n as u32).div_ceil(2));
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.k.relu_inplace);
        b.arg(&mut *x).arg(&n_i);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// LayerNorm over last dim of [rows, dim]: y = LN(x, w, b, eps).
    pub fn layer_norm(
        &self,
        x: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> anyhow::Result<CudaSlice<f16>> {
        // block.x = next pow2 >= dim, capped at 1024; shared mem = 2 * block * 4 bytes.
        let block = dim.next_power_of_two().min(1024).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (2 * block * 4) as u32,
        };
        let mut out = self.alloc_uninit_f16(rows * dim)?;
        let rows_i = rows as i32;
        let dim_i = dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.layer_norm);
        bb.arg(&mut out).arg(x).arg(w).arg(b).arg(&rows_i).arg(&dim_i).arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// SiLU(x + bias): y[r,c] = silu(x[r,c] + bias[c]). `n` = rows*cols.
    pub fn silu_bias(
        &self,
        x: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        n: usize,
        cols: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let cols_i = cols as i32;
        let mut out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.silu_bias);
        bb.arg(&mut out).arg(x).arg(bias).arg(&n_i).arg(&cols_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// In-place bias add: x[r,c] += bias[c]. `n` = rows*cols.
    pub fn add_bias_inplace(
        &self,
        x: &mut CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        n: usize,
        cols: usize,
    ) -> anyhow::Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let cols_i = cols as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_bias_inplace);
        bb.arg(&mut *x).arg(bias).arg(&n_i).arg(&cols_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// y = out + bias + residual. `n` = rows*cols.
    pub fn bias_residual(
        &self,
        out: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        residual: &CudaSlice<f16>,
        n: usize,
        cols: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let cols_i = cols as i32;
        let mut y = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.bias_residual);
        bb.arg(&mut y).arg(out).arg(bias).arg(residual).arg(&n_i).arg(&cols_i);
        unsafe { bb.launch(cfg) }?;
        Ok(y)
    }

    /// Softmax over last dim of [rows_ab, dim].
    pub fn softmax_last_dim(
        &self,
        x: &CudaSlice<f16>,
        rows_ab: usize,
        dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let block = dim.next_power_of_two().min(1024).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows_ab as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (2 * block * 4) as u32,
        };
        let mut out = self.alloc_uninit_f16(rows_ab * dim)?;
        let rows_i = rows_ab as i32;
        let dim_i = dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.softmax_last_dim);
        bb.arg(&mut out).arg(x).arg(&rows_i).arg(&dim_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Relative-position shift on [heads, q_len, pos_len].
    pub fn rel_shift_rank3(
        &self,
        bd: &CudaSlice<f16>,
        heads: usize,
        q_len: usize,
        pos_len: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let total = heads * q_len * pos_len;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let heads_i = heads as i32;
        let q_len_i = q_len as i32;
        let pos_len_i = pos_len as i32;
        let mut out = self.alloc_uninit_f16(total)?;
        let mut bb = self.stream.launch_builder(&self.k.rel_shift_rank3);
        bb.arg(bd)
            .arg(&mut out)
            .arg(&heads_i)
            .arg(&q_len_i)
            .arg(&pos_len_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// In-place scalar multiply: x *= scale. `n` = numel.
    pub fn scale_inplace(&self, x: &mut CudaSlice<f16>, n: usize, scale: f32) -> anyhow::Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.scale_inplace);
        bb.arg(&mut *x).arg(&n_i).arg(&scale);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Elementwise add: y = a + b. `n` = numel.
    pub fn add(&self, a: &CudaSlice<f16>, b: &CudaSlice<f16>, n: usize) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut y = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.add);
        bb.arg(&mut y).arg(a).arg(b).arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(y)
    }

    /// GLU + pointwise bias + depthwise conv1d k9 + SiLU, fused.
    /// x: [tokens, 2*C], bias: [2*C], cdw_params: [C, 10] (weight[0..8] + bias[9]).
    /// Output: [tokens, C].
    pub fn glu_depthwise_conv(
        &self,
        x: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        cdw_params: &CudaSlice<f16>,
        tokens: usize,
        channels: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = tokens * channels;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let tokens_i = tokens as i32;
        let channels_i = channels as i32;
        // Pass 1: GLU gate → gated [tokens, C]
        let mut gated = self.alloc_uninit_f16(n)?;
        let mut b1 = self.stream.launch_builder(&self.k.glu_gate);
        b1.arg(&mut gated).arg(x).arg(bias).arg(&tokens_i).arg(&channels_i);
        unsafe { b1.launch(cfg) }?;
        // Pass 2: depthwise conv k9 + SiLU
        let mut y = self.alloc_uninit_f16(n)?;
        let mut b2 = self.stream.launch_builder(&self.k.dw_conv_silu);
        b2.arg(&mut y).arg(&gated).arg(cdw_params).arg(&tokens_i).arg(&channels_i);
        unsafe { b2.launch(cfg) }?;
        Ok(y)
    }

    /// Fused qkv split + head reshape + pos bias: produces q_u, q_v, k, v.
    /// qkv: [tokens, 3*D], pos_bias_u/v: [D], outputs: [heads, tokens, head_dim].
    pub fn split_qkv_heads_bias(
        &self,
        qkv: &CudaSlice<f16>,
        pos_bias_u: &CudaSlice<f16>,
        pos_bias_v: &CudaSlice<f16>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<(CudaSlice<f16>, CudaSlice<f16>, CudaSlice<f16>, CudaSlice<f16>)> {
        let n = heads * tokens * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let tokens_i = tokens as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut q_u = self.alloc_uninit_f16(n)?;
        let mut q_v = self.alloc_uninit_f16(n)?;
        let mut k = self.alloc_uninit_f16(n)?;
        let mut v = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.split_qkv_heads_bias);
        bb.arg(qkv).arg(pos_bias_u).arg(pos_bias_v)
            .arg(&mut q_u).arg(&mut q_v).arg(&mut k).arg(&mut v)
            .arg(&tokens_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok((q_u, q_v, k, v))
    }

    /// Merge heads: [heads, tokens, head_dim] → [tokens, D].
    pub fn merge_heads(
        &self,
        inp: &CudaSlice<f16>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = heads * tokens * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let tokens_i = tokens as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.merge_heads);
        bb.arg(inp).arg(&mut out).arg(&tokens_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Reshape [tokens, D] → [heads, tokens, head_dim].
    pub fn split_to_heads(
        &self,
        inp: &CudaSlice<f16>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = heads * tokens * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let tokens_i = tokens as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.split_to_heads);
        bb.arg(inp).arg(&mut out).arg(&tokens_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Fused scale + causal mask + softmax over last dim of [heads, tokens, tokens].
    /// Upper triangle (j > i) is masked to -inf before softmax.
    pub fn causal_softmax(
        &self,
        x: &CudaSlice<f16>,
        heads: usize,
        tokens: usize,
        scale: f32,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let rows = heads * tokens;
        let block = tokens.next_power_of_two().min(1024).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (2 * block * 4) as u32,
        };
        let mut out = self.alloc_uninit_f16(rows * tokens)?;
        let heads_i = heads as i32;
        let tokens_i = tokens as i32;
        let mut bb = self.stream.launch_builder(&self.k.causal_softmax);
        bb.arg(&mut out).arg(x).arg(&heads_i).arg(&tokens_i).arg(&scale);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// ReLU + bias: y[r, c] = relu(x[r, c] + bias[c]).
    pub fn relu_bias(
        &self,
        x: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        n: usize,
        cols: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let cols_i = cols as i32;
        let mut y = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.relu_bias);
        bb.arg(&mut y).arg(x).arg(bias).arg(&n_i).arg(&cols_i);
        unsafe { bb.launch(cfg) }?;
        Ok(y)
    }

    /// Scatter-write one token's KV ([heads, head_dim]) into cache at row `pos`.
    /// cache: [heads, max_seq, head_dim].
    pub fn scatter_kv(
        &self,
        cache: &mut CudaSlice<f16>,
        inp: &CudaSlice<f16>,
        pos: usize,
        max_seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<()> {
        let n = heads * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let pos_i = pos as i32;
        let max_seq_i = max_seq as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.scatter_kv);
        bb.arg(&mut *cache).arg(inp).arg(&pos_i).arg(&max_seq_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Merge heads for a single token: [heads, stride_seq, head_dim] @ row 0 → [heads*head_dim].
    pub fn merge_heads_single(
        &self,
        inp: &CudaSlice<f16>,
        stride_seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = heads * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let stride_i = stride_seq as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.merge_heads_single);
        bb.arg(inp).arg(&mut out).arg(&stride_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Embedding gather: copy row `id` from table [vocab, dim] → out[dim].
    pub fn embed_gather(
        &self,
        table: &CudaSlice<f16>,
        id: usize,
        dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(dim as u32);
        let id_i = id as i32;
        let dim_i = dim as i32;
        let mut out = self.alloc_uninit_f16(dim)?;
        let mut bb = self.stream.launch_builder(&self.k.embed_gather);
        bb.arg(&mut out).arg(table).arg(&id_i).arg(&dim_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Standard 3x3 stride-2 pad-1 conv2d (groups=1) + ReLU.
    /// in: [N,Cin,H,W], w: [Cout,Cin,3,3], bias: [Cout]. out: [N,Cout,Hout,Wout].
    pub fn conv2d3x3_s2_relu(
        &self,
        inp: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        n: usize, cin: usize, cout: usize, h: usize, ww: usize,
    ) -> anyhow::Result<(CudaSlice<f16>, usize, usize)> {
        let hout = (h + 1) / 2;
        let wout = (ww + 1) / 2;
        let total = n * cout * hout * wout;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut out = self.alloc_uninit_f16(total)?;
        let (n_i, cin_i, cout_i, h_i, ww_i, hout_i, wout_i) =
            (n as i32, cin as i32, cout as i32, h as i32, ww as i32, hout as i32, wout as i32);
        let mut bb = self.stream.launch_builder(&self.k.conv2d3x3_s2_relu);
        bb.arg(&mut out).arg(inp).arg(w).arg(bias)
            .arg(&n_i).arg(&cin_i).arg(&cout_i).arg(&h_i).arg(&ww_i).arg(&hout_i).arg(&wout_i);
        unsafe { bb.launch(cfg) }?;
        Ok((out, hout, wout))
    }

    /// Depthwise 3x3 stride-2 pad-1 conv2d (groups=C).
    /// in: [N,C,H,W], w: [C,1,3,3], bias: [C]. out: [N,C,Hout,Wout].
    pub fn depthwise_conv2d3x3_s2(
        &self,
        inp: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        n: usize, c: usize, h: usize, ww: usize,
    ) -> anyhow::Result<(CudaSlice<f16>, usize, usize)> {
        let hout = (h + 1) / 2;
        let wout = (ww + 1) / 2;
        let total = n * c * hout * wout;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut out = self.alloc_uninit_f16(total)?;
        let (n_i, c_i, h_i, ww_i, hout_i, wout_i) =
            (n as i32, c as i32, h as i32, ww as i32, hout as i32, wout as i32);
        let mut bb = self.stream.launch_builder(&self.k.depthwise_conv2d3x3_s2);
        bb.arg(&mut out).arg(inp).arg(w).arg(bias)
            .arg(&n_i).arg(&c_i).arg(&h_i).arg(&ww_i).arg(&hout_i).arg(&wout_i);
        unsafe { bb.launch(cfg) }?;
        Ok((out, hout, wout))
    }

    /// Pointwise (1x1) conv2d over channels + ReLU. in: [N,Cin,H,W] → [N,Cout,H,W].
    pub fn pointwise_conv_relu(
        &self,
        inp: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        n: usize, cin: usize, cout: usize, h: usize, ww: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let total = n * cout * h * ww;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut out = self.alloc_uninit_f16(total)?;
        let (n_i, cin_i, cout_i, h_i, ww_i) = (n as i32, cin as i32, cout as i32, h as i32, ww as i32);
        let mut bb = self.stream.launch_builder(&self.k.pointwise_conv_relu);
        bb.arg(&mut out).arg(inp).arg(w).arg(bias)
            .arg(&n_i).arg(&cin_i).arg(&cout_i).arg(&h_i).arg(&ww_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Reshape NCHW [1,C,T,F] → [T, C*F].
    pub fn nchw_to_tokens(
        &self,
        inp: &CudaSlice<f16>,
        c: usize, t: usize, f: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let total = t * c * f;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut out = self.alloc_uninit_f16(total)?;
        let (c_i, t_i, f_i) = (c as i32, t as i32, f as i32);
        let mut bb = self.stream.launch_builder(&self.k.nchw_to_tokens);
        bb.arg(&mut out).arg(inp).arg(&c_i).arg(&t_i).arg(&f_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Fused encoder attention: rel_shift(bd) + ac, scale, softmax in one kernel.
    /// ac, bd: [heads, q_len, k_len]. Returns softmax [heads, q_len, k_len].
    pub fn fused_attn_scores_softmax(
        &self,
        ac: &CudaSlice<f16>,
        bd: &CudaSlice<f16>,
        heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let rows = heads * q_len;
        let block = k_len.next_power_of_two().min(1024).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (2 * block * 4) as u32,
        };
        let mut out = self.alloc_uninit_f16(rows * k_len)?;
        let heads_i = heads as i32;
        let q_len_i = q_len as i32;
        let k_len_i = k_len as i32;
        let mut bb = self.stream.launch_builder(&self.k.fused_attn_scores_softmax);
        bb.arg(&mut out).arg(ac).arg(bd).arg(&heads_i).arg(&q_len_i).arg(&k_len_i).arg(&scale);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Fused single-token self-QKV: split fused qkv + bias → Q [heads, head_dim],
    /// scatter K/V into cache at row `pos`. Returns Q [heads, head_dim].
    pub fn split_qkv_step_cached(
        &self,
        qkv: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        k_cache: &mut CudaSlice<f16>,
        v_cache: &mut CudaSlice<f16>,
        pos: usize,
        max_seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = heads * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let pos_i = pos as i32;
        let max_seq_i = max_seq as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut q_out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.split_qkv_step_cached);
        bb.arg(qkv).arg(bias).arg(&mut q_out)
            .arg(&mut *k_cache).arg(&mut *v_cache)
            .arg(&pos_i).arg(&max_seq_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok(q_out)
    }

    /// Batch embedding gather for prefill: out[s,d] = token_emb[ids[s],d] + pos_emb[s,d].
    /// `ids`: [seq] i32 device buffer. Returns [seq, dim] f16.
    pub fn embed_batch(
        &self,
        token_emb: &CudaSlice<f16>,
        pos_emb: &CudaSlice<f16>,
        ids: &CudaSlice<i32>,
        seq: usize,
        dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = seq * dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let seq_i = seq as i32;
        let dim_i = dim as i32;
        let mut out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.embed_gather_batch);
        bb.arg(&mut out).arg(token_emb).arg(pos_emb).arg(ids).arg(&seq_i).arg(&dim_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Single-token fused gather+add: out[d] = token_emb[tok,d] + pos_emb[pos,d].
    /// Replaces 2 gather kernels + 1 add kernel on the per-token decode path.
    pub fn embed_gather_add(
        &self,
        token_emb: &CudaSlice<f16>,
        pos_emb: &CudaSlice<f16>,
        tok: usize,
        pos: usize,
        dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(dim as u32);
        let tok_i = tok as i32;
        let pos_i = pos as i32;
        let dim_i = dim as i32;
        let mut out = self.alloc_uninit_f16(dim)?;
        let mut bb = self.stream.launch_builder(&self.k.embed_gather_add);
        bb.arg(&mut out).arg(token_emb).arg(pos_emb).arg(&tok_i).arg(&pos_i).arg(&dim_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Sinusoidal position encoding on GPU: out[pos_len, D]. Matches the CPU
    /// reference to within sub-ULP (GPU sinf/cosf/expf vs host f32), but that
    /// drift is enough to perturb attention — the encoder path keeps the CPU
    /// version. Retained for reference / non-encoder uses.
    #[allow(dead_code)]
    pub fn position_encoding(&self, tokens: usize, d_model: usize) -> anyhow::Result<CudaSlice<f16>> {
        let pos_len = 2 * tokens - 1;
        let n = pos_len * d_model;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let tokens_i = tokens as i32;
        let pos_len_i = pos_len as i32;
        let d_i = d_model as i32;
        let mut out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.position_encoding);
        bb.arg(&mut out).arg(&tokens_i).arg(&pos_len_i).arg(&d_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Batch QKV split + bias + K/V scatter for prefill. qkv: [seq, 3*D].
    /// Writes K/V into cache rows 0..seq-1, returns Q [heads, seq, head_dim].
    pub fn split_qkv_batch_scatter(
        &self,
        qkv: &CudaSlice<f16>,
        bias: &CudaSlice<f16>,
        k_cache: &mut CudaSlice<f16>,
        v_cache: &mut CudaSlice<f16>,
        seq: usize,
        max_seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let n = heads * seq * head_dim;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let seq_i = seq as i32;
        let max_seq_i = max_seq as i32;
        let heads_i = heads as i32;
        let hd_i = head_dim as i32;
        let mut q_out = self.alloc_uninit_f16(n)?;
        let mut bb = self.stream.launch_builder(&self.k.split_qkv_batch_scatter);
        bb.arg(qkv).arg(bias).arg(&mut q_out)
            .arg(&mut *k_cache).arg(&mut *v_cache)
            .arg(&seq_i).arg(&max_seq_i).arg(&heads_i).arg(&hd_i);
        unsafe { bb.launch(cfg) }?;
        Ok(q_out)
    }

    // ---- INT8 DP4A path (Stage 6) -----------------------------------------

    /// Reduce max(|a|) over n f16 elements → 1-element i32 buffer (int bit-repr).
    pub fn max_abs_reduce(&self, a: &CudaSlice<f16>, n: usize) -> anyhow::Result<CudaSlice<i32>> {
        let mut out = self.alloc_zeros_i32(1)?;
        let block = 1024u32;
        let grid = ((n as u32).div_ceil(block)).min(256);
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.max_abs_reduce);
        bb.arg(a).arg(&n_i).arg(&mut out);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Quantize f16 → i8 using per-tensor scale from `max_buf`.
    pub fn quantize_f16_i8(&self, a: &CudaSlice<f16>, n: usize, max_buf: &CudaSlice<i32>) -> anyhow::Result<CudaSlice<i8>> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut out = unsafe { self.stream.alloc::<i8>(n) }?;
        let mut bb = self.stream.launch_builder(&self.k.quantize_f16_i8);
        bb.arg(&mut out).arg(a).arg(&n_i).arg(max_buf);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// Dequantize int32 GEMM output → f16, applying act scale (from max_buf) and
    /// per-channel weight scale `wt_inv`. numel = M*out_dim.
    pub fn dequant_i32_f16(
        &self,
        c_i32: &CudaSlice<i32>,
        wt_inv: &CudaSlice<f16>,
        max_buf: &CudaSlice<i32>,
        numel: usize,
        out_dim: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let numel_i = numel as i32;
        let out_dim_i = out_dim as i32;
        let mut out = self.alloc_uninit_f16(numel)?;
        let mut bb = self.stream.launch_builder(&self.k.dequant_i32_f16);
        bb.arg(&mut out).arg(c_i32).arg(wt_inv).arg(max_buf).arg(&numel_i).arg(&out_dim_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }

    /// INT8 GEMM via cublasGemmEx (DP4A on sm_61). C = act @ weight^T.
    /// act_i8: [M, in_dim], weight_i8: [out_dim, in_dim], out_i32: [M, out_dim].
    pub fn linear_gpu_int8(
        &self,
        act_i8: &CudaSlice<i8>,
        weight_i8: &CudaSlice<i8>,
        out_i32: &mut CudaSlice<i32>,
        m: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> anyhow::Result<()> {
        use cudarc::driver::safe::{DevicePtr, DevicePtrMut};
        use std::ffi::c_void;
        let (a_ptr, _ga) = weight_i8.device_ptr(&self.stream);
        let (b_ptr, _gb) = act_i8.device_ptr(&self.stream);
        let (c_ptr, _gc) = out_i32.device_ptr_mut(&self.stream);
        let alpha: i32 = 1;
        let beta: i32 = 0;
        unsafe {
            sys::cublasGemmEx(
                *self.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                out_dim as i32, m as i32, in_dim as i32,
                &alpha as *const i32 as *const c_void,
                a_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_8I, in_dim as i32,
                b_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_8I, in_dim as i32,
                &beta as *const i32 as *const c_void,
                c_ptr as *mut c_void, sys::cudaDataType_t::CUDA_R_32I, out_dim as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            );
        }
        Ok(())
    }

    /// High-level INT8 linear: y = x[m, in_dim] @ w[out_dim, in_dim]^T → [m, out_dim]
    /// High-level INT8 linear: y = x[m, in] @ w[out, in]^T → [m, out] f16.
    /// Performs per-tensor activation quantize → cublasGemmEx (DP4A) →
    /// dequantize with the weight's per-channel scale. Returns f16 output ready
    /// for the next op. Takes the raw i8 weight + per-channel f16 scale as flat
    /// buffers (engine-level; the trait-level `linear_int8` wraps this from the
    /// `Int8Weight<B>` container).
    pub fn linear_int8_f16_raw(
        &self,
        x: &CudaSlice<f16>,
        w_data: &CudaSlice<i8>,
        w_wt_inv: &CudaSlice<f16>,
        out_dim: usize,
        in_dim: usize,
        m: usize,
    ) -> anyhow::Result<CudaSlice<f16>> {
        let mx = self.max_abs_reduce(x, m * in_dim)?;
        let x_i8 = self.quantize_f16_i8(x, m * in_dim, &mx)?;
        let mut c_i32 = self.alloc_uninit_i32(m * out_dim)?;
        self.linear_gpu_int8(&x_i8, w_data, &mut c_i32, m, in_dim, out_dim)?;
        self.dequant_i32_f16(&c_i32, w_wt_inv, &mx, m * out_dim, out_dim)
    }

    // ---- argmax (LM head / token selection) ---------------------------------

    /// Argmax over `n` f16 logits (starting at `offset` in `x`) → single-element
    /// i32 device buffer holding the winning index (0-based within the window).
    /// One 1024-thread block; shared mem = 2*block floats. Caller does D2H to
    /// read the result. Replaces a full-logits D2H + CPU reduction.
    pub fn argmax(&self, x: &CudaSlice<f16>, offset: usize, n: usize) -> anyhow::Result<CudaSlice<i32>> {
        let block = 1024u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (2 * block as usize * 4) as u32,
        };
        let mut out = self.alloc_uninit_i32(1)?;
        let off_i = offset as i32;
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.argmax);
        bb.arg(&mut out).arg(x).arg(&off_i).arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(out)
    }
}
