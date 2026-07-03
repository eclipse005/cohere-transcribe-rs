//! Backend abstraction: a single `Backend` trait whose two implementations
//! (`CudaBackend` via cudarc, `CpuBackend` via pure-Rust f32) let the encoder,
//! decoder, and pre-encoder run unchanged on either device.
//!
//! ## Storage contract
//! Both backends store activations and weights as **f16** (`half::f16`) and
//! accumulate in **f32**, mirroring cuBLAS `hgemm` semantics. This keeps the
//! CUDA and CPU numerical paths aligned (same rounding, same accumulation
//! order up to f16 input truncation), so the parity tests hold on both.
//!
//! ## INT8 / precision
//! INT8 (DP4A on sm_61, AVX-VNNI on x86) is an *optional* fast path layered on
//! top of the f16 storage, exposed via `Int8Weight` + `linear_int8`. A backend
//! that doesn't implement it returns `None` from `int8_enabled`, and the
//! generic forward code falls back to the f16 GEMM. This keeps precision
//! orthogonal to the backend — both CUDA and CPU can opt into INT8.
//!
//! ## Why a trait, not separate code paths
//! The encoder/decoder/pre-encoder forward logic is identical across backends
//! (same math, same fusion order). A trait gives one generic code path that
//! the parity tests pin for both, avoiding drift between two hand-written
//! copies.

use anyhow::Result;
use half::f16;

#[cfg(feature = "cuda")]
mod cuda;
mod cpu;
mod cpu_f16;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
pub use cpu::CpuBackend;
pub use cpu_f16::{CpuBackendF16, CpuWeightF16};

/// The element type held in a backend's storage buffer. Always f16 by contract
/// (see module docs); this alias exists so call sites read as `B::Buf` contents.
pub type Elem = f16;

/// A quantized weight for the INT8 fast path. Backend-agnostic in shape but the
/// `data` buffer type is backend-specific (`CudaSlice<i8>` / `Vec<i8>`).
pub struct Int8Weight<B: Backend> {
    pub data: B::IBuf,
    /// Per-output-channel scale f16: W[o,:] ≈ data[o,:] * wt_inv[o].
    pub wt_inv: B::Buf,
    pub rows: usize, // out
    pub cols: usize, // in
}

/// Backend abstraction. `Buf` is the f16 storage vector; `Weight` is an f16
/// weight matrix `[rows, cols]` row-major; `IBuf` is the i8 storage vector for
/// the optional INT8 path.
pub trait Backend: Sized {
    /// 1-D f16 storage buffer (CUDA: `CudaSlice<f16>`, CPU: `Vec<f16>`).
    type Buf: Clone;
    /// f16 weight matrix `[rows, cols]` row-major, with shape metadata.
    type Weight: Clone;
    /// 1-D i8 storage buffer for INT8 weights (CUDA: `CudaSlice<i8>`, CPU: `Vec<i8>`).
    type IBuf: Clone;

    /// Human-readable device name ("cuda:0" / "cpu").
    fn name(&self) -> &str;

    /// Whether this backend has the INT8 fast path wired up. Generic forward
    /// code checks this + weight presence to decide f16 vs int8.
    fn int8_enabled(&self) -> bool {
        false
    }

    // ---- memory primitives --------------------------------------------------

    /// Allocate an uninitialized f16 buffer of `n` elements. Caller must write
    /// every element before reading (β=0 GEMM outputs, fused kernels).
    fn alloc_uninit(&self, n: usize) -> Result<Self::Buf>;
    /// Upload a host f16 slice to the backend.
    fn upload_f16(&self, data: &[f16]) -> Result<Self::Buf>;
    /// Download an f16 buffer to the host.
    fn download_f16(&self, b: &Self::Buf) -> Result<Vec<f16>>;
    /// Block until all queued work is done. No-op on CPU (synchronous).
    fn synchronize(&self) -> Result<()>;

    // ---- weight construction ------------------------------------------------

    /// Build a `Weight` from host f16 data `[rows, cols]` row-major.
    fn upload_weight(&self, data: &[f16], rows: usize, cols: usize) -> Result<Self::Weight>;
    /// Build an INT8 weight from host data (only called when `int8_enabled`).
    fn upload_int8_weight(
        &self,
        data: &[i8],
        wt_inv: &[f16],
        rows: usize,
        cols: usize,
    ) -> Result<Int8Weight<Self>>;

    // ---- GEMM ---------------------------------------------------------------

    /// y = x @ W^T  (x: `[m, k]`, W: `[n, k]` → y: `[m, n]`). β=0.
    /// `x` raw buffer of `m*k` elements; returns raw buffer of `m*n`.
    fn linear(&self, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<Self::Buf>;

    /// y = y + x @ W^T (β=1 accumulation into `y`). Currently unused (dead code
    /// in the f16 paths) but part of the contract for future residual folding.
    fn linear_accum(&self, y: &mut Self::Buf, x: &Self::Buf, m: usize, w: &Self::Weight) -> Result<()>;

    /// Batched Q @ K^T. q: `[heads, m, d]`, k: `[heads, n, d]` (k read from a
    /// `[heads, k_stride, d]` cache with `k_seq` valid rows) → `[heads, m, k_seq]`.
    /// `alpha` folds in the head_dim^-0.5 scale (avoids a separate scale pass).
    /// Unified API: the encoder (batch=1, alpha=1.0 then scale) and the decoder
    /// (cached, alpha=scale) both call this.
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
    ) -> Result<Self::Buf>;

    /// Batched A @ V. a: `[heads, m, k_seq]`, v: `[heads, v_stride, d]` → `[heads, m, d]`.
    fn attention_av(
        &self,
        a: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        m: usize,
        k_seq: usize,
        d: usize,
        v_stride: usize,
    ) -> Result<Self::Buf>;

    /// INT8 linear: y = x[m, in] @ w[out, in]^T → [m, out] f16. Per-tensor
    /// activation quantize → int8 GEMM → per-channel dequant. Only called when
    /// `int8_enabled()` and an `Int8Weight` is present.
    fn linear_int8(&self, x: &Self::Buf, w: &Int8Weight<Self>, m: usize) -> Result<Self::Buf>;

    // ---- elementwise / fusion kernels --------------------------------------
    // Each mirrors a kernel in kernels.cu. CPU implements these in pure f32;
    // CUDA launches the corresponding NVRTC kernel.

    /// LayerNorm over last dim of `[rows, dim]`.
    fn layer_norm(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        b: &Self::Buf,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Self::Buf>;

    /// SiLU(x + bias): y[r,c] = silu(x[r,c] + bias[c]). `n` = rows*cols.
    fn silu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf>;

    /// In-place bias add: x[r,c] += bias[c]. `n` = rows*cols.
    fn add_bias_inplace(&self, x: &mut Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<()>;

    /// y = out + bias + residual. `n` = rows*cols.
    fn bias_residual(
        &self,
        out: &Self::Buf,
        bias: &Self::Buf,
        residual: &Self::Buf,
        n: usize,
        cols: usize,
    ) -> Result<Self::Buf>;

    /// Softmax over last dim of `[rows, dim]`.
    fn softmax_last_dim(&self, x: &Self::Buf, rows: usize, dim: usize) -> Result<Self::Buf>;

    /// In-place scalar multiply: x *= scale.
    fn scale_inplace(&self, x: &mut Self::Buf, n: usize, scale: f32) -> Result<()>;

    /// Elementwise add: y = a + b.
    fn add(&self, a: &Self::Buf, b: &Self::Buf, n: usize) -> Result<Self::Buf>;

    /// GLU + pointwise bias + depthwise conv1d k9 + SiLU, fused.
    /// `x`: `[tokens, 2*C]`, `bias`: `[2*C]`, `cdw_params`: `[C, 10]` → `[tokens, C]`.
    fn glu_depthwise_conv(
        &self,
        x: &Self::Buf,
        bias: &Self::Buf,
        cdw_params: &Self::Buf,
        tokens: usize,
        channels: usize,
    ) -> Result<Self::Buf>;

    /// Fused qkv split + head reshape + pos bias. qkv: `[tokens, 3*D]`,
    /// pos_bias_u/v: `[D]` → (q_u, q_v, k, v) each `[heads, tokens, head_dim]`.
    fn split_qkv_heads_bias(
        &self,
        qkv: &Self::Buf,
        pos_bias_u: &Self::Buf,
        pos_bias_v: &Self::Buf,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(Self::Buf, Self::Buf, Self::Buf, Self::Buf)>;

    /// Merge heads: `[heads, tokens, head_dim]` → `[tokens, D]`.
    fn merge_heads(&self, inp: &Self::Buf, tokens: usize, heads: usize, head_dim: usize)
        -> Result<Self::Buf>;

    /// Reshape `[tokens, D]` → `[heads, tokens, head_dim]`.
    fn split_to_heads(&self, inp: &Self::Buf, tokens: usize, heads: usize, head_dim: usize)
        -> Result<Self::Buf>;

    /// Fused scale + causal mask + softmax over last dim of `[heads, tokens, tokens]`.
    fn causal_softmax(&self, x: &Self::Buf, heads: usize, tokens: usize, scale: f32)
        -> Result<Self::Buf>;

    /// ReLU + bias: y[r,c] = relu(x[r,c] + bias[c]).
    fn relu_bias(&self, x: &Self::Buf, bias: &Self::Buf, n: usize, cols: usize) -> Result<Self::Buf>;

    /// Merge heads for a single token: `[heads, stride, head_dim]` row 0 → `[heads*head_dim]`.
    fn merge_heads_single(&self, inp: &Self::Buf, stride: usize, heads: usize, head_dim: usize)
        -> Result<Self::Buf>;

    /// Fused single-token QKV split + bias + K/V scatter into cache at `pos`.
    /// qkv: `[3*D]`, returns Q `[heads, head_dim]`, writes k_cache/v_cache.
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
    ) -> Result<Self::Buf>;

    /// Batch QKV split + bias + K/V scatter for prefill. qkv: `[seq, 3*D]`,
    /// writes cache rows 0..seq-1, returns Q `[heads, seq, head_dim]`.
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
    ) -> Result<Self::Buf>;

    /// Fused encoder attention scores: rel_shift(bd) + ac, scale, softmax.
    /// ac, bd: `[heads, q_len, k_len]` → softmax `[heads, q_len, k_len]`.
    fn fused_attn_scores_softmax(
        &self,
        ac: &Self::Buf,
        bd: &Self::Buf,
        heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
    ) -> Result<Self::Buf>;

    // ---- embedding ----------------------------------------------------------

    /// Single-token fused gather+add: out[d] = token_emb[tok,d] + pos_emb[pos,d].
    fn embed_gather_add(
        &self,
        token_emb: &Self::Buf,
        pos_emb: &Self::Buf,
        tok: usize,
        pos: usize,
        dim: usize,
    ) -> Result<Self::Buf>;

    /// Batch embedding: out[s,d] = token_emb[ids[s],d] + pos_emb[s,d].
    fn embed_batch(
        &self,
        token_emb: &Self::Buf,
        pos_emb: &Self::Buf,
        ids: &[i32],
        seq: usize,
        dim: usize,
    ) -> Result<Self::Buf>;

    // ---- pre-encoder conv stack --------------------------------------------

    /// Standard 3x3 stride-2 pad-1 conv2d (groups=1) + ReLU.
    /// in: `[n, cin, h, w]`, w: `[cout, cin, 3, 3]`, bias: `[cout]`.
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
    ) -> Result<(Self::Buf, usize, usize)>;

    /// Depthwise 3x3 stride-2 pad-1 conv2d (groups=C).
    fn depthwise_conv2d3x3_s2(
        &self,
        inp: &Self::Buf,
        w: &Self::Buf,
        bias: &Self::Buf,
        n: usize,
        c: usize,
        h: usize,
        ww: usize,
    ) -> Result<(Self::Buf, usize, usize)>;

    /// Pointwise (1x1) conv2d + ReLU.
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
    ) -> Result<Self::Buf>;

    /// Reshape NCHW `[1, C, T, F]` → `[T, C*F]`.
    fn nchw_to_tokens(&self, inp: &Self::Buf, c: usize, t: usize, f: usize) -> Result<Self::Buf>;

    // ---- token selection ----------------------------------------------------

    /// Argmax over `n` f16 logits (starting at `offset`) → host i32 index.
    /// On CUDA this does a device reduction + small D2H; on CPU it's a host loop.
    /// Returning the host i32 (not a device buffer) keeps the trait sync-free.
    fn argmax(&self, x: &Self::Buf, offset: usize, n: usize) -> Result<i32>;

    // ---- buffer access helpers ---------------------------------------------
    // CPU needs to read buffer contents for some ops (e.g. mel input reshape
    // in pre-encoder). These give a typed view; CUDA downloads on demand.

    /// Length (number of f16 elements) of a buffer.
    fn buf_len(b: &Self::Buf) -> usize;

    /// A borrow/clone of the underlying f16 storage of a `Weight` (CUDA:
    /// `GpuWeight.data`, CPU: `CpuWeight.data`). Needed for embedding gather
    /// and any op that reads weight values element-wise rather than as a GEMM.
    fn weight_data(&self, w: &Self::Weight) -> Self::Buf;
}
