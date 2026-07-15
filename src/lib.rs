//! Hand-written cudarc + cuBLAS CUDA engine for Cohere ASR.
//!
//! Kernels ship as precompiled multi-arch PTX (scheme B) — no NVRTC at
//! runtime. Weights are f16-resident on the device; matrix products go through
//! cuBLAS `hgemm` (f16 storage / f32 accumulate — ≈ FP32 throughput on sm_61,
//! since GP104 consumer Pascal has only 1/64-rate native f16 ALUs but cuBLAS
//! accumulates in fp32). Elementwise and fusion kernels accumulate in f32.
//!
//! The three RTFx levers over the candle baseline, in order of impact on this
//! GPU (P104-100, sm_61):
//!  1. Zero-allocation scratch buffers in the encode/decode hot loops
//!     (`EncodeScratch` / `DecodeScratch`) — every per-layer temp is
//!     pre-allocated once and written into via `_into` entry points.
//!  2. `alloc_uninit` instead of `alloc_zeros` — Pascal's driver throttles
//!     `cudaMemset`; any fully-overwritten output skips it.
//!  3. β=1 residual fold into cuBLAS GEMM — `out_proj` / `down_proj` residuals
//!     are folded into the GEMM output, removing one `add_inplace` launch.
//!
//! All CUDA-only. Builds without the `cuda` feature produce a no-CUDA stub.

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

pub mod backend;
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod prebuilt_ptx;
pub mod raw_tensor;
pub mod weights;
pub mod audio;
pub mod features;
pub mod prepare;
pub mod tokenizer;

#[cfg(feature = "cuda")]
pub mod engine;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod weights_gpu;

#[cfg(feature = "cuda")]
pub mod pre_encoder;
#[cfg(feature = "cuda")]
pub mod encoder;
#[cfg(feature = "cuda")]
pub mod decoder;
#[cfg(feature = "cuda")]
pub mod transcribe;

pub fn run_cli() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    {
        cli::run()
    }
    #[cfg(not(feature = "cuda"))]
    {
        anyhow::bail!("native-transcribe was built without the `cuda` feature; rebuild with --features cuda")
    }
}

#[cfg(feature = "cuda")]
mod cli;

/// Re-exported pure-Rust audio/prep/tokenizer pieces for downstream code.
pub mod host {
    pub use crate::audio::{build_prompt, join_chunk_texts, split_audio_chunks_energy_meta};
    pub use crate::features::{FeatureConfig, FeatureExtractor, FeatureOutput};
    pub use crate::prepare::{
        NativePrepareOptions, NativePreparedAudio, NativePreparedChunk, prepare_native_audio,
    };
    pub use crate::tokenizer::CohereTokenizer;
}
