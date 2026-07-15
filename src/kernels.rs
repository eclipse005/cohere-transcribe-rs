//! Precompiled multi-arch PTX kernel registry (scheme B).
//!
//! CUDA kernels are compiled offline (`scripts/compile-ptx.ps1`) and embedded
//! via [`crate::prebuilt_ptx`]. At engine init we select the highest prebuilt
//! arch ≤ the device compute capability and load the module with the CUDA
//! driver — no NVRTC on the end-user machine.

#[cfg(feature = "cuda")]
mod inner {
    use anyhow::Context;

    /// Registry of loaded kernel functions. Add a `pub` field per kernel and a
    /// matching `module.load_function(...)?` line in `CudaKernels::load_all`.
    #[derive(Debug)]
    pub struct CudaKernels {
        pub relu_inplace: cudarc::driver::safe::CudaFunction,
        pub layer_norm: cudarc::driver::safe::CudaFunction,
        pub silu_bias: cudarc::driver::safe::CudaFunction,
        pub add_bias_inplace: cudarc::driver::safe::CudaFunction,
        pub bias_residual: cudarc::driver::safe::CudaFunction,
        pub softmax_last_dim: cudarc::driver::safe::CudaFunction,
        pub rel_shift_rank3: cudarc::driver::safe::CudaFunction,
        pub scale_inplace: cudarc::driver::safe::CudaFunction,
        pub add: cudarc::driver::safe::CudaFunction,
        pub glu_depthwise_conv: cudarc::driver::safe::CudaFunction,
        pub glu_gate: cudarc::driver::safe::CudaFunction,
        pub dw_conv_silu: cudarc::driver::safe::CudaFunction,
        pub split_qkv_heads_bias: cudarc::driver::safe::CudaFunction,
        pub merge_heads: cudarc::driver::safe::CudaFunction,
        pub split_to_heads: cudarc::driver::safe::CudaFunction,
        pub causal_softmax: cudarc::driver::safe::CudaFunction,
        pub relu_bias: cudarc::driver::safe::CudaFunction,
        pub scatter_kv: cudarc::driver::safe::CudaFunction,
        pub merge_heads_single: cudarc::driver::safe::CudaFunction,
        pub embed_gather: cudarc::driver::safe::CudaFunction,
        pub conv2d3x3_s2_relu: cudarc::driver::safe::CudaFunction,
        pub depthwise_conv2d3x3_s2: cudarc::driver::safe::CudaFunction,
        pub pointwise_conv_relu: cudarc::driver::safe::CudaFunction,
        pub nchw_to_tokens: cudarc::driver::safe::CudaFunction,
        pub max_abs_reduce: cudarc::driver::safe::CudaFunction,
        pub quantize_f16_i8: cudarc::driver::safe::CudaFunction,
        pub dequant_i32_f16: cudarc::driver::safe::CudaFunction,
        pub fused_attn_scores_softmax: cudarc::driver::safe::CudaFunction,
        pub split_qkv_step_cached: cudarc::driver::safe::CudaFunction,
        pub argmax: cudarc::driver::safe::CudaFunction,
        pub embed_gather_batch: cudarc::driver::safe::CudaFunction,
        pub embed_gather_add: cudarc::driver::safe::CudaFunction,
        pub position_encoding: cudarc::driver::safe::CudaFunction,
        pub split_qkv_batch_scatter: cudarc::driver::safe::CudaFunction,
    }

    impl CudaKernels {
        /// Load precompiled PTX for this device's compute capability and
        /// bind every kernel function into a fresh `CudaKernels`.
        pub fn load_all(
            ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        ) -> anyhow::Result<Self> {
            use cudarc::nvrtc::Ptx;

            let (major, minor) = ctx
                .compute_capability()
                .map_err(|e| anyhow::anyhow!("failed to query compute capability: {e:?}"))?;
            let (ptx_src, selected_sm) = crate::prebuilt_ptx::resolve_ptx_for_device(major, minor)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            log::info!(
                "loading prebuilt CUDA kernels for device sm_{}{} (selected sm_{})",
                major,
                minor,
                selected_sm
            );
            let module = ctx
                .load_module(Ptx::from_src(ptx_src))
                .context("loading prebuilt PTX module")?;

            let load_fn = |name: &str| {
                module
                    .load_function(name)
                    .with_context(|| format!("loading kernel {name}"))
            };

            Ok(Self {
                relu_inplace: load_fn("relu_inplace_f16")?,
                layer_norm: load_fn("layer_norm_f16")?,
                silu_bias: load_fn("silu_bias_f16")?,
                add_bias_inplace: load_fn("add_bias_inplace_f16")?,
                bias_residual: load_fn("bias_residual_f16")?,
                softmax_last_dim: load_fn("softmax_last_dim_f16")?,
                rel_shift_rank3: load_fn("rel_shift_rank3_f16")?,
                scale_inplace: load_fn("scale_inplace_f16")?,
                add: load_fn("add_f16")?,
                glu_depthwise_conv: load_fn("glu_depthwise_conv_f16")?,
                glu_gate: load_fn("glu_gate_f16")?,
                dw_conv_silu: load_fn("dw_conv_silu_f16")?,
                split_qkv_heads_bias: load_fn("split_qkv_heads_bias_f16")?,
                merge_heads: load_fn("merge_heads_f16")?,
                split_to_heads: load_fn("split_to_heads_f16")?,
                causal_softmax: load_fn("causal_softmax_f16")?,
                relu_bias: load_fn("relu_bias_f16")?,
                scatter_kv: load_fn("scatter_kv_f16")?,
                merge_heads_single: load_fn("merge_heads_single_f16")?,
                embed_gather: load_fn("embed_gather_f16")?,
                conv2d3x3_s2_relu: load_fn("conv2d3x3_s2_relu_f16")?,
                depthwise_conv2d3x3_s2: load_fn("depthwise_conv2d3x3_s2_f16")?,
                pointwise_conv_relu: load_fn("pointwise_conv_relu_f16")?,
                nchw_to_tokens: load_fn("nchw_to_tokens_f16")?,
                max_abs_reduce: load_fn("max_abs_reduce_f16")?,
                quantize_f16_i8: load_fn("quantize_f16_i8")?,
                dequant_i32_f16: load_fn("dequant_i32_f16")?,
                fused_attn_scores_softmax: load_fn("fused_attn_scores_softmax_f16")?,
                split_qkv_step_cached: load_fn("split_qkv_step_cached_f16")?,
                argmax: load_fn("argmax_f16")?,
                embed_gather_batch: load_fn("embed_gather_batch_f16")?,
                embed_gather_add: load_fn("embed_gather_add_f16")?,
                position_encoding: load_fn("position_encoding_f16")?,
                split_qkv_batch_scatter: load_fn("split_qkv_batch_scatter_f16")?,
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::CudaKernels;
