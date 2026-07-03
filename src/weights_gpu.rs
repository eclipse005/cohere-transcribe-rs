//! Backend-resident model weights, built from the mmap'd safetensors with all
//! load-time rewrites done on the host (weight-transpose, 0.5 macaron FFN
//! residual bake, fused QKV, BatchNorm-folded depthwise conv packed to
//! `[C, 10]`). Everything uploaded once as f16 and held for the session.
//!
//! Fully generic over `B: Backend` — the same code builds a CUDA-resident
//! (`CudaBackend`) or CPU-resident (`CpuBackend`) weight set. Host rewrites are
//! identical across backends; only the storage type (`B::Buf` / `B::Weight`)
//! differs.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use anyhow::Context;
use half::f16;

use crate::backend::{Backend, Int8Weight};
use crate::raw_tensor::{
    RawTensor, as_2d, fold_bn_into_depthwise_conv1d, pack_depthwise_conv1d_params, scale_f16,
};
use crate::weights::{get, load_weights};

// ---------------------------------------------------------------------------
// Pre-encoder: dw_striding conv stack (8x subsampling) + out projection.
// ---------------------------------------------------------------------------

/// Pre-encoder conv layer (pointwise or depthwise-stride-2).
pub struct PreConv<B: Backend> {
    /// Flattened weight, shape per variant stored in `weight_shape`.
    pub weight: B::Buf,
    pub bias: B::Buf,
    pub kind: PreConvKind,
}

#[derive(Debug, Clone, Copy)]
pub enum PreConvKind {
    /// Standard 2-D conv, stride 2, pad 1, groups = `groups`, kernel 3x3.
    /// Weight layout in safetensors: [out, in/groups, 3, 3].
    Stride2 { groups: usize },
    /// 1x1 (kernel 1, no pad) pointwise 2-D conv = a linear over channels.
    /// Weight layout: [out, in, 1, 1].
    Pointwise1x1,
}

pub struct PreEncoderWeights<B: Backend> {
    pub conv0: PreConv<B>, // stride2 groups=1
    pub conv2: PreConv<B>, // stride2 groups=256
    pub conv3: PreConv<B>, // pointwise 1x1
    pub conv5: PreConv<B>, // stride2 groups=256
    pub conv6: PreConv<B>, // pointwise 1x1
    /// out projection: weight-transposed [d_model=1280, flattened_in], bias [1280].
    pub out_w_t: B::Weight,
    pub out_b: B::Buf,
    pub d_model: usize,
}

// ---------------------------------------------------------------------------
// Conformer encoder layer (macaron: FFN1 -> self-attn -> conv -> FFN2 -> LN).
// ---------------------------------------------------------------------------

pub struct EncoderLayerWeights<B: Backend> {
    // FFN1 (LN -> l1 -> silu -> l2(+0.5 baked into l2) -> residual)
    pub ffn1_norm_w: B::Buf,
    pub ffn1_norm_b: B::Buf,
    pub ffn1_l1_w_t: B::Weight,
    pub ffn1_l1_b: B::Buf,
    pub ffn1_l2_w_t: B::Weight, // 0.5 baked in
    pub ffn1_l2_b: B::Buf,     // 0.5 baked in
    // Self-attention (rel-pos)
    pub att_norm_w: B::Buf,
    pub att_norm_b: B::Buf,
    /// Fused [q|k|v] weight, weight-transposed: [3*d_model, d_model].
    pub qkv_w_t: B::Weight,
    /// Fused q/k/v bias: [3*d_model].
    pub qkv_b: B::Buf,
    /// Position projection weight-transposed [d_model, d_model], no bias.
    pub p_w_t: B::Weight,
    /// Output projection weight-transposed [d_model, d_model] + bias.
    pub out_w_t: B::Weight,
    pub out_b: B::Buf,
    /// Rel-pos content/position biases: [d_model] each (split across heads at runtime).
    pub pos_bias_u: B::Buf,
    pub pos_bias_v: B::Buf,
    // Conv module (LN -> pointwise1 -> GLU+depthwise-k9+silu -> pointwise2 -> residual)
    pub conv_norm_w: B::Buf,
    pub conv_norm_b: B::Buf,
    /// pointwise1 weight-transposed [2*d_model, d_model] + bias [2*d_model].
    pub cpw1_w_t: B::Weight,
    pub cpw1_b: B::Buf,
    /// Packed depthwise params [d_model, 10] (9-tap folded weight + bias).
    pub cdw_params: B::Buf,
    /// pointwise2 weight-transposed [d_model, d_model] + bias [d_model].
    pub cpw2_w_t: B::Weight,
    pub cpw2_b: B::Buf,
    // FFN2 (same shape as FFN1, 0.5 baked into l2)
    pub ffn2_norm_w: B::Buf,
    pub ffn2_norm_b: B::Buf,
    pub ffn2_l1_w_t: B::Weight,
    pub ffn2_l1_b: B::Buf,
    pub ffn2_l2_w_t: B::Weight,
    pub ffn2_l2_b: B::Buf,
    // Final output LayerNorm (no residual).
    pub out_norm_w: B::Buf,
    pub out_norm_b: B::Buf,
    // INT8 DP4A FFN weights (Option — only loaded when int8 mode requested).
    pub ffn1_l1_i8: Option<Int8Weight<B>>,
    pub ffn1_l2_i8: Option<Int8Weight<B>>,
    pub ffn2_l1_i8: Option<Int8Weight<B>>,
    pub ffn2_l2_i8: Option<Int8Weight<B>>,
    // INT8 conv pointwise weights (cpw1, cpw2).
    pub cpw1_i8: Option<Int8Weight<B>>,
    pub cpw2_i8: Option<Int8Weight<B>>,
    // INT8 attention projection weights (qkv, pos, out). Attention is more
    // sensitive to quantization than FFN, so gated behind the same int8 flag.
    pub qkv_w_i8: Option<Int8Weight<B>>,
    pub p_w_i8: Option<Int8Weight<B>>,
    pub out_w_i8: Option<Int8Weight<B>>,
}

pub struct EncoderWeights<B: Backend> {
    pub layers: Vec<EncoderLayerWeights<B>>,
    /// Encoder -> decoder projection weight-transposed [1024, 1280] + bias [1024].
    pub enc_proj_w_t: B::Weight,
    pub enc_proj_b: B::Buf,
}

// ---------------------------------------------------------------------------
// Decoder (8-layer transformer encoder-decoder) + embeddings + LM head.
// ---------------------------------------------------------------------------

pub struct DecoderLayerWeights<B: Backend> {
    pub norm1_w: B::Buf,
    pub norm1_b: B::Buf,
    pub self_q_w_t: B::Weight,
    pub self_q_b: B::Buf,
    pub self_k_w_t: B::Weight,
    pub self_k_b: B::Buf,
    pub self_v_w_t: B::Weight,
    pub self_v_b: B::Buf,
    /// Fused [q|k|v] self-attention weight [3*d, d] + bias [3*d] (one GEMM vs three).
    pub self_qkv_w_t: B::Weight,
    pub self_qkv_b: B::Buf,
    pub self_out_w_t: B::Weight,
    pub self_out_b: B::Buf,
    pub norm2_w: B::Buf,
    pub norm2_b: B::Buf,
    pub cross_q_w_t: B::Weight,
    pub cross_q_b: B::Buf,
    pub cross_k_w_t: B::Weight,
    pub cross_k_b: B::Buf,
    pub cross_v_w_t: B::Weight,
    pub cross_v_b: B::Buf,
    pub cross_out_w_t: B::Weight,
    pub cross_out_b: B::Buf,
    pub norm3_w: B::Buf,
    pub norm3_b: B::Buf,
    pub ffn_in_w_t: B::Weight,
    pub ffn_in_b: B::Buf,
    pub ffn_out_w_t: B::Weight,
    pub ffn_out_b: B::Buf,
    // INT8 decoder projections (optional). FFN first (K=4096 benefits most from
    // DP4A even at M=1); attention projections gated on the same flag.
    pub self_qkv_w_i8: Option<Int8Weight<B>>,
    pub self_out_w_i8: Option<Int8Weight<B>>,
    pub cross_q_w_i8: Option<Int8Weight<B>>,
    pub cross_out_w_i8: Option<Int8Weight<B>>,
    pub ffn_in_w_i8: Option<Int8Weight<B>>,
    pub ffn_out_w_i8: Option<Int8Weight<B>>,
}

pub struct DecoderWeights<B: Backend> {
    pub layers: Vec<DecoderLayerWeights<B>>,
    /// Token embedding [vocab, hidden=1024].
    pub token_embedding: B::Weight, // rows=vocab, cols=hidden
    /// Learned position embedding [max_seq=1024, hidden=1024].
    pub position_embedding: B::Weight, // rows=max_seq, cols=hidden
    pub emb_norm_w: B::Buf,
    pub emb_norm_b: B::Buf,
    pub final_norm_w: B::Buf,
    pub final_norm_b: B::Buf,
    /// LM head weight-transposed [vocab=16384, hidden=1024] + bias.
    pub lm_w_t: B::Weight,
    pub lm_b: B::Buf,
}

pub struct ModelWeights<B: Backend> {
    pub pre_encoder: PreEncoderWeights<B>,
    pub encoder: EncoderWeights<B>,
    pub decoder: DecoderWeights<B>,
}

// ---------------------------------------------------------------------------
// Helpers — all host rewrites are backend-independent; only the final upload
// differs. Each takes `&B` and builds a `B::Buf` / `B::Weight` / `Int8Weight<B>`.
// ---------------------------------------------------------------------------

/// Load a tensor, decode to f16, return the flat vec + shape.
fn raw_f16(
    weights: &HashMap<String, RawTensor>,
    name: &str,
) -> anyhow::Result<(Vec<f16>, Vec<usize>)> {
    let t = get(weights, name).with_context(|| format!("weight {name}"))?;
    Ok((t.to_f16_vec()?, t.shape.clone()))
}

/// Load a 1-D f16 tensor as a backend buffer.
fn vec1<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    name: &str,
    backend: &B,
) -> anyhow::Result<B::Buf> {
    let (data, shape) = raw_f16(weights, name)?;
    let n = shape.iter().product::<usize>();
    assert_eq!(data.len(), n, "vec1 {name}: len {} != product {:?}", data.len(), shape);
    backend.upload_f16(&data)
}

/// Quantize a 2-D weight `[out, in]` to INT8 per-output-channel. `scale` bakes
/// in a constant factor (e.g. 0.5 for macaron FFN l2). Uploads via the backend's
/// INT8 weight constructor.
fn weight_t_int8<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    name: &str,
    backend: &B,
    scale: f32,
) -> anyhow::Result<Int8Weight<B>> {
    let (data, shape) = raw_f16(weights, name)?;
    let (rows, cols) = as_2d(&shape)?; // [out, in]
    int8_from_f16(data, rows, cols, scale, backend)
}

/// Per-output-channel INT8 quantization of an already-assembled f16 weight
/// `[out, in]` row-major. Shared by tensor-name-based and fused-weight callers.
fn int8_from_f16<B: Backend>(
    data: Vec<f16>,
    rows: usize,
    cols: usize,
    scale: f32,
    backend: &B,
) -> anyhow::Result<Int8Weight<B>> {
    let mut wq = vec![0i8; rows * cols];
    let mut wt_inv = vec![f16::ZERO; rows];
    for o in 0..rows {
        let row = &data[o * cols..(o + 1) * cols];
        let mut mx = 0.0f32;
        for &v in row {
            let a = (v.to_f32() * scale).abs();
            if a > mx { mx = a; }
        }
        let inv = if mx > 0.0 { 127.0f32 / mx } else { 0.0 };
        wt_inv[o] = f16::from_f32(mx / 127.0);
        for k in 0..cols {
            let q = ((row[k].to_f32() * scale) * inv).round() as i32;
            let q = q.clamp(-127, 127) as i8;
            wq[o * cols + k] = q;
        }
    }
    backend.upload_int8_weight(&wq, &wt_inv, rows, cols)
}

/// INT8 quantize a Conv1d pointwise weight stored as `[out, in, 1]` → flatten to
/// `[out, in]` and per-channel quantize.
fn weight_t_int8_conv1d<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    name: &str,
    backend: &B,
) -> anyhow::Result<Int8Weight<B>> {
    let (data, shape) = raw_f16(weights, name)?;
    let (rows, cols) = match shape.as_slice() {
        [out, _in, 1] => (*out, *_in),
        _ => anyhow::bail!("expected [out,in,1] for {name}, got {shape:?}"),
    };
    let mut wq = vec![0i8; rows * cols];
    let mut wt_inv = vec![f16::ZERO; rows];
    for o in 0..rows {
        let row = &data[o * cols..(o + 1) * cols];
        let mut mx = 0.0f32;
        for &v in row { let a = v.to_f32().abs(); if a > mx { mx = a; } }
        let inv = if mx > 0.0 { 127.0f32 / mx } else { 0.0 };
        wt_inv[o] = f16::from_f32(mx / 127.0);
        for k in 0..cols {
            let q = (row[k].to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
            wq[o * cols + k] = q;
        }
    }
    backend.upload_int8_weight(&wq, &wt_inv, rows, cols)
}

/// Load a 2-D weight `[out=N, in=K]` (safetensors layout) and upload as-is.
/// The linear op reads this buffer under `OP_T`, computing
/// `y = x[...,K] @ W^T` -> `[...,N]`.
/// B::Weight { rows=N=out, cols=K=in } with the data uploaded as [out, in] row-major.
fn weight_t<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    name: &str,
    backend: &B,
) -> anyhow::Result<B::Weight> {
    let (data, shape) = raw_f16(weights, name)?;
    let (rows, cols) = as_2d(&shape)?; // [out, in]
    let n = rows; // out
    let k = cols; // in
    assert_eq!(data.len(), n * k);
    backend.upload_weight(&data, n, k)
}

/// Like `weight_t` but scale every element by `s` (the 0.5 macaron bake).
fn weight_t_scaled<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    name: &str,
    backend: &B,
    s: f32,
) -> anyhow::Result<B::Weight> {
    let (data, shape) = raw_f16(weights, name)?;
    let (rows, cols) = as_2d(&shape)?;
    let scaled = scale_f16(&data, s);
    let n = rows;
    let k = cols;
    backend.upload_weight(&scaled, n, k)
}

impl<B: Backend> PreEncoderWeights<B> {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        backend: &B,
    ) -> anyhow::Result<Self> {
        let load_conv = |idx: usize, kind: PreConvKind| -> anyhow::Result<PreConv<B>> {
            let (w, _) = raw_f16(weights, &format!("encoder.pre_encode.conv.{idx}.weight"))?;
            let (b, _) = raw_f16(weights, &format!("encoder.pre_encode.conv.{idx}.bias"))?;
            Ok(PreConv {
                weight: backend.upload_f16(&w)?,
                bias: backend.upload_f16(&b)?,
                kind,
            })
        };

        let conv0 = load_conv(0, PreConvKind::Stride2 { groups: 1 })?;
        let conv2 = load_conv(2, PreConvKind::Stride2 { groups: 256 })?;
        let conv3 = load_conv(3, PreConvKind::Pointwise1x1)?;
        let conv5 = load_conv(5, PreConvKind::Stride2 { groups: 256 })?;
        let conv6 = load_conv(6, PreConvKind::Pointwise1x1)?;

        let out_w_t = weight_t(weights, "encoder.pre_encode.out.weight", backend)?;
        let out_b = vec1(weights, "encoder.pre_encode.out.bias", backend)?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out_w_t,
            out_b,
            d_model: 1280,
        })
    }
}

impl<B: Backend> EncoderLayerWeights<B> {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        layer_idx: usize,
        backend: &B,
        int8: bool,
    ) -> anyhow::Result<Self> {
        let p = |name: &str| format!("encoder.layers.{layer_idx}.{name}");

        // FFN1
        let ffn1_norm_w = vec1(weights, &p("norm_feed_forward1.weight"), backend)?;
        let ffn1_norm_b = vec1(weights, &p("norm_feed_forward1.bias"), backend)?;
        let ffn1_l1_w_t = weight_t(weights, &p("feed_forward1.linear1.weight"), backend)?;
        let ffn1_l1_b = vec1(weights, &p("feed_forward1.linear1.bias"), backend)?;
        let ffn1_l2_w_t = weight_t_scaled(weights, &p("feed_forward1.linear2.weight"), backend, 0.5)?;
        let ffn1_l2_b = {
            let (d, _) = raw_f16(weights, &p("feed_forward1.linear2.bias"))?;
            backend.upload_f16(&scale_f16(&d, 0.5))?
        };

        // Self-attention norms + fused QKV + pos + out + biases
        let att_norm_w = vec1(weights, &p("norm_self_att.weight"), backend)?;
        let att_norm_b = vec1(weights, &p("norm_self_att.bias"), backend)?;
        let qkv_w_t = fuse_qkv_weight(weights, &p("self_attn.linear_q.weight"), &p("self_attn.linear_k.weight"), &p("self_attn.linear_v.weight"), backend)?;
        let qkv_b = fuse_qkv_bias(weights, &p("self_attn.linear_q.bias"), &p("self_attn.linear_k.bias"), &p("self_attn.linear_v.bias"), backend)?;
        let p_w_t = weight_t(weights, &p("self_attn.linear_pos.weight"), backend)?;
        let out_w_t = weight_t(weights, &p("self_attn.linear_out.weight"), backend)?;
        let out_b = vec1(weights, &p("self_attn.linear_out.bias"), backend)?;
        let pos_bias_u = vec1(weights, &p("self_attn.pos_bias_u"), backend)?;
        let pos_bias_v = vec1(weights, &p("self_attn.pos_bias_v"), backend)?;

        // Conv module with BatchNorm folded into depthwise conv.
        let conv_norm_w = vec1(weights, &p("norm_conv.weight"), backend)?;
        let conv_norm_b = vec1(weights, &p("norm_conv.bias"), backend)?;
        let cpw1_w_t = weight_t_reshape1x(weights, &p("conv.pointwise_conv1.weight"), backend)?;
        let cpw1_b = vec1(weights, &p("conv.pointwise_conv1.bias"), backend)?;
        let cdw_params = load_packed_depthwise(weights, &p("conv."), backend)?;
        let cpw2_w_t = weight_t_reshape1x(weights, &p("conv.pointwise_conv2.weight"), backend)?;
        let cpw2_b = vec1(weights, &p("conv.pointwise_conv2.bias"), backend)?;

        // FFN2
        let ffn2_norm_w = vec1(weights, &p("norm_feed_forward2.weight"), backend)?;
        let ffn2_norm_b = vec1(weights, &p("norm_feed_forward2.bias"), backend)?;
        let ffn2_l1_w_t = weight_t(weights, &p("feed_forward2.linear1.weight"), backend)?;
        let ffn2_l1_b = vec1(weights, &p("feed_forward2.linear1.bias"), backend)?;
        let ffn2_l2_w_t = weight_t_scaled(weights, &p("feed_forward2.linear2.weight"), backend, 0.5)?;
        let ffn2_l2_b = {
            let (d, _) = raw_f16(weights, &p("feed_forward2.linear2.bias"))?;
            backend.upload_f16(&scale_f16(&d, 0.5))?
        };

        let out_norm_w = vec1(weights, &p("norm_out.weight"), backend)?;
        let out_norm_b = vec1(weights, &p("norm_out.bias"), backend)?;

        // INT8 DP4A FFN weights (optional). l2 weights have 0.5 baked in already
        // (weight_t_scaled), so the int8 quant operates on the scaled values.
        let (ffn1_l1_i8, ffn1_l2_i8, ffn2_l1_i8, ffn2_l2_i8, cpw1_i8, cpw2_i8,
             qkv_w_i8, p_w_i8, out_w_i8) = if int8 {
            // Fused [q|k|v] INT8: assemble the same [3d, d] layout as
            // fuse_qkv_weight, then per-channel quantize.
            let (q, _) = raw_f16(weights, &p("self_attn.linear_q.weight"))?;
            let (k, _) = raw_f16(weights, &p("self_attn.linear_k.weight"))?;
            let (v, _) = raw_f16(weights, &p("self_attn.linear_v.weight"))?;
            let mut fused = Vec::with_capacity(3 * q.len());
            fused.extend_from_slice(&q);
            fused.extend_from_slice(&k);
            fused.extend_from_slice(&v);
            // q/k/v are each [d, d] with d = d_model = 1280. Fused → [3d, d].
            let d_dim = 1280usize;
            let qkv_i8 = int8_from_f16(fused, 3 * d_dim, d_dim, 1.0, backend)?;
            (
                Some(weight_t_int8(weights, &p("feed_forward1.linear1.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("feed_forward1.linear2.weight"), backend, 0.5)?),
                Some(weight_t_int8(weights, &p("feed_forward2.linear1.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("feed_forward2.linear2.weight"), backend, 0.5)?),
                Some(weight_t_int8_conv1d(weights, &p("conv.pointwise_conv1.weight"), backend)?),
                Some(weight_t_int8_conv1d(weights, &p("conv.pointwise_conv2.weight"), backend)?),
                Some(qkv_i8),
                Some(weight_t_int8(weights, &p("self_attn.linear_pos.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("self_attn.linear_out.weight"), backend, 1.0)?),
            )
        } else {
            (None, None, None, None, None, None, None, None, None)
        };

        Ok(Self {
            ffn1_norm_w, ffn1_norm_b, ffn1_l1_w_t, ffn1_l1_b, ffn1_l2_w_t, ffn1_l2_b,
            att_norm_w, att_norm_b, qkv_w_t, qkv_b, p_w_t, out_w_t, out_b,
            pos_bias_u, pos_bias_v,
            conv_norm_w, conv_norm_b, cpw1_w_t, cpw1_b, cdw_params, cpw2_w_t, cpw2_b,
            ffn2_norm_w, ffn2_norm_b, ffn2_l1_w_t, ffn2_l1_b, ffn2_l2_w_t, ffn2_l2_b,
            out_norm_w, out_norm_b,
            ffn1_l1_i8, ffn1_l2_i8, ffn2_l1_i8, ffn2_l2_i8,
            cpw1_i8, cpw2_i8,
            qkv_w_i8, p_w_i8, out_w_i8,
        })
    }
}

impl<B: Backend> EncoderWeights<B> {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        num_layers: usize,
        backend: &B,
        int8: bool,
    ) -> anyhow::Result<Self> {
        let layers = (0..num_layers)
            .map(|i| EncoderLayerWeights::load(weights, i, backend, int8))
            .collect::<Result<Vec<_>, _>>()?;
        let enc_proj_w_t = weight_t(weights, "encoder_decoder_proj.weight", backend)?;
        let enc_proj_b = vec1(weights, "encoder_decoder_proj.bias", backend)?;
        Ok(Self { layers, enc_proj_w_t, enc_proj_b })
    }
}

impl<B: Backend> DecoderLayerWeights<B> {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        layer_idx: usize,
        backend: &B,
        int8: bool,
    ) -> anyhow::Result<Self> {
        let p = |name: &str| format!("transf_decoder._decoder.layers.{layer_idx}.{name}");

        let (self_qkv_w_i8, self_out_w_i8, cross_q_w_i8, cross_out_w_i8,
             ffn_in_w_i8, ffn_out_w_i8) = if int8 {
            // Fused self-QKV INT8: [3d, d] with d = DEC_DIM = 1024.
            let (q, _) = raw_f16(weights, &p("first_sub_layer.query_net.weight"))?;
            let (k, _) = raw_f16(weights, &p("first_sub_layer.key_net.weight"))?;
            let (v, _) = raw_f16(weights, &p("first_sub_layer.value_net.weight"))?;
            let mut fused = Vec::with_capacity(3 * q.len());
            fused.extend_from_slice(&q);
            fused.extend_from_slice(&k);
            fused.extend_from_slice(&v);
            let d = 1024usize;
            let qkv_i8 = int8_from_f16(fused, 3 * d, d, 1.0, backend)?;
            (
                Some(qkv_i8),
                Some(weight_t_int8(weights, &p("first_sub_layer.out_projection.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("second_sub_layer.query_net.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("second_sub_layer.out_projection.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("third_sub_layer.dense_in.weight"), backend, 1.0)?),
                Some(weight_t_int8(weights, &p("third_sub_layer.dense_out.weight"), backend, 1.0)?),
            )
        } else {
            (None, None, None, None, None, None)
        };

        Ok(Self {
            norm1_w: vec1(weights, &p("layer_norm_1.weight"), backend)?,
            norm1_b: vec1(weights, &p("layer_norm_1.bias"), backend)?,
            self_q_w_t: weight_t(weights, &p("first_sub_layer.query_net.weight"), backend)?,
            self_q_b: vec1(weights, &p("first_sub_layer.query_net.bias"), backend)?,
            self_k_w_t: weight_t(weights, &p("first_sub_layer.key_net.weight"), backend)?,
            self_k_b: vec1(weights, &p("first_sub_layer.key_net.bias"), backend)?,
            self_v_w_t: weight_t(weights, &p("first_sub_layer.value_net.weight"), backend)?,
            self_v_b: vec1(weights, &p("first_sub_layer.value_net.bias"), backend)?,
            self_qkv_w_t: fuse_qkv_weight(
                weights,
                &p("first_sub_layer.query_net.weight"),
                &p("first_sub_layer.key_net.weight"),
                &p("first_sub_layer.value_net.weight"),
                backend,
            )?,
            self_qkv_b: fuse_qkv_bias(
                weights,
                &p("first_sub_layer.query_net.bias"),
                &p("first_sub_layer.key_net.bias"),
                &p("first_sub_layer.value_net.bias"),
                backend,
            )?,
            self_out_w_t: weight_t(weights, &p("first_sub_layer.out_projection.weight"), backend)?,
            self_out_b: vec1(weights, &p("first_sub_layer.out_projection.bias"), backend)?,
            norm2_w: vec1(weights, &p("layer_norm_2.weight"), backend)?,
            norm2_b: vec1(weights, &p("layer_norm_2.bias"), backend)?,
            cross_q_w_t: weight_t(weights, &p("second_sub_layer.query_net.weight"), backend)?,
            cross_q_b: vec1(weights, &p("second_sub_layer.query_net.bias"), backend)?,
            cross_k_w_t: weight_t(weights, &p("second_sub_layer.key_net.weight"), backend)?,
            cross_k_b: vec1(weights, &p("second_sub_layer.key_net.bias"), backend)?,
            cross_v_w_t: weight_t(weights, &p("second_sub_layer.value_net.weight"), backend)?,
            cross_v_b: vec1(weights, &p("second_sub_layer.value_net.bias"), backend)?,
            cross_out_w_t: weight_t(weights, &p("second_sub_layer.out_projection.weight"), backend)?,
            cross_out_b: vec1(weights, &p("second_sub_layer.out_projection.bias"), backend)?,
            norm3_w: vec1(weights, &p("layer_norm_3.weight"), backend)?,
            norm3_b: vec1(weights, &p("layer_norm_3.bias"), backend)?,
            ffn_in_w_t: weight_t(weights, &p("third_sub_layer.dense_in.weight"), backend)?,
            ffn_in_b: vec1(weights, &p("third_sub_layer.dense_in.bias"), backend)?,
            ffn_out_w_t: weight_t(weights, &p("third_sub_layer.dense_out.weight"), backend)?,
            ffn_out_b: vec1(weights, &p("third_sub_layer.dense_out.bias"), backend)?,
            self_qkv_w_i8, self_out_w_i8, cross_q_w_i8, cross_out_w_i8,
            ffn_in_w_i8, ffn_out_w_i8,
        })
    }
}

impl<B: Backend> DecoderWeights<B> {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        num_layers: usize,
        backend: &B,
        int8: bool,
    ) -> anyhow::Result<Self> {
        let layers = (0..num_layers)
            .map(|i| DecoderLayerWeights::load(weights, i, backend, int8))
            .collect::<Result<Vec<_>, _>>()?;

        // Token embedding [vocab, hidden] — kept as B::Weight {rows=vocab, cols=hidden}.
        let (te_data, te_shape) = raw_f16(weights, "transf_decoder._embedding.token_embedding.weight")?;
        let (vocab, hidden) = as_2d(&te_shape)?;
        let token_embedding = backend.upload_weight(&te_data, vocab, hidden)?;

        let (pe_data, pe_shape) = raw_f16(weights, "transf_decoder._embedding.position_embedding.pos_enc")?;
        let (max_seq, hidden2) = as_2d(&pe_shape)?;
        assert_eq!(hidden2, hidden, "position/token embedding hidden mismatch");
        let position_embedding = backend.upload_weight(&pe_data, max_seq, hidden)?;

        Ok(Self {
            layers,
            token_embedding,
            position_embedding,
            emb_norm_w: vec1(weights, "transf_decoder._embedding.layer_norm.weight", backend)?,
            emb_norm_b: vec1(weights, "transf_decoder._embedding.layer_norm.bias", backend)?,
            final_norm_w: vec1(weights, "transf_decoder._decoder.final_layer_norm.weight", backend)?,
            final_norm_b: vec1(weights, "transf_decoder._decoder.final_layer_norm.bias", backend)?,
            lm_w_t: weight_t(weights, "log_softmax.mlp.layer0.weight", backend)?,
            lm_b: vec1(weights, "log_softmax.mlp.layer0.bias", backend)?,
        })
    }
}

impl<B: Backend> ModelWeights<B> {
    /// Load all weights from `<model_dir>`, rewrite on host, upload as f16.
    /// `int8`: quantize encoder FFN + attention projection weights to INT8
    /// (DP4A path). Decoder always runs f16 — at M=1 its GEMMs are
    /// memory-bound, so INT8's quant/dequant overhead costs more than DP4A
    /// saves (measured: ~2× slower decode).
    pub fn load(model_dir: &std::path::Path, backend: &B, int8: bool) -> anyhow::Result<Self> {
        let weights = load_weights(model_dir)?;
        let pre_encoder = PreEncoderWeights::load(&weights, backend)?;
        let encoder = EncoderWeights::load(&weights, 48, backend, int8)?;
        let decoder = DecoderWeights::load(&weights, 8, backend, false)?;
        Ok(Self { pre_encoder, encoder, decoder })
    }
}

// ---------------------------------------------------------------------------
// Specialized fusers
// ---------------------------------------------------------------------------

/// Fuse q/k/v 2-D weights `[d,d]` each into one matrix `[3d, d]`.
/// Order matches candle's `cat([q,k,v], 0)` -> `[3d, d]`.
fn fuse_qkv_weight<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    q_name: &str,
    k_name: &str,
    v_name: &str,
    backend: &B,
) -> anyhow::Result<B::Weight> {
    let (q, qshape) = raw_f16(weights, q_name)?;
    let (k, kshape) = raw_f16(weights, k_name)?;
    let (v, vshape) = raw_f16(weights, v_name)?;
    let (qr, qc) = as_2d(&qshape)?;
    let (kr, kc) = as_2d(&kshape)?;
    let (vr, vc) = as_2d(&vshape)?;
    anyhow::ensure!(
        qr == qc && qr == kr && kr == kc && kr == vr && vr == vc,
        "qkv shapes differ: q{:?} k{:?} v{:?}",
        qshape,
        kshape,
        vshape
    );
    let d = qr; // = qc, square
    // [3d, d] row-major: q-block, then k-block, then v-block.
    let mut fused = Vec::with_capacity(3 * d * d);
    fused.extend_from_slice(&q);
    fused.extend_from_slice(&k);
    fused.extend_from_slice(&v);
    backend.upload_weight(&fused, 3 * d, d)
}

/// Fuse q/k/v 1-D biases [d] each into [3d].
fn fuse_qkv_bias<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    q_name: &str,
    k_name: &str,
    v_name: &str,
    backend: &B,
) -> anyhow::Result<B::Buf> {
    let (q, _) = raw_f16(weights, q_name)?;
    let (k, _) = raw_f16(weights, k_name)?;
    let (v, _) = raw_f16(weights, v_name)?;
    anyhow::ensure!(q.len() == k.len() && k.len() == v.len(), "qkv bias len mismatch");
    let mut fused = Vec::with_capacity(3 * q.len());
    fused.extend_from_slice(&q);
    fused.extend_from_slice(&k);
    fused.extend_from_slice(&v);
    backend.upload_f16(&fused)
}

/// A Conv1d pointwise weight stored as `[out, in, 1]` (kernel=1, the trailing
/// 1 is the time dimension). Flatten to `[out, in]` — the trailing dim is degenerate.
fn weight_t_reshape1x<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    name: &str,
    backend: &B,
) -> anyhow::Result<B::Weight> {
    let (data, shape) = raw_f16(weights, name)?;
    match shape.as_slice() {
        [out, _in, 1] => {
            // data is already row-major [out, in] (trailing 1 is degenerate).
            backend.upload_weight(&data, *out, *_in)
        }
        _ => anyhow::bail!("expected [out,in,1] for {name}, got {shape:?}"),
    }
}

/// Fold BN into the depthwise conv and pack to [C, 10]. `prefix` is e.g.
/// `encoder.layers.{i}.conv.`.
fn load_packed_depthwise<B: Backend>(
    weights: &HashMap<String, RawTensor>,
    prefix: &str,
    backend: &B,
) -> anyhow::Result<B::Buf> {
    let (cw, cwshape) = raw_f16(weights, &format!("{prefix}depthwise_conv.weight"))?;
    let (cb, _) = raw_f16(weights, &format!("{prefix}depthwise_conv.bias"))?;
    let (bnw, _) = raw_f16(weights, &format!("{prefix}batch_norm.weight"))?;
    let (bnb, _) = raw_f16(weights, &format!("{prefix}batch_norm.bias"))?;
    let (bnm, _) = raw_f16(weights, &format!("{prefix}batch_norm.running_mean"))?;
    let (bnv, _) = raw_f16(weights, &format!("{prefix}batch_norm.running_var"))?;
    let channels = match cwshape.as_slice() {
        [c, 1, 9] => *c,
        _ => anyhow::bail!("depthwise weight must be [C,1,9], got {cwshape:?}"),
    };
    let (fw, fb) = fold_bn_into_depthwise_conv1d(&cw, &cb, &bnw, &bnb, &bnm, &bnv, 1e-5, channels, 9);
    let packed = pack_depthwise_conv1d_params(&fw, &fb, channels);
    backend.upload_f16(&packed)
}
