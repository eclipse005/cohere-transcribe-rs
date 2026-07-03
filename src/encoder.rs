//! 48-layer macaron Conformer encoder. Stage 3.3: full encoder forward with
//! fused data-layout kernels (zero CPU round-trips in the hot path).
//!
//! The Conformer layer (macaron structure):
//!   FFN1 (half-step) → Self-Attention (rel-pos) → Conv (GLU+depthwise) → FFN2 → LN
//!
//! Generic over `B: Backend` — the same forward runs on CUDA or CPU. CUDA is
//! the RTFx-tuned path; CPU is the correctness/portability path.

use anyhow::Result;
use half::f16;

use crate::backend::{Backend, Int8Weight};
use crate::weights_gpu::{EncoderLayerWeights, EncoderWeights};

const D_MODEL: usize = 1280;
const HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / HEADS; // 160
const FFN_EXPAND: usize = 5120; // 4 * D_MODEL
const EPS: f32 = 1e-5;

static mut ENC_ACC: [f64; 4] = [0.0; 4]; // [ffn1, attn, conv, ffn2] accumulated

/// A single Conformer encoder layer with backend-resident weights.
pub struct EncoderLayer<'a, B: Backend> {
    pub w: &'a EncoderLayerWeights<B>,
}

impl<'a, B: Backend> EncoderLayer<'a, B> {
    // ------------------------------------------------------------------
    // FFN half-step: LN → l1 → SiLU → l2(0.5 baked) → bias+residual
    // ------------------------------------------------------------------
    fn ffn_forward(
        &self,
        backend: &B,
        x: &B::Buf,
        tokens: usize,
        norm_w: &B::Buf,
        norm_b: &B::Buf,
        l1_w_t: &B::Weight,
        l1_b: &B::Buf,
        l2_w_t: &B::Weight,
        l2_b: &B::Buf,
    ) -> Result<B::Buf> {
        let normed = backend.layer_norm(x, norm_w, norm_b, tokens, D_MODEL, EPS)?;
        let h = backend.linear(&normed, tokens, l1_w_t)?;
        let h_silu = backend.silu_bias(&h, l1_b, tokens * FFN_EXPAND, FFN_EXPAND)?;
        let out = backend.linear(&h_silu, tokens, l2_w_t)?;
        backend.bias_residual(&out, l2_b, x, tokens * D_MODEL, D_MODEL)
    }

    /// INT8 DP4A FFN: same math as ffn_forward but both GEMMs via the backend's
    /// INT8 path (quantize → int8 GEMM → dequant), wrapped in `linear_int8`.
    fn ffn_forward_int8(
        &self,
        backend: &B,
        x: &B::Buf,
        tokens: usize,
        norm_w: &B::Buf,
        norm_b: &B::Buf,
        l1: &Int8Weight<B>,
        l1_b: &B::Buf,
        l2: &Int8Weight<B>,
        l2_b: &B::Buf,
    ) -> Result<B::Buf> {
        let normed = backend.layer_norm(x, norm_w, norm_b, tokens, D_MODEL, EPS)?;
        // l1: [tokens, D_MODEL] @ [FFN, D_MODEL]^T -> [tokens, FFN] (INT8)
        let h = backend.linear_int8(&normed, l1, tokens)?;
        let h_silu = backend.silu_bias(&h, l1_b, tokens * FFN_EXPAND, FFN_EXPAND)?;
        // l2: [tokens, FFN] @ [D_MODEL, FFN]^T -> [tokens, D_MODEL] (INT8)
        let out = backend.linear_int8(&h_silu, l2, tokens)?;
        backend.bias_residual(&out, l2_b, x, tokens * D_MODEL, D_MODEL)
    }

    fn attention_forward(
        &self,
        backend: &B,
        x: &B::Buf,
        pos: &B::Buf,
        tokens: usize,
    ) -> Result<B::Buf> {
        // 1. LayerNorm
        let normed = backend.layer_norm(x, &self.w.att_norm_w, &self.w.att_norm_b, tokens, D_MODEL, EPS)?;

        // 2. QKV projection + bias: [tokens, 3*D_MODEL]. INT8 when available.
        let mut qkv = match self.w.qkv_w_i8.as_ref() {
            Some(w) => backend.linear_int8(&normed, w, tokens)?,
            None => backend.linear(&normed, tokens, &self.w.qkv_w_t)?,
        };
        backend.add_bias_inplace(&mut qkv, &self.w.qkv_b, tokens * 3 * D_MODEL, 3 * D_MODEL)?;

        // 3. pos_proj = pos @ p_w_t  → [pos_len, D_MODEL]
        //    `pos` is the full relative-position table [pos_len=2*tokens-1, D_MODEL]
        //    (built by Encoder::generate_position_encoding). The projection must
        //    cover ALL pos_len rows so that the position-content score `bd` is
        //    [heads, tokens, pos_len] — matching candle's
        //    `linear_pos(pos) → pos_split_transpose → [heads, pos_len, head_dim]`
        //    and `matrix_bd = q_v @ pos_proj^T → [heads, tokens, pos_len]`.
        //    (Previously this used `tokens` as the row count, truncating pos_proj
        //    to [tokens, D_MODEL] and making bd [heads, tokens, tokens], which
        //    fed the rel_shift with the wrong shape and corrupted relative-pos
        //    attention → per-token greedy drift / word substitutions.)
        let pos_len = 2 * tokens - 1;
        let pos_proj = match self.w.p_w_i8.as_ref() {
            Some(w) => backend.linear_int8(pos, w, pos_len)?,
            None => backend.linear(pos, pos_len, &self.w.p_w_t)?,
        };

        // 4. Fused split + head reshape + pos bias → q_u, q_v, k, v [heads, tokens, head_dim]
        let (q_u, q_v, k, v) = backend.split_qkv_heads_bias(
            &qkv, &self.w.pos_bias_u, &self.w.pos_bias_v,
            tokens, HEADS, HEAD_DIM,
        )?;

        // 5. pos_proj reshape to [heads, pos_len, head_dim] (full pos_len rows).
        let pos_proj_heads = backend.split_to_heads(&pos_proj, pos_len, HEADS, HEAD_DIM)?;

        // 6. Attention scores: ac = q_u @ k^T  [heads, tokens, tokens],
        //    bd = q_v @ pos_proj^T [heads, tokens, pos_len]. alpha=1.0 (scale
        //    applied later in fused_attn_scores_softmax).
        let ac = backend.attention_qk(
            &q_u, &k, HEADS, tokens, tokens, HEAD_DIM, tokens, 1.0,
        )?;
        let bd = backend.attention_qk(
            &q_v, &pos_proj_heads, HEADS, tokens, pos_len, HEAD_DIM, pos_len, 1.0,
        )?;

        // 7. Fused rel_shift(bd) + ac, scale, softmax → [heads, tokens, tokens].
        //    k_len=tokens (the output/`ac` width); the shift reads bd's pos_len
        //    columns internally (pos_len = 2*k_len-1, asserted inside).
        let attn = backend.fused_attn_scores_softmax(
            &ac, &bd, HEADS, tokens, tokens, (HEAD_DIM as f32).powf(-0.5),
        )?;

        // 8. Attend: attn @ V over [heads, tokens, head_dim]
        let context = backend.attention_av(
            &attn, &v, HEADS, tokens, tokens, HEAD_DIM, tokens,
        )?;

        // 9. Merge heads: [heads, tokens, head_dim] → [tokens, D_MODEL]
        let merged = backend.merge_heads(&context, tokens, HEADS, HEAD_DIM)?;

        // 10. Output projection + bias + residual. INT8 when available.
        let out = match self.w.out_w_i8.as_ref() {
            Some(w) => backend.linear_int8(&merged, w, tokens)?,
            None => backend.linear(&merged, tokens, &self.w.out_w_t)?,
        };
        backend.bias_residual(&out, &self.w.out_b, x, tokens * D_MODEL, D_MODEL)
    }

    // ------------------------------------------------------------------
    // Conv module: LN → pointwise1 → GLU+depthwise k9+SiLU → pointwise2 → residual
    // ------------------------------------------------------------------
    fn conv_forward(
        &self,
        backend: &B,
        x: &B::Buf,
        tokens: usize,
    ) -> Result<B::Buf> {
        let normed = backend.layer_norm(x, &self.w.conv_norm_w, &self.w.conv_norm_b, tokens, D_MODEL, EPS)?;

        // pointwise1: [tokens, D_MODEL] → [tokens, 2*D_MODEL]
        let conv_in = backend.linear(&normed, tokens, &self.w.cpw1_w_t)?;

        // GLU + depthwise conv k9 + SiLU (fused kernel)
        let conv_mid = backend.glu_depthwise_conv(
            &conv_in,
            &self.w.cpw1_b,
            &self.w.cdw_params,
            tokens,
            D_MODEL,
        )?;

        // pointwise2: [tokens, D_MODEL] → [tokens, D_MODEL]
        let conv_out = backend.linear(&conv_mid, tokens, &self.w.cpw2_w_t)?;

        backend.bias_residual(&conv_out, &self.w.cpw2_b, x, tokens * D_MODEL, D_MODEL)
    }

    /// INT8 conv module: pointwise1 (int8) → GLU+depthwise → pointwise2 (int8).
    fn conv_forward_int8(
        &self,
        backend: &B,
        x: &B::Buf,
        tokens: usize,
        cpw1: &Int8Weight<B>,
        cpw2: &Int8Weight<B>,
    ) -> Result<B::Buf> {
        let normed = backend.layer_norm(x, &self.w.conv_norm_w, &self.w.conv_norm_b, tokens, D_MODEL, EPS)?;
        // pointwise1: [tokens, D] @ [2D, D]^T -> [tokens, 2D] (INT8)
        let conv_in = backend.linear_int8(&normed, cpw1, tokens)?;
        // GLU + depthwise k9 + SiLU (f16)
        let conv_mid = backend.glu_depthwise_conv(&conv_in, &self.w.cpw1_b, &self.w.cdw_params, tokens, D_MODEL)?;
        // pointwise2: [tokens, D] @ [D, D]^T -> [tokens, D] (INT8)
        let out = backend.linear_int8(&conv_mid, cpw2, tokens)?;
        backend.bias_residual(&out, &self.w.cpw2_b, x, tokens * D_MODEL, D_MODEL)
    }

    // ------------------------------------------------------------------
    // Full layer forward
    // ------------------------------------------------------------------
    pub fn forward(
        &self,
        backend: &B,
        x: &B::Buf,
        pos: &B::Buf,
        tokens: usize,
    ) -> Result<B::Buf> {
        let prof = std::env::var("ENC_PROF").is_ok();
        let mut acc = [0f64; 4];

        macro_rules! phase {
            ($i:expr, $body:expr) => {{
                let t0 = std::time::Instant::now();
                let r = $body;
                if prof { backend.synchronize().ok(); acc[$i] += t0.elapsed().as_secs_f64(); }
                r
            }};
        }

        let x = phase!(0, match (self.w.ffn1_l1_i8.as_ref(), self.w.ffn1_l2_i8.as_ref()) {
            (Some(l1), Some(l2)) => self.ffn_forward_int8(
                backend, x, tokens,
                &self.w.ffn1_norm_w, &self.w.ffn1_norm_b,
                l1, &self.w.ffn1_l1_b, l2, &self.w.ffn1_l2_b,
            )?,
            _ => self.ffn_forward(
                backend, x, tokens,
                &self.w.ffn1_norm_w, &self.w.ffn1_norm_b,
                &self.w.ffn1_l1_w_t, &self.w.ffn1_l1_b,
                &self.w.ffn1_l2_w_t, &self.w.ffn1_l2_b,
            )?,
        });

        let x = phase!(1, self.attention_forward(backend, &x, pos, tokens)?);

        let x = phase!(2, match (self.w.cpw1_i8.as_ref(), self.w.cpw2_i8.as_ref()) {
            (Some(c1), Some(c2)) => self.conv_forward_int8(backend, &x, tokens, c1, c2)?,
            _ => self.conv_forward(backend, &x, tokens)?,
        });

        let x = phase!(3, match (self.w.ffn2_l1_i8.as_ref(), self.w.ffn2_l2_i8.as_ref()) {
            (Some(l1), Some(l2)) => self.ffn_forward_int8(
                backend, &x, tokens,
                &self.w.ffn2_norm_w, &self.w.ffn2_norm_b,
                l1, &self.w.ffn2_l1_b, l2, &self.w.ffn2_l2_b,
            )?,
            _ => self.ffn_forward(
                backend, &x, tokens,
                &self.w.ffn2_norm_w, &self.w.ffn2_norm_b,
                &self.w.ffn2_l1_w_t, &self.w.ffn2_l1_b,
                &self.w.ffn2_l2_w_t, &self.w.ffn2_l2_b,
            )?,
        });

        let r = backend.layer_norm(&x, &self.w.out_norm_w, &self.w.out_norm_b, tokens, D_MODEL, EPS);
        if prof {
            backend.synchronize().ok();
            unsafe { ENC_ACC[0] += acc[0]; ENC_ACC[1] += acc[1]; ENC_ACC[2] += acc[2]; ENC_ACC[3] += acc[3]; }
        }
        r
    }
}

// ---------------------------------------------------------------------------
// Full 48-layer encoder
// ---------------------------------------------------------------------------
pub struct Encoder<'a, B: Backend> {
    pub layers: Vec<EncoderLayer<'a, B>>,
    /// Encoder → decoder projection: [1024, 1280] weight-transposed + bias [1024].
    pub enc_proj_w_t: &'a B::Weight,
    pub enc_proj_b: &'a B::Buf,
}

impl<'a, B: Backend> Encoder<'a, B> {
    pub fn new(weights: &'a EncoderWeights<B>) -> Self {
        let layers: Vec<EncoderLayer<'a, B>> = weights.layers.iter().map(|w| EncoderLayer { w }).collect();
        Self {
            layers,
            enc_proj_w_t: &weights.enc_proj_w_t,
            enc_proj_b: &weights.enc_proj_b,
        }
    }

    /// Generate sinusoidal position encodings for `tokens` positions.
    /// Returns [pos_len=2*tokens-1, D_MODEL] f16 on the backend.
    ///
    /// Kept on the CPU (then uploaded) rather than a device kernel: the GPU
    /// sinf/cosf/expf approximations differ slightly from the host f32 path,
    /// and that sub-ULP drift is enough to shift attention at sensitive
    /// tokens. Since this runs once per chunk (a few hundred K sin/cos ≈ <1ms),
    /// there is no RTFx reason to move it off the host.
    pub fn generate_position_encoding(
        backend: &B,
        tokens: usize,
    ) -> Result<B::Buf> {
        let pos_len = 2 * tokens - 1; // relative position encoding needs full range
        let mut values = vec![0f32; pos_len * D_MODEL];
        for idx in 0..pos_len {
            let position = (tokens as isize - 1 - idx as isize) as f32;
            for dim in (0..D_MODEL).step_by(2) {
                let div = (dim as f32 * -(10000.0f32.ln() / D_MODEL as f32)).exp();
                values[idx * D_MODEL + dim] = (position * div).sin();
                values[idx * D_MODEL + dim + 1] = (position * div).cos();
            }
        }
        let pos_f16: Vec<f16> = values.iter().map(|v| f16::from_f32(*v)).collect();
        backend.upload_f16(&pos_f16)
    }

    /// Run all 48 encoder layers on the input.
    /// `x`: [tokens, D_MODEL] f16 on backend.
    /// `pos`: [pos_len, D_MODEL] f16 position encodings (pos_len = 2*tokens-1).
    /// Returns encoder output [tokens, D_MODEL] (before enc_dec_proj).
    pub fn forward(
        &self,
        backend: &B,
        x: &B::Buf,
        pos: &B::Buf,
        tokens: usize,
    ) -> Result<B::Buf> {
        // First layer: input is x (borrowed)
        let mut out = self.layers[0].forward(backend, x, pos, tokens)?;

        // Remaining layers
        for layer in &self.layers[1..] {
            out = layer.forward(backend, &out, pos, tokens)?;
        }

        if std::env::var("ENC_PROF").is_ok() {
            backend.synchronize().ok();
            unsafe {
                let t = ENC_ACC;
                eprintln!("[enc] ffn1={:.1}ms attn={:.1}ms conv={:.1}ms ffn2={:.1}ms (x48)",
                    t[0]*1e3, t[1]*1e3, t[2]*1e3, t[3]*1e3);
                ENC_ACC = [0.0; 4];
            }
        }
        Ok(out)
    }
}
