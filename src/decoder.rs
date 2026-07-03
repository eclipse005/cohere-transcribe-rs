//! 8-layer transformer encoder-decoder + greedy token loop.
//!
//! Decoder architecture:
//!   Self-Attn (causal) → Cross-Attn (to encoder) → FFN (ReLU)
//! Hidden dim: 1024, heads: 8, head_dim: 128, FFN expand: 4096.
//!
//! Generic over `B: Backend` — the same forward runs on CUDA or CPU. The KV
//! cache, prefill, and single-token step paths all use the unified
//! `attention_qk`/`attention_av` trait API.

use anyhow::Result;

use crate::backend::{Backend, Int8Weight};
use crate::weights_gpu::{DecoderLayerWeights, DecoderWeights};

const DEC_DIM: usize = 1024;
const DEC_HEADS: usize = 8;
const DEC_HEAD_DIM: usize = DEC_DIM / DEC_HEADS; // 128
const DEC_FFN: usize = 4096; // 4 * DEC_DIM
const DEC_VOCAB: usize = 16384;
const EPS: f32 = 1e-5;

/// Pick the INT8 linear path when an INT8 weight is available, else the f16
/// GEMM. `x` is the [m, in_dim] activation, `f16_w` the f16 fallback. Returns
/// the [m, out_dim] f16 result.
///
/// Currently unused on the decoder M=1 path — INT8's per-tensor quant/dequant
/// overhead (3 extra kernel launches per linear) outweighs DP4A's gain at
/// batch=1, where the GEMM is already memory-bound. Kept for future use on
/// the prefill (M=prompt_len) and batched decode paths, where M is large
/// enough for DP4A to win.
#[allow(dead_code)]
fn lin_or_i8<B: Backend>(
    backend: &B,
    x: &B::Buf,
    m: usize,
    i8_w: Option<&Int8Weight<B>>,
    f16_w: &B::Weight,
) -> Result<B::Buf> {
    match i8_w {
        Some(w) => backend.linear_int8(x, w, m),
        None => backend.linear(x, m, f16_w),
    }
}

/// A single decoder layer.
pub struct DecoderLayer<'a, B: Backend> {
    pub w: &'a DecoderLayerWeights<B>,
}

impl<'a, B: Backend> DecoderLayer<'a, B> {
    /// Scaled dot-product attention (Q from input, K/V from context).
    /// Fully on-device — no CPU round-trips. `causal`: mask upper triangle.
    /// Batch=1 over heads (4-D semantics folded into the flat trait API).
    fn attention(
        &self,
        backend: &B,
        input: &B::Buf,  // [tokens, DEC_DIM]
        context: &B::Buf, // [ctx_tokens, DEC_DIM]
        q_w_t: &B::Weight,
        q_b: &B::Buf,
        k_w_t: &B::Weight,
        k_b: &B::Buf,
        v_w_t: &B::Weight,
        v_b: &B::Buf,
        out_w_t: &B::Weight,
        out_b: &B::Buf,
        residual: &B::Buf,
        causal: bool,
        tokens: usize,
        ctx_tokens: usize,
    ) -> Result<B::Buf> {
        // Q = input @ q_w + q_b, reshape to [heads, tokens, head_dim]
        let mut q = backend.linear(input, tokens, q_w_t)?;
        backend.add_bias_inplace(&mut q, q_b, tokens * DEC_DIM, DEC_DIM)?;
        let q_heads = backend.split_to_heads(&q, tokens, DEC_HEADS, DEC_HEAD_DIM)?;

        // K = context @ k_w + k_b, reshape to [heads, ctx_tokens, head_dim]
        let mut k = backend.linear(context, ctx_tokens, k_w_t)?;
        backend.add_bias_inplace(&mut k, k_b, ctx_tokens * DEC_DIM, DEC_DIM)?;
        let k_heads = backend.split_to_heads(&k, ctx_tokens, DEC_HEADS, DEC_HEAD_DIM)?;

        // V = context @ v_w + v_b, reshape to [heads, ctx_tokens, head_dim]
        let mut v = backend.linear(context, ctx_tokens, v_w_t)?;
        backend.add_bias_inplace(&mut v, v_b, ctx_tokens * DEC_DIM, DEC_DIM)?;
        let v_heads = backend.split_to_heads(&v, ctx_tokens, DEC_HEADS, DEC_HEAD_DIM)?;

        // Scores = Q @ K^T → [heads, tokens, ctx_tokens] (alpha=1.0; scale
        // applied below via scale_inplace or folded into causal_softmax).
        let scores = backend.attention_qk(
            &q_heads, &k_heads, DEC_HEADS, tokens, ctx_tokens, DEC_HEAD_DIM, ctx_tokens, 1.0,
        )?;

        // Softmax (causal or standard)
        let scale = (DEC_HEAD_DIM as f32).powf(-0.5);
        let attn = if causal {
            backend.causal_softmax(&scores, DEC_HEADS, tokens, scale)?
        } else {
            let mut s = scores;
            backend.scale_inplace(&mut s, DEC_HEADS * tokens * ctx_tokens, scale)?;
            backend.softmax_last_dim(&s, DEC_HEADS * tokens, ctx_tokens)?
        };

        // Attend: attn @ V → [heads, tokens, head_dim]
        let ctx = backend.attention_av(
            &attn, &v_heads, DEC_HEADS, tokens, ctx_tokens, DEC_HEAD_DIM, ctx_tokens,
        )?;

        // Merge heads: [heads, tokens, head_dim] → [tokens, DEC_DIM]
        let merged = backend.merge_heads(&ctx, tokens, DEC_HEADS, DEC_HEAD_DIM)?;

        // Output projection + bias + residual
        let out = backend.linear(&merged, tokens, out_w_t)?;
        backend.bias_residual(&out, out_b, residual, tokens * DEC_DIM, DEC_DIM)
    }

    /// Full decoder layer forward (batched, non-cached — used for parity tests).
    pub fn forward(
        &self,
        backend: &B,
        x: &B::Buf,        // [tokens, DEC_DIM]
        encoder_states: &B::Buf, // [enc_tokens, DEC_DIM] (after proj)
        tokens: usize,
        enc_tokens: usize,
    ) -> Result<B::Buf> {
        // 1. Self-attention (causal)
        let normed = backend.layer_norm(x, &self.w.norm1_w, &self.w.norm1_b, tokens, DEC_DIM, EPS)?;
        let x = self.attention(
            backend, &normed, &normed, // input = context for self-attention
            &self.w.self_q_w_t, &self.w.self_q_b,
            &self.w.self_k_w_t, &self.w.self_k_b,
            &self.w.self_v_w_t, &self.w.self_v_b,
            &self.w.self_out_w_t, &self.w.self_out_b,
            x, // residual
            true, tokens, tokens,
        )?;

        // 2. Cross-attention (to encoder states)
        let normed = backend.layer_norm(&x, &self.w.norm2_w, &self.w.norm2_b, tokens, DEC_DIM, EPS)?;
        let x = self.attention(
            backend, &normed, encoder_states,
            &self.w.cross_q_w_t, &self.w.cross_q_b,
            &self.w.cross_k_w_t, &self.w.cross_k_b,
            &self.w.cross_v_w_t, &self.w.cross_v_b,
            &self.w.cross_out_w_t, &self.w.cross_out_b,
            &x,
            false, tokens, enc_tokens,
        )?;

        // 3. FFN (ReLU)
        let normed = backend.layer_norm(&x, &self.w.norm3_w, &self.w.norm3_b, tokens, DEC_DIM, EPS)?;
        let hidden = backend.linear(&normed, tokens, &self.w.ffn_in_w_t)?;
        let hidden_relu = backend.relu_bias(&hidden, &self.w.ffn_in_b, tokens * DEC_FFN, DEC_FFN)?;
        let out = backend.linear(&hidden_relu, tokens, &self.w.ffn_out_w_t)?;
        backend.bias_residual(&out, &self.w.ffn_out_b, &x, tokens * DEC_DIM, DEC_DIM)
    }
}

// ---------------------------------------------------------------------------
// Full decoder
// ---------------------------------------------------------------------------

pub struct Decoder<'a, B: Backend> {
    pub layers: Vec<DecoderLayer<'a, B>>,
    pub w: &'a DecoderWeights<B>,
}

impl<'a, B: Backend> Decoder<'a, B> {
    pub fn new(weights: &'a DecoderWeights<B>) -> Self {
        let layers: Vec<DecoderLayer<'a, B>> = weights.layers.iter().map(|w| DecoderLayer { w }).collect();
        Self { layers, w: weights }
    }

    /// Run all 8 decoder layers.
    pub fn forward(
        &self,
        backend: &B,
        x: &B::Buf,        // [tokens, DEC_DIM]
        encoder_states: &B::Buf, // [enc_tokens, DEC_DIM]
        tokens: usize,
        enc_tokens: usize,
    ) -> Result<B::Buf> {
        let mut out = self.layers[0].forward(backend, x, encoder_states, tokens, enc_tokens)?;
        for layer in &self.layers[1..] {
            out = layer.forward(backend, &out, encoder_states, tokens, enc_tokens)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// KV-cached incremental decoding (Stage 4.1 — O(n) decode)
// ---------------------------------------------------------------------------

/// Per-layer KV cache buffers for incremental decoding.
pub struct DecoderKvCache<B: Backend> {
    /// Self-attention K/V: [DEC_HEADS, max_seq, DEC_HEAD_DIM] per layer.
    pub self_k: Vec<B::Buf>,
    pub self_v: Vec<B::Buf>,
    /// Cross-attention K/V: [DEC_HEADS, enc_tokens, DEC_HEAD_DIM] per layer (constant).
    pub cross_k: Vec<B::Buf>,
    pub cross_v: Vec<B::Buf>,
    pub max_seq: usize,
    pub enc_tokens: usize,
}

impl<B: Backend> DecoderKvCache<B> {
    pub fn new(backend: &B, num_layers: usize, max_seq: usize, enc_tokens: usize) -> Result<Self> {
        let self_elt = || backend.alloc_uninit(DEC_HEADS * max_seq * DEC_HEAD_DIM);
        let cross_elt = || backend.alloc_uninit(DEC_HEADS * enc_tokens * DEC_HEAD_DIM);
        Ok(Self {
            self_k: (0..num_layers).map(|_| self_elt()).collect::<Result<Vec<_>>>()?,
            self_v: (0..num_layers).map(|_| self_elt()).collect::<Result<Vec<_>>>()?,
            cross_k: (0..num_layers).map(|_| cross_elt()).collect::<Result<Vec<_>>>()?,
            cross_v: (0..num_layers).map(|_| cross_elt()).collect::<Result<Vec<_>>>()?,
            max_seq,
            enc_tokens,
        })
    }
}

impl<'a, B: Backend> DecoderLayer<'a, B> {
    /// Process ONE token through a layer using the KV cache.
    /// `x`: [1, DEC_DIM]. Writes new self K/V into cache at row `pos`.
    /// `max_seq`: allocated seq dim of the self cache buffers.
    /// Returns [1, DEC_DIM].
    pub fn forward_step(
        &self,
        backend: &B,
        x: &B::Buf,
        self_k_cache: &mut B::Buf,
        self_v_cache: &mut B::Buf,
        cross_k: &B::Buf,
        cross_v: &B::Buf,
        pos: usize,
        max_seq: usize,
        enc_tokens: usize,
    ) -> Result<B::Buf> {
        let scale = (DEC_HEAD_DIM as f32).powf(-0.5);

        // 1. Self-attention — fused QKV (one GEMM) + split/scatter (one kernel)
        let normed = backend.layer_norm(x, &self.w.norm1_w, &self.w.norm1_b, 1, DEC_DIM, EPS)?;
        let qkv = backend.linear(&normed, 1, &self.w.self_qkv_w_t)?; // [1, 3*DEC_DIM]
        let q_heads = backend.split_qkv_step_cached(
            &qkv, &self.w.self_qkv_b, self_k_cache, self_v_cache,
            pos, max_seq, DEC_HEADS, DEC_HEAD_DIM,
        )?;
        let k_seq = pos + 1;

        // scores = Q @ cached_K^T → [heads, 1, k_seq] (alpha folds in scale)
        let scores = backend.attention_qk(
            &q_heads, self_k_cache, DEC_HEADS, 1, k_seq, DEC_HEAD_DIM, max_seq, scale,
        )?;
        let attn = backend.softmax_last_dim(&scores, DEC_HEADS, k_seq)?;

        let ctx = backend.attention_av(
            &attn, self_v_cache, DEC_HEADS, 1, k_seq, DEC_HEAD_DIM, max_seq,
        )?;
        let merged = backend.merge_heads_single(&ctx, 1, DEC_HEADS, DEC_HEAD_DIM)?;
        let out = backend.linear(&merged, 1, &self.w.self_out_w_t)?;
        let x = backend.bias_residual(&out, &self.w.self_out_b, x, DEC_DIM, DEC_DIM)?;

        // 2. Cross-attention (constant cross K/V)
        let normed = backend.layer_norm(&x, &self.w.norm2_w, &self.w.norm2_b, 1, DEC_DIM, EPS)?;
        let mut cq = backend.linear(&normed, 1, &self.w.cross_q_w_t)?;
        backend.add_bias_inplace(&mut cq, &self.w.cross_q_b, DEC_DIM, DEC_DIM)?;
        let cq_heads = backend.split_to_heads(&cq, 1, DEC_HEADS, DEC_HEAD_DIM)?;

        let cscores = backend.attention_qk(
            &cq_heads, cross_k, DEC_HEADS, 1, enc_tokens, DEC_HEAD_DIM, enc_tokens, scale,
        )?;
        let cattn = backend.softmax_last_dim(&cscores, DEC_HEADS, enc_tokens)?;

        let cctx = backend.attention_av(
            &cattn, cross_v, DEC_HEADS, 1, enc_tokens, DEC_HEAD_DIM, enc_tokens,
        )?;
        let cmerged = backend.merge_heads_single(&cctx, 1, DEC_HEADS, DEC_HEAD_DIM)?;
        let cout = backend.linear(&cmerged, 1, &self.w.cross_out_w_t)?;
        let x = backend.bias_residual(&cout, &self.w.cross_out_b, &x, DEC_DIM, DEC_DIM)?;

        // 3. FFN (ReLU) on single token
        let normed = backend.layer_norm(&x, &self.w.norm3_w, &self.w.norm3_b, 1, DEC_DIM, EPS)?;
        let hidden = backend.linear(&normed, 1, &self.w.ffn_in_w_t)?;
        let hidden_relu = backend.relu_bias(&hidden, &self.w.ffn_in_b, DEC_FFN, DEC_FFN)?;
        let ffn_out = backend.linear(&hidden_relu, 1, &self.w.ffn_out_w_t)?;
        backend.bias_residual(&ffn_out, &self.w.ffn_out_b, &x, DEC_DIM, DEC_DIM)
    }

    /// Prefill: process the whole `seq`-token prompt in one batched forward,
    /// while scattering the self-attention K/V into the cache so subsequent
    /// `forward_step` calls read the correct history. One GEMM per projection
    /// (M=seq) instead of `seq` separate M=1 calls.
    /// `x`: [seq, DEC_DIM]. Returns [seq, DEC_DIM].
    pub fn forward_prefill(
        &self,
        backend: &B,
        x: &B::Buf,
        self_k_cache: &mut B::Buf,
        self_v_cache: &mut B::Buf,
        cross_k: &B::Buf,
        cross_v: &B::Buf,
        seq: usize,
        max_seq: usize,
        enc_tokens: usize,
    ) -> Result<B::Buf> {
        let scale = (DEC_HEAD_DIM as f32).powf(-0.5);

        // 1. Self-attention — fused QKV (one GEMM, M=seq) + split/scatter into cache.
        let normed = backend.layer_norm(x, &self.w.norm1_w, &self.w.norm1_b, seq, DEC_DIM, EPS)?;
        let qkv = backend.linear(&normed, seq, &self.w.self_qkv_w_t)?; // [seq, 3*DEC_DIM]
        let q_heads = backend.split_qkv_batch_scatter(
            &qkv, &self.w.self_qkv_b, self_k_cache, self_v_cache,
            seq, max_seq, DEC_HEADS, DEC_HEAD_DIM,
        )?; // [heads, seq, head_dim]

        // Q @ K^T over the filled cache rows [0..seq) → [heads, seq, seq].
        // attention_qk reads K from [heads, max_seq, head_dim] with only
        // the first `seq` rows valid — exactly the prefill case.
        // alpha=1.0: the scale is applied once by `causal_softmax` below (it is
        // a fused scale + causal-mask + softmax). Passing scale here too would
        // double-scale the scores (scale^2 ≈ 7.8e-3), collapsing softmax and
        // corrupting the prefill — matching the (non-cached) `attention` helper
        // and candle's `fused_scaled_softmax_last_dim(scores, scale, causal)`.
        let scores = backend.attention_qk(
            &q_heads, self_k_cache, DEC_HEADS, seq, seq, DEC_HEAD_DIM, max_seq, 1.0,
        )?; // [heads, seq, seq]
        let attn = backend.causal_softmax(&scores, DEC_HEADS, seq, scale)?;
        let ctx = backend.attention_av(
            &attn, self_v_cache, DEC_HEADS, seq, seq, DEC_HEAD_DIM, max_seq,
        )?; // [heads, seq, head_dim]
        let merged = backend.merge_heads(&ctx, seq, DEC_HEADS, DEC_HEAD_DIM)?;
        let out = backend.linear(&merged, seq, &self.w.self_out_w_t)?;
        let x = backend.bias_residual(&out, &self.w.self_out_b, x, seq * DEC_DIM, DEC_DIM)?;

        // 2. Cross-attention (constant cross K/V from cache). Q over all seq.
        let normed = backend.layer_norm(&x, &self.w.norm2_w, &self.w.norm2_b, seq, DEC_DIM, EPS)?;
        let mut cq = backend.linear(&normed, seq, &self.w.cross_q_w_t)?;
        backend.add_bias_inplace(&mut cq, &self.w.cross_q_b, seq * DEC_DIM, DEC_DIM)?;
        let cq_heads = backend.split_to_heads(&cq, seq, DEC_HEADS, DEC_HEAD_DIM)?;
        // cross_k/v are [heads, enc_tokens, head_dim]; reuse attention_qk
        // with m=seq (it accepts any m). scale folded into alpha.
        let cscores = backend.attention_qk(
            &cq_heads, cross_k, DEC_HEADS, seq, enc_tokens, DEC_HEAD_DIM, enc_tokens, scale,
        )?;
        let cattn = backend.softmax_last_dim(&cscores, DEC_HEADS * seq, enc_tokens)?;
        let cctx = backend.attention_av(
            &cattn, cross_v, DEC_HEADS, seq, enc_tokens, DEC_HEAD_DIM, enc_tokens,
        )?;
        let cmerged = backend.merge_heads(&cctx, seq, DEC_HEADS, DEC_HEAD_DIM)?;
        let cout = backend.linear(&cmerged, seq, &self.w.cross_out_w_t)?;
        let x = backend.bias_residual(&cout, &self.w.cross_out_b, &x, seq * DEC_DIM, DEC_DIM)?;

        // 3. FFN (ReLU) over all seq tokens.
        let normed = backend.layer_norm(&x, &self.w.norm3_w, &self.w.norm3_b, seq, DEC_DIM, EPS)?;
        let hidden = backend.linear(&normed, seq, &self.w.ffn_in_w_t)?;
        let hidden_relu = backend.relu_bias(&hidden, &self.w.ffn_in_b, seq * DEC_FFN, DEC_FFN)?;
        let ffn_out = backend.linear(&hidden_relu, seq, &self.w.ffn_out_w_t)?;
        backend.bias_residual(&ffn_out, &self.w.ffn_out_b, &x, seq * DEC_DIM, DEC_DIM)
    }
}

impl<'a, B: Backend> Decoder<'a, B> {
    /// Build the cross-attention KV cache (constant across decode steps).
    pub fn build_cross_kv_cache(
        &self,
        backend: &B,
        cache: &mut DecoderKvCache<B>,
        encoder_states: &B::Buf,
    ) -> Result<()> {
        for (li, layer) in self.layers.iter().enumerate() {
            let mut k = backend.linear(encoder_states, cache.enc_tokens, &layer.w.cross_k_w_t)?;
            backend.add_bias_inplace(&mut k, &layer.w.cross_k_b, cache.enc_tokens * DEC_DIM, DEC_DIM)?;
            cache.cross_k[li] = backend.split_to_heads(&k, cache.enc_tokens, DEC_HEADS, DEC_HEAD_DIM)?;

            let mut v = backend.linear(encoder_states, cache.enc_tokens, &layer.w.cross_v_w_t)?;
            backend.add_bias_inplace(&mut v, &layer.w.cross_v_b, cache.enc_tokens * DEC_DIM, DEC_DIM)?;
            cache.cross_v[li] = backend.split_to_heads(&v, cache.enc_tokens, DEC_HEADS, DEC_HEAD_DIM)?;
        }
        Ok(())
    }

    /// Embed a single token (id, position) → [1, DEC_DIM] with emb norm. Fully on-device.
    /// Fused gather+add (one kernel) followed by LN — was 4 launches, now 2.
    pub fn embed_one(
        &self,
        backend: &B,
        token_id: i32,
        pos: usize,
    ) -> Result<B::Buf> {
        let tok_emb = backend.weight_data(&self.w.token_embedding);
        let pos_emb = backend.weight_data(&self.w.position_embedding);
        let emb = backend.embed_gather_add(
            &tok_emb, &pos_emb,
            token_id as usize, pos, DEC_DIM,
        )?;
        backend.layer_norm(&emb, &self.w.emb_norm_w, &self.w.emb_norm_b, 1, DEC_DIM, EPS)
    }

    /// Embed all `seq` prompt tokens in one batched on-device pass: gather +
    /// position add + LN. `ids`: [seq] on host. Returns [seq, DEC_DIM].
    pub fn embed_batch(
        &self,
        backend: &B,
        ids: &[i32],
    ) -> Result<B::Buf> {
        let seq = ids.len();
        let tok_emb = backend.weight_data(&self.w.token_embedding);
        let pos_emb = backend.weight_data(&self.w.position_embedding);
        let emb = backend.embed_batch(
            &tok_emb, &pos_emb,
            ids, seq, DEC_DIM,
        )?;
        backend.layer_norm(&emb, &self.w.emb_norm_w, &self.w.emb_norm_b, seq, DEC_DIM, EPS)
    }

    /// Parallel prefill: run the whole `seq`-token prompt through all decoder
    /// layers in one batched forward (M=seq GEMMs), filling the self-attention
    /// KV cache. Returns the next token id predicted from the last prompt
    /// position. After this, decoding continues at position `seq` via
    /// `decode_step_cached`.
    pub fn prefill(
        &self,
        backend: &B,
        cache: &mut DecoderKvCache<B>,
        prompt_ids: &[i32],
    ) -> Result<i32> {
        let seq = prompt_ids.len();
        let x = self.embed_batch(backend, prompt_ids)?; // [seq, DEC_DIM]

        let mut out = self.layers[0].forward_prefill(
            backend, &x,
            &mut cache.self_k[0], &mut cache.self_v[0],
            &cache.cross_k[0], &cache.cross_v[0],
            seq, cache.max_seq, cache.enc_tokens,
        )?;
        for li in 1..self.layers.len() {
            out = self.layers[li].forward_prefill(
                backend, &out,
                &mut cache.self_k[li], &mut cache.self_v[li],
                &cache.cross_k[li], &cache.cross_v[li],
                seq, cache.max_seq, cache.enc_tokens,
            )?;
        }
        // Final norm + LM head on the last prompt token → first generated token.
        let normed = backend.layer_norm(&out, &self.w.final_norm_w, &self.w.final_norm_b, seq, DEC_DIM, EPS)?;
        let mut logits = backend.linear(&normed, seq, &self.w.lm_w_t)?;
        backend.add_bias_inplace(&mut logits, &self.w.lm_b, seq * DEC_VOCAB, DEC_VOCAB)?;
        // argmax returns host i32 directly (trait handles any D2H).
        backend.argmax(&logits, (seq - 1) * DEC_VOCAB, DEC_VOCAB)
    }

    /// Decode one token using the KV cache. `x`: [1, DEC_DIM] embedded token at position `pos`.
    /// Returns the next token id (argmax of LM head).
    pub fn decode_step_cached(
        &self,
        backend: &B,
        cache: &mut DecoderKvCache<B>,
        x: &B::Buf,
        pos: usize,
    ) -> Result<i32> {
        let mut out = self.layers[0].forward_step(
            backend, x,
            &mut cache.self_k[0], &mut cache.self_v[0],
            &cache.cross_k[0], &cache.cross_v[0],
            pos, cache.max_seq, cache.enc_tokens,
        )?;
        for li in 1..self.layers.len() {
            out = self.layers[li].forward_step(
                backend, &out,
                &mut cache.self_k[li], &mut cache.self_v[li],
                &cache.cross_k[li], &cache.cross_v[li],
                pos, cache.max_seq, cache.enc_tokens,
            )?;
        }
        // Final norm + LM head on single token
        let normed = backend.layer_norm(&out, &self.w.final_norm_w, &self.w.final_norm_b, 1, DEC_DIM, EPS)?;
        let mut logits = backend.linear(&normed, 1, &self.w.lm_w_t)?;
        backend.add_bias_inplace(&mut logits, &self.w.lm_b, DEC_VOCAB, DEC_VOCAB)?;
        // argmax → host i32 (CUDA does a device reduction + small D2H; CPU is a host loop).
        backend.argmax(&logits, 0, DEC_VOCAB)
    }
}
