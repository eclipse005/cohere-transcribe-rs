//! Pre-encoder: the `dw_striding` conv stack that 8×-subsamples log-mel features
//! ([1,128,frames] → [tokens, 1280]). Stage 5.1: hand-written conv stack
//! matching candle's GPU pre-encoder numerical path (fixes the CPU-f32
//! divergence). Generic over `B: Backend` — runs on CUDA or CPU unchanged.

#![cfg(feature = "cuda")]

use anyhow::Result;
use half::f16;

use crate::backend::Backend;
use crate::weights_gpu::PreEncoderWeights;

const MEL_BINS: usize = 128;

impl<B: Backend> PreEncoderWeights<B> {
    /// Run the pre-encoder conv stack (f16 storage). Returns [tokens, 1280] + token count.
    /// `mel`: [128, frames] row-major (mel[f * frames + t]).
    pub fn forward(
        &self,
        backend: &B,
        mel: &[f32],
        frames: usize,
    ) -> Result<(B::Buf, usize)> {
        // Build NCHW [1, 1, frames, 128]: nchw[t*128 + f] = mel[f*frames + t].
        let mut nchw = vec![f16::ZERO; frames * MEL_BINS];
        for t in 0..frames {
            for f in 0..MEL_BINS {
                nchw[t * MEL_BINS + f] = f16::from_f32(mel[f * frames + t]);
            }
        }
        let mut x = backend.upload_f16(&nchw)?;
        let mut h = frames;
        let mut w = MEL_BINS;

        // conv0: standard 3x3 stride2 + ReLU, in=1 → out=256
        let (o, h2, w2) = backend.conv2d3x3_s2_relu(&x, &self.conv0.weight, &self.conv0.bias, 1, 1, 256, h, w)?;
        x = o; h = h2; w = w2;

        // conv2: depthwise 3x3 stride2 (groups=256)
        let (o, h2, w2) = backend.depthwise_conv2d3x3_s2(&x, &self.conv2.weight, &self.conv2.bias, 1, 256, h, w)?;
        x = o; h = h2; w = w2;

        // conv3: pointwise 1x1 + ReLU
        x = backend.pointwise_conv_relu(&x, &self.conv3.weight, &self.conv3.bias, 1, 256, 256, h, w)?;

        // conv5: depthwise 3x3 stride2
        let (o, h2, w2) = backend.depthwise_conv2d3x3_s2(&x, &self.conv5.weight, &self.conv5.bias, 1, 256, h, w)?;
        x = o; h = h2; w = w2;

        // conv6: pointwise 1x1 + ReLU. Now x is NCHW [1, 256, tokens, 16].
        x = backend.pointwise_conv_relu(&x, &self.conv6.weight, &self.conv6.bias, 1, 256, 256, h, w)?;
        let tokens = h;

        // Reshape [1, 256, tokens, 16] → [tokens, 4096], then out-proj → [tokens, 1280].
        let flat = backend.nchw_to_tokens(&x, 256, tokens, 16)?;
        let mut out = backend.linear(&flat, tokens, &self.out_w_t)?;
        backend.add_bias_inplace(&mut out, &self.out_b, tokens * 1280, 1280)?;
        Ok((out, tokens))
    }
}
