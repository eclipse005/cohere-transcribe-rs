//! Host-side view over a safetensors tensor: a refcounted zero-copy byte
//! slice plus shape/dtype. Conversion to `Vec<f16>` happens on demand at
//! upload time.

use bytes::Bytes;
use safetensors::Dtype;

/// A raw weight tensor backed by an mmap'd safetensors region.
///
/// `data` is an O(1) refcounted slice (`bytes::Bytes`) into the owning mmap;
/// cloning it is cheap and keeps the whole region alive.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    pub fn new(data: Bytes, shape: Vec<usize>, dtype: Dtype) -> Self {
        Self { data, shape, dtype }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Decode this tensor to a flat `Vec<f16>`, regardless of source dtype
    /// (F32 / F16 / BF16). Used right before H2D upload.
    pub fn to_f16_vec(&self) -> anyhow::Result<Vec<half::f16>> {
        match self.dtype {
            Dtype::F16 => {
                // safetensors stores little-endian f16; reinterpret-cast.
                let bytes = self.data.as_ref();
                anyhow::ensure!(
                    bytes.len() == self.numel() * 2,
                    "f16 tensor byte length {} != numel*2 {}",
                    bytes.len(),
                    self.numel() * 2
                );
                let out: Vec<half::f16> = bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                Ok(out)
            }
            Dtype::BF16 => {
                let bytes = self.data.as_ref();
                let out: Vec<half::f16> = bytes
                    .chunks_exact(2)
                    .map(|c| {
                        // bf16 -> f32 (pad low 16 bits zero) -> f16
                        let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
                        let f = f32::from_bits(bits << 16);
                        half::f16::from_f32(f)
                    })
                    .collect();
                Ok(out)
            }
            Dtype::F32 => {
                let bytes = self.data.as_ref();
                let out: Vec<half::f16> = bytes
                    .chunks_exact(4)
                    .map(|c| {
                        let f = f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                        half::f16::from_f32(f)
                    })
                    .collect();
                Ok(out)
            }
            other => anyhow::bail!("unsupported weight dtype for f16 conversion: {other:?}"),
        }
    }

    /// Decode to f32 (used for sanity / parity comparisons against candle).
    pub fn to_f32_vec(&self) -> anyhow::Result<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => {
                let bytes = self.data.as_ref();
                let out: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect();
                Ok(out)
            }
            Dtype::F16 => Ok(self.to_f16_vec()?.iter().map(|h| h.to_f32()).collect()),
            Dtype::BF16 => Ok(self.to_f16_vec()?.iter().map(|h| h.to_f32()).collect()),
            other => anyhow::bail!("unsupported weight dtype for f32 conversion: {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side load-time rewrites (mirror cohere-transcribe-rs app.rs:4702-4826,
// but done in f16 on the host before H2D upload). All operate on Vec<f16> +
// explicit shape so we avoid any tensor library.
// ---------------------------------------------------------------------------

/// Shape helper: row-major 2-D as (rows, cols).
pub fn as_2d(shape: &[usize]) -> anyhow::Result<(usize, usize)> {
    match shape {
        [r, c] => Ok((*r, *c)),
        _ => anyhow::bail!("expected 2-D shape, got {shape:?}"),
    }
}

/// Transpose a row-major 2-D f16 buffer: [rows, cols] -> [cols, rows].
pub fn transpose_f16(data: &[half::f16], rows: usize, cols: usize) -> Vec<half::f16> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![half::f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Scale every element by `s` (e.g. the 0.5 macaron FFN residual bake).
pub fn scale_f16(data: &[half::f16], s: f32) -> Vec<half::f16> {
    data.iter().map(|x| half::f16::from_f32(x.to_f32() * s)).collect()
}

/// Vertical concat of two 2-D f16 buffers sharing the same column count:
/// `[a_rows, cols]` ++ `[b_rows, cols]` -> `[a_rows + b_rows, cols]`.
pub fn vconcat_f16(
    a: &[half::f16],
    a_rows: usize,
    b: &[half::f16],
    b_rows: usize,
    cols: usize,
) -> Vec<half::f16> {
    assert_eq!(a.len(), a_rows * cols);
    assert_eq!(b.len(), b_rows * cols);
    let mut out = Vec::with_capacity((a_rows + b_rows) * cols);
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

/// Broadcast-multiply a 2-D weight by a per-row scale (reserved for future
/// BatchNorm-into-pointwise-conv folds; currently unused but kept for symmetry
/// with the candle reference path).
#[allow(dead_code)]
pub fn broadcast_mul_row_f16(
    weight: &[half::f16],
    rows: usize,
    cols: usize,
    scale: &[half::f16],
) -> Vec<half::f16> {
    assert_eq!(weight.len(), rows * cols);
    assert_eq!(scale.len(), rows);
    let mut out = vec![half::f16::ZERO; rows * cols];
    for r in 0..rows {
        let s = scale[r];
        for c in 0..cols {
            out[r * cols + c] = half::f16::from_f32(weight[r * cols + c].to_f32() * s.to_f32());
        }
    }
    out
}

/// Fold BatchNorm1d running statistics into a depthwise conv1d weight + bias,
/// matching `app.rs::fold_batch_norm_into_conv1d`.
///
/// - conv_weight: [C, 1, 9] (flattened to [C, 9])
/// - conv_bias:   [C]
/// - bn_weight/running_mean/running_var: [C], bn_bias: [C], eps: scalar
/// Returns (folded_weight [C*9], folded_bias [C]).
pub fn fold_bn_into_depthwise_conv1d(
    conv_weight: &[half::f16], // [C, 9]
    conv_bias: &[half::f16],   // [C]
    bn_weight: &[half::f16],   // [C]
    bn_bias: &[half::f16],     // [C]
    running_mean: &[half::f16], // [C]
    running_var: &[half::f16], // [C]
    eps: f32,
    channels: usize,
    kernel: usize,
) -> (Vec<half::f16>, Vec<half::f16>) {
    assert_eq!(conv_weight.len(), channels * kernel);
    assert_eq!(conv_bias.len(), channels);
    let mut fw = vec![half::f16::ZERO; channels * kernel];
    let mut fb = vec![half::f16::ZERO; channels];
    for c in 0..channels {
        let scale = bn_weight[c].to_f32() / (running_var[c].to_f32() + eps).sqrt();
        for k in 0..kernel {
            fw[c * kernel + k] =
                half::f16::from_f32(conv_weight[c * kernel + k].to_f32() * scale);
        }
        fb[c] = half::f16::from_f32(
            (conv_bias[c].to_f32() - running_mean[c].to_f32()) * scale + bn_bias[c].to_f32(),
        );
    }
    (fw, fb)
}

/// Pack a folded depthwise conv weight `[C, 9]` and bias `[C]` into the
/// `[C, 10]` row-major layout the fused GLU+depthwise kernel consumes
/// (mirrors `app.rs::pack_depthwise_conv1d_params`).
pub fn pack_depthwise_conv1d_params(
    weight: &[half::f16], // [C, 9]
    bias: &[half::f16],   // [C]
    channels: usize,
) -> Vec<half::f16> {
    let kernel = 9usize;
    assert_eq!(weight.len(), channels * kernel);
    assert_eq!(bias.len(), channels);
    let mut out = vec![half::f16::ZERO; channels * (kernel + 1)];
    for c in 0..channels {
        out[c * (kernel + 1)..c * (kernel + 1) + kernel]
            .copy_from_slice(&weight[c * kernel..c * kernel + kernel]);
        out[c * (kernel + 1) + kernel] = bias[c];
    }
    out
}
