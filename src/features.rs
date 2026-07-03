use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::MmapOptions;
use rustfft::{FftPlanner, num_complex::Complex32};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;

type PreprocessorBuffers = (Option<Vec<f32>>, Option<Vec<f32>>);

#[derive(Debug, Clone)]
pub struct FeatureConfig {
    pub sample_rate: usize,
    pub feature_size: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub normalize: String,
    pub dither: f32,
    pub pad_to: usize,
    pub preemph: f32,
    pub frontend_window: Option<Vec<f32>>,
    pub frontend_mel: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct FeatureOutput {
    pub data: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
    pub length: usize,
}

impl FeatureOutput {
    pub fn get(&self, channel: usize, frame: usize) -> f32 {
        self.data[channel * self.frames + frame]
    }
}

#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    config: FeatureConfig,
    window_padded: Vec<f32>,
    mel: Vec<f32>,
}

impl FeatureConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let (frontend_window, frontend_mel) = load_preprocessor_buffers(model_dir)?;
        Ok(Self {
            sample_rate: get_usize(&config, "sampling_rate", 16_000),
            feature_size: get_usize(&config, "feature_size", 128),
            n_fft: get_usize(&config, "n_fft", 512),
            win_length: get_usize(&config, "n_window_size", 400),
            hop_length: get_usize(&config, "n_window_stride", 160),
            normalize: config
                .get("normalize")
                .and_then(Value::as_str)
                .unwrap_or("per_feature")
                .to_string(),
            dither: config.get("dither").and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            pad_to: get_usize(&config, "pad_to", 0),
            preemph: 0.97,
            frontend_window,
            frontend_mel,
        })
    }

    pub fn sequence_len(&self, samples: usize) -> usize {
        samples / self.hop_length
    }
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Result<Self> {
        if config.n_fft == 0 || config.win_length == 0 || config.hop_length == 0 {
            bail!("n_fft, win_length and hop_length must be > 0");
        }
        if config.win_length > config.n_fft {
            bail!("win_length must be <= n_fft");
        }
        if config.normalize != "per_feature" {
            bail!("only per_feature normalization is supported");
        }

        let mut window_padded = vec![0.0; config.n_fft];
        let left = (config.n_fft - config.win_length) / 2;
        if let Some(window) = &config.frontend_window {
            if window.len() != config.win_length {
                bail!(
                    "checkpoint frontend window len {} does not match win_length {}",
                    window.len(),
                    config.win_length
                );
            }
            window_padded[left..left + config.win_length].copy_from_slice(window);
        } else {
            for i in 0..config.win_length {
                let value = 0.5
                    - 0.5
                        * ((2.0 * std::f32::consts::PI * i as f32)
                            / (config.win_length - 1) as f32)
                            .cos();
                window_padded[left + i] = value;
            }
        }

        let mel = if let Some(mel) = &config.frontend_mel {
            let expected = config.feature_size * (config.n_fft / 2 + 1);
            if mel.len() != expected {
                bail!(
                    "checkpoint frontend mel len {} does not match expected {}",
                    mel.len(),
                    expected
                );
            }
            mel.clone()
        } else {
            build_mel_filterbank(
                config.sample_rate,
                config.n_fft,
                config.feature_size,
                0.0,
                config.sample_rate as f32 / 2.0,
            )
        };
        Ok(Self {
            config,
            window_padded,
            mel,
        })
    }

    pub fn extract(&self, samples: &[f32]) -> Result<FeatureOutput> {
        let valid_len = samples.len();
        let length = self.config.sequence_len(valid_len);

        let mut x = samples.to_vec();
        // Python adds deterministic 1e-5 dither from torch RNG. Skipping it keeps
        // native extraction deterministic and has negligible ASR impact; exact RNG
        // parity can be added once the full native path is wired.
        apply_preemphasis(&mut x, self.config.preemph);

        let pad = self.config.n_fft / 2;
        let mut centered = vec![0.0; pad + x.len() + pad];
        centered[pad..pad + x.len()].copy_from_slice(&x);
        let frames = if centered.len() < self.config.n_fft {
            0
        } else {
            (centered.len() - self.config.n_fft) / self.config.hop_length + 1
        };

        let bins = self.config.n_fft / 2 + 1;
        let mut spectrum = vec![0.0_f32; bins * frames];
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.config.n_fft);
        let mut buffer = vec![Complex32::new(0.0, 0.0); self.config.n_fft];
        for frame in 0..frames {
            let offset = frame * self.config.hop_length;
            for i in 0..self.config.n_fft {
                buffer[i] = Complex32::new(centered[offset + i] * self.window_padded[i], 0.0);
            }
            fft.process(&mut buffer);
            for bin in 0..bins {
                let value = buffer[bin];
                spectrum[bin * frames + frame] = value.re * value.re + value.im * value.im;
            }
        }

        let mut data = vec![0.0_f32; self.config.feature_size * frames];
        for mel_idx in 0..self.config.feature_size {
            for frame in 0..frames {
                let mut value = 0.0;
                for bin in 0..bins {
                    value += self.mel[mel_idx * bins + bin] * spectrum[bin * frames + frame];
                }
                data[mel_idx * frames + frame] = (value + LOG_ZERO_GUARD).ln();
            }
        }

        normalize_per_feature(&mut data, self.config.feature_size, frames, length);
        for channel in 0..self.config.feature_size {
            for frame in length..frames {
                data[channel * frames + frame] = 0.0;
            }
        }
        if self.config.pad_to > 0 {
            bail!("pad_to > 0 is not implemented in native Rust feature extraction yet");
        }

        Ok(FeatureOutput {
            data,
            channels: self.config.feature_size,
            frames,
            length,
        })
    }
}

fn get_usize(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or(default as u64) as usize
}

fn load_preprocessor_buffers(model_dir: &Path) -> Result<PreprocessorBuffers> {
    let path = model_dir.join("model.safetensors");
    if !path.exists() {
        return Ok((None, None));
    }
    let file =
        std::fs::File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    // The checkpoint is multi-GB; mmap lets us copy only the two tiny frontend buffers.
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to mmap {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let window = match tensors.tensor("preprocessor.featurizer.window") {
        Ok(tensor) => Some(tensor_to_f32_vec(tensor)?),
        Err(_) => None,
    };
    let mel = match tensors.tensor("preprocessor.featurizer.fb") {
        Ok(tensor) => Some(tensor_to_f32_vec(tensor)?),
        Err(_) => None,
    };
    Ok((window, mel))
}

fn tensor_to_f32_vec(tensor: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    match tensor.dtype() {
        Dtype::BF16 => Ok(tensor
            .data()
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
                f32::from_bits(bits << 16)
            })
            .collect()),
        Dtype::F32 => Ok(tensor
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect()),
        dtype => bail!("unsupported frontend tensor dtype {dtype:?}"),
    }
}

fn apply_preemphasis(samples: &mut [f32], preemph: f32) {
    if samples.len() < 2 {
        return;
    }
    for i in (1..samples.len()).rev() {
        samples[i] -= preemph * samples[i - 1];
    }
}

fn normalize_per_feature(data: &mut [f32], channels: usize, frames: usize, length: usize) {
    if length == 0 {
        return;
    }
    for channel in 0..channels {
        let row = &mut data[channel * frames..(channel + 1) * frames];
        let mean = row[..length].iter().sum::<f32>() / length as f32;
        let denom = (length as f32 - 1.0).max(1.0);
        let var = row[..length]
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f32>()
            / denom;
        let std = var.sqrt() + NORMALIZE_EPS;
        for value in row.iter_mut() {
            *value = (*value - mean) / std;
        }
    }
}

fn build_mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let fft_freqs = (0..bins)
        .map(|bin| bin as f32 * sample_rate as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points = (0..n_mels + 2)
        .map(|idx| mel_min + (mel_max - mel_min) * idx as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz)
        .collect::<Vec<_>>();

    let mut weights = vec![0.0; n_mels * bins];
    for mel_idx in 0..n_mels {
        let lower = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let upper = mel_points[mel_idx + 2];
        let enorm = 2.0 / (upper - lower);
        for (bin, freq) in fft_freqs.iter().copied().enumerate() {
            let left = (freq - lower) / (center - lower);
            let right = (upper - freq) / (upper - center);
            weights[mel_idx * bins + bin] = left.min(right).max(0.0) * enorm;
        }
    }
    weights
}

fn hz_to_mel(freq: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * f_sp
    }
}
