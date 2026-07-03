use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::audio::{AudioChunk, build_prompt, load_wav_mono, split_audio_chunks_energy_meta};
use crate::features::{FeatureConfig, FeatureExtractor, FeatureOutput};
use crate::tokenizer::CohereTokenizer;

#[derive(Debug, Clone)]
pub struct NativePrepareOptions {
    pub model_dir: PathBuf,
    pub audio: PathBuf,
    pub language: String,
    pub punctuation: bool,
}

#[derive(Debug, Clone)]
pub struct NativePreparedAudio {
    pub audio: PathBuf,
    pub sample_rate: usize,
    pub audio_samples: usize,
    pub audio_duration_s: f64,
    pub prompt: String,
    pub prompt_ids: Vec<u32>,
    pub chunks: Vec<NativePreparedChunk>,
}

#[derive(Debug, Clone)]
pub struct NativePreparedChunk {
    pub index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub duration_s: f64,
    pub features: FeatureOutput,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativePrepareReport {
    pub audio: String,
    pub sample_rate: usize,
    pub audio_samples: usize,
    pub audio_duration_s: f64,
    pub prompt: String,
    pub prompt_ids: Vec<u32>,
    pub chunks: Vec<NativePrepareChunkReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativePrepareChunkReport {
    pub index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub duration_s: f64,
    pub feature_channels: usize,
    pub feature_frames: usize,
    pub feature_length: usize,
    pub feature_checksum: f64,
    pub feature_first: f32,
}

impl NativePreparedChunk {
    pub fn feature_checksum(&self) -> f64 {
        self.features
            .data
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
    }
}

impl NativePreparedAudio {
    pub fn report(&self) -> NativePrepareReport {
        NativePrepareReport {
            audio: self.audio.display().to_string(),
            sample_rate: self.sample_rate,
            audio_samples: self.audio_samples,
            audio_duration_s: self.audio_duration_s,
            prompt: self.prompt.clone(),
            prompt_ids: self.prompt_ids.clone(),
            chunks: self
                .chunks
                .iter()
                .map(|chunk| NativePrepareChunkReport {
                    index: chunk.index,
                    start_sample: chunk.start_sample,
                    end_sample: chunk.end_sample,
                    duration_s: chunk.duration_s,
                    feature_channels: chunk.features.channels,
                    feature_frames: chunk.features.frames,
                    feature_length: chunk.features.length,
                    feature_checksum: chunk.feature_checksum(),
                    feature_first: chunk.features.data.first().copied().unwrap_or(0.0),
                })
                .collect(),
        }
    }
}

pub fn prepare_native_audio(options: &NativePrepareOptions) -> Result<NativePreparedAudio> {
    let feature_config = FeatureConfig::from_model_dir(&options.model_dir)?;
    let sample_rate = feature_config.sample_rate;
    let split_config = SplitConfig::from_model_dir(&options.model_dir)?;
    let (samples, loaded_sample_rate) = load_wav_mono(&options.audio, sample_rate)?;
    let chunks_meta = split_audio_chunks_energy_meta(
        &samples,
        loaded_sample_rate,
        split_config.max_audio_clip_s,
        split_config.overlap_chunk_second,
        split_config.min_energy_window_samples,
    )?;
    let extractor = FeatureExtractor::new(feature_config)?;
    let tokenizer = CohereTokenizer::from_model_dir(&options.model_dir)?;
    let prompt = build_prompt(&options.language, options.punctuation);
    let prompt_ids = tokenizer.encode(&prompt, false)?;

    let mut chunks = Vec::with_capacity(chunks_meta.len());
    for (index, (start_sample, end_sample)) in chunks_meta.into_iter().enumerate() {
        let features = extractor.extract(&samples[start_sample..end_sample])?;
        chunks.push(NativePreparedChunk {
            index,
            start_sample,
            end_sample,
            duration_s: (end_sample.saturating_sub(start_sample)) as f64
                / loaded_sample_rate as f64,
            features,
        });
    }

    Ok(NativePreparedAudio {
        audio: options.audio.clone(),
        sample_rate: loaded_sample_rate,
        audio_samples: samples.len(),
        audio_duration_s: samples.len() as f64 / loaded_sample_rate as f64,
        prompt,
        prompt_ids,
        chunks,
    })
}

pub fn prepare_native_audio_report(options: &NativePrepareOptions) -> Result<NativePrepareReport> {
    Ok(prepare_native_audio(options)?.report())
}

#[derive(Debug, Clone)]
struct SplitConfig {
    max_audio_clip_s: f32,
    overlap_chunk_second: f32,
    min_energy_window_samples: usize,
}

impl SplitConfig {
    fn from_model_dir(model_dir: &std::path::Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Self {
            max_audio_clip_s: config
                .get("max_audio_clip_s")
                .and_then(Value::as_f64)
                .unwrap_or(35.0) as f32,
            overlap_chunk_second: config
                .get("overlap_chunk_second")
                .and_then(Value::as_f64)
                .unwrap_or(5.0) as f32,
            min_energy_window_samples: config
                .get("min_energy_window_samples")
                .and_then(Value::as_u64)
                .unwrap_or(1600) as usize,
        })
    }
}

impl From<&NativePreparedChunk> for AudioChunk {
    fn from(value: &NativePreparedChunk) -> Self {
        Self {
            index: value.index,
            start_sample: value.start_sample,
            end_sample: value.end_sample,
            duration_s: value.duration_s,
        }
    }
}
