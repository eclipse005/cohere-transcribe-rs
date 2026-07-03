use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AudioChunk {
    pub index: usize,
    pub start_sample: usize,
    pub end_sample: usize,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioInfo {
    pub path: String,
    pub sample_rate: usize,
    pub samples: usize,
    pub duration_s: f64,
    pub mean: f64,
    pub std: f64,
    pub rms: f64,
    pub min: f32,
    pub max: f32,
    pub chunks: Vec<AudioChunk>,
}

pub fn build_prompt(language: &str, punctuation: bool) -> String {
    let pnc = if punctuation { "<|pnc|>" } else { "<|nopnc|>" };
    let task = "<|noitn|>";
    format!(
        "<|startofcontext|><|startoftranscript|><|emo:undefined|><|{language}|><|{language}|>{pnc}{task}<|notimestamp|><|nodiarize|>"
    )
}

pub fn join_chunk_texts(texts: &[String], language: &str) -> String {
    let pieces = texts
        .iter()
        .map(|text| text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if matches!(language, "zh" | "ja" | "ko") {
        pieces.join("")
    } else {
        pieces.join(" ")
    }
}

pub fn load_wav_mono(path: &Path, target_sr: usize) -> Result<(Vec<f32>, usize)> {
    if !is_wav_path(path) {
        bail!("only WAV input is supported, got: {}", path.display());
    }

    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav: {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("invalid wav channels=0: {}", path.display());
    }

    let mono = match spec.sample_format {
        hound::SampleFormat::Float => {
            let all = reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read float wav samples")?;
            downmix(&all, channels)
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits <= 16 {
                let all = reader
                    .samples::<i16>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("failed to read int16 wav samples")?;
                let as_f32 = all
                    .into_iter()
                    .map(|value| (value as f32) / 32768.0)
                    .collect::<Vec<_>>();
                downmix(&as_f32, channels)
            } else {
                let all = reader
                    .samples::<i32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("failed to read int32 wav samples")?;
                let max_amp = (1_i64 << (bits - 1)) as f64;
                let as_f32 = all
                    .into_iter()
                    .map(|value| ((value as f64) / max_amp) as f32)
                    .collect::<Vec<_>>();
                downmix(&as_f32, channels)
            }
        }
    };

    let src_sr = spec.sample_rate as usize;
    if src_sr == target_sr {
        Ok((mono, target_sr))
    } else {
        Ok((resample_linear(&mono, src_sr, target_sr), target_sr))
    }
}

fn is_wav_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
}

pub fn inspect_wav(
    path: &Path,
    target_sr: usize,
    max_audio_clip_s: f32,
    overlap_chunk_second: f32,
    min_energy_window_samples: usize,
) -> Result<AudioInfo> {
    let (samples, sample_rate) = load_wav_mono(path, target_sr)?;
    let chunks_meta = split_audio_chunks_energy_meta(
        &samples,
        sample_rate,
        max_audio_clip_s,
        overlap_chunk_second,
        min_energy_window_samples,
    )?;
    let chunks = chunks_meta
        .into_iter()
        .enumerate()
        .map(|(index, (start_sample, end_sample))| AudioChunk {
            index,
            start_sample,
            end_sample,
            duration_s: (end_sample.saturating_sub(start_sample)) as f64 / sample_rate as f64,
        })
        .collect::<Vec<_>>();
    let (mean, std, rms, min, max) = stats(&samples);
    Ok(AudioInfo {
        path: path.display().to_string(),
        sample_rate,
        samples: samples.len(),
        duration_s: samples.len() as f64 / sample_rate as f64,
        mean,
        std,
        rms,
        min,
        max,
        chunks,
    })
}

pub fn split_audio_chunks_energy_meta(
    waveform: &[f32],
    sample_rate: usize,
    max_audio_clip_s: f32,
    overlap_chunk_second: f32,
    min_energy_window_samples: usize,
) -> Result<Vec<(usize, usize)>> {
    if waveform.is_empty() {
        return Ok(vec![(0, 0)]);
    }
    if sample_rate == 0 {
        bail!("sample_rate must be > 0");
    }
    if min_energy_window_samples == 0 {
        bail!("min_energy_window_samples must be > 0");
    }

    let chunk_size = ((max_audio_clip_s * sample_rate as f32).round() as usize).max(1);
    let boundary_context_size =
        ((overlap_chunk_second * sample_rate as f32).round() as usize).max(1);
    let total_samples = waveform.len();
    if total_samples <= chunk_size {
        return Ok(vec![(0, total_samples)]);
    }

    let mut chunks_meta = Vec::new();
    let mut idx = 0usize;
    while idx < total_samples {
        if idx + chunk_size >= total_samples {
            chunks_meta.push((idx, total_samples));
            break;
        }
        let search_start = idx.max(idx + chunk_size - boundary_context_size);
        let search_end = (idx + chunk_size).min(total_samples);
        let mut split_point = if search_end <= search_start {
            idx + chunk_size
        } else {
            find_split_point_energy(
                waveform,
                search_start,
                search_end,
                min_energy_window_samples,
            )
        };
        split_point = split_point.max(idx + 1).min(total_samples);
        chunks_meta.push((idx, split_point));
        idx = split_point;
    }
    Ok(chunks_meta)
}

pub fn stats(samples: &[f32]) -> (f64, f64, f64, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|value| *value as f64).sum::<f64>() / n;
    let mut var = 0.0;
    let mut sq = 0.0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in samples {
        let x = *value as f64;
        var += (x - mean) * (x - mean);
        sq += x * x;
        min = min.min(*value);
        max = max.max(*value);
    }
    (mean, (var / n).sqrt(), (sq / n).sqrt(), min, max)
}

fn downmix(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], src_sr: usize, dst_sr: usize) -> Vec<f32> {
    if samples.is_empty() || src_sr == 0 || dst_sr == 0 || src_sr == dst_sr {
        return samples.to_vec();
    }
    let out_len =
        ((samples.len() as u128 * dst_sr as u128 + src_sr as u128 / 2) / src_sr as u128) as usize;
    let ratio = src_sr as f64 / dst_sr as f64;
    (0..out_len)
        .map(|index| {
            let pos = index as f64 * ratio;
            let left = pos.floor() as usize;
            let frac = (pos - left as f64) as f32;
            let a = samples.get(left).copied().unwrap_or(0.0);
            let b = samples.get(left + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

fn find_split_point_energy(
    waveform: &[f32],
    start_idx: usize,
    end_idx: usize,
    min_energy_window_samples: usize,
) -> usize {
    let start_idx = start_idx.min(waveform.len());
    let end_idx = end_idx.min(waveform.len());
    if end_idx <= start_idx {
        return start_idx;
    }
    let window = min_energy_window_samples.max(1);
    if end_idx - start_idx <= window {
        return (start_idx + end_idx) / 2;
    }

    let mut best_idx = start_idx;
    let mut best_energy = f64::INFINITY;
    let upper = end_idx - start_idx - window;
    for offset in (0..upper).step_by(window) {
        let idx = start_idx + offset;
        let energy = waveform[idx..idx + window]
            .iter()
            .map(|value| {
                let value = *value as f64;
                value * value
            })
            .sum::<f64>()
            / window as f64;
        let energy = energy.sqrt();
        if energy < best_energy {
            best_energy = energy;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_prompt_tags() {
        assert_eq!(
            build_prompt("en", true),
            "<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"
        );
        assert!(build_prompt("ja", false).contains("<|nopnc|>"));
    }

    #[test]
    fn joins_cjk_without_spaces() {
        assert_eq!(
            join_chunk_texts(&["你好".to_string(), "世界".to_string()], "zh"),
            "你好世界"
        );
        assert_eq!(
            join_chunk_texts(&["hello".to_string(), "world".to_string()], "en"),
            "hello world"
        );
    }

    #[test]
    fn splits_short_audio_as_one_chunk() {
        let chunks =
            split_audio_chunks_energy_meta(&vec![0.0; 1000], 16_000, 35.0, 5.0, 1600).unwrap();
        assert_eq!(chunks, vec![(0, 1000)]);
    }

    #[test]
    fn split_energy_matches_python_window_start() {
        let mut samples = vec![0.5; 12];
        samples[4..8].fill(0.0);
        assert_eq!(find_split_point_energy(&samples, 0, samples.len(), 4), 4);
    }

    #[test]
    fn rejects_non_wav_extension() {
        let error = load_wav_mono(Path::new("input.mp3"), 16_000).unwrap_err();
        assert!(error.to_string().contains("only WAV input is supported"));
    }

    #[test]
    fn loads_wav_as_target_rate_mono() {
        let path = std::env::temp_dir().join(format!(
            "cohere_transcribe_native_audio_resample_test_{}_{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for index in 0..4_800 {
                let value = ((index as f32 * 0.01).sin() * i16::MAX as f32) as i16;
                writer.write_sample(value).unwrap();
                writer.write_sample(value).unwrap();
            }
            writer.finalize().unwrap();
        }

        let (samples, sample_rate) = load_wav_mono(&path, 16_000).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(sample_rate, 16_000);
        assert!(samples.len() > 1_500);
        assert!(samples.len() < 1_700);
    }
}
