//! End-to-end transcription orchestration. Stage 5: wire everything together
//! and measure RTFx.
//!
//! The transcriber is generic over `B: Backend`. The CUDA backend wraps the
//! hand-tuned `CudaState` (cuBLAS + NVRTC); the CPU backend runs pure-Rust f32.
//! Both share the exact same encoder/decoder/pre-encoder forward code.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::backend::{Backend, CpuBackend, CpuBackendF16, CudaBackend};
use crate::decoder::{Decoder, DecoderKvCache};
use crate::encoder::Encoder;
use crate::host;
use crate::weights_gpu::ModelWeights;

/// Which device to run inference on. `Auto` picks CUDA if available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
    Cpu,
    Auto,
}

impl Device {
    /// Parse a `--device` string ("cuda" | "cpu" | "auto").
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cuda" => Ok(Device::Cuda),
            "cpu" => Ok(Device::Cpu),
            "auto" => Ok(Device::Auto),
            other => anyhow::bail!("unknown device '{other}': expected cuda|cpu|auto"),
        }
    }
}

/// The two resident-backend variants, each carrying its own weight set. All
/// forward code is generic over `B: Backend`, so the two variants run identical
/// math — only storage (`CudaSlice` vs `Vec`) and the tuned kernels differ.
enum AnyBackend {
    #[cfg(feature = "cuda")]
    Cuda {
        backend: CudaBackend,
        weights: ModelWeights<CudaBackend>,
    },
    Cpu {
        // f32 backend: pre-encoder + encoder + projection (compute-bound M>1 GEMMs).
        backend: CpuBackend,
        weights: ModelWeights<CpuBackend>,
        // f16 backend: decoder only (M=1 GEMVs, bandwidth-bound — f16 is ~1.6×
        // faster and needs no per-op activation conversion). Holds the decoder
        // weights reloaded into f16 storage.
        dec_backend: CpuBackendF16,
        dec_weights: crate::weights_gpu::DecoderWeights<CpuBackendF16>,
    },
}

impl AnyBackend {
    fn device_name(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            AnyBackend::Cuda { backend, .. } => backend.name(),
            AnyBackend::Cpu { backend, .. } => backend.name(),
        }
    }
}

/// Same field set as `cohere_transcribe_rs::TranscribeReport` (the candle
/// baseline) so benchmark tooling can diff them directly.
#[derive(Debug, Serialize)]
pub struct TranscribeReport {
    pub device: String,
    pub dtype: String,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub max_new_tokens: usize,
    pub kv_cache: bool,
    pub batch_size: usize,
    pub encoder_pos_cache_capacity: usize,
    pub audio_duration_s: f64,
    pub chunks: usize,
    pub text: String,
    pub load_s: f64,
    pub prepare_s: f64,
    pub encoder_s: f64,
    pub projection_s: f64,
    pub generate_s: f64,
    pub generate_cross_kv_s: f64,
    pub generate_prefill_s: f64,
    pub generate_token_loop_s: f64,
    pub detok_s: f64,
    pub total_infer_s: f64,
    pub rtfx: f64,
    pub process_cpu_s: f64,
    pub cpu_per_wall: f64,
}

/// Resident transcribe session (model loaded once). Holds either a CUDA or a
/// CPU backend with its weight set; the forward math is identical across both
/// (generic over `B: Backend`).
pub struct Transcriber {
    backend: AnyBackend,
    tokenizer: host::CohereTokenizer,
    model_dir: std::path::PathBuf,
    bos_token: i32,
    eos_token: i32,
    max_new_tokens: usize,
    int8: bool,
}

impl Transcriber {
    /// Load the model onto `device`. `int8`: quantize encoder weights to INT8
    /// (CUDA DP4A only — the CPU backend ignores the flag and runs f16).
    pub fn load_with_device(model_dir: &Path, device: Device, int8: bool) -> Result<Self> {
        let t0 = Instant::now();
        let backend = match device {
            Device::Cuda | Device::Auto => {
                #[cfg(feature = "cuda")]
                {
                    let cuda = CudaBackend::new(0).context("initializing CUDA")?;
                    // INT8 only on CUDA (DP4A); CPU runs f16 regardless of flag.
                    let weights = ModelWeights::load(model_dir, &cuda, int8)
                        .context("loading weights")?;
                    AnyBackend::Cuda { backend: cuda, weights }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = int8;
                    anyhow::bail!("CUDA device requested but this build has no `cuda` feature");
                }
            }
            Device::Cpu => {
                let cpu = CpuBackend::new();
                let weights = ModelWeights::load(model_dir, &cpu, false)?;
                // Reload decoder weights into an f16 backend for the bandwidth-
                // bound M=1 GEMV decode path (encoder keeps the f32 backend).
                let dec_backend = CpuBackendF16::new();
                let raw = crate::weights::load_weights(model_dir)?;
                let dec_weights = crate::weights_gpu::DecoderWeights::load(&raw, 8, &dec_backend, false)?;
                AnyBackend::Cpu { backend: cpu, weights, dec_backend, dec_weights }
            }
        };
        let tokenizer = host::CohereTokenizer::from_model_dir(model_dir)?;
        let bos_token = tokenizer.bos_token_id() as i32;
        let eos_token = tokenizer.eos_token_id() as i32;
        let load_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "model loaded in {load_s:.2}s on {} (bos={bos_token} eos={eos_token})",
            backend.device_name(),
        );

        Ok(Self {
            backend,
            tokenizer,
            model_dir: model_dir.to_path_buf(),
            bos_token,
            eos_token,
            // Match candle's per-language default (`app.rs::default_max_new_tokens_for_language`):
            // 256 for Latin-script langs, 384 for zh/ja/ko. The native default
            // used to be 220, which truncated dense ~30s chunks before EOS and
            // dropped the last ~1.5s of speech per chunk (compounding across a
            // long file). 256 lets each chunk run to its natural EOS like candle.
            max_new_tokens: default_max_new_tokens_for_language("en"),
            int8,
        })
    }

    /// Backwards-compatible loader: defaults to CUDA (the original behaviour).
    pub fn load(model_dir: &Path, int8: bool) -> Result<Self> {
        Self::load_with_device(model_dir, Device::Auto, int8)
    }

    /// Encode + decode one chunk → text. Returns (text, encoder_s, projection_s, gen_s).
    fn transcribe_chunk(
        &self,
        chunk: &host::NativePreparedChunk,
        prompt_ids: &[u32],
    ) -> Result<(String, f64, f64, f64)> {
        match &self.backend {
            #[cfg(feature = "cuda")]
            AnyBackend::Cuda { backend, weights } => transcribe_chunk_on(
                backend, weights, chunk, prompt_ids,
                self.max_new_tokens, self.bos_token, self.eos_token, &self.tokenizer,
            ),
            AnyBackend::Cpu { backend, weights, dec_backend, dec_weights } => {
                // Encoder on the f32 backend, decoder on the f16 backend.
                transcribe_chunk_cpu_dual(
                    backend, weights, dec_backend, dec_weights, chunk, prompt_ids,
                    self.max_new_tokens, self.bos_token, self.eos_token, &self.tokenizer,
                )
            }
        }
    }

    /// Transcribe raw f32 mono samples (single segment, caller-owned
    /// chunking) → text. Reuses the full encode/decode pipeline without
    /// touching `prepare_native_audio`, so the caller keeps full control of
    /// VAD segmentation. `language` is a short code ("en"/"zh"/...) that
    /// becomes the `<|lang|>` prompt tag.
    ///
    /// This mirrors the `transcribe_samples(&[f32], ..)` entry point the
    /// voxtrans ASR layer expects, keeping Cohere's call shape identical to
    /// qwen3-asr's: the host (voxtrans) owns VAD/splitting and feeds one
    /// segment at a time.
    ///
    /// Long audio (> `max_audio_clip_s`, default 35s) is split here into
    /// energy-based chunks — exactly what the official `transformers` processor
    /// does (and what `prepare_native_audio` does in the file path). Each chunk
    /// is transcribed independently and the texts are reassembled with
    /// [`host::join_chunk_texts`]. This makes the method safe for any input
    /// length, mirroring how `transcribe_file_with_lang` handles long audio.
    pub fn transcribe_samples(&self, samples: &[f32], language: &str) -> Result<String> {
        // Split config: read from model config.json (same fields the file path
        // uses via SplitConfig::from_model_dir), with the documented defaults.
        let split_cfg = split_config_from_model_dir(&self.model_dir)?;
        let sample_rate = 16000; // Cohere ASR frontend is 16kHz mono.

        // Energy-based chunking. Returns (start, end) sample ranges.
        let chunk_ranges = host::split_audio_chunks_energy_meta(
            samples,
            sample_rate,
            split_cfg.max_audio_clip_s,
            split_cfg.overlap_chunk_second,
            split_cfg.min_energy_window_samples,
        )?;
        if chunk_ranges.is_empty() {
            anyhow::bail!("audio chunking produced no chunks");
        }

        // Build the feature extractor + prompt once (reused across chunks).
        let feature_config = host::FeatureConfig::from_model_dir(&self.model_dir)?;
        let extractor = host::FeatureExtractor::new(feature_config)?;
        let prompt = host::build_prompt(language, true);
        let prompt_ids = self.tokenizer.encode(&prompt, false)?;

        // Transcribe each chunk independently, collect per-chunk text.
        let mut texts = Vec::with_capacity(chunk_ranges.len());
        for (start, end) in chunk_ranges {
            if end <= start {
                continue;
            }
            let features = extractor.extract(&samples[start..end])?;
            let chunk = host::NativePreparedChunk {
                index: 0,
                start_sample: start,
                end_sample: end,
                duration_s: (end - start) as f64 / sample_rate as f64,
                features,
            };
            let (text, _enc_s, _proj_s, _gen_s) =
                self.transcribe_chunk(&chunk, &prompt_ids)?;
            texts.push(text);
        }

        // Reassemble: CJK chunks join without spaces, Latin with spaces.
        Ok(host::join_chunk_texts(&texts, language))
    }

    pub fn transcribe_file(&self, audio: &Path) -> Result<TranscribeReport> {
        self.transcribe_file_with_lang(audio, "en")
    }

    pub fn transcribe_file_with_lang(&self, audio: &Path, language: &str) -> Result<TranscribeReport> {
        let t_total = Instant::now();

        let t0 = Instant::now();
        let prepared = host::prepare_native_audio(&host::NativePrepareOptions {
            model_dir: self.model_dir.clone(),
            audio: audio.to_path_buf(),
            language: language.to_string(),
            punctuation: true,
        })?;
        let prepare_s = t0.elapsed().as_secs_f64();

        // Process ALL chunks (long audio is split into ~30s chunks).
        let prompt_ids = &prepared.prompt_ids;
        let mut texts = Vec::new();
        let (mut enc_s, mut proj_s, mut gen_s, mut dur_s) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for chunk in &prepared.chunks {
            let (text, e, p, g) = self.transcribe_chunk(chunk, prompt_ids)?;
            texts.push(text);
            enc_s += e; proj_s += p; gen_s += g; dur_s += chunk.duration_s;
        }
        let text = texts.join(" ");
        let n_chunks = prepared.chunks.len();

        let total_infer_s = t_total.elapsed().as_secs_f64();
        let rtfx = if total_infer_s > 0.0 { dur_s / total_infer_s } else { 0.0 };

        Ok(TranscribeReport {
            device: self.backend.device_name().into(),
            dtype: if self.int8 { "int8".into() } else { "f16".into() },
            encoder_layers: 48,
            decoder_layers: 8,
            max_new_tokens: self.max_new_tokens,
            kv_cache: true,
            batch_size: 1,
            encoder_pos_cache_capacity: 0,
            audio_duration_s: dur_s,
            chunks: n_chunks,
            text,
            load_s: 0.0,
            prepare_s,
            encoder_s: enc_s,
            projection_s: proj_s,
            generate_s: gen_s,
            generate_cross_kv_s: 0.0,
            generate_prefill_s: 0.0,
            generate_token_loop_s: 0.0,
            detok_s: 0.0,
            total_infer_s,
            rtfx,
            process_cpu_s: 0.0,
            cpu_per_wall: 0.0,
        })
    }
}

/// Backend-generic chunk transcription: pre-encoder → encoder → projection →
/// greedy decode. Free function so the enum dispatcher can call it for each
/// `B` variant without re-monornomorphizing the orchestration logic.
fn transcribe_chunk_on<B: Backend>(
    backend: &B,
    weights: &ModelWeights<B>,
    chunk: &host::NativePreparedChunk,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    bos_token: i32,
    eos_token: i32,
    tokenizer: &host::CohereTokenizer,
) -> Result<(String, f64, f64, f64)> {
    // Pre-encoder (f16)
    let frames = chunk.features.frames;
    let (x, enc_tokens) = weights.pre_encoder.forward(backend, &chunk.features.data, frames)?;
    let pos = Encoder::generate_position_encoding(backend, enc_tokens)?;

    // Encoder forward (48 layers)
    let t_enc = Instant::now();
    let encoder = Encoder::new(&weights.encoder);
    let enc_out = encoder.forward(backend, &x, &pos, enc_tokens)?;
    backend.synchronize()?;
    let encoder_s = t_enc.elapsed().as_secs_f64();

    // Encoder → decoder projection (with bias)
    let t_proj = Instant::now();
    let mut enc_proj = backend.linear(&enc_out, enc_tokens, &weights.encoder.enc_proj_w_t)?;
    backend.add_bias_inplace(&mut enc_proj, &weights.encoder.enc_proj_b, enc_tokens * 1024, 1024)?;
    backend.synchronize()?;
    let projection_s = t_proj.elapsed().as_secs_f64();

    // Greedy decode (KV-cached)
    let t_gen = Instant::now();
    let text = greedy_decode_on(
        backend, &weights.decoder, &enc_proj, enc_tokens, prompt_ids,
        max_new_tokens, bos_token, eos_token, tokenizer,
    )?;
    backend.synchronize()?;
    let generate_s = t_gen.elapsed().as_secs_f64();

    Ok((text, encoder_s, projection_s, generate_s))
}

/// CPU dual-backend chunk path: pre-encoder + encoder + projection on the f32
/// `CpuBackend` (compute-bound M>1 GEMMs are ~3× faster in f32), then greedy
/// decode on the f16 `CpuBackendF16` (M=1 GEMVs are bandwidth-bound, ~1.6×
/// faster in f16, and f16 storage needs no per-op activation conversion).
/// The encoder projection output is downcast f32→f16 once at the boundary.
#[allow(clippy::too_many_arguments)]
fn transcribe_chunk_cpu_dual(
    enc_backend: &CpuBackend,
    weights: &ModelWeights<CpuBackend>,
    dec_backend: &CpuBackendF16,
    dec_weights: &crate::weights_gpu::DecoderWeights<CpuBackendF16>,
    chunk: &host::NativePreparedChunk,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    bos_token: i32,
    eos_token: i32,
    tokenizer: &host::CohereTokenizer,
) -> Result<(String, f64, f64, f64)> {
    // Pre-encoder + encoder + projection on f32.
    let frames = chunk.features.frames;
    let (x, enc_tokens) = weights.pre_encoder.forward(enc_backend, &chunk.features.data, frames)?;
    let pos = Encoder::generate_position_encoding(enc_backend, enc_tokens)?;

    let t_enc = Instant::now();
    let encoder = Encoder::new(&weights.encoder);
    let enc_out = encoder.forward(enc_backend, &x, &pos, enc_tokens)?;
    enc_backend.synchronize()?;
    let encoder_s = t_enc.elapsed().as_secs_f64();

    let t_proj = Instant::now();
    let mut enc_proj = enc_backend.linear(&enc_out, enc_tokens, &weights.encoder.enc_proj_w_t)?;
    enc_backend.add_bias_inplace(&mut enc_proj, &weights.encoder.enc_proj_b, enc_tokens * 1024, 1024)?;
    enc_backend.synchronize()?;
    let projection_s = t_proj.elapsed().as_secs_f64();

    // Boundary: f32 encoder states → f16 for the decoder backend (one pass).
    let enc_proj_f16: <CpuBackendF16 as Backend>::Buf =
        enc_proj.iter().map(|v| half::f16::from_f32(*v)).collect();

    // Greedy decode on the f16 backend.
    let t_gen = Instant::now();
    let text = greedy_decode_on(
        dec_backend, dec_weights, &enc_proj_f16, enc_tokens, prompt_ids,
        max_new_tokens, bos_token, eos_token, tokenizer,
    )?;
    dec_backend.synchronize()?;
    let generate_s = t_gen.elapsed().as_secs_f64();

    Ok((text, encoder_s, projection_s, generate_s))
}

/// Greedy decode with KV cache (O(n) decode), generic over the backend.
fn greedy_decode_on<B: Backend>(
    backend: &B,
    dec_weights: &crate::weights_gpu::DecoderWeights<B>,
    encoder_states: &B::Buf,
    enc_tokens: usize,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    bos_token: i32,
    eos_token: i32,
    tokenizer: &host::CohereTokenizer,
) -> Result<String> {
    let decoder = Decoder::new(dec_weights);
    let max_seq = max_new_tokens + prompt_ids.len().max(1) + 8;

    let mut cache = DecoderKvCache::new(backend, decoder.layers.len(), max_seq, enc_tokens)?;
    decoder.build_cross_kv_cache(backend, &mut cache, encoder_states)?;

    let mut token_ids: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
    if token_ids.is_empty() || token_ids[0] != bos_token {
        token_ids.insert(0, bos_token);
    }
    let prompt_len = token_ids.len();

    // Parallel prefill: run the whole prompt through the decoder in one
    // batched forward (M=prompt_len GEMMs), filling the self-attention KV
    // cache. Returns the first generated token from the last prompt position.
    let mut next_token = decoder.prefill(backend, &mut cache, &token_ids)?;

    // Autoregressive generation: embed predicted token, predict next.
    let mut pos = prompt_len;
    for _step in 0..max_new_tokens {
        token_ids.push(next_token);
        if next_token == eos_token {
            break;
        }
        let x = decoder.embed_one(backend, next_token, pos)?;
        next_token = decoder.decode_step_cached(backend, &mut cache, &x, pos)?;
        pos += 1;
    }

    let text = tokenizer.decode(&token_ids[1..].iter().map(|&id| id as u32).collect::<Vec<_>>(), true)?;
    Ok(text)
}

/// Chunking parameters read from model `config.json`. Mirrors the
/// `SplitConfig::from_model_dir` used by `prepare_native_audio`, so the
/// samples path and the file path chunk identically.
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
        let config: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Self {
            max_audio_clip_s: config
                .get("max_audio_clip_s")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(35.0) as f32,
            overlap_chunk_second: config
                .get("overlap_chunk_second")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(5.0) as f32,
            min_energy_window_samples: config
                .get("min_energy_window_samples")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1600) as usize,
        })
    }
}

fn split_config_from_model_dir(model_dir: &std::path::Path) -> Result<SplitConfig> {
    SplitConfig::from_model_dir(model_dir)
}

/// Per-language decoder token cap. Mirrors candle's
/// `default_max_new_tokens_for_language` (`src/app.rs`) so the native path
/// gives each chunk the same generation budget (and thus the same natural EOS
/// point) as the candle baseline — not a shorter cap that truncates speech.
fn default_max_new_tokens_for_language(language: &str) -> usize {
    if matches!(language, "zh" | "ja" | "ko") {
        384
    } else {
        256
    }
}
