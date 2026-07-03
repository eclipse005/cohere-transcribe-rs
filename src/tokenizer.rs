use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use prost::Message;
use serde_json::Value;

#[derive(Debug)]
pub struct CohereTokenizer {
    pieces: Vec<TokenPiece>,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    special_ids: HashSet<u32>,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    unk_token_id: u32,
}

impl CohereTokenizer {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let sp_path = model_dir.join("tokenizer.model");
        let vocab_path = model_dir.join("vocab.json");
        let config_path = model_dir.join("tokenizer_config.json");

        let pieces = load_unigram_pieces(&sp_path)?;
        let vocab_text = std::fs::read_to_string(&vocab_path)
            .with_context(|| format!("failed to read {}", vocab_path.display()))?;
        let raw_vocab: HashMap<String, String> = serde_json::from_str(&vocab_text)
            .with_context(|| format!("failed to parse {}", vocab_path.display()))?;

        let mut id_to_token = vec![String::new(); raw_vocab.len()];
        let mut token_to_id = HashMap::with_capacity(raw_vocab.len());
        for (id_text, token) in raw_vocab {
            let id = id_text
                .parse::<usize>()
                .with_context(|| format!("invalid vocab id {id_text:?}"))?;
            if id >= id_to_token.len() {
                bail!("vocab id {id} is outside vocab size {}", id_to_token.len());
            }
            id_to_token[id] = token.clone();
            token_to_id.insert(token, id as u32);
        }

        let config_text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config: Value = serde_json::from_str(&config_text)
            .with_context(|| format!("failed to parse {}", config_path.display()))?;
        let special_ids = special_ids_from_config(&config)?;

        let bos_token_id = id_for_token(&token_to_id, "<|startoftranscript|>")?;
        let eos_token_id = id_for_token(&token_to_id, "<|endoftext|>")?;
        let pad_token_id = id_for_token(&token_to_id, "<pad>")?;
        let unk_token_id = id_for_token(&token_to_id, "<unk>")?;

        Ok(Self {
            pieces,
            id_to_token,
            token_to_id,
            special_ids,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let normalized = normalize_sentencepiece_text(text);
        let mut ids = self.encode_normalized(&normalized)?;
        if add_special_tokens {
            ids.insert(0, self.bos_token_id);
            ids.push(self.eos_token_id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let pieces = ids
            .iter()
            .copied()
            .filter(|id| !skip_special_tokens || !self.special_ids.contains(id))
            .map(|id| {
                self.id_to_token(id)
                    .map(str::to_string)
                    .with_context(|| format!("token id {id} outside vocab"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(decode_sentencepiece_pieces(&pieces))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    pub fn byte_fallback_value_for_id(&self, id: u32) -> Option<u8> {
        self.id_to_token(id).and_then(byte_fallback_value)
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    fn encode_normalized(&self, normalized: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut pos = 0usize;
        while pos < normalized.len() {
            let mut matched = None;
            for piece in &self.pieces {
                if piece.is_byte_fallback {
                    continue;
                }
                if normalized[pos..].starts_with(&piece.text) {
                    matched = Some((piece.text.len(), piece.id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                ids.push(id);
                pos += len;
            } else {
                ids.push(self.unk_token_id);
                pos += normalized[pos..].chars().next().unwrap().len_utf8();
            }
        }
        Ok(ids)
    }
}

fn id_for_token(token_to_id: &HashMap<String, u32>, token: &str) -> Result<u32> {
    token_to_id
        .get(token)
        .copied()
        .with_context(|| format!("token {token:?} missing from vocab"))
}

fn special_ids_from_config(config: &Value) -> Result<HashSet<u32>> {
    let mut ids = HashSet::new();
    let decoder = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .with_context(|| "tokenizer_config.json missing added_tokens_decoder")?;

    for (id_text, entry) in decoder {
        if entry
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let id = id_text
                .parse::<u32>()
                .with_context(|| format!("invalid added token id {id_text:?}"))?;
            ids.insert(id);
        }
    }
    Ok(ids)
}

#[derive(Debug, Clone)]
struct TokenPiece {
    text: String,
    id: u32,
    is_byte_fallback: bool,
}

fn load_unigram_pieces(path: &Path) -> Result<Vec<TokenPiece>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let proto = ModelProto::decode(bytes.as_slice())
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let mut pieces = proto
        .pieces
        .into_iter()
        .enumerate()
        .map(|(id, piece)| {
            let text = piece.piece.unwrap_or_default();
            TokenPiece {
                is_byte_fallback: is_byte_fallback_piece(&text),
                text,
                id: id as u32,
            }
        })
        .collect::<Vec<_>>();
    pieces.sort_by(|a, b| {
        b.text
            .len()
            .cmp(&a.text.len())
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(pieces)
}

fn is_byte_fallback_piece(token: &str) -> bool {
    token.len() == 6 && token.starts_with("<0x") && token.ends_with('>')
}

fn normalize_sentencepiece_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 1);
    out.push('▁');
    for ch in text.chars() {
        if ch.is_whitespace() {
            out.push('▁');
        } else {
            out.push(ch);
        }
    }
    out
}

fn decode_sentencepiece_pieces(pieces: &[String]) -> String {
    let mut text = decode_byte_fallback_pieces(pieces).replace('▁', " ");
    if text.starts_with(' ') {
        text.remove(0);
    }
    text
}

fn decode_byte_fallback_pieces(pieces: &[String]) -> String {
    let mut out = String::new();
    let mut bytes = Vec::new();
    for piece in pieces {
        if let Some(byte) = byte_fallback_value(piece) {
            bytes.push(byte);
        } else {
            flush_byte_fallback(&mut out, &mut bytes);
            out.push_str(piece);
        }
    }
    flush_byte_fallback(&mut out, &mut bytes);
    out
}

fn flush_byte_fallback(out: &mut String, bytes: &mut Vec<u8>) {
    if bytes.is_empty() {
        return;
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        out.push_str(text);
    } else if let Err(err) = std::str::from_utf8(bytes) {
        let valid_up_to = err.valid_up_to();
        if valid_up_to > 0 {
            out.push_str(std::str::from_utf8(&bytes[..valid_up_to]).unwrap_or_default());
        }
    }
    bytes.clear();
}

fn byte_fallback_value(token: &str) -> Option<u8> {
    if !is_byte_fallback_piece(token) {
        return None;
    }
    u8::from_str_radix(&token[3..5], 16).ok()
}

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SentencePiece>,
}

#[derive(Clone, PartialEq, Message)]
struct SentencePiece {
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    #[prost(float, optional, tag = "2")]
    score: Option<f32>,
}
