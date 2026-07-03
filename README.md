# cohere-transcribe-rs

[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) 的 Rust 推理引擎。支持 CUDA 和 CPU 双后端，零深度学习框架依赖。

Cohere Transcribe 是 Cohere Labs 开源的 2B 参数语音识别模型，基于 Conformer encoder + Transformer decoder 架构，支持 14 种语言（英/法/德/意/西/葡/希腊/荷兰/波兰/中/日/韩/越南/阿拉伯）。

## 安装

```toml
[dependencies]
native-transcribe = { git = "https://github.com/eclipse005/cohere-transcribe-rs.git" }
```

CPU-only 构建：

```toml
native-transcribe = { git = "https://github.com/eclipse005/cohere-transcribe-rs.git", default-features = false }
```

## 使用

### 作为库

```rust
use native_transcribe::{Engine, Device};

let engine = Engine::load("path/to/model", Device::Cuda)?;
let result = engine.transcribe("audio.wav")?;
println!("{}", result.text);
```

### 命令行

```bash
cargo run --release -- --model path/to/model --audio audio.wav --device cuda
```

## 模型下载

从 HuggingFace 下载：
- [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)

## Features

| Feature | 说明 |
|---------|------|
| `cuda`（默认） | CUDA 后端，需要 CUDA 12.8+ |
| — | CPU 后端始终可用 |

## License

MIT
