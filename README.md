# Cohere ASR — Native Rust Engine

> Cohere ASR 模型的纯 Rust 推理引擎。**手写 CUDA + CPU 双后端**，零深度学习框架依赖，目标 RTFx 远超 Candle 基线。

## 为什么不用 Candle / Burn？

Cohere ASR（也称 Canary-1B）的官方实现基于 PyTorch，社区 Rust 移植版用 [`candle`](https://github.com/huggingface/candle)。在推理场景有几个痛点：

- **conformer encoder 慢**：candle 的通用 conv + attention 对 conformer 结构没有针对性 fusion
- **rel_pos attention 开销大**：candle 每层重新计算 relative position bias，无缓存
- **CPU 单线程瓶颈**：conformer 的 depthwise conv 在 candle 下走单线程
- **内存峰值高**：长音频的 attention scores 分配完整 (seq × seq) 内存

本项目直接用 `cudarc` + `cuBLAS` + `NVRTC` 手写所有 kernel，CPU 路径用 `gemm` + `rayon`。**热路径上没有任何深度学习框架**。

## 特性

- **双后端，单一二进制**：CUDA + CPU 同时编译进同一个库，运行时通过 `--device cuda|cpu` 切换
- **CUDA 路径**：cuBLAS HGEMM + NVRTC 手写 kernel（fused conformer conv、rel_pos attention、RMSNorm）
- **CPU 路径**：gemm + rayon（AVX2/FMA，f16 权重存储 + f32 计算）
- **零拷贝权重加载**：mmap safetensors + `Bytes::from_owner`
- **完整 conformer 支持**：encoder + decoder + CTC + transformer decoder
- **多语言**：支持 Cohere/Canary 模型的多语言转录和翻译

## 快速开始

```rust
use native_transcribe::{Engine, Device};

let engine = Engine::load("path/to/model", Device::Cuda)?;
let result = engine.transcribe("audio.wav")?;
println!("{}", result.text);
```

命令行使用：

```bash
cargo run --release -- --model path/to/model --audio audio.wav --device cuda
```

## Features

```toml
default = ["cuda"]       # CUDA 后端（+ CPU 总是可用）
cuda = ["dep:cudarc"]    # CUDA 后端
```

CPU-only 构建：`cargo build --no-default-features`

## 项目结构

```
src/
├── engine.rs             # 主推理引擎：encoder → decoder → CTC
├── encoder.rs            # Conformer encoder（手写 depthwise conv + attention）
├── decoder.rs            # Transformer decoder（cuBLAS HGEMM + NVRTC kernel）
├── features.rs           # mel spectrogram（CPU, f32）
├── kernels.cu            # 所有 CUDA kernel（NVRTC 运行时编译）
├── kernels.rs            # CUDA kernel host 绑定
├── weights.rs            # safetensors mmap 零拷贝权重加载
├── weights_gpu.rs        # GPU 权重上传
├── raw_tensor.rs         # safetensors 原始字节 view
├── backend/
│   ├── cpu.rs            # CPU 后端（gemm + rayon）
│   ├── cpu_f16.rs        # CPU f16 权重处理
│   └── cuda.rs           # CUDA 后端（cudarc + cuBLAS）
└── tokenizer.rs          # 分词器
```

## License

MIT
