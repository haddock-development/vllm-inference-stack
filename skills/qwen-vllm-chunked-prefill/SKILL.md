---
name: qwen-vllm-chunked-prefill
description: Deploy Qwen3.5 models with vLLM chunked prefill for long context handling. Use PROACTIVELY for high-throughput batch processing, long context inference, or when needing paged attention features.
---

# Qwen vLLM Chunked Prefill Service

Deploy Qwen3.5 models with vLLM for advanced memory management.

## Models

| Model | Params | Q4 Size | Max Context |
|-------|--------|---------|-------------|
| Qwen3.5-0.8B | 0.8B | ~550MB | 32K |
| Qwen3.5-2B | 2B | ~1.2GB | 32K |
| Qwen3.5-4B | 4B | ~2.4GB | 32K |

## Features

- ✅ **Paged Attention** - Efficient KV-cache management
- ✅ **Chunked Prefill** - Handle long contexts without OOM
- ✅ **Continuous Batching** - Higher throughput
- 🚀 Fast inference with vLLM runtime

## Installation

```bash
pip install vllm
```

## Start Services

### Qwen3.5-0.8B (GPU - Laptop)

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 18070 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8
```

### Qwen3.5-2B (GPU - Laptop)

```bash
vllm serve Qwen/Qwen2.5-2B-Instruct \
  --host 0.0.0.0 \
  --port 18072 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9
```

### Qwen3.5-2B with Chunked Prefill (CPU - Spawn)

```bash
vllm serve Qwen/Qwen2.5-2B-Instruct \
  --host 0.0.0.0 \
  --port 18072 \
  --max-model-len 32768 \
  --device cpu \
  --enable-chunked-prefill \
  --chunked-prefill-tokens 512
```

### Qwen3.5-2B Long Context (32K)

```bash
vllm serve Qwen/Qwen2.5-2B-Instruct \
  --host 0.0.0.0 \
  --port 18072 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.95
```

## Chunked Prefill Options

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--enable-chunked-prefill` | Enable chunked prefill | True for ctx > 16K |
| `--max-model-len` | Maximum context length | 8192-32768 |
| `--gpu-memory-utilization` | VRAM usage (GPU) | 0.8-0.95 |
| `--tensor-parallel-size` | Multi-GPU | 2 for 2 GPUs |

## API Usage

### Chat Completion

```bash
curl http://192.168.0.50:18072/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-2B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

### Batch Processing

```bash
curl http://192.168.0.50:18072/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-2B-Instruct",
    "prompt": ["Hello", "Hi there", "Hey"],
    "max_tokens": 100
  }'
```

### Long Context Request

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.50:18072/v1",
    api_key="dummy"
)

# Load a long document
with open("document.txt", "r") as f:
    long_text = f.read()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-2B-Instruct",
    messages=[
        {"role": "user", "content": f"Summarize this document:\n\n{long_text}"}
    ],
    max_tokens=1024
)
print(response.choices[0].message.content)
```

## Service Registry

| Service | Node | Port | Runtime | Model |
|---------|------|------|---------|-------|
| Qwen 0.8B | laptop | 18070 | GPU | 0.8B |
| Qwen 2B | laptop | 18072 | GPU | 2B |
| Qwen 2B (long ctx) | spawn | 18072 | CPU | 2B |

## Use Cases

| Use Case | Recommended Model |
|----------|------------------|
| Fast chat | Qwen3.5-0.8B |
| General chat | Qwen3.5-2B |
| Long documents | Qwen3.5-2B + chunked prefill |
| Batch processing | Qwen3.5-2B + vLLM |

## Performance Tips

### GPU Memory Optimization

```bash
# For limited VRAM
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--enforce-eager  # Disable CUDA graphs (saves memory)
```

### CPU Mode Optimization

```bash
# For CPU-only inference
--device cpu \
--enable-chunked-prefill \
--max-model-len 16384
```

## Comparison: llama.cpp vs vLLM for Qwen

| Feature | llama.cpp | vLLM |
|---------|-----------|------|
| Startup Time | <1s | 10-30s |
| Memory Efficiency | Best | Good |
| Batch Throughput | OK | **Best** |
| Long Context | Limited | **Best** (chunked) |
| CPU Mode | Excellent | Good |

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (2GB+ VRAM)
- **RAM**: 8GB+ for CPU mode
- **Python**: 3.8+

## Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen Models](https://huggingface.co/Qwen)
- [Paged Attention Paper](https://arxiv.org/abs/2309.06180)
