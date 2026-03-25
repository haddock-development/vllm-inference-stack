---
name: glm-vllm-long-context
description: Deploy GLM-4-9B-1M with vLLM for ultra-long context (1M tokens) with MLA architecture. Use PROACTIVELY for long document processing, code analysis, or when user needs 1M context length.
---

# GLM-4 Long Context Service

Deploy GLM-4-9B-1M with vLLM for ultra-long context inference.

## Model

| Model | Params | Q4 Size | Context |
|-------|--------|---------|---------|
| glm-4-9b-chat-1m | 9B | ~5.5GB | **1,048,576 tokens** |

## Features

- ✅ **MLA Architecture** - Multi-head Latent Attention
- ✅ **1M Context** - Ultra-long context support
- 🇨🇳 Chinese-optimized
- 🧠 Great for code understanding
- 💪 CPU-friendly with vLLM chunked prefill

## Start Service (GPU)

```bash
vllm serve THUDM/glm-4-9b-chat-1m \
  --host 0.0.0.0 \
  --port 18103 \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

## Start Service (CPU - Spawn 32GB RAM)

```bash
vllm serve THUDM/glm-4-9b-chat-1m \
  --host 0.0.0.0 \
  --port 18103 \
  --max-model-len 65536 \
  --device cpu \
  --enable-chunked-prefill
```

## Full Context Attention

GLM-4 nutzt **MLA (Multi-head Latent Attention)** für effiziente Long-Context-Verarbeitung.

| Mode | Beschreibung | Use Case |
|------|-------------|----------|
| Sparse (Default) | Nur aktiver KV-Cache Teil | Kurze Prompts |
| **Full Context** | Kompletter KV-Cache | Long Docs, Code Analysis |

### Chunked Prefill aktivieren

```bash
--enable-chunked-prefill         # Verhindert OOM bei langen Contexten
--max-model-len 131072          # 128K context
--gpu-memory-utilization 0.95    # Max VRAM nutzen
```

## API Usage

### Long Document Analysis

```bash
curl http://192.168.0.50:18103/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "THUDM/glm-4-9b-chat-1m",
    "messages": [
      {"role": "user", "content": "Analyze this long document: [128K tokens of text]..."}
    ],
    "max_tokens": 4096
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.50:18103/v1",
    api_key="dummy"
)

# Load a long document
with open("document.txt", "r") as f:
    long_text = f.read()

response = client.chat.completions.create(
    model="THUDM/glm-4-9b-chat-1m",
    messages=[
        {"role": "user", "content": f"Analyze this document:\n\n{long_text}"}
    ],
    max_tokens=4096
)
print(response.choices[0].message.content)
```

## Service Registry

| Service | Node | Port | Runtime | VRAM/RAM |
|---------|------|------|---------|----------|
| GLM-4-9B-1M | spawn | 18103 | CPU | ~12GB RAM |
| GLM-4-9B-1M | laptop | 18103 | GPU | ~8GB VRAM |

## Use Cases

- **Document Analysis** - 1M context für komplette Dokumente
- **Code Review** - Ganze Repositories analysieren
- **Long Conversation Memory** - Extended chat history
- **Chinese Language Tasks** - Optimiert für Chinesisch

## Memory Requirements

| Context Length | VRAM (GPU) | RAM (CPU) |
|----------------|------------|-----------|
| 32K | 8GB | 16GB |
| 64K | 12GB | 24GB |
| 128K | 16GB | 32GB |
| 1M | 64GB+ | 128GB+ |

## Comparison

| Feature | Qwen3.5-4B | GLM-4-9B-1M |
|---------|-------------|-------------|
| Context | 32K | **1M** |
| Architecture | Standard | ✅ MLA |
| Chinese | Good | ✅ Excellent |
| Long Docs | Limited | ✅ Perfect |

## Links

- [GLM-4 Paper](https://arxiv.org/abs/2406.07894)
- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Model](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
