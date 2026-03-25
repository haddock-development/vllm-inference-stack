# vLLM Inference Stack

> High-performance inference with vLLM runtime - **Full Context Attention, Paged Attention, Chunked Prefill**

## Attention Mechanisms in vLLM

### What is "Full Context Attention"?

**WICHTIG:** "Full Context Attention" ist **kein Quantizer-Format** sondern ein **Runtime-Schalter**!

| Begriff | Typ | Erklärung |
|---------|-----|-----------|
| GGUF / GPTQ / AWQ | Format | Quantisierung des Modells |
| **Full Context Attention** | Runtime | Erweitert aktiven Kontextzugriff bei sparse/indexed attention |
| Distributed Compute | Runtime | Verteilung über mehrere Devices/Nodes |

**Du brauchst:**
1. Modell mit MLA/DSA Architektur (GLM-5, DeepSeek-V3)
2. Runtime die den Schalter exponiert (vLLM, SGLang, KTransformers)

---

## Modelle mit Full Context Attention Support

### MLA/DSA Architektur (Sparse Attention)

| Modell | Params | MLA/DSA | Max Context | GGUF Größe |
|--------|--------|---------|-------------|------------|
| **GLM-4-9B-1M** | 9B | ✅ MLA | **1M tokens** | ~5.5GB Q4 |
| DeepSeek-V2-Lite | 15.7B | ✅ MLA | 128K | ~10GB Q4 |
| DeepSeek-V3.2-Exp | 685B | ✅ DSA | 128K | Riesig |
| Kimi-K2.5 | 1T | ✅ MLA | 128K | Riesig |

### Standard Architektur (Kein Sparse Attention)

| Modell | Params | MLA/DSA | Max Context | GGUF Größe |
|--------|--------|---------|-------------|------------|
| Qwen3.5-0.8B | 0.8B | ❌ | 32K | ~550MB |
| Qwen3.5-2B | 2B | ❌ | 32K | ~1.2GB |
| Qwen3.5-4B | 4B | ❌ | 32K | ~2.4GB |

---

## vLLM Attention Features

### 1. Paged Attention (Default)

```
┌─────────────────────────────────────────────┐
│  Paged Attention                            │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
│  │ K0 │ │ K1 │ │ K2 │ │ K3 │  KV-Cache    │
│  │ V0 │ │ V1 │ │ V2 │ │ V3 │  in Pages    │
│  └────┘ └────┘ └────┘ └────┘              │
│  → Memory efficient, keine Fragmentierung   │
└─────────────────────────────────────────────┘
```

**Aktivierung:** Automatisch (Default)

### 2. Chunked Prefill (Long Context)

```
┌─────────────────────────────────────────────┐
│  Chunked Prefill                            │
│  ┌──────┐ ┌──────┐ ┌──────┐               │
│  │ 512  │ │ 512  │ │ 512  │  Tokens       │
│  │ tok  │ │ tok  │ │ tok  │  per Chunk    │
│  └──────┘ └──────┘ └──────┘               │
│  → Verhindert OOM bei langen Contexten      │
└─────────────────────────────────────────────┘
```

**Aktivierung:** `--enable-chunked-prefill`

### 3. Full Context Attention (GLM/DeepSeek)

```
┌─────────────────────────────────────────────┐
│  Sparse Attention (Default)                 │
│  ┌───────────────────────────────┐         │
│  │ ░░░░░██████░░░░░██████░░░░░   │         │
│  │ ░░░░░██████░░░░░██████░░░░░   │ ← Sparse│
│  └───────────────────────────────┘         │
│                                             │
│  Full Context Attention (Aktiviert)         │
│  ┌───────────────────────────────┐         │
│  │ ████████████████████████████  │         │
│  │ ████████████████████████████  │ ← Full  │
│  └───────────────────────────────┘         │
│  → Bessere Qualität bei langen Contexten    │
└─────────────────────────────────────────────┘
```

**WICHTIG:** Nur bei MLA/DSA Modellen (GLM-4, DeepSeek-V2+)!

---

## Quick Start

### Installation

```bash
pip install vllm

# Mit CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### GLM-4-9B-1M (Full Context Attention mit MLA)

```bash
vllm serve THUDM/glm-4-9b-chat-1m \
  --host 0.0.0.0 \
  --port 18103 \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

### Qwen3.5-2B (Chunked Prefill für Long Context)

```bash
vllm serve Qwen/Qwen2.5-2B-Instruct \
  --host 0.0.0.0 \
  --port 18072 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --chunked-prefill-tokens 512
```

---

## Full Context Attention Details

### Was passiert bei "Full Context Attention"?

Bei Modellen mit **MLA (Multi-head Latent Attention)** oder **DSA (DeepSeek Attention)**:

1. **Default (Sparse):** Nur ein Teil des KV-Cache wird aktiv gehalten
2. **Full Context:** Kompletter KV-Cache wird für alle Tokens zugänglich

### Wann aktivieren?

| Szenario | Empfehlung |
|----------|------------|
| Kurze Prompts (<4K) | Sparse (Default) |
| Lange Dokumente (>16K) | **Full Context** |
| Code-Analysis (gesamtes Repo) | **Full Context** |
| Multi-Turn Chat | Sparse OK |

### vLLM Flags für Long Context

```bash
--max-model-len 131072          # 128K context
--enable-chunked-prefill         # Verhindert OOM
--gpu-memory-utilization 0.95    # Max VRAM nutzen
--tensor-parallel-size 2         # Multi-GPU (falls verfügbar)
```

---

## Vergleich: llama.cpp vs vLLM

| Feature | llama.cpp | vLLM |
|---------|-----------|------|
| **GGUF** | ✅ Native | ✅ Via loader |
| **Paged Attention** | ❌ | ✅ |
| **Chunked Prefill** | ❌ | ✅ |
| **Continuous Batching** | ❌ | ✅ |
| **Full Context Toggle** | ❌ | ✅ (bei MLA/DSA) |
| **CPU Efficiency** | ✅ Best | ⚠️ OK |
| **RAM Usage** | ✅ Minimal | ⚠️ Higher |
| **Edge/Mobile** | ✅ Perfect | ❌ |
| **Multi-User Throughput** | ⚠️ OK | ✅ Best |

---

## Hardware Requirements

### Für Full Context Attention mit GLM-4-9B

| Context Length | VRAM (GPU) | RAM (CPU) |
|----------------|------------|-----------|
| 32K | 8GB | 16GB |
| 64K | 12GB | 24GB |
| 128K | 16GB | 32GB |
| 1M | 64GB+ | 128GB+ |

### Für Chunked Prefill mit Qwen2.5-2B

| Context Length | VRAM (GPU) | RAM (CPU) |
|----------------|------------|-----------|
| 8K | 2GB | 4GB |
| 16K | 3GB | 6GB |
| 32K | 4GB | 8GB |
| 64K | 6GB | 12GB |

---

## API Usage

### OpenAI-Compatible

```bash
curl http://192.168.0.50:18103/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "THUDM/glm-4-9b-chat-1m",
    "messages": [
      {"role": "user", "content": "Analysiere dieses lange Dokument: ..."}
    ],
    "max_tokens": 4096
  }'
```

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.50:18103/v1",
    api_key="dummy"
)

# Long context request
with open("document.txt", "r") as f:
    long_text = f.read()

response = client.chat.completions.create(
    model="THUDM/glm-4-9b-chat-1m",
    messages=[{"role": "user", "content": f"Analysiere: {long_text}"}],
    max_tokens=4096
)
print(response.choices[0].message.content)
```

---

## Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [Paged Attention Paper](https://arxiv.org/abs/2309.06180)
- [GLM-4 Paper](https://arxiv.org/abs/2406.07894)
- [DeepSeek-V2 Paper (MLA)](https://arxiv.org/abs/2405.04434)

---

*Last updated: 2026-03-25*
