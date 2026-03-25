# Full Context Attention Models

> Kleine Modelle mit MLA/DSA Architektur oder Long Context Support

## Was ist Full Context Attention?

**WICHTIG:** Full Context Attention ist ein **Runtime-Feature**, kein Modell-Feature!

| Begriff | Typ | Erklärung |
|---------|-----|-----------|
| GGUF / GPTQ / AWQ | Format | Quantisierung des Modells |
| **Full Context Attention** | Runtime | Erweitert aktiven Kontextzugriff bei sparse/indexed attention |
| MLA / DSA | Architektur | Multi-head Latent Attention / DeepSeek Attention |

**Du brauchst:**
1. Modell mit MLA/DSA Architektur
2. Runtime die den Switch exponiert (vLLM, SGLang, KTransformers)

---

## Kleine Modelle mit MLA/DSA (Sparse Attention)

### DeepSeek-V2-Lite Familie

| Modell | Params | Active | MLA | Max Context | Q4 Größe | VRAM |
|--------|--------|--------|-----|-------------|----------|------|
| **DeepSeek-V2-Lite** | 15.7B | 2.4B | ✅ | 128K | ~10GB | 12GB+ |
| **DeepSeek-Coder-V2-Lite** | 15.7B | 2.4B | ✅ | 128K | ~10GB | 12GB+ |

**GGUF Downloads:**
```bash
# DeepSeek-V2-Lite (Chat)
hf download mradermacher/DeepSeek-V2-Lite-GGUF DeepSeek-V2-Lite.Q4_K_M.gguf --local-dir ~/models

# DeepSeek-Coder-V2-Lite (Code)
hf download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf --local-dir ~/models
```

**vLLM Start:**
```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
  --host 0.0.0.0 \
  --port 18090 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

### GLM-4 Familie (z.ai / GLM Tech) - Alle mit MLA! 🎯

| Modell | Params | MLA | Vision | Max Context | Q4 Größe | VRAM | Status |
|--------|--------|-----|--------|-------------|----------|------|--------|
| **GLM-4-9B-Chat** | 9B | ✅ | ❌ | 128K | ~5.5GB | 6GB+ | ✅ Passt |
| **GLM-4-9B-Chat-1M** | 9B | ✅ | ❌ | **1M** | ~5.5GB | 8GB+ | ✅ Passt |
| **GLM-4.6V-Flash** | 10.3B | ✅ | ✅ | 128K | ~6GB | 10GB+ | ⚠️ Knapp |
| **GLM-4.7-Flash** | 31B MoE | ✅ | ❌ | 128K | ~17GB | 20GB+ | ❌ Zu groß |

**GGUF Downloads:**
```bash
# GLM-4-9B-Chat-1M (Passt auf Laptop!)
hf download gaianet/glm-4-9b-chat-1m-GGUF glm-4-9b-chat-1m-Q4_0.gguf --local-dir ~/models

# GLM-4.6V-Flash (Vision, MLA - braucht 10GB+ VRAM)
hf download unsloth/GLM-4.6V-Flash-GGUF GLM-4.6V-Flash-Q4_K_M.gguf --local-dir ~/models
hf download unsloth/GLM-4.6V-Flash-GGUF mmproj-F16.gguf --local-dir ~/models

# GLM-4.7-Flash (31B MoE - braucht 20GB+ VRAM)
hf download unsloth/GLM-4.7-Flash-GGUF GLM-4.7-Flash-Q4_K_M.gguf --local-dir ~/models
```

**vLLM Start:**
```bash
# GLM-4-9B-Chat-1M (Laptop 8GB)
vllm serve THUDM/glm-4-9b-chat-1m \
  --host 0.0.0.0 \
  --port 18103 \
  --max-model-len 65536 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9

# GLM-4.6V-Flash (12GB+ VRAM)
vllm serve zai-org/GLM-4.6V-Flash \
  --host 0.0.0.0 \
  --port 18104 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9

# GLM-4.7-Flash (24GB+ VRAM)
vllm serve zai-org/GLM-4.7-Flash \
  --host 0.0.0.0 \
  --port 18105 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.95
```

**GLM-4.7-Flash Highlights:**
- 🏆 **31B MoE** (Multi-Expert Architecture)
- ✅ **MLA** (Multi-head Latent Attention)
- 🚀 **MIT License** - Full Open Source
- 📊 **3.7M Downloads** - Very Popular
- 🌐 Bilingual (EN, ZH)

---

## Aktuelle Top Small Models (2026)

### Qwen3.5 Familie (Alibaba - Neueste Generation)

| Modell | Params | MLA | Max Context | Q4 Größe | VRAM | Bewertung |
|--------|--------|-----|-------------|----------|------|-----------|
| **Qwen3.5-0.8B** | 0.8B | ❌ | 32K | ~550MB | 1GB | ⭐ Ultra-fast |
| **Qwen3.5-2B** | 2B | ❌ | 32K | ~1.2GB | 2GB | ⭐⭐ Balanced |
| **Qwen3.5-4B** | 4B | ❌ | 32K | ~2.4GB | 4GB | ⭐⭐⭐ Best Quality |

**GGUF Downloads:**
```bash
# Qwen3.5-0.8B (Schnellste)
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf --local-dir ~/models

# Qwen3.5-2B (Balance)
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf --local-dir ~/models

# Qwen3.5-4B (Beste Qualität)
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir ~/models
```

### Gemma 3 Familie (Google, März 2026)

| Modell | Params | Vision | Max Context | Q4 Größe | VRAM | Bewertung |
|--------|--------|--------|-------------|----------|------|-----------|
| **Gemma-3-270M** | 270M | ❌ | 32K | ~180MB | 0.5GB | ⭐ Ultra-fast |
| **Gemma-3-1B** | 1B | ❌ | 32K | ~700MB | 1GB | ⭐⭐ Fast |
| **Gemma-3-4B-VL** | 4B | ✅ | **128K** | ~2.4GB | 4GB | ⭐⭐⭐ Best Value |
| **Gemma-3-12B-VL** | 12B | ✅ | 128K | ~7GB | 10GB | ⭐⭐⭐ High Quality |

**GGUF Downloads:**
```bash
# Gemma-3-1B
hf download unsloth/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf --local-dir ~/models

# Gemma-3-4B-VL (Vision!)
hf download unsloth/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir ~/models
hf download unsloth/gemma-3-4b-it-GGUF mmproj-F16.gguf --local-dir ~/models
```

### Granite 4.0 Familie (IBM, 2026)

| Modell | Params | Type | Max Context | Q4 Größe | VRAM |
|--------|--------|------|-------------|----------|------|
| **Granite-4.0-1B-Speech** | 1B | ASR/TTS | 16K | ~700MB | 2GB |
| **Granite-3.3-2B** | 2B | Text | 32K | ~1.2GB | 3GB |
| **Granite-3.3-8B** | 8B | Text | 128K | ~5GB | 8GB |

**Hinweis:** IBM Granite ist **Apache 2.0** lizenziert!

**GGUF Downloads:**
```bash
# Granite Speech (ASR)
hf download ibm-granite/granite-4.0-1b-speech-GGUF granite-4.0-1b-speech-Q4_K_M.gguf --local-dir ~/models
```

### Kimi K2.5 Familie (Moonshot, 2026)

| Modell | Params | MLA | Max Context | Q4 Größe | VRAM |
|--------|--------|-----|-------------|----------|------|
| **Kimi-K2.5** | 1T MoE | ✅ | 128K | Various | 12GB+ |

**Hinweis:** Kimi nutzt MLA-ähnliche Architektur für Long Context.

**GGUF Downloads:**
```bash
# Kimi K2.5
hf download unsloth/Kimi-K2.5-GGUF Kimi-K2.5-IQ4_NL-*.gguf --local-dir ~/models
```

### LFM2.5 Familie (Liquid AI, 2026)

| Modell | Params | Type | Max Context | Q4 Größe | VRAM |
|--------|--------|------|-------------|----------|------|
| **LFM2.5-VL-1.6B** | 1.6B | Vision | 32K | ~700MB | 2GB |
| **LFM2.5-1.2B-Thinking** | 1.2B | Reasoning | 32K | ~700MB | 2GB |
| **LFM2.5-1.2B-Instruct** | 1.2B | Chat | 32K | ~700MB | 2GB |

**GGUF Downloads:**
```bash
# LFM2.5-VL (Vision)
hf download unsloth/LFM2.5-VL-1.6B-GGUF LFM2.5-VL-1.6B-Q4_K_M.gguf --local-dir ~/models

# LFM2.5-Thinking (Reasoning)
hf download LiquidAI/LFM2.5-1.2B-Thinking-GGUF LFM2.5-1.2B-Thinking-Q4_K_M.gguf --local-dir ~/models
```

### Phi-3.5 Familie (Microsoft, 2024)

| Modell | Params | MLA | Max Context | Q4 Größe | VRAM |
|--------|--------|-----|-------------|----------|------|
| **Phi-3.5-mini** | 3.8B | ❌ | 128K | ~2.3GB | 4GB+ |

**GGUF Downloads:**
```bash
hf download bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir ~/models
```

**vLLM Start:**
```bash
vllm serve microsoft/Phi-3.5-mini-instruct \
  --host 0.0.0.0 \
  --port 18095 \
  --max-model-len 32768 \
  --enable-chunked-prefill
```

### MiniCPM 4 Familie (OpenBMB, 2026)

| Modell | Params | MLA | Max Context | Q4 Größe | VRAM |
|--------|--------|-----|-------------|----------|------|
| **MiniCPM-4-0.5B** | 0.5B | ❌ | 32K | ~300MB | 0.5GB |
| **MiniCPM-4-2B** | 2B | ❌ | 32K | ~1.2GB | 2GB |

---

## Vergleichstabelle: Alle Modelle

### MLA/DSA Modelle (Full Context Attention) 🎯

| Modell | Params | MLA | Vision | Max Ctx | Q4 Size | VRAM | Status |
|--------|--------|-----|--------|---------|---------|------|--------|
| **GLM-4-9B-Chat-1M** | 9B | ✅ | ❌ | **1M** | 5.5GB | 8GB | ✅ Laptop |
| **GLM-4-9B-Chat** | 9B | ✅ | ❌ | 128K | 5.5GB | 6GB | ✅ Laptop |
| **GLM-4.6V-Flash** | 10.3B | ✅ | ✅ | 128K | 6GB | 10GB | ⚠️ Knapp |
| **GLM-4.7-Flash** | 31B | ✅ | ❌ | 128K | 17GB | 20GB | ❌ Server |
| **DeepSeek-V2-Lite** | 15.7B | ✅ | ❌ | 128K | 10GB | 12GB | ⚠️ Knapp |
| **DeepSeek-Coder-V2-Lite** | 15.7B | ✅ | ❌ | 128K | 10GB | 12GB | ⚠️ Code |

### Standard Long Context Modelle

| Modell | Params | MLA | Vision | Max Ctx | Q4 Size | Best For |
| **Gemma-3-4B-VL** | 4B | ❌ | 128K | 2.4GB | Vision + Long Ctx |
| **Qwen3.5-4B** | 4B | ❌ | 32K | 2.4GB | Text Quality |
| **LFM2.5-VL-1.6B** | 1.6B | ❌ | 32K | 700MB | Vision, Edge |
| **Phi-3.5-mini** | 3.8B | ❌ | 128K | 2.3GB | Fast, Long Ctx |
| **Gemma-3-1B** | 1B | ❌ | 32K | 700MB | Fast |
| **Qwen3.5-2B** | 2B | ❌ | 32K | 1.2GB | Balanced |
| **MiniCPM-4-2B** | 2B | ❌ | 32K | 1.2GB | Edge |
| **Gemma-3-270M** | 270M | ❌ | 32K | 180MB | Ultra-fast |
| **Qwen3.5-0.8B** | 0.8B | ❌ | 32K | 550MB | Ultra-fast |

---

## Hardware Requirements

### Für MLA/DSA Modelle (DeepSeek-V2-Lite, GLM-4)

| Context | GPU VRAM | CPU RAM |
|---------|----------|---------|
| 8K | 6GB | 8GB |
| 16K | 8GB | 12GB |
| 32K | 10GB | 16GB |
| 64K | 14GB | 24GB |
| 128K | 18GB | 32GB |

### Für Standard Long Context (Gemma-3, Qwen, Phi)

| Context | GPU VRAM | CPU RAM |
|---------|----------|---------|
| 8K | 3GB | 4GB |
| 16K | 4GB | 6GB |
| 32K | 5GB | 8GB |
| 64K | 6GB | 12GB |
| 128K | 8GB | 16GB |

---

## Empfehlung für dein Setup

### Laptop (GTX 1070, 8GB VRAM)

| Modell | Context | Machbar? |
|--------|---------|----------|
| **Gemma-3-4B-VL** | 64K | ✅ Top Choice |
| **Qwen3.5-4B** | 32K | ✅ |
| **GLM-4-9B-Chat** | 32K | ✅ Mit Chunked Prefill |
| **LFM2.5-VL-1.6B** | 32K | ✅ Vision |
| **Phi-3.5-mini** | 64K | ✅ |
| **DeepSeek-V2-Lite** | 8K | ⚠️ Knapp |

### Spawn (32GB RAM, CPU)

| Modell | Context | Machbar? |
|--------|---------|----------|
| **GLM-4-9B-1M** | 64K | ✅ Mit Chunked Prefill |
| **Gemma-3-4B-VL** | 128K | ✅ |
| **DeepSeek-V2-Lite** | 32K | ✅ CPU Mode |
| **Qwen3.5-4B** | 32K | ✅ |

---

## Download Script (Alle Modelle)

```bash
#!/bin/bash
MODELS_DIR=~/models
mkdir -p $MODELS_DIR

# === MLA/DSA Modelle (Full Context Attention) ===
echo "=== GLM-4 Familie (MLA) - Alle mit Full Context Attention! ==="
hf download gaianet/glm-4-9b-chat-1m-GGUF glm-4-9b-chat-1m-Q4_0.gguf --local-dir $MODELS_DIR
hf download unsloth/GLM-4.6V-Flash-GGUF GLM-4.6V-Flash-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download unsloth/GLM-4.6V-Flash-GGUF mmproj-F16.gguf --local-dir $MODELS_DIR
hf download unsloth/GLM-4.7-Flash-GGUF GLM-4.7-Flash-Q4_K_M.gguf --local-dir $MODELS_DIR

echo "=== DeepSeek-V2-Lite (MLA) ==="
hf download mradermacher/DeepSeek-V2-Lite-GGUF DeepSeek-V2-Lite.Q4_K_M.gguf --local-dir $MODELS_DIR
hf download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf --local-dir $MODELS_DIR

# === Aktuelle Top Modelle (2026) ===
echo "=== Qwen3.5 ==="
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir $MODELS_DIR

echo "=== Gemma 3 ==="
hf download unsloth/gemma-3-270m-it-GGUF gemma-3-270m-it-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download unsloth/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download unsloth/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download unsloth/gemma-3-4b-it-GGUF mmproj-F16.gguf --local-dir $MODELS_DIR

echo "=== LFM2.5 (Liquid AI) ==="
hf download unsloth/LFM2.5-VL-1.6B-GGUF LFM2.5-VL-1.6B-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download LiquidAI/LFM2.5-1.2B-Thinking-GGUF LFM2.5-1.2B-Thinking-Q4_K_M.gguf --local-dir $MODELS_DIR
hf download LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf --local-dir $MODELS_DIR

echo "=== Granite (IBM) ==="
hf download ibm-granite/granite-4.0-1b-speech-GGUF granite-4.0-1b-speech-Q4_K_M.gguf --local-dir $MODELS_DIR

echo "=== Phi-3.5 ==="
hf download bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir $MODELS_DIR

echo "Done! Alle Modelle in $MODELS_DIR"
```

---

## Links

- [DeepSeek-V2 Paper (MLA)](https://arxiv.org/abs/2405.04434)
- [GLM-4 Paper](https://arxiv.org/abs/2406.07894)
- [Qwen3.5 Release](https://qwenlm.github.io/)
- [Gemma 3 Release](https://blog.google/technology/gemma/)
- [LFM2.5 Release](https://www.liquid.ai/)
- [Granite Models](https://www.ibm.com/granite)
- [vLLM Documentation](https://docs.vllm.ai/)
