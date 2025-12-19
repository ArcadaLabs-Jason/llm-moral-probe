# Open-Source Base Models for Moral Probe Research
## Full Precision (FP16/BF16) VRAM Requirements — December 2025

> **VRAM Formula**: `(Parameters × 2 bytes) × 1.2 overhead`  
> **Note**: For MoE models, ALL parameters must be loaded into VRAM, not just active parameters.

---

## 🔷 DENSE MODELS (Recommended for Consistent Results)

### Tier 1: Consumer GPU (≤24GB) — RTX 4090 / RTX 3090

| Model | HuggingFace ID | Params | VRAM (BF16) | Release | License | Context |
|-------|----------------|--------|-------------|---------|---------|---------|
| Qwen3-0.6B-Base | `Qwen/Qwen3-0.6B-Base` | 0.6B | ~2 GB | Apr 2025 | Apache 2.0 | 32K |
| Gemma-3-1B-PT | `google/gemma-3-1b-pt` | 1B | ~3 GB | Mar 2025 | Gemma | 32K |
| Qwen3-1.7B-Base | `Qwen/Qwen3-1.7B-Base` | 1.7B | ~4 GB | Apr 2025 | Apache 2.0 | 32K |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | 1B | ~3 GB | Sep 2024 | Llama 3.2 | 128K |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | 3B | ~8 GB | Sep 2024 | Llama 3.2 | 128K |
| Gemma-3-4B-PT | `google/gemma-3-4b-pt` | 4B | ~10 GB | Mar 2025 | Gemma | 128K |
| Qwen3-4B-Base | `Qwen/Qwen3-4B-Base` | 4B | ~10 GB | Apr 2025 | Apache 2.0 | 32K |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | 7B | ~17 GB | May 2024 | Apache 2.0 | 32K |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | 7B | ~17 GB | Sep 2024 | Apache 2.0 | 128K |
| **Qwen3-8B-Base** | `Qwen/Qwen3-8B-Base` | 8B | ~19 GB | Apr 2025 | Apache 2.0 | 32K |
| **Llama-3.1-8B** | `meta-llama/Llama-3.1-8B` | 8B | ~19 GB | Jul 2024 | Llama 3.1 | 128K |
| Gemma-2-9B | `google/gemma-2-9b` | 9B | ~22 GB | Jun 2024 | Gemma | 8K |

### Tier 2: Professional GPU (40-48GB) — A100-40GB / RTX 6000 Ada

| Model | HuggingFace ID | Params | VRAM (BF16) | Release | License | Context |
|-------|----------------|--------|-------------|---------|---------|---------|
| Gemma-3-12B-PT | `google/gemma-3-12b-pt` | 12B | ~29 GB | Mar 2025 | Gemma | 128K |
| **Qwen3-14B-Base** | `Qwen/Qwen3-14B-Base` | 14B | ~34 GB | Apr 2025 | Apache 2.0 | 32K |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B` | 14B | ~34 GB | Sep 2024 | Apache 2.0 | 128K |
| Phi-4 | `microsoft/phi-4` | 14B | ~34 GB | Dec 2024 | MIT | 16K |

### Tier 3: High-End GPU (80GB) — A100-80GB / H100

| Model | HuggingFace ID | Params | VRAM (BF16) | Release | License | Context |
|-------|----------------|--------|-------------|---------|---------|---------|
| Gemma-3-27B-PT | `google/gemma-3-27b-pt` | 27B | ~65 GB | Mar 2025 | Gemma | 128K |
| **Qwen3-32B-Base** | `Qwen/Qwen3-32B-Base` | 32B | ~77 GB | Apr 2025 | Apache 2.0 | 32K |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B` | 32B | ~77 GB | Sep 2024 | Apache 2.0 | 128K |

### Tier 4: Multi-GPU (2× A100-80GB / 2× H100)

| Model | HuggingFace ID | Params | VRAM (BF16) | Release | License | Context |
|-------|----------------|--------|-------------|---------|---------|---------|
| **Llama-3.1-70B** | `meta-llama/Llama-3.1-70B` | 70B | ~168 GB | Jul 2024 | Llama 3.1 | 128K |
| Qwen2.5-72B | `Qwen/Qwen2.5-72B` | 72B | ~173 GB | Sep 2024 | Apache 2.0 | 128K |

### Tier 5: Large Cluster (8× A100-80GB / 8× H100)

| Model | HuggingFace ID | Params | VRAM (BF16) | Release | License | Context |
|-------|----------------|--------|-------------|---------|---------|---------|
| Llama-3.1-405B | `meta-llama/Llama-3.1-405B` | 405B | ~972 GB | Jul 2024 | Llama 3.1 | 128K |

---

## 🔶 MIXTURE-OF-EXPERTS (MoE) MODELS

> ⚠️ **Important**: MoE models load ALL parameters into VRAM. Active parameters only affect compute, not memory.

### Tier 3: High-End GPU (80GB) — Single H100

| Model | HuggingFace ID | Total / Active | VRAM (BF16) | Release | License | Context |
|-------|----------------|----------------|-------------|---------|---------|---------|
| Qwen3-30B-A3B-Base | `Qwen/Qwen3-30B-A3B-Base` | 30B / 3B | ~72 GB | Apr 2025 | Apache 2.0 | 32K |

### Tier 4: Multi-GPU (2-4× H100)

| Model | HuggingFace ID | Total / Active | VRAM (BF16) | Release | License | Context |
|-------|----------------|----------------|-------------|---------|---------|---------|
| Mixtral-8x7B-v0.1 | `mistralai/Mixtral-8x7B-v0.1` | 47B / 13B | ~113 GB | Dec 2023 | Apache 2.0 | 32K |
| Llama-4-Scout | `meta-llama/Llama-4-Scout-17B-16E` | 109B / 17B | ~262 GB | Apr 2025 | Llama 4 | 10M |
| Mixtral-8x22B-v0.1 | `mistralai/Mixtral-8x22B-v0.1` | 141B / 39B | ~338 GB | Apr 2024 | Apache 2.0 | 64K |

### Tier 5: Large Cluster (8× H100)

| Model | HuggingFace ID | Total / Active | VRAM (BF16) | Release | License | Context |
|-------|----------------|----------------|-------------|---------|---------|---------|
| Qwen3-235B-A22B-Base | `Qwen/Qwen3-235B-A22B-Base` | 235B / 22B | ~564 GB | Apr 2025 | Apache 2.0 | 32K |
| Llama-4-Maverick | `meta-llama/Llama-4-Maverick-17B-128E` | 400B / 17B | ~960 GB | Apr 2025 | Llama 4 | 1M |

### Tier 6: Data Center (16× H100 or FP8 optimized)

| Model | HuggingFace ID | Total / Active | VRAM | Release | License | Context |
|-------|----------------|----------------|------|---------|---------|---------|
| DeepSeek-V3-Base | `deepseek-ai/DeepSeek-V3-Base` | 671B / 37B | ~700 GB (FP8) / ~1.4 TB (BF16) | Dec 2024 | DeepSeek | 128K |
| DeepSeek-R1-Zero | `deepseek-ai/DeepSeek-R1-Zero` | 671B / 37B | ~700 GB (FP8) / ~1.4 TB (BF16) | Jan 2025 | MIT | 128K |

---

## 📊 QUICK REFERENCE: GPU → Best Base Model Match

| Your Hardware | VRAM | Recommended Base Models |
|---------------|------|-------------------------|
| RTX 3060 12GB | 12 GB | Qwen3-4B-Base, Gemma-3-4B-PT |
| RTX 4070 Ti 16GB | 16 GB | Mistral-7B-v0.3, Qwen2.5-7B |
| RTX 4090 / 3090 | 24 GB | **Qwen3-8B-Base**, **Llama-3.1-8B**, Gemma-2-9B |
| A100-40GB | 40 GB | **Qwen3-14B-Base**, Phi-4, Gemma-3-12B-PT |
| RTX 6000 Ada | 48 GB | Qwen3-14B-Base + headroom for long context |
| A100-80GB / H100 | 80 GB | **Qwen3-32B-Base**, Gemma-3-27B-PT |
| 2× A100-80GB | 160 GB | **Llama-3.1-70B**, Qwen2.5-72B |
| 2× H100 | 160 GB | Llama-4-Scout (with FP8), Mixtral-8x22B |
| 4× H100 | 320 GB | Llama-4-Scout (BF16), Qwen3-235B-A22B-Base |
| 8× H100 | 640 GB | DeepSeek-V3-Base (FP8), Llama-4-Maverick |
| 16× H100 | 1.28 TB | DeepSeek-V3-Base (BF16), Llama-3.1-405B |

---

## 🎯 RECOMMENDED TEST SUITE FOR MORAL PROBE RESEARCH

For testing the hypothesis that "scale doesn't clearly improve moral coherence":

| Scale Tier | Model | Why |
|------------|-------|-----|
| **Small (≤4B)** | Qwen3-4B-Base | Latest architecture, strong for size |
| **Medium (7-8B)** | Qwen3-8B-Base | Latest Qwen, outperforms Qwen2.5-14B on many benchmarks |
| **Medium (7-8B)** | Llama-3.1-8B | Baseline Meta model, widely studied |
| **Large (14B)** | Qwen3-14B-Base | Competes with Qwen2.5-32B |
| **Large (27-32B)** | Qwen3-32B-Base | Largest dense Qwen3 |
| **XL (70B)** | Llama-3.1-70B | Gold standard for scale comparison |

### Minimum Viable Test Set (fits on 24GB GPU):
1. `Qwen/Qwen3-4B-Base` — 10 GB
2. `Qwen/Qwen3-8B-Base` — 19 GB  
3. `meta-llama/Llama-3.1-8B` — 19 GB

### Extended Test Set (requires 80GB GPU):
Add: `Qwen/Qwen3-32B-Base` — 77 GB

### Full Scale Test (requires 2× 80GB):
Add: `meta-llama/Llama-3.1-70B` — 168 GB

---

## 📝 NOTES

### Why These Models?

1. **Qwen3 series** (April 2025) — Latest architecture with thinking/non-thinking modes, strong benchmarks
2. **Llama 3.1 series** (July 2024) — Industry standard, extensive documentation
3. **Gemma 3 series** (March 2025) — Google's best open model, multimodal capable
4. **Phi-4** (December 2024) — Microsoft's latest, excellent efficiency

### Base vs Instruct Models

For moral probe research testing emergent ethics in **pretraining alone**:
- ✅ Use `-Base` or no suffix (e.g., `Qwen3-8B-Base`, `Llama-3.1-8B`)
- ❌ Avoid `-Instruct`, `-IT`, `-Chat` variants (these have RLHF/DPO alignment)

### KV Cache Considerations

The VRAM figures above assume minimal context. For long prompts, add:
- 8B model @ 32K context: +4.5 GB for KV cache
- 32B model @ 32K context: +18 GB for KV cache
- 70B model @ 32K context: +40 GB for KV cache

### Inference Frameworks

For running at full BF16 precision:
- **vLLM** — Best for throughput, supports tensor parallelism
- **Hugging Face Transformers** — Most flexible, `device_map="auto"`
- **SGLang** — Optimized for Qwen/Llama, good for batching
- **TGI** — Production-ready, HTTP API
