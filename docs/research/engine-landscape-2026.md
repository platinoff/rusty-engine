# Engine landscape — research & comparison (2026)

**Product:** `rusty-engine` · збір: 2026-10-14 (agi-сесія «verify + research»).
Мета: зафіксувати актуальний стан ринку MoE-inference та **чесно позиціонувати** `rusty-engine`
(що не конкурує за H100-throughput, а займає іншу нішу). Джерела — знизу; llama-rs/poolAI —
локальна генетична кодова база.

## 1. TL;DR позиціонування

| | vLLM · SGLang · TensorRT-LLM | llama.cpp / ktransformers | **rusty-engine** |
|--|------------------------------|---------------------------|------------------|
| Мова ядра | Python + CUDA/C++ | C/C++ (ggml) | **pure Rust** (без C++/Python-ядра) |
| Артефакт | pip + контейнер, GPU-сервер | `llama-server` бінарник | **один portable `rusty` exe/bin/apk** (amd64+arm64) |
| MoE-ядро | grouped-GEMM на GPU | у C++ llama.cpp / kt-kernel | **Rust router+experts+shared** (PH-401..406a) |
| Розподіл | TP/EP/DP (GPU-кластери, NVIDIA/AMD) | RPC / `-sm layer` (одна машина) | **GSV-native master/slave «Rust-k8s-lite»** по локальному Wi-Fi/LAN + Android-slave |
| Edge/CPU | слабко (GPU-centric) | сильно (SIMD/NEON) | **сильно + Android без Docker** |

`rusty-engine` **не** намагається перемогти SGLang на 8×H200. Його вісь: локальний/гетерогенний
(first-person) MoE, один Rust-бінарник на OS×arch, розподіл по пристроях під GSV-хабом,
і **нативний Rust-MoE** (якого немає в llama-rs — той делегує на C++ llama.cpp).

## 2. Стан рушіїв (вересень–листопад 2026)

- **vLLM** — `v0.28.0` (серп. 2026), ~91k ★, PyTorch Foundation, **Apache-2.0**. Найширша
  підтримка HW (NVIDIA, AMD ROCm, TPU, Gaudi, Neuron, Ascend) і **найширший квант-стек**
  (FP8/MXFP8/**MXFP4**/NVFP4/INT8/INT4/GPTQ/AWQ/**GGUF**/compressed-tensors + quantized KV).
  Expert-parallel `--enable-expert-parallel`; PagedAttention. MLPerf v6.0 (gpt-oss-120b, 8×B200):
  Red Hat llm-d+vLLM **93k tok/s Offline**, 71.6k Server.
- **SGLang** — `v0.5.19` (вер. 2026), ~35.5k ★; governance-зміна: виділився у компанію
  **RadixArk** (~$400M пост-мани, NVIDIA/AMD venture), лишається Apache-2.0. Фірмові речі:
  **RadixAttention** (KV у radix-дереві, longest-prefix) + **zero-overhead CPU scheduler**;
  EP `--moe-dp-size`; day-0 для DeepSeek/Kimi/GLM. Еталон MoE-throughput: **~2.5× vLLM** на
  DeepSeek-R1/V3 (52 tok/s на 8×H200 LMSYS). Структурований JSON як kernel-проблема.
- **TensorRT-LLM** — `v1.2.1` стабільний, NVIDIA, ~14.5k ★, Apache-2.0, **NVIDIA-only**.
  Розворот: `v1.0` (вер. 2025) прибрав TensorRT-backend → **PyTorch-бекенд, без обовʼязкового
  AOT-компілювання**, pip-install, OpenAI-endpoint. Найглибший NVFP4/fused-MoE; унікальна
  фіча: **guided decoding + speculative decoding** разом. MLPerf Server 87.4k (Nebius+Dynamo).
- **llama.cpp** — GGUF, CPU-SIMD (AVX/AVX2/AVX512/**AMX**/**NEON**/RVV) + Metal/CUDA/Vulkan;
  **MTP** влито в main; **Paged KV лише opt-in** (PR#22569: 247 vs 26 послідов. paged-vs-unified);
  `--n-cpu-moe` — **layer-uniform** offload, **без router-aware кешування** (FR#20757 відкритий).
- **ktransformers / ik_llama.cpp** — ніша «big-MoE > VRAM на потребительському залізі»:
  **router-aware hot-expert cache** (kt) — 1.5–2× TPS над llama.cpp при ~половині VRAM
  (DeepSeek-V2: 9 TPS@45%VRAM vs 7.5@95%); ik_llama.cpp додає IQ_* кванти + fused-MoE.

## 3. Чому MoE важкий (загальний урок для нашого плану)

- Ефективність MoE-декоду нижча за dense: Qwen3-30B-A3B (3.3B active) на A100 — **~46%**
  теоретичної пропускної здатності проти **63%** у dense FP16; AWQ-Marlin-MoE падає до
  **11–18%** через сумарні витрати dequant + expert-loading. → наш **native-Rust** шлях (PH-401..406)
  має битися саме за це: router-overhead і деquant-на-льоту; тому й робимо fused kernels (PH-201+/B2).
- «MoE-throughput ∝ active-параметри» — підтверджує ставку на A3B-клас (Qwen3-30B-A3B) як бенчмарк-підлога.
- Префікс-кешинг (RadixAttention) — великий виграш для агентських/багатоточкових навантажень;
  наш `PagedKvCache` (PH-303) — база; radix/prefix-Reuse → майбутній band.

## 4. Ризики / уроки (власна чесність)

- Реальний `PH-305` multi-layer forward ще **не верифікований числово** — потрібен референс
  (llama.cpp/SGLang) → gate `PH-405`. Поки `PH-305a/c/d/e` дають **геометрію + деquant +
  single-step** (усе green + real-Qwen2.5 `rusty validate` = OK).
- Контрибути llama.cpp-семантики змінюються (Q5_K тепер ділить 6-bit layout Q4_K; `block_q5_0`
  порядок полів `{d,qh,qs}` vs старий `{d,qs,qh}`; фінальна норма qwen2 = `output_norm.weight`) —
  тож **кешуємо фактаж у `docs/research/`** і тримаємо регресійні тести.
- Не претендуємо на GPU-cluster-throughput проти TRT-LLM/SGLang — це інша ліга й інша апаратна
  модель; наша диференціація = pure-Rust + edge/Android + GSV-fleet.

## 5. Джерела (веб, 2026)

- mlai.qa — *vLLM vs SGLang vs TensorRT-LLM 2026* (реleases/stars/governance/MLPerf v6.0).
- masternodeai — *Model serving runtimes for MoE* (2.5× SGLang, W4A16, EP-таблиці).
- inferenceengineering.tech — *vLLM vs SGLang vs TRT-LLM* (feature matrix, HW breadth).
- ree2raz/inference-bench — *vLLM vs SGLang vs llama.cpp* (Qwen3-30B-A3B MoE: 46% vs 63% eff.).
- stefanopineda.com — *Three engines, three memory models* (KV: pages/tree/pool; PR#22569, #4130).
- noonghunha/club-3090 `docs/INFERENCE_ENGINES.md` — +ktransformers/ik_llama.cpp (router-aware cache, #20757).
- Локально: `S:/rust/llama-rs` (GGUF/staged/MTP/OpenAI server, MoE у C++), `S:/rust/poolAI` (FM edge job-lease — генетична база для GSV-fleet).
