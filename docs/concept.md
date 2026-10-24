# rusty-engine — концепція проєкту

**Status:** Draft v0.1 (owner 2026-10-14 кампанія «унікальний MoE-двигун»)
**Product id:** `rusty-engine` (crate `rusty-engine`, lib `rusty_engine`, bin `rusty`)
**Root:** `S:/rust/rusty-engine`
**Nature:** Rust-first portable plugin під GSV hub (host = `S:/rust/GSV`, `:9999`).
**License:** MIT (`LICENSE`, `Copyright (c) 2026 platinoff / Madevinc`).

---

## 1. Vision (одне речення)

Єдиний **pure-Rust** двигун виводу **Mixture-of-Experts** моделей (GGUF-first), що
збирається в **один портативний бінарник** (`rusty.exe` / `rusty` / `rusty.apk`) під
amd64/arm64, розміщує MoE на локальних Wi-Fi-хостах (Windows·Linux через Rust-docker,
Android через cargo-ndk/Termux) і керується **GSV-native Rust-супервізором** (hub =
master, пристрої = slaves) — вбираючи найкращі методи llama.cpp, llama-rs, vLLM,
SGLang, KTransformers і MixtureServe, але **без C++-ядра та без Go k8s**.

## 2. Клієнт і ринок

- Власники локальних MoE-моделей (Qwen3-MoE, Mixtral, DeepSeek-V3, GPT-OSS), які хочуть
  інференс на **своєму** залізі: десктоп → кластер → телефон.
- Edge/IoT-оператори: аarch64-бокси й Android як розподілені expert-вузли.
- Ком'юніті rust-first LLM (llama-rs) — відсутній власний Rust-MoE-шлях.

## 3. Модель доходу та економіка

Відкритий MIT-двигун (безкоштовно), монетизація через **сервісний шар GSV-хаба**
(fleet-оркестрація, телеметрія, алерти) та підтримку портативних дистр-каналів.
Non-goal: збирати власну хмару.

## 4. Функціонал (Map of the engine)

- **4.1 GGUF-чанкер (pure Rust).** Header v3 + KV-metadata, tensor-info, mmap/mlock
  staged loader, квант-блоки Q4_0/Q4_K/Q5_K/Q6_K/Q8_0.
- **4.2 Quantized kernels + SIMD.** скалярний референс → repack → SSE/AVX/NEON, runtime
  dispatch заfeatures.
- **4.3 Dense forward (control path).** RMSNorm/SiLU/MatMul, causal attention GQA/MQA,
  RoPE, paged KV-cache, sampler chain.
- **4.4 MoE-ядро (ДИФЕРЕНЦІЯТОР).** top-k router (softmax / sigmoid / noaux_tc
  group-gating), expert-FFN SwiGLU, shared-expert merge, expert-cache + NVMe/RAM
  offload, hot-expert prefetch. Усе в Rust, не в llama.cpp.
- **4.5 Single-node serve.** `rusty serve` — OpenAI-compat `/v1/models`,
  `/v1/chat/completions`, SSE-стримінг, prompt-cache, метрики tok/s·RAM·активні experts.
- **4.6 Build matrix.** `rusty.exe` (win gnu+msvc), `rusty` (linux amd64/arm64),
  `rusty.apk/.so` (android arm64 cargo-ndk) — `cargo xtask build-matrix`.
- **4.7 Rust-docker на хостах.** Win/Linux-хости: Rust-docker-клієнт (bollard) піднімає
  slave-образ на localhost, exposing через локальний Wi-Fi/LAN. Android — без Docker.
- **4.8 GSV-native supervisor («Rust k8s-lite»).** master = GSV-hаб `:9999`; slave
  реєстрація + heartbeat + lease + placement по моделі; fail-open при падінні вузла.
- **4.9 Розподілений MoE.** expert-parallel через воркерів (MixtureServe/MoE-Exchange),
  token-routing до expert-вузла, MTP/speculative decoding cross-node.
- **4.10 UI (Design System).** Rust-render сторінки (`src/ui.rs` `/page/{name}` +
  `GET /api/ui/card/{name}`), JS = лише chrome (тема/сайдбар/хоткеї/`?`). Див. `docs/UI.md`.
- **4.11 Distribs + версії + publish.** semver, CHANGELOG, portable tar.gz/zip/apk +
  checksums, update-channel, THIRD-PARTY-NOTICES, GitHub (private→MIT).

## 5. Потік даних

`GGUF file → gguf(mmap) → core(dequant/repack) → infer(attention+KV) → moe(router+experts) →
serve(stream)`. У оркестрації: `hub(master, GSV) ⇄ node(slave)` по LAN/Wi-Fi, placement
leases → expert-паралелізм.

## 6. Технічні специфікації

- **6.1** rust-toolchain pin `1.92.0-x86_64-pc-windows-gnu`; MSYS2 bash; shell ≠ PowerShell.
- **6.2** Ratio: Rust **95–100%**; `ui/` — тонкий glue; plugin-гейт тестами (LOC-ratchet +
  UI-інваріанти + scaffold-contracts).
- **6.3** Мережевий код (docker/orchestrator) — dry-run за замовчуванням, `--live` за
  явною згодою owner. Токени в `data/` (gitignored), ніколи в коді/логах.
- **6.4** Жодного C++-ядра; llama.cpp-методи **переносимо** в Rust, а не лінкуємо.

## 7. Product-Market fit

llama-rs вже біжить GGUF-MoE, але **роутер/expert — у C++ llama.cpp**. `rusty-engine`
дає **той самий GGUF-MoE у чистому Rust** + розподіл expert по локальних Wi-Fi-вузлах
через GSV-хаб — ніхто з існуючих двигунів не дає ані Rust-MoE-ядра, ні «Rust-k8s-lite»
для експертного паралелізму на edge.

## 8. Ризики

- Productive-шwindigkeit: quant-kernels + MoE — найважча частина (B2/B4); мітимо
  llama-rs-бенчмарк як підлога (Qwen3-30B-A3B ≈ 0.497 tok/s decode).
- Android/Docker: Docker на Android відсутній → свідомо «без Docker», cargo-ndk/Termux.
- «Rust k8s»: літеральний k8s (Go) — нe-Rust → робимо власний supervisor (вирішення owner 3).

## 9. Non-goals

- Не копіюємо llama.cpp як залежність; не лінкуємо C++.
- Не піднімаємо другий `:9999` / інший MCP-hub; не пишемо heavy-JS рендер.
- Не збираємо власну хмару і не публікуємо моделі/токени.

## 10. Джерела (методи, які інкорпорують)

llama.cpp (GGUF quants, MoE routing), llama-rs (staged loader, MTP, OpenAI server,
GSV heartbeat), vLLM (paged attention, continuous batching), SGLang (RadixAttention /
KV-reuse), KTransformers (CPU/GPU expert offload), MixtureServe / MoE-Exchange
(expert-parallel), PowerInfer (activation-sparsity expert prefetch).

## 11. Вирішення owner (2026-10-14)

1. Окреме portable-plugin git-репо `S:/rust/rusty-engine` (MIT).
2. Crate `rusty-engine`, lib `rusty_engine`, бінарники `rusty` (exe/bin/apk).
3. Оркестрація = **GSV-native Rust-супервізор** (master/slave), не k3s.
4. Формат моделей = **GGUF-first** (safetensors — пізніше, опційно).
5. Android — **без Docker** (cargo-ndk/Termux `.apk/.bin` як slave); Docker лише Win/Linux.

## 12. Вирішення owner (2026-10-24) — цільова модель

1. Цільова модель двигуна = **MiniCPM-V 4.7** (1B dense + 35B-A3B MoE), official-GGUF
   з HF (`openbmb/MiniCPM-V-4.7-*-gguf`, Apache-2.0 і на ваги, і на код) — **без Python
   у рантаймі**; їхній репозитарій — лише джерело ваг і специфікації архітектури.
2. **Інші модельні цілі з роадмепу прибрано** (Qwen3-30B-A3B / Mixtral — не підтримуємо
   як продуктові моделі). Qwen2.5-0.5B лишається **внутрішнім числовим оракулом**
   (dev-фікстура, golden `PH-405`, llama-rs-оракул — інструмент, не продукт).
3. Віжен-шлях реалізується у Rust: `src/vit.rs` (vision tower) + 4×/16× visual-token
   compressor + projector + image-prefix у prefill — бенд **B12** у `docs/roadmap.md`.
4. Чисельний оракул для vision-фіксайтів — `llama.cpp-omni` (mtmd) CLI як
   вимірювальний інструмент; до продукту не входить.
