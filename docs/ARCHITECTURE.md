# rusty-engine — ARCHITECTURE

**Product id:** `rusty-engine` · **Root** `S:/rust/rusty-engine` · Rust-first plugin.
Canon-концепція: [`concept.md`](./concept.md) · смуги: [`roadmap.md`](./roadmap.md).

## 1. Principle: the MoE kernel is Rust, not C++

llama-rs (sibling plugin) runs GGUF MoE today, but `router → top-k gating → per-expert
FFN` is computed **inside llama.cpp** (`llama.cpp/src/models/qwen3moe.cpp`). rusty-engine
lifts that whole path into Rust so we can extend it (shared experts, noaux_tc group
gating, NVMe/RAM expert offload, cross-node expert-parallel) without a C++ backend. We
**port llama.cpp methods**, we do not link them.

## 2. Module map (B0 crate graph)

Single crate `rusty-engine` (lib `rusty_engine`) + `rusty` bin + `xtask`; workspace with
`members = ["xtask"]`. A crate-per-layer split is a B2+ decision, deferred until kernels
need their own feature gates.

| Module | Responsibility | Band |
|--------|----------------|------|
| `core` | `DType` (F32/F16/Q4_0..Q8_0), `TensorInfo`, `EngineError`/`Result` | B0→B2 |
| `gguf` | GGUF v3 header + KV-metadata + tensor-info; mmap/mlock staged loader | B1 |
| `moe` | **DIFFERENTIATOR**: softmax, top-k gating, expert FFN, shared experts | B0→B4 |
| `infer` | attention (GQA/MQA + RoPE), paged KV-cache, sampler chain | B3 |
| `serve` | OpenAI-compat HTTP (`/v1/models`, `/v1/chat/completions`), SSE stream | B5 |
| `node` | slave/worker runtime over local Wi-Fi/LAN | B7/B8 |
| `hub` | GSV-native master supervisor («Rust k8s-lite»): registry, heartbeat, placement | B8 |

```
GGUF file
  └─ gguf(mmap) ── core(dequant/repack) ── infer(attention+KV)
                                              └─ moe(router + experts) ── serve(stream)

Orchestration:
  hub (master = GSV :9999) ⇄ node (slave)   over LAN/Wi-Fi
       └─ register/heartbeat/lease/placement ─► expert-parallel sharding (B9)
```

## 3. Quantized kernel strategy (B2)

1. scalar reference kernels (`dot`, `gemv`) for each GGUF block format — ground truth;
2. `repack` to compute-friendly layout;
3. SIMD backends: SSE2/AVX/AVX2 (x86_64) + NEON (arm64), runtime feature dispatch;
4. fused dequant+dot inner loops; benches vs scalar + vs llama-rs baseline.

## 4. MoE forward (B4)

- router logits → gating: softmax-renorm (Mixtral), **sigmoid + noaux_tc** group top-k
  (DeepSeek-V3 / Qwen3), optional shared expert added outside the routing;
- per-expert SwiGLU FFN over `blk.N.ffn_{gate,up,down}_exps` tensors;
- expert cache (LRU) + hot-expert prefetch (PowerInfer-style) + RAM/NVMe offload
  (KTransformers-style) for 30B-A3B-class models on consumer RAM.

## 5. Portability targets (B6)

| Target | Artifact | Route |
|--------|----------|-------|
| Windows x86_64 | `rusty.exe` | gnu (primary) + msvc (`LIBCLANG_PATH`) |
| Linux x86_64 | `rusty` | native |
| Linux aarch64 | `rusty` | cross / native on Pi-class |
| Android arm64 | `rusty.apk` / `librusty.so` | cargo-ndk / Termux — **no Docker** |

`cargo xtask build-matrix` emits one artifact per target + SHA-256.

## 6. Docker + fleet (B7/B8)

- Rust docker client (**bollard**) builds/runs the slave image on **Win/Linux** hosts
  only, exposed on the LAN address (GSV band-233 LAN-first), never loopback-forced.
- **Android slave runs the same Rust runtime as `.apk`/`.bin`, no Docker** (owner decision 5).
- GSV hub = **master**: worker registry, 15s heartbeat (llama-rs keep_live cadence),
  leases + reclaim (GSV ticket-lease model), model placement, fail-open watchdog.
  Reuses poolAI **ed25519 job-lease** signing + `gsv_grid` ALLBGP mirror shape.
  This is the «Rust kubernetes» — homegrown master/slave supervisor, not k3s/Go.

## 7. UI (B10) — see `docs/UI.md`

Rust renders every page (`src/ui.rs`: `/page/{name}` + `GET /api/ui/card/{name}`);
`ui/js/app.js` is chrome ONLY (theme, sidebar, hotkeys, `?` help panel, no innerHTML).
Actions are plain GET/POST forms. Ratio ratchet stays ≥ 0.95.

## 8. GSV interaction (master hub)

- keep-live: `rusty serve` writes `target/live/rusty_heartbeat.json` under `GSV_LIVE=1`
  (mirrors llama-rs band-225 contract, so GSV `keep_live` can read it).
- fleet + tickets + alerts via the hub `/mcp` + `/api/edge`; plugin never opens a second
  `:9999` and never ships GSV kit (see plugin `AGENTS.md`).
