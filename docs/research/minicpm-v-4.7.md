# MiniCPM-V 4.7 — GGUF layout spike (PH-1201, 2026-10-24)

Цільова модель двигуна (owner 2026-10-24). Офіційні ваги: HF
`openbmb/MiniCPM-V-4.7-1B-gguf` (Apache-2.0; код OpenBMB — не форкається, Python у
рантаймі не використовується). Spike міряв `rusty inspect` (з новим повним KV-дампом +
prefix-histogram + `[N]` tensor list) на двох файлах у `data/models/` (gitignored):

| Файл | Байтів | Роль |
|---|---|---|
| `MiniCPM-V-4_7-1B-Q4_K_M.gguf` | 672 003 680 | text core, arch **`qwen35`** |
| `mmproj-model-f16.gguf` | 1 110 117 984 | vision, arch **`clip`**, type `mmproj`, 548M, **усі ваги BF16 (ggml=30)** |

## 1. Header LLM: qwen35 — ГІБРИД, не qwen2

24 шари, d_model **1024**, vocab **248 094**, ctx **262 144**, rms_eps 1e-6,
rope base 1e7. Ключі `qwen35.*`:

```
attention.head_count = 8, head_count_kv = 2
attention.key_length = value_length = 256
rope.dimension_sections = [11, 11, 10, 0]   rope.section_order = "zyxt"
rope.dimension_count = 64
full_attention_interval = 4,  attention.recurrent_layers = Bool[24]
ssm.conv_kernel = 4, state_size = 128, group_count = 16,
time_step_rank = 16, inner_size = 2048
tokenizer.ggml.pre = "qwen35"  (gpt2-подібний pre, НОВИЙ regex)
bos 248045 / eos 248046 / unk 248077 / pad 248044, add_bos=false
chat_template: 7 249 chars (Jinja, hard-disable thinking)
```

Тензорні схеми по шарах (321 tensor, 18+6=24):

- **18 GDN-шарів** (рекурентні, 3 з кожного 4-го): `blk.N.attn_qkv` [1024×6144] (q/k/v
  mixed 3×2048, Q6_K), `attn_gate` [1024×2048] (Q4_K), `ssm_a` [16], `ssm_alpha`
  [1024×16], `ssm_beta` [1024×16], `ssm_conv1d` [4×6144] (F32, depthwise kernel 4),
  `ssm_dt.bias` [16], `ssm_norm` [128], `ssm_out` [2048×1024], `attn_norm`,
  `post_attention_norm`, ffn трійка.
- **6 full-attention-шарів** (idx 3,7,11,15,19,23): класичний роздільний `attn_q`
  [1024×4096] (2048 q + 2048 gate), `attn_k` [1024×512], `attn_v` [1024×512],
  **`attn_q_norm`/`attn_k_norm` [256]** (QK-norm, head_dim 256), `attn_output`
  [2048×1024], ffn трійка. Немає `ssm_*`.
- FFN усіх шарів: `ffn_gate/up` [1024×3584] Q4_K + `ffn_down` [3584×1024]
  (Q6_K у GDN-шарів, Q4_K у full — квант змішаний).
- `token_embd` Q4_K / `output` Q6_K / `output_norm` F32.

Тобто text core = **Qwen3-Next-подібний hybrid (Gated DeltaNet linear attention +
повноувагова кожна 4-та шар)**. Існуючий qwen2-шлях (`forward.rs`, `model.rs`) його
НЕ покриває — це новий бенд **B13** (див. §3).

> **PH-1301 (того ж дня): `rusty validate data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf` = OK.**
> 24 шари, 6 full (blk.3 включно), всі dims збігаються з config-формулами
> (`model.rs`: `LayerKind::{Recurrent, Full}`, qkv 3·inner, attn_q 2·q). Додатковий
> факт із валідації: **`ffn_norm.weight` у qwen35 немає** — перед-FFN норму грає
> `post_attention_norm.weight` (дві норми на блок); гибридна гілка `validate` саме це
> й вимагає.

## 2. Vision mmproj: clip, 548M, повний BF16

```
clip.vision: image_size 448, patch 14, embed 1152, ffn 4304, blocks 27,
  heads 16, GELU, LN eps 1e-6, mean/std 0.5, projection_dim 1024
clip.projector_type = "minicpmv4_7", scale_factor = 4, max_slice_nums = 9,
  wa_layer_indexes = [6]
460 tensors: v.blk.0..26 (16/блок: attn_q/k/v/out + ffn_up/down + ln1/ln2, weight
  BF16 + bias F32) · v.patch_embd [14,14,3→1152] · v.position_embd [1152×4900]
  (2D sin-cos table 70×70 — hi-res інтерполяція) · v.tok_embd_sep [1024×4]
  · v.post_ln · v.vit_merger.{ln1, attn q/k/v/out, ds_ln [4608],
  ds_ffn_up 4608→17216, ds_ffn_down 17216→1152} (це 4×/16× token-компресор)
  · mm.{input_norm, up 4608→4608, down 4608→1024} — фінальний MLP-projection.
```

**Глюк:** `GGML_TYPE_BF16 = 30` (канон ggml.h: I8=24, IQ1_M=29, BF16=30). Таблиця
rusty-mape мала хибний `24 → BF16` — виправлено цим зрізом (`gguf::ggml_dtype`,
`quant::block_info`, `dequant` arm 30: bf16 = верхні 16 біт f32, widening = `<< 16`;
тест `dequant_bf16_widens_by_left_shift`). Усі vision-ваги — 2-байтові без блоків →
ViT-матмоди йдуть поверх існуючих f32-ядер після дейкванта.

## 3. Що це змінює в плані (B12/B13)

1. **B13 (новий бенд, текст): qwen35-hybrid core** — без нього vision не має куди
   вставляти токени. Слайси: config/geometry+validate (`model.rs`), GDN-стейт декод
   (conv1d k4 + delta-rule, O(1)/крок — наш диференціатор лінійної уваги), full-attn
   шари (QK-norm, gate, zyxt-sectioned rope), qwen35 pre-tokenizer, greedy-паритет
   text-шляху проти `llama.cpp-omni` CLI (golden-ораккул, як llama-rs для qwen2).
2. **B12 (vision, mmproj)** — переформульовано під реальний layout: PH-1202 ViT 27×
   (attn 16 heads + GELU MLP, BF16-dequant), PH-1203 **vit_merger + mm MLP**
   (scale_factor 4, slice-nums 9, wa_layer 6), PH-1204 image-prefix чекає B13-core.
3. Перф-профіль 1B: 24 шари × 1024 ≈ половина qwen2-0.5B на повноувагових шарах,
   18 GDN-шарів — рекурентний стан 16 груп × 128×128 → декод лінійний, prefill
   паралельний — rusty-ядра (QMat/matmul/AVX2) покривають обидві форми.
4. 35B-A3B варіант (MoE) — окремий spike пізніше (PH-1206 тримаємо); його text core
   теж qwen35-родини (LES-планування expert-load — див. llama.cpp-omni).

Файли: `data/models/{MiniCPM-V-4_7-1B-Q4_K_M,mmproj-model-f16}.gguf` (gitignored),
повні дампи — `target/live/inspect_ph1201_{main,mmproj}_full.txt`.
