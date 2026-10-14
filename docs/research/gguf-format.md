# GGUF format — local reference (cached research)

Закешований фактаж, щоб наступна agi-сесія **не перезбирав контекст** з llama.cpp/вебу.
Канон джерела: `ggml/gguf.cpp` + `ggml-common.h` (GGML_TYPE_*). Перевіряти обмін-сумісність
проти **реального** фікстуру в `PH-106`.

## Header layout (little-endian)

| Поле | v1/v2 | v3 |
|------|-------|----|
| magic | `GGUF` 4B | те саме |
| version | u32 | u32 |
| n_tensors | u32 | **u64** |
| n_kv | u32 | **u64** |
| KV × n_kv | `string key` + `u32 type` + value | — |
| tensor-info × n_tensors | name, n_dims(u32), dims**(v3: u64)**, ggml_type(u32), offset(u64) | — |

String = `u64 len` + UTF-8 bytes. `data_offset` = позиція після останньої tensor-info;
tensor data вирівняна на `general.alignment` (u32, дефолт **32**).

## GGUF_TYPE_* (value kind)

`0 U8 · 1 I8 · 2 U16 · 3 I16 · 4 U32 · 5 I32 · 6 F32 · 7 Bool(u8) · 8 String · 9 Array · 10 U64 · 11 I64 · 12 F64`
Array = `u32 elem_type` + `u64 count` + count·value (elem може бути Array → вкладено).

## GGML_TYPE_* (tensor dtype) → `core::DType` map (PH-102)

| code | GGML | → DType |
|------|------|---------|
| 0 | F32 | F32 |
| 1 | F16 | F16 |
| 2 | Q4_0 | Q4_0 |
| 8 | Q8_0 | Q8_0 |
| 14 | Q4_K | Q4_K |
| 15 | Q5_K | Q5_K |
| 16 | Q6_K | Q6_K |
| 24 | BF16 | BF16 |
| 12/13 | Q2_K/Q3_K | (не в DType → `dtype=None`, лишаємо `ggml_type` raw) |
| 18+ | IQ* | (B2) |

**Не-відомі коди:** НЕ панікувати — `ggml_dtype()` → `Option::None`, raw `ggml_type` збережено.

## Ключі, що читатимемо далі

- `general.architecture` (str) — arch-діспатч (`qwen2moe`, `qwen3moe`, `mixtral`, `deepseek2`).
- `general.block_count`, `general.embedding_length`, `general.attention.head_count`,
  `general.attention.head_count_kv` (GQA) — PH-4xx MoE-ядро.
- MoE: `general.expert_count`, `general.expert_used_count` (top-k), `general.moe_layer_count`.
- Tokenizer (`PH-105`): `tokenizer.ggml.model`(str `gpt2`), `tokenizer.ggml.tokens`(arr),
  `tokenizer.ggml.scores`(arr f32), `tokenizer.ggml.token_type`(arr i32), `tokenizer.ggml.merges`(arr).

## Block formats (PH-104 sizes) — для перевірки offset-математики

| type | block elems | block bytes |
|------|-------------|-------------|
| Q4_0 | 32 | 2 + 16 |
| Q8_0 | 32 | 2 + 32 |
| Q4_K | 256 | 2 + 2 + 12 + 128 = 144 |
| Q5_K | 256 | 2 + 2 + 12 + 128 + 64 = ... (перевірити в ggml-common.h) |
| Q6_K | 256 | 2 + 256 + 128 + 64 (перевірити) |

> TODO(PH-104): точні байт-сайзи Q*_K перечитати з `ggml-common.h`, бо вони визначають
> обчислення `data_start + Σ size(tensor)`. Зараз парсимо ЛИШЕ header+info, data не чіпаємо.
