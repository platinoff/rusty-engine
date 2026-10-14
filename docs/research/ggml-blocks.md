# GGML quant block layout — local reference (cached research)

Джерело: `ggml/src/ggml-common.h` + `ggml-quants.c` (через explore-subagent 2026-10-14).
Кешуємо, щоб `PH-104`/`B2` не перезбирали llama.cpp-семантику. Перевіряти parity з
реальним фікстуром (B4 `PH-405`), не лише synth.

## Табло блоків (size для offset-математики tensor bytes)

| ggml # | тип | elements/block | bytes/block | bpw |
|--------|-----|----------------|-------------|-----|
| 0 | F32 | 1 | 4 | 32 |
| 1 | F16 | 1 | 2 | 16 |
| 2 | Q4_0 | 32 | **18** | 4.5 |
| 8 | Q8_0 | 32 | **34** | 8.5 |
| 12 | Q4_K | 256 | **144** | 4.5 |
| 13 | Q5_K | 256 | **176** | 5.5 |
| 14 | Q6_K | 256 | **210** | 6.5625 |
| 24 | BF16 | 1 | 2 | 16 |

`tensor_bytes = numel / QK * block_bytes` (перевірити `numel % QK == 0`).

## Q4_0 (реалізовано у `src/quant.rs`)

```c
struct block_q4_0 { ggml_half d; uint8_t qs[16]; }   // 2 + 16 = 18
// dequant (С): for byte j in 0..15:
//   x0 = (qs[j] & 0x0F) - 8;  y[j]      = x0 * d;   // LOW nibble  -> element j      (0..15)
//   x1 = (qs[j] >>   4) - 8;  y[j + 16] = x1 * d;   // HIGH nibble -> element j + 16 (16..31)
```
> ⚠ SPLIT-HALF, НЕ 2i/2i+1. Низький нібл → перша половина, високий → друга.

## Q8_0 (реалізовано)

```c
struct block_q8_0 { ggml_half d; int8_t qs[32]; }     // 2 + 32 = 34
// y[j] = qs[j] * d   (qs signed int8)
```

## K-кванти — розміри відомі, dequant → B2 (`src/core/kernel/*`)

- Q4_K: {half d @0, half dmin @2, u8 scales[12] @4, u8 qs[128] @16} = 144. 8 sub-blocks×32,
  scales 6-bit (d + dmin, super-block scale).
- Q5_K: {half d, dmin, u8 scales[12] @4, u8 qh[32] @16 (5th bit), u8 qs[128] @48} = 176.
- Q6_K: {u8 ql[128] @0, u8 qh[64] @128 (upper 2 bits), i8 scales[16] @192, half d @208} = 210.

## FP16 (scale) семантика

`ggml_half` = `uint16_t` little-endian, IEEE-754 binary16 (sign1 / exp8 bias15 / mant10).
У Rust: читати `blk[0..1]` як `u16::from_le_bytes` → `f16_to_f32()` (реалізовано через f64:
exp==0 subnormal `frac/1024*2^-14`, exp==31 → inf/nan, інакше `(1+frac/1024)*2^(exp-15)`).

## GGUF tokenizer keys (для `PH-105`, `src/tokenizer.rs`)

| key | GGUF type |
|-----|-----------|
| tokenizer.ggml.model | str (`gpt2` BPE / `llama` SPM) |
| tokenizer.ggml.tokens | array[str] |
| tokenizer.ggml.token_type | array[i32] (0 undef,1 NORMAL,2 UNKNOWN,3 CONTROL,4 USER_DEF) |
| tokenizer.ggml.scores | array[f32] |
| tokenizer.ggml.merges | array[str] (`"tokA tokB"`) |
| tokenizer.ggml.bos_token_id / eos_token_id / padding_token_id / unknown_token_id | u32 |
| tokenizer.ggml.add_bos_token / add_eos_token / add_space_prefix / remove_extra_whitespaces | bool |
| tokenizer.ggml.pre | str (pre-tokenizer regex: gpt2/llama3/qwen2/...) |

> `byte_fallback`/`byte_to_char`/`utf8_min_byte` — НЕ ключі GGUF; llama.cpp виводить з
> vocab (token_type==BYTE). PH-105 = **ingest метаданих** у `TokenizerMeta`, не сам BPE-алгоритм.
