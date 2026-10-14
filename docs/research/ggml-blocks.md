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

## Q5_0 (PH-201c, `src/quant.rs dequant_q5_0`)

`block_q5_0` = **22 B** / 32 elem, current-master order **`[d@0(f16), qh@2(u32 LE 5th-bits), qs@6(16)]`**.
`value = d * ( (nibble | (qh_bit << 4)) - 16 )` — bias **−16** (range −16..15), **не −8**.
qh bit index == element index (low nibble qs[j]→elem j uses qh bit j; high→elem j+16 uses qh bit j+16).
> CAVEAT: pre-refactor GGUFs = `[d, qs, qh]`. Реальна Qwen2.5 деquantиться до малих finite
> embeddings + self-consistent offsets → master `{d,qh,qs}` вірна для неї; при garbage у файлі —
> спробувати старий порядок.

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

## K-quant scalar dequant (референс `dequantize_row_q{4,5,6}_K`, ggml-quants.c)

Підтверджено (subagent, llama.cpp-мастер): **Q5_K спільний** 6-bit layout з Q4_K
(`get_scale_min_k4`) — НЕМАЄ `get_scale_min_k5`; **Q6_K БЕЗ lookup-таблиці**
(помиłka про `{32,32,32,32,...}` — її нема); `scales` Q6_K = signed `int8[16]`.

```c
// 6-bit scale/min extraction (Q4_K + Q5_K shared). q = scales[12]. j = 0..7.
get_scale_min_k4(j, q, &d, &m):
  if j<4: d = q[j]&63;        m = q[j+4]&63;
  else:   d = (q[j+4]&0xF)|((q[j-4]>>6)<<4);  m = (q[j+4]>>4)|((q[j]>>6)<<4);

// Q4_K (144B): d@0,dmin@2,scales@4(12),qs@16(128). 4 groups (j+=64), is+=2, base=is*16 bytes:
//   y[l]      = d*sc0 * (qs[base+l]&0xF) - dmin*m0     // low nibbles,  0..31
//   y[l+32]   = d*sc1 * (qs[base+l]>>4)  - dmin*m1     // high nibbles, 32..63

// Q5_K (176B): d@0,dmin@2,scales@4(12),qh@16(32),qs@48(128). masks u1=1<<(2*grp),u2=2<<(2*grp):
//   y[l]    = d*sc0*((qs[base+l]&0xF)+(qh[l]&u1?16:0)) - dmin*m0
//   y[l+32] = d*sc1*((qs[base+l]>>4)+(qh[l]&u2?16:0))  - dmin*m1

// Q6_K (210B): ql@0(128),qh@128(64),scales@192(16 i8),d@208. 2 halves (n+=128):
//   per l in 0..32, is=l/16, q = (ql.. &0xF or >>4) | (qh 2bits <<4) - 32   // signed [-32..31]
//   y[l]=d*sc[sb+is+0]*q1; y[l+32]=...*sc[sb+is+2]*q2; y[l+64]=..*sc[+4]*q3; y[l+96]=..*sc[+6]*q4
//   q1 low-nibble(l), q2 low-nibble(l+32), q3 high-nibble(l), q4 high-nibble(l+32);
//   qh bits: q1<<0 &3, q2>>2 &3, q3>>4 &3, q4>>6 &3
```
**Пастки:** signedness — Q6_K `q` та `sc` СІГНОВАНІ (−32 bias, `sc` i8), Q4_K/Q5_K `sc/m`
без знаку (0..63) і `q` без знаку (0..15 / 0..31), `min` віднімається; low-nibble → перші
32, high-nibble → наступні 32 в межах 64-групи. Реалізовано: `src/quant.rs`
`dequant_q4_k`/`dequant_q5_k`/`dequant_q6_k` (+ `get_scale_min_k4`), PH-202.

> SIMD (`PH-203`) + repack (`PH-204`) мають відтворювати ЦЕЙ скалярний oracle;
> числовий parity проти llama.cpp — на реальній моделі, `PH-405`.
