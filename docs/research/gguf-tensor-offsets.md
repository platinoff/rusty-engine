# GGUF tensor byte-offset semantics — cached research (PH-305a)

Джерело: `ggml/src/gguf.cpp`, `ggml/src/ggml.c`, `src/llama/llama-model-loader.{h,cpp}`
(explore-subagent 2026-10-14). Кеш — щоб `PH-305` multi-layer не перезбирав.

## Правило (як читає llama.cpp)

- `general.alignment` default **32** (`GGUF_DEFAULT_ALIGNMENT`), power-of-two інакше load fails.
- **data_start** (абсолютна позиція даних) = `pad_up(metadata_end - gguf_begin, alignment)`
  (для standalone `.gguf` `gguf_begin=0` → `pad_up(header_len, align)`). Pad лише коли `n_tensors>0`.
- Поле `offset` у tensor-info = **byte delta від data_start, УЖЕ ×alignment**. НЕ елементи,
  НЕ ділити на type_size. `start = data_start + offset`.
- `nbytes` = `ggml_nbytes` = `numel / QK(type) * block_bytes(type)` = наш `bytes_for_numel`
  (підтверджено субагентом; GGUF гарантує `ne[0] % QK == 0`).
- Bounds (як loader): `start + nbytes <= file_len` і без переповнення.

```
pad_up(x, a) = (x + a - 1) & ~(a - 1)          # GGML_PAD
for t in tensors:                               # stored order == ascending offset
    start  = data_start + t.offset
    nbytes = bytes_for_numel(t.ggml_type, numel(t.dims))
    assert start + nbytes <= file_len
    range = [start, start + nbytes)
```

`ggml_type_size` = block_bytes (напр. Q4_0=18, Q8_0=34, Q4_K=144, Q5_K=176, Q6_K=210);
`ggml_blck_size` = QK (32 / 32 / 256 / 256 / 256). Реалізація: `src/loader.rs`
`MmapModel::tensor_bytes` (довіра stored offset), `src/gguf.rs` `TensorEntry::nbytes`.

## Перевірено на реальній моделі (Qwen2.5-0.5B Q4_K_M)

`data_start = 5 947 744` (= PAD(header_len,32)); `blk.0.ffn_down.weight` offset=238 239 232,
numel=4 358 144 → nbytes 3 575 040 (=4358144/256·210) → **dequant_q6_k → 4 358 144 finite
значень** (магітуди ~0.006–0.028 = справжні FFN-ваги). `blk.0.attn_norm.weight` F32 → 896 float.

## Далі (PH-305 multi-layer loop)

Порядок шару: `token_embd` → для кожного `blk.i`: rmsnorm(`blk.i.attn_norm`) → attn
(q/k/v/o = `blk.i.attn_*.weight`, GQA via `general.attention.head_count{,_kv}`, RoPE
`general.rope.freq_base`) + residual → rmsnorm(`ffn_norm`) → MoE/dense FFN (`blk.i.ffn_*`,
для MoE: `blk.i.ffn_gate_exps/up_exps/down_exps` + `blk.i.ffn_gate_shexp/ffn_up_shexp/ffn_down_shexp`
shared + `blk.i.ffn_gate.weight` router) + residual → фінальний norm → `output.weight` → logits.
+ BPE-decode (`tokenizer.ggml.pre`) → токен-текст.
