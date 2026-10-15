# llama-rs parity — PH-405b facts (live oracle)

**Product:** `rusty-engine` · зріз: 2026-10-15 (agi-сесія, тікет `PH-405b`).
Оракул: **llama-rs** `llama_serve` на тій самій моделі. Мета: семантичний
greedy-parity токен-стрімів + **byte-exact** перевірка tokenizer з HF
`tokenizer.json`. Цей файл — тільки зафіксовані факти; код гейтів —
`tests/moe_parity.rs`.

## Спільний вхід

- Модель: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, sha256 `74a4da8c9fdbcd15bd1f6d01…d7a9db`
  — **byte-identical** у `rusty-engine/data/models/` і `llama-rs/models/`.
- Оракул: `target/release/llama_serve.exe models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
  --port 8085 --model-name qwen05-oracle` (без `GSV_LIVE`/`LLAMA_RS_HEARTBEAT` —
  спільний heartbeat `target/live/llama_heartbeat.json` не перебивається; живою
  залишається штатна `llama-fast`-ланка :8082 на 1.5B).
- API-контракт: `POST /v1/chat/completions`, raw `{"prompt":…,"temperature":0.0,
  "max_tokens":12}` → greedy-декод у llama.cpp-семантиці; `stream:false`.

## Гейт A — tokenizer: PASS (byte-exact)

- Vocab GGUF = HF `tokenizer.json` за іменами: `You`=2610, `The`=785, `Ġare`=525,
  `Ġa`=264, `Ġhelpful`=10950, `Ġassistant`=17847, `Ġcapital`=6722, `.`=13,
  `Ġ`=220, `Ċ`=198 (10 spots, усе збігається індекс-в-індекс).
- Merges head (перші 8) GGUF ≡ HF: `Ġ Ġ`, `ĠĠ ĠĠ`, `i n`, `Ġ t`, `ĠĠĠĠ ĠĠĠĠ`, `e r`, …
- Pre-tokenizer regex із `tokenizer.json` (`Sequence[Split,ByteLevel(add_prefix_space=false)]`):
  голова `(?i:'s|'t|'re|'ve|'m|'ll|'d)` — як GPT-2, **АЛЕ** цифрова альтернатива —
  одинарний `\p{N}`, НЕ `\p{N}{1,3}`. Знайджено й виправлено в `PH-405b`:
  `src/gpt2.rs` тепер має `qwen2_pre_split` (цифра = окремий шматок) і
  `Bpe::encode` вибирає варіант за `tokenizer.ggml.pre == "qwen2"`.
  До фікса runs цифр ("1234567") групувались по 3 — розбіжність з HF.
- Atomic-id goldens (після фікса): `You are a helpful assistant.` →
  `[2610, 525, 264, 10950, 17847, 13]`; `The capital of France is` →
  `[785, 6722, 315, 9625, 374]`; `2+2=` → `[17, 10, 17, 28]` (байт-простір).

## Гейт B — semantic greedy parity: DIVERGES at the FIRST token (PH-405 open)

Оракул стабільний (повтори тексту 1:1). Наші стріми детерміновані
(`full == incremental` на реальній моделі — новий факт, раніше перевірялось
лише synth). Але top-1 після префіксу — сміття, префікс-збіг = 0:

| prompt | ids | oracle 1-й токен | rusty 1-й токен | lcp |
|--------|-----|------------------|-----------------|-----|
| `The capital of France is` | `785 6722 315 9625 374` | `ĠParis` (12095) | `122007` ("èĹ¿") | 0 |
| `2+2=` | `17 10 17 28` | `4` (19) | `6186` ("izont") | 0 |
| `Berlin is a` | `94409 374 264` | `Ġcity` (3283) | `17528` ("flix") | 0 |

Висновок: **вхід правильний** (токенізація byte-exact, геометри config валідний,
усі 151 936 logits finite), **помилка всередині forward-стека** — не numerical
accumulation, а систематична. Логи аргмаксу поводяться як після невдалої
проєкції (безкаркадне «шумове» argmax).

### Підозрювані (порядок для PH-405-фікса)

1. **Q4_K_M dequant block math** (super-block scales/mins, interleave 32×N) —
   `quant.rs` vs `ggml-common.h` (кеш: `docs/research/ggml-blocks.md`).
2. **RoPE positions у prefill** (`block_forward_seq` base-pos) — all-positions
   на pos 0 дадуть саме такий тип деградації.
3. **GQA head-mapping** (14 q-heads / 2 kv-heads) у `forward.rs`.
4. **SwiGLU orientation** gate/up у `ffn`.
5. `rms_eps` / final norm.

Наступний зріз: one-prompt diff-профіль per-layer hidden проти llama-rs
(його heartbeat/API не віддає hidden — порівнювати логіти через
`/v1/completions` топи чи local-зріз `llama-rs` як dev-залежність).
Тести-харнес (`tests/moe_parity.rs`) готові: floor-и `min_prefix` підняти
разом з фіксом.

## Як повторити

```bash
# оракул (S:/rust/llama-rs)
./target/release/llama_serve.exe models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --port 8085 --model-name qwen05-oracle --max-tokens 12 --temperature 0.0
# гейт A (debug, ~1 s + read)
cargo test --test moe_parity -- --ignored tokenizer_json_byte_exact
# гейт B (release, ~2 min)
PH405B_ORACLE_URL=http://127.0.0.1:8085/v1/chat/completions \
  cargo test --release --test moe_parity -- --ignored greedy_parity
```
