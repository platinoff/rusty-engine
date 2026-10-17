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

## Гейт B — semantic greedy parity: PASS (byte-for-byte, PH-405 fixed)

**Фікс (2026-10-17, та сама agi-сесія).** Оракул був стабільний, вхід
правильний (токенізація byte-exact, геометрія валідна, усі 151 936 logits
finite), але top-1 був сміттям — префікс-збіг = 0 на кожному рядку. Знайдено
**дві** систематичні помилки forward-стека (обидві в `rusty-engine`):

1. **Пропущені q/k/v-біаси Qwen2.** GGUF несе `blk.N.attn_{q,k,v}.bias`
   (291 тензор = 219 + 3×24), `model.rs` їх лише *валідував*, а forward їх не
   додавав. Qwen2 (на відміну від Llama) має qkv-біаси — без них вихід шумовий.
   Фікс: `BlockWeights.{q,k,v}_bias` + `add_bias`/`add_bias_rows` у всіх
   attention-шляхах; `runner.rs` тягне біаси опційно (порожні для Llama-style).
   **Це головна причина.**
2. **RoPE-mapping.** `infer::rope` робив GPT-J interleaved (пари `2i,2i+1`),
   тоді як Qwen2/llama.cpp використовують **NeoX rotate-half** (пари
   `i, i+d/2`). Фікс: rotate-half. (RoPE діє з pos≥1, тож сам по собі він
   давав шум; разом із біасами стек тепер точний.)

Результат на зафіксованих goldens (`tests/moe_parity.rs`,
`greedy_text_against_recorded_oracle` — офлайн, без мережі):

| prompt | ids | oracle текст | rusty | lcp |
|--------|-----|--------------|-------|-----|
| `The capital of France is` | `785 6722 315 9625 374` | ` Paris. It is the largest city in Europe and the second` | той самий | **55/55** |
| `2+2=` | `17 10 17 28` | `4, 3+3=6, 4+` | той самий | **12/12** |
| `Berlin is a` | `94409 374 264` | ` city in the state of Berlin, Germany. It is the` | той самий | **48/48** |

`full == incremental` на реальній моделі збережено. Floor-и `min_prefix` у
`PARITY_CASES` піднято до повної довжини мінус невеликий запас (52 / 10 / 45) —
тепер гейт ловить будь-який регрес forward.

Числовий стрічковий відпечаток `tests/data/qwen25_golden.txt` **перезахоплено**
(`RUSTY_CAPTURE_GOLDEN=1`) — старі «шумові» токени (`127498 27277 …`) замінено
на зв'язні (`7063 8749 14358 …`, `198`=нова строка).

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
