# Uk-golden — поведінковий оціночний набор для MiniCPM (Rust warp)

**Дата:** 2026-11-15 · **Збірка:** rusty-engine @ `7f7c1ad` (release) · **Стан:** **DRAFT, очікує owner review**
**Залежності гейту:** Uk-golden обов'язковий крок MC-6 (WSP gate on MiniCPM path) та кожного offloaded matvec assert.

## Що це

18 укр. промптів (класи: fact / grammar / code / reasoning / translation / idiom / tech / chat / long / gen),
критичні (абсолютна правильність) — `uk-fact-1`, `uk-fact-2`, `uk-fact-3`, `uk-reasoning-1`.

`fixtures.json` — масив записів `{id, class, critical, prompt, expectation, baseline}`:

- **`expectation`** — еталон людини (що має бути у правильній відповіді). НЕ те, що зараз видає модель.
- **`baseline`** — вербатим запис поточної відповіді (pre-patch baseline для before/after gate).
- **`critical`** — що MC-6 gate перевіряє на абсолютну правильність/відсутність деградації.

## Метод запису

Serve: `rusty serve -m MiniCPM-V-4_7-1B-Q4_K_M.gguf :8088` (release), chat/completions, `temperature 0`,
`max_tokens 160`, **без ChatML-обгортки** (поточна стейт serve конкатенує messages content без шаблону).

## Виміряно (2026-11-15, після двох багфіксів: UTF-8 serve `ae39cb5`, gpt2 contraction `7f7c1ad`)

Щодо поточної моделі (1B, Q4_K_M, Uk):

- Сильні відповіді: `uk-fact-1` (Kyiv ✅), `uk-reasoning-1` (43 ✅), `uk-code-1` (працююча fn ✅)
- Частково коректні: `uk-chat-2`, `uk-tech-1`, `uk-tech-2`, `uk-translate-2` (спроба), `uk-reasoning-2` (семантика 3,2,5 ✅ без чіткої 3/5)
- Деградовані/невірні: `uk-fact-2` (**Пушкін замість «Кобзарі» Шевченка** — критичний клас фейлу), `uk-fact-3`, `uk-grammar-1`, `uk-grammar-2`, `uk-idiom-1`, `uk-code-2`, `uk-translate-1`, `uk-long-1`, `uk-gen-1`, `uk-chat-1` — повтори/галюцинації/«Візуальна оцінка: 5/5»-цикли.
- Класична властивість 1B на Uk: повтори, тертя в граматиці, підміна фактів. Це і є простір для WSP-патчів (MC-6) — але тільки через цей gate.

## Чесні застереження

- Модель мовою міркування за замовчуванням — англійська/російська, хоча розуміє укр. вхід.
- Багато baseline відповідей — деградовані; fixture фіксує **стан до патча**, не «хорошість».
- Proposed WSP-фікс класу `uk-fact-2` (фактична підміна на Uk-факт) — пріоритет №1 для MC-6:
  подібні патчі мають найбільший шанс на keep, і тут же найбільший ризик пошкодити інші кластери.
  `uk-fact-2` — canary-кандидат.

## Owner review checklist (до канону)

1. Чи підходять 18 промптів / класи / очікування? (можна додати/прибрати рядки)
2. Чи `uk-fact-2` — прийнятний перший клас фейлу для MC-6?
3. Чи допускаємо baseline відповідей як «pre-patch» стан (включно з деградованими)?
4. Параметри запису (temperature 0, max_tokens 160) — фіксовані?

## Як використати

- MC-6: `rusty engine` оновлює serve; прогін fixtures до і після патча; keep/rollback за:
  (a) critical — усі 4 не деградували; (b) решта — token-overlap / exact збіг ≥ порогу; (c) Uk-nkл
  `uk-fact-2` залишається коректною або залишається прийнятно схожою.
- Не комітити GGUF/ваги; fixtures = лише текст.
