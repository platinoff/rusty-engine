# MoE router math — local reference (self-authored, standard activations)

Формули `rusty-engine` `src/moe.rs` (PH-401..403). Точні heuristic-деталі `noaux_tc`
(top-2-per-group + group bias, DeepSeek-V3 / Qwen3) уточнювати при parity-тесті
(`PH-405`/B9) проти llama.cpp `qwen3moe`/`deepseek2.cpp`; тут — канонічна база + unit-тести.

## Активації
- `sigmoid(x) = 1/(1+e^-x)`
- `silu(x) = x * sigmoid(x)`  (Swish)
- `swiglu(gate, up) = silu(gate) * up`  (елементно)
- `softmax` — стабільний (max-subtract)

## Роутер (PH-401)
- **Mixtral-style:** `score = softmax(logits)` → top-k → (вже нормовано).
- **Qwen3/DeepSeek-style (sigmoid):** `score = sigmoid(logits)` → top-k →
  **renorm** діленням на Σ обраних (суми ваг = 1).
- **noaux_tc group (спрощено):** `exp[i] = sigmoid(logits[i] + bias[i])`;
  groups = `i*ng/n` (контiguous); `group_score[g] = Σ exp` у групі; вибір `top_k_group`
  груп; candidates = experts у вибраних групах; top-k за `exp`; renorm. Точний
  llama.cpp топ-2-на-групу → `PH-405` (не впливає на API).

## Expert FFN (PH-402)
`out = down_proj( swiglu( gate_proj(x), up_proj(x) ) )`
матвек: `matvec(W[m×n], x[n]) -> [m]`, row-major. `hidden` = in/out dim, `inter` = ffn dim.

## Shared expert merge (PH-403)
`out = Σ_k weight[k] * routed_out[k]  +  shared_out`  (DeepSeek/Qwen3 have a dense shared
expert always added; Mixtral has none → `shared = None`).
