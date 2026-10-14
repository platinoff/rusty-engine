# rusty-engine — Web UI Design System

**Продукт:** `rusty_engine` · сторінки рендерить **Rust** (`src/ui.rs`, band B10).
`ui/js/app.js` = **chrome ONLY**. Дії = звичайні GET/POST-форми. Ratio-рашет ≥ **0.95**
(не послаблювати). Ніколи не повертати рендер у JS (канон-закон плагіна).

## 1. Токени (dark-first)

`ui/css/style.css` (`:root` custom properties; `[data-theme]` overrides):

```css
:root{
  --bg:#0b0f14; --bg-soft:#11161d; --panel:#151b24; --panel-2:#1a2230;
  --border:#232b38; --text:#d7dee8; --muted:#7d8a9c;
  --accent:#3aa0ff; --accent-2:#22c55e; --warn:#f5b041; --err:#ef4444;
  --radius:10px; --shadow:0 2px 10px rgba(0,0,0,.35);
  --mono:"Cascadia Mono","JetBrains Mono",Consolas,monospace;
  --sidebar-w:272px; --ui:14px;
}
```

Схема: `prefers-color-scheme` + явний перемикач `auto → light → dark`
(`localStorage["re.theme"]`, `document.body.dataset.theme`). `color-scheme:dark`.

## 2. Сторінки (Rust `/page/{name}` + fragments `GET /api/ui/card/{name}`)

| name | Призначення |
|------|-------------|
| `dashboard` | флот master/slave, tok/s, активні experts, GPU/RAM, lease-статус |
| `models` | GGUF-картки моделей (arch, quant, size, placement) |
| `inference` | playground: prompt, стрим-токени, usage, stop-tokens |
| `tuning` | temperature / top_p / top_k / **router-α** / shared-expert / KV-preserve |
| `nodes` | edge-вузли: addr, role, health, fail-open, last-heartbeat |
| `orchestration` | placement-leases, expert-parallel шартинг, queue |
| `settings` | порти, LAN-bind, `--live`-гейт, GSV-хаб-з'єднання |
| `logs` | telemetry/ticker |

## 3. Chrome (JS only, XSS-safe)

- **Тема** — 3-way cycle, `#theme-btn`.
- **Сайдбар** — `#nav-toggle` → `body.classList.toggle("nav-collapsed")` + `aria-expanded`.
- **Хоткеї** (GET-навігація):
  ```js
  const HOTKEYS = { d:"/page/dashboard", m:"/page/models", i:"/page/inference",
    k:"/page/tuning", f:"/page/nodes", s:"/page/settings",
    o:"/page/orchestration", l:"/page/logs" };
  ```
- **`?` help-panel** — `#help.help-overlay` з `<kbd>` на кожен хоткей;
  будується `createElement`/`textContent`, **ніколи `innerHTML`** (XSS-law).

## 4. Layout

`body{grid-template-columns:var(--sidebar-w) 1fr}` — sidebar + workspace;
`.panel-dock` для згорнутих панелей; focus-visible rings; monospace для метрик/логів.

## 5. Контракт-гейт (B10)

`tests/ui_glue_contracts.rs`:
- **LOC ratchet**: rust-частка ≥ 0.95 (аналог CoreBoard `RUST_RATIO_RATCHET`);
- **UI-інваріант**: жодного `.innerHTML`/`document.write` у `ui/js`; рендер сторінок —
  лише Rust (`src/ui.rs`).
