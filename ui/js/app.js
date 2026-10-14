// rusty-engine UI chrome ONLY (B0 placeholder). Theme, sidebar, hotkeys, `?` help.
// NEVER renders pages — pages come from Rust (src/ui.rs) in B10.
// XSS law: build nodes with createElement/textContent, never innerHTML.
(function () {
  "use strict";

  var THEME_KEY = "re.theme";
  var THEMES = ["auto", "light", "dark"];
  var HOTKEYS = {
    d: "/page/dashboard",
    m: "/page/models",
    i: "/page/inference",
    k: "/page/tuning",
    f: "/page/nodes",
    s: "/page/settings",
    o: "/page/orchestration",
    l: "/page/logs"
  };

  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    document.body.dataset.theme = t;
  }
  function cycleTheme() {
    var cur = localStorage.getItem(THEME_KEY) || "auto";
    var next = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  }

  function toggleSidebar() {
    document.body.classList.toggle("nav-collapsed");
    var btn = document.getElementById("nav-toggle");
    if (btn) btn.setAttribute("aria-expanded", String(!document.body.classList.contains("nav-collapsed")));
  }

  function markActive() {
    var path = location.pathname;
    Array.prototype.forEach.call(document.querySelectorAll(".nav a"), function (a) {
      if (a.getAttribute("href") === path) a.setAttribute("aria-current", "page");
    });
  }

  function toggleHelp() {
    var existing = document.getElementById("help");
    if (existing) { existing.remove(); return; }
    var overlay = document.createElement("div");
    overlay.id = "help";
    overlay.className = "help-overlay";
    var card = document.createElement("div");
    card.className = "help-card";
    var h = document.createElement("h2");
    h.textContent = "Hotkeys";
    card.appendChild(h);
    Object.keys(HOTKEYS).forEach(function (key) {
      var row = document.createElement("div");
      row.className = "help-row";
      var k = document.createElement("kbd");
      k.textContent = key;
      var v = document.createElement("span");
      v.textContent = HOTKEYS[key];
      row.appendChild(k);
      row.appendChild(v);
      card.appendChild(row);
    });
    overlay.appendChild(card);
    overlay.addEventListener("click", function (e) { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
  }

  document.addEventListener("keydown", function (e) {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key === "?") { toggleHelp(); return; }
    var route = HOTKEYS[e.key.toLowerCase()];
    if (route && !e.ctrlKey && !e.metaKey && !e.altKey) { location.href = route; }
  });

  document.addEventListener("DOMContentLoaded", function () {
    applyTheme(localStorage.getItem(THEME_KEY) || "auto");
    markActive();
    var tb = document.getElementById("theme-btn");
    if (tb) tb.addEventListener("click", cycleTheme);
    var nb = document.getElementById("nav-toggle");
    if (nb) nb.addEventListener("click", toggleSidebar);
  });
})();
