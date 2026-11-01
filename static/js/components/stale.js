// A banner when the page is newer than the server it talks to (the app was updated, the server was not restarted).
//
// Detected two ways: /api/info reports `build` (fingerprint of the running code) and `restart_needed` (the files on
// disk changed since the start); a server too old to know them reports neither. And a known GET endpoint answering
// 404 / 422 without an error code (api.js calls `markStale`), which is what an old server does with a new URL.

import { setMismatchHandler } from "../core/api.js";
import "../i18n/stale.js";
import { applyI18n, onLangChange } from "../core/i18n.js";

const RECHECK_MS = 60_000;
let banner = null;
let hidden = false;

function show() {
  if (banner || hidden) return;
  banner = document.createElement("div");
  banner.className = "stale-banner";
  banner.id = "stale-banner";
  banner.setAttribute("role", "alert");
  const fill = () => {
    banner.innerHTML = `<span data-i18n="stale.text"></span>
      <button type="button" class="btn small" id="stale-reload" data-i18n="stale.reload"></button>
      <button type="button" class="btn small ghost" id="stale-hide" data-i18n="stale.dismiss"></button>`;
    applyI18n(banner);
    banner.querySelector("#stale-reload").addEventListener("click", () => location.reload());
    banner.querySelector("#stale-hide").addEventListener("click", () => {
      hidden = true;
      banner.remove();
      banner = null;
    });
  };
  fill();
  document.body.prepend(banner);
  onLangChange(() => banner && fill());
}

async function check() {
  try {
    const res = await fetch("/api/info", { cache: "no-store" });
    if (!res.ok) return;
    const info = await res.json();
    if (!("build" in info) || info.restart_needed) show();
  } catch {}
}

export function mountStaleWatch() {
  setMismatchHandler(show);
  check();
  setInterval(() => !document.hidden && check(), RECHECK_MS);
}

