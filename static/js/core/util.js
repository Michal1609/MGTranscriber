// Small DOM, formatting and UI helpers shared by all pages.

import { locale, t } from "./i18n.js";

// Poll timer that does not poll a hidden tab: while the tab is hidden the call is skipped and the timer is armed
// again. `setHandle` stores each timeout id (so clearTimeout of the caller's variable always cancels the poll).
export function pollTimeout(setHandle, fn, ms) {
  const tick = () => {
    if (document.hidden) setHandle(setTimeout(tick, ms));
    else fn();
  };
  setHandle(setTimeout(tick, ms));
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pad = (n) => String(n).padStart(2, "0");

export function clock(sec) {
  const s = Math.floor(sec);
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

export function shortClock(sec) {
  const s = Math.floor(sec);
  return s >= 3600 ? clock(s) : `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

export function duration(sec) {
  const m = Math.round(sec / 60);
  if (m < 1) return t("common.duration_s", { n: Math.round(sec) });
  return m >= 60 ? t("common.duration_h", { h: Math.floor(m / 60), m: m % 60 }) : t("common.duration_min", { n: m });
}

export function date(ts) {
  return new Date(ts * 1000).toLocaleString(locale(), {
    day: "numeric", month: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export const number = (n) => n.toLocaleString(locale());

let toastTimer;
export function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2200);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // the clipboard API is unavailable outside a secure context (http from another computer)
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    ta.style.cssText = "position:fixed;opacity:0";
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

/* --- speakers --- */

const COLORS = 8;
export const speakerOrder = (stats) => Object.keys(stats).sort();
// The speakers to count and list: UNKNOWN is words no speaker was assigned to, not a person.
export const UNASSIGNED_MIN_S = 0.5; // shorter than this it would show as "00:00"
export const realSpeakers = (stats) => speakerOrder(stats).filter((s) => s !== "UNKNOWN");
export const speakerColor = (index) => `var(--spk-${index % COLORS})`;

export function speakerLabel(id, names = {}) {
  if (names[id]) return names[id];
  if (id === "UNKNOWN") return t("speaker.unknown");
  const m = id.match(/^SPEAKER_0*(\d+)$/);
  return m ? t("speaker.n", { n: Number(m[1]) + 1 }) : id;
}

export function languageName(code) {
  return code && (code === "auto" || ["cs", "en"].includes(code)) ? t(`lang.${code}`) : t("lang.other", { code: code || "?" });
}
