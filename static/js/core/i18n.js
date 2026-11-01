// Tiny i18n: nested dictionaries per language, dotted keys, {param} interpolation and
// plural forms via Intl.PluralRules ({ one, few, many, other }).
// Add a language: create i18n/<code>.js and list it in LANGUAGES below.

import en from "../i18n/en.js";
import cs from "../i18n/cs.js";

export const LANGUAGES = { en, cs };

// A page with many texts can ship its own dictionaries: extendLanguages({ en: {...}, cs: {...} }) merges them
// (deeply) into the main ones, so the shared en.js / cs.js stay small.
export function extendLanguages(dicts) {
  const merge = (target, source) => {
    for (const [key, value] of Object.entries(source)) {
      if (value && typeof value === "object" && target[key] && typeof target[key] === "object") merge(target[key], value);
      else target[key] = value;
    }
  };
  for (const [code, dict] of Object.entries(dicts)) if (LANGUAGES[code]) merge(LANGUAGES[code], dict);
}
const STORAGE_KEY = "lang";
const listeners = new Set();

function detect() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved in LANGUAGES) return saved;
  } catch {}
  for (const tag of navigator.languages || [navigator.language || "en"]) {
    const code = tag.toLowerCase().split("-")[0];
    if (code in LANGUAGES) return code;
    if (code === "sk") return "cs";
  }
  return "en";
}

let current = detect();

export const getLang = () => current;
// BCP 47 locale for Intl formatting
export const locale = () => LANGUAGES[current]._locale || current;

export function setLang(code) {
  if (!(code in LANGUAGES) || code === current) return;
  current = code;
  try {
    localStorage.setItem(STORAGE_KEY, code);
  } catch {}
  document.documentElement.lang = code;
  applyI18n();
  for (const fn of listeners) fn(code);
}

export function onLangChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function lookup(dict, key) {
  return key.split(".").reduce((node, part) => (node && typeof node === "object" ? node[part] : undefined), dict);
}

export function has(key) {
  return lookup(LANGUAGES[current], key) !== undefined || lookup(LANGUAGES.en, key) !== undefined;
}

// t("jobs.queued_at", { n: 2 }); a plural entry is chosen by params.n; numbers are locale-formatted
export function t(key, params = {}) {
  let value = lookup(LANGUAGES[current], key) ?? lookup(LANGUAGES.en, key);
  if (value === undefined) return key;
  if (typeof value === "object") {
    const form = new Intl.PluralRules(locale()).select(params.n ?? 0);
    value = value[form] ?? value.other ?? "";
  }
  const fmt = (v) => (typeof v === "number" ? v.toLocaleString(locale()) : v);
  return String(value).replace(/\{(\w+)\}/g, (m, name) => (name in params ? fmt(params[name]) : m));
}

// Fill elements marked with data-i18n (text), data-i18n-placeholder, data-i18n-title, data-i18n-aria-label.
export function applyI18n(root = document) {
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const attr of ["placeholder", "title", "aria-label"]) {
    const data = `i18n${attr.replace(/(^|-)(\w)/g, (_, __, c) => c.toUpperCase())}`;
    for (const el of root.querySelectorAll(`[data-i18n-${attr}]`)) el.setAttribute(attr, t(el.dataset[data]));
  }
}

document.documentElement.lang = current;
