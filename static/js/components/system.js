// Pieces shared by the Settings page and the first-run guide on the home page: status icons, translated
// check messages and the HuggingFace token form. Texts live in i18n/system.js.

import "../i18n/system.js";
import { api } from "../core/api.js";
import { applyI18n, has, t } from "../core/i18n.js";
import { $, esc, toast } from "../core/util.js";

const ICON_PATHS = {
  ok: "M9.5 16.2 5.3 12l-1.4 1.4 5.6 5.6L21 7.5 19.6 6z",
  warn: "M12 3 1.5 21h21zm1 14h-2v-2h2zm0-4h-2V9h2z",
  error: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm5 13.6-1.4 1.4-3.6-3.6L8.4 17 7 15.6l3.6-3.6L7 8.4 8.4 7l3.6 3.6L15.6 7 17 8.4 13.4 12z",
  na: "M5 11h14v2H5z",
};

// The shape differs per status as well as the colour, so the state is never conveyed by colour alone.
export const statusIcon = (status) =>
  `<span class="st st-${status}" role="img" aria-label="${esc(t(`system.status.${status}`))}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="${ICON_PATHS[status]}"/></svg></span>`;

// A server error as text: our translation of its code when there is one, else the server's own message.
export const errorText = (err) => (err.code && has(`system.errors.${err.code}`) ? t(`system.errors.${err.code}`) : err.message);

// Escapes text and turns `code` spans into <code> (the dictionary writes commands in backticks).
export const rich = (text) => esc(text).replace(/`([^`]+)`/g, "<code>$1</code>");

export function checkMessage(check) {
  const key = `system.items.${check.id}.${check.code}`;
  if (has(key)) return t(key, check.params || {});
  return has(`system.generic.${check.code}`) ? t(`system.generic.${check.code}`) : check.detail;
}

export const loadSystem = (refresh = false) => api(`/api/system${refresh ? "?refresh=true" : ""}`);

// Token form: paste, validate with HuggingFace, save to .env. `onDone(system)` gets a fresh report.
export function hfActions(check, system) {
  const links = check.params || {};
  const parts = [];
  if (check.code === "no_access") {
    parts.push(`<a class="btn" href="${esc(links.model_url)}" target="_blank" rel="noopener noreferrer">${esc(t("system.hf.open_model"))}</a>`);
  }
  if (["no_token", "invalid_token"].includes(check.code)) {
    parts.push(`<a class="btn" href="${esc(links.tokens_url)}" target="_blank" rel="noopener noreferrer">${esc(t("system.hf.open_tokens"))}</a>`);
  }
  return parts.join(" ");
}

export function tokenForm(system, onDone) {
  const wrap = document.createElement("div");
  wrap.className = "token-form";
  if (system.hf_token_source === "environment") {
    wrap.innerHTML = `<p class="meta">${esc(t("system.hf.from_env"))}</p>`;
    return wrap;
  }
  if (!system.can_save_token) {
    wrap.innerHTML = `<p class="meta">${esc(t("system.hf.remote_hint"))}</p>`;
    return wrap;
  }
  const id = `hf-token-${Math.random().toString(36).slice(2, 7)}`;
  wrap.innerHTML = `
    <form class="token-row" autocomplete="off">
      <label for="${id}" class="visually-hidden">${esc(t("system.hf.token"))}</label>
      <input id="${id}" type="password" name="hf-token" spellcheck="false" autocomplete="off" required
             placeholder="${esc(t("system.hf.token_placeholder"))}">
      <button class="btn primary" type="submit">${esc(t("system.hf.save"))}</button>
      ${system.hf_token_source === ".env" ? `<button class="btn danger" type="button" data-act="remove">${esc(t("system.hf.remove"))}</button>` : ""}
    </form>
    <p class="form-msg" role="status"></p>`;
  const form = $("form", wrap);
  const msg = $(".form-msg", wrap);
  const busy = (on) => form.querySelectorAll("button").forEach((b) => (b.disabled = on));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.textContent = "";
    busy(true);
    const submit = $("button[type=submit]", form);
    submit.textContent = t("system.hf.saving");
    try {
      const res = await api("/api/system/hf-token", { method: "PUT", body: JSON.stringify({ token: $("input", form).value.trim() }) });
      toast(t(res.verified ? "system.hf.saved" : "system.hf.saved_unverified"));
      onDone(await loadSystem(true));
    } catch (err) {
      msg.textContent = errorText(err);
      submit.textContent = t("system.hf.save");
      busy(false);
    }
  });
  $("[data-act=remove]", form)?.addEventListener("click", async () => {
    busy(true);
    try {
      await api("/api/system/hf-token", { method: "DELETE" });
      toast(t("system.hf.removed"));
      onDone(await loadSystem(true));
    } catch (err) {
      msg.textContent = errorText(err);
      busy(false);
    }
  });
  applyI18n(wrap);
  return wrap;
}

// One cloud API key: password field + Save, Test and Remove. The value is write-only (never shown again).
// `key` is an item of `system.keys`; `onDone(system)` gets a fresh report after a change.
export function keyForm(key, system, onDone) {
  const wrap = document.createElement("div");
  wrap.className = "token-form";
  if (key.source === "environment") {
    wrap.innerHTML = `<p class="meta">${esc(t("system.keys.from_env"))}</p>`;
    return wrap;
  }
  if (!system.can_save_keys) {
    wrap.innerHTML = `<p class="meta">${esc(t("system.keys.remote_hint"))}</p>`;
    return wrap;
  }
  const id = `key-${key.env.toLowerCase()}`;
  wrap.innerHTML = `
    <form class="token-row" autocomplete="off">
      <label for="${id}" class="visually-hidden">${esc(key.env)}</label>
      <input id="${id}" type="password" name="${esc(key.env)}" spellcheck="false" autocomplete="new-password" required
             placeholder="${esc(t(key.set ? "system.keys.replace_placeholder" : "system.keys.placeholder"))}">
      <button class="btn primary" type="submit">${esc(t("system.keys.save"))}</button>
      ${key.set ? `<button class="btn" type="button" data-act="test">${esc(t("system.keys.test"))}</button>
      <button class="btn danger" type="button" data-act="remove">${esc(t("system.keys.remove"))}</button>` : ""}
      <a class="btn" href="${esc(key.url)}" target="_blank" rel="noopener noreferrer">${esc(t("system.keys.get"))}</a>
    </form>
    <p class="form-msg" role="status"></p>`;
  const form = $("form", wrap);
  const msg = $(".form-msg", wrap);
  const busy = (on) => form.querySelectorAll("button").forEach((b) => (b.disabled = on));
  const url = `/api/system/keys/${encodeURIComponent(key.env)}`;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.textContent = "";
    busy(true);
    try {
      await api(url, { method: "PUT", body: JSON.stringify({ value: $("input", form).value.trim() }) });
      $("input", form).value = "";
      toast(t("system.keys.saved"));
      onDone(await loadSystem(true));
    } catch (err) {
      msg.textContent = errorText(err);
      busy(false);
    }
  });
  $("[data-act=test]", form)?.addEventListener("click", async () => {
    busy(true);
    msg.textContent = t("system.keys.testing");
    try {
      const res = await api(`${url}/test`, { method: "POST" });
      msg.textContent = t(`system.keys.test_${res.code}`);
    } catch (err) {
      msg.textContent = errorText(err);
    }
    busy(false);
  });
  $("[data-act=remove]", form)?.addEventListener("click", async () => {
    busy(true);
    try {
      await api(url, { method: "DELETE" });
      toast(t("system.keys.removed"));
      onDone(await loadSystem(true));
    } catch (err) {
      msg.textContent = errorText(err);
      busy(false);
    }
  });
  return wrap;
}
