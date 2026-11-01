// First-run guide on the home page: instead of an error in the middle of a transcription, a new user gets the
// missing pieces (ffmpeg, HuggingFace token and model terms) step by step, and may continue without speakers.

import "../i18n/system.js";
import { getInfo } from "../core/api.js";
import { applyI18n, t } from "../core/i18n.js";
import { href } from "../core/router.js";
import { $, esc } from "../core/util.js";
import { loadSystem, rich, tokenForm } from "./system.js";

const SKIP_KEY = "setup.skip-speakers";
const skipped = () => {
  try {
    return localStorage.getItem(SKIP_KEY) === "1";
  } catch {
    return false;
  }
};
const setSkipped = (on) => {
  try {
    on ? localStorage.setItem(SKIP_KEY, "1") : localStorage.removeItem(SKIP_KEY);
  } catch {}
};

// onState({ blocked }): blocked = speaker recognition cannot run now (the form turns the option off).
export function mountSetup(box, { onState }) {
  let alive = true;
  refresh();

  async function refresh() {
    let system, info;
    try {
      [system, info] = await Promise.all([loadSystem(), getInfo()]);
    } catch {
      return; // the status is a convenience; the page works without it
    }
    if (!alive) return;
    const by = Object.fromEntries(system.checks.map((c) => [c.id, c]));
    const needsHf = info.defaults.diarize && system.pyannote_engines.includes(info.defaults.engine);
    const hf = by.hf;
    const blocked = needsHf && hf.status !== "ok";
    const parts = [];

    if (by.ffmpeg.status === "error") {
      parts.push(`<section class="card setup-block" role="alert"><h2>${esc(t("setup.ffmpeg_title"))}</h2>
        <p>${rich(t("setup.ffmpeg_body"))}</p>
        <a class="btn" href="${href("/system")}">${esc(t("setup.open_settings"))}</a></section>`);
    }
    if (blocked && !skipped()) parts.push(guide(hf, system));
    else if (blocked) {
      parts.push(`<div class="card setup-note"><span>${esc(t("setup.skipped"))}</span>
        <button class="btn" type="button" data-act="resume">${esc(t("setup.resume"))}</button></div>`);
    }
    const notes = [];
    if (by.gpu?.code === "no_nvidia") notes.push(t("setup.no_gpu_note"));
    if (by.jsruntime?.status === "warn") notes.push(t("setup.youtube_note"));
    if (notes.length) {
      parts.push(`<div class="card setup-note"><span>${esc(notes.join(" "))}</span>
        <a href="${href("/system")}">${esc(t("setup.open_settings"))}</a></div>`);
    }

    box.innerHTML = parts.join("");
    box.hidden = !parts.length;
    applyI18n(box);
    const form = $(".token-slot", box);
    if (form) form.replaceWith(tokenForm(system, () => refresh()));
    $("[data-act=skip]", box)?.addEventListener("click", () => (setSkipped(true), refresh()));
    $("[data-act=resume]", box)?.addEventListener("click", () => (setSkipped(false), refresh()));
    $("[data-act=check]", box)?.addEventListener("click", async (e) => {
      e.currentTarget.disabled = true;
      e.currentTarget.textContent = t("setup.checking");
      try {
        await loadSystem(true);
      } catch {}
      refresh();
    });
    onState({ blocked });
  }

  function guide(hf, system) {
    const code = hf.code;
    const links = hf.params || {};
    const done1 = ["no_access", "unreachable"].includes(code);
    const done2 = done1;
    const step = (n, done, active, title, body) => `
      <li class="setup-step${done ? " done" : ""}${active ? " active" : ""}">
        <span class="step-no" aria-hidden="true">${done ? "✓" : n}</span>
        <div class="step-body"><div class="step-title">${esc(title)}<span class="visually-hidden"> — ${esc(done ? t("setup.done") : t("setup.todo"))}</span></div>${body}</div>
      </li>`;
    const link = (url, label) => `<a class="btn" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`;
    const status = ["invalid_token", "unreachable", "offline_missing"].includes(code)
      ? `<p class="setup-status" role="status">${esc(t(`system.items.hf.${code}`))}</p>` : "";
    return `<section class="card setup-guide" aria-labelledby="setup-title">
      <h2 id="setup-title">${esc(t("setup.title"))}</h2>
      <p>${esc(t("setup.intro"))}</p>
      <ol class="setup-steps">
        ${step(1, done1, false, t("setup.step1"), `<div>${link("https://huggingface.co/join", t("system.hf.open_join"))}</div>`)}
        ${step(2, done2, ["no_token", "invalid_token"].includes(code), t("setup.step2"),
          `<div>${link(links.tokens_url, t("system.hf.open_tokens"))}</div>${["no_token", "invalid_token"].includes(code) ? '<div class="token-slot"></div>' : ""}`)}
        ${step(3, false, code === "no_access", t("setup.step3"),
          `<p class="step-note">${esc(t("setup.step3_note"))}</p>${link(links.model_url, t("system.hf.open_model"))}`)}
      </ol>
      ${status}
      <div class="setup-actions">
        <button class="btn primary" type="button" data-act="check">${esc(t("setup.check"))}</button>
        <button class="btn" type="button" data-act="skip">${esc(t("setup.skip"))}</button>
      </div>
    </section>`;
  }

  return () => { alive = false; };
}
