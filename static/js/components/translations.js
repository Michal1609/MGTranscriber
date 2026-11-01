// Language switcher and "Translate to…" of the transcript detail page.
// mountTranslations({ id, box, sourceLang, initial, onView }) fills `box` and returns
//   { stop(), lang(): shown language ("" = the original), show(lang) }.
// onView(lang, data) is called when the view changes: data = the stored translation ({segments, stale, ...}) or null.
// Translation itself is a queued job (task "translate"); progress comes from GET .../translations (active jobs).

import "../i18n/translate.js";
import { api } from "../core/api.js";
import { applyI18n, getLang, t } from "../core/i18n.js";
import { date, esc, pollTimeout, toast } from "../core/util.js";

const POLL_MS = 2000;

export function loadStyles() {
  if (document.querySelector("link[data-translate-css]")) return;
  const link = Object.assign(document.createElement("link"), { rel: "stylesheet", href: "translate.css" });
  link.dataset.translateCss = "";
  document.head.append(link);
}
const CHOICE_KEY = "llm.choice"; // shared with the Summary tab: provider and model last used

function readChoice() {
  try {
    return JSON.parse(localStorage.getItem(CHOICE_KEY)) || {};
  } catch {
    return {};
  }
}

// "čeština" for cs in the interface language; the server's English name when Intl has none.
export function langName(code, fallback) {
  try {
    const name = new Intl.DisplayNames([getLang()], { type: "language" }).of(code);
    if (name && name !== code) return name.charAt(0).toLocaleUpperCase(getLang()) + name.slice(1);
  } catch {}
  return fallback || code.toUpperCase();
}

export function mountTranslations({ id, box, sourceLang, initial = "", onView }) {
  loadStyles();
  const base = `/api/transcripts/${encodeURIComponent(id)}`;
  const state = {
    list: [], jobs: [], languages: {}, lang: "", data: null, providers: null, watched: new Map(), message: null,
    formOpen: false, timer: null, stopped: false, loadingLang: null,
  };

  box.innerHTML = `
    <div class="lang-row">
      <div class="lang-switch" role="group" data-i18n-aria-label="translate.group"></div>
      <button type="button" class="btn lang-open" aria-expanded="false" aria-controls="lang-form-${esc(id.length)}" data-i18n="translate.open"></button>
    </div>
    <form class="card lang-form" id="lang-form-${esc(id.length)}" hidden>
      <label>${esc(t("translate.target"))}<select class="lf-lang"></select></label>
      <label>${esc(t("translate.provider"))}<select class="lf-provider"></select></label>
      <label>${esc(t("translate.model"))}<input class="lf-model" type="text" autocomplete="off" list="lf-models-${esc(id.length)}"><datalist id="lf-models-${esc(id.length)}" class="lf-models"></datalist></label>
      <div class="lang-actions">
        <button type="submit" class="btn primary lf-start" data-i18n="translate.start"></button>
        <button type="button" class="btn lf-cancel" data-i18n="translate.cancel"></button>
      </div>
      <p class="llm-note lf-note"></p>
    </form>
    <div class="lang-status" role="status" aria-live="polite"></div>
    <div class="lang-banner" hidden></div>`;
  applyI18n(box);
  const $q = (sel) => box.querySelector(sel);

  /* --- rendering --- */

  function nameOf(code) {
    return langName(code, state.languages[code]);
  }

  function renderSwitch() {
    const original = sourceLang ? t("translate.original", { lang: sourceLang.toUpperCase() }) : t("translate.original_unknown");
    const items = [{ lang: "", label: original, stale: false }, ...state.list.map((x) => ({ lang: x.lang, label: nameOf(x.lang), stale: x.stale }))];
    $q(".lang-switch").innerHTML = items
      .map((x) => `<button type="button" class="lang-btn${x.lang === state.lang ? " active" : ""}" data-lang="${esc(x.lang)}" aria-pressed="${x.lang === state.lang}"${x.lang ? ` lang="${esc(x.lang)}"` : ""}>${esc(x.label)}${x.stale ? ` <small>(${esc(t("translate.stale_mark"))})</small>` : ""}</button>`)
      .join("");
    $q(".lang-switch").hidden = state.list.length === 0;
    $q(".lang-open").classList.toggle("lang-open-first", state.list.length === 0);
  }

  function renderStatus() {
    const parts = state.jobs.map((j) => {
      const lang = nameOf(j.language);
      const text = j.status === "running"
        ? t("translate.running", { lang, pct: Math.round(j.percent ?? 0) })
        : j.position ? t("translate.queued", { lang, n: j.position }) : t("translate.queued_plain", { lang });
      const bar = j.status === "running"
        ? `<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(j.percent ?? 0)}" aria-label="${esc(text)}"><i style="width:${Math.round(j.percent ?? 0)}%"></i></div>`
        : "";
      return `<div class="lang-job"><span>${esc(text)}</span>${bar}<button type="button" class="btn small" data-cancel="${j.id}">${esc(t("translate.cancel"))}</button></div>`;
    });
    if (state.message) {
      parts.push(`<div class="${state.message.error ? "llm-error" : "lang-msg"}"${state.message.error ? ' role="alert"' : ""}>${esc(state.message.text)}</div>`);
    }
    $q(".lang-status").innerHTML = parts.join("");
  }

  function renderBanner() {
    const el = $q(".lang-banner");
    const entry = state.list.find((x) => x.lang === state.lang);
    el.hidden = !entry;
    if (!entry) return;
    const when = entry.created ? date(entry.created) : "";
    const lines = [];
    if (entry.stale) lines.push(`<p class="lang-stale">${esc(t("translate.stale"))}</p>`);
    lines.push(`<p class="meta">${esc(t("translate.made_by", { provider: entry.provider || "?", model: entry.model || "?", date: when }))}</p>`);
    if (entry.untranslated) lines.push(`<p class="meta">${esc(t("translate.untranslated", { n: entry.untranslated }))}</p>`);
    lines.push(`<p class="meta">${esc(t("translate.readonly"))} ${esc(t("translate.word_times"))}</p>`);
    lines.push(`<div class="lang-actions"><button type="button" class="btn" data-again="${esc(entry.lang)}">${esc(t("translate.again"))}</button>
      <button type="button" class="btn danger" data-delete="${esc(entry.lang)}">${esc(t("translate.delete"))}</button></div>`);
    el.innerHTML = lines.join("");
    el.classList.toggle("stale", !!entry.stale);
  }

  function render() {
    renderSwitch();
    renderStatus();
    renderBanner();
  }

  /* --- data --- */

  async function refresh() {
    let out;
    try {
      out = await api(`${base}/translations`);
    } catch {
      return;
    }
    if (state.stopped) return;
    state.list = out.translations;
    state.jobs = out.jobs;
    state.languages = out.languages || {};
    for (const j of out.jobs) state.watched.set(j.id, j.language);
    const active = new Set(out.jobs.map((j) => j.id));
    for (const [jobId, lang] of [...state.watched]) {
      if (active.has(jobId)) continue;
      state.watched.delete(jobId);
      await finished(jobId, lang);
    }
    if (state.lang && !state.list.some((x) => x.lang === state.lang)) await show("", { quiet: true }); // deleted elsewhere
    else if (state.lang) {
      const entry = state.list.find((x) => x.lang === state.lang);
      if (state.data && entry && entry.created !== state.data.created) await show(state.lang, { quiet: true, force: true });
    }
    render();
    schedule();
  }

  async function finished(jobId, lang) {
    let job = null;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch {}
    if (state.stopped) return;
    if (job?.status === "error") {
      state.message = { error: true, text: t("translate.failed", { lang: nameOf(lang), error: job.error || "" }) };
    } else if (job?.status === "done") {
      state.message = { error: false, text: t("translate.done", { lang: nameOf(lang) }) };
      toast(state.message.text);
      state.list = (await api(`${base}/translations`).catch(() => ({ translations: state.list }))).translations;
      await show(lang, { quiet: true, force: true });
    } else if (job?.status === "cancelled") {
      state.message = null;
    }
  }

  function schedule() {
    clearTimeout(state.timer);
    if (state.stopped) return;
    if (state.jobs.length || state.watched.size) pollTimeout((h) => (state.timer = h), refresh, POLL_MS);
  }

  async function show(lang, { quiet = false, force = false } = {}) {
    if (lang === state.lang && !force) return;
    if (!lang) {
      state.lang = "";
      state.data = null;
      onView("", null);
      render();
      return;
    }
    state.loadingLang = lang;
    try {
      const data = await api(`${base}/translations/${encodeURIComponent(lang)}`);
      if (state.stopped || state.loadingLang !== lang) return;
      state.lang = lang;
      state.data = data;
      onView(lang, data);
    } catch (err) {
      if (!quiet) state.message = { error: true, text: t("translate.load_failed", { error: err.message }) };
      if (state.lang === lang) {
        state.lang = "";
        state.data = null;
        onView("", null);
      }
    }
    render();
  }

  /* --- the form --- */

  function openForm(lang = "") {
    state.formOpen = true;
    $q(".lang-form").hidden = false;
    $q(".lang-open").setAttribute("aria-expanded", "true");
    loadProviders().then(() => {
      fillLanguages(lang);
      $q(".lf-lang").focus();
    });
  }

  function closeForm() {
    state.formOpen = false;
    $q(".lang-form").hidden = true;
    $q(".lang-open").setAttribute("aria-expanded", "false");
    $q(".lang-open").focus();
  }

  function fillLanguages(preferred) {
    const taken = new Set(sourceLang ? [sourceLang.split("-")[0]] : []);
    const codes = Object.keys(state.languages).filter((c) => !taken.has(c));
    const first = preferred || (getLang() !== sourceLang && codes.includes(getLang()) ? getLang() : codes.includes("cs") ? "cs" : codes[0]);
    const sorted = codes.sort((a, b) => nameOf(a).localeCompare(nameOf(b), getLang()));
    $q(".lf-lang").innerHTML = sorted.map((c) => `<option value="${esc(c)}"${c === first ? " selected" : ""}>${esc(nameOf(c))}</option>`).join("");
    if (!sorted.length) $q(".lf-note").textContent = t("translate.no_languages");
  }

  async function loadProviders() {
    if (state.providers) return;
    try {
      const info = await api("/api/llm");
      state.providers = info.providers;
      const choice = readChoice();
      const defaults = info.config || {};
      const sel = $q(".lf-provider");
      sel.innerHTML = state.providers
        .map((p) => `<option value="${esc(p.name)}">${esc(p.label)}${p.cloud ? " ☁" : ""}</option>`)
        .join("");
      const wanted = state.providers.some((p) => p.name === choice.provider) ? choice.provider : defaults.provider;
      if (wanted) sel.value = wanted;
      const p = state.providers.find((x) => x.name === sel.value);
      $q(".lf-model").value = choice.provider === sel.value && choice.model ? choice.model : p?.default_model || "";
      updateProvider();
    } catch {
      state.providers = [];
    }
  }

  function updateProvider() {
    const p = (state.providers || []).find((x) => x.name === $q(".lf-provider").value);
    $q(".lf-models").innerHTML = (p?.models || []).map((m) => `<option value="${esc(m)}"></option>`).join("");
    let text = t("translate.queue_note");
    if (p && !p.available) text = p.cloud ? t("translate.no_key", { env: p.key_env }) : t("translate.no_ollama");
    else if (p?.cloud) text = `${t("translate.cloud_warning")} ${text}`;
    $q(".lf-note").textContent = text;
    $q(".lf-note").classList.toggle("warn", !!p && !p.available);
  }

  async function start(lang, provider, model) {
    state.message = null;
    try {
      const job = await api(`${base}/translations`, { method: "POST", body: JSON.stringify({ language: lang, provider: provider || null, model: model || null }) });
      state.watched.set(job.id, job.language);
      state.jobs = [...state.jobs.filter((j) => j.id !== job.id), { id: job.id, language: job.language, status: job.status, percent: job.percent, position: job.position }];
      closeForm();
      render();
      schedule();
      refresh();
    } catch (err) {
      state.message = { error: true, text: err.message };
      render();
    }
  }

  /* --- events --- */

  $q(".lang-open").addEventListener("click", () => (state.formOpen ? closeForm() : openForm()));
  $q(".lf-cancel").addEventListener("click", closeForm);
  $q(".lf-provider").addEventListener("change", () => {
    const p = state.providers.find((x) => x.name === $q(".lf-provider").value);
    $q(".lf-model").value = p?.default_model || "";
    updateProvider();
  });
  $q(".lang-form").addEventListener("keydown", (e) => e.key === "Escape" && closeForm());
  $q(".lang-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const lang = $q(".lf-lang").value;
    if (lang) start(lang, $q(".lf-provider").value, $q(".lf-model").value.trim());
  });
  $q(".lang-switch").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-lang]");
    if (btn) show(btn.dataset.lang);
  });
  box.addEventListener("click", async (e) => {
    const again = e.target.closest("[data-again]");
    if (again) {
      const entry = state.list.find((x) => x.lang === again.dataset.again);
      return start(again.dataset.again, entry?.provider, entry?.model);
    }
    const del = e.target.closest("[data-delete]");
    if (del) {
      const lang = del.dataset.delete;
      if (!confirm(t("translate.delete_confirm", { lang: nameOf(lang) }))) return;
      try {
        await api(`${base}/translations/${encodeURIComponent(lang)}`, { method: "DELETE" });
        toast(t("translate.deleted"));
        state.list = state.list.filter((x) => x.lang !== lang);
        await show("", { quiet: true });
        render();
        $q(".lang-open").focus();
      } catch (err) {
        state.message = { error: true, text: err.message };
        render();
      }
      return;
    }
    const cancel = e.target.closest("[data-cancel]");
    if (cancel) {
      try {
        await api(`/api/jobs/${encodeURIComponent(cancel.dataset.cancel)}`, { method: "DELETE" });
      } catch {}
      refresh();
    }
  });

  render();
  refresh().then(() => initial && state.list.some((x) => x.lang === initial) && show(initial));

  return {
    stop() {
      state.stopped = true;
      clearTimeout(state.timer);
    },
    lang: () => state.lang,
    show,
  };
}
