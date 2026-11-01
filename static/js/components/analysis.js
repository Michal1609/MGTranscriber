// LLM panels of the transcript detail page: "Summary" (analysis) and "Ask".
// mountAnalysis() fills two containers and returns a cleanup function.

import { api } from "../core/api.js";
import { applyI18n, t } from "../core/i18n.js";
import { $, date, esc, pollTimeout, shortClock, speakerLabel, toast } from "../core/util.js";

const CHOICE_KEY = "llm.choice";
const LANGS = ["auto", "en", "cs"];  // "auto" = the transcript's language

function loadChoice() {
  try {
    return JSON.parse(localStorage.getItem(CHOICE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveChoice(choice) {
  try {
    localStorage.setItem(CHOICE_KEY, JSON.stringify(choice));
  } catch {}
}

const timeButton = (seconds, label = shortClock(seconds)) =>
  `<button type="button" class="time-link" data-seek="${seconds}">${esc(label)}</button>`;

function errorHtml(err) {
  const code = err.code || "analysis_failed";
  const head = t(`llm.err.${code}`);
  const headline = head === `llm.err.${code}` ? t("llm.err.generic") : head;
  const detail = err.message && err.message !== headline ? `<small>${esc(err.message)}</small>` : "";
  return `<div class="llm-error" role="alert"><strong>${esc(headline)}</strong>${detail}</div>`;
}

export function mountAnalysis({ id, summaryEl, askEl, names, colorOf, seek }) {
  const base = `/api/transcripts/${encodeURIComponent(id)}`;
  const choice = loadChoice();
  let providers = [];
  let defaults = {};
  let poll = null;
  let disposed = false;
  let analysis = null;
  let run = null;

  summaryEl.innerHTML = `
    <div class="card llm-controls">
      <label>${esc(t("llm.provider"))}<select class="l-provider"></select></label>
      <label>${esc(t("llm.model"))}<input class="l-model" list="l-models-${id.length}" autocomplete="off"><datalist id="l-models-${id.length}" class="l-models"></datalist></label>
      <label>${esc(t("llm.language"))}<select class="l-lang"></select></label>
      <div class="llm-actions">
        <button class="btn primary l-run"></button>
        <button class="btn l-cancel" hidden></button>
        <button class="btn danger l-delete" hidden></button>
      </div>
      <p class="llm-note l-note" hidden></p>
    </div>
    <div class="l-progress" hidden><div class="bar"><i></i></div><small class="l-progress-text"></small></div>
    <div class="l-error"></div>
    <div class="l-result"></div>`;
  askEl.innerHTML = `
    <form class="card llm-ask l-form">
      <input class="l-question" type="text" maxlength="1000" autocomplete="off" data-i18n-placeholder="llm.ask_placeholder" data-i18n-aria-label="llm.ask_placeholder">
      <button class="btn primary l-send" type="submit" data-i18n="llm.ask_send"></button>
    </form>
    <p class="llm-note l-ask-note"></p>
    <div class="l-answers"></div>`;
  applyI18n(summaryEl);
  applyI18n(askEl);

  const s = (cls) => $(`.${cls}`, summaryEl);
  const a = (cls) => $(`.${cls}`, askEl);

  /* --- provider controls --- */

  function selected() {
    return { provider: s("l-provider").value, model: s("l-model").value.trim(), language: s("l-lang").value };
  }

  function fillModels() {
    const p = providers.find((x) => x.name === s("l-provider").value);
    s("l-models").innerHTML = (p?.models || []).map((m) => `<option value="${esc(m)}">`).join("");
    const note = s("l-note");
    let text = "";
    if (p && !p.available) text = p.cloud ? t("llm.no_key", { env: p.key_env }) : t("llm.no_ollama");
    else if (p?.cloud) text = t("llm.cloud_warning");
    note.hidden = !text;
    note.textContent = text;
    note.classList.toggle("warn", !!p?.cloud);
    a("l-ask-note").textContent = p?.cloud ? t("llm.cloud_warning") : "";
  }

  async function loadProviders() {
    try {
      const info = await api("/api/llm");
      providers = info.providers;
      defaults = info.config;
    } catch (err) {
      s("l-error").innerHTML = errorHtml(err);
      return;
    }
    if (disposed) return;
    s("l-provider").innerHTML = providers
      .map((p) => `<option value="${esc(p.name)}">${esc(p.label)}${p.available ? "" : " ⚠"}</option>`)
      .join("");
    s("l-provider").value = providers.some((p) => p.name === choice.provider) ? choice.provider : defaults.provider;
    const p = providers.find((x) => x.name === s("l-provider").value);
    s("l-model").value = choice.provider === p.name && choice.model ? choice.model : p.default_model;
    s("l-lang").innerHTML = LANGS.map((l) => `<option value="${l}">${esc(t(l === "auto" ? "llm.lang_auto" : `lang.${l}`))}</option>`).join("");
    s("l-lang").value = LANGS.includes(choice.language) ? choice.language : LANGS.includes(defaults.language) ? defaults.language : "auto";
    fillModels();
  }

  s("l-provider").addEventListener("change", () => {
    const p = providers.find((x) => x.name === s("l-provider").value);
    s("l-model").value = p.default_model;
    fillModels();
    remember();
  });
  for (const cls of ["l-model", "l-lang"]) s(cls).addEventListener("change", remember);

  function remember() {
    Object.assign(choice, selected());
    saveChoice(choice);
  }

  /* --- running and showing the analysis --- */

  async function refresh() {
    let state;
    try {
      state = await api(`${base}/analysis`);
    } catch (err) {
      s("l-error").innerHTML = errorHtml(err);
      return;
    }
    if (disposed) return;
    analysis = state.analysis;
    run = state.run;
    render();
    clearTimeout(poll);
    if (run?.status === "running") pollTimeout((h) => (poll = h), refresh, 1500);
  }

  function render() {
    const running = run?.status === "running";
    s("l-run").textContent = t(analysis ? "llm.rerun" : "llm.run");
    s("l-run").disabled = running;
    s("l-cancel").hidden = !running;
    s("l-cancel").textContent = t("llm.cancel");
    s("l-delete").hidden = running || !analysis;
    s("l-delete").textContent = t("llm.delete");
    s("l-progress").hidden = !running;
    if (running) {
      $(".bar i", summaryEl).style.width = `${run.percent}%`;
      const phase = String(run.message || "").split(" ")[0];
      s("l-progress-text").textContent = `${t(`llm.phase.${phase}`) === `llm.phase.${phase}` ? run.message : t(`llm.phase.${phase}`)}${
        run.message.includes(" ") ? " " + run.message.split(" ")[1] : ""} · ${Math.round(run.percent)} %`;
    }
    s("l-error").innerHTML = run?.status === "error" && run.error ? errorHtml(run.error) : "";
    const box = s("l-result");
    if (!analysis) {
      box.innerHTML = running ? "" : `<div class="empty">${esc(t("llm.empty"))}</div>`;
      return;
    }
    box.innerHTML = (analysis.stale ? `<div class="empty">${esc(t("llm.stale"))}</div>` : "") + renderAnalysis(analysis);
  }

  function who(speaker) {
    return `<span class="who" style="color:${colorOf[speaker] || "var(--muted)"}">${esc(speakerLabel(speaker, names))}</span>`;
  }

  function renderAnalysis(r) {
    const parts = [];
    parts.push(`<p class="meta llm-meta"><span>${esc(t("llm.made_by", { provider: r.provider, model: r.model }))}</span>
      <span>${esc(date(Date.parse(r.created) / 1000))}</span><span>${esc(t("common.duration_s", { n: Math.round(r.seconds || 0) }))}</span></p>`);
    parts.push(`<section class="card llm-section"><h2>${esc(t("llm.summary"))}</h2><p>${esc(r.summary)}</p></section>`);
    if (r.key_points?.length)
      parts.push(`<section class="card llm-section"><h2>${esc(t("llm.key_points"))}</h2><ul>${r.key_points
        .map((p) => `<li>${esc(p)}</li>`).join("")}</ul></section>`);
    if (r.chapters?.length)
      parts.push(`<section class="card llm-section"><h2>${esc(t("llm.chapters"))}</h2><ol class="chapters">${r.chapters
        .map((c) => `<li>${timeButton(c.start)}<div><strong>${esc(c.title)}</strong>${c.summary ? `<p>${esc(c.summary)}</p>` : ""}</div></li>`)
        .join("")}</ol></section>`);
    if (r.quotes?.length)
      parts.push(`<section class="card llm-section"><h2>${esc(t("llm.quotes"))}</h2>${r.quotes
        .map((q) => `<blockquote>${esc(q.text)}<footer>${who(q.speaker)} ${timeButton(q.time)}</footer></blockquote>`).join("")}</section>`);
    parts.push(`<section class="card llm-section"><h2>${esc(t("llm.actions"))}</h2>${
      r.action_items?.length
        ? `<ul class="actions-list">${r.action_items.map((x) => `<li>${x.owner ? `<strong>${esc(x.owner)}:</strong> ` : ""}${esc(x.text)}${
            x.time != null ? " " + timeButton(x.time) : ""}</li>`).join("")}</ul>`
        : `<p class="muted">${esc(t("llm.no_actions"))}</p>`}</section>`);
    return parts.join("");
  }

  s("l-run").addEventListener("click", async () => {
    remember();
    s("l-error").innerHTML = "";
    const body = selected();
    if (!body.model) delete body.model;
    try {
      await api(`${base}/analysis`, { method: "POST", body: JSON.stringify(body) });
    } catch (err) {
      s("l-error").innerHTML = errorHtml(err);
      return;
    }
    refresh();
  });

  s("l-cancel").addEventListener("click", async () => {
    await api(`${base}/analysis/cancel`, { method: "POST" }).catch(() => {});
    refresh();
  });

  s("l-delete").addEventListener("click", async () => {
    if (!confirm(t("llm.delete_confirm"))) return;
    await api(`${base}/analysis`, { method: "DELETE" }).catch((err) => toast(err.message));
    refresh();
  });

  summaryEl.addEventListener("click", (e) => {
    const link = e.target.closest("[data-seek]");
    if (link) seek(Number(link.dataset.seek));
  });

  /* --- questions --- */

  a("l-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = a("l-question");
    const question = input.value.trim();
    if (question.length < 2) return;
    remember();
    const entry = document.createElement("article");
    entry.className = "card qa";
    entry.innerHTML = `<h3>${esc(question)}</h3><div class="qa-body"><span class="loading">${esc(t("llm.thinking"))}</span></div>`;
    a("l-answers").prepend(entry);
    input.value = "";
    a("l-send").disabled = true;
    const body = { ...selected(), question };
    if (!body.model) delete body.model;
    try {
      const res = await api(`${base}/ask`, { method: "POST", body: JSON.stringify(body) });
      $(".qa-body", entry).innerHTML = res.found
        ? `<p>${esc(res.answer)}</p>${res.citations.map((c) =>
            `<blockquote>${esc(c.text)}<footer>${who(c.speaker)} ${timeButton(c.time)}</footer></blockquote>`).join("")}`
        : `<p class="muted">${esc(t("llm.not_found"))}</p>`;
    } catch (err) {
      $(".qa-body", entry).innerHTML = errorHtml(err);
    } finally {
      a("l-send").disabled = false;
    }
  });

  askEl.addEventListener("click", (e) => {
    const link = e.target.closest("[data-seek]");
    if (link) seek(Number(link.dataset.seek));
  });

  loadProviders().then(refresh);
  return () => {
    disposed = true;
    clearTimeout(poll);
  };
}
