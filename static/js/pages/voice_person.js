// One remembered person across the whole archive: transcripts to confirm (with a 5 s excerpt to listen
// to), everywhere they speak, merging a duplicate person into this one, and a search limited to them.

import { api } from "../core/api.js";
import { applyI18n, locale, t } from "../core/i18n.js";
import { href, isCurrent, registerPage } from "../core/router.js";
import { $, date, duration, esc, pollTimeout, shortClock, toast } from "../core/util.js";

const TEMPLATE = `
  <header class="detail-head">
    <a class="muted" href="#/voices" data-i18n="voices.back"></a>
    <h1 class="vp-name"></h1>
    <p class="meta vp-meta"></p>
  </header>
  <section class="card vp-box vp-actions">
    <button class="btn vp-scan" data-i18n="voices.scan_person"></button>
    <span class="muted vp-job" role="status"></span>
  </section>
  <section class="card vp-box" aria-labelledby="vp-sugg-h">
    <h2 id="vp-sugg-h" data-i18n="voices.sugg_title"></h2>
    <div class="vp-bulk">
      <label for="vp-min" data-i18n="voices.accept_above"></label>
      <input id="vp-min" class="vp-min" type="number" min="0" max="1" step="0.05" value="0.60">
      <button class="btn vp-accept-all" data-i18n="voices.accept_all"></button>
    </div>
    <ul class="vp-list vp-sugg"></ul>
  </section>
  <section class="card vp-box" aria-labelledby="vp-app-h">
    <h2 id="vp-app-h" data-i18n="voices.app_title"></h2>
    <ul class="vp-list vp-app"></ul>
  </section>
  <section class="card vp-box" aria-labelledby="vp-search-h">
    <h2 id="vp-search-h" data-i18n="voices.search_title"></h2>
    <input class="vp-q" type="search" data-i18n-placeholder="voices.search_ph" data-i18n-aria-label="voices.search_ph">
    <ul class="vp-list vp-hits" aria-live="polite"></ul>
  </section>
  <section class="card vp-box" aria-labelledby="vp-merge-h">
    <h2 id="vp-merge-h" data-i18n="voices.merge_title"></h2>
    <p class="muted" data-i18n="voices.merge_hint"></p>
    <div class="vp-merge-row">
      <select class="vp-other" data-i18n-aria-label="voices.merge_pick"></select>
      <button class="btn danger vp-merge" data-i18n="voices.merge"></button>
    </div>
  </section>
  <audio class="vp-audio" preload="none"></audio>`;

const score = (s) => (s == null ? "" : s.toLocaleString(locale(), { minimumFractionDigits: 2, maximumFractionDigits: 2 }));

function mount(view, { id }) {
  // the view element outlives the page: listeners go on an element of our own
  view.innerHTML = `<div class="vp">${TEMPLATE}</div>`;
  const el = view.firstElementChild;
  applyI18n(el);
  const q = (cls) => $(`.${cls}`, el);
  const base = `/api/voices/people/${encodeURIComponent(id)}`;
  const audio = q("vp-audio");
  let name = "";
  let alive = true;
  let timer = null;
  let armed = null;
  let searchTimer = null;
  let stopAt = null;

  audio.addEventListener("timeupdate", () => {
    if (stopAt != null && audio.currentTime >= stopAt) {
      audio.pause();
      stopAt = null;
    }
  });

  load();

  async function load() {
    clearTimeout(timer);
    let sugg, apps, all;
    try {
      [sugg, apps, all] = await Promise.all([api(`${base}/suggestions`), api(`${base}/appearances`), api("/api/voices")]);
    } catch (err) {
      if (alive && isCurrent("voice-person", { id })) view.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
      return;
    }
    if (!alive || !isCurrent("voice-person", { id })) return;
    name = apps.name;
    q("vp-name").textContent = name;
    q("vp-meta").textContent = [t("voices.appearances_n", { n: apps.transcripts }), duration(apps.speech_s)].join(" · ");
    q("vp-scan").textContent = t("voices.scan_person", { name });
    renderSuggestions(sugg.suggestions);
    renderAppearances(apps.appearances);
    const others = all.people.filter((p) => p.id !== id);
    q("vp-other").innerHTML = `<option value="">${esc(t("voices.merge_pick"))}</option>` +
      others.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("");
    q("vp-merge").disabled = !others.length;
    const job = sugg.scan_job || all.scan_job;
    q("vp-scan").disabled = !!job;
    q("vp-job").textContent = !job ? "" : job.status === "running"
      ? t("voices.scan_running", { pct: job.percent != null ? Math.round(job.percent) + " %" : "" })
      : t("voices.scan_waiting");
    if (job) pollTimeout((h) => (timer = h), load, 2000);
  }

  const status = (s) => `<span class="voice-tag${s === "confirmed" || s === "manual" ? " ok" : ""}">${esc(t(`voices.status.${s}`))}</span>`;

  function renderSuggestions(items) {
    const list = q("vp-sugg");
    q("vp-accept-all").disabled = !items.length;
    if (!items.length) {
      list.innerHTML = `<li class="empty">${esc(t("voices.sugg_empty"))}</li>`;
      return;
    }
    list.innerHTML = items.map((s) => `
      <li data-transcript="${esc(s.transcript_id)}" data-label="${esc(s.label)}"
          data-audio="${esc(s.audio || "")}" data-start="${s.start ?? ""}">
        <div class="vp-row-head">
          <a href="${href(`/t/${encodeURIComponent(s.transcript_id)}`, s.start != null ? { t: s.start } : undefined)}">${esc(s.title)}</a>
          <span class="meta"><span>${esc(s.label)}</span><span>${esc(score(s.score))}</span><span>${esc(duration(s.speech_s || 0))}</span></span>
          ${status(s.status)}
        </div>
        ${s.snippet ? `<p class="vp-snippet">“${esc(s.snippet)}”</p>` : ""}
        <div class="vp-row-actions">
          ${s.audio && s.start != null ? `<button class="btn vp-play">${esc(t("voices.play"))}</button>` : ""}
          <button class="btn vp-yes">${esc(t("voices.accept"))}</button>
          <button class="btn vp-no">${esc(t("voices.reject"))}</button>
        </div>
      </li>`).join("");
  }

  function renderAppearances(items) {
    const list = q("vp-app");
    if (!items.length) {
      list.innerHTML = `<li class="empty">${esc(t("voices.app_empty"))}</li>`;
      return;
    }
    list.innerHTML = items.map((a) => `
      <li>
        <div class="vp-row-head">
          <a href="${href(`/t/${encodeURIComponent(a.transcript_id)}`)}">${esc(a.title)}</a>
          <span class="meta"><span>${esc(a.label)}</span>${a.speech_s ? `<span>${esc(duration(a.speech_s))}</span>` : ""}
            ${a.date ? `<span>${esc(date(a.date))}</span>` : ""}</span>
          ${status(a.status)}
        </div>
      </li>`).join("");
  }

  async function post(path, body, okText) {
    try {
      const res = await api(`${base}/${path}`, { method: "POST", body: JSON.stringify(body) });
      if (okText) toast(okText(res));
      return res;
    } catch (err) {
      toast(err.message);
    }
  }

  el.addEventListener("click", async (e) => {
    const row = e.target.closest("li[data-transcript]");
    if (e.target.closest(".vp-play") && row) {
      const file = `/api/transcripts/${encodeURIComponent(row.dataset.transcript)}/files/${encodeURIComponent(row.dataset.audio)}`;
      const start = Number(row.dataset.start);
      const go = () => {
        audio.currentTime = start;
        stopAt = start + 5;
        audio.play().catch(() => toast(t("errors.network")));
      };
      if (!audio.src.endsWith(file)) {
        audio.src = file;
        audio.addEventListener("loadedmetadata", go, { once: true });
        audio.load();
      } else go();
    } else if (e.target.closest(".vp-yes, .vp-no") && row) {
      const yes = !!e.target.closest(".vp-yes");
      const res = await post(`suggestions/${yes ? "accept" : "reject"}`,
        { items: [{ transcript: row.dataset.transcript, label: row.dataset.label }] });
      if (res && res.failed?.length) toast(t(`errors.${res.failed[0].code}`));
      load();
    } else if (e.target.closest(".vp-accept-all")) {
      const min = Number(q("vp-min").value);
      const res = await post("suggestions/accept", { min_score: Number.isFinite(min) ? min : 0.6 });
      if (res) toast(t("voices.bulk_done", { n: res.accepted }) + (res.failed.length ? " · " + t("voices.bulk_failed", { n: res.failed.length }) : ""));
      load();
    } else if (e.target.closest(".vp-scan")) {
      e.target.disabled = true;
      await api("/api/voices/scan", { method: "POST", body: JSON.stringify({ person_id: id }) }).then(
        () => toast(t("voices.scan_queued")), (err) => toast(err.message));
      load();
    } else if (e.target.closest(".vp-merge")) {
      const other = q("vp-other").value;
      if (!other) return toast(t("voices.merge_pick"));
      const button = e.target.closest(".vp-merge");
      if (armed !== button) { // second click confirms
        armed = button;
        button.dataset.label = button.textContent;
        button.textContent = t("voices.confirm");
        setTimeout(() => { if (armed === button) { button.textContent = button.dataset.label; armed = null; } }, 4000);
        return;
      }
      armed = null;
      button.textContent = button.dataset.label;
      const res = await post("merge", { other }, () => t("voices.merged"));
      if (res) load();
    }
  });

  q("vp-q").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(search, 250);
  });

  async function search() {
    const text = q("vp-q").value.trim();
    const list = q("vp-hits");
    if (text.length < 2) return list.replaceChildren();
    try {
      const data = await api(`/api/search?${new URLSearchParams({ q: text, person: id, limit: 30 })}`);
      list.innerHTML = data.total
        ? data.results.map((h) => `<li><a class="hit" href="${href(`/t/${encodeURIComponent(h.transcript_id)}`, { t: Math.round(h.start * 10) / 10 })}">
            <span class="time">${esc(shortClock(h.start))}</span><span class="who">${esc(h.title)}</span>
            <span class="snippet">${esc(h.snippet)}</span></a></li>`).join("")
        : `<li class="empty">${esc(t("search.none"))}</li>`;
    } catch (err) {
      list.innerHTML = `<li class="empty">${esc(err.message)}</li>`;
    }
  }

  return () => {
    alive = false;
    clearTimeout(timer);
    clearTimeout(searchTimer);
    audio.pause();
  };
}

registerPage({
  id: "voice-person",
  path: "/voices/:id",
  title: () => t("voices.title"),
  mount,
});
