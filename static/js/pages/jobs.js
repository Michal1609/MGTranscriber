// Jobs: running and waiting jobs, history, retry, remove. Server-side filters and paging (GET /api/jobs).

import { api } from "../core/api.js";
import { applyI18n, has, t } from "../core/i18n.js";
import { href, isCurrent, registerPage } from "../core/router.js";
import { refreshSummary } from "../core/jobsummary.js";
import { $, date, esc, shortClock, toast } from "../core/util.js";
import { confirmDialog } from "../components/folders.js";
import { langName } from "../components/translations.js";
import {
  cancelJob, errorParts, isActive, isFailed, isUpload, jobLabel, phaseLabel, statusLabel, stepsHtml, streamLog,
} from "../components/jobs.js";

const PAGE_SIZE = 20;
const POLL_MS = 2500;
// tab id = the server's status group (see JobQueue.GROUPS); "all" has no filter
const TABS = [
  { id: "active", label: "jobs.tab.active" },
  { id: "done", label: "jobs.tab.done" },
  { id: "failed", label: "jobs.tab.failed" },
  { id: "all", label: "jobs.tab.all" },
];

const TEMPLATE = `
  <div class="section-head jobs-head">
    <h1 data-i18n="jobs.title"></h1>
    <button type="button" class="btn danger" id="j-clear" data-i18n="jobs.clear"></button>
  </div>
  <div class="tabs j-tabs" role="tablist" data-i18n-aria-label="jobs.tabs">
    ${TABS.map((tab) => `<button type="button" class="tab" role="tab" id="jt-${tab.id}" data-tab="${tab.id}"
      aria-controls="j-panel"><span data-i18n="${tab.label}"></span> <span class="tab-count" aria-hidden="true"></span></button>`).join("")}
  </div>
  <p class="sr-only" id="j-live" role="status" aria-live="polite"></p>
  <div id="j-panel" role="tabpanel" tabindex="-1">
    <p class="j-empty" id="j-empty" hidden></p>
    <ul class="jobs-list" id="j-list"></ul>
    <nav class="pager" id="j-pager" data-i18n-aria-label="jobs.pager" hidden>
      <button type="button" class="btn" id="j-prev" data-i18n="jobs.prev"></button>
      <span id="j-range"></span>
      <button type="button" class="btn" id="j-next" data-i18n="jobs.next"></button>
    </nav>
  </div>`;

function mount(el, params, query) {
  el.innerHTML = TEMPLATE;
  applyI18n(el);
  const list = $("#j-list", el);
  const live = $("#j-live", el);
  const rows = new Map(); // id -> { el, job, open, close }
  let tab = TABS.some((x) => x.id === query.tab) ? query.tab : null; // null: pick on the first load
  let offset = 0;
  let counts = null;
  let openWanted = query.open ? Number(query.open) : null;
  let timer = null;
  let loading = false;

  const say = (text) => {
    live.textContent = "";
    setTimeout(() => (live.textContent = text), 50);
  };

  /* --- rows --- */

  function createRow(job) {
    const li = document.createElement("li");
    li.className = "card job-row";
    li.dataset.id = job.id;
    li.innerHTML = `
      <div class="jr-head">
        <button type="button" class="jr-toggle" aria-expanded="false">
          <span class="jr-title"></span>
        </button>
        <span class="pill"></span>
      </div>
      <p class="jr-source"></p>
      <p class="jr-meta"></p>
      <p class="jr-nospeech" role="note" hidden></p>
      <p class="jr-translation" role="note" hidden></p>
      <div class="steps" hidden></div>
      <div class="job-error" hidden>
        <p class="je-text"></p>
        <details class="je-details"><summary></summary><pre></pre></details>
      </div>
      <div class="jr-actions">
        <button type="button" class="btn danger j-cancel"></button>
        <button type="button" class="btn j-retry"></button>
        <a class="btn j-open"></a>
        <button type="button" class="btn j-remove"></button>
      </div>
      <div class="jr-panel" hidden><pre class="log" tabindex="0"></pre></div>`;
    const row = { el: li, job, open: false, close: null };
    const q = (s) => li.querySelector(s);
    const panelId = `jr-panel-${job.id}`;
    q(".jr-panel").id = panelId;
    q(".jr-toggle").setAttribute("aria-controls", panelId);
    q(".je-details summary").textContent = t("jobs.error_details");
    q(".j-cancel").textContent = t("jobs.cancel");
    q(".j-retry").textContent = t("jobs.retry");
    q(".j-open").textContent = t("jobs.open");
    q(".j-remove").textContent = t("jobs.remove");
    q(".jr-toggle").addEventListener("click", () => toggle(row, !row.open));
    q(".j-cancel").addEventListener("click", async () => {
      const updated = await cancelJob(row.job);
      if (updated) load();
    });
    q(".j-retry").addEventListener("click", () => retry(row));
    q(".j-remove").addEventListener("click", () => remove(row));
    return row;
  }

  function updateRow(row, job) {
    row.job = job;
    const q = (s) => row.el.querySelector(s);
    q(".jr-title").textContent = jobLabel(job);
    const toggle = q(".jr-toggle");
    toggle.setAttribute("aria-label", `${jobLabel(job)} — ${t(row.open ? "jobs.hide_log" : "jobs.show_log")}`);
    const pill = q(".pill");
    pill.className = `pill ${job.status}`;
    pill.textContent = statusLabel(job);

    const source = q(".jr-source");
    const isVoiceScan = job.task === "voice_scan"; // no audio source and no engine: "all" / "default engine" would mislead
    const showSource = job.title && job.source && !isUpload(job) && !isVoiceScan;
    source.hidden = !showSource;
    source.textContent = showSource ? job.source : "";

    const bits = [];
    if (job.engine) bits.push(job.model ? `${job.engine} / ${job.model}` : job.engine);
    else if (!isVoiceScan) bits.push(t("jobs.engine_default"));
    if (job.separate_speakers) bits.push(t("jobs.separated"));
    bits.push(t("jobs.created", { time: date(job.created) }));
    if (job.finished) bits.push(t("jobs.finished", { time: date(job.finished) }));
    if (job.started) bits.push(t("jobs.took", { time: shortClock((job.finished || Date.now() / 1000) - job.started) }));
    const phase = phaseLabel(job);
    if (phase) bits.push(phase);
    q(".jr-meta").textContent = bits.join(" · ");

    const tr = q(".jr-translation"); // "Translate to" of the form: why no translation was queued
    tr.hidden = !job.translation?.error;
    tr.textContent = job.translation?.error ? t("jobs.translation_failed", { lang: langName(job.translation.language), reason: job.translation.error }) : "";

    const quiet = q(".jr-nospeech"); // the recording had no speech: the transcript is kept, but empty
    quiet.hidden = !job.no_speech;
    quiet.textContent = job.no_speech ? t("jobs.no_speech") : "";
    if (job.translation?.skipped) {
      tr.hidden = false;
      tr.textContent = t("jobs.translation_skipped", { lang: langName(job.translation.language) });
    }

    const steps = q(".steps");
    steps.hidden = job.status !== "running";
    if (job.status === "running") steps.innerHTML = stepsHtml(job);

    const err = q(".job-error");
    const { headline: raw, details } = errorParts(job.error);
    const headline = job.error_code && has(`errors.${job.error_code}`) ? t(`errors.${job.error_code}`) : raw;
    err.hidden = !job.error;
    q(".je-text").textContent = headline;
    q(".je-details").hidden = !details;
    q(".je-details pre").textContent = details;

    q(".j-cancel").hidden = !isActive(job);
    q(".j-retry").hidden = !isFailed(job) || isUpload(job); // an upload's file is deleted when its job ends
    const open = q(".j-open");
    open.hidden = !(job.status === "done" && job.result) || job.task === "voice_scan"; // its result is not a transcript
    if (job.result) open.href = href(`/t/${encodeURIComponent(job.result)}`);
    q(".j-remove").hidden = isActive(job);
  }

  function toggle(row, open) {
    row.open = open;
    const q = (s) => row.el.querySelector(s);
    q(".jr-toggle").setAttribute("aria-expanded", String(open));
    q(".jr-toggle").setAttribute("aria-label", `${jobLabel(row.job)} — ${t(open ? "jobs.hide_log" : "jobs.show_log")}`);
    q(".jr-panel").hidden = !open;
    row.el.classList.toggle("open", open);
    row.close?.();
    row.close = null;
    if (open) row.close = streamLog(q(".log"), row.job.id, { onState: (job) => updateRow(row, job) });
  }

  async function retry(row) {
    try {
      await api(`/api/jobs/${row.job.id}/retry`, { method: "POST" });
      toast(t("jobs.retried_toast"));
      say(t("jobs.retried_toast"));
      refreshSummary();
      await load();
    } catch (err) {
      toast(err.message);
    }
  }

  async function remove(row) {
    try {
      await api(`/api/jobs/${row.job.id}/history`, { method: "DELETE" });
      say(t("jobs.removed_toast"));
      toast(t("jobs.removed_toast"));
      $("#j-panel", el).focus();
      await load();
    } catch (err) {
      toast(err.message);
    }
  }

  /* --- loading, tabs, paging --- */

  async function load() {
    if (loading) return;
    loading = true;
    try {
      const status = tab === "all" || tab === null ? "" : tab;
      const qs = new URLSearchParams({ limit: PAGE_SIZE, offset });
      if (status) qs.set("status", status);
      let data = await api(`/api/jobs?${qs}`);
      if (!isCurrent("jobs")) return;
      if (tab === null) { // first visit without a tab: Active when something runs or waits, else the whole history
        tab = data.counts.active > 0 ? "active" : "all";
        if (tab === "active") data = await api(`/api/jobs?status=active&limit=${PAGE_SIZE}&offset=0`);
        if (!isCurrent("jobs")) return;
      }
      if (data.total > 0 && data.items.length === 0 && offset > 0) { // the last page emptied out
        offset = Math.max(0, Math.floor((data.total - 1) / PAGE_SIZE) * PAGE_SIZE);
        loading = false;
        return load();
      }
      counts = data.counts;
      render(data);
    } catch (err) {
      if (isCurrent("jobs")) $("#j-empty", el).textContent = t("jobs.load_failed", { error: err.message });
      $("#j-empty", el).hidden = false;
    } finally {
      loading = false;
    }
  }

  function render(data) {
    for (const x of TABS) {
      const button = $(`#jt-${x.id}`, el);
      const selected = x.id === tab;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      button.querySelector(".tab-count").textContent = counts ? `(${counts[x.id]})` : "";
    }
    $("#j-panel", el).setAttribute("aria-labelledby", `jt-${tab}`);
    $("#j-clear", el).disabled = !counts || counts.finished === 0;

    const seen = new Set();
    for (const job of data.items) {
      seen.add(job.id);
      let row = rows.get(job.id);
      if (!row) {
        row = createRow(job);
        rows.set(job.id, row);
      }
      list.append(row.el); // server order
      updateRow(row, job);
    }
    for (const [id, row] of rows) {
      if (!seen.has(id)) {
        row.close?.();
        row.el.remove();
        rows.delete(id);
      }
    }
    const empty = $("#j-empty", el);
    empty.hidden = data.items.length > 0;
    empty.textContent = t(`jobs.empty.${tab}`);

    const pager = $("#j-pager", el);
    pager.hidden = data.total <= PAGE_SIZE;
    $("#j-prev", el).disabled = offset === 0;
    $("#j-next", el).disabled = offset + PAGE_SIZE >= data.total;
    $("#j-range", el).textContent = t("jobs.range", {
      from: data.total ? offset + 1 : 0, to: Math.min(offset + PAGE_SIZE, data.total), total: data.total,
    });

    if (openWanted != null) { // arrived from the home page's "log" link
      const row = rows.get(openWanted);
      openWanted = null;
      if (row) {
        toggle(row, true);
        row.el.scrollIntoView({ block: "center" });
        row.el.querySelector(".jr-toggle").focus({ preventScroll: true });
      }
    }
    history.replaceState(null, "", href("/jobs", { tab, ...(offset ? { page: offset / PAGE_SIZE + 1 } : {}) }));
  }

  function setTab(id, focus = false) {
    if (id === tab) return;
    tab = id;
    offset = 0;
    for (const row of rows.values()) row.close?.();
    rows.clear();
    list.replaceChildren();
    load().then(() => focus && $(`#jt-${id}`, el).focus());
  }

  const tabs = $(".j-tabs", el);
  tabs.addEventListener("click", (e) => {
    const button = e.target.closest("[data-tab]");
    if (button) setTab(button.dataset.tab);
  });
  tabs.addEventListener("keydown", (e) => {
    const ids = TABS.map((x) => x.id);
    const i = ids.indexOf(document.activeElement?.dataset?.tab);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: ids.length - 1 }[e.key];
    if (i < 0 || next === undefined) return;
    e.preventDefault();
    const id = ids[(next + ids.length) % ids.length];
    setTab(id);
    $(`#jt-${id}`, el).focus();
  });

  const turn = (delta) => {
    offset = Math.max(0, offset + delta * PAGE_SIZE);
    for (const row of rows.values()) row.close?.();
    rows.clear();
    list.replaceChildren();
    load().then(() => $("#j-panel", el).focus());
  };
  $("#j-prev", el).addEventListener("click", () => turn(-1));
  $("#j-next", el).addEventListener("click", () => turn(1));

  $("#j-clear", el).addEventListener("click", async () => {
    const n = counts?.finished || 0;
    const ok = await confirmDialog({
      title: t("jobs.clear_title"), text: t("jobs.clear_text", { n }), confirm: t("jobs.clear_confirm"), danger: true,
    });
    if (!ok) return;
    try {
      const { removed } = await api("/api/jobs/clear?status=finished", { method: "POST" });
      toast(t("jobs.cleared_toast", { n: removed }));
      say(t("jobs.cleared_toast", { n: removed }));
      offset = 0;
      refreshSummary();
      await load();
    } catch (err) {
      toast(err.message);
    }
  });

  if (query.page && Number(query.page) > 1) offset = (Number(query.page) - 1) * PAGE_SIZE;
  load();
  timer = setInterval(() => !document.hidden && !document.querySelector("dialog[open]") && load(), POLL_MS);
  return () => {
    clearInterval(timer);
    for (const row of rows.values()) row.close?.();
  };
}

registerPage({ id: "jobs", path: "/jobs", nav: { label: "nav.jobs", order: 5, badge: true }, title: () => t("jobs.title"), mount });
