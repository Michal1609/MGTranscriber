// Job helpers shared by the home page's "now running" strip and the Jobs page.

import "../i18n/translate.js";
import { api } from "../core/api.js";
import { has, t } from "../core/i18n.js";
import { href } from "../core/router.js";
import { refreshSummary, watchSummary } from "../core/jobsummary.js";
import { esc, toast } from "../core/util.js";

// Steps shown on a job and which job phases (see mgtranscriber/progress.py) belong to them.
const STEPS = [
  { key: "prepare", phases: ["queued", "starting", "loading"] },
  { key: "download", phases: ["download"] },
  { key: "transcribe", phases: ["transcribe"] },
  { key: "align", phases: ["align"] },
  { key: "diarize", phases: ["diarize", "writing"] },
];

// A translation job goes through different steps (task "translate": mgtranscriber/llm/translate.py).
const TRANSLATE_STEPS = [
  { key: "prepare", phases: ["queued", "starting", "loading"] },
  { key: "translate", phases: ["translate"] },
  { key: "save", phases: ["writing"] },
];
// Separated voices (mgtranscriber/separated.py): separation first, then every voice is transcribed and aligned in
// turn; transcribe and align are one step so that the steps do not jump back with each voice.
const SEPARATED_STEPS = [
  { key: "prepare", phases: ["queued", "starting", "loading"] },
  { key: "download", phases: ["download"] },
  { key: "separate", phases: ["separate"] },
  { key: "transcribe", phases: ["transcribe", "align", "diarize"] },
  { key: "save", phases: ["writing"] },
];
const stepsOf = (job) => (job.task === "translate" ? TRANSLATE_STEPS : job.separate_speakers ? SEPARATED_STEPS : STEPS);

export const isActive = (job) => job.status === "queued" || job.status === "running";
export const isFailed = (job) => ["error", "cancelled", "interrupted"].includes(job.status);

// A web upload's source is a path inside the server's data folder (deleted when its job ends).
export const isUpload = (job) => job.upload === true || /(^|[\\/])uploads[\\/]/.test(job.source || "");
const baseName = (source) => String(source).split(/[\\/]/).pop();

// "Voiceprints: <transcript>" for task jobs (see mgtranscriber/tasks.py), else the title or source.
export function jobLabel(job) {
  const task = job.task && job.task !== "transcribe" ? (has(`jobs.task.${job.task}`) ? t(`jobs.task.${job.task}`) : job.task) : "";
  const name = job.title || (isUpload(job) ? baseName(job.source) : job.source);
  const into = job.task === "translate" && job.language ? ` → ${job.language}` : "";
  return (task ? `${task}: ` : "") + name + into;
}

export function statusLabel(job) {
  return job.status === "queued" && job.position ? t("jobs.queued_at", { n: job.position }) : t(`jobs.status.${job.status}`);
}

export function phaseLabel(job) {
  const step = stepsOf(job).find((s) => s.phases.includes(job.phase));
  if (!step || job.status !== "running") return "";
  const pct = job.percent != null && job.phase !== "writing" ? ` ${Math.round(job.percent)} %` : "";
  return t(`jobs.step.${step.key}`) + pct;
}

// A failure is shown as one readable line; anything further (a traceback) goes behind "details".
export function errorParts(error) {
  const text = String(error || "").trim();
  const [headline, ...rest] = text.split("\n");
  return { headline: headline.trim(), details: rest.join("\n").trim() };
}

export function stepsHtml(job) {
  const steps = stepsOf(job).filter((s) => !(job.no_diarize && s.key === "diarize"));
  let current = steps.findIndex((s) => s.phases.includes(job.phase));
  if (job.status === "done") current = steps.length;
  return steps
    .map((s, i) => {
      const active = i === current && job.status === "running";
      const state = i < current ? "done" : active ? "active" : "";
      const pct = active && job.percent != null && job.phase !== "writing" ? Math.round(job.percent) : null;
      const style = pct != null ? ` style="--p:${pct}%"` : "";
      const label = t(`jobs.step.${s.key}`) + (pct != null ? ` ${pct} %` : "");
      return `<div class="step ${state}${pct != null ? " determinate" : ""}"${style}><i></i><span>${esc(label)}</span></div>`;
    })
    .join("");
}

// Live log of a job (SSE: log lines, state changes, end). Returns the function that closes the stream.
export function streamLog(pre, jobId, { onState } = {}) {
  pre.textContent = "";
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.addEventListener("log", (e) => {
    const nearBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.append(JSON.parse(e.data) + "\n");
    if (nearBottom) pre.scrollTop = pre.scrollHeight;
  });
  source.addEventListener("state", (e) => onState?.(JSON.parse(e.data)));
  source.addEventListener("end", () => source.close());
  return () => source.close();
}

export async function cancelJob(job) {
  try {
    const updated = await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    toast(t("jobs.cancelling"));
    refreshSummary();
    return updated;
  } catch (err) {
    toast(err.message);
    return null;
  }
}

// Home page: the single running job (title, phase, %, Cancel, link to its log) and a link to the Jobs page with
// "+N queued · M failed recently". Hidden while nothing runs or waits. onFinished(job) fires when a job ends.
export function mountRunning(box, { onFinished } = {}) {
  box.className = "card running-strip";
  box.hidden = true;
  box.innerHTML = `
    <div class="rs-main">
      <span class="pill running"></span>
      <span class="rs-title"></span>
      <span class="rs-phase"></span>
    </div>
    <div class="rs-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" hidden><i></i></div>
    <div class="rs-foot">
      <a class="rs-more"></a>
      <span class="rs-actions">
        <a class="btn rs-log"></a>
        <button type="button" class="btn danger rs-cancel"></button>
      </span>
    </div>
    <p class="sr-only" role="status" aria-live="polite"></p>`;
  const q = (s) => box.querySelector(s);
  const live = q('[role="status"]');
  let runningId = null;
  let announced = "";

  q(".rs-cancel").textContent = t("jobs.cancel");
  q(".rs-log").textContent = t("jobs.log_link");
  q(".pill").textContent = t("jobs.status.running");
  q(".rs-cancel").addEventListener("click", async () => {
    if (runningId != null) await cancelJob({ id: runningId });
  });

  async function ended(id) {
    try {
      const job = await api(`/api/jobs/${id}`);
      if (job.status === "done") toast(t("jobs.finished_toast"));
      else if (job.status === "error") toast(t("jobs.failed_toast"));
      onFinished?.(job);
    } catch {
      onFinished?.(null); // removed from the history meanwhile
    }
  }

  function render(s) {
    const job = s.running;
    box.hidden = !job && s.queued === 0;
    const more = [];
    if (s.queued) more.push(t("jobs.more_queued", { n: s.queued }));
    if (s.failed_recent) more.push(t("jobs.more_failed", { n: s.failed_recent }));
    const link = q(".rs-more");
    link.textContent = more.join(" · ") || t("jobs.all_jobs");
    link.href = href("/jobs");
    q(".rs-main").hidden = q(".rs-foot .rs-actions").hidden = !job;
    if (job) {
      q(".rs-title").textContent = job.title || job.source;
      q(".rs-title").title = job.title ? job.source : "";
      const pct = job.percent != null && job.phase !== "writing" ? Math.round(job.percent) : null;
      q(".rs-phase").textContent = phaseLabel(job);
      const bar = q(".rs-bar");
      bar.hidden = pct == null;
      if (pct != null) {
        bar.setAttribute("aria-valuenow", pct);
        bar.firstElementChild.style.width = `${pct}%`;
      }
      q(".rs-log").href = href("/jobs", { open: job.id });
      const now = t("jobs.now_running", { title: job.title || job.source });
      if (job.id !== runningId && announced !== now) live.textContent = announced = now;
    }
    const id = job ? job.id : null;
    if (runningId != null && id !== runningId) ended(runningId);
    runningId = id;
  }

  return watchSummary(render);
}
