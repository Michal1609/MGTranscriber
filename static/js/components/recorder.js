// Recorder: record the microphone and / or a shared tab in the browser and transcribe it afterwards (not live).
//
// The audio is cut in chunks (MediaRecorder timeslice) which are uploaded to /api/recordings WHILE recording, so a
// crashed tab, a closed lid or a sleeping computer loses at most the last chunk. The recording state lives in this
// module (not in the page), so it keeps running when the user opens another page of the app; the panel just binds
// to it again. After a reload, unfinished sessions on the server are listed and can be finished or discarded.

import { showLogin } from "./login.js";
import { refreshSummary } from "../core/jobsummary.js";
import { has, t } from "../core/i18n.js";
import { $, esc, toast } from "../core/util.js";
import "../i18n/recorder.js";

const SLICE_MS = 5000;
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4;codecs=mp4a.40.2", "audio/mp4"];
const SILENCE_AFTER_MS = 8000;
const TITLE_PREFIX = /^● \S+ — /;

// --- the one recording of this browser tab ---------------------------------------------------------------------
const S = {
  phase: "idle", // idle | starting | recording | paused | finishing | failed
  id: null,
  mime: "",
  source: "microphone",
  total: 0, // chunks produced so far
  acked: 0, // chunks the server has (contiguous from 0)
  pending: [], // [{ n, blob, t }] waiting for the server
  uploadedMs: 0,
  elapsedBase: 0, // ms recorded before the current run (pause)
  runStart: 0,
  level: 0,
  silentSince: 0,
  error: "",
  note: "",
  announce: "",
  lastMinute: 0,
  fields: {},
  getFields: null,
  onJob: null,
  recorder: null,
  ctx: null,
  streams: [],
  wake: null,
  timer: null,
  pumping: false,
  fatal: null,
  retrying: false,
};
const views = new Set(); // mounted panels
const emit = () => views.forEach((v) => v.update());
const rerender = () => views.forEach((v) => v.render());

const elapsedMs = () => S.elapsedBase + (S.phase === "recording" ? performance.now() - S.runStart : 0);

export function clock(ms) {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(h ? 2 : 1, "0");
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

const pad = (n) => String(n).padStart(2, "0");
const stamp = (d = new Date()) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}-${pad(d.getMinutes())}`;

export function support() {
  if (typeof MediaRecorder === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return window.isSecureContext ? "unsupported" : "insecure";
  }
  return window.isSecureContext ? "ok" : "insecure";
}

function pickMime() {
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) || "";
}

// --- server calls -----------------------------------------------------------------------------------------------
async function call(path, options = {}) {
  const res = await fetch(path, options);
  if (res.ok) return res.json();
  let code = null;
  let message = t("common.error_status", { status: res.status });
  try {
    const body = await res.json();
    code = body.code || null;
    if (code && has(`errors.${code}`)) message = t(`errors.${code}`);
    else if (typeof body.detail === "string") message = body.detail;
  } catch {}
  if (code === "auth_required") showLogin();
  throw Object.assign(new Error(message), { status: res.status, code });
}
const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// Upload chunks one at a time, in order. A network error or 5xx is retried with a growing delay; anything else
// (the session is gone, the size limit) stops the recording, what the server has stays finishable.
async function pump() {
  if (S.pumping) return;
  S.pumping = true;
  let delay = 1000;
  try {
    while (S.pending.length && S.id && !S.fatal) {
      const item = S.pending[0];
      try {
        await call(`/api/recordings/${S.id}/chunks/${item.n}?t=${Math.round(item.t)}`, {
          method: "PUT",
          headers: { "Content-Type": item.blob.type || "application/octet-stream" },
          body: item.blob,
        });
        S.pending.shift();
        S.acked = item.n + 1;
        S.uploadedMs = Math.max(S.uploadedMs, item.t);
        S.retrying = false;
        delay = 1000;
        emit();
      } catch (err) {
        if (err.status && err.status < 500 && err.status !== 401 && err.status !== 408 && err.status !== 429) {
          S.fatal = err;
          break;
        }
        S.retrying = true;
        emit();
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(30000, delay * 2);
      }
    }
  } finally {
    S.pumping = false;
  }
  if (S.fatal && (S.phase === "recording" || S.phase === "paused")) {
    S.error = S.fatal.code === "recording_not_found" ? t("recorder.gone") : S.fatal.message;
    stopCapture();
    S.phase = "failed";
    announce(t("recorder.failed_live", { msg: S.error }));
    rerender();
  }
}

// --- capture ---------------------------------------------------------------------------------------------------
function stopCapture() {
  try {
    if (S.recorder && S.recorder.state !== "inactive") S.recorder.stop();
  } catch {}
  for (const stream of S.streams) stream.getTracks().forEach((track) => track.stop());
  S.streams = [];
  S.ctx?.close().catch(() => {});
  S.ctx = null;
  S.analyser = null;
  clearInterval(S.timer);
  S.timer = null;
  S.wake?.release?.().catch(() => {});
  S.wake = null;
  document.title = document.title.replace(TITLE_PREFIX, "");
}

function announce(text) {
  S.announce = text;
  views.forEach((v) => v.say(text));
}

async function openStreams(source, deviceId) {
  const streams = [];
  const wantMic = source === "microphone" || source === "mix";
  const wantTab = source === "tab" || source === "mix";
  let mic = null;
  let display = null;
  try {
    if (wantMic) {
      mic = await navigator.mediaDevices.getUserMedia({ audio: deviceId ? { deviceId: { exact: deviceId } } : true });
      streams.push(mic);
    }
    if (wantTab) {
      display = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      streams.push(display);
      if (display.getAudioTracks().length === 0) throw Object.assign(new Error("no tab audio"), { code: "no_tab_audio" });
    }
  } catch (err) {
    streams.forEach((s) => s.getTracks().forEach((tr) => tr.stop()));
    throw err;
  }
  return { streams, mic, display };
}

async function beginCapture(source, deviceId) {
  const mime = pickMime();
  if (!mime) throw Object.assign(new Error("no mime"), { code: "unsupported" });
  const { streams, mic, display } = await openStreams(source, deviceId);
  S.streams = streams;
  let recordStream;
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  S.ctx = ctx;
  if (source === "mix") {
    const dest = ctx.createMediaStreamDestination();
    ctx.createMediaStreamSource(new MediaStream(mic.getAudioTracks())).connect(dest);
    ctx.createMediaStreamSource(new MediaStream(display.getAudioTracks())).connect(dest);
    recordStream = dest.stream;
    S.streams.push(recordStream);
  } else {
    recordStream = new MediaStream((mic || display).getAudioTracks()); // the video of a shared tab is not recorded
  }
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(recordStream).connect(analyser); // not connected to the speakers
  S.analyser = analyser;
  S.mime = mime;
  // the user ended the share / unplugged the microphone: finish what was recorded
  for (const tr of streams.flatMap((s) => s.getTracks())) {
    tr.addEventListener("ended", () => {
      if (S.phase === "recording" || S.phase === "paused") {
        announce(t("recorder.source_ended"));
        stop();
      }
    });
  }
  return recordStream;
}

function startRecorder(stream) {
  const recorder = new MediaRecorder(stream, { mimeType: S.mime, audioBitsPerSecond: 64000 });
  S.recorder = recorder;
  recorder.ondataavailable = (e) => {
    if (!e.data || e.data.size === 0) return;
    const n = S.total++;
    S.pending.push({ n, blob: e.data, t: elapsedMs() });
    pump();
    emit();
  };
  recorder.onerror = () => {
    S.error = t("recorder.recorder_error");
  };
  recorder.start(SLICE_MS);
  S.runStart = performance.now();
  S.elapsedBase = 0;
  S.lastMinute = 0;
  S.level = 0;
  S.silentSince = performance.now();
  S.timer = setInterval(tick, 200);
}

function tick() {
  if (S.analyser && S.phase === "recording") {
    const data = new Float32Array(S.analyser.fftSize);
    S.analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (const v of data) sum += v * v;
    const rms = Math.sqrt(sum / data.length);
    const db = rms > 0 ? 20 * Math.log10(rms) : -100;
    S.level = Math.max(0, Math.min(1, (db + 60) / 60));
    if (db > -55) {
      S.silentSince = performance.now();
      if (S.note) {
        S.note = "";
        announce("");
      }
    } else if (performance.now() - S.silentSince > SILENCE_AFTER_MS && !S.note) {
      S.note = t("recorder.silent");
      announce(S.note);
    }
  }
  const minute = Math.floor(elapsedMs() / 60000);
  if (S.phase === "recording" && minute > S.lastMinute && (minute <= 5 || minute % 5 === 0)) {
    S.lastMinute = minute;
    announce(t("recorder.minutes_live", { n: minute }));
  } else if (minute > S.lastMinute) {
    S.lastMinute = minute;
  }
  const base = document.title.replace(TITLE_PREFIX, "");
  if (S.phase === "recording" || S.phase === "paused") {
    document.title = `● ${clock(elapsedMs())} — ${base}`;
  } else {
    document.title = base;
  }
  emit();
}

// --- actions ---------------------------------------------------------------------------------------------------
async function start({ source, deviceId }) {
  if (S.phase !== "idle" && S.phase !== "failed") return;
  S.error = "";
  S.note = "";
  S.fatal = null;
  S.phase = "starting";
  rerender();
  try {
    const stream = await beginCapture(source, deviceId);
    const session = await call("/api/recordings", json("POST", { mime: S.mime, source, slice_ms: SLICE_MS }));
    S.id = session.id;
    S.source = source;
    S.total = 0;
    S.acked = 0;
    S.pending = [];
    S.uploadedMs = 0;
    S.fields = S.getFields?.() || {};
    startRecorder(stream);
    try {
      S.wake = (await navigator.wakeLock?.request("screen")) || null; // keeps the screen (and the laptop) awake
    } catch {}
    S.phase = "recording";
    announce(t("recorder.started_live"));
  } catch (err) {
    stopCapture();
    S.id = null;
    S.phase = "idle";
    S.error = startError(err);
  }
  rerender();
}

function startError(err) {
  if (err.code === "no_tab_audio") return t("recorder.err_no_tab_audio");
  if (err.name === "NotAllowedError" || err.name === "SecurityError") return t("recorder.err_denied");
  if (err.name === "NotFoundError" || err.name === "OverconstrainedError") return t("recorder.err_no_device");
  if (err.name === "NotReadableError") return t("recorder.err_busy");
  if (err.code === "unsupported") return t("recorder.err_format");
  return err.message || t("recorder.err_generic");
}

function pause() {
  if (S.phase !== "recording") return;
  S.elapsedBase = elapsedMs();
  S.recorder.pause();
  S.phase = "paused";
  S.level = 0;
  announce(t("recorder.paused_live", { time: clock(S.elapsedBase) }));
  rerender();
}

function resume() {
  if (S.phase !== "paused") return;
  S.recorder.resume();
  S.runStart = performance.now();
  S.phase = "recording";
  announce(t("recorder.resumed_live"));
  rerender();
}

// Stop recording and queue the transcription. The last chunk arrives with the recorder's "stop".
async function stop() {
  if (S.phase !== "recording" && S.phase !== "paused") return;
  S.elapsedBase = elapsedMs();
  S.phase = "finishing";
  S.error = "";
  const recorder = S.recorder;
  const ended = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
  stopCapture();
  await ended;
  announce(t("recorder.sending_live"));
  rerender();
  await finish();
}

async function drain() {
  pump();
  while ((S.pumping || S.pending.length) && !S.fatal) await new Promise((r) => setTimeout(r, 150));
}

async function finish() {
  S.phase = "finishing";
  S.error = "";
  rerender();
  await drain();
  const count = S.fatal ? S.acked : S.total;
  if (!count) {
    S.error = S.fatal ? S.fatal.message : t("recorder.nothing");
    S.phase = "failed";
    rerender();
    return;
  }
  try {
    const fields = S.getFields?.() || S.fields;
    const job = await call(`/api/recordings/${S.id}/finish`, json("POST", { ...clean(fields), chunks: count, title: `${t("recorder.file_name")} ${stamp()}` }));
    reset();
    toast(job.position > 1 ? t("home.queued_toast", { n: job.position }) : t("recorder.queued_toast"));
    refreshSummary();
    S.onJob?.(job);
    announce(t("recorder.queued_live"));
  } catch (err) {
    S.error = t("recorder.finish_failed", { msg: err.message });
    S.phase = "failed";
  }
  rerender();
}

function clean(fields) {
  const out = {};
  for (const [k, v] of Object.entries(fields || {})) {
    if (v !== null && v !== undefined && v !== "") out[k] = v;
  }
  if (out.no_diarize === "true") out.no_diarize = true;
  if (out.no_diarize === false) delete out.no_diarize;
  if (out.separate_speakers === "true") out.separate_speakers = true;
  if (out.separate_speakers === false) delete out.separate_speakers;
  return out;
}

function reset() {
  stopCapture();
  Object.assign(S, { phase: "idle", id: null, total: 0, acked: 0, pending: [], uploadedMs: 0, elapsedBase: 0, error: "", note: "", fatal: null });
}

async function discard() {
  if (S.phase !== "recording" && S.phase !== "paused" && S.phase !== "failed") return;
  const id = S.id;
  S.pending = [];
  reset();
  announce(t("recorder.discarded_live"));
  rerender();
  if (id) await call(`/api/recordings/${id}`, { method: "DELETE" }).catch(() => {});
}

// --- the panel --------------------------------------------------------------------------------------------------
const ICONS = {
  rec: '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><circle cx="12" cy="12" r="6" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><path fill="currentColor" d="M7 5h4v14H7zm6 0h4v14h-4z"/></svg>',
  play: '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><path fill="currentColor" d="M6 6h12v12H6z"/></svg>',
  bin: '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><path fill="currentColor" d="M6 19a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V8H6zM19 4h-3.5l-1-1h-5l-1 1H5v2h14z"/></svg>',
};

// card: element holding #rec-btn and #recorder; getFields(): the form's job options (language, engine, ...);
// onJob(job): the recording was queued. Returns { cleanup }.
export function mountRecorder(root, { getFields, onJob = () => {} } = {}) {
  const button = $("#rec-btn", root);
  const panel = $("#recorder", root);
  const live = $("#rec-live", root);
  let open = S.phase !== "idle";
  let devices = [];
  let source = S.source || "microphone";
  let deviceId = "";
  let confirming = false;
  let recoverable = [];
  let recoverError = "";
  let busyIds = new Set();
  let mounted = true;
  const mode = support();
  const hasDisplay = Boolean(navigator.mediaDevices?.getDisplayMedia);
  S.getFields = getFields;
  S.onJob = onJob;

  async function loadDevices() {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      devices = all.filter((d) => d.kind === "audioinput" && d.deviceId);
    } catch {
      devices = [];
    }
  }

  async function loadRecoverable() {
    try {
      const list = await call("/api/recordings");
      recoverable = list.filter((r) => r.id !== S.id);
      recoverError = "";
    } catch (err) {
      recoverable = [];
      recoverError = "";
    }
    if (mounted) render();
  }

  function sourceLabel(s) {
    return t(`recorder.source_${s || "microphone"}`);
  }

  function localTime(iso) {
    const d = iso ? new Date(iso) : null;
    if (!d || Number.isNaN(d.getTime())) return "?";
    const same = d.toDateString() === new Date().toDateString();
    return same ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : d.toLocaleString([], { dateStyle: "short", timeStyle: "short" });
  }

  function approx(seconds) {
    const min = Math.round(seconds / 60);
    return seconds < 60 ? t("recorder.under_minute") : t("recorder.duration_min", { n: min });
  }

  function recoverHtml() {
    if (!recoverable.length) return "";
    const items = recoverable.map((r) => {
      const fresh = r.updated && Date.now() - new Date(r.updated).getTime() < 20000;
      return `<li class="rec-recover-item" data-id="${esc(r.id)}">
        <span class="rec-recover-text">${esc(t("recorder.recover_item", { time: localTime(r.created), len: approx(r.duration), src: sourceLabel(r.source) }))}${
          r.contiguous ? "" : ` <em>${esc(t("recorder.has_gaps"))}</em>`}${fresh ? ` <em>${esc(t("recorder.maybe_live"))}</em>` : ""}</span>
        <span class="rec-recover-actions">
          <button type="button" class="btn small" data-act="recover" ${busyIds.has(r.id) ? "disabled" : ""}>${esc(t("recorder.recover_finish"))}</button>
          <button type="button" class="btn small ghost" data-act="drop" ${busyIds.has(r.id) ? "disabled" : ""}
                  aria-label="${esc(t("recorder.recover_discard_label", { time: localTime(r.created) }))}">${esc(t("recorder.discard"))}</button>
        </span>
      </li>`;
    });
    return `<section class="rec-recover" aria-labelledby="rec-recover-h">
      <h3 id="rec-recover-h">${esc(t("recorder.recover_title"))}</h3>
      <p class="rec-hint">${esc(t("recorder.recover_hint"))}</p>
      <ul>${items.join("")}</ul>
      ${recoverError ? `<p class="rec-error" role="alert">${esc(recoverError)}</p>` : ""}
    </section>`;
  }

  function render() {
    if (!mounted) return;
    button.setAttribute("aria-expanded", String(open));
    button.classList.toggle("recording", S.phase === "recording" || S.phase === "paused");
    panel.hidden = !open;
    if (!open) return;
    const focused = panel.contains(document.activeElement) ? document.activeElement.id || document.activeElement.dataset.act : null;
    const active = S.phase === "recording" || S.phase === "paused";
    let html = "";
    if (S.phase === "idle" || S.phase === "starting") {
      html += `<p class="rec-consent">${esc(t("recorder.consent"))}</p>`;
      if (mode !== "ok") {
        html += `<p class="rec-error" role="alert">${esc(t(mode === "insecure" ? "recorder.insecure" : "recorder.unsupported", { host: location.host }))}</p>`;
      } else {
        const starting = S.phase === "starting";
        html += `<fieldset class="rec-sources" ${starting ? "disabled" : ""}>
          <legend>${esc(t("recorder.source"))}</legend>
          ${["microphone", ...(hasDisplay ? ["tab", "mix"] : [])].map((s) => `
            <label class="rec-radio"><input type="radio" name="rec-source" value="${s}" ${s === source ? "checked" : ""}>
              <span>${esc(sourceLabel(s))}</span></label>`).join("")}
        </fieldset>`;
        html += `<label class="field rec-device" ${source === "tab" ? "hidden" : ""}>
          <span>${esc(t("recorder.device"))}</span>
          <select id="rec-device" class="select" ${starting ? "disabled" : ""}>
            <option value="">${esc(t("recorder.device_default"))}</option>
            ${devices.map((d, i) => `<option value="${esc(d.deviceId)}" ${d.deviceId === deviceId ? "selected" : ""}>${esc(d.label || t("recorder.device_n", { n: i + 1 }))}</option>`).join("")}
          </select>
        </label>`;
        if (source !== "microphone") {
          html += `<p class="rec-hint">${esc(t("recorder.tab_hint"))}</p>`;
        }
        html += `<div class="rec-actions">
          <button type="button" class="btn primary" id="rec-start" ${starting ? "disabled" : ""}>${ICONS.rec}<span>${esc(t(starting ? "recorder.starting" : "recorder.start"))}</span></button>
        </div>`;
      }
      if (S.error) html += `<p class="rec-error" role="alert">${esc(S.error)}</p>`;
    } else if (active) {
      html += `<div class="rec-status ${S.phase}">
          <span class="rec-dot" aria-hidden="true"></span>
          <strong id="rec-state">${esc(t(S.phase === "paused" ? "recorder.paused" : "recorder.recording"))}</strong>
          <span class="rec-time" id="rec-time" role="timer" aria-live="off">0:00</span>
          <span class="rec-src">${esc(sourceLabel(S.source))}</span>
        </div>
        <div class="rec-meter" aria-hidden="true"><span id="rec-meter-bar"></span></div>
        <p class="rec-sync" id="rec-sync"></p>
        <p class="rec-warn" id="rec-note"></p>
        <div class="rec-actions">
          ${S.phase === "recording"
            ? `<button type="button" class="btn" id="rec-pause">${ICONS.pause}<span>${esc(t("recorder.pause"))}</span></button>`
            : `<button type="button" class="btn" id="rec-resume">${ICONS.play}<span>${esc(t("recorder.resume"))}</span></button>`}
          <button type="button" class="btn primary" id="rec-stop">${ICONS.stop}<span>${esc(t("recorder.stop"))}</span></button>
          ${confirming
            ? `<span class="rec-confirm" role="group" aria-label="${esc(t("recorder.discard_confirm"))}">
                 <button type="button" class="btn danger" id="rec-discard-yes">${esc(t("recorder.discard_yes"))}</button>
                 <button type="button" class="btn" id="rec-discard-no">${esc(t("recorder.discard_no"))}</button></span>`
            : `<button type="button" class="btn ghost" id="rec-discard">${ICONS.bin}<span>${esc(t("recorder.discard"))}</span></button>`}
        </div>
        <p class="rec-hint">${esc(t("recorder.keep_open"))}</p>`;
    } else if (S.phase === "finishing") {
      html += `<p class="rec-status finishing"><span class="rec-spinner" aria-hidden="true"></span><strong>${esc(t("recorder.sending"))}</strong></p>`;
    } else if (S.phase === "failed") {
      html += `<p class="rec-error" role="alert">${esc(S.error || t("recorder.err_generic"))}</p>
        <div class="rec-actions">
          ${S.id && (S.acked || S.total) ? `<button type="button" class="btn primary" id="rec-retry">${esc(t("recorder.retry_finish"))}</button>` : ""}
          <button type="button" class="btn ghost" id="rec-discard-failed">${esc(t("recorder.discard"))}</button>
        </div>`;
    }
    if (S.phase === "idle") html += recoverHtml();
    panel.innerHTML = html;
    bind();
    update();
    const target = focused && (document.getElementById(focused) || panel.querySelector(`[data-act="${focused}"]`));
    if (focused && !target) panel.querySelector("button:not([disabled])")?.focus();
    else target?.focus();
  }

  function bind() {
    panel.querySelectorAll('input[name="rec-source"]').forEach((r) => r.addEventListener("change", () => {
      source = r.value;
      render();
    }));
    $("#rec-device", panel)?.addEventListener("change", (e) => { deviceId = e.target.value; });
    $("#rec-start", panel)?.addEventListener("click", async () => {
      S.source = source;
      await start({ source, deviceId });
      loadDevices().then(() => mounted && S.phase === "idle" && render()); // labels appear after permission
    });
    $("#rec-pause", panel)?.addEventListener("click", pause);
    $("#rec-resume", panel)?.addEventListener("click", resume);
    $("#rec-stop", panel)?.addEventListener("click", stop);
    $("#rec-retry", panel)?.addEventListener("click", finish);
    $("#rec-discard-failed", panel)?.addEventListener("click", () => { discard(); });
    $("#rec-discard", panel)?.addEventListener("click", () => { confirming = true; render(); $("#rec-discard-no", panel)?.focus(); });
    $("#rec-discard-no", panel)?.addEventListener("click", () => { confirming = false; render(); $("#rec-discard", panel)?.focus(); });
    $("#rec-discard-yes", panel)?.addEventListener("click", () => { confirming = false; discard(); });
    panel.querySelectorAll(".rec-recover-item").forEach((li) => {
      const id = li.dataset.id;
      li.querySelector('[data-act="recover"]').addEventListener("click", () => recover(id));
      li.querySelector('[data-act="drop"]').addEventListener("click", () => drop(id));
    });
  }

  async function recover(id) {
    const item = recoverable.find((r) => r.id === id);
    busyIds.add(id);
    recoverError = "";
    render();
    try {
      const fields = getFields?.() || {};
      const job = await call(`/api/recordings/${id}/finish`, json("POST", {
        ...clean(fields), allow_gaps: item ? !item.contiguous : false,
        title: `${t("recorder.file_name")} ${stamp(item?.created ? new Date(item.created) : new Date())}`,
      }));
      toast(job.position > 1 ? t("home.queued_toast", { n: job.position }) : t("recorder.queued_toast"));
      refreshSummary();
      onJob(job);
      recoverable = recoverable.filter((r) => r.id !== id);
    } catch (err) {
      recoverError = t("recorder.finish_failed", { msg: err.message });
    }
    busyIds.delete(id);
    render();
  }

  async function drop(id) {
    busyIds.add(id);
    render();
    try {
      await call(`/api/recordings/${id}`, { method: "DELETE" });
    } catch (err) {
      if (err.status !== 404) recoverError = err.message;
    }
    busyIds.delete(id);
    recoverable = recoverable.filter((r) => r.id !== id);
    render();
    announce(t("recorder.discarded_live"));
  }

  // cheap, targeted updates (timer, meter, upload state): the panel is not rebuilt, so focus stays where it is
  function update() {
    if (!mounted) return;
    const time = $("#rec-time", panel);
    if (time) time.textContent = clock(elapsedMs());
    const bar = $("#rec-meter-bar", panel);
    if (bar) bar.style.width = `${Math.round(S.level * 100)}%`;
    const sync = $("#rec-sync", panel);
    if (sync) {
      const waiting = S.pending.length;
      sync.textContent = S.retrying
        ? t("recorder.sync_retry", { n: waiting })
        : t("recorder.sync_ok", { time: clock(S.uploadedMs) });
      sync.classList.toggle("warn", S.retrying);
    }
    const note = $("#rec-note", panel);
    if (note) note.textContent = S.note;
  }

  function say(text) {
    if (live) live.textContent = text;
  }

  const view = { update, render, say };
  views.add(view);
  button.addEventListener("click", onToggle);
  function onToggle() {
    open = !open;
    render();
    if (open) {
      panel.querySelector("button:not([disabled]), input")?.focus();
      loadRecoverable();
      loadDevices().then(() => mounted && S.phase === "idle" && render());
    }
  }
  if (open) {
    render();
  } else {
    panel.hidden = true;
  }
  loadRecoverable().then(() => {
    if (recoverable.length && !open && S.phase === "idle") { // unfinished sessions: show them without a click
      open = true;
      render();
    }
  });

  return {
    cleanup() {
      mounted = false;
      views.delete(view);
      button.removeEventListener("click", onToggle);
      if (S.getFields === getFields) S.getFields = null;
      if (S.onJob === onJob) S.onJob = null;
    },
  };
}

window.addEventListener("beforeunload", (e) => {
  if (S.phase === "recording" || S.phase === "paused" || S.phase === "finishing") {
    e.preventDefault();
    e.returnValue = "";
  }
});

// test hook: the QA drives the state without touching module internals
window.__recorderState = () => ({ phase: S.phase, id: S.id, total: S.total, acked: S.acked, pending: S.pending.length, mime: S.mime });
