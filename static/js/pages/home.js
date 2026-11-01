// Home: new-job form, the running job and the transcript library (the job list is on the Jobs page).

import { api, getInfo } from "../core/api.js";
import { applyI18n, has, t } from "../core/i18n.js";
import { registerPage } from "../core/router.js";
import { $, esc, toast } from "../core/util.js";
import { mountRunning } from "../components/jobs.js";
import { refreshSummary } from "../core/jobsummary.js";
import { mountFolderField } from "../components/folders.js";
import { langName } from "../components/translations.js";
import { mountLibrary } from "../components/library.js";
import { mountSetup } from "../components/setup.js";
import { mountUpload } from "../components/upload.js";
import { mountRecorder } from "../components/recorder.js";

const TEMPLATE = `
  <div id="setup" hidden></div>
  <section class="card new-job" id="new-job">
    <h1 data-i18n="home.title"></h1>
    <form id="job-form" autocomplete="off">
      <label class="field url-field">
        <span data-i18n="home.url"></span>
        <input id="url" type="url" inputmode="url" data-i18n-placeholder="home.url_placeholder">
      </label>
      <div class="upload-field">
        <button type="button" class="btn" id="pick-btn" aria-describedby="upload-help">
          <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><path fill="currentColor" d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>
          <span data-i18n="home.upload_pick"></span>
        </button>
        <span class="upload-help" id="upload-help"></span>
        <input type="file" id="file-input" multiple hidden>
        <button type="button" class="btn" id="rec-btn" aria-expanded="false" aria-controls="recorder">
          <svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18"><path fill="currentColor" d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.9V22h2v-3.1A7 7 0 0 0 19 12z"/></svg>
          <span data-i18n="recorder.button"></span>
        </button>
      </div>
      <div class="rec-panel" id="recorder" role="group" data-i18n-aria-label="recorder.panel" hidden></div>
      <p class="sr-only" id="rec-live" role="status" aria-live="polite"></p>
      <ul id="upload-list" class="upload-list" data-i18n-aria-label="home.upload_list"></ul>
      <p class="sr-only" id="upload-live" role="status" aria-live="polite"></p>
      <div class="row">
        <label class="field">
          <span data-i18n="home.language"></span>
          <div class="segmented" role="radiogroup" data-i18n-aria-label="home.language">
            <input type="radio" name="language" id="lang-auto" value="auto" checked><label for="lang-auto" data-i18n="home.lang_auto"></label>
            <input type="radio" name="language" id="lang-en" value="en"><label for="lang-en" data-i18n="home.lang_en"></label>
            <input type="radio" name="language" id="lang-cs" value="cs"><label for="lang-cs" data-i18n="home.lang_cs"></label>
          </div>
        </label>
        <label class="field speakers-field">
          <span data-i18n="home.speakers"></span>
          <input id="speakers" type="number" min="1" max="20" data-i18n-placeholder="home.speakers_placeholder">
        </label>
        <label class="field diarize-field">
          <input type="checkbox" id="diarize" checked>
          <span data-i18n="setup.diarize"></span>
        </label>
        <button class="btn primary" type="submit" id="start-btn">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg><span data-i18n="home.start"></span>
        </button>
      </div>
      <div class="row engine-row">
        <label class="field engine-field">
          <span data-i18n="home.engine"></span>
          <select id="engine" class="select"></select>
        </label>
        <label class="field model-field">
          <span data-i18n="home.model"></span>
          <select id="model" class="select"></select>
        </label>
      </div>
      <p class="engine-note" id="engine-note"></p>
      <div class="row separate-row">
        <label class="field diarize-field separate-field">
          <input type="checkbox" id="separate">
          <span data-i18n="home.separate"></span>
        </label>
        <p class="engine-note" id="separate-note" role="note"></p>
      </div>
      <div class="row folder-row">
        <div class="field folder-field" id="folder-field">
          <label for="job-folder" data-i18n="folders.form_label"></label>
          <select id="job-folder" class="select"></select>
          <input id="job-folder-new" type="text" maxlength="255" hidden data-i18n-placeholder="folders.form_new_name"
                 data-i18n-aria-label="folders.form_new_name" autocomplete="off">
        </div>
        <label class="field translate-field">
          <span data-i18n="home.translate_to"></span>
          <select id="job-translate" class="select" aria-describedby="translate-note"></select>
        </label>
      </div>
      <p class="engine-note" id="translate-note" role="note" hidden></p>
      <p class="form-msg" id="form-msg" role="status"></p>
    </form>
    <div class="drop-overlay" aria-hidden="true" data-i18n="home.upload_drop"></div>
  </section>

  <section id="running" aria-label="" data-i18n-aria-label="jobs.now_running_label" hidden></section>

  <section>
    <div class="section-head">
      <h2 class="section-title" data-i18n="library.title"></h2>
      <input id="library-filter" class="search-input small" type="search" data-i18n-placeholder="library.filter"
             data-i18n-aria-label="library.filter">
    </div>
    <div class="library-layout">
      <aside class="folder-side" id="folder-side"></aside>
      <div class="library-main">
        <div class="folder-bar" id="folder-bar"></div>
        <div class="selection-bar" id="selection-bar" hidden></div>
        <div id="search-results" class="search-results" aria-live="polite" hidden></div>
        <div id="library" class="library"></div>
      </div>
    </div>
  </section>`;

function mount(el) {
  el.innerHTML = TEMPLATE;
  applyI18n(el);

  let folderField = null;
  const library = mountLibrary($("#library", el), $("#library-filter", el), $("#search-results", el), {
    side: $("#folder-side", el),
    bar: $("#folder-bar", el),
    selection: $("#selection-bar", el),
    onFolder: (path) => folderField?.follow(path),
  });
  // the form's folder follows the folder open in the library (nothing when "All" is shown)
  folderField = mountFolderField($("#folder-field", el), () => {
    const open = library.current();
    return typeof open === "string" ? open : "";
  });
  const separate = $("#separate", el);
  // "Translate to": the finished transcript is translated like the detail page's Translate does (default: no)
  const translate = $("#job-translate", el);
  const fillTranslate = (languages) => {
    const chosen = translate.value;
    translate.innerHTML = `<option value="">${esc(t("home.translate_none"))}</option>` +
      Object.entries(languages).map(([code, name]) => `<option value="${esc(code)}">${esc(langName(code, name))}</option>`).join("");
    translate.value = chosen;
  };
  let translateLanguages = {};
  fillTranslate(translateLanguages);
  translate.addEventListener("change", () => {
    const note = $("#translate-note", el);
    note.hidden = !translate.value;
    note.textContent = translate.value ? t("home.translate_hint") : "";
  });
  // the transcription runs anyway; a translation that cannot be queued is announced now, not only when it ends
  const warnTranslation = (job) => {
    if (job?.translation_warning) msg.textContent = t("home.translate_warning", { reason: job.translation_warning });
  };
  // the job options shared by the submit handler and the recorder
  const jobFields = () => {
    const fields = {
      language: el.querySelector('input[name="language"]:checked').value,
      speakers: $("#speakers", el).value ? Number($("#speakers", el).value) : null,
      no_diarize: !diarize.checked,
      ...(separate.checked ? { separate_speakers: true } : {}),
      ...(translate.value ? { translate_to: translate.value } : {}),
      ...engines.value(),
    };
    const folder = folderField.value();
    if (folder !== undefined) fields.folder = folder;
    return fields;
  };
  const engines = mountEngines(el, (allowed) => {
    // overlapping speech is separated only for the local engines (the cloud ones get the whole recording)
    separate.disabled = !allowed;
    if (!allowed) separate.checked = false;
    $("#separate-note", el).textContent = allowed ? t("home.separate_hint") : t("home.separate_blocked");
  });
  // first-run guide; while speaker recognition cannot work, the option is off instead of failing mid-transcription
  const diarize = $("#diarize", el);
  let diarizeDefault = true;
  getInfo().then((info) => {
    diarizeDefault = info.defaults.diarize;
    translateLanguages = info.translate_languages || {};
    fillTranslate(translateLanguages);
    if (!diarize.disabled) diarize.checked = diarizeDefault;
    el.querySelector(`input[name="language"][value="${info.defaults.language}"]`)?.click();  // a language set in Settings
  }).catch(() => {});
  const setup = mountSetup($("#setup", el), {
    onState({ blocked }) {
      diarize.disabled = blocked;
      diarize.checked = blocked ? false : diarizeDefault;
      diarize.title = blocked ? t("setup.diarize_blocked") : "";
    },
  });
  const stopRunning = mountRunning($("#running", el), { onFinished: (job) => job?.status === "done" && library.reload() });
  const msg = $("#form-msg", el);
  const uploads = mountUpload($("#new-job", el), el, {
    onChange: (count) => engines.setFiles(count > 0),
    onProblem: (text) => { msg.textContent = text; },
  });

  // the recorder uploads while it records and queues a normal job with the form's options
  const recorder = mountRecorder(el, {
    getFields: jobFields,
    onJob(job) {
      warnTranslation(job);
      library.refreshFolders().then(() => folderField.refresh(library.folders())).catch(() => {});
    },
  });

  $("#job-form", el).addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.textContent = "";
    const urlInput = $("#url", el);
    const link = urlInput.value.trim();
    if (!link && uploads.count() === 0) {
      msg.textContent = t("home.need_source");
      urlInput.focus();
      return;
    }
    const fields = jobFields();
    const folder = fields.folder;
    const btn = $("#start-btn", el);
    btn.disabled = true;
    try {
      if (link) {
        const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ source: link, ...fields }) });
        urlInput.value = "";
        folderField.reset();
        toast(job.position > 1 ? t("home.queued_toast", { n: job.position }) : t("home.started_toast"));
        warnTranslation(job);
        refreshSummary();
      }
      if (uploads.count() > 0) {
        const { ok } = await uploads.run({ ...fields, no_diarize: fields.no_diarize ? "true" : "", separate_speakers: fields.separate_speakers ? "true" : "" }, (job) => { warnTranslation(job); refreshSummary(); });
        if (ok) toast(t("home.upload_done_toast", { n: ok }));
      }
      if (folder !== undefined) library.refreshFolders().then((tree) => folderField.refresh(library.folders())).catch(() => {});
    } catch (err) {
      msg.textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  });

  return () => {
    setup();
    uploads.cleanup();
    recorder.cleanup();
    stopRunning();
    library.cleanup();
  };
}

// Engine and model choice from /api/info. Unavailable engines stay visible (disabled, with the
// reason); cloud engines get a warning that the audio leaves the computer. The YouTube captions
// engine is offered only while the source is a link.
const URL_ONLY = new Set(["youtube"]);
// engines that can run voice separation (mgtranscriber/separated.py LOCAL_ENGINES)
const SEPARATION_ENGINES = new Set(["whisperx", "parakeet", "canary"]);

function mountEngines(el, onSeparation = () => {}) {
  const engineSel = $("#engine", el);
  const modelSel = $("#model", el);
  const note = $("#engine-note", el);
  const source = $("#url", el);
  let list = [];
  let preferredName = "";
  let hasFiles = false;

  const reason = (m) => t(`engines.missing.${m.code}`, { detail: m.detail });
  const current = () => list.find((e) => e.name === engineSel.value);

  function update() {
    const engine = current();
    if (!engine) return;
    modelSel.innerHTML = engine.models.length
      ? engine.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("")
      : `<option value="">${esc(t("home.model_default"))}</option>`;
    modelSel.disabled = engine.models.length < 2;
    const lang = el.querySelector('input[name="language"]:checked')?.value;
    const parts = [];
    if (engine.cloud) parts.push(`<strong class="cloud-warning">⚠ ${esc(t("engines.cloud_warning"))}</strong>`);
    if (engine.missing) parts.push(`<span class="engine-missing">${esc(reason(engine.missing))}</span>`);
    if (engine.needs_language && lang === "auto") parts.push(`<span class="engine-missing">${esc(t("engines.needs_language"))}</span>`);
    const desc = has(`engines.desc.${engine.name}`) ? t(`engines.desc.${engine.name}`) : engine.description;
    parts.push(`<span class="engine-desc">${esc(desc)}</span>`);
    note.innerHTML = parts.join(" ");
    note.classList.toggle("cloud", engine.cloud);
    onSeparation(SEPARATION_ENGINES.has(engine.name));
  }

  // a typed source that is not a link is a local file: no captions to read
  function updateSource() {
    const value = source.value.trim();
    const local = hasFiles || (value !== "" && !/^https?:\/\//i.test(value));
    for (const option of engineSel.options) {
      const engine = list.find((e) => e.name === option.value);
      if (!engine || !URL_ONLY.has(engine.name)) continue;
      option.disabled = Boolean(engine.missing) || local;
      option.textContent = engine.label + (engine.missing ? ` — ${reason(engine.missing)}` : local ? ` — ${t("engines.url_only")}` : "");
    }
    if (engineSel.selectedOptions[0]?.disabled) {
      engineSel.value = preferredName;
      update();
    }
  }

  getInfo().then((info) => {
    list = info.engines.filter((e) => e.available);
    engineSel.innerHTML = list.map((e) => {
      const label = e.label + (e.missing ? ` — ${reason(e.missing)}` : "");
      return `<option value="${esc(e.name)}"${e.missing ? " disabled" : ""}>${esc(label)}</option>`;
    }).join("");
    const preferred = list.find((e) => e.name === info.defaults.engine && !e.missing && !URL_ONLY.has(e.name))
      || list.find((e) => !e.missing && !URL_ONLY.has(e.name));
    preferredName = preferred?.name || "";
    if (preferred) engineSel.value = preferred.name;
    update();
    updateSource();
    const engine = current();
    if (engine?.name === info.defaults.engine && engine.models.includes(info.defaults.model)) {
      modelSel.value = info.defaults.model;
    }
  }).catch(() => {});
  engineSel.addEventListener("change", update);
  source.addEventListener("input", updateSource);
  el.querySelectorAll('input[name="language"]').forEach((r) => r.addEventListener("change", update));

  return {
    setFiles(on) {
      hasFiles = on;
      updateSource();
    },
    value() {
      const engine = current();
      if (!engine) return {};
      return { engine: engine.name, ...(modelSel.value ? { model: modelSel.value } : {}) };
    },
  };
}

registerPage({ id: "home", path: "/", nav: { label: "nav.home", order: 0 }, mount });
