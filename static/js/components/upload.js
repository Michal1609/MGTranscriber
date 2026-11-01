// File upload in the "New transcript" form: a button and drag & drop on the whole card stage files, the form's
// Start button uploads them one after another (each file is its own job) with a progress bar per file.

import { getInfo, upload } from "../core/api.js";
import "../i18n/upload.js";
import { t } from "../core/i18n.js";
import { $, esc, number } from "../core/util.js";

const GENERIC_TYPES = ["", "application/octet-stream", "binary/octet-stream", "application/ogg", "application/mp4",
  "application/x-matroska", "application/vnd.apple.mpegurl"];
const FALLBACK_EXTENSIONS = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mkv", ".mov"];
const hasFiles = (e) => Boolean(e.dataTransfer) && [...e.dataTransfer.types].includes("Files");

function size(bytes) {
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${number(Math.round((mb / 1024) * 10) / 10)} GB` : `${number(Math.max(0.1, Math.round(mb * 10) / 10))} MB`;
}

// card: the form's card (drop target), root: element holding #pick-btn, #file-input, #upload-help, #upload-list,
// #upload-live. onChange(count): staged files changed. onProblem(text): files that were refused or failed.
export function mountUpload(card, root, { onChange = () => {}, onProblem = () => {} } = {}) {
  const input = $("#file-input", root);
  const list = $("#upload-list", root);
  const live = $("#upload-live", root);
  const pick = $("#pick-btn", root);
  let limits = { max_mb: 0, extensions: FALLBACK_EXTENSIONS };
  let items = []; // { file, el, state: "staged" | "uploading", abort }
  let busy = false;

  getInfo().then((info) => {
    if (info.upload) limits = info.upload;
    $("#upload-help", root).textContent = limits.max_mb > 0 ? t("home.upload_help", { mb: limits.max_mb }) : t("home.upload_help_nolimit");
  }).catch(() => {});

  // same rules as the server (POST /api/jobs/upload), so a refused file never travels
  const problem = (file) => {
    const dot = file.name.lastIndexOf(".");
    const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    const kind = (file.type || "").split(";")[0].toLowerCase();
    if (!limits.extensions.includes(ext) || !(kind.startsWith("audio/") || kind.startsWith("video/") || GENERIC_TYPES.includes(kind))) {
      return t("home.upload_bad_type", { name: file.name });
    }
    if (file.size === 0) return t("home.upload_empty", { name: file.name });
    if (limits.max_mb > 0 && file.size > limits.max_mb * 1024 * 1024) return t("home.upload_too_large", { name: file.name, mb: limits.max_mb });
    return null;
  };

  function add(files) {
    if (busy) return;
    const refused = [];
    for (const file of files) {
      const why = problem(file);
      if (why) {
        refused.push(why);
      } else if (!items.some((i) => i.file.name === file.name && i.file.size === file.size && i.file.lastModified === file.lastModified)) {
        const item = { file, state: "staged", abort: null, el: document.createElement("li") };
        item.el.className = "upload-item";
        items.push(item);
        list.append(item.el);
        render(item);
      }
    }
    onProblem(refused.join(" "));
    onChange(items.length);
  }

  function render(item, pct = 0) {
    const uploading = item.state === "uploading";
    const label = uploading ? t("home.upload_cancel", { name: item.file.name }) : t("home.upload_remove", { name: item.file.name });
    item.el.innerHTML = `
      <span class="upload-name" title="${esc(item.file.name)}">${esc(item.file.name)}</span>
      <span class="upload-size">${esc(size(item.file.size))}</span>
      <span class="upload-state">${esc(uploading ? t("home.upload_uploading", { pct }) : t("home.upload_waiting"))}</span>
      <button type="button" class="icon-btn" aria-label="${esc(label)}">✕</button>
      ${uploading ? `<progress max="100" value="${pct}" aria-label="${esc(item.file.name)}"></progress>` : ""}`;
    $("button", item.el).addEventListener("click", () => (item.abort ? item.abort() : remove(item)));
  }

  function remove(item) {
    items = items.filter((i) => i !== item);
    const hadFocus = item.el.contains(document.activeElement);
    item.el.remove();
    onChange(items.length);
    if (hadFocus) pick.focus();
  }

  // Upload every staged file; `fields` are the form fields of the job. onJob(job) per accepted file.
  // Resolves with { ok, failed }; a failed file stays in the list, the reason goes to onProblem.
  async function run(fields, onJob = () => {}) {
    if (busy) return { ok: 0, failed: 0 };
    busy = true;
    const queue = items.filter((i) => i.state === "staged");
    let ok = 0;
    const errors = [];
    for (const [index, item] of queue.entries()) {
      const controller = new AbortController();
      item.state = "uploading";
      item.abort = () => controller.abort();
      render(item);
      const bar = $("progress", item.el);
      const state = $(".upload-state", item.el);
      let announced = -1;
      const form = new FormData();
      form.append("file", item.file, item.file.name);
      for (const [key, value] of Object.entries(fields)) {
        if (value !== null && value !== undefined && value !== "") form.append(key, value);
      }
      try {
        const job = await upload("/api/jobs/upload", form, {
          signal: controller.signal,
          onProgress(fraction) {
            const pct = Math.round(fraction * 100);
            bar.value = pct;
            state.textContent = t("home.upload_uploading", { pct });
            if (pct >= announced + 10 || pct === 100) { // a screen reader hears every tenth, not every chunk
              announced = pct;
              live.textContent = t("home.upload_sending", { name: item.file.name, pct }) + (queue.length > 1 ? ` (${index + 1}/${queue.length})` : "");
            }
          },
        });
        ok += 1;
        remove(item);
        onJob(job);
      } catch (err) {
        item.state = "staged";
        item.abort = null;
        if (err.code === "aborted") {
          remove(item);
        } else {
          render(item);
          errors.push(`${item.file.name}: ${err.message}`);
        }
      }
    }
    busy = false;
    live.textContent = t("home.upload_summary", { ok, total: queue.length });
    onProblem(errors.join(" "));
    return { ok, failed: errors.length };
  }

  // --- picking and dropping ---------------------------------------------------------------------------
  pick.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    add([...input.files]);
    input.value = ""; // the same file can be chosen again after it was removed
  });

  let depth = 0;
  card.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth += 1;
    card.classList.add("dragging");
  });
  card.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  card.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) card.classList.remove("dragging");
  });
  card.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    card.classList.remove("dragging");
    add([...e.dataTransfer.files]);
  });
  // a file dropped beside the card must not make the browser open it and leave the app
  const strayDrag = (e) => {
    if (hasFiles(e) && !card.contains(e.target)) e.preventDefault();
  };
  window.addEventListener("dragover", strayDrag);
  window.addEventListener("drop", strayDrag);

  return {
    count: () => items.length,
    run,
    cleanup() {
      window.removeEventListener("dragover", strayDrag);
      window.removeEventListener("drop", strayDrag);
      for (const item of items) item.abort?.();
    },
  };
}
