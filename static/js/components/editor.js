// Transcript editor (detail page, "Edit"): text of each segment in place while listening, speaker per
// segment, split and merge, redaction, undo/redo, autosave with version history. See docs/EDITOR.md.
//
// Invariant shared with the server (mgtranscriber/editor.py): a segment's words correspond 1:1 to the
// whitespace-separated tokens of its text. Typing re-aligns the words (realign()), so times and the
// redaction flag stay with the words that were not changed.

import { api } from "../core/api.js";
import { applyI18n, t } from "../core/i18n.js";
import { clock, esc, speakerLabel, toast } from "../core/util.js";

const SAVE_DELAY = 1200; // ms after the last change
const TYPING_GAP = 1500; // a pause this long starts a new undo step
const MAX_UNDO = 200;
const SEEK_STEP = 5;

export function loadStyles() {
  if (document.querySelector("link[data-editor-css]")) return;
  const link = Object.assign(document.createElement("link"), { rel: "stylesheet", href: "editor.css" });
  link.dataset.editorCss = "";
  document.head.append(link);
}

/* --- words <-> text ----------------------------------------------------------------------- */

const tokensOf = (text) => text.split(/\s+/).filter(Boolean);
const keyOf = (token) => token.replace(/[^\p{L}\p{N}_]+/gu, "").toLowerCase() || token;

const LCS_LIMIT = 400000; // cells of the LCS table; the same limit as editor.LCS_LIMIT on the server

// Longest common subsequence of the keys: pairs [i, j] of old word i = new token j. The common
// prefix and suffix are matched first; a larger middle stays unmatched. Same as editor.lcs_pairs.
function lcsPairs(a, b) {
  const n = a.length, m = b.length;
  let lo = 0, hi = 0;
  while (lo < n && lo < m && a[lo] === b[lo]) lo++;
  while (hi < n - lo && hi < m - lo && a[n - 1 - hi] === b[m - 1 - hi]) hi++;
  const pairs = [];
  for (let k = 0; k < lo; k++) pairs.push([k, k]);
  const rn = n - lo - hi, rm = m - lo - hi;
  if (rn && rm && rn * rm <= LCS_LIMIT) {
    const A = a.slice(lo, n - hi), B = b.slice(lo, m - hi);
    const dp = Array.from({ length: rn + 1 }, () => new Uint32Array(rm + 1));
    for (let i = rn - 1; i >= 0; i--)
      for (let j = rm - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    let i = 0, j = 0;
    while (i < rn && j < rm) {
      if (A[i] === B[j]) { pairs.push([lo + i, lo + j]); i++; j++; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
      else j++;
    }
  }
  for (let k = 0; k < hi; k++) pairs.push([n - hi + k, m - hi + k]);
  return pairs;
}

// Words for the new text, one per token (the same rules as editor.realign_words on the server).
// Redaction errs on the safe side: a changed stretch that had a redacted word is redacted as a
// whole, and tokens added next to a redacted word are redacted too.
export function realign(old, text, start, end, speaker) {
  const toks = tokensOf(text);
  if (old.length === toks.length && old.every((w, i) => w.word === toks[i])) return old;
  const pairs = lcsPairs(old.map((w) => keyOf(w.word)), toks.map(keyOf));
  const out = new Array(toks.length);
  const estimated = new Set();
  const inserted = [];
  let pi = 0, pj = 0;
  for (const [mi, mj] of [...pairs, [old.length, toks.length]]) {
    // changed stretch old[pi..mi) -> new[pj..mj): the old words' times one for one first, the rest new
    const red = old.slice(pi, mi).some((w) => w.redacted);
    const paired = Math.min(mi - pi, mj - pj);
    for (let k = 0; k < paired; k++) {
      const { score, redacted, ...w } = old[pi + k];
      out[pj + k] = { ...w, word: toks[pj + k], ...(red ? { redacted: true } : {}) };
    }
    for (let j = pj + paired; j < mj; j++) {
      out[j] = { word: toks[j], ...(speaker ? { speaker } : {}), ...(red ? { redacted: true } : {}) };
      estimated.add(j);
    }
    if (pj + paired < mj) inserted.push([pj + paired, mj]);
    if (mi < old.length) out[mj] = { ...old[mi], word: toks[mj] };
    pi = mi + 1;
    pj = mj + 1;
  }
  for (const [j1, j2] of inserted)
    if ((j1 > 0 && out[j1 - 1].redacted) || (j2 < out.length && out[j2].redacted))
      for (let j = j1; j < j2; j++) out[j].redacted = true;
  spreadTimes(out, estimated, start, end);
  return out;
}

function spreadTimes(words, estimated, start, end) {
  for (let i = 0; i < words.length; i++) {
    if (!estimated.has(i)) continue;
    let j = i;
    while (j < words.length && estimated.has(j)) j++;
    let left = start, right = end;
    for (let k = i - 1; k >= 0; k--) if (words[k].end != null) { left = words[k].end; break; }
    for (let k = j; k < words.length; k++) if (words[k].start != null) { right = words[k].start; break; }
    if (right < left) right = left;
    const weights = words.slice(i, j).map((w) => w.word.length + 1);
    const total = weights.reduce((a, b) => a + b, 0);
    let time = left;
    for (let k = i; k < j; k++) {
      words[k].start = Math.round(time * 1000) / 1000;
      time += ((right - left) * weights[k - i]) / total;
      words[k].end = Math.round(time * 1000) / 1000;
    }
    i = j;
  }
}

// the editor works on the text as shown (prepis.json keeps a leading space, the server adds it back)
const trimmed = (seg) => ({ ...seg, text: seg.text.trim() });
const firstStart = (words) => words.find((w) => w.start != null)?.start;
const lastEnd = (words) => [...words].reverse().find((w) => w.end != null)?.end;

/* --- caret and selection helpers ---------------------------------------------------------- */

function caretOffset(el, node, offset) {
  const range = document.createRange();
  range.setStart(el, 0);
  range.setEnd(node, offset);
  return range.toString().length;
}

function selectionIn(el) {
  const sel = window.getSelection();
  if (!sel.rangeCount) return null;
  const r = sel.getRangeAt(0);
  if (!el.contains(r.startContainer) && !el.contains(r.endContainer) && !r.intersectsNode(el)) return null;
  const start = el.contains(r.startContainer) ? caretOffset(el, r.startContainer, r.startOffset) : 0;
  const end = el.contains(r.endContainer) ? caretOffset(el, r.endContainer, r.endOffset) : el.textContent.length;
  return { start, end, collapsed: r.collapsed };
}

function setCaret(el, offset) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let node, left = offset;
  while ((node = walker.nextNode())) {
    if (left <= node.length) {
      const sel = window.getSelection();
      sel.collapse(node, left);
      return;
    }
    left -= node.length;
  }
  const sel = window.getSelection();
  sel.selectAllChildren(el);
  sel.collapseToEnd();
}

// Token index ranges [first, last] overlapping the character range [start, end).
function tokensInRange(text, start, end) {
  const out = [];
  let i = 0;
  for (const m of text.matchAll(/\S+/g)) {
    const a = m.index, b = m.index + m[0].length;
    if (start === end ? start >= a && start <= b : a < end && b > start) out.push(i);
    i++;
  }
  return out;
}

/* --- the editor ---------------------------------------------------------------------------- */

export function mountEditor({ id, box, audio, hasAudio, names, colorOf, onClose }) {
  loadStyles();
  const base = `/api/transcripts/${encodeURIComponent(id)}`;
  const state = {
    segs: [], stamp: null, versions: [], undo: [], redo: [], dirty: false, saving: null, timer: null,
    lastInput: 0, typingRow: -1, playing: -1, conflict: false, closed: false, active: 0,
  };

  box.innerHTML = `
    <div class="ed-bar" role="toolbar" data-i18n-aria-label="editor.toolbar">
      <button class="btn ed-undo" data-i18n="editor.undo" data-i18n-title="editor.undo_key" disabled></button>
      <button class="btn ed-redo" data-i18n="editor.redo" data-i18n-title="editor.redo_key" disabled></button>
      <button class="btn ed-split" data-i18n="editor.split" data-i18n-title="editor.split_key"></button>
      <button class="btn ed-merge" data-i18n="editor.merge" data-i18n-title="editor.merge_key"></button>
      <button class="btn ed-redact" data-i18n="editor.redact" data-i18n-title="editor.redact_key"></button>
      <button class="btn ed-keys-btn" aria-expanded="false" data-i18n="editor.keys"></button>
      <button class="btn ed-versions-btn" aria-expanded="false" data-i18n="editor.versions"></button>
      <span class="ed-status" role="status" aria-live="polite"></span>
      <button class="btn primary ed-done" data-i18n="editor.done"></button>
    </div>
    <div class="ed-keys card" hidden>
      <dl>
        <dt><kbd>Ctrl</kbd>+<kbd>Space</kbd></dt><dd data-i18n="editor.k_play"></dd>
        <dt><kbd>Ctrl</kbd>+<kbd>,</kbd> / <kbd>Ctrl</kbd>+<kbd>.</kbd></dt><dd data-i18n="editor.k_seek"></dd>
        <dt><kbd>Enter</kbd> · <kbd>Alt</kbd>+<kbd>↓</kbd> / <kbd>Alt</kbd>+<kbd>↑</kbd></dt><dd data-i18n="editor.k_next"></dd>
        <dt><kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd></dt><dd data-i18n="editor.k_speaker"></dd>
        <dt><kbd>Ctrl</kbd>+<kbd>Enter</kbd></dt><dd data-i18n="editor.k_split"></dd>
        <dt><kbd>Backspace</kbd> <span data-i18n="editor.at_start"></span></dt><dd data-i18n="editor.k_merge"></dd>
        <dt><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>X</kbd></dt><dd data-i18n="editor.k_redact"></dd>
        <dt><kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd></dt><dd data-i18n="editor.k_undo"></dd>
      </dl>
    </div>
    <div class="ed-versions card" hidden></div>
    <p class="ed-hint muted" data-i18n="editor.hint"></p>
    <div class="ed-rows"><div class="loading" data-i18n="detail.loading"></div></div>`;
  applyI18n(box);
  const $ = (cls) => box.querySelector(`.${cls}`);
  const rowsEl = $("ed-rows");

  load();

  async function load() {
    let data;
    try {
      data = await api(`${base}/segments`);
    } catch (err) {
      rowsEl.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
      return;
    }
    if (state.closed) return;
    state.segs = data.segments.map(trimmed);
    state.stamp = data.stamp;
    state.versions = data.versions;
    if (data.language) rowsEl.lang = data.language;
    renderAll();
    setStatus("saved");
  }

  /* --- rendering --- */

  function speakers() {
    const set = new Set(Object.keys(colorOf));
    for (const s of state.segs) set.add(s.speaker || "UNKNOWN");
    return [...set].sort();
  }

  function speakerOptions(current) {
    const opts = speakers().map((s) =>
      `<option value="${esc(s)}"${s === (current || "UNKNOWN") ? " selected" : ""}>${esc(speakerLabel(s, names))}</option>`);
    opts.push(`<option value="__new">${esc(t("editor.new_speaker"))}</option>`);
    return opts.join("");
  }

  function textHtml(seg) {
    return seg.words.map((w) => (w.redacted ? `<span class="red">${esc(w.word)}</span>` : esc(w.word))).join(" ");
  }

  function rowHtml(seg, i) {
    const spk = seg.speaker || "UNKNOWN";
    return `<div class="ed-row" data-i="${i}" style="--c:${colorOf[spk] || "var(--muted)"}">
      <div class="ed-side">
        <button type="button" class="ed-time" tabindex="-1" title="${esc(t("editor.play_from"))}">${clock(seg.start)}</button>
        <select class="ed-speaker" tabindex="-1" aria-label="${esc(t("editor.speaker_of", { n: i + 1 }))}">${speakerOptions(seg.speaker)}</select>
      </div>
      <div class="ed-text" contenteditable="plaintext-only" spellcheck="true" role="textbox" aria-multiline="false"
        tabindex="${i === state.active ? 0 : -1}"
        aria-label="${esc(t("editor.segment_n", { n: i + 1, time: clock(seg.start) }))}">${textHtml(seg)}</div>
    </div>`;
  }

  function renderAll() {
    state.active = Math.max(0, Math.min(state.active, state.segs.length - 1));
    rowsEl.innerHTML = state.segs.map(rowHtml).join("") || `<div class="empty">${esc(t("editor.empty"))}</div>`;
    state.playing = -1;
  }

  function renderRow(i) {
    const el = rowEl(i);
    if (!el) return renderAll();
    el.outerHTML = rowHtml(state.segs[i], i);
  }

  const rowEl = (i) => rowsEl.querySelector(`.ed-row[data-i="${i}"]`);
  const textEl = (i) => rowEl(i)?.querySelector(".ed-text");
  const rowOf = (node) => node?.closest?.(".ed-row") || node?.parentElement?.closest(".ed-row");
  const indexOf = (node) => {
    const row = rowOf(node);
    return row ? Number(row.dataset.i) : -1;
  };

  // roving tabindex: the whole list is one tab stop (the text of the row last focused)
  rowsEl.addEventListener("focusin", (e) => {
    const i = indexOf(e.target);
    if (i < 0 || i === state.active) return;
    const old = textEl(state.active);
    if (old) old.tabIndex = -1;
    state.active = i;
    const el = textEl(i);
    if (el) el.tabIndex = 0;
  });

  function focusRow(i, where = "start") {
    const el = textEl(i);
    if (!el) return;
    el.focus();
    setCaret(el, where === "end" ? el.textContent.length : typeof where === "number" ? where : 0);
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  /* --- history and saving --- */

  const snapshot = () => JSON.stringify(state.segs);

  function pushUndo() {
    state.undo.push(snapshot());
    if (state.undo.length > MAX_UNDO) state.undo.shift();
    state.redo = [];
    updateButtons();
  }

  function restoreFrom(from, to) {
    if (!from.length) return;
    const focused = indexOf(document.activeElement);
    to.push(snapshot());
    state.segs = JSON.parse(from.pop());
    state.typingRow = -1;
    renderAll();
    changed(false);
    if (focused >= 0) focusRow(Math.min(focused, state.segs.length - 1), "end");
  }

  function updateButtons() {
    $("ed-undo").disabled = !state.undo.length;
    $("ed-redo").disabled = !state.redo.length;
  }

  function changed(undoStep = true) {
    if (undoStep) pushUndo();
    state.dirty = true;
    updateButtons();
    setStatus("unsaved");
    clearTimeout(state.timer);
    state.timer = setTimeout(save, SAVE_DELAY);
  }

  function setStatus(kind, extra = "") {
    const el = $("ed-status");
    el.dataset.kind = kind;
    el.textContent = t(`editor.status_${kind}`, { error: extra });
  }

  async function save() {
    clearTimeout(state.timer);
    state.timer = null;
    if (state.saving) return state.saving.then(() => state.dirty && save());
    if (!state.dirty || state.conflict) return;
    state.dirty = false;
    setStatus("saving");
    const body = JSON.stringify({ base: state.stamp, segments: state.segs });
    state.saving = (async () => {
      try {
        const res = await api(`${base}/segments`, { method: "PUT", body });
        state.stamp = res.stamp;
        state.versions = res.versions;
        if (!state.dirty) setStatus("saved");
        if (!$("ed-versions").hidden) renderVersions();
      } catch (err) {
        state.dirty = true;
        if (err.code === "edit_conflict") {
          state.conflict = true;
          setStatus("conflict");
        } else {
          setStatus("error", err.message);
          state.timer = setTimeout(save, 5000); // try again
        }
      } finally {
        state.saving = null;
      }
    })();
    await state.saving;
    if (state.dirty && !state.conflict && !state.timer) state.timer = setTimeout(save, SAVE_DELAY);
  }

  const beforeUnload = (e) => {
    if (state.dirty || state.saving) {
      e.preventDefault();
      e.returnValue = "";
    }
  };
  window.addEventListener("beforeunload", beforeUnload);

  /* --- editing --- */

  rowsEl.addEventListener("input", (e) => {
    const i = indexOf(e.target);
    if (i < 0 || !e.target.classList.contains("ed-text")) return;
    const now = Date.now();
    if (state.typingRow !== i || now - state.lastInput > TYPING_GAP) pushUndo();
    state.typingRow = i;
    state.lastInput = now;
    const seg = state.segs[i];
    const text = e.target.textContent.replace(/\s+/g, " ").trim();
    seg.text = text;
    seg.words = realign(seg.words, text, seg.start, seg.end, seg.speaker);
    changed(false);
  });

  // after typing, show the words again as the server keeps them (redacted words highlighted)
  rowsEl.addEventListener("focusout", (e) => {
    if (!e.target.classList?.contains("ed-text")) return;
    const i = indexOf(e.target);
    state.typingRow = -1;
    if (i >= 0 && e.target.innerHTML !== textHtml(state.segs[i])) e.target.innerHTML = textHtml(state.segs[i]);
    if (state.dirty) save();
  });

  rowsEl.addEventListener("change", (e) => {
    if (!e.target.classList.contains("ed-speaker")) return;
    const i = indexOf(e.target);
    let value = e.target.value;
    if (value === "__new") {
      const used = new Set(speakers());
      let n = 0;
      while (used.has(`SPEAKER_${String(n).padStart(2, "0")}`)) n++;
      value = `SPEAKER_${String(n).padStart(2, "0")}`;
      colorOf[value] = `var(--spk-${speakers().length % 8})`;
    }
    pushUndo();
    setSpeaker(state.segs[i], value === "UNKNOWN" ? null : value);
    renderRow(i);
    changed(false);
    rowEl(i)?.querySelector(".ed-speaker").focus();
  });

  function setSpeaker(seg, speaker) {
    seg.speaker = speaker;
    seg.words = seg.words.map((w) => {
      const { speaker: _, ...rest } = w;
      return speaker ? { ...rest, speaker } : rest;
    });
  }

  rowsEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".ed-time");
    if (!btn || !hasAudio) return;
    audio.currentTime = state.segs[indexOf(btn)].start;
    audio.play();
  });

  function split(i, offset) {
    const seg = state.segs[i];
    const text = seg.text;
    // split at the end of the word the caret is in, so no word is cut in two
    let cut = offset;
    while (cut < text.length && /\S/.test(text[cut]) && cut > 0 && /\S/.test(text[cut - 1])) cut++;
    const k = tokensOf(text.slice(0, cut)).length;
    if (k === 0 || k >= seg.words.length) return toast(t("editor.cannot_split"));
    pushUndo();
    const left = seg.words.slice(0, k), right = seg.words.slice(k);
    const share = text.slice(0, cut).length / text.length;
    const mid = seg.start + (seg.end - seg.start) * share;
    const a = { ...seg, text: left.map((w) => w.word).join(" "), words: left, end: lastEnd(left) ?? mid };
    const b = { ...seg, text: right.map((w) => w.word).join(" "), words: right, start: firstStart(right) ?? mid };
    if (b.start < a.end) b.start = a.end;
    state.segs.splice(i, 1, a, b);
    renderAll();
    changed(false);
    focusRow(i + 1);
  }

  function merge(i) {
    if (i <= 0) return;
    pushUndo();
    const prev = state.segs[i - 1], cur = state.segs[i];
    const join = prev.text.length + 1;
    const merged = { ...prev, end: Math.max(prev.end, cur.end), text: `${prev.text} ${cur.text}`.trim(),
      words: [...prev.words, ...cur.words] };
    setSpeaker(merged, prev.speaker);
    state.segs.splice(i - 1, 2, merged);
    renderAll();
    changed(false);
    focusRow(i - 1, join);
  }

  function redactSelection() {
    const touched = [];
    rowsEl.querySelectorAll(".ed-text").forEach((el) => {
      const sel = selectionIn(el);
      if (!sel) return;
      const i = indexOf(el);
      const idx = tokensInRange(el.textContent, sel.start, sel.end);
      if (idx.length) touched.push([i, idx]);
    });
    if (!touched.length) return toast(t("editor.select_first"));
    const all = touched.every(([i, idx]) => idx.every((k) => state.segs[i].words[k]?.redacted));
    pushUndo();
    for (const [i, idx] of touched) {
      for (const k of idx) {
        const w = state.segs[i].words[k];
        if (!w) continue;
        if (all) delete w.redacted;
        else w.redacted = true;
      }
      renderRow(i);
    }
    changed(false);
    toast(t(all ? "editor.unredacted" : "editor.redacted"));
  }

  /* --- keyboard --- */

  function onKey(e) {
    if (!box.isConnected) return;
    const inText = e.target.classList?.contains("ed-text");
    const inEditor = box.contains(e.target) || e.target === document.body;
    const ctrl = e.ctrlKey || e.metaKey;
    if (!inEditor) return;
    if (ctrl && e.code === "Space") {
      e.preventDefault();
      if (hasAudio) audio.paused ? audio.play() : audio.pause();
    } else if (ctrl && (e.code === "Comma" || e.code === "Period")) {
      e.preventDefault();
      if (hasAudio) audio.currentTime = Math.max(0, audio.currentTime + (e.code === "Comma" ? -SEEK_STEP : SEEK_STEP));
    } else if (ctrl && e.shiftKey && e.code === "KeyX") {
      e.preventDefault();
      redactSelection();
    } else if (inText && e.altKey && e.shiftKey && e.code === "KeyS") {
      e.preventDefault();
      rowOf(e.target)?.querySelector(".ed-speaker")?.focus();
    } else if (ctrl && !e.shiftKey && e.code === "KeyZ") {
      e.preventDefault();
      restoreFrom(state.undo, state.redo);
    } else if (ctrl && (e.code === "KeyY" || (e.shiftKey && e.code === "KeyZ"))) {
      e.preventDefault();
      restoreFrom(state.redo, state.undo);
    } else if (e.target.classList?.contains("ed-speaker") && e.key === "Escape") {
      e.preventDefault();
      focusRow(indexOf(e.target), "end");
    } else if (inText && ctrl && e.key === "Enter") {
      e.preventDefault();
      const sel = selectionIn(e.target);
      split(indexOf(e.target), sel ? sel.start : 0);
    } else if (inText && (e.key === "Enter" || (e.altKey && (e.key === "ArrowDown" || e.key === "ArrowUp")))) {
      e.preventDefault();
      const i = indexOf(e.target) + (e.key === "ArrowUp" ? -1 : 1);
      if (i < 0 || i >= state.segs.length) return;
      focusRow(i);
      if (hasAudio) audio.currentTime = state.segs[i].start;
    } else if (inText && e.key === "Backspace") {
      const sel = selectionIn(e.target);
      if (sel && sel.collapsed && sel.start === 0 && indexOf(e.target) > 0) {
        e.preventDefault();
        merge(indexOf(e.target));
      }
    }
  }
  document.addEventListener("keydown", onKey, true);

  /* --- toolbar --- */

  $("ed-undo").addEventListener("click", () => restoreFrom(state.undo, state.redo));
  $("ed-redo").addEventListener("click", () => restoreFrom(state.redo, state.undo));
  // keep the selection/caret in the text when a toolbar button is pressed
  box.querySelector(".ed-bar").addEventListener("mousedown", (e) => {
    if (e.target.closest(".ed-split, .ed-merge, .ed-redact")) e.preventDefault();
  });
  $("ed-split").addEventListener("click", () => {
    const el = document.activeElement;
    if (!el?.classList.contains("ed-text")) return toast(t("editor.place_caret"));
    const sel = selectionIn(el);
    split(indexOf(el), sel ? sel.start : 0);
  });
  $("ed-merge").addEventListener("click", () => {
    const i = indexOf(document.activeElement);
    if (i <= 0) return toast(t("editor.place_caret"));
    merge(i);
  });
  $("ed-redact").addEventListener("click", redactSelection);
  for (const [btnCls, panelCls] of [["ed-keys-btn", "ed-keys"], ["ed-versions-btn", "ed-versions"]]) {
    $(btnCls).addEventListener("click", () => {
      const panel = $(panelCls);
      panel.hidden = !panel.hidden;
      $(btnCls).setAttribute("aria-expanded", String(!panel.hidden));
      if (panelCls === "ed-versions" && !panel.hidden) renderVersions();
    });
  }
  $("ed-done").addEventListener("click", close);

  function renderVersions() {
    const panel = $("ed-versions");
    if (!state.versions.length) {
      panel.innerHTML = `<p class="muted">${esc(t("editor.no_versions"))}</p>`;
      return;
    }
    panel.innerHTML = `<p class="muted">${esc(t("editor.versions_hint"))}</p><ul class="ed-version-list">${state.versions
      .map((v) => `<li><span>${esc(new Date(v.time).toLocaleString())}</span> <span class="pill">${esc(t(`editor.reason_${v.reason}`))}</span>
        <button class="btn ed-restore" data-v="${esc(v.id)}">${esc(t("editor.restore"))}</button></li>`)
      .join("")}</ul>`;
  }

  $("ed-versions").addEventListener("click", async (e) => {
    const btn = e.target.closest(".ed-restore");
    if (!btn) return;
    if (btn.dataset.armed !== "1") {
      btn.dataset.armed = "1";
      btn.textContent = t("editor.restore_confirm");
      return;
    }
    await save();
    try {
      const data = await api(`${base}/versions/${encodeURIComponent(btn.dataset.v)}/restore`, { method: "POST" });
      state.segs = data.segments.map(trimmed);
      state.stamp = data.stamp;
      state.versions = data.versions;
      state.undo = [];
      state.redo = [];
      state.conflict = false;
      renderAll();
      renderVersions();
      updateButtons();
      setStatus("saved");
      toast(t("editor.restored"));
    } catch (err) {
      toast(err.message);
    }
  });

  $("ed-status").addEventListener("click", () => {
    if (state.conflict) location.reload();
  });

  /* --- player: highlight the segment being spoken --- */

  function onTime() {
    const now = audio.currentTime;
    let found = -1;
    for (let lo = 0, hi = state.segs.length - 1; lo <= hi;) {
      const mid = (lo + hi) >> 1;
      if (state.segs[mid].start <= now) { found = mid; lo = mid + 1; } else hi = mid - 1;
    }
    if (found >= 0 && now > state.segs[found].end + 1.5) found = -1;
    if (found === state.playing) return;
    rowEl(state.playing)?.classList.remove("playing");
    state.playing = found;
    const row = rowEl(found);
    if (!row) return;
    row.classList.add("playing");
    const editing = document.activeElement?.classList.contains("ed-text");
    if (!audio.paused && !editing) {
      const r = row.getBoundingClientRect();
      if (r.top < 140 || r.bottom > window.innerHeight - 90) row.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
  audio.addEventListener("timeupdate", onTime);

  /* --- leaving --- */

  async function close() {
    await save();
    if (state.dirty) return; // not saved: stay so nothing is lost
    stop();
    onClose?.();
  }

  function stop() {
    if (state.closed) return;
    state.closed = true;
    if (state.dirty) save(); // leaving the page: last attempt
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("beforeunload", beforeUnload);
    audio.removeEventListener("timeupdate", onTime);
  }

  return { stop, flush: save, get dirty() { return state.dirty || !!state.saving; } };
}
