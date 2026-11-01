// Transcript library: folder tree, tiles with a title filter and multi-select, and full-text search results
// (GET /api/search) whose hits link to the moment in the transcript (#/t/<id>?t=<seconds>).
//
// The folder shown is `current`: null = all transcripts, "" = unfiled, "A/B" = that folder and what is below it.
// Folders are virtual (docs/ARCHITECTURE.md): moving a transcript only changes its folder path.

import { api } from "../core/api.js";
import { t } from "../core/i18n.js";
import { href } from "../core/router.js";
import { clock, date, duration, esc, number, speakerColor, realSpeakers, speakerLabel, speakerOrder, toast } from "../core/util.js";
import "../i18n/folders.js";
import "../i18n/translate.js";
import "../i18n/remove.js";
import { deleteTranscripts } from "./remove.js";
import { loadStyles as loadTranslateStyles } from "./translations.js";
import {
  STORE_FOLDER, askFolderName, assignFolder, baseName, confirmDelete, createFolder, deleteFolder, loadFolders, moveFolder, pickFolder,
} from "./folders.js";

const SEARCH_MIN = 2; // characters before the full-text search runs
const SEARCH_LIMIT = 60;
const SEARCH_DELAY_MS = 250;
const DRAG_TYPE = "application/x-mgtranscriber-ids";
const STORE_OPEN = "library.folders.open";
const STORE_HERE = "library.search_here";

const read = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
};
const write = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
};
const inFolder = (itemFolder, folder) => itemFolder === folder || (folder !== "" && itemFolder.startsWith(`${folder}/`));

// ui: { side, bar, selection, onFolder(path|null) }; elements the page provides (all optional)
export function mountLibrary(container, filterInput, resultsEl, ui = {}) {
  loadTranslateStyles();
  let items = [];
  let tree = { folders: [], unfiled: 0, total: 0 };
  let current = read(STORE_FOLDER, null);
  let open = new Set(read(STORE_OPEN, []));
  let searchHere = read(STORE_HERE, false);
  const selected = new Set();
  let searchTimer;
  let searchSeq = 0;
  let loaded = false;

  async function reload() {
    try {
      [items, tree] = await Promise.all([api("/api/transcripts"), loadFolders()]);
    } catch (err) {
      container.innerHTML = `<div class="empty">${esc(t("library.load_failed", { error: err.message }))}</div>`;
      return;
    }
    loaded = true;
    // a remembered folder that is gone (renamed or deleted elsewhere): back to everything
    if (typeof current === "string" && current !== "" && !tree.folders.some((f) => f.path === current)) current = null;
    for (const id of [...selected]) if (!items.some((i) => i.id === id)) selected.delete(id);
    renderAll();
  }

  function renderAll() {
    renderSide();
    renderBar();
    renderSelection();
    render();
  }

  function setFolder(path) {
    current = path;
    write(STORE_FOLDER, current);
    if (typeof path === "string") for (const a of ancestors(path)) open.add(a);
    write(STORE_OPEN, [...open]);
    selected.clear();
    ui.onFolder?.(current);
    renderAll();
    search();
  }

  const ancestors = (path) => {
    const parts = path.split("/");
    return parts.slice(0, -1).map((_, i) => parts.slice(0, i + 1).join("/"));
  };

  /* --- folder tree (nested lists of buttons; works with the keyboard and a screen reader) --- */

  function renderSide() {
    if (!ui.side) return;
    const active = document.activeElement;
    const keep = ui.side.contains(active) ? (active.dataset.folder !== undefined ? `f:${active.dataset.folder}` : active.dataset.toggle !== undefined ? `t:${active.dataset.toggle}` : null) : null;
    const children = new Map();
    for (const f of tree.folders) {
      if (!children.has(f.parent)) children.set(f.parent, []);
      children.get(f.parent).push(f);
    }
    const item = (f) => {
      const kids = children.get(f.path) || [];
      const expanded = open.has(f.path);
      const toggle = kids.length
        ? `<button type="button" class="ft-toggle" data-toggle="${esc(f.path)}" aria-expanded="${expanded}"
             aria-label="${esc(t(expanded ? "folders.collapse" : "folders.expand", { name: f.name }))}"><span aria-hidden="true">${expanded ? "▾" : "▸"}</span></button>`
        : `<span class="ft-spacer"></span>`;
      return `<li class="ft-item">
        <div class="ft-row" data-drop="${esc(f.path)}">${toggle}
          <button type="button" class="ft-label" data-folder="${esc(f.path)}" ${current === f.path ? 'aria-current="true"' : ""} title="${esc(f.path)}">
            <span class="ft-name">${esc(f.name)}</span><span class="ft-count">${number(f.total)}</span></button>
        </div>
        ${kids.length ? `<ul class="ft-children"${expanded ? "" : " hidden"}>${kids.map(item).join("")}</ul>` : ""}
      </li>`;
    };
    const top = (children.get("") || []).map(item).join("");
    ui.side.innerHTML = `
      <div class="fs-select">
        <label for="fs-select-input" class="sr-only">${esc(t("folders.pick"))}</label>
        <select id="fs-select-input" class="select"></select>
      </div>
      <nav class="fs-tree" aria-label="${esc(t("folders.nav"))}">
        <div class="fs-head"><h3>${esc(t("folders.title"))}</h3>
          <button type="button" class="icon-btn fs-new" data-action="new" title="${esc(t("folders.new"))}" aria-label="${esc(t("folders.new"))}">＋</button></div>
        <ul class="ft-root">
          <li class="ft-item"><div class="ft-row">
            <span class="ft-spacer"></span>
            <button type="button" class="ft-label" data-folder="*" ${current === null ? 'aria-current="true"' : ""}>
              <span class="ft-name">${esc(t("folders.all"))}</span><span class="ft-count">${number(tree.total)}</span></button></div></li>
          <li class="ft-item"><div class="ft-row" data-drop="">
            <span class="ft-spacer"></span>
            <button type="button" class="ft-label" data-folder="" ${current === "" ? 'aria-current="true"' : ""}>
              <span class="ft-name">${esc(t("folders.unfiled"))}</span><span class="ft-count">${number(tree.unfiled)}</span></button></div></li>
          ${top}
        </ul>
      </nav>`;
    // the mobile <select>: counts in the option text
    const select = ui.side.querySelector("select");
    select.innerHTML =
      `<option value="*"${current === null ? " selected" : ""}>${esc(t("folders.all"))} (${number(tree.total)})</option>` +
      `<option value=""${current === "" ? " selected" : ""}>${esc(t("folders.unfiled"))} (${number(tree.unfiled)})</option>` +
      tree.folders.map((f) => `<option value="${esc(f.path)}"${current === f.path ? " selected" : ""}>${" ".repeat(f.depth - 1)}${esc(f.name)} (${number(f.total)})</option>`).join("");
    select.addEventListener("change", () => setFolder(select.value === "*" ? null : select.value));
    if (keep) {
      const [kind, value] = [keep[0], keep.slice(2)];
      ui.side.querySelector(kind === "f" ? `[data-folder="${CSS.escape(value)}"]` : `[data-toggle="${CSS.escape(value)}"]`)?.focus();
    }
  }

  function onSideClick(e) {
    const label = e.target.closest(".ft-label");
    const toggle = e.target.closest(".ft-toggle");
    const create = e.target.closest("[data-action=new]");
    if (label) {
      setFolder(label.dataset.folder === "*" ? null : label.dataset.folder);
    } else if (toggle) {
      const path = toggle.dataset.toggle;
      if (open.has(path)) open.delete(path);
      else open.add(path);
      write(STORE_OPEN, [...open]);
      renderSide();
    } else if (create) {
      newFolder("");
    }
  }

  /* --- breadcrumb and folder actions --- */

  function renderBar() {
    if (!ui.bar) return;
    const real = typeof current === "string" && current !== "";
    const crumbs = [`<li><button type="button" class="crumb" data-crumb="*">${esc(t("folders.all"))}</button></li>`];
    if (current === "") {
      crumbs.push(`<li><span class="crumb current" aria-current="page">${esc(t("folders.unfiled"))}</span></li>`);
    } else if (real) {
      const parts = current.split("/");
      parts.forEach((name, i) => {
        const path = parts.slice(0, i + 1).join("/");
        crumbs.push(i === parts.length - 1
          ? `<li><span class="crumb current" aria-current="page">${esc(name)}</span></li>`
          : `<li><button type="button" class="crumb" data-crumb="${esc(path)}">${esc(name)}</button></li>`);
      });
    }
    const where = current === null ? "" : `<label class="here"><input type="checkbox" id="search-here"${searchHere ? " checked" : ""}> <span>${esc(t("folders.search_here"))}</span></label>`;
    const actions = real
      ? `<div class="folder-actions" role="group" aria-label="${esc(t("folders.actions"))}">
           <button type="button" class="btn small" data-action="sub">${esc(t("folders.new_sub"))}</button>
           <button type="button" class="btn small" data-action="rename">${esc(t("folders.rename"))}</button>
           <button type="button" class="btn small danger" data-action="delete">${esc(t("folders.delete"))}</button>
         </div>`
      : "";
    ui.bar.innerHTML = `<nav class="crumbs" aria-label="${esc(t("folders.path"))}"><ol>${crumbs.join("")}</ol></nav>${actions}${where}`;
  }

  async function onBarClick(e) {
    const crumb = e.target.closest("[data-crumb]");
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (crumb) return setFolder(crumb.dataset.crumb === "*" ? null : crumb.dataset.crumb);
    if (e.target.id === "search-here") {
      searchHere = e.target.checked;
      write(STORE_HERE, searchHere);
      return search();
    }
    if (action === "sub") return newFolder(`${current}/`);
    if (action === "rename") return renameFolder();
    if (action === "delete") return removeFolder();
  }

  async function guarded(fn) {
    try {
      await fn();
    } catch (err) {
      toast(err.message);
      reload();
    }
  }

  const newFolder = (prefix) => guarded(async () => {
    const name = await askFolderName({
      title: prefix ? t("folders.create_sub_title", { parent: prefix.replace(/\/$/, "") }) : t("folders.create_title"),
      value: prefix, confirm: t("folders.save"),
    });
    if (name === null) return;
    const made = await createFolder(name);
    await reload();
    setFolder(made.path);
    toast(t("folders.created_toast", { name: baseName(made.path) }));
  });

  const renameFolder = () => guarded(async () => {
    const name = await askFolderName({ title: t("folders.rename_title"), value: current, hint: t("folders.rename_hint") });
    if (name === null || name === current) return;
    const old = current;
    const moved = await moveFolder(old, name);
    current = moved.path;
    write(STORE_FOLDER, current);
    open = new Set([...open].map((p) => (inFolder(p, old) ? moved.path + p.slice(old.length) : p)));
    write(STORE_OPEN, [...open]);
    await reload();
    ui.onFolder?.(current);
    renderAll();
    toast(t("folders.renamed_toast", { name: baseName(moved.path) }));
  });

  const removeFolder = () => guarded(async () => {
    const info = tree.folders.find((f) => f.path === current);
    const subfolders = tree.folders.filter((f) => f.path.startsWith(`${current}/`)).length;
    const to = await confirmDelete({ path: current, total: info ? info.total : 0, subfolders });
    if (to === null) return;
    const result = await deleteFolder(current, to);
    const next = result.target === "" ? null : result.target;
    await reload();
    setFolder(next);
    toast(t("folders.deleted_toast"));
  });

  /* --- tiles --- */

  const visible = () => {
    const q = filterInput.value.trim().toLowerCase();
    return items.filter((item) =>
      (current === null || inFolder(item.folder || "", current)) && (!q || item.title.toLowerCase().includes(q)));
  };

  function render() {
    const q = filterInput.value.trim();
    const shown = visible();
    if (!shown.length) {
      // while searching, the text hits above are the answer; a big "nothing matches" box would mislead
      container.innerHTML = q
        ? `<p class="muted-note">${esc(t("library.no_title_match"))}</p>`
        : `<div class="empty">${esc(t(items.length ? (current === null ? "library.no_match" : "folders.empty") : "library.empty"))}</div>`;
      return;
    }
    container.innerHTML = shown.map(tile).join("");
  }

  function tile(item) {
    const order = speakerOrder(item.speakers);
    const total = Object.values(item.speakers).reduce((a, b) => a + b, 0) || 1;
    const bar = order.map((s, i) => `<i style="width:${(item.speakers[s] / total) * 100}%;background:${speakerColor(i)}"></i>`).join("");
    const chips = order
      .map((s, i) => `<span class="chip"><span class="dot" style="background:${speakerColor(i)}"></span>${esc(speakerLabel(s, item.names))}</span>`)
      .join("");
    const folder = item.folder && item.folder !== current
      ? `<span class="tile-folder" title="${esc(t("folders.in_folder", { folder: item.folder }))}">${esc(item.folder)}</span>` : "";
    return `<div class="card tile${selected.has(item.id) ? " selected" : ""}" draggable="true" data-id="${esc(item.id)}">
      <label class="tile-check"><input type="checkbox" data-select="${esc(item.id)}"${selected.has(item.id) ? " checked" : ""}
        aria-label="${esc(t("folders.select_tile", { title: item.title }))}"></label>
      <button type="button" class="icon-btn tile-del" data-del="${esc(item.id)}" title="${esc(t("remove.action"))}"
        aria-label="${esc(t("remove.aria", { title: item.title }))}">🗑</button>
      <a class="tile-link" draggable="false" href="${href(`/t/${encodeURIComponent(item.id)}`)}">
        <h3>${esc(item.title)}</h3>
        <p class="meta"><span>${date(item.date)}</span><span>${duration(item.duration)}</span><span>${t("common.speakers", { n: realSpeakers(item.speakers).length })}</span>${item.has_audio ? `<span>${t("common.audio")}</span>` : ""}${folder}</p>
        <div class="bar">${bar}</div>
        <div class="chips">${chips}</div>
      </a>
    </div>`;
  }

  /* --- selection ("Move to…") --- */

  function renderSelection() {
    const box = ui.selection;
    if (!box) return;
    box.hidden = selected.size === 0;
    if (!selected.size) {
      box.replaceChildren();
      return;
    }
    box.innerHTML = `<span class="sel-count" role="status">${esc(t("folders.selected", { n: selected.size }))}</span>
      <button type="button" class="btn small primary" data-sel="move">${esc(t("folders.move"))}</button>
      <button type="button" class="btn small danger" data-sel="delete">${esc(t("remove.action"))}</button>
      <button type="button" class="btn small" data-sel="all">${esc(t("folders.select_all"))}</button>
      <button type="button" class="btn small" data-sel="clear">${esc(t("folders.clear"))}</button>`;
  }

  function onContainerChange(e) {
    const box = e.target.closest("[data-select]");
    if (!box) return;
    if (box.checked) selected.add(box.dataset.select);
    else selected.delete(box.dataset.select);
    box.closest(".tile").classList.toggle("selected", box.checked);
    renderSelection();
  }

  async function onSelectionClick(e) {
    const what = e.target.closest("[data-sel]")?.dataset.sel;
    if (what === "clear") {
      selected.clear();
      renderSelection();
      render();
    } else if (what === "all") {
      for (const item of visible()) selected.add(item.id);
      renderSelection();
      render();
    } else if (what === "move") {
      moveSelected([...selected]);
    } else if (what === "delete") {
      removeTranscripts([...selected]);
    }
  }

  const removeTranscripts = (ids) => guarded(async () => {
    const chosen = ids.map((id) => items.find((i) => i.id === id)).filter(Boolean);
    const result = await deleteTranscripts(chosen);
    if (!result) return;
    for (const id of result.deleted) selected.delete(id);
    await reload();
    search();
    container.setAttribute("tabindex", "-1");
    container.focus({ preventScroll: true });
  });

  function onContainerClick(e) {
    const del = e.target.closest("[data-del]");
    if (!del) return;
    e.preventDefault();
    removeTranscripts([del.dataset.del]);
  }

  const moveSelected = (ids) => guarded(async () => {
    if (!ids.length) return;
    const one = ids.length === 1 ? items.find((i) => i.id === ids[0]) : null;
    const folder = await pickFolder({
      title: one ? t("folders.move_title_one", { title: one.title }) : t("folders.move_title", { n: ids.length }),
      tree, selected: typeof current === "string" ? current : "",
    });
    if (folder === null) return;
    await moveTo(ids, folder);
  });

  async function moveTo(ids, folder) {
    const result = await assignFolder(ids, folder);
    selected.clear();
    await reload();
    toast(t("folders.moved_toast", { n: result.moved || ids.length, folder: folder === "" ? t("folders.unfiled") : folder }));
    container.setAttribute("tabindex", "-1");
    container.focus({ preventScroll: true });
  }

  /* --- drag & drop: tiles onto a folder in the tree --- */

  function onDragStart(e) {
    const tileEl = e.target.closest?.(".tile");
    if (!tileEl || !e.dataTransfer) return;
    const id = tileEl.dataset.id;
    const ids = selected.has(id) ? [...selected] : [id];
    e.dataTransfer.setData(DRAG_TYPE, JSON.stringify(ids));
    e.dataTransfer.setData("text/plain", ids.join("\n"));
    e.dataTransfer.effectAllowed = "move";
    for (const el of container.querySelectorAll(".tile")) el.classList.toggle("dragging", ids.includes(el.dataset.id));
    ui.side?.classList.add("drop-active");
  }
  const endDrag = () => {
    ui.side?.classList.remove("drop-active");
    for (const el of document.querySelectorAll(".drop-over")) el.classList.remove("drop-over");
    for (const el of container.querySelectorAll(".dragging")) el.classList.remove("dragging");
  };
  const hasIds = (e) => e.dataTransfer && [...e.dataTransfer.types].includes(DRAG_TYPE);
  function onSideDragOver(e) {
    const row = e.target.closest?.("[data-drop]");
    if (!row || !hasIds(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    row.classList.add("drop-over");
  }
  function onSideDragLeave(e) {
    const row = e.target.closest?.("[data-drop]");
    if (row && !row.contains(e.relatedTarget)) row.classList.remove("drop-over");
  }
  function onSideDrop(e) {
    const row = e.target.closest?.("[data-drop]");
    if (!row || !hasIds(e)) return;
    e.preventDefault();
    const ids = JSON.parse(e.dataTransfer.getData(DRAG_TYPE) || "[]");
    endDrag();
    guarded(() => moveTo(ids, row.dataset.drop));
  }

  /* --- full-text search --- */

  async function search() {
    const q = filterInput.value.trim();
    const seq = ++searchSeq;
    if (q.length < SEARCH_MIN) {
      resultsEl.hidden = true;
      resultsEl.replaceChildren();
      return;
    }
    const params = { q, limit: SEARCH_LIMIT };
    if (searchHere && current !== null) params.folder = current; // "" = unfiled; subfolders are included
    let data;
    try {
      data = await api(`/api/search?${new URLSearchParams(params)}`);
    } catch (err) {
      data = { error: err.message };
    }
    if (seq !== searchSeq) return; // a newer query is on its way
    resultsEl.hidden = false;
    if (data.error) {
      resultsEl.innerHTML = `<p class="muted-note">${esc(data.error)}</p>`;
      return;
    }
    if (!data.total) {
      resultsEl.innerHTML = `<p class="muted-note">${esc(t("search.none"))}</p>`;
      return;
    }
    const groups = new Map();
    for (const hit of data.results) {
      if (!groups.has(hit.transcript_id)) groups.set(hit.transcript_id, []);
      groups.get(hit.transcript_id).push(hit);
    }
    const more = data.total - data.results.length;
    resultsEl.innerHTML =
      `<p class="muted-note">${esc(t("search.summary", { n: data.total }))}</p>` +
      [...groups.values()].map(group).join("") +
      (more > 0 ? `<p class="muted-note">${esc(t("search.more", { n: more }))}</p>` : "");
  }

  function group(hits) {
    const first = hits[0];
    const rows = hits
      .map((hit) => {
        const who = hit.speaker ? speakerLabel(hit.speaker, hit.speaker_name ? { [hit.speaker]: hit.speaker_name } : {}) : "";
        const link = href(`/t/${encodeURIComponent(hit.transcript_id)}`,
          { t: Math.round(hit.start * 10) / 10, ...(hit.lang ? { lang: hit.lang } : {}) });
        return `<li><a class="hit" href="${link}"><span class="time">${clock(hit.start)}</span>` +
          (who ? `<span class="who">${esc(who)}</span>` : "") +
          `<span class="snippet">` +
          (hit.lang ? `<abbr class="hit-lang" title="${esc(t("translate.hit_lang", { lang: hit.lang }))}">${esc(hit.lang.toUpperCase())}</abbr> ` : "") +
          `${marked(hit)}</span></a></li>`;
      })
      .join("");
    return `<section class="card hit-group">
      <h3><a href="${href(`/t/${encodeURIComponent(first.transcript_id)}`)}">${esc(first.title)}</a></h3>
      <ul>${rows}</ul>
    </section>`;
  }

  // snippet with <mark> around the highlight ranges. The server counts characters (code points), JS strings
  // count UTF-16 units, so slice an array of code points: emoji and other astral characters stay aligned.
  function marked(hit) {
    const chars = Array.from(hit.snippet);
    const part = (a, b) => esc(chars.slice(a, b).join(""));
    let out = "";
    let pos = 0;
    for (const [a, b] of hit.highlights) {
      out += part(pos, a) + `<mark>${part(a, b)}</mark>`;
      pos = b;
    }
    return out + part(pos);
  }

  /* --- wiring --- */

  filterInput.addEventListener("input", () => {
    render();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(search, SEARCH_DELAY_MS);
  });
  container.addEventListener("change", onContainerChange);
  container.addEventListener("click", onContainerClick);
  container.addEventListener("dragstart", onDragStart);
  container.addEventListener("dragend", endDrag);
  ui.side?.addEventListener("click", onSideClick);
  ui.side?.addEventListener("dragover", onSideDragOver);
  ui.side?.addEventListener("dragleave", onSideDragLeave);
  ui.side?.addEventListener("drop", onSideDrop);
  ui.bar?.addEventListener("click", onBarClick);
  ui.selection?.addEventListener("click", onSelectionClick);
  reload().then(() => ui.onFolder?.(current));
  return {
    reload() {
      reload();
      search();
    },
    current: () => current,
    folders: () => tree,
    refreshFolders: async () => {
      tree = await loadFolders();
      renderSide();
    },
    cleanup: () => clearTimeout(searchTimer),
  };
}
