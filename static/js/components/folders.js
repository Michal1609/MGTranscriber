// Library folders: API calls, the dialogs (name, "Move to…", delete) and the folder <select> of the new-job form.
// Folders are virtual: a transcript has one folder path ("Podcasts/Cílek", "" = unfiled), see docs/ARCHITECTURE.md.

import { api } from "../core/api.js";
import "../i18n/folders.js";
import { t } from "../core/i18n.js";
import { esc, toast } from "../core/util.js";

export const STORE_FOLDER = "library.folder"; // localStorage: the folder open in the library (JSON: null | "" | path)

export const loadFolders = () => api("/api/folders");
export const baseName = (path) => path.split("/").pop();
export const parentOf = (path) => path.split("/").slice(0, -1).join("/");
export const folderLabel = (path) => (path === "" ? t("folders.unfiled") : path);

// ---- API ---------------------------------------------------------------------------------------------

export const createFolder = (path) => api("/api/folders", { method: "POST", body: JSON.stringify({ path }) });
export const moveFolder = (path, to) => api("/api/folders/move", { method: "POST", body: JSON.stringify({ path, to }) });
export const deleteFolder = (path, moveTo) =>
  api(`/api/folders?${new URLSearchParams({ path, move_to: moveTo })}`, { method: "DELETE" });
export const assignFolder = (ids, folder) =>
  api("/api/folders/assign", { method: "POST", body: JSON.stringify({ ids, folder }) });

// ---- options -----------------------------------------------------------------------------------------

// <option>s for a folder <select>: the tree flattened, children indented. `exclude` leaves out a folder and what is in it.
export function folderOptions(tree, { selected = "", exclude = null, unfiled = true } = {}) {
  const out = [];
  if (unfiled) out.push(`<option value=""${selected === "" ? " selected" : ""}>${esc(t("folders.unfiled"))}</option>`);
  for (const f of tree.folders) {
    if (exclude && (f.path === exclude || f.path.startsWith(`${exclude}/`))) continue;
    const pad = " ".repeat(f.depth - 1);
    out.push(`<option value="${esc(f.path)}"${f.path === selected ? " selected" : ""}>${pad}${esc(f.name)}</option>`);
  }
  return out.join("");
}

// ---- dialogs -----------------------------------------------------------------------------------------

let dialogCount = 0;

// Native <dialog>: focus is trapped, Escape cancels, focus returns to the opener. `body` is HTML; `collect(form)`
// returns the result (or { error } to keep the dialog open). Resolves with the result, or null when cancelled.
export function modal({ title, body, confirm, danger = false, collect, wide = false, focusCancel = false }) {
  return new Promise((resolve) => {
    const id = `fx-dialog-${++dialogCount}`;
    const opener = document.activeElement;
    const dlg = document.createElement("dialog");
    dlg.className = `fx-dialog${wide ? " wide" : ""}`;
    dlg.setAttribute("aria-labelledby", `${id}-title`);
    dlg.innerHTML = `<form method="dialog" novalidate>
      <h2 id="${id}-title">${esc(title)}</h2>
      <div class="fx-body">${body}</div>
      <p class="fx-error" role="alert"></p>
      <div class="fx-actions">
        <button type="button" class="btn fx-cancel">${esc(t("folders.cancel"))}</button>
        <button type="submit" class="btn ${danger ? "danger" : "primary"} fx-ok">${esc(confirm)}</button>
      </div>
    </form>`;
    document.body.append(dlg);
    const form = dlg.querySelector("form");
    let result = null;
    const finish = (value) => {
      result = value;
      dlg.close();
    };
    // A route change (browser Back, a link) cancels the dialog like Cancel does; the listener lives as long as it.
    const onRoute = () => {
      if (result === null) finish(null);  // a value already confirmed wins over a racing route change
    };
    window.addEventListener("hashchange", onRoute);
    dlg.addEventListener("close", () => {
      window.removeEventListener("hashchange", onRoute);
      dlg.remove();
      if (opener && opener.isConnected) opener.focus();
      resolve(result);
    });
    dlg.querySelector(".fx-cancel").addEventListener("click", () => finish(null));
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const value = collect(form);
      if (value && value.error) {
        form.querySelector(".fx-error").textContent = value.error;
        return;
      }
      finish(value);
    });
    dlg.showModal();
    (form.querySelector("[autofocus], input[type=text], select, input") || form.querySelector(focusCancel ? ".fx-cancel" : ".fx-ok"))?.focus();
  });
}

// A yes/no question with explaining text. Resolves true or false.
export function confirmDialog({ title, text, confirm, danger = false }) {
  return modal({ title, confirm, danger, body: `<p>${esc(text)}</p>`, collect: () => true }).then(Boolean);
}

// One name / path field. Resolves with the text, or null.
export function askFolderName({ title, value = "", hint = t("folders.name_hint"), confirm = t("folders.save") }) {
  return modal({
    title,
    confirm,
    body: `<label class="field"><span>${esc(t("folders.name_label"))}</span>
        <input type="text" name="name" value="${esc(value)}" maxlength="255" autocomplete="off" aria-describedby="fx-hint"></label>
      <p class="fx-hint" id="fx-hint">${esc(hint)}</p>`,
    collect(form) {
      const name = form.elements.name.value.trim();
      return name ? name : { error: t("errors.folder_invalid") };
    },
  });
}

// Choose a destination folder: existing ones, Unfiled, or a new one. Resolves with the path ("" = unfiled) or null.
export function pickFolder({ title, tree, selected = "", exclude = null, confirm = t("folders.move_confirm") }) {
  return modal({
    title,
    confirm,
    body: `<label class="field"><span>${esc(t("folders.destination"))}</span>
        <select name="folder" class="select">${folderOptions(tree, { selected, exclude })}
          <option value="__new__">${esc(t("folders.new_option"))}</option></select></label>
      <label class="field fx-new" hidden><span>${esc(t("folders.new_name"))}</span>
        <input type="text" name="newname" maxlength="255" autocomplete="off" aria-describedby="fx-hint"></label>
      <p class="fx-hint fx-new" id="fx-hint" hidden>${esc(t("folders.name_hint"))}</p>`,
    collect(form) {
      const choice = form.elements.folder.value;
      if (choice !== "__new__") return { folder: choice };
      const name = form.elements.newname.value.trim();
      return name ? { folder: name } : { error: t("errors.folder_invalid") };
    },
  }).then((result) => (result ? result.folder : null));
}

// Wire the "New folder…" option of a pickFolder dialog (called from the form's change event, see below).
document.addEventListener("change", (e) => {
  const select = e.target;
  if (!(select instanceof HTMLSelectElement) || select.name !== "folder" || !select.closest(".fx-dialog")) return;
  const isNew = select.value === "__new__";
  for (const el of select.closest("form").querySelectorAll(".fx-new")) el.hidden = !isNew;
  if (isNew) select.closest("form").elements.newname.focus();
});

// Delete a folder: where do its transcripts go? Resolves with "parent" | "unfiled" | null.
export function confirmDelete({ path, total, subfolders }) {
  const parent = parentOf(path);
  const info = [total ? t("folders.delete_info", { n: total }) : t("folders.delete_empty")];
  if (subfolders) info.push(t("folders.delete_subfolders", { n: subfolders }));
  const options = parent
    ? `<label class="radio"><input type="radio" name="to" value="parent" checked> ${esc(t("folders.delete_to_parent", { parent }))}</label>
       <label class="radio"><input type="radio" name="to" value="unfiled"> ${esc(t("folders.delete_to_unfiled"))}</label>`
    : `<input type="hidden" name="to" value="unfiled">${total ? `<p>${esc(t("folders.delete_to_unfiled"))}</p>` : ""}`;
  return modal({
    title: t("folders.delete_title", { name: baseName(path) }),
    confirm: t("folders.delete_confirm"),
    danger: true,
    body: `<p>${esc(info.join(" "))}</p>${total ? options : '<input type="hidden" name="to" value="parent">'}`,
    collect: (form) => form.elements.to.value || "parent",
  });
}

// ---- folder line of the detail page ------------------------------------------------------------------

// "Folder: Podcasts / Cílek  [Move to…]". Clicking a part opens the library at that folder. `onMoved(path)` after a move.
export function mountFolderLine(box, { id, title, folder, onMoved }) {
  const render = () => {
    const parts = folder ? folder.split("/") : [];
    const crumbs = parts.length
      ? parts.map((name, i) => `<li><button type="button" class="crumb" data-path="${esc(parts.slice(0, i + 1).join("/"))}">${esc(name)}</button></li>`).join("")
      : `<li><button type="button" class="crumb" data-path="">${esc(t("folders.detail_unfiled"))}</button></li>`;
    box.innerHTML = `<span class="d-folder-label">${esc(t("folders.detail_label"))}:</span>
      <nav class="crumbs" aria-label="${esc(t("folders.path"))}"><ol>${crumbs}</ol></nav>
      <button type="button" class="btn small d-move">${esc(t("folders.detail_move"))}</button>`;
  };
  render();
  box.addEventListener("click", async (e) => {
    const crumb = e.target.closest(".crumb");
    if (crumb) {
      try {
        localStorage.setItem(STORE_FOLDER, JSON.stringify(crumb.dataset.path));
      } catch {}
      location.hash = "#/";
      return;
    }
    if (!e.target.closest(".d-move")) return;
    try {
      const tree = await loadFolders();
      const target = await pickFolder({ title: t("folders.move_title_one", { title }), tree, selected: folder });
      if (target === null || target === folder) return;
      const done = await api(`/api/transcripts/${encodeURIComponent(id)}/folder`, { method: "PUT", body: JSON.stringify({ folder: target }) });
      folder = done.folder;
      render();
      box.querySelector(".d-move").focus();
      toast(t("folders.moved_toast", { n: 1, folder: folderLabel(folder) }));
      onMoved?.(folder);
    } catch (err) {
      toast(err.message);
    }
  });
}

// ---- the folder <select> of the new-job form ---------------------------------------------------------

// Mount a folder picker into `box` (needs <select class="select" id="job-folder"> + <input id="job-folder-new">).
// `value()` is what to send as `folder`: undefined when none is chosen (the server then keeps / leaves unfiled).
export function mountFolderField(box, getDefault) {
  const select = box.querySelector("select");
  const newInput = box.querySelector("input");
  let tree = { folders: [] };
  function fill(selected) {
    const sel = selected ?? "";
    const opt = (value, label) => `<option value="${esc(value)}"${value === sel ? " selected" : ""}>${esc(label)}</option>`;
    select.innerHTML = opt("", t("folders.form_none")) + tree.folders.map((f) => opt(f.path, f.path)).join("") +
      `<option value="__new__">${esc(t("folders.form_new"))}</option>`;
    newInput.hidden = select.value !== "__new__";
  }
  function sync() {
    newInput.hidden = select.value !== "__new__";
  }
  select.addEventListener("change", () => {
    select.dataset.touched = "1"; // the user chose: the library's folder no longer moves the choice
    sync();
    if (!newInput.hidden) newInput.focus();
  });
  loadFolders().then((data) => {
    tree = data;
    fill(getDefault() ?? "");
  }).catch(() => fill(""));
  return {
    // call when the folder open in the library changes: the form follows it unless the user chose another
    follow(path) {
      if (select.dataset.touched === "1") return;
      if ([...select.options].some((o) => o.value === (path ?? ""))) select.value = path ?? "";
      sync();
    },
    refresh(data) {
      tree = data;
      const keep = select.value === "__new__" ? "" : select.value;
      fill(keep);
    },
    value() {
      if (select.value === "__new__") return newInput.value.trim() || undefined;
      return select.value || undefined;
    },
    reset() {
      newInput.value = "";
    },
  };
}
