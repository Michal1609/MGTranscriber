// "Watching": folders, podcast feeds and YouTube channels/playlists that are checked automatically.
// New items are added to the job queue; the page lists sources with state and last error, and lets you
// add, edit, pause, check now (also as a dry run) and delete them.

import { api, getInfo } from "../core/api.js";
import { applyI18n, t } from "../core/i18n.js";
import { isCurrent, registerPage } from "../core/router.js";
import { $, date, esc, pollTimeout, toast } from "../core/util.js";
import { loadFolders } from "../components/folders.js";

const TEMPLATE = `
  <header class="detail-head"><h1 data-i18n="watch.title"></h1>
    <button class="btn primary w-add" data-i18n="watch.add"></button></header>
  <p class="meta" data-i18n="watch.intro"></p>
  <section class="card w-form-card" hidden></section>
  <section class="w-list"><div class="loading" data-i18n="common.loading"></div></section>`;

const FORM = `
  <form class="w-form" autocomplete="off">
    <h2 class="w-form-title"></h2>
    <div class="row">
      <label class="field"><span data-i18n="watch.kind"></span>
        <select name="kind" class="select">
          <option value="folder" data-i18n="watch.kind_folder"></option>
          <option value="rss" data-i18n="watch.kind_rss"></option>
          <option value="youtube" data-i18n="watch.kind_youtube"></option>
        </select></label>
      <label class="field grow"><span class="w-target-label"></span><input name="target" type="text" required></label>
    </div>
    <div class="row">
      <label class="field grow"><span data-i18n="watch.name"></span><input name="name" type="text" maxlength="120"></label>
      <label class="field"><span data-i18n="home.language"></span>
        <select name="language" class="select">
          <option value="" data-i18n="watch.lang_default"></option>
          <option value="auto" data-i18n="home.lang_auto"></option>
          <option value="en" data-i18n="home.lang_en"></option>
          <option value="cs" data-i18n="home.lang_cs"></option>
        </select></label>
      <label class="field speakers-field"><span data-i18n="home.speakers"></span>
        <input name="speakers" type="number" min="1" max="20" data-i18n-placeholder="home.speakers_placeholder"></label>
      <label class="field"><span data-i18n="home.engine"></span><select name="engine" class="select"></select></label>
    </div>
    <div class="row">
      <label class="field"><span data-i18n="watch.mode"></span>
        <select name="mode" class="select">
          <option value="new" data-i18n="watch.mode_new"></option>
          <option value="last_n" data-i18n="watch.mode_last_n"></option>
        </select></label>
      <label class="field speakers-field w-lastn"><span data-i18n="watch.last_n"></span>
        <input name="last_n" type="number" min="1" max="50" value="5"></label>
      <label class="field speakers-field"><span class="w-interval-label"></span>
        <input name="interval" type="number" min="1" step="1" required></label>
      <label class="field speakers-field"><span data-i18n="watch.max_per_check"></span>
        <input name="max_per_check" type="number" min="1" max="20" value="5"></label>
    </div>
    <div class="row">
      <label class="check w-recursive"><input name="recursive" type="checkbox"> <span data-i18n="watch.recursive"></span></label>
      <label class="check"><input name="analyze" type="checkbox"> <span data-i18n="watch.analyze"></span></label>
      <label class="field"><span data-i18n="watch.translate_to"></span>
        <select name="translate_to" class="select"></select></label>
      <label class="field grow"><span data-i18n="watch.webhook"></span>
        <input name="webhook_url" type="url" data-i18n-placeholder="watch.webhook_placeholder"></label>
    </div>
    <div class="row">
      <label class="field grow"><span data-i18n="folders.watch_label"></span>
        <input name="folder" type="text" list="watch-folders" maxlength="255" autocomplete="off" aria-describedby="watch-folder-hint">
        <datalist id="watch-folders"></datalist></label>
      <p class="meta" id="watch-folder-hint" data-i18n="folders.watch_hint"></p>
    </div>
    <p class="meta w-hint"></p>
    <p class="form-msg w-msg" role="status"></p>
    <div class="row">
      <button class="btn primary" type="submit" data-i18n="watch.save"></button>
      <button class="btn w-preview" type="button" data-i18n="watch.preview"></button>
      <button class="btn w-cancel" type="button" data-i18n="watch.cancel"></button>
    </div>
    <ul class="w-preview-list"></ul>
  </form>`;

function mount(el) {
  el.innerHTML = TEMPLATE;
  applyI18n(el);
  const list = $(".w-list", el);
  const formCard = $(".w-form-card", el);
  let limits = null;
  let sources = [];
  let editing = null; // id being edited, null = new
  let armed = null;
  let timer = null;
  const open = new Set(); // sources whose history is expanded

  const minutes = (s) => Math.round((s / 60) * 10) / 10;
  const when = (ts) => (ts ? date(ts) : t("watch.never"));

  async function load() {
    try {
      const data = await api("/api/watch");
      if (!isCurrent("watch")) return;
      sources = data.sources;
      limits = data.limits;
      if (!armed) renderList();
    } catch (err) {
      if (isCurrent("watch")) list.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    }
    clearTimeout(timer);
    if (isCurrent("watch")) pollTimeout((h) => (timer = h), load, 5000);
  }

  function status(s) {
    if (!s.enabled) return `<span class="w-state paused">${esc(t("watch.paused"))}</span>`;
    if (s.last_error) return `<span class="w-state bad">${esc(t("watch.state_error"))}</span>`;
    return `<span class="w-state ok">${esc(t("watch.state_ok"))}</span>`;
  }

  function renderList() {
    if (!sources.length) {
      list.innerHTML = `<div class="empty">${esc(t("watch.empty"))}</div>`;
      return;
    }
    list.innerHTML = sources.map((s) => `
      <article class="card w-source" data-id="${esc(s.id)}">
        <div class="w-head">
          <strong>${esc(s.name || s.target)}</strong>
          <span class="voice-tag">${esc(t(`watch.kind_${s.kind}`))}</span>
          ${status(s)}
        </div>
        <div class="meta"><span class="w-target">${esc(s.target)}</span></div>
        <div class="meta">
          <span>${esc(t("watch.last_check", { when: when(s.last_check) }))}</span>
          ${s.enabled && s.next_check ? `<span>${esc(t("watch.next_check", { when: when(s.next_check) }))}</span>` : ""}
          <span>${esc(t("watch.every", { n: minutes(s.interval_s) }))}</span>
          <span>${esc(t(s.mode === "new" ? "watch.mode_new_short" : "watch.mode_last_n_short", { n: s.last_n }))}</span>
          <span>${esc(t("watch.seen", { n: s.seen_count }))}</span>
          ${s.translate_to ? `<span>${esc(t("watch.translate_short", { lang: s.translate_to }))}</span>` : ""}
        </div>
        ${s.last_error ? `<div class="form-msg w-error">${esc(s.last_error)}${s.fails > 1 ? ` (${esc(t("watch.fails", { n: s.fails }))})` : ""}</div>` : ""}
        <div class="w-actions">
          <button class="btn w-check" data-i18n="watch.check_now"></button>
          <button class="btn w-dry" data-i18n="watch.dry_run"></button>
          <button class="btn w-toggle">${esc(t(s.enabled ? "watch.pause" : "watch.resume"))}</button>
          <button class="btn w-edit" data-i18n="watch.edit"></button>
          <button class="btn w-history" data-i18n="watch.history"></button>
          <button class="btn danger w-delete" data-i18n="watch.delete"></button>
        </div>
        <div class="w-detail" ${open.has(s.id) ? "" : "hidden"}></div>
      </article>`).join("");
    applyI18n(list);
    for (const id of open) fillHistory(id);
  }

  async function fillHistory(id) {
    const box = $(`[data-id="${CSS.escape(id)}"] .w-detail`, list);
    if (!box) return;
    try {
      const data = await api(`/api/watch/${encodeURIComponent(id)}/history`);
      if (!isCurrent("watch")) return;
      const translation = (i) => {
        const tr = i.translation;
        if (!tr) return "";
        if (tr.error) return " · " + esc(t("watch.tr_failed", { reason: tr.error }));
        if (tr.status === "unknown") return " · " + esc(t("watch.tr_unknown", { lang: tr.language }));
        const state = ["done", "error", "cancelled"].includes(tr.status) ? tr.status : "pending";
        return " · " + esc(t(`watch.tr_${state}`, { lang: tr.language }));
      };
      const items = data.items.map((i) => `<li><span>${esc(i.title || i.key)}</span>
          <span class="meta">${esc(t(`watch.item.${i.status}`))} · ${esc(date(i.queued))}${i.error ? " · " + esc(i.error) : ""}${translation(i)}</span></li>`).join("");
      const checks = data.history.map((h) => `<li class="${h.ok ? "" : "w-bad"}">
          <span>${esc(date(h.time))}${h.dry_run ? " · " + esc(t("watch.dry_run")) : ""}${h.manual ? " · " + esc(t("watch.manual")) : ""}</span>
          <span class="meta">${h.ok ? esc(t("watch.check_result", { found: h.found, queued: h.queued })) : esc(h.error || "")}</span></li>`).join("");
      box.innerHTML = `<h3>${esc(t("watch.recent_items"))}</h3><ul class="w-log">${items || `<li class="meta">${esc(t("watch.none"))}</li>`}</ul>
        <h3>${esc(t("watch.recent_checks"))}</h3><ul class="w-log">${checks || `<li class="meta">${esc(t("watch.none"))}</li>`}</ul>`;
    } catch (err) {
      box.textContent = err.message;
    }
  }

  // --- add / edit form ---

  async function showForm(source) {
    editing = source?.id || null;
    formCard.hidden = false;
    formCard.innerHTML = FORM;
    applyI18n(formCard);
    const form = $(".w-form", formCard);
    const f = form.elements;
    const info = await getInfo().catch(() => ({ engines: [] }));
    f.engine.innerHTML = `<option value="">${esc(t("watch.engine_default"))}</option>` +
      (info.engines || []).filter((e) => e.available !== false)
        .map((e) => `<option value="${esc(e.name)}">${esc(e.label || e.name)}</option>`).join("");
    f.translate_to.innerHTML = `<option value="">${esc(t("watch.translate_none"))}</option>` +
      Object.entries(limits?.translate_languages || {})
        .map(([code, name]) => `<option value="${esc(code)}">${esc(name)} (${esc(code)})</option>`).join("");
    $(".w-form-title", form).textContent = t(source ? "watch.edit_title" : "watch.add_title");
    const v = source || { kind: "youtube", mode: "new", last_n: 5, max_per_check: 5, interval_s: limits?.default_interval_s?.youtube || 3600 };
    f.kind.value = v.kind;
    f.kind.disabled = !!source;
    f.target.value = v.target || "";
    f.name.value = v.name || "";
    f.language.value = v.language || "";
    f.speakers.value = v.speakers || "";
    f.engine.value = v.engine || "";
    f.mode.value = v.mode;
    f.last_n.value = v.last_n;
    f.max_per_check.value = v.max_per_check;
    f.interval.value = minutes(v.interval_s);
    f.recursive.checked = !!v.recursive;
    f.analyze.checked = !!v.analyze;
    f.translate_to.value = v.translate_to || "";
    f.webhook_url.value = v.webhook_url || "";
    f.folder.value = v.folder || "";
    loadFolders().then((tree) => {
      $("#watch-folders", form).innerHTML = tree.folders.map((x) => `<option value="${esc(x.path)}"></option>`).join("");
    }).catch(() => {});

    const sync = (resetInterval) => {
      const kind = f.kind.value;
      $(".w-target-label", form).textContent = t(`watch.target_${kind}`);
      f.target.placeholder = t(`watch.target_${kind}_placeholder`);
      $(".w-interval-label", form).textContent = t("watch.interval");
      $(".w-recursive", form).hidden = kind !== "folder";
      $(".w-lastn", form).hidden = f.mode.value !== "last_n";
      const min = limits?.min_interval_s?.[kind] || 60;
      f.interval.min = String(Math.max(0.1, min / 60));
      f.interval.step = "any"; // minutes may be fractional (10 s = 0.17); a step of 1 from a fractional min would reject whole minutes
      $(".w-hint", form).textContent = t(`watch.hint_${kind}`, { n: minutes(min) });
      if (resetInterval) f.interval.value = minutes(limits?.default_interval_s?.[kind] || 3600);
    };
    f.kind.addEventListener("change", () => sync(true));
    f.mode.addEventListener("change", () => sync(false));
    sync(false);

    const body = () => ({
      kind: f.kind.value,
      target: f.target.value.trim(),
      name: f.name.value.trim(),
      language: f.language.value || null,
      speakers: f.speakers.value ? Number(f.speakers.value) : null,
      engine: f.engine.value || null,
      mode: f.mode.value,
      last_n: Number(f.last_n.value) || 5,
      max_per_check: Number(f.max_per_check.value) || 5,
      interval_s: Math.round(Number(f.interval.value) * 60),
      recursive: f.recursive.checked,
      analyze: f.analyze.checked,
      translate_to: f.translate_to.value || null,
      webhook_url: f.webhook_url.value.trim() || null,
      folder: f.folder.value.trim(),
    });
    const msg = $(".w-msg", form);

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      msg.textContent = "";
      const data = body();
      if (editing) delete data.kind;
      try {
        await api(editing ? `/api/watch/${encodeURIComponent(editing)}` : "/api/watch", {
          method: editing ? "PATCH" : "POST", body: JSON.stringify(data),
        });
        formCard.hidden = true;
        toast(t("watch.saved"));
        load();
      } catch (err) {
        msg.textContent = err.message;
      }
    });
    $(".w-cancel", form).addEventListener("click", () => (formCard.hidden = true));
    $(".w-preview", form).addEventListener("click", async () => {
      const out = $(".w-preview-list", form);
      msg.textContent = t("watch.checking");
      out.innerHTML = "";
      try {
        const res = await api("/api/watch/preview", { method: "POST", body: JSON.stringify(body()) });
        msg.textContent = t("watch.preview_total", { n: res.total });
        out.innerHTML = res.items.map((i) => `<li>${esc(i.title || i.source)}</li>`).join("");
      } catch (err) {
        msg.textContent = err.message;
      }
    });
    f.target.focus();
  }

  // --- actions ---

  function confirmed(button) {
    if (armed === button) return true;
    if (armed) armed.textContent = armed.dataset.label;
    armed = button;
    button.dataset.label = button.textContent;
    button.textContent = t("watch.confirm");
    setTimeout(() => {
      if (armed === button) {
        button.textContent = button.dataset.label;
        armed = null;
      }
    }, 4000);
    return false;
  }

  $(".w-add", el).addEventListener("click", () => showForm(null));

  list.addEventListener("click", async (e) => {
    const card = e.target.closest("[data-id]");
    const button = e.target.closest("button");
    if (!card || !button) return;
    const id = card.dataset.id;
    const base = `/api/watch/${encodeURIComponent(id)}`;
    const source = sources.find((s) => s.id === id);
    try {
      if (button.matches(".w-check, .w-dry")) {
        const dry = button.matches(".w-dry");
        button.disabled = true;
        const res = await api(`${base}/check${dry ? "?dry_run=true" : ""}`, { method: "POST" });
        const r = res.result;
        if (!r.ok) toast(r.error);
        else if (dry) toast(r.queued_items.length ? t("watch.dry_result", { n: r.queued_items.length, items: r.queued_items.map((i) => i.title).join(", ") })
          : t("watch.dry_none", { found: r.found }));
        else toast(t("watch.check_result", { found: r.found, queued: r.queued }));
        load();
      } else if (button.matches(".w-toggle")) {
        await api(base, { method: "PATCH", body: JSON.stringify({ enabled: !source.enabled }) });
        load();
      } else if (button.matches(".w-edit")) {
        showForm(source);
        formCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else if (button.matches(".w-history")) {
        open.has(id) ? open.delete(id) : open.add(id);
        $(".w-detail", card).hidden = !open.has(id);
        if (open.has(id)) fillHistory(id);
      } else if (button.matches(".w-delete") && confirmed(button)) {
        await api(base, { method: "DELETE" });
        open.delete(id);
        armed = null;
        load();
      }
    } catch (err) {
      toast(err.message);
      button.disabled = false;
    }
  });

  load();
  return () => clearTimeout(timer);
}

registerPage({
  id: "watch",
  path: "/watch",
  nav: { label: "nav.watch", order: 40 },
  title: () => t("watch.title"),
  mount,
});
