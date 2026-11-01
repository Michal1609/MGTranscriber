// Sensitive-data mode in the UI: the "offline mode" badge in the top bar, and on the detail page the
// redacted exports, chain of custody with the audit log, and deleting a transcript for good.
// See docs/PRIVACY.md.

import { api, getInfo } from "../core/api.js";
import { applyI18n, onLangChange, t } from "../core/i18n.js";
import { esc, toast } from "../core/util.js";
import { loadStyles } from "./editor.js";

/* --- top bar badge (imported once by main.js) --- */

async function showBadge() {
  loadStyles();
  let info;
  try {
    info = await getInfo();
  } catch {
    return;
  }
  const end = document.querySelector(".topbar-end");
  let badge = document.querySelector(".offline-badge");
  if (!info.privacy?.offline) return badge?.remove();
  if (!badge) {
    badge = document.createElement("span");
    badge.className = "offline-badge";
    end?.prepend(badge);
  }
  badge.textContent = t("privacy.offline_badge");
  badge.title = t("privacy.offline_title");
}
showBadge();
onLangChange?.(showBadge);

/* --- detail page panel --- */

const TEMPLATE = `
  <summary data-i18n="privacy.panel"></summary>
  <div class="pv-body">
    <section>
      <h2 data-i18n="privacy.redacted_title"></h2>
      <p class="muted pv-redacted-note"></p>
      <div class="actions pv-redacted"></div>
    </section>
    <section>
      <h2 data-i18n="privacy.custody_title"></h2>
      <p class="muted" data-i18n="privacy.custody_hint"></p>
      <div class="pv-custody"><div class="loading" data-i18n="detail.loading"></div></div>
    </section>
    <section>
      <h2 data-i18n="privacy.delete_title"></h2>
      <p class="muted" data-i18n="privacy.delete_hint"></p>
      <label class="field pv-confirm-field" hidden>
        <span data-i18n="privacy.delete_type"></span>
        <input class="pv-confirm" autocomplete="off" spellcheck="false">
      </label>
      <div class="actions">
        <button class="btn danger pv-delete" data-i18n="privacy.delete"></button>
        <button class="btn pv-cancel" data-i18n="privacy.cancel" hidden></button>
      </div>
    </section>
  </div>`;

export function mountPrivacy({ id, box, exporters, hasAudio, onDeleted }) {
  box.innerHTML = TEMPLATE;
  applyI18n(box);
  const $ = (cls) => box.querySelector(`.${cls}`);
  const base = `/api/transcripts/${encodeURIComponent(id)}`;
  let loaded = false;

  $("pv-redacted-note").textContent = t("privacy.redacted_hint");
  $("pv-redacted").innerHTML = exporters
    .map((e) => `<a class="btn" download href="${base}/export/${esc(e.name)}?redact=true">${esc(e.label)}</a>`)
    .join(" ") + (hasAudio ? ` <a class="btn" download href="${base}/redacted/audio">${esc(t("privacy.redacted_audio"))}</a>` : "");

  box.addEventListener("toggle", () => {
    if (box.open && !loaded) {
      loaded = true;
      loadCustody();
    }
  });

  async function loadCustody(method = "GET") {
    const el = $("pv-custody");
    try {
      renderCustody(await api(`${base}/custody`, { method }));
    } catch (err) {
      el.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    }
  }

  function renderCustody(data) {
    const stored = data.stored || {};
    const rows = [];
    if (stored.source) rows.push([t("privacy.source_file", { name: stored.source.file }), stored.source.sha256, data.check.source]);
    for (const [name, digest] of Object.entries(stored.outputs || {})) rows.push([name, digest, data.check[name]]);
    const table = rows.length
      ? `<table class="pv-hashes"><tbody>${rows
          .map(([name, digest, check]) => `<tr><th scope="row">${esc(name)}</th><td><code>${esc(digest)}</code></td>
            <td><span class="pill pv-${esc(check || "none")}">${esc(t(`privacy.check_${check || "none"}`))}</span></td></tr>`)
          .join("")}</tbody></table>`
      : `<p class="muted">${esc(t("privacy.no_hashes"))}</p>`;
    const audit = data.audit.length
      ? `<ol class="pv-audit">${data.audit
          .slice()
          .reverse()
          .map((e) => `<li><time>${esc(e.time ? new Date(e.time).toLocaleString() : "?")}</time> <b>${esc(t(`privacy.event_${e.event}`, {}) || e.event)}</b>${auditDetail(e)}</li>`)
          .join("")}</ol>`
      : `<p class="muted">${esc(t("privacy.no_audit"))}</p>`;
    $("pv-custody").innerHTML = `${table}
      <p class="pv-chain ${data.audit_intact ? "ok" : "bad"}">${esc(t(data.audit_intact ? "privacy.chain_ok" : "privacy.chain_broken"))}</p>
      <div class="actions"><button class="btn pv-hash">${esc(t("privacy.hash_now"))}</button></div>
      <h3>${esc(t("privacy.audit_title"))}</h3>${audit}`;
  }

  function auditDetail(e) {
    const parts = [];
    if (e.format) parts.push(e.format.toUpperCase());
    if (e.redacted) parts.push(t("privacy.redacted_short"));
    if (e.segments_changed) parts.push(t("privacy.segments_changed", { n: e.segments_changed }));
    if (e.redactions_changed) parts.push(t("privacy.redactions_changed"));
    if (e.engine) parts.push(e.engine);
    return parts.length ? ` <span class="muted">· ${esc(parts.join(" · "))}</span>` : "";
  }

  box.addEventListener("click", async (e) => {
    if (e.target.closest(".pv-hash")) {
      await loadCustody("POST");
      toast(t("privacy.hashed"));
    } else if (e.target.closest(".pv-delete")) {
      const field = $("pv-confirm-field");
      if (field.hidden) {
        field.hidden = false;
        $("pv-cancel").hidden = false;
        $("pv-confirm").placeholder = id;
        $("pv-confirm").focus();
        return;
      }
      if ($("pv-confirm").value.trim() !== id) {
        toast(t("privacy.delete_mismatch"));
        $("pv-confirm").focus();
        return;
      }
      try {
        await api(`${base}?confirm=${encodeURIComponent(id)}`, { method: "DELETE" });
        toast(t("privacy.deleted"));
        onDeleted?.();
      } catch (err) {
        toast(err.message);
      }
    } else if (e.target.closest(".pv-cancel")) {
      $("pv-confirm-field").hidden = true;
      $("pv-cancel").hidden = true;
      $("pv-confirm").value = "";
    }
  });
}
