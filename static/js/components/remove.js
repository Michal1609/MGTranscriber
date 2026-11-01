// Quick delete of transcripts: an in-page dialog that names the transcripts (no ID typing; the API's `confirm`
// is the transcript's own id), then DELETE /api/transcripts/{id}?confirm={id} for each. The Privacy panel on the
// detail page keeps its own, typed confirmation (components/privacy.js).

import { api } from "../core/api.js";
import { t } from "../core/i18n.js";
import { esc, toast } from "../core/util.js";
import "../i18n/remove.js";
import { modal } from "./folders.js";

const LISTED = 5; // titles named in the dialog

// items: [{ id, title }]. Resolves with { deleted: [ids], failed: [{ id, error }] }, or null when cancelled.
export async function deleteTranscripts(items) {
  if (!items.length) return null;
  const shown = items.slice(0, LISTED).map((i) => `<li>${esc(i.title || i.id)}</li>`).join("");
  const more = items.length > LISTED ? `<li class="muted">${esc(t("remove.more", { n: items.length - LISTED }))}</li>` : "";
  const ok = await modal({
    title: t("remove.title", { n: items.length }),
    confirm: t("remove.confirm"),
    danger: true,
    focusCancel: true,
    body: `<ul class="rm-list">${shown}${more}</ul><p>${esc(t("remove.text"))}</p>`,
    collect: () => true,
  });
  if (!ok) return null;
  const deleted = [];
  const failed = [];
  for (const { id } of items) {
    try {
      await api(`/api/transcripts/${encodeURIComponent(id)}?confirm=${encodeURIComponent(id)}`, { method: "DELETE" });
      deleted.push(id);
    } catch (err) {
      failed.push({ id, error: err.message });
    }
  }
  if (!failed.length) toast(t("remove.done", { n: deleted.length }));
  else if (!deleted.length) toast(t("remove.failed", { error: failed[0].error }));
  else toast(t("remove.partial", { ok: deleted.length, failed: failed.length, error: failed[0].error }));
  return { deleted, failed };
}
