// "Voices": the people whose voices are remembered, their samples and where they came from.
// Rename, delete a sample or a person, export, and delete everything.

import { api } from "../core/api.js";
import { applyI18n, t } from "../core/i18n.js";
import { href, isCurrent, registerPage } from "../core/router.js";
import { $, date, duration, esc, pollTimeout, toast } from "../core/util.js";

const TEMPLATE = `
  <header class="detail-head"><h1 data-i18n="voices.title"></h1></header>
  <p class="voice-privacy card" data-i18n="voices.privacy"></p>
  <section class="card vp-box v-archive" aria-labelledby="v-archive-h">
    <h2 id="v-archive-h" data-i18n="voices.archive_title"></h2>
    <p class="muted" data-i18n="voices.archive_hint"></p>
    <div class="vp-bulk">
      <button class="btn v-scan" data-i18n="voices.scan_all"></button>
      <label class="v-compute-label"><input type="checkbox" class="v-compute"> <span data-i18n="voices.compute_missing"></span></label>
      <span class="muted v-job" role="status"></span>
    </div>
  </section>
  <section class="card voice-people v-people"><div class="loading" data-i18n="common.loading"></div></section>
  <div class="voice-actions">
    <a class="btn v-export" download href="/api/voices/export" data-i18n="voices.export"></a>
    <button class="btn danger v-delete-all" data-i18n="voices.delete_all"></button>
  </div>
  <p class="meta v-file"></p>`;

function mount(el) {
  el.innerHTML = TEMPLATE;
  applyI18n(el);
  const list = $(".v-people", el);
  let armed = null; // button waiting for a second click (instead of window.confirm)
  let jobTimer = null;

  load();

  async function load() {
    let data;
    try {
      data = await api("/api/voices");
    } catch (err) {
      if (isCurrent("voices")) list.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
      return;
    }
    if (!isCurrent("voices")) return;
    showJob(data.scan_job);
    $(".v-file", el).textContent = t("voices.file", { path: data.file });
    if (!data.enabled) $(".v-file", el).textContent += " · " + t("voices.disabled");
    if (!data.people.length) {
      $(".v-job", el).textContent = t("voices.scan_empty"); // rather than "waiting": there is nothing to recognise
      list.innerHTML = `<div class="empty v-empty">
        <h2>${esc(t("voices.empty_title"))}</h2>
        <p>${esc(t("voices.empty_intro"))}</p>
        <ol class="voice-steps">
          ${["open", "name", "remember", "scan"].map((k) => `<li>${esc(t(`voices.step_${k}`))}</li>`).join("")}
        </ol></div>`;
      return;
    }
    list.innerHTML = data.people.map((p) => `
      <article class="voice-person" data-person="${esc(p.id)}">
        <div class="voice-person-head">
          <input type="text" class="v-name" maxlength="60" value="${esc(p.name)}" aria-label="${esc(t("voices.name"))}">
          <span class="meta">${esc(t("voices.samples", { n: p.samples.length }))} · ${esc(duration(p.speech_s))} ·
            ${esc(t("voices.appearances_n", { n: p.appearances || 0 }))}</span>
          <a class="btn" href="${href(`/voices/${encodeURIComponent(p.id)}`)}" data-i18n="voices.open_person"></a>
          <button class="btn danger v-del-person" data-i18n="voices.delete_person"></button>
        </div>
        <ul class="voice-samples">${p.samples.map((s) => `
          <li data-sample="${esc(s.id)}">
            ${s.source?.transcript
              ? `<a href="${href(`/t/${encodeURIComponent(s.source.transcript)}`)}">${esc(s.source.title || s.source.transcript)}</a>
                 <span class="meta">${esc(s.source.label || "")}</span>`
              : `<span class="meta">${esc(t("voices.unknown_source"))}</span>`}
            <span class="meta">${esc(duration(s.speech_s || 0))}${s.added ? " · " + esc(date(Date.parse(s.added) / 1000)) : ""}</span>
            ${s.outdated ? `<span class="voice-tag" title="${esc(t("voices.outdated_title"))}">${esc(t("voices.outdated"))}</span>` : ""}
            <button class="link-btn v-del-sample" data-i18n="voices.delete_sample"></button>
          </li>`).join("")}
        </ul>
      </article>`).join("");
    applyI18n(list);
  }

  function showJob(job) {
    clearTimeout(jobTimer);
    $(".v-scan", el).disabled = !!job;
    $(".v-job", el).textContent = !job ? "" : job.status === "running"
      ? t("voices.scan_running", { pct: job.percent != null ? Math.round(job.percent) + " %" : "" })
      : t("voices.scan_waiting");
    if (job) pollTimeout((h) => (jobTimer = h), load, 2000);
  }

  $(".v-scan", el).addEventListener("click", async () => {
    try {
      const res = await api("/api/voices/scan", { method: "POST", body: JSON.stringify({ compute_missing: $(".v-compute", el).checked }) });
      if (res.empty) {
        // nothing to recognise: say why instead of queueing a job that reports zeros
        toast(t("voices.scan_empty"));
        return load();
      }
      toast(t("voices.scan_queued"));
    } catch (err) {
      toast(err.message);
    }
    load();
  });

  // a destructive button needs a second click within 4 s
  function confirmed(button) {
    if (armed === button) return true;
    if (armed) armed.textContent = armed.dataset.label;
    armed = button;
    button.dataset.label = button.textContent;
    button.textContent = t("voices.confirm");
    setTimeout(() => {
      if (armed === button) {
        button.textContent = button.dataset.label;
        armed = null;
      }
    }, 4000);
    return false;
  }

  async function call(path, options, okText) {
    try {
      await api(path, options);
      if (okText) toast(okText);
    } catch (err) {
      toast(err.message);
    }
    armed = null;
    load();
  }

  list.addEventListener("click", (e) => {
    const person = e.target.closest("[data-person]")?.dataset.person;
    if (!person) return;
    const base = `/api/voices/people/${encodeURIComponent(person)}`;
    if (e.target.closest(".v-del-person") && confirmed(e.target)) {
      call(base, { method: "DELETE" }, t("voices.deleted"));
    } else if (e.target.closest(".v-del-sample") && confirmed(e.target)) {
      const sample = e.target.closest("[data-sample]").dataset.sample;
      call(`${base}/samples/${encodeURIComponent(sample)}`, { method: "DELETE" }, t("voices.deleted"));
    }
  });

  list.addEventListener("change", (e) => {
    if (!e.target.matches(".v-name")) return;
    const person = e.target.closest("[data-person]").dataset.person;
    const name = e.target.value.trim();
    if (name) call(`/api/voices/people/${encodeURIComponent(person)}`, { method: "PATCH", body: JSON.stringify({ name }) },
      t("voices.renamed"));
  });
  list.addEventListener("keydown", (e) => e.key === "Enter" && e.target.matches(".v-name") && e.target.blur());

  $(".v-delete-all", el).addEventListener("click", (e) => {
    if (confirmed(e.target)) call("/api/voices", { method: "DELETE" }, t("voices.deleted_all"));
  });
}

registerPage({
  id: "voices",
  path: "/voices",
  nav: { label: "nav.voices", order: 30 },
  title: () => t("voices.title"),
  mount,
});
