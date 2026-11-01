// Voice memory on the transcript detail page: per speaker "Remember voice", suggestions
// (accept / reject) with their score, and computing voiceprints for older transcripts.
// mountVoices() decorates the speaker rows ([data-speaker]) inside `box` and returns a cleanup.

import { api } from "../core/api.js";
import { locale, t } from "../core/i18n.js";
import { href, isCurrent } from "../core/router.js";
import { esc, pollTimeout, toast } from "../core/util.js";

const score = (s) => (s == null ? "" : s.toLocaleString(locale(), { minimumFractionDigits: 2, maximumFractionDigits: 2 }));

export function mountVoices({ id, box, getName, onNames }) {
  const base = `/api/voices/transcripts/${encodeURIComponent(id)}`;
  let state = null;
  let pollTimer = null;
  let alive = true;
  const footer = document.createElement("div");
  footer.className = "voice-footer";
  box.after(footer);
  // result of the last action for screen readers; lives outside render(), which rebuilds the rows
  const live = document.createElement("div");
  live.className = "sr-only";
  live.setAttribute("role", "status");
  live.setAttribute("aria-live", "polite");
  footer.after(live);
  let formLabel = null; // the speaker whose inline "name this voice" form is open
  let focusForm = false;
  const notice = {}; // label -> error text shown next to the speaker until the next attempt
  const offered = new Set(); // labels renamed on this page that may get the "remember?" prompt
  let known = null; // names at the last render, to see which one the user just changed

  const dismissKey = (label) => `${id}|${label}`;
  function dismissed(label) {
    try {
      return !!JSON.parse(localStorage.getItem("voices.not_now") || "{}")[dismissKey(label)];
    } catch {
      return false;
    }
  }
  function dismiss(label) {
    try {
      const all = JSON.parse(localStorage.getItem("voices.not_now") || "{}");
      all[dismissKey(label)] = 1;
      localStorage.setItem("voices.not_now", JSON.stringify(all));
    } catch {
      // private mode: the prompt just returns on the next visit
    }
  }
  function say(text) {
    live.textContent = "";
    setTimeout(() => (live.textContent = text), 30);
  }

  // detail.js saved the speaker names: a changed name may get the "remember this voice?" prompt
  const onSaved = () => {
    if (!state?.available) return;
    for (const label of Object.keys(state.speakers || {})) {
      const name = getName(label);
      if (name && name !== (known?.[label] || "")) offered.add(label);
      else if (!name) offered.delete(label);
    }
    render();
  };
  box.addEventListener("names-saved", onSaved);

  load();

  async function load() {
    const hadJob = !!state?.job;
    try {
      state = await api(base);
    } catch {
      state = null; // voice memory is optional; the page works without it
    }
    if (!alive || !isCurrent("detail", { id })) return;
    if (hadJob && state && !state.job) onNames(state.names); // the job may have named recognised voices
    render();
    clearTimeout(pollTimer);
    if (state?.job) pollTimeout((h) => (pollTimer = h), load, 3000);
  }

  async function act(path, body, okText) {
    try {
      const res = await api(`${base}/${path}`, { method: "POST", body: JSON.stringify(body || {}) });
      if (res.names) onNames(res.names);
      if (body?.label) delete notice[body.label];
      if (okText) {
        toast(okText);
        say(okText);
      }
      await load();
      return res;
    } catch (err) {
      if (err.code === "voice_mismatch" && body?.name &&
          (await confirmInline(t("voices.mismatch_confirm", { name: body.name })))) {
        return act(path, { ...body, force: true }, okText);
      }
      if (body?.label) notice[body.label] = err.message;
      toast(err.message);
      say(err.message);
      render();
    }
  }

  // two-step confirmation in the footer instead of window.confirm()
  function confirmInline(text) {
    return new Promise((resolve) => {
      footer.innerHTML = `<span>${esc(text)}</span> <button class="btn danger v-yes"></button> <button class="btn v-no"></button>`;
      footer.querySelector(".v-yes").textContent = t("voices.yes");
      footer.querySelector(".v-no").textContent = t("voices.no");
      footer.querySelector(".v-yes").onclick = () => resolve(true);
      footer.querySelector(".v-no").onclick = () => (resolve(false), render());
    });
  }

  function render() {
    for (const old of box.querySelectorAll(".voice")) old.remove();
    const hadFocus = focusForm;
    focusForm = false;
    footer.innerHTML = "";
    if (!state || !state.enabled) return;

    if (!state.available) {
      if (state.job) {
        footer.innerHTML = `<span class="muted">${esc(t("voices.computing"))}</span>`;
      } else if (state.can_compute) {
        footer.innerHTML = `<span class="muted">${esc(t("voices.missing"))}</span> <button class="btn v-compute"></button>`;
        footer.querySelector(".v-compute").textContent = t("voices.compute");
        footer.querySelector(".v-compute").onclick = async (e) => {
          e.target.disabled = true;
          try {
            await api(`${base}/compute`, { method: "POST" });
            toast(t("voices.compute_queued"));
          } catch (err) {
            toast(err.message);
          }
          load();
        };
      }
      return;
    }

    for (const row of box.querySelectorAll("[data-speaker]")) {
      const label = row.dataset.speaker;
      const info = state.speakers[label];
      if (!info) continue;
      const m = info.match || {};
      const el = document.createElement("div");
      el.className = "voice";
      const name = getName(label);
      const matched = m.status === "auto" && name === m.name;
      if (m.confirmed || info.enrolled) {
        const who = m.name || name;
        el.innerHTML = `<span class="voice-tag ok">${esc(who ? t("voices.saved_as", { name: who }) : t("voices.saved"))}</span>`;
      } else if (matched) {
        el.innerHTML = `<span class="voice-tag ok" title="${esc(t("voices.score_title"))}">${esc(t("voices.recognised", { score: score(m.score) }))}</span>
          <button class="link-btn v-reject">${esc(t("voices.not_them"))}</button>`;
      } else if (m.status === "auto" || m.status === "suggest") {
        el.innerHTML = `<span class="voice-tag" title="${esc(t("voices.score_title"))}">${esc(t("voices.maybe", { name: m.name, score: score(m.score) }))}</span>
          <button class="link-btn v-accept">${esc(t("voices.accept"))}</button>
          <button class="link-btn v-reject">${esc(t("voices.reject"))}</button>`;
      }
      if (!m.confirmed && !info.enrolled && !matched) {
        const enough = (info.speech_s || 0) >= 20;
        if (!enough) {
          el.insertAdjacentHTML("beforeend",
            `<span class="muted" title="${esc(t("voices.too_short_title"))}">${esc(t("voices.too_short"))}</span>`);
        } else if (formLabel === label && !name) {
          el.insertAdjacentHTML("beforeend", `<form class="voice-form">
            <label class="sr-only" for="vf-${esc(label)}">${esc(t("voices.form_label"))}</label>
            <input type="text" id="vf-${esc(label)}" class="v-form-name" maxlength="60" placeholder="${esc(t("voices.form_placeholder"))}">
            <button type="submit" class="btn">${esc(t("voices.save_voice"))}</button>
            <button type="button" class="link-btn v-form-cancel">${esc(t("voices.cancel"))}</button></form>`);
        } else if (offered.has(label) && name && !dismissed(label)) {
          el.insertAdjacentHTML("beforeend", `<span class="voice-offer" role="group" aria-label="${esc(t("voices.remember"))}">
            <span>${esc(t("voices.offer", { name }))}</span>
            <button class="btn v-enroll">${esc(t("voices.offer_yes"))}</button>
            <button class="link-btn v-not-now">${esc(t("voices.offer_no"))}</button></span>`);
        } else {
          el.insertAdjacentHTML("beforeend", `<button class="link-btn v-enroll">${esc(t("voices.remember"))}</button>`);
        }
        if (notice[label]) {
          el.insertAdjacentHTML("beforeend", `<span class="voice-error" role="alert">${esc(notice[label])}</span>`);
        }
      }
      el.querySelector(".v-accept")?.addEventListener("click", () =>
        act("accept", { label }, t("voices.accepted", { name: m.name })));
      el.querySelector(".v-reject")?.addEventListener("click", () => act("reject", { label }));
      // one step: a named speaker is saved straight away; an unnamed one gets a small form to type the name
      el.querySelector(".v-enroll")?.addEventListener("click", () => {
        const value = getName(label);
        if (!value) {
          formLabel = label;
          focusForm = true;
          render();
          return;
        }
        offered.delete(label);
        act("enroll", { label, name: value }, t("voices.remembered", { name: value }));
      });
      el.querySelector(".v-not-now")?.addEventListener("click", () => {
        dismiss(label);
        offered.delete(label);
        render();
      });
      const form = el.querySelector(".voice-form");
      if (form) {
        const field = form.querySelector("input");
        const close = () => {
          formLabel = null;
          render();
          row.querySelector(".v-enroll")?.focus();
        };
        form.addEventListener("submit", (e) => {
          e.preventDefault();
          const value = field.value.trim();
          if (!value) {
            notice[label] = t("errors.voice_name_required");
            say(notice[label]);
            focusForm = true;
            render();
            return;
          }
          formLabel = null;
          act("enroll", { label, name: value }, t("voices.remembered", { name: value }));
        });
        field.addEventListener("keydown", (e) => e.key === "Escape" && close());
        form.querySelector(".v-form-cancel").addEventListener("click", close);
        if (hadFocus) queueMicrotask(() => field.focus());
      }
      if (el.childElementCount) row.append(el);
    }
    known = Object.fromEntries(Object.keys(state.speakers).map((l) => [l, getName(l)]));
    footer.innerHTML = `<small class="muted">${esc(t("voices.privacy_short"))} <a href="${href("/voices")}">${esc(t("voices.manage"))}</a></small>`;
  }

  return () => {
    alive = false;
    clearTimeout(pollTimer);
    box.removeEventListener("names-saved", onSaved);
    footer.remove();
    live.remove();
  };
}
