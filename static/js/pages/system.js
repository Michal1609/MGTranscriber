// "Settings": system status (is everything needed there, and if not, what to do), cloud keys (set / missing only)
// and the preferences that may be saved to settings.json. Backend: /api/system and /api/settings.

import { api, getInfo } from "../core/api.js";
import { applyI18n, has, t } from "../core/i18n.js";
import { isCurrent, registerPage } from "../core/router.js";
import { $, esc, toast } from "../core/util.js";
import { checkMessage, errorText, hfActions, loadSystem, keyForm, rich, statusIcon, tokenForm } from "../components/system.js";

const GROUPS = ["hardware", "tools", "models", "services", "app"];

const TEMPLATE = `
  <header class="detail-head sys-head">
    <div>
      <h1 data-i18n="system.title"></h1>
      <p class="meta"><span class="sys-version"></span></p>
    </div>
    <button class="btn" id="recheck" type="button" data-i18n="system.recheck"></button>
  </header>
  <p class="sys-intro" data-i18n="system.intro"></p>
  <div id="sys-summary" class="sys-summary card" role="status"></div>
  <div id="sys-checks"></div>
  <section aria-labelledby="eng-title">
    <h2 class="section-title" id="eng-title" data-i18n="system.engines.title"></h2>
    <div class="card sys-list" id="sys-engines"></div>
  </section>
  <section aria-labelledby="keys-title">
    <h2 class="section-title" id="keys-title" data-i18n="system.keys.title"></h2>
    <div class="card sys-list" id="sys-keys"></div>
  </section>
  <section aria-labelledby="prefs-title">
    <h2 class="section-title" id="prefs-title" data-i18n="system.settings.title"></h2>
    <form class="card prefs" id="prefs" autocomplete="off"></form>
  </section>`;

function mount(el) {
  el.innerHTML = TEMPLATE;
  applyI18n(el);
  let stopPull = null;

  const recheck = $("#recheck", el);
  recheck.addEventListener("click", () => load(true));
  load(false);
  loadPrefs();

  async function load(refresh) {
    recheck.disabled = true;
    recheck.textContent = t("system.checking");
    let system;
    try {
      system = await loadSystem(refresh);
    } catch (err) {
      if (isCurrent("system")) $("#sys-summary", el).innerHTML = `<span class="st-text error">${esc(t("system.load_failed", { error: err.message }))}</span>`;
      return;
    } finally {
      recheck.disabled = false;
      recheck.textContent = t("system.recheck");
    }
    if (isCurrent("system")) render(system);
  }

  function render(system) {
    stopPull?.();
    $(".sys-version", el).textContent = t("system.version", { version: system.version, python: system.python });
    const counts = { warn: 0, error: 0 };
    system.checks.forEach((c) => c.status in counts && counts[c.status]++);
    const worst = counts.error ? "error" : counts.warn ? "warn" : "ok";
    $("#sys-summary", el).innerHTML = `${statusIcon(worst)}<strong>${esc(worst === "ok" ? t("system.summary.ok") : t(`system.summary.${worst}`, { n: counts[worst] }))}</strong>`;

    $("#sys-checks", el).innerHTML = GROUPS.map((group) => {
      const rows = system.checks.filter((c) => c.group === group);
      return rows.length
        ? `<section aria-labelledby="g-${group}"><h2 class="section-title" id="g-${group}">${esc(t(`system.group.${group}`))}</h2>
             <ul class="card sys-list">${rows.map((c) => `<li class="check" data-check="${esc(c.id)}">${checkRow(c)}</li>`).join("")}</ul></section>`
        : "";
    }).join("");

    for (const check of system.checks) {
      const li = $(`[data-check="${check.id}"] .check-actions`, el);
      if (!li) continue;
      if (check.id === "hf" && ["no_token", "invalid_token"].includes(check.code)) li.append(tokenForm(system, (fresh) => isCurrent("system") && render(fresh)));
      if (check.id === "ollama" && check.code === "model_missing") li.append(pullControl(check));
      if (check.id === "gpu" && ["ok", "warn"].includes(check.status) && check.code !== "no_nvidia") li.append(gpuTest());
    }
    applyI18n($("#sys-checks", el));

    $("#sys-engines", el).innerHTML = system.engines.map((e) => {
      const why = !e.missing ? t("system.engines.ready") : e.missing.code === "offline" ? t("system.engines.offline")
        : t(`engines.missing.${e.missing.code}`, { detail: e.missing.detail });
      return `<div class="check">${statusIcon(!e.missing ? "ok" : ["missing_key", "offline"].includes(e.missing.code) ? "na" : "warn")}
        <div class="check-body"><div class="check-title">${esc(e.label)}${e.cloud ? ` <span class="pill">${esc(t("system.engines.cloud"))}</span>` : ""}</div>
        <p class="check-msg">${esc(why)}</p></div></div>`;
    }).join("");

    $("#sys-keys", el).innerHTML = `<p class="meta pad">${esc(t("system.keys.intro"))}</p>` + system.keys.map((k) => `
      <div class="check">${statusIcon(k.set ? "ok" : "na")}
        <div class="check-body"><div class="check-title"><code>${esc(k.env)}</code> · ${esc(t(k.set ? "system.keys.set" : "system.keys.missing"))}</div>
        <p class="check-msg">${esc(t("system.keys.used_by", { list: k.used_by.join(", ") }))}</p>
        <div class="check-actions"></div></div></div>`).join("")
      + `<p class="meta pad">${esc(t("system.keys.privacy"))}</p>`;
    $("#sys-keys", el).querySelectorAll(".check-actions").forEach((slot, i) =>
      slot.append(keyForm(system.keys[i], system, (fresh) => isCurrent("system") && render(fresh))));
  }

  function checkRow(c) {
    const extra = c.id === "hf" ? hfActions(c) : "";
    return `${statusIcon(c.status)}
      <div class="check-body">
        <div class="check-title">${esc(has(`system.item.${c.id}`) ? t(`system.item.${c.id}`) : c.id)}</div>
        <p class="check-msg">${rich(checkMessage(c))}</p>
        <div class="check-actions">${extra}</div>
      </div>`;
  }

  // Ollama: download the missing model with a progress bar (polls the server's download state).
  function pullControl(check) {
    const box = document.createElement("div");
    box.className = "pull";
    box.innerHTML = `<button class="btn primary" type="button">${esc(t("system.ollama.pull"))}</button>
      <progress max="100" value="0" hidden aria-label="${esc(t("system.ollama.pull"))}"></progress>
      <span class="meta pull-text" role="status"></span>`;
    const button = $("button", box);
    const bar = $("progress", box);
    const text = $(".pull-text", box);
    let timer = null;
    stopPull = () => clearTimeout(timer);

    async function poll() {
      let state;
      try {
        state = await api("/api/system/ollama/pull");
      } catch {
        timer = setTimeout(poll, 2000);
        return;
      }
      if (!isCurrent("system")) return;
      bar.hidden = state.status === "idle";
      bar.value = state.percent;
      if (state.status === "running") {
        button.disabled = true;
        text.textContent = t("system.ollama.pulling", { model: state.model, percent: Math.round(state.percent) });
        timer = setTimeout(poll, 1000);
      } else if (state.status === "done") {
        toast(t("system.ollama.done"));
        load(true);
      } else if (state.status === "error") {
        button.disabled = false;
        text.textContent = t("system.ollama.failed", { error: state.error });
      }
    }
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await api("/api/system/ollama/pull", { method: "POST", body: JSON.stringify({ model: check.params.model }) });
      } catch (err) {
        if (err.code !== "pull_running") {
          text.textContent = t("system.ollama.failed", { error: errorText(err) });
          button.disabled = false;
          return;
        }
      }
      poll();
    });
    poll();
    return box;
  }

  function gpuTest() {
    const box = document.createElement("div");
    box.className = "pull";
    box.innerHTML = `<button class="btn" type="button">${esc(t("system.gpu_test.button"))}</button><span class="meta" role="status"></span>`;
    const button = $("button", box);
    const out = $("span", box);
    button.addEventListener("click", async () => {
      button.disabled = true;
      out.textContent = t("system.gpu_test.running");
      try {
        const res = await api("/api/system/gpu-test", { method: "POST" });
        out.innerHTML = `${statusIcon(res.status)} ${rich(checkMessage(res))}`;
      } catch (err) {
        out.textContent = errorText(err);
      }
      button.disabled = false;
    });
    return box;
  }

  // --- preferences ---------------------------------------------------------------------------------

  async function loadPrefs() {
    const form = $("#prefs", el);
    let data, info;
    try {
      [data, info] = await Promise.all([api("/api/settings"), getInfo()]);
    } catch (err) {
      form.innerHTML = `<p class="form-msg">${esc(err.message)}</p>`;
      return;
    }
    if (!isCurrent("system")) return;
    const engineLabels = Object.fromEntries(info.engines.map((e) => [e.name, e.label]));
    const byGroup = {};
    data.settings.forEach((s) => (byGroup[s.group] ||= []).push(s));

    form.innerHTML = Object.entries(byGroup).map(([group, items]) => `
      <fieldset><legend>${esc(t(`system.settings.group.${group}`))}</legend>
        ${items.map((s) => field(s, engineLabels, data.suggestions)).join("")}
      </fieldset>`).join("") + `
      <div class="prefs-actions"><button class="btn primary" type="submit">${esc(t("system.settings.save"))}</button>
      <span class="meta" id="prefs-msg" role="status"></span></div>
      <p class="meta">${esc(t("system.settings.intro"))}</p>`;

    const initial = Object.fromEntries(data.settings.map((s) => [s.key, s.value]));
    const modelOptions = (key) => {
      const sel = key === "model" ? form.elements.engine?.value : form.elements.llm_provider?.value;
      const map = key === "model" ? data.suggestions.by_engine : data.suggestions.by_provider;
      return map[sel] || [];
    };
    // the model suggestions follow the chosen engine / provider
    for (const [key, source] of [["model", "engine"], ["llm_model", "llm_provider"]]) {
      form.elements[source]?.addEventListener("change", () => {
        $(`#dl-${key}`, form).innerHTML = modelOptions(key).map((m) => `<option value="${esc(m)}"></option>`).join("");
      });
    }
    form.addEventListener("click", (e) => {
      const reset = e.target.closest("[data-reset]");
      if (!reset) return;
      const s = data.settings.find((x) => x.key === reset.dataset.reset);
      const input = form.elements[s.key];
      if (s.kind === "bool") input.checked = !!s.default;
      else input.value = s.default ?? "";
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    // Offline mode protects sensitive recordings, so it must not wait for the Save button where a
    // forgotten click leaves the protection off. It saves at once; the page reloads because the badge
    // and the engine lists were read when the page opened.
    form.elements.offline_mode?.addEventListener("change", async (e) => {
      const input = e.target;
      const msg = $("#prefs-msg", form);
      input.disabled = true;
      try {
        await api("/api/settings", { method: "PUT", body: JSON.stringify({ values: { offline_mode: input.checked } }) });
        toast(t("system.settings.saved"));
        setTimeout(() => location.reload(), 500);
      } catch (err) {
        input.checked = !input.checked;
        input.disabled = false;
        msg.textContent = errorText(err);
      }
    });
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const values = {};
      for (const s of data.settings) {
        if (s.locked || s.key === "offline_mode") continue; // saved on its own, see above
        const input = form.elements[s.key];
        const now = s.kind === "bool" ? input.checked : input.value.trim();
        if (now !== initial[s.key]) values[s.key] = now;
      }
      const msg = $("#prefs-msg", form);
      if (!Object.keys(values).length) {
        msg.textContent = t("system.settings.unchanged");
        return;
      }
      try {
        await api("/api/settings", { method: "PUT", body: JSON.stringify({ values }) });
        toast(t("system.settings.saved"));
        loadPrefs();
        load(false);
      } catch (err) {
        msg.textContent = errorText(err);
      }
    });
  }

  function field(s, engineLabels, suggestions) {
    const id = `set-${esc(s.key)}`;
    const label = t(`system.settings.key.${s.key}`);
    const help = `<small id="${id}-help" class="meta">${esc(t(`system.settings.help.${s.key}`))}${
      s.locked ? ` <strong>${esc(t("system.settings.locked", { name: `MGTRANSCRIBER_${s.key.toUpperCase()}` }))}</strong>` : ""}</small>`;
    const disabled = s.locked ? " disabled" : "";
    let control;
    if (s.kind === "bool") {
      return `<div class="pref pref-bool"><label class="check-label" for="${id}">
        <input type="checkbox" id="${id}" name="${esc(s.key)}" ${s.value ? "checked" : ""}${disabled} aria-describedby="${id}-help"> <span>${esc(label)}</span></label>${help}</div>`;
    }
    if (s.kind === "choice") {
      const names = s.choices.includes(s.value) ? s.choices : [...s.choices, s.value];
      const text = (c) => s.key === "engine" ? engineLabels[c] || c : s.key === "llm_provider" ? t(`system.settings.provider.${c}`)
        : has(`lang.${c}`) ? t(`lang.${c}`) : c.toUpperCase();
      control = `<select class="select" id="${id}" name="${esc(s.key)}"${disabled} aria-describedby="${id}-help">${
        names.map((c) => `<option value="${esc(c)}"${c === s.value ? " selected" : ""}>${esc(text(c))}</option>`).join("")}</select>`;
    } else {
      const list = s.key === "model" ? suggestions.model : suggestions.llm_model;
      control = `<input type="text" id="${id}" name="${esc(s.key)}" value="${esc(s.value)}" list="dl-${esc(s.key)}" maxlength="100"${disabled}
        aria-describedby="${id}-help" ${s.key === "llm_model" ? `placeholder="${esc(t("system.settings.model_default"))}"` : ""}>
        <datalist id="dl-${esc(s.key)}">${list.map((m) => `<option value="${esc(m)}"></option>`).join("")}</datalist>`;
    }
    return `<div class="pref"><label for="${id}">${esc(label)}</label>
      <div class="pref-control">${control}${s.locked || s.source === "default" ? "" : `<button type="button" class="link-btn" data-reset="${esc(s.key)}">${esc(t("system.settings.reset"))}</button>`}</div>${help}</div>`;
  }

  return () => stopPull?.();
}

registerPage({
  id: "system",
  path: "/system",
  nav: { label: "nav.system", order: 90 },
  title: () => t("system.title"),
  mount,
});
