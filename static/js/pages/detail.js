// Transcript detail: speaker names, search, copy, exports and the audio player.

import { api, getInfo } from "../core/api.js";
import { applyI18n, t } from "../core/i18n.js";
import { mountAnalysis } from "../components/analysis.js";
import { mountVoices } from "../components/voices.js";
import { mountEditor } from "../components/editor.js";
import { mountPrivacy } from "../components/privacy.js";
import { mountFolderLine } from "../components/folders.js";
import { deleteTranscripts } from "../components/remove.js";
import { mountTranslations } from "../components/translations.js";
import { href, isCurrent, navigate, registerPage } from "../core/router.js";
import { $, clock, copyText, date, duration, esc, realSpeakers, shortClock, speakerColor, speakerLabel, speakerOrder, toast, UNASSIGNED_MIN_S } from "../core/util.js";

const TEMPLATE = `
  <a class="back" href="${href("/")}" data-i18n="detail.back"></a>
  <header class="detail-head">
    <div class="d-titlebar"><h1 class="d-title"></h1>
      <button type="button" class="btn small danger d-delete" data-i18n="remove.action"></button></div>
    <p class="meta d-meta"></p>
    <p class="meta d-origin" hidden></p>
    <p class="meta d-nospeech" role="note" hidden></p>
    <div class="d-folder" hidden></div>
  </header>

  <section class="speakers card d-speakers" data-i18n-aria-label="detail.speakers" hidden></section>

  <div class="tabs d-tabs" role="tablist" hidden>
    <button class="tab active" role="tab" aria-selected="true" tabindex="0" id="tab-transcript" aria-controls="pane-transcript" data-tab="transcript" data-i18n="detail.tab_transcript"></button>
    <button class="tab" role="tab" aria-selected="false" tabindex="-1" id="tab-analysis" aria-controls="pane-analysis" data-tab="analysis" data-i18n="llm.tab"></button>
    <button class="tab" role="tab" aria-selected="false" tabindex="-1" id="tab-ask" aria-controls="pane-ask" data-tab="ask" data-i18n="llm.ask_tab"></button>
  </div>
  <section class="d-pane d-analysis" data-pane="analysis" role="tabpanel" id="pane-analysis" aria-labelledby="tab-analysis" hidden></section>
  <section class="d-pane d-ask" data-pane="ask" role="tabpanel" id="pane-ask" aria-labelledby="tab-ask" hidden></section>

  <div class="d-pane" data-pane="transcript" role="tabpanel" id="pane-transcript" aria-labelledby="tab-transcript">
  <div class="d-langbar" hidden></div>
  <div class="toolbar d-toolbar" hidden>
    <div class="search">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 4a6 6 0 1 0 3.9 10.6l4.7 4.7 1.4-1.4-4.7-4.7A6 6 0 0 0 10 4zm0 2a4 4 0 1 1 0 8 4 4 0 0 1 0-8z"/></svg>
      <input class="d-search" type="search" data-i18n-placeholder="detail.search" data-i18n-aria-label="detail.search">
      <span class="search-count d-count"></span>
      <button class="icon-btn d-prev" data-i18n-title="detail.prev" data-i18n-aria-label="detail.prev_aria">↑</button>
      <button class="icon-btn d-next" data-i18n-title="detail.next" data-i18n-aria-label="detail.next_aria">↓</button>
    </div>
    <div class="actions">
      <button class="btn d-edit" data-i18n="editor.edit" data-i18n-title="editor.edit_title"></button>
      <button class="btn d-copy" data-i18n="detail.copy"></button>
      <span class="d-exports"></span>
    </div>
  </div>

  <article class="blocks d-blocks"><div class="loading" data-i18n="detail.loading"></div></article>
  <section class="d-editor" hidden></section>
  <details class="card privacy-panel d-privacy" hidden></details>
  </div>

  <div class="player d-player" hidden>
    <audio class="d-audio" controls preload="metadata"></audio>
    <label class="follow"><input type="checkbox" class="d-follow" checked> <span data-i18n="detail.follow"></span></label>
  </div>`;

const external = (url, text) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>`;

/** One "Source" line: link to the original, channel, the watch source it came from (or the local file name). */
function renderOrigin(box, origin) {
  const parts = [];
  if (origin?.url) {
    let host = origin.url;
    try { host = new URL(origin.url).hostname.replace(/^www\./, ""); } catch { /* keep the whole URL */ }
    parts.push(`${esc(t("detail.sourceVideo"))}: ${external(origin.url, host)}`);
  } else if (origin?.file) {
    parts.push(`${esc(t("detail.sourceFile"))}: <span title="${esc(origin.path || "")}">${esc(origin.file)}</span>`);
  }
  if (origin?.channel) {
    parts.push(`${esc(t("detail.sourceChannel"))}: ${origin.channel_url ? external(origin.channel_url, origin.channel) : esc(origin.channel)}`);
  }
  if (origin?.watch) {
    parts.push(`${esc(t("detail.sourceWatch"))}: <a href="${href("/watch")}">${esc(origin.watch.name || origin.watch.id)}</a>`);
  }
  box.hidden = !parts.length;
  box.innerHTML = parts.length ? `<strong>${esc(t("detail.sourceLabel"))}</strong> ` + parts.map((s) => `<span>${s}</span>`).join("") : "";
}

function mount(el, { id }, query = {}) {
  el.innerHTML = TEMPLATE;
  applyI18n(el);
  const q = (cls) => $(`.${cls}`, el);
  q("d-title").textContent = id;

  // state of this page instance
  const state = { data: null, names: {}, segs: [], whoEls: {}, colorOf: {}, matches: [], matchIndex: -1, playing: null };
  const audio = q("d-audio");
  let gone = false; // set by the cleanup: events from the dying <audio> must not touch the page
  // Every play() caller (here and in the editor) gets a promise that is already handled: a play()
  // interrupted by pause()/load() (AbortError, e.g. on leaving the page) is expected and quiet.
  const rawPlay = audio.play.bind(audio);
  audio.play = () => {
    const p = rawPlay();
    return p?.catch((e) => { if (e?.name !== "AbortError") console.warn("audio.play() failed:", e); });
  };
  let saveTimer = null;
  let stopAnalysis = () => {};
  let stopVoices = () => {};
  let editor = null;
  let translations = null;
  let pendingTime = query.t ? Number(query.t) : null;
  let searchTimer;

  load();

  async function load() {
    let data, info;
    try {
      [data, info] = await Promise.all([api(`/api/transcripts/${encodeURIComponent(id)}`), getInfo()]);
    } catch (err) {
      if (isCurrent("detail", { id })) q("d-blocks").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
      return;
    }
    if (!isCurrent("detail", { id })) return;

    document.title = `${data.title} · ${t("app.name")}`;
    state.data = data;
    state.names = { ...data.names };
    const order = speakerOrder(data.speakers);
    state.colorOf = Object.fromEntries(order.map((s, i) => [s, speakerColor(i)]));

    q("d-title").textContent = data.title;
    q("d-meta").innerHTML = [date(data.date), duration(data.duration), t("common.speakers", { n: realSpeakers(data.speakers).length }),
      t("common.words", { n: data.words }),
      ...(data.metadata?.extra?.separation?.used ? [t("detail.separated")] : [])]
      .map((s) => `<span>${esc(s)}</span>`).join("");
    renderOrigin(q("d-origin"), data.origin);
    const quiet = q("d-nospeech");
    quiet.hidden = !data.no_speech;
    quiet.textContent = data.no_speech ? t("detail.noSpeech") : "";

    q("d-delete").addEventListener("click", async () => {
      const result = await deleteTranscripts([{ id, title: data.title }]);
      if (result?.deleted.length) navigate("/");
    });
    q("d-folder").hidden = false;
    mountFolderLine(q("d-folder"), { id, title: data.title, folder: data.folder || "" });

    const base = `/api/transcripts/${encodeURIComponent(id)}`;
    state.exporters = info.exporters;
    renderExports();

    q("d-toolbar").hidden = q("d-speakers").hidden = false;
    renderSpeakers(order);
    stopVoices = mountVoices({ id, box: q("d-speakers"), getName: (spk) => (state.names[spk] || "").trim(),
      onNames: applyNames });
    renderBlocks();
    q("d-langbar").hidden = false;  // language switcher: a translated view re-renders the blocks (onView)
    translations = mountTranslations({ id, box: q("d-langbar"), sourceLang: data.metadata?.language || "",
      initial: query.lang || "", onView: showLanguage });
    q("d-tabs").hidden = false;
    stopAnalysis = mountAnalysis({ id, summaryEl: q("d-analysis"), askEl: q("d-ask"), names: state.names,
      colorOf: state.colorOf, seek: seekTo });

    if (data.audio) {
      audio.src = `${base}/files/${encodeURIComponent(data.audio)}`;
      q("d-player").hidden = false;
    }
    el.classList.toggle("has-audio", !!data.audio);
    q("d-privacy").hidden = false;
    mountPrivacy({ id, box: q("d-privacy"), exporters: info.exporters, hasAudio: !!data.audio,
      onDeleted: () => navigate("/") });
    if (query.edit === "1") startEditing();
    else if (query.t) showTime(Number(query.t));
  }

  /* --- languages (components/translations.js): a translation has the segments of the original, 1:1 --- */

  function renderExports() {
    const base = `/api/transcripts/${encodeURIComponent(id)}`;
    const lang = state.lang ? `?lang=${encodeURIComponent(state.lang)}` : "";
    q("d-exports").innerHTML = state.exporters
      .map((e) => `<a class="btn" download href="${base}/export/${esc(e.name)}${lang}">${esc(e.label)}</a>`)
      .join(" ");
  }

  function showLanguage(lang, data) {
    state.lang = lang;
    state.translation = data;
    el.classList.toggle("translated-view", !!lang);  // the editor edits the original only (button hidden)
    renderExports();
    renderBlocks();
    if (search.value) runSearch();
    if (lang && Number.isFinite(pendingTime)) showTime(pendingTime);  // a search hit in a translation: ?lang=cs&t=12
    pendingTime = null;
    const keep = location.hash.split("?")[0];
    history.replaceState(null, "", lang ? href(`/t/${id}`, { lang }) : keep);  // a reload keeps the language
  }

  /* --- editor (components/editor.js): replaces the reading view until "Done" --- */

  function startEditing() {
    if (editor || !state.data) return;
    q("d-blocks").hidden = q("d-toolbar").hidden = true;
    q("d-editor").hidden = false;
    history.replaceState(null, "", href(`/t/${id}`, { edit: 1 })); // a reload stays in the editor
    editor = mountEditor({ id, box: q("d-editor"), audio, hasAudio: !!state.data.audio, names: state.names,
      colorOf: state.colorOf, onClose: () => {
        editor = null;
        // mount the page again with the saved text, at the moment being listened to
        const target = href(`/t/${id}`, { t: (audio.currentTime || 0).toFixed(1) });
        if (location.hash === target) window.dispatchEvent(new HashChangeEvent("hashchange"));
        else location.hash = target;
      } });
  }
  q("d-edit").addEventListener("click", startEditing);

  function renderSpeakers(order) {
    const { data, names } = state;
    const total = Object.values(data.speakers).reduce((a, b) => a + b, 0) || 1;
    const box = q("d-speakers");
    box.innerHTML = "";
    // people first; words with no speaker ("unassigned") only when there are more than a blink of them
    const shown = [...order.filter((s) => s !== "UNKNOWN"),
      ...order.filter((s) => s === "UNKNOWN" && data.speakers[s] >= UNASSIGNED_MIN_S)];
    for (const spk of shown) {
      const unassigned = spk === "UNKNOWN";
      const pct = (data.speakers[spk] / total) * 100;
      const row = document.createElement("div");
      row.className = unassigned ? "speaker unassigned" : "speaker";
      row.dataset.speaker = spk;
      row.innerHTML = `
        <div class="speaker-name"><span class="dot" style="background:${state.colorOf[spk]}"></span><input type="text" maxlength="60" aria-label="${esc(t("detail.speaker_name_for", { id: spk }))}"></div>
        <div class="bar"><i style="width:${pct}%;background:${state.colorOf[spk]}"></i></div>
        <small>${esc(unassigned ? t("speaker.unassigned") : spk)} · ${shortClock(data.speakers[spk])} · ${Math.round(pct)} %</small>`;
      const input = row.querySelector("input");
      input.value = names[spk] || "";
      input.placeholder = unassigned ? t("speaker.unassigned") : speakerLabel(spk);
      input.addEventListener("input", () => {
        names[spk] = input.value.trim();
        for (const who of state.whoEls[spk] || []) who.textContent = speakerLabel(spk, names);
        clearTimeout(saveTimer);
        saveTimer = setTimeout(saveNames, 700);
      });
      input.addEventListener("keydown", (e) => e.key === "Enter" && input.blur());
      box.append(row);
    }
  }

  // names changed on the server (voice memory named a speaker)
  function applyNames(names) {
    for (const spk of Object.keys(state.data.speakers)) {
      state.names[spk] = names[spk] || "";
      const input = q("d-speakers").querySelector(`[data-speaker="${CSS.escape(spk)}"] input`);
      if (input && document.activeElement !== input) input.value = state.names[spk];
      for (const who of state.whoEls[spk] || []) who.textContent = speakerLabel(spk, state.names);
    }
  }

  async function saveNames() {
    saveTimer = null;
    const clean = Object.fromEntries(Object.entries(state.names).filter(([, v]) => v));
    try {
      await api(`/api/transcripts/${encodeURIComponent(id)}/speakers`, { method: "PUT", body: JSON.stringify(clean) });
      toast(t("detail.names_saved"));
      q("d-speakers").dispatchEvent(new CustomEvent("names-saved"));
    } catch (err) {
      toast(t("detail.save_failed", { error: err.message }));
    }
  }

  function renderBlocks() {
    const frag = document.createDocumentFragment();
    let block = null;
    let text = null;
    state.segs = [];
    state.whoEls = {};
    state.playing = null;
    activeSeg = 0;
    const shown = state.lang && state.translation
      ? state.translation.segments.map((s) => ({ ...s, speaker: s.speaker || "UNKNOWN" }))
      : state.data.segments;
    shown.forEach((seg, i) => {
      if (!block || block.dataset.speaker !== seg.speaker) {
        block = document.createElement("section");
        block.className = "block";
        block.dataset.speaker = seg.speaker;
        block.style.setProperty("--c", state.colorOf[seg.speaker] || "var(--muted)");
        const side = document.createElement("div");
        side.className = "block-side";
        const who = document.createElement("span");
        who.className = "who";
        who.textContent = speakerLabel(seg.speaker, state.names);
        (state.whoEls[seg.speaker] ||= []).push(who);
        const time = document.createElement("button");
        time.className = "time";
        time.type = "button";
        time.tabIndex = -1; // one tab stop per segment would bury the player; the text spans are clickable too
        time.dataset.i = i;
        time.textContent = clock(seg.start);
        side.append(who, time);
        text = document.createElement("p");
        text.className = "block-text";
        block.append(side, text);
        frag.append(block);
      } else {
        text.append(" ");
      }
      const span = document.createElement("span");
      span.className = "seg";
      span.dataset.i = i;
      span.textContent = seg.text;
      text.append(span);
      state.segs.push({ el: span, text: seg.text, start: seg.start, end: seg.end, speaker: seg.speaker });
    });
    q("d-blocks").replaceChildren(frag);
    if (state.data.audio) {  // keyboard: one tab stop, arrows move between segments, Enter/Space plays
      state.segs.forEach((s, k) => {
        s.el.tabIndex = k === 0 ? 0 : -1;
        s.el.setAttribute("role", "button");
      });
    }
  }

  let activeSeg = 0;
  q("d-blocks").addEventListener("focusin", (e) => {
    const i = Number(e.target.closest?.(".seg")?.dataset.i);
    if (Number.isNaN(i) || i === activeSeg) return;
    state.segs[activeSeg].el.tabIndex = -1;
    state.segs[i].el.tabIndex = 0;
    activeSeg = i;
  });
  q("d-blocks").addEventListener("keydown", (e) => {
    const seg = e.target.closest?.(".seg");
    if (!seg || e.altKey || e.ctrlKey || e.metaKey) return;
    const i = Number(seg.dataset.i);
    const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
    let to = step ? i + step : e.key === "Home" ? 0 : e.key === "End" ? state.segs.length - 1 : null;
    if (to !== null) {
      e.preventDefault();
      to = Math.max(0, Math.min(state.segs.length - 1, to));
      state.segs[to].el.focus();
      state.segs[to].el.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      audio.currentTime = state.segs[i].start;
      audio.play();
    }
  });

  q("d-blocks").addEventListener("click", (e) => {
    if (!state.data?.audio || window.getSelection().toString()) return;
    const target = e.target.closest("[data-i]");
    if (!target) return;
    audio.currentTime = state.segs[Number(target.dataset.i)].start;
    audio.play();
  });

  /* --- tabs and jumping to a time (summary, chapters, quotes and answers link here) --- */

  function showTab(name) {
    for (const tab of el.querySelectorAll(".d-tabs .tab")) {
      tab.classList.toggle("active", tab.dataset.tab === name);
      tab.setAttribute("aria-selected", tab.dataset.tab === name);
      tab.tabIndex = tab.dataset.tab === name ? 0 : -1; // roving tabindex: arrows move between tabs
    }
    for (const pane of el.querySelectorAll(".d-pane")) pane.hidden = pane.dataset.pane !== name;
  }
  q("d-tabs").addEventListener("click", (e) => {
    const tab = e.target.closest("[data-tab]");
    if (tab) showTab(tab.dataset.tab);
  });
  q("d-tabs").addEventListener("keydown", (e) => {
    const tabs = [...el.querySelectorAll(".d-tabs .tab")];
    const i = tabs.indexOf(document.activeElement);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (i < 0 || next === undefined) return;
    e.preventDefault();
    const tab = tabs[(next + tabs.length) % tabs.length];
    showTab(tab.dataset.tab);
    tab.focus();
  });

  function seekTo(seconds) {
    const seg = state.segs.find((s) => s.start >= seconds - 0.01) || state.segs[state.segs.length - 1];
    if (state.data?.audio) {
      audio.currentTime = seconds;
      audio.play();
    } else {
      showTab("transcript");
      seg?.el.scrollIntoView({ block: "center", behavior: "smooth" });
      seg?.el.classList.add("flash");
      setTimeout(() => seg?.el.classList.remove("flash"), 2200);
    }
  }

  /* --- search --- */

  const search = q("d-search");
  search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(runSearch, 150);
  });
  search.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      stepMatch(e.shiftKey ? -1 : 1);
    } else if (e.key === "Escape") {
      search.value = "";
      runSearch();
    }
  });
  q("d-next").addEventListener("click", () => stepMatch(1));
  q("d-prev").addEventListener("click", () => stepMatch(-1));

  function runSearch() {
    const query = search.value.trim();
    // capture group: split() then returns the matches at odd indexes
    const re = query ? new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "i") : null;
    state.matches = [];
    for (const seg of state.segs) {
      const parts = re ? seg.text.split(re) : [];
      if (parts.length > 1) {
        seg.el.innerHTML = parts.map((p, i) => (i % 2 ? `<mark>${esc(p)}</mark>` : esc(p))).join("");
        state.matches.push(...seg.el.querySelectorAll("mark"));
        seg.marked = true;
      } else if (seg.marked) {
        seg.el.textContent = seg.text;
        seg.marked = false;
      }
    }
    state.matchIndex = -1;
    const n = state.matches.length;
    q("d-count").textContent = query ? (n ? t("detail.matches", { n }) : t("detail.no_matches")) : "";
    if (n) stepMatch(1);
  }

  function stepMatch(dir) {
    const { matches } = state;
    if (!matches.length) return;
    matches[state.matchIndex]?.classList.remove("current");
    state.matchIndex = (state.matchIndex + dir + matches.length) % matches.length;
    const m = matches[state.matchIndex];
    m.classList.add("current");
    m.scrollIntoView({ block: "center", behavior: "smooth" });
    q("d-count").textContent = `${state.matchIndex + 1} / ${matches.length}`;
  }

  /* --- #/t/<id>?t=<seconds> (search results, MCP links): show that moment without playing --- */

  function showTime(seconds) {
    if (!Number.isFinite(seconds)) return;
    let seg = state.segs[0];
    for (const s of state.segs) {
      if (s.start > seconds + 0.06) break; // links carry tenths of a second
      seg = s;
    }
    if (!seg) return;
    seg.el.classList.add("target");
    requestAnimationFrame(() => seg.el.scrollIntoView({ block: "center" }));
    if (state.data.audio) {
      const seek = () => (audio.currentTime = Math.max(seconds, seg.start));
      if (audio.readyState >= 1) seek();
      else audio.addEventListener("loadedmetadata", seek, { once: true });
    }
  }

  /* --- copy --- */

  q("d-copy").addEventListener("click", async () => {
    const out = [];
    let last = null;
    for (const seg of state.segs) {
      if (!last || last.speaker !== seg.speaker) {
        last = { speaker: seg.speaker, lines: [], head: `[${clock(seg.start)}] ${speakerLabel(seg.speaker, state.names)}:` };
        out.push(last);
      }
      last.lines.push(seg.text);
    }
    await copyText(out.map((b) => `${b.head}\n${b.lines.join(" ")}`).join("\n\n"));
    toast(t("detail.copied"));
  });

  /* --- player: highlight the segment being spoken --- */

  audio.addEventListener("timeupdate", () => {
    if (gone) return;
    const now = audio.currentTime;
    const segs = state.segs;
    let lo = 0, hi = segs.length - 1, found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (segs[mid].start <= now) { found = mid; lo = mid + 1; } else hi = mid - 1;
    }
    // voices may overlap: of the segments that started, prefer the latest one still being spoken
    let seg = null;
    for (let k = found; k >= 0 && segs[k].start >= now - 30; k--) {
      if (now <= segs[k].end) { seg = segs[k]; break; }
    }
    if (!seg && found >= 0 && now <= segs[found].end + 1.5) seg = segs[found];
    if (seg === state.playing) return;
    state.playing?.el.classList.remove("playing");
    state.playing = seg;
    if (!seg) return;
    seg.el.classList.add("playing");
    if (q("d-follow").checked && !audio.paused && !search.value) {
      const r = seg.el.getBoundingClientRect();
      if (r.top < 120 || r.bottom > window.innerHeight - 90) seg.el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  });

  return () => {
    gone = true;
    editor?.stop();
    stopAnalysis();
    stopVoices();
    translations?.stop();
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveNames(); // do not lose a rename typed just before leaving
    }
    clearTimeout(searchTimer);
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  };
}

registerPage({ id: "detail", path: "/t/:id", title: ({ id }) => id, mount });
