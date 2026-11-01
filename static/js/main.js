// Entry point: shell setup, page registration, router start.
// To add a page: create js/pages/<name>.js that calls registerPage(), then import it below.

import { applyI18n, getLang, LANGUAGES, setLang } from "./core/i18n.js";
import { startRouter } from "./core/router.js";
import { mountAuthControls } from "./components/login.js";
import { $ } from "./core/util.js";
import { mountNavBadge } from "./components/navbadge.js";
import { mountStaleWatch } from "./components/stale.js";

import "./pages/home.js";
import "./pages/jobs.js";
import "./pages/detail.js";
import "./pages/voices.js";
import "./pages/voice_person.js";
import "./pages/watch.js";
import "./pages/system.js";

const select = $("#lang");
select.innerHTML = Object.keys(LANGUAGES)
  .map((code) => `<option value="${code}">${code.toUpperCase()}</option>`)
  .join("");
select.value = getLang();
select.addEventListener("change", () => setLang(select.value));

applyI18n();
startRouter();
mountAuthControls($(".topbar-end"));
mountNavBadge($("#nav"));
mountStaleWatch();
