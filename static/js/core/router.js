// Hash router and page registry.
//
// A page is a module that calls registerPage({...}):
//   id        unique name
//   path      "/" or a pattern with params, e.g. "/t/:id"  (matched against location.hash)
//   nav       optional { label: "<i18n key>", order: 10, badge: true } to get a link in the top bar
//             (badge: an empty <span data-badge="<id>"> in the link; components/navbadge.js fills it)
//   title     (params) => document title part
//   mount     (el, params, query) => cleanup?   render into el; may return a cleanup function.
//             query holds the hash's query string: "#/t/abc?t=12" -> { t: "12" }
// Pages are re-mounted when the language changes.

import { onLangChange, t } from "./i18n.js";
import { $ } from "./util.js";

const pages = [];
let current = null; // { page, params, cleanup }

export function registerPage(page) {
  const names = [];
  const source = page.path.replace(/:(\w+)/g, (_, name) => (names.push(name), "([^/]+)"));
  pages.push({ ...page, regex: new RegExp(`^${source}$`), names });
}

export const href = (path, query) => `#${path}${query ? `?${new URLSearchParams(query)}` : ""}`;
export const navigate = (path) => (location.hash = href(path));

function match(path) {
  for (const page of pages) {
    const m = path.match(page.regex);
    if (m) return { page, params: Object.fromEntries(page.names.map((n, i) => [n, decodeURIComponent(m[i + 1])])) };
  }
  return null;
}

function render() {
  const [path, search = ""] = (location.hash.replace(/^#/, "") || "/").split("?");
  const found = match(path || "/") || match("/");
  const query = Object.fromEntries(new URLSearchParams(search));
  current?.cleanup?.();
  const view = $("#view");
  view.replaceChildren();
  view.className = "page";
  window.scrollTo(0, 0);
  current = { ...found, query, cleanup: null };
  const name = t("app.name");
  const part = found.page.title?.(found.params);
  document.title = part ? `${part} · ${name}` : name;
  current.cleanup = found.page.mount(view, found.params, query) || null;
  renderNav(found.page);
}

function renderNav(active) {
  const nav = $("#nav");
  const items = pages.filter((p) => p.nav).sort((a, b) => (a.nav.order ?? 50) - (b.nav.order ?? 50));
  nav.hidden = items.length < 2; // a single entry is just the brand link
  nav.innerHTML = items
    .map((p) => `<a href="${href(p.path)}" class="${p === active ? "active" : ""}">${t(p.nav.label)}${p.nav.badge ? ` <span class="nav-badge" data-badge="${p.id}" hidden></span>` : ""}</a>`)
    .join("");
}

export function startRouter() {
  window.addEventListener("hashchange", render);
  onLangChange(render);
  render();
}

// Lets a page check that it is still the one on screen after an await.
export const isCurrent = (pageId, params = {}) =>
  current?.page.id === pageId && Object.entries(params).every(([k, v]) => current.params[k] === v);
