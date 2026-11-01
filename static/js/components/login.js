// API key login for servers started with MGTRANSCRIBER_API_KEY. The key is exchanged for an
// HttpOnly session cookie (POST /api/auth/login), which also covers SSE, audio and downloads.

import { t } from "../core/i18n.js";
import { esc } from "../core/util.js";

const post = (path, body) =>
  fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

// Shown when any API call answers 401 auth_required.
export function showLogin() {
  if (document.querySelector(".login-overlay")) return;
  const overlay = document.createElement("div");
  overlay.className = "login-overlay";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-labelledby", "login-title");
  overlay.innerHTML = `
    <form class="card login" autocomplete="on">
      <h2 id="login-title">${esc(t("auth.title"))}</h2>
      <p>${esc(t("auth.prompt"))}</p>
      <input type="text" name="username" value="${esc(t("app.name"))}" autocomplete="username" hidden>
      <input class="search-input" type="password" name="key" autocomplete="current-password" required
             placeholder="${esc(t("auth.key"))}" aria-label="${esc(t("auth.key"))}">
      <p class="form-msg" role="status"></p>
      <button class="btn primary" type="submit">${esc(t("auth.submit"))}</button>
    </form>`;
  const form = overlay.querySelector("form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const res = await post("/api/auth/login", { key: form.key.value });
    if (res.ok) location.reload();
    else {
      form.querySelector(".form-msg").textContent = t(res.status === 401 ? "auth.invalid" : "common.error_status", { status: res.status });
      form.key.select();
    }
  });
  document.body.append(overlay);
  form.key.focus();
}

// "Log out" in the top bar when the server requires a key and this browser is logged in.
export async function mountAuthControls(container) {
  let status;
  try {
    status = await (await fetch("/api/auth/status")).json();
  } catch {
    return;
  }
  if (!status.required || !status.authenticated) return;
  const button = document.createElement("button");
  button.className = "btn logout";
  button.type = "button";
  button.dataset.i18n = "auth.logout";
  button.textContent = t("auth.logout");
  button.addEventListener("click", async () => {
    await post("/api/auth/logout");
    location.reload();
  });
  container.append(button);
}
