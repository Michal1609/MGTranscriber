// fetch wrapper: JSON in/out, errors become Error objects with a translated message.

import { has, t } from "./i18n.js";
import { showLogin } from "../components/login.js";

// An error response (status + body text) -> Error with a translated message and the server's error code.
function failure(status, text) {
  let msg = t("common.error_status", { status });
  let code = null;
  try {
    const body = JSON.parse(text);
    code = body.code || null;
    if (code && has(`errors.${code}`)) msg = t(`errors.${code}`);
    else if (typeof body.detail === "string") msg = body.detail;
    else if (Array.isArray(body.detail)) msg = t("common.invalid_form");
  } catch {}
  if (code === "auth_required") showLogin();
  return Object.assign(new Error(msg), { status, code });
}

// components/stale.js registers a callback for "this page is newer than the server it talks to"
let mismatch = null;
export function setMismatchHandler(fn) {
  mismatch = fn;
}

export async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    const err = failure(res.status, await res.text().catch(() => ""));
    // our own errors carry a code; a bare 404 / 422 on a GET means the server does not know this endpoint (old server)
    const method = (options.method || "GET").toUpperCase();
    if (method === "GET" && (res.status === 404 || res.status === 422) && !err.code && path.startsWith("/api/")) mismatch?.();
    throw err;
  }
  return res.json();
}

// POST a FormData with upload progress (fetch cannot report it). onProgress(fraction 0..1); `signal` aborts it and
// the promise then rejects with code "aborted". Resolves with the JSON body.
export function upload(path, form, { onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) return reject(failure(xhr.status, xhr.responseText));
      try {
        resolve(JSON.parse(xhr.responseText));
      } catch (err) {
        reject(err);
      }
    };
    xhr.onerror = () => reject(Object.assign(new Error(t("errors.network")), { code: "network" }));
    xhr.onabort = () => reject(Object.assign(new Error("aborted"), { code: "aborted" }));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(form);
  });
}

// /api/info (name, version, defaults, engines, exporters), fetched once.
let infoPromise = null;
export function getInfo() {
  infoPromise ||= api("/api/info").catch((err) => {
    infoPromise = null;
    throw err;
  });
  return infoPromise;
}
