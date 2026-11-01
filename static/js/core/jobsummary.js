// Shared poll of GET /api/jobs/summary: the running job, queue length and recent failures.
// Used by the nav badge and the home page's "now running" strip; one timer however many listeners.

import { api } from "./api.js";

const POLL_MS = 2500;
const listeners = new Set();
let timer = null;
let last = null;
let inflight = null;

export function currentSummary() {
  return last;
}

export function refreshSummary() {
  inflight ||= api("/api/jobs/summary")
    .then((summary) => {
      last = summary;
      for (const fn of [...listeners]) fn(summary);
      return summary;
    })
    .catch(() => last)
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

// fn(summary) is called now (when known) and after each poll; returns the unsubscribe function.
export function watchSummary(fn) {
  listeners.add(fn);
  if (last) fn(last);
  if (!timer) timer = setInterval(() => !document.hidden && refreshSummary(), POLL_MS);
  refreshSummary();
  return () => {
    listeners.delete(fn);
    if (!listeners.size) {
      clearInterval(timer);
      timer = null;
    }
  };
}
