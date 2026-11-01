// Count of running + queued jobs on the "Jobs" nav link. The nav is re-rendered on every route change, so the
// badge is filled again whenever its links are replaced.

import { t } from "../core/i18n.js";
import { watchSummary } from "../core/jobsummary.js";

export function mountNavBadge(nav) {
  let active = 0;

  function paint() {
    for (const badge of nav.querySelectorAll('[data-badge="jobs"]')) {
      badge.hidden = active === 0;
      badge.innerHTML = active
        ? `<span aria-hidden="true">${active}</span><span class="sr-only"> ${t("jobs.badge_hint", { n: active })}</span>`
        : "";
    }
  }

  new MutationObserver(paint).observe(nav, { childList: true }); // only the links being replaced, not the badge text
  return watchSummary((s) => {
    active = s.active;
    paint();
  });
}
