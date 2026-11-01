// Texts of the "server runs an older version" banner.

import { extendLanguages } from "../core/i18n.js";

const en = {
  stale: {
    text: "The server is running an older version than this page, so some things will not work. Restart it (close its window and start web.bat again), then reload the page.",
    reload: "Reload the page",
    dismiss: "Hide",
  },
};

const cs = {
  stale: {
    text: "Server běží ve starší verzi než tato stránka, takže některé věci nebudou fungovat. Restartujte ho (zavřete jeho okno a znovu spusťte web.bat) a stránku načtěte znovu.",
    reload: "Načíst stránku znovu",
    dismiss: "Skrýt",
  },
};

extendLanguages({ en, cs });
