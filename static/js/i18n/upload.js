// Texts of the file upload in the "New transcript" form. Importing this module merges them into the main
// dictionaries (extendLanguages).

import { extendLanguages } from "../core/i18n.js";

const en = {
  home: {
    upload_pick: "Choose files…",
    upload_help: "or drop them anywhere on this card. Audio and video, up to {mb} MB each.",
    upload_help_nolimit: "or drop them anywhere on this card. Audio and video.",
    upload_drop: "Drop files to transcribe them",
    upload_list: "Files to transcribe",
    upload_remove: "Remove {name}",
    upload_cancel: "Cancel upload of {name}",
    upload_waiting: "Waiting",
    upload_uploading: "Uploading {pct} %",
    upload_sending: "Uploading {name}: {pct} %",
    upload_done_toast: { one: "{n} file uploaded", other: "{n} files uploaded" },
    upload_summary: "Uploaded {ok} of {total} files.",
    upload_bad_type: "“{name}” is not an audio or video file.",
    upload_too_large: "“{name}” is larger than the limit of {mb} MB.",
    upload_empty: "“{name}” is empty.",
    need_source: "Enter a link or add a file.",
  },
  errors: {
    network: "The upload failed: the server cannot be reached.",
    unsupported_media_type: "Only audio and video files can be transcribed.",
    request_too_large: "The file is larger than the server's limit (max_upload_mb).",
    empty_file: "The file is empty.",
  },
};

const cs = {
  home: {
    upload_pick: "Vybrat soubory…",
    upload_help: "nebo je přetáhněte kamkoli na tuto kartu. Zvuk i video, každý do {mb} MB.",
    upload_help_nolimit: "nebo je přetáhněte kamkoli na tuto kartu. Zvuk i video.",
    upload_drop: "Pusťte soubory k přepisu",
    upload_list: "Soubory k přepisu",
    upload_remove: "Odebrat {name}",
    upload_cancel: "Zrušit nahrávání souboru {name}",
    upload_waiting: "Čeká",
    upload_uploading: "Nahrávám {pct} %",
    upload_sending: "Nahrávám {name}: {pct} %",
    upload_done_toast: { one: "Nahrán {n} soubor", few: "Nahrány {n} soubory", many: "Nahráno {n} souborů", other: "Nahráno {n} souborů" },
    upload_summary: "Nahráno {ok} z {total} souborů.",
    upload_bad_type: "„{name}“ není zvukový ani video soubor.",
    upload_too_large: "„{name}“ je větší než limit {mb} MB.",
    upload_empty: "„{name}“ je prázdný.",
    need_source: "Zadejte odkaz nebo přidejte soubor.",
  },
  errors: {
    network: "Nahrávání selhalo: server není dostupný.",
    unsupported_media_type: "Přepsat jde jen zvukové a video soubory.",
    request_too_large: "Soubor je větší než limit serveru (max_upload_mb).",
    empty_file: "Soubor je prázdný.",
  },
};

extendLanguages({ en, cs });
