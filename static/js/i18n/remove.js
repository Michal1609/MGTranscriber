// Texts of the quick "Delete" action for transcripts (library tiles, selection bar, detail page).
// Importing this module merges them into the main dictionaries. Used by components/remove.js.

import { extendLanguages } from "../core/i18n.js";

const en = {
  remove: {
    action: "Delete",
    aria: "Delete {title}",
    title: { one: "Delete this transcript?", other: "Delete {n} transcripts?" },
    text: "This cannot be undone.",
    more: "…and {n} more",
    confirm: "Delete",
    done: { one: "Transcript deleted", other: "{n} transcripts deleted" },
    partial: "{ok} deleted, {failed} not deleted: {error}",
    failed: "Not deleted: {error}",
  },
};

const cs = {
  remove: {
    action: "Smazat",
    aria: "Smazat {title}",
    title: { one: "Smazat tento přepis?", few: "Smazat {n} přepisy?", many: "Smazat {n} přepisů?", other: "Smazat {n} přepisů?" },
    text: "Tuto akci nelze vrátit.",
    more: "…a dalších {n}",
    confirm: "Smazat",
    done: { one: "Přepis smazán", few: "{n} přepisy smazány", many: "{n} přepisů smazáno", other: "{n} přepisů smazáno" },
    partial: "Smazáno: {ok}, nesmazáno: {failed} ({error})",
    failed: "Nesmazáno: {error}",
  },
};

extendLanguages({ en, cs });
