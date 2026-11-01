"""Neutral demo data for screenshots and manual testing: invented transcripts, no third-party audio or text.

    uv run python -m tools.make_demo_data OUTPUT_DIR DATA_DIR

writes three fictional transcripts (a two-person interview, a three-person meeting, a one-person lecture) with
speaker names, an LLM-style analysis and a silent placeholder audio track, plus a voice database, a watch
list and a few library folders (one empty) in DATA_DIR. Point MGTRANSCRIBER_OUTPUT_DIR / MGTRANSCRIBER_DATA_DIR at the two folders to look at it.
Everything here is made up; the "voices" are random vectors, but built so that the archive has something to
show: automatic matches, suggestions of different strength (a voice at a chosen cosine to a person), an unknown
voice and a duplicate person who can be merged.
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

from mgtranscriber import voices
from mgtranscriber.config import Settings
from mgtranscriber.engines.common import spread_words
from mgtranscriber.folders import FolderStore
from mgtranscriber.library import Library, Metadata, transcript_id_for
from mgtranscriber.llm.store import save_analysis
from mgtranscriber.watch.models import Source

SECONDS_PER_WORD = 0.36

INTERVIEW = [
    ("A", "Vítejte u dalšího dílu. Dnes si povídáme o tom, jak si zařídit malou zahrádku na balkoně. Mým hostem je Petr Dvořák, který pěstuje bylinky už deset let."),
    ("B", "Dobrý den, děkuji za pozvání. Je pravda, že začínal jsem s jediným květináčem bazalky, který mi hned na první zimu zmrzl."),
    ("A", "To zní povědomě. Co byste tedy začátečníkovi poradil jako první?"),
    ("B", "Vybrat správné místo. Většina bylinek potřebuje alespoň pět hodin slunce denně, takže jižní nebo západní balkon je ideální."),
    ("A", "A když mám balkon na sever?"),
    ("B", "Pak sázejte máty, petrželku nebo pažitku. Ty snesou i polostín a odmění se vám i tak."),
    ("A", "Jak je to se zálivkou? Já své rostliny většinou utopím."),
    ("B", "To je nejčastější chyba. Bylinky chtějí spíš méně vody, ale pravidelně. Prstem zkuste půdu do hloubky dvou centimetrů a zalévejte, až když je suchá."),
    ("A", "Jakou zeminu používáte?"),
    ("B", "Obyčejnou univerzální zeminu smíchanou s třetinou písku nebo drobného keramzitu. Důležité je, aby měl květináč na dně otvory pro odtok vody."),
    ("A", "Dá se na jednom balkoně pěstovat i něco většího, třeba rajčata?"),
    ("B", "Samozřejmě. Existují nízké keříčkové odrůdy, které vyrostou v nádobě o objemu deseti litrů. Stačí jim pravidelné hnojení jednou týdně."),
    ("A", "Co sousedé a hmyz? Slyšela jsem, že bylinky přitahují včely."),
    ("B", "A to je na tom to nejlepší. Levandule a tymián v květu přilákají čmeláky i včely, a vy tak pomůžete i okolní přírodě."),
    ("A", "Jak dlouho vám trvá péče o balkon každý den?"),
    ("B", "Ráno asi deset minut. Zkontroluji zeminu, utrhnu pár lístků do snídaně a obhlédnu, jestli se neobjevil mšice."),
    ("A", "A na mšice máte nějaký trik?"),
    ("B", "Postříkat rostlinu zředěným mýdlovým roztokem. Funguje to překvapivě dobře a nic to neničí."),
    ("A", "Skvělé. Můžete na závěr říct, jaká bylinka je podle vás nejlepší na začátek?"),
    ("B", "Určitě pažitka. Roste skoro sama, nevadí jí chyby při zálivce a hodí se na všechno od chleba s máslem po polévku."),
    ("A", "Děkuji, Petře. A vám posluchačům přeji, ať se vašim bylinkám daří. Na slyšenou příště."),
]

MEETING = [
    ("A", "Dobrý den všem, začneme přípravou na jarní akci. Máme tři body: rozpočet, termín a rozdělení úkolů."),
    ("B", "K rozpočtu mám jednu poznámku. Tisk plakátů vyjde o pětinu dráž, než jsme plánovali."),
    ("C", "Mohu zjistit u druhé tiskárny, jestli nabídne lepší cenu. Dám vám vědět do čtvrtka."),
    ("A", "Výborně. Termín navrhuji třetí sobotu v dubnu, protože o týden dřív je státní svátek."),
    ("B", "Souhlasím. Sál si můžu rezervovat hned dnes."),
    ("B", "Pokud jde o sál, mají volno jen tuto sobotu a pak až v dubnu. Navrhuji tedy zálohu zaplatit hned, aby nám termín nikdo nepřebral, a smlouvu pošlu ostatním k nahlédnutí."),
    ("B", "A ještě poznámka k ozvučení: techniku si můžeme půjčit zdarma od sousedního spolku, stačí jim napsat zhruba čtrnáct dní předem."),
    ("C", "Já se postarám o občerstvení a o seznam hostů."),
    ("A", "Tak to máme. Shrnu to v e-mailu a pošlu všem ještě dnes večer."),
]

LECTURE = [
    ("A", "V této přednášce si ukážeme, jak funguje kváskový chléb od mouky až po pečení."),
    ("A", "Kvásek je směs mouky a vody, ve které žijí přirozené kvasinky a mléčné bakterie. Krmíme ho pravidelně každý den."),
    ("A", "Těsto necháme kynout několik hodin při pokojové teplotě, nebo přes noc v lednici, aby získalo plnější chuť."),
    ("A", "Pečeme v rozpálené troubě pod poklopem, aby se na povrchu vytvořila lesklá křupavá kůrka."),
    ("A", "Chléb před krájením nechte vychladnout alespoň hodinu, jinak zůstane střída vlhká."),
    ("A", "Tím se dostáváme na konec. Děkuji za pozornost a přeji dobrou chuť."),
]


def near(vector: np.ndarray, cos: float, rng: np.random.Generator) -> list[float]:
    """A vector whose cosine similarity to ``vector`` is ``cos``."""
    unit = vector / np.linalg.norm(vector)
    noise = rng.normal(size=vector.shape)
    noise -= (noise @ unit) * unit
    noise /= np.linalg.norm(noise)
    return (cos * unit + np.sqrt(1 - cos**2) * noise).tolist()


def timeline(lines: list[tuple[str, str]], names: dict[str, str]) -> tuple[list[dict], float]:
    t, segments = 2.0, []
    for label, text in lines:
        speaker = f"SPEAKER_{ord(label) - ord('A'):02d}"
        dur = max(2.5, len(text.split()) * SECONDS_PER_WORD)
        seg = spread_words({"start": round(t, 2), "end": round(t + dur, 2), "text": " " + text})
        seg["speaker"] = speaker
        for w in seg["words"]:
            w["speaker"], w["score"] = speaker, 0.9
        segments.append(seg)
        t += dur + 0.6
    return segments, round(t + 1, 1)


def silent_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80" * int(seconds * 8000))


def write_transcript(out: Path, title: str, source: str, lines, names, days_ago: float, language="cs") -> tuple[str, list[dict], float]:
    segments, duration = timeline(lines, names)
    tid = transcript_id_for(title, "demo:" + source)
    folder = out / tid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prepis.json").write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    speakers = sorted({s["speaker"] for s in segments})
    date = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - days_ago * 86400))
    meta = Metadata(title=title, source=source, date=date, duration=duration, language=language, engine="whisperx",
                    model="large-v3", speakers=speakers, source_key="demo:" + source)
    (folder / "metadata.json").write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    if names:
        (folder / "speakers.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    silent_wav(folder / "audio.wav", duration)
    return tid, segments, duration


MARKER = ".mgtranscriber-demo"


def claim(folder: Path) -> None:
    """Use ``folder`` for demo data only if it is empty or was made by this script. Never write into (or delete
    from) a real library or data folder."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / MARKER).exists():
        return
    if any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty and was not created by make_demo_data; refusing to use it.")
    (folder / MARKER).write_text("invented demo data, safe to delete", encoding="utf-8")


def build(out: Path, data: Path) -> dict:
    claim(out)
    claim(data)
    for stale in ("voiceprints.json", "watch.json"):  # running it twice must not pile up samples
        (data / stale).unlink(missing_ok=True)
    rng = np.random.default_rng(7)
    people = {"Jana Nováková": rng.normal(size=voices.DIM), "Petr Dvořák": rng.normal(size=voices.DIM),
              "Marta Svobodová": rng.normal(size=voices.DIM)}

    names = {"SPEAKER_00": "Jana Nováková", "SPEAKER_01": "Petr Dvořák"}
    tid, segments, dur = write_transcript(out, "Bylinky na balkoně: rozhovor", "https://example.org/demo/bylinky",
                                          INTERVIEW, names, days_ago=2)
    starts = [s["start"] for s in segments]

    def at(i: int) -> float:
        return starts[i]

    analysis = {
        "version": 1, "language": "cs", "provider": "ollama", "model": "gemma3:4b", "final_model": None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "duration": dur,
        "summary": ("Moderátorka si povídá s Petrem Dvořákem o pěstování bylinek na malém balkoně. Host radí, jak vybrat "
                    "místo podle slunce, jak správně zalévat a jakou použít zeminu. Zmiňuje i nízká rajčata, bylinky "
                    "přitahující včely a jednoduchou ochranu před mšicemi. Jako nejlepší začátek doporučuje pažitku."),
        "key_points": [
            "Většina bylinek potřebuje alespoň pět hodin slunce denně.",
            "Na severní balkon se hodí máty, petržel a pažitka.",
            "Zalévat až ve chvíli, kdy je půda do hloubky dvou centimetrů suchá.",
            "Zemina s třetinou písku a květináč s otvory pro odtok vody.",
            "Mšice pomůže odehnat zředěný mýdlový roztok.",
        ],
        "chapters": [
            {"start": at(0), "title": "Úvod a představení hosta", "summary": "Téma dílu a první zkušenosti hosta."},
            {"start": at(3), "title": "Světlo a umístění", "summary": "Kolik slunce bylinky potřebují a co na severní balkon."},
            {"start": at(6), "title": "Zálivka a zemina", "summary": "Jak nezahubit rostliny a jaká zemina se hodí."},
            {"start": at(10), "title": "Rajčata a hmyz", "summary": "Co ještě na balkon a proč přilákat včely."},
            {"start": at(14), "title": "Denní péče a mšice", "summary": "Rychlá ranní rutina a ochrana rostlin."},
            {"start": at(18), "title": "Závěr: nejlepší začátečnická bylinka", "summary": "Doporučení hosta pro úplné začátečníky."},
        ],
        "quotes": [
            {"time": at(7), "speaker": "SPEAKER_01", "text": "Bylinky chtějí spíš méně vody, ale pravidelně."},
            {"time": at(13), "speaker": "SPEAKER_01", "text": "Levandule a tymián v květu přilákají čmeláky i včely."},
            {"time": at(19), "speaker": "SPEAKER_01", "text": "Určitě pažitka. Roste skoro sama, nevadí jí chyby při zálivce."},
        ],
        "action_items": [
            {"time": at(5), "owner": "", "text": "Na severní balkon vybrat máty, petrželku nebo pažitku."},
            {"time": at(17), "owner": "", "text": "Připravit si zředěný mýdlový roztok proti mšicím."},
        ],
        "validation": {"chunks": 1, "max_time_snap_s": 0.0, "quotes_dropped": 0, "quotes_kept": 3},
        "usage": {"calls": 4, "prompt_tokens": 5200, "output_tokens": 900}, "seconds": 41.0,
    }
    lib = Library(out)
    save_analysis(lib, tid, analysis)

    tid2, _, _ = write_transcript(out, "Porada: jarní akce spolku", "https://example.org/demo/porada", MEETING,
                                  {"SPEAKER_00": "Jana Nováková", "SPEAKER_01": "Marta Svobodová"}, days_ago=9)
    tid3, _, _ = write_transcript(out, "Kváskový chléb krok za krokem", "https://example.org/demo/chleb", LECTURE, {},
                                  days_ago=30)
    tid4, _, _ = write_transcript(out, "Zahrada v říjnu: druhý rozhovor", "https://example.org/demo/zahrada",
                                  INTERVIEW[:14], {}, days_ago=5)
    tid5, _, _ = write_transcript(out, "Porada: podzimní brigáda", "https://example.org/demo/brigada", MEETING, {},
                                  days_ago=12)

    # voice memory: Jana and Petr are known people with a sample from the interview
    settings = Settings(root=data.parent, output_dir=out, data_dir=data)
    db = voices.open_db(settings)
    for name, vec, transcript, label, title in [
        ("Jana Nováková", people["Jana Nováková"], tid, "SPEAKER_00", "Bylinky na balkoně: rozhovor"),
        ("Petr Dvořák", people["Petr Dvořák"], tid, "SPEAKER_01", "Bylinky na balkoně: rozhovor"),
        ("Marta Svobodová", people["Marta Svobodová"], tid2, "SPEAKER_01", "Porada: jarní akce spolku"),
    ]:
        spoken = voices.speech_seconds(lib.segments(transcript)).get(label, 0.0)
        db.enroll(name, vec, speech_s=round(spoken, 1), force=True, source={"transcript": transcript, "label": label, "title": title})

    def noisy(name: str) -> list[float]:
        return (people[name] + 0.25 * rng.normal(size=voices.DIM)).tolist()

    voices.after_transcription(lib, tid, {"SPEAKER_00": noisy("Jana Nováková"), "SPEAKER_01": noisy("Petr Dvořák")}, settings)
    voices.after_transcription(lib, tid2, {"SPEAKER_00": noisy("Jana Nováková"), "SPEAKER_01": noisy("Marta Svobodová"),
                                           "SPEAKER_02": rng.normal(size=voices.DIM).tolist()}, settings)

    # the archive: three more transcripts found only through voice memory (nothing is named yet)
    voices.after_transcription(lib, tid3, {"SPEAKER_00": near(people["Marta Svobodová"], 0.55, rng)}, settings)  # suggestion
    voices.after_transcription(lib, tid4, {"SPEAKER_00": near(people["Jana Nováková"], 0.93, rng),
                                           "SPEAKER_01": near(people["Petr Dvořák"], 0.74, rng)}, settings)
    voices.after_transcription(lib, tid5, {"SPEAKER_00": near(people["Jana Nováková"], 0.5, rng),  # weak suggestion
                                           "SPEAKER_01": near(people["Marta Svobodová"], 0.8, rng),
                                           "SPEAKER_02": rng.normal(size=voices.DIM).tolist()}, settings)  # unknown
    # a duplicate: the same voice as Petr under another name (to try "merge people")
    spoken = voices.speech_seconds(lib.segments(tid4)).get("SPEAKER_01", 0.0)
    db.enroll("P. Dvořák", near(people["Petr Dvořák"], 0.9, rng), speech_s=round(spoken, 1), force=True,
              source={"transcript": tid4, "label": "SPEAKER_01", "title": "Zahrada v říjnu: druhý rozhovor"})

    # library folders: a small tree, an empty folder, and one transcript left unfiled
    folders = FolderStore(lib, data / "folders.json")
    folders.assign([tid, tid4], "Podcasty/Zahrada")
    folders.assign([tid2, tid5], "Spolek/Porady")
    folders.create("Klient Novák/2026")  # empty

    # watch list: plausible state, but never due, so no network access happens
    now = time.time()
    watch = [
        Source(id="a1b2c3d4", kind="folder", target=r"D:\Nahrávky\schůzky", name="Nahrávky ze schůzek", interval_s=60,
               baselined=True, next_check=now + 3600, last_check=now - 120, last_ok=now - 120,
               history=[{"time": now - 120, "ok": True, "found": 3, "queued": 0, "error": None, "manual": False, "dry_run": False}],
               items=[{"key": "file:1", "title": "porada-2026-10-02", "source": "porada.m4a", "queued": now - 86400,
                       "job": 1, "status": "done", "result": tid2, "error": None}]),
        Source(id="e5f6a7b8", kind="rss", target="https://example.org/demo/podcast.xml", name="Podcast o zahradě", interval_s=3600,
               baselined=True, next_check=now + 3000, last_check=now - 600, last_ok=now - 600,
               history=[{"time": now - 600, "ok": True, "found": 15, "queued": 1, "error": None, "manual": False, "dry_run": False}],
               items=[{"key": "guid:7", "title": "Epizoda 7: Kompost", "source": "https://example.org/demo/ep7.mp3",
                       "queued": now - 600, "job": 2, "status": "done", "result": tid, "error": None}]),
    ]
    (data / "watch.json").write_text(json.dumps({"version": 1, "sources": [s.to_dict() for s in watch]}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return {"transcripts": [tid, tid2, tid3, tid4, tid5],
            "folders": [f["path"] for f in folders.tree()["folders"]],
            "people": {p["name"]: p["id"] for p in db.list()}}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(build(Path(args[0]).resolve(), Path(args[1]).resolve()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
