"""One transcription from source to library entry, reporting structured progress.

Steps: pick engine -> preflight check -> resolve source -> library folder -> fetch audio
-> engine.transcribe -> save segments, stored exports and metadata.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from . import editor, exporters, privacy, separated, voices
from .config import Settings
from .folders import safe_path
from .engines import TranscribeOptions, get_engine
from .engines.base import EngineError
from .library import METADATA_FILE, Library, Metadata, now_iso
from .pathguard import UNC_MESSAGE, unc_blocked
from .progress import Phase, Progress, ProgressCallback, no_progress
from .sources import fetch_audio, is_url, probe, source_key


@dataclass
class TranscriptionRequest:
    source: str  # URL or local file path
    language: str | None = None  # "auto" = detect; None = settings default
    speakers: int | None = None
    diarize: bool | None = None  # None = settings default
    engine: str | None = None  # None = settings default
    model: str | None = None  # None = settings default (or the engine's own default)
    batch_size: int | None = None
    task: str = "transcribe"  # another value runs a registered task instead (see tasks.py)
    folder: str | None = None  # virtual library folder for the transcript; None = keep its folder (new: unfiled)
    translate_to: str | None = None  # translate the finished transcript into this language (a LANGUAGES key; see llm/auto_translate.py)
    separate_speakers: bool = False  # one track per voice, each transcribed alone (overlapping speech; separated.py)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionRequest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def run(request: TranscriptionRequest, settings: Settings, progress: ProgressCallback = no_progress,
        library: Library | None = None) -> str:
    """Transcribe ``request`` into the library and return the transcript id."""
    library = library or Library(settings.output_dir)
    engine_name = request.engine or settings.engine
    privacy.check_source(settings, request.source)  # offline mode: no URLs (get_engine refuses cloud engines)
    engine = get_engine(engine_name, settings)
    language = request.language or settings.language
    options = TranscribeOptions(
        language=None if language == "auto" else language,
        model=request.model or (settings.model if engine_name == settings.engine else None),
        speakers=request.speakers,
        # separated voices need no diarization (a track is one speaker), so no diarization-model access either
        diarize=(settings.diarize if request.diarize is None else request.diarize) and engine.supports_diarization
                and not request.separate_speakers,
        batch_size=request.batch_size or settings.batch_size,
    )

    progress(Progress(Phase.LOADING, message=f"Engine: {engine_name}"))
    engine.check(options)
    if request.separate_speakers:
        separated.check(settings, engine_name)  # local engines only, and the separation models must be there
    if not is_url(request.source) and unc_blocked(settings, request.source):  # whoever queued it
        raise EngineError(UNC_MESSAGE)
    media = probe(request.source)
    progress(Progress(Phase.LOADING, message=f"Title: {media.title}", data={"title": media.title}))

    # The new run works in a staging folder; the old transcript (audio, names, voices, analysis) stays
    # untouched until this run has succeeded, so a cancelled or failed re-transcription loses nothing.
    run_ = library.begin_run(media.title, source_key(media))
    folder = run_.folder
    transcript_id = folder.name
    media.workdir = run_.stage
    report = progress

    def progress(event: Progress) -> None:  # noqa: F811 - every report also tells the sweeper the stage is alive
        run_.touch()
        report(event)

    # tells the job queue where this run's stage is, so that it can remove it when the worker is killed (cancel)
    progress(Progress(Phase.LOADING, data={"stage": str(run_.stage)}))
    # a phase can be silent for hours (CPU transcription): the stage must still look alive to the sweeps of other processes
    alive = threading.Event()
    threading.Thread(target=_heartbeat, args=(run_, alive), name="stage-heartbeat", daemon=True).start()
    try:
        if engine.needs_audio:
            fetch_audio(media, run_.stage, progress)

        separation_info = None
        if request.separate_speakers:
            result, separation_info = separated.transcribe(engine, media, options, settings, progress, request.speakers)
        else:
            result = engine.transcribe(media, options, progress)

        progress(Progress(Phase.WRITING, message="Writing outputs..."))
        stored = [e.stored_as for e in exporters.list_exporters() if e.stored_as]
        # under the editor's lock: an editor save cannot overwrite this run unnoticed (its stamp check fails)
        with editor.edit_lock(library, transcript_id):
            previous = editor.backup(library, transcript_id, "retranscribe")  # edits of an earlier run stay restorable
            try:
                # new audio in; audio and names of the old run moved aside (not deleted) until all is saved
                library.commit_run(run_, snapshot=[editor.SEGMENTS_FILE, METADATA_FILE, *stored])
                if media.audio_path is not None and media.audio_path.parent == run_.stage:
                    media.audio_path = folder / media.audio_path.name  # the staged audio is in its place now
                library.save_segments(transcript_id, result.segments)
                for exporter in exporters.list_exporters():
                    if exporter.stored_as:
                        content = exporter.render(result.segments, {})
                        target = folder / exporter.stored_as
                        target.write_bytes(content) if isinstance(content, bytes) else target.write_text(content, encoding="utf-8")
                # the folder survives a re-transcription unless the request names one (read under the lock)
                folder = library.folder_of(transcript_id) if request.folder is None else safe_path(request.folder)
                library.save_metadata(transcript_id, Metadata(
                    folder=folder,
                    title=media.title,
                    source=request.source,
                    date=now_iso(),
                    duration=result.duration or max((s.end for s in result.segments), default=0),
                    language=result.language or options.language,
                    engine=engine_name,
                    model=result.model or options.model,
                    speakers=sorted({s.speaker for s in result.segments if s.speaker}),
                    extra={**({"source_info": media.info} if media.info else {}),
                           **({separated.METADATA_KEY: separation_info} if separation_info else {}),
                           # chain of custody: hashes of the source media and the outputs (docs/PRIVACY.md)
                           "custody": privacy.custody_record(library, transcript_id, [editor.SEGMENTS_FILE] + stored,
                                                             source=media.audio_path)},
                ))
                custody = library.metadata(transcript_id).extra["custody"]
                privacy.audit(library, transcript_id, "created" if previous is None else "retranscribed",
                              source=request.source, engine=engine_name, model=result.model or options.model,
                              source_sha256=(custody.get("source") or {}).get("sha256"),
                              sha256=custody["outputs"].get(editor.SEGMENTS_FILE), previous_version=previous)
            except BaseException:
                library.abort_run(run_)  # still under the lock: the old transcript is back before anyone looks
                _drop_backup(library, transcript_id, previous)
                raise
        library.finish_run(run_)
        alive.set()
    except BaseException:
        alive.set()
        library.abort_run(run_)
        raise
    # voice memory: store speaker embeddings, name known voices (never raises)
    voices.after_transcription(library, transcript_id, result.speaker_embeddings, settings)
    progress(Progress(Phase.WRITING, 100, data={"result": transcript_id}))
    return transcript_id


HEARTBEAT_S = 60.0


def _heartbeat(run_, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_S):
        run_.touch(every=0)


def _drop_backup(library: Library, transcript_id: str, version: str | None) -> None:
    """The run failed: its pre-commit copy of prepis.json is not a version worth keeping."""
    if not version:
        return
    try:
        folder = editor.versions_dir(library, transcript_id)
        (folder / version).unlink(missing_ok=True)
        folder.rmdir()  # only when empty
    except OSError:
        pass
