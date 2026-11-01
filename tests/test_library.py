import json

import pytest

from mgtranscriber.config import load_settings
from mgtranscriber.library import InvalidTranscriptId, Metadata, TranscriptNotFound, slugify, transcript_id_for


def test_legacy_folder_without_metadata(library, legacy_transcript):
    meta = library.metadata(legacy_transcript)
    assert meta.title == "Old video"
    assert meta.duration == 4.25
    assert meta.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert meta.date

    [item] = library.list()
    assert item["id"] == "Old video"
    assert item["speakers"] == {"SPEAKER_00": 1.5, "SPEAKER_01": 2.25}
    assert item["words"] == 5
    assert item["has_audio"] and item["audio"] == "audio.webm"


def test_metadata_roundtrip_and_cache_invalidation(library, legacy_transcript):
    library.summary(legacy_transcript)
    library.save_metadata(legacy_transcript, Metadata(title="Nice title", language="cs", engine="whisperx",
                                                      model="large-v3", date="2026-01-02T03:04:05+00:00",
                                                      extra={"tags": ["a"]}))
    meta = library.metadata(legacy_transcript)
    assert meta.title == "Nice title" and meta.extra == {"tags": ["a"]}
    summary = library.summary(legacy_transcript)
    assert summary["title"] == "Nice title"
    assert summary["language"] == "cs"


def test_unknown_metadata_keys_are_kept_in_extra(library, legacy_transcript):
    path = library.path(legacy_transcript) / "metadata.json"
    path.write_text(json.dumps({"title": "T", "summary": "from a future feature"}), encoding="utf-8")
    assert library.metadata(legacy_transcript).extra == {"summary": "from a future feature"}


def test_speaker_names(library, legacy_transcript):
    saved = library.save_names(legacy_transcript, {"SPEAKER_00": "  Alice ", "SPEAKER_01": "", "../evil": "x"})
    assert saved == {"SPEAKER_00": "Alice"}
    assert library.names(legacy_transcript) == {"SPEAKER_00": "Alice"}


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "C:x", "..\\output"])
def test_rejects_ids_outside_library(library, bad):
    with pytest.raises(InvalidTranscriptId):
        library.path(bad)


def test_missing_transcript(library):
    with pytest.raises(TranscriptNotFound):
        library.path("nope")


def test_folder_without_segments_is_ignored(library, legacy_transcript):
    (library.root / "half-done").mkdir()
    assert [t["id"] for t in library.list()] == [legacy_transcript]


def test_ids_are_unique_per_source_and_stable():
    assert slugify('Přepis: "Díl 1"?') == "prepis-dil-1"
    assert slugify("...") == "transcript"
    a, b = transcript_id_for("Same title", "file:/a/x.wav"), transcript_id_for("Same title", "file:/b/x.wav")
    assert a != b and a.startswith("same-title-")
    assert transcript_id_for("Renamed title", "Youtube:abc")[-8:] == transcript_id_for("Other", "Youtube:abc")[-8:]


def test_same_title_from_two_sources_does_not_overwrite(library):
    first = library.create("Talk", "file:/one/talk.wav")
    second = library.create("Talk", "file:/two/talk.wav")
    assert first != second


def test_retranscription_replaces_audio_and_names_only_on_commit(library):
    folder = library.create("Talk", "Youtube:abc")
    (folder / "audio.webm").write_bytes(b"old")
    (folder / "prepis.json").write_text("[]", encoding="utf-8")
    library.save_names(folder.name, {"SPEAKER_00": "Old name"})
    assert library.create("Talk", "Youtube:abc") == folder  # creating the folder again changes nothing
    assert (folder / "audio.webm").read_bytes() == b"old" and library.names(folder.name) == {"SPEAKER_00": "Old name"}

    run = library.begin_run("Talk", "Youtube:abc")
    assert run.folder == folder and run.existed
    (run.stage / "audio.webm").write_bytes(b"new")
    library.abort_run(run)  # failed: everything as it was
    assert (folder / "audio.webm").read_bytes() == b"old" and library.names(folder.name) == {"SPEAKER_00": "Old name"}
    assert not run.stage.exists()

    run = library.begin_run("Talk", "Youtube:abc")
    (run.stage / "audio.webm").write_bytes(b"new")
    library.commit_run(run)
    assert (folder / "audio.webm").read_bytes() == b"new" and library.names(folder.name) == {}
    assert (run.stage / "old" / "audio.webm").read_bytes() == b"old"  # kept until everything is saved
    library.finish_run(run)
    assert not run.stage.exists()


def test_legacy_unknown_speaker_key(library, legacy_transcript):
    (library.path(legacy_transcript) / "speakers.json").write_text(json.dumps({"NEZNÁMÝ": "Moderátor"}), encoding="utf-8")
    assert library.names(legacy_transcript) == {"UNKNOWN": "Moderátor"}


def test_settings_from_env_and_file(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"model": "small", "language": "en", "feature_x": 1}))
    settings = load_settings(tmp_path, env={"MGTRANSCRIBER_OUTPUT_DIR": "lib", "MGTRANSCRIBER_DIARIZE": "false",
                                            "MGTRANSCRIBER_PORT": "9000", "HF_TOKEN": "t"})
    assert settings.output_dir == tmp_path / "lib"
    assert settings.model == "small" and settings.language == "en"
    assert settings.diarize is False and settings.port == 9000 and settings.hf_token == "t"
    assert settings.extra == {"feature_x": 1}


def test_same_source_with_a_new_title_reuses_its_folder(library):
    from mgtranscriber.library import Metadata

    folder = library.create("Old title", "Youtube:abc")
    (folder / "prepis.json").write_text("[]", encoding="utf-8")
    library.save_metadata(folder.name, Metadata(title="Old title"))
    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["source_key"] == "Youtube:abc"
    assert library.create("Renamed video", "Youtube:abc") == folder
    assert library.create("Old title", "Youtube:other") != folder


def test_folders_from_before_source_keys_are_found(library, tmp_path):
    """Folders named after the title (round-1 layout) carry the source in metadata.json."""
    video = library.root / "Some video title"
    video.mkdir()
    (video / "prepis.json").write_text("[]", encoding="utf-8")
    (video / "metadata.json").write_text(json.dumps({"title": "Some video title", "source": "https://youtu.be/abc",
                                                     "extra": {"source_info": {"id": "abc", "extractor_key": "Youtube"}}}))
    assert library.create("Some video title (updated)", "Youtube:abc") == video

    media = tmp_path / "talk.wav"
    media.write_bytes(b"RIFF")
    local = library.root / "talk"
    local.mkdir()
    (local / "prepis.json").write_text("[]", encoding="utf-8")
    (local / "metadata.json").write_text(json.dumps({"title": "talk", "source": str(media)}))
    from mgtranscriber.sources import Media, source_key

    assert library.create("talk", source_key(Media(str(media), "file", "talk", audio_path=media))) == local


def test_segments_stamp_tracks_transcript_changes(library, legacy_transcript):
    before = library.segments_stamp(legacy_transcript)
    assert before == library.segments_stamp(legacy_transcript)
    segments = library.segments(legacy_transcript)
    segments[0].text = " Changed."
    library.save_segments(legacy_transcript, segments)
    assert library.segments_stamp(legacy_transcript) != before
