"""The MCP server, driven by the official MCP SDK client: in-process, over stdio and over Streamable HTTP."""

import asyncio
import json
import os
import sys

import httpx2
import numpy as np
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from mgtranscriber import __version__, voices
from mgtranscriber.config import ENV_PREFIX, PROJECT_ROOT
from mgtranscriber.jobs import InProcessRunner, JobQueue
from mgtranscriber.mcp_server import LocalJobs, build_server
from mgtranscriber.search import SearchIndex

TOOLS = {"transcribe", "job_status", "list_transcripts", "search_transcripts", "get_transcript", "rename_speakers",
         "list_speakers", "find_speaker", "list_folders"}


def run(coro):
    return asyncio.run(coro)


async def call(client, tool, **args):
    result = await client.call_tool(tool, args)
    if result.is_error:
        raise AssertionError(result.content[0].text)
    return result.structured_content


async def error_text(client, tool, **args) -> str:
    result = await client.call_tool(tool, args)
    assert result.is_error
    return result.content[0].text


@pytest.fixture
def server(settings, library, legacy_transcript):
    queue = JobQueue(settings, runner=InProcessRunner())
    index = SearchIndex(library, settings.search_file)
    library.listeners.append(index.reindex)
    queue.on_finished.append(lambda job: job.result and index.reindex(job.result))
    index.sync()
    yield build_server(settings, library, index, LocalJobs(queue), allow_local_paths=True,
                       web_url="http://127.0.0.1:8000")
    queue.stop()
    index.close()


def test_tools_and_reading(server):
    async def scenario():
        async with Client(server) as client:
            assert {t.name for t in (await client.list_tools()).tools} == TOOLS
            listing = await call(client, "list_transcripts")
            assert listing["total"] == 1 and listing["transcripts"][0]["id"] == "Old video"
            assert (await call(client, "list_transcripts", query="OLD"))["total"] == 1
            assert (await call(client, "list_transcripts", query="nothing"))["total"] == 0

            found = await call(client, "search_transcripts", q="mas")
            [hit] = found["hits"]
            assert hit["snippet"] == "jak se **máš**?" and hit["time"] == "00:00:03"
            assert hit["citation"] == "Old video @ 00:00:03"
            assert hit["url"] == "http://127.0.0.1:8000/#/t/Old%20video?t=3"

            text = await call(client, "get_transcript", transcript_id="Old video")
            assert text["text"] == "[00:00:00] SPEAKER_00: Ahoj.\n[00:00:02] SPEAKER_01: Nazdar, jak se máš?"
            part = await call(client, "get_transcript", transcript_id="Old video", format="segments", start=2.5)
            assert [s["text"] for s in part["segments"]] == ["Nazdar,", "jak se máš?"]
            assert "No transcript" in await error_text(client, "get_transcript", transcript_id="nope")
            assert "format" in await error_text(client, "get_transcript", transcript_id="Old video", format="xml")

            renamed = await call(client, "rename_speakers", transcript_id="Old video", mapping={"SPEAKER_01": "Petr"})
            assert renamed["speakers"] == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "Petr"}
            await call(client, "rename_speakers", transcript_id="Old video", mapping={"SPEAKER_00": "Jana"})
            text = await call(client, "get_transcript", transcript_id="Old video")
            assert "Jana: Ahoj." in text["text"] and "Petr: Nazdar" in text["text"]  # merged, not replaced
            assert (await call(client, "search_transcripts", q="nazdar"))["hits"][0]["speaker"] == "Petr"
            assert "Unknown speaker" in await error_text(client, "rename_speakers", transcript_id="Old video",
                                                        mapping={"SPEAKER_09": "X"})

            resources = await client.list_resources()
            assert "mgtranscriber://transcripts" in [str(r.uri) for r in resources.resources]
            content = await client.read_resource("mgtranscriber://transcripts/Old%20video")
            assert content.contents[0].text.startswith("# Old video\n\n[00:00:00] Jana: Ahoj.")

    run(scenario())


def make_episode(library, tid, parts):
    """A transcript from (label, start, end) parts, each saying 'tereza hovori'."""
    folder = library.root / tid
    folder.mkdir()
    segs = [{"start": a, "end": b, "text": f" {label} hovori o kavě.", "speaker": label, "words": []}
            for label, a, b in parts]
    (folder / "prepis.json").write_text(json.dumps(segs), encoding="utf-8")


def test_speakers_tools(server, settings, library):
    rng = np.random.default_rng(7)
    jana = rng.standard_normal(voices.DIM)
    jana /= np.linalg.norm(jana)
    voices.open_db(settings).enroll("Jana Nováková", jana, speech_s=60)
    make_episode(library, "ep one", [("SPEAKER_00", 0, 100), ("SPEAKER_01", 100, 130), ("SPEAKER_00", 135, 200)])
    other = rng.standard_normal(voices.DIM)
    voices.after_transcription(library, "ep one", {"SPEAKER_00": list(jana), "SPEAKER_01": list(other)}, settings)
    make_episode(library, "ep two", [("SPEAKER_00", 0, 90)])
    voices.after_transcription(library, "ep two", {"SPEAKER_00": list(jana)}, settings)
    make_episode(library, "ep three", [("SPEAKER_00", 0, 40)])  # matched by voice, but the user named it otherwise
    voices.after_transcription(library, "ep three", {"SPEAKER_00": list(jana)}, settings)
    library.save_names("ep three", {"SPEAKER_00": "Somebody Else"})

    async def scenario():
        async with Client(server) as client:
            people = await call(client, "list_speakers")
            assert people["enabled"] and people["total"] == 1
            [jana_row] = people["people"]
            assert jana_row["name"] == "Jana Nováková" and jana_row["transcripts"] == 2
            # speech_seconds = the sure appearances (100 + 65 s in ep one, 90 s in ep two), never the voice samples
            assert jana_row["speech_seconds"] == 255 and jana_row["voice_sample_seconds"] == 60
            assert jana_row["suggested_transcripts"] == 1 and jana_row["suggested_seconds"] == 40

            found = await call(client, "find_speaker", name="jana novakova")  # case and diacritics ignored
            assert found["name"] == "Jana Nováková" and found["total"] == 2
            first = next(a for a in found["appearances"] if a["transcript_id"] == "ep one")
            assert first["label"] == "SPEAKER_00" and first["status"] == "auto"
            assert [s["start"] for s in first["spans"]] == [0.0, 135.0]  # SPEAKER_01 in between: two spans
            assert first["spans"][0]["time"] == "00:00:00-00:01:40" and first["first_time"] == "00:00:00"
            assert first["url"] == "http://127.0.0.1:8000/#/t/ep%20one?t=0"
            assert first["span_count"] == 2 and first["speech_seconds"] == 165 and first["needs_confirmation"] is False
            assert found["suggested_total"] == 0 and found["sure_speech_seconds"] == 255
            assert found["transcripts_truncated"] is False and found["spans_truncated"] is False
            # the same numbers as list_speakers
            assert found["sure_total"] == jana_row["transcripts"]
            assert found["sure_speech_seconds"] == jana_row["speech_seconds"]

            # suggestions: an unverified guess, flagged as such, never mixed into the sure totals
            withs = await call(client, "find_speaker", name="Jana Nováková", include_suggestions=True)
            [guess] = [a for a in withs["appearances"] if a["status"] == "suggest"]
            assert guess["transcript_id"] == "ep three" and guess["needs_confirmation"] is True
            assert withs["suggested_total"] == 1 and withs["sure_total"] == 2 and withs["sure_speech_seconds"] == 255
            assert all(a["needs_confirmation"] is False for a in withs["appearances"] if a is not guess)

            # a summary without the spans
            brief = await call(client, "find_speaker", name="Jana Nováková", include_spans=False)
            assert all("spans" not in a and a["span_count"] >= 1 and a["first_time"] for a in brief["appearances"])
            by_id = await call(client, "find_speaker", person_id=people["people"][0]["person_id"], limit=1)
            assert by_id["total"] == 2 and len(by_id["appearances"]) == 1

            # the voice-memory name is a display name now: search by speaker, with and without a text
            hits = await call(client, "search_transcripts", q="kavě", speaker="Jana Nováková")
            assert hits["total"] == 3 and {h["speaker"] for h in hits["hits"]} == {"Jana Nováková"}
            assert (await call(client, "search_transcripts", speaker="jana novakova"))["total"] == 3
            assert (await call(client, "search_transcripts", q="kavě", speaker="SPEAKER_01"))["total"] == 1

            # an unknown speaker is an error with a pointer, not a silent 0
            err = await error_text(client, "search_transcripts", q="kavě", speaker="Nobody Atall")
            assert "Unknown speaker" in err and "list_speakers" in err
            near = await call(client, "search_transcripts", q="kavě", speaker="Somebody")  # unique close name
            assert near["speaker_matched"] == "Somebody Else" and near["total"] == 1

            assert "exactly one" in await error_text(client, "find_speaker")
            assert "exactly one" in await error_text(client, "find_speaker", name="a", person_id="b")
            assert "No person named" in await error_text(client, "find_speaker", name="Nobody")
            assert "No person with id" in await error_text(client, "find_speaker", person_id="zzz")
            assert "Give a search text" in await error_text(client, "search_transcripts")

    run(scenario())


def test_find_speaker_output_is_capped(server, settings, library):
    from mgtranscriber.mcp_server import MAX_CHARS, MAX_FIND, MAX_SPANS, MAX_SPANS_TOTAL

    rng = np.random.default_rng(3)
    jana = rng.standard_normal(voices.DIM)
    jana /= np.linalg.norm(jana)
    voices.open_db(settings).enroll("Jana Nováková", jana, speech_s=60)
    for n in range(40):  # 40 transcripts, 60 separate spans each
        parts = [(("SPEAKER_00", "SPEAKER_01")[i % 2], i * 10, i * 10 + 9) for i in range(120)]
        make_episode(library, f"long {n:02d}", parts)
        voices.after_transcription(library, f"long {n:02d}", {"SPEAKER_00": list(jana),
                                                              "SPEAKER_01": list(rng.standard_normal(voices.DIM))}, settings)

    async def scenario():
        async with Client(server) as client:
            found = await call(client, "find_speaker", name="Jana Nováková", limit=1000)
            assert found["total"] == 40 and found["transcripts_truncated"] is True and found["spans_truncated"] is True
            assert found["returned"] == len(found["appearances"]) <= MAX_FIND
            spans = [len(a["spans"]) for a in found["appearances"]]
            assert max(spans) <= MAX_SPANS and sum(spans) <= MAX_SPANS_TOTAL
            assert all(a["span_count"] == 60 for a in found["appearances"])  # the real count is still reported
            assert len(json.dumps(found, ensure_ascii=False)) < 2 * MAX_CHARS
            small = await call(client, "find_speaker", name="Jana Nováková", limit=2)
            assert small["returned"] == 2 and small["transcripts_truncated"] is True
            brief = await call(client, "find_speaker", name="Jana Nováková", limit=2, include_spans=False)
            assert brief["spans_truncated"] is False and brief["transcripts_truncated"] is True

    run(scenario())


def test_speakers_tools_with_voice_memory_off(server, settings, monkeypatch):
    monkeypatch.setenv("MGTRANSCRIBER_VOICE_MEMORY", "0")

    async def scenario():
        async with Client(server) as client:
            assert await call(client, "list_speakers") == {"enabled": False, "total": 0, "people": []}
            assert "switched off" in await error_text(client, "find_speaker", name="Jana")

    run(scenario())


def test_server_json():
    """The MCP Registry file: the owner namespace, the released version, a stdio PyPI package."""
    data = json.loads((PROJECT_ROOT / "server.json").read_text(encoding="utf-8"))
    assert data["name"] == "io.github.michal1609/mgtranscriber"
    assert data["version"] == __version__ and len(data["description"]) <= 100
    [package] = data["packages"]
    assert (package["registryType"], package["identifier"], package["version"]) == ("pypi", "mgtranscriber", __version__)
    assert package["transport"] == {"type": "stdio"} and package["packageArguments"][0]["value"] == "mcp"


def test_pagination(server, library):
    from mgtranscriber.models import Segment

    library.save_segments("Old video", [Segment(i * 10.0, i * 10.0 + 9, f" Sentence number {i} " + "x" * 80,
                                                f"SPEAKER_0{i % 2}") for i in range(60)])

    async def scenario():
        async with Client(server) as client:
            pages, start = [], None
            while True:
                page = await call(client, "get_transcript", transcript_id="Old video", format="segments",
                                  start=start, max_chars=1000)
                pages.append(page["segments"])
                if not page["truncated"]:
                    break
                start = page["next_start"]
            texts = [s["text"] for p in pages for s in p]
            assert len(pages) > 3 and len(texts) == 60 and texts[59].startswith("Sentence number 59")

    run(scenario())


def test_search_hint_for_multi_word_misses(server):
    async def scenario():
        async with Client(server) as client:
            miss = await call(client, "search_transcripts", q="ahoj kavarna sova")
            assert miss["total"] == 0 and "1-2" in miss["hint"]
            assert "hint" not in await call(client, "search_transcripts", q="kavarna")  # one word: nothing to loosen
            assert "hint" not in await call(client, "search_transcripts", q="ahoj")

    run(scenario())


def test_get_transcript_argument_checks(server):
    async def scenario():
        async with Client(server) as client:
            assert "before end" in await error_text(client, "get_transcript", transcript_id="Old video", start=5, end=2)
            assert "after the end" in await error_text(client, "get_transcript", transcript_id="Old video", start=999)
            assert "max_chars" in await error_text(client, "get_transcript", transcript_id="Old video", max_chars=0)
            assert "negative" in await error_text(client, "get_transcript", transcript_id="Old video", start=-1)
            small = await call(client, "get_transcript", transcript_id="Old video", max_chars=10)
            assert "1000" in small["warnings"][0]
            body = await call(client, "get_transcript", transcript_id="Old video")
            assert body["translation_lang"] is None and "language" not in body and "warnings" not in body
            assert "transcript_language" in body

    run(scenario())


def test_transcribe_checks_language_and_existing_source(server, media_file):
    async def scenario():
        async with Client(server) as client:
            path = str(media_file("again.wav"))
            assert "Invalid language" in await error_text(client, "transcribe", url_or_path=path, language="czech")
            assert "Invalid language" in await error_text(client, "transcribe", url_or_path="https://x.example/v",
                                                          language="e")
            first = await call(client, "transcribe", url_or_path=path, wait=True)
            assert first["status"] == "done" and "warning" not in first
            tid = first["transcript_id"]

            err = await error_text(client, "transcribe", url_or_path=path, replace=False)
            assert tid in err and "already transcribed" in err
            second = await call(client, "transcribe", url_or_path=path, language="en")  # replace defaults to true
            assert second["existing_transcript_id"] == tid and "REPLACES" in second["warning"]

    run(scenario())


def test_wait_limit_leaves_the_job_running(server, media_file, monkeypatch):
    from mgtranscriber import mcp_server

    monkeypatch.setattr(mcp_server, "WAIT_LIMIT", 0.0)

    async def scenario():
        async with Client(server) as client:
            job = await call(client, "transcribe", url_or_path=str(media_file("late.wav")), wait=True)
            assert job["status"] in ("queued", "running") and job["wait_timed_out"] is True
            assert "job_status" in job["hint"]

    run(scenario())


def test_transcribe_job(server, media_file):
    async def scenario():
        async with Client(server) as client:
            job = await call(client, "transcribe", url_or_path=str(media_file("mcp talk.wav")), language="en")
            assert job["status"] in ("queued", "running", "done") and job["job_id"] == 1
            for _ in range(100):
                status = await call(client, "job_status", job_id=job["job_id"])
                if status["status"] == "done":
                    break
                await asyncio.sleep(0.1)
            assert status["transcript_id"].startswith("mcp-talk-")
            assert (await call(client, "search_transcripts", q="hello"))["hits"][0]["transcript_id"] == status["transcript_id"]

            progress = []

            async def on_progress(value, total, message):
                progress.append((value, message))

            waited = await client.call_tool("transcribe", {"url_or_path": str(media_file("slow.wav")), "wait": True},
                                            progress_callback=on_progress)
            assert waited.structured_content["status"] == "done"
            assert progress and progress[-1][0] > 15  # the transcribe phase starts at 15 %

            assert "File not found" in await error_text(client, "transcribe", url_or_path="C:/no/such/file.wav")
            assert "No job" in await error_text(client, "job_status", job_id=999)

    run(scenario())


def test_local_paths_can_be_refused(settings, library, legacy_transcript, media_file):
    queue = JobQueue(settings, runner=InProcessRunner(), autostart=False)
    index = SearchIndex(library, settings.search_file)
    server = build_server(settings, library, index, LocalJobs(queue), allow_local_paths=False)

    async def scenario():
        async with Client(server) as client:
            assert "give a URL" in await error_text(client, "transcribe", url_or_path=str(media_file()))

    run(scenario())
    index.close()


def stdio_params(settings, *args) -> StdioServerParameters:
    env = {**os.environ, ENV_PREFIX + "OUTPUT_DIR": str(settings.output_dir),
           ENV_PREFIX + "DATA_DIR": str(settings.data_dir), ENV_PREFIX + "ENGINE": "mock",
           ENV_PREFIX + "PORT": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    return StdioServerParameters(command=sys.executable, args=["-m", "mgtranscriber", "mcp", *args], env=env,
                                 cwd=str(PROJECT_ROOT))


def test_stdio_transport(settings, legacy_transcript, media_file):
    """A real ``mgtranscriber mcp`` subprocess with its own job queue (no web server is listening on port 1)."""

    async def scenario():
        async with Client(stdio_params(settings), read_timeout_seconds=60) as client:
            assert {t.name for t in (await client.list_tools()).tools} == TOOLS
            assert (await call(client, "search_transcripts", q="ahoj"))["total"] == 1
            job = await call(client, "transcribe", url_or_path=str(media_file("stdio.wav")), wait=True)
            assert job["status"] == "done", job
            text = await call(client, "get_transcript", transcript_id=job["transcript_id"])
            assert "Hello there." in text["text"]

    run(scenario())
    assert (settings.data_dir / "mcp-jobs.json").is_file()  # its own store, never the web server's jobs.json


def test_stdio_delegates_jobs_to_a_running_web_server(settings, legacy_transcript, media_file, live_server):
    url = live_server()

    async def scenario():
        async with Client(stdio_params(settings, "--server", url), read_timeout_seconds=60) as client:
            job = await call(client, "transcribe", url_or_path=str(media_file("via server.wav")), wait=True)
            assert job["status"] == "done", job

    run(scenario())
    jobs = json.loads((settings.data_dir / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    assert [j["request"]["source"].endswith("via server.wav") for j in jobs] == [True]
    assert not (settings.data_dir / "mcp-jobs.json").exists()


def test_streamable_http_transport(live_server, media_file):
    url = live_server(api_key="k2")

    async def scenario():
        headers = {"Authorization": "Bearer k2"}
        async with httpx2.AsyncClient(headers=headers, timeout=60) as http:
            async with Client(streamable_http_client(f"{url}/mcp", http_client=http)) as client:
                assert {t.name for t in (await client.list_tools()).tools} == TOOLS
                job = await call(client, "transcribe", url_or_path=str(media_file("http.wav")), wait=True)
                assert job["status"] == "done"
                assert (await call(client, "search_transcripts", q="fine"))["total"] >= 1
        async with httpx2.AsyncClient(timeout=10) as http:
            assert (await http.post(f"{url}/mcp", json={})).status_code == 401

    run(scenario())
