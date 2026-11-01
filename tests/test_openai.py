"""The OpenAI-compatible API, exercised with the official ``openai`` Python client against a live server."""

import pytest

openai = pytest.importorskip("openai")

from mgtranscriber.library import Library  # noqa: E402


@pytest.fixture
def server(live_server):
    return live_server()


@pytest.fixture
def client(server):
    with openai.OpenAI(base_url=f"{server}/v1", api_key="unused", max_retries=0, timeout=60) as c:
        yield c


def upload(media_file, name="talk.wav"):
    return open(media_file(name), "rb")


def test_models(client):
    ids = [m.id for m in client.models.list()]
    assert ids[0] == "whisper-1" and "large-v3" in ids and "mock" in ids
    assert client.models.retrieve("whisper-1").id == "whisper-1"
    with pytest.raises(openai.NotFoundError):
        client.models.retrieve("dall-e-3")


def test_open_webui_default_model_is_accepted(client, media_file):
    """Open WebUI's current default STT model name."""
    with upload(media_file) as f:
        assert client.audio.transcriptions.create(model="gpt-transcribe", file=f).text.startswith("Hello there.")


def test_json_and_text(client, media_file):
    with upload(media_file) as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    assert result.text == "Hello there. Hi, how are you? Fine."
    with upload(media_file) as f:
        text = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
    assert text.strip() == "Hello there. Hi, how are you? Fine."


def test_verbose_json_with_words_and_speakers(client, media_file, settings):
    with upload(media_file, "dialog.wav") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en", response_format="verbose_json",
            timestamp_granularities=["word", "segment"], extra_body={"diarize": True, "num_speakers": 2})
    assert result.language == "english" and result.duration == 5.0
    assert [s.start for s in result.segments] == [0.5, 2.5, 4.2]
    assert result.segments[1].model_extra["speaker"] == "SPEAKER_01"
    assert [w.word for w in result.words][:2] == ["Hello", "there."]
    assert result.words[0].model_extra["speaker"] == "SPEAKER_00"
    assert all(w.start is not None and w.end >= w.start for w in result.words)
    # the transcript is in the library, with the upload kept as its audio and the original file name as source
    library = Library(settings.output_dir)
    transcript_id = result.model_extra["transcript_id"]
    assert library.metadata(transcript_id).source == "dialog.wav"
    assert library.audio_file(transcript_id).name == "audio.wav"
    assert not list((settings.data_dir / "uploads").iterdir())


def test_diarization_is_off_by_default(client, media_file):
    with upload(media_file) as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json")
    assert all(s.model_extra["speaker"] is None for s in result.segments)
    assert result.words is None  # only with timestamp_granularities=["word"]


def test_srt_and_vtt(client, media_file):
    with upload(media_file) as f:
        srt = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="srt")
    assert srt.startswith("1\n00:00:00,500 --> 00:00:02,000\nHello there.")
    with upload(media_file) as f:
        vtt = client.audio.transcriptions.create(model="mock", file=f, response_format="vtt",
                                                 extra_body={"diarize": "true"})
    assert vtt.startswith("WEBVTT\n\n00:00:00.500 --> 00:00:02.000\n<v SPEAKER_00>Hello there.")


def test_errors_in_openai_format(client, media_file):
    with upload(media_file) as f, pytest.raises(openai.BadRequestError) as e:
        client.audio.transcriptions.create(model="whisper-1", file=f, response_format="docx")
    assert e.value.body["param"] == "response_format"
    with upload(media_file) as f, pytest.raises(openai.NotFoundError) as e:
        client.audio.transcriptions.create(model="gpt-5", file=f)
    assert e.value.body["code"] == "model_not_found"
    with upload(media_file, "fail.wav") as f, pytest.raises(openai.InternalServerError) as e:
        client.audio.transcriptions.create(model="whisper-1", file=f)
    assert "mock failure" in e.value.body["message"]
    with upload(media_file) as f, pytest.raises(openai.BadRequestError):
        client.audio.transcriptions.create(model="whisper-1", file=f, extra_body={"num_speakers": 99})


def test_api_key(live_server, media_file):
    url = live_server(api_key="k1")
    with openai.OpenAI(base_url=f"{url}/v1", api_key="wrong", max_retries=0) as c, pytest.raises(openai.AuthenticationError):
        c.models.list()
    with openai.OpenAI(base_url=f"{url}/v1", api_key="k1", max_retries=0) as c, upload(media_file) as f:
        assert c.audio.transcriptions.create(model="whisper-1", file=f).text.startswith("Hello")


def test_jobs_are_queued_one_at_a_time(server, client, media_file):
    """Two parallel requests are served by the job queue one after the other."""
    import json
    import threading
    import urllib.request

    results = []

    def call(name):
        with upload(media_file, name) as f:
            results.append(client.audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json"))

    threads = [threading.Thread(target=call, args=(n,)) for n in ("a slow.wav", "b slow.wav")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert len(results) == 2
    with urllib.request.urlopen(f"{server}/api/jobs") as r:
        jobs = json.loads(r.read())
    spans = sorted((j["started"], j["finished"]) for j in jobs)
    assert spans[0][1] <= spans[1][0]  # no overlap
