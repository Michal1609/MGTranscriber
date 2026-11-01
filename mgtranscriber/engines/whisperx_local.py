"""WhisperX on the local GPU: faster-whisper transcription, wav2vec2 word alignment, pyannote diarization."""

from __future__ import annotations

from ..models import Segment, TranscriptionResult
from ..progress import Phase, Progress, ProgressCallback
from ..sources import Media
from . import common
from .base import Engine, TranscribeOptions

DIARIZATION_MODEL = common.DIARIZATION_MODEL
SAMPLE_RATE = common.SAMPLE_RATE


class WhisperXEngine(Engine):
    name = "whisperx"
    supports_diarization = True
    default_model = "large-v3"

    def check(self, options: TranscribeOptions) -> None:
        if options.diarize:
            common.check_diarization_access(self.settings)

    def transcribe(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        import torch
        import whisperx

        device = common.device()
        compute_type = "float16" if device == "cuda" else "int8"
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        model_name = options.model or self.default_model
        progress(Progress(Phase.LOADING, message=f"Device: {gpu}"))

        audio, original = common.load_audio_pair(media.audio_path, self.settings.level_audio)
        duration = len(audio) / SAMPLE_RATE

        progress(Progress(Phase.TRANSCRIBE, 0, f"Transcribing ({model_name})..."))
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=options.language)
        result = model.transcribe(audio, batch_size=options.batch_size,
                                  progress_callback=lambda pct: progress(Progress(Phase.TRANSCRIBE, pct)))
        language = result["language"]
        del model
        common.free_gpu()

        segments = common.align(result["segments"], audio, language, progress)
        voices: dict = {}
        if options.diarize:
            segments = common.diarize(segments, original, self.settings, options.speakers, progress, voices)

        return TranscriptionResult([Segment.from_dict(s) for s in segments], language=language, duration=duration,
                                   model=model_name, speaker_embeddings=voices or None)
