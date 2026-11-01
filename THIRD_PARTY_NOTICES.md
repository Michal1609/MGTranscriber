# Third-party notices

MGTranscriber's own code is licensed under the [Apache License 2.0](LICENSE). It uses third-party
software and machine-learning models under their own licenses, summarised below. Nothing here is
legal advice; the linked license texts are authoritative.

## What is distributed where

| distribution | contains third-party code? |
|---|---|
| source ZIP, git checkout, setup `.exe` | **No.** Only MGTranscriber's code. Python packages are downloaded by `uv` from PyPI and download.pytorch.org on your computer; uv, ffmpeg, Deno and Ollama are installed by `install.ps1` from their official sources. |
| Docker image | **Yes.** Ubuntu 24.04 packages (including FFmpeg), Python, Deno, tini and all Python packages listed in `uv.lock`, including NVIDIA CUDA libraries (`nvidia-*` wheels, Linux CUDA build). |
| models | **Never bundled.** Downloaded at first use from Hugging Face into `data/models` or the Hugging Face cache, under the model's own license. |

## Models (downloaded at runtime)

| model | used by | license | notes |
|---|---|---|---|
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | speaker diarization, voice memory | CC-BY-4.0 | **Gated:** each user logs in to Hugging Face, accepts the conditions (sharing contact details with pyannote) and uses their own token. |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (part of the pipeline) | speaker embeddings | CC-BY-4.0 | |
| [pyannote/speech-separation-ami-1.0](https://huggingface.co/pyannote/speech-separation-ami-1.0) and its network [pyannote/separation-ami-1.0](https://huggingface.co/pyannote/separation-ami-1.0) | speech separation (overlapping speakers, optional) | MIT | **Gated:** both repositories need their conditions accepted with the same Hugging Face account as the token. |
| [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | speaker embedding inside the separation pipeline | Apache-2.0 | |
| Whisper large-v3 / large-v3-turbo etc. ([Systran/faster-whisper-*](https://huggingface.co/Systran)) | `whisperx` engine | MIT | OpenAI Whisper weights, converted for CTranslate2 |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (ONNX export via onnx-asr) | `parakeet` engine | CC-BY-4.0 | |
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) (ONNX export via onnx-asr) | `canary` engine | CC-BY-4.0 | |
| wav2vec2 alignment models chosen by WhisperX per language, e.g. torchaudio `WAV2VEC2_ASR_BASE_960H` (en), [comodoro/wav2vec2-xls-r-300m-cs-250](https://huggingface.co/comodoro/wav2vec2-xls-r-300m-cs-250) (cs) | word timestamps | Apache-2.0 (per model card) | other languages: see the model card of the model WhisperX downloads |
| [Silero VAD](https://github.com/snakers4/silero-vad) | voice activity detection | MIT | |
| [gemma3](https://ollama.com/library/gemma3), [bge-m3](https://ollama.com/library/bge-m3) (optional, via Ollama) | summaries, Q&A | Gemma Terms of Use; MIT | pulled by the user into Ollama |

### Attribution for CC-BY-4.0 models

> Speaker diarization uses **pyannote speaker-diarization-community-1** by pyannoteAI / Hervé Bredin
> (CC-BY-4.0, https://huggingface.co/pyannote/speaker-diarization-community-1), based on the pyannote
> segmentation model, WeSpeaker ResNet34 embeddings and VBx clustering.
>
> Speech recognition can use **NVIDIA Parakeet-TDT-0.6B-v3** and **NVIDIA Canary-1B-v2**
> (© NVIDIA, CC-BY-4.0, https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3,
> https://huggingface.co/nvidia/canary-1b-v2), run as ONNX exports through onnx-asr.
>
> Models were used unmodified. CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/

If you publish transcripts or research made with these models, the model cards ask you to cite the
corresponding papers.

## Main software dependencies

| component | license |
|---|---|
| [WhisperX](https://github.com/m-bain/whisperX) | BSD-2-Clause |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) and pyannote-* packages | MIT |
| [PyTorch](https://github.com/pytorch/pytorch), torchaudio, torchvision, torchcodec | BSD-3-Clause |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) (speaker embedding of the speech separation model) | Apache-2.0 |
| [Asteroid](https://github.com/asteroid-team/asteroid) 0.5.2: the DPRNN network and `pad_x_to_y` are vendored in `mgtranscriber/_vendor/asteroid_min.py` (needed by the speech separation model; the package itself pins an incompatible pytorch-lightning) | MIT, Copyright (c) 2019 Pariente Manuel |
| [onnx-asr](https://github.com/istupakov/onnx-asr) | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (onnxruntime-gpu) | MIT |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp), yt-dlp-ejs | Unlicense (yt-dlp-ejs: Unlicense AND MIT AND ISC) |
| [mutagen](https://github.com/quodlibet/mutagen) (optional dependency of yt-dlp) | **GPL-2.0-or-later**; MGTranscriber does not import it |
| [FastAPI](https://github.com/fastapi/fastapi), Starlette, Uvicorn | MIT / BSD-3-Clause |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | MIT |
| anthropic, google-genai SDKs (optional cloud providers) | MIT, Apache-2.0 |
| python-docx | MIT |
| [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause; its wheels bundle FFmpeg libraries under LGPL-2.1+ |
| [FFmpeg](https://ffmpeg.org/legal.html) (installed separately; in the Docker image from Ubuntu) | LGPL-2.1+ / GPL-2.0+ depending on the build. Ubuntu's build is GPL; its source is available from Ubuntu (`apt-get source ffmpeg`). |
| [Deno](https://github.com/denoland/deno) | MIT |
| [uv](https://github.com/astral-sh/uv) | MIT OR Apache-2.0 |
| [tini](https://github.com/krallin/tini) (Docker image) | MIT |
| NVIDIA CUDA libraries (`nvidia-*` wheels in the Linux CUDA build / Docker image) | NVIDIA Software License Agreement / CUDA EULA (redistributable components) |
| Python | PSF-2.0 |

The full list of Python packages with exact versions is in `uv.lock`. To print the license of every
installed package: `uv run python -c "import importlib.metadata as m; [print(d.metadata['Name'], d.metadata.get('License-Expression') or d.metadata.get('License')) for d in m.distributions()]"`.

## Trademarks

Whisper is a trademark of OpenAI; NVIDIA, Parakeet and Canary are trademarks of NVIDIA; YouTube is a
trademark of Google. MGTranscriber is not affiliated with or endorsed by them.
