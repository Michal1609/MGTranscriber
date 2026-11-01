# Benchmark: English

Measured on **2026-10-10** on a laptop with an **NVIDIA GeForce RTX 5070 Laptop GPU (8 GB)**, Windows 11, Python 3.12,
WhisperX 3.8.6 (faster-whisper 1.2.1), onnx-asr 0.12.0 / ONNX Runtime 1.24.4 (CUDA 12), PyTorch 2.8.0+cu128,
pyannote.audio 4.0.7 (`speaker-diarization-community-1`). Script: [`benchmarks/bench_cs.py`](../benchmarks/bench_cs.py)
(`--dataset fleurs-en | librispeech-clean | librispeech-other | ami`) and
[`benchmarks/longform.py`](../benchmarks/longform.py). The metrics and per-utterance error counts are in
[`benchmarks/results/`](../benchmarks/results/) (the reference and hypothesis text is not published); the tables below can be regenerated with
`uv run python -m benchmarks.bench_cs report benchmarks/results/*-ami-*.json` (or any other group of result files).

All numbers are **through the application** (`Engine.transcribe`, the path a user gets), one model per process,
one run per model. All engines are local; nothing was sent to the cloud.

## Summary

| | Whisper large-v3 | Whisper large-v3-turbo | Parakeet-TDT-0.6B-v3 | Canary-1B-v2 |
|---|---|---|---|---|
| LibriSpeech test-clean, WER % | 1.97 | **1.85** | 3.05 | 2.27 |
| LibriSpeech test-other, WER % | **3.75** | 4.05 | 4.12 | 4.05 |
| FLEURS en_us, WER % | 6.36 | 7.30 | 6.56 | **6.34** |
| AMI meetings (86 min), WER % | 25.6 | 25.7 | **22.4** | 26.2 |
| AMI meetings, DER % (collar 0.25 s) | 31.5 | 30.8 | **25.0** | 30.7 |
| AMI, whole transcript incl. speakers | 7.3 min | 5.3 min | **4.8 min** | 7.1 min |

What it says, in short: on clean read English the four engines are within about one point of each other; on a real
meeting Parakeet was the most accurate and the fastest in this test; the speaker part (who speaks when) is accurate
for all, and the DER is dominated by speech that was not transcribed, not by speakers mixed up. Details, intervals
and limits below.

## Method

- **Read speech (FLEURS, LibriSpeech):** 300 utterances per set. The utterances are joined into one recording (1.5 s of
  silence between them) and transcribed by `Engine.transcribe` without speaker detection; words are assigned back to
  the utterances by their start time. RTF includes loading the model and alignment, i.e. the time a user waits.
- **Meetings (AMI):** four whole meetings, each transcribed **with speaker detection** and an unknown number of
  speakers. The time is the whole transcript (recognition, alignment, speakers, model loading).
- **Language:** Whisper and Canary are given `en`; Parakeet detects the language itself.
- **Text normalization**, applied to reference and hypothesis alike: Whisper's `EnglishTextNormalizer` (with its
  British-to-American spelling table). It makes "twenty twenty-three" and "2023", "don't" and "do not" equal,
  and drops fillers (uh, um, mm, hmm). Without it Parakeet and Canary, which write digits and punctuation,
  would be penalized for format rather than recognition. Numbers are therefore not a separate column here.
- **WER / CER** are corpus rates (errors / reference words). Intervals are 95% bootstrap intervals over utterances
  (AMI: over 60-second windows), 2 000 samples. Δ is a paired bootstrap of the WER difference against the best run
  on the same items.
- **DER** (pyannote.metrics 4.1) with a 0.25 s collar around reference boundaries, overlapped speech included,
  computed from the speakers and times of **the words in the application's output**, not from raw pyannote output.
  It therefore also counts speech that the recognizer did not transcribe as missed speech. Pooled over the four
  meetings (error time / reference speech time). "No overlap" DER scores only the time with one reference speaker.
- **Speed:** RTF = processing time / audio length; "×realtime" is 1 / RTF. Read-speech runs use one process per
  model on a laptop that also ran other processes, so differences of a few ×realtime are not conclusive.

## Read speech

### LibriSpeech test-clean (300 utterances, 6 336 words; fixed random subset, at most 10 utterances per speaker)

| run | WER % (95% CI) | errors / words | CER % | Δ WER vs best (95% CI) | RTF | speed |
|---|---|---|---|---|---|---|
| Whisper large-v3-turbo | **1.85** (1.4–2.3) | 117 / 6 336 | 0.65 | – | 0.044 | 22× |
| Whisper large-v3 | 1.97 (1.5–2.5) | 125 / 6 336 | **0.64** | +0.13 (−0.19…+0.54) | 0.039 | 26× |
| Canary-1B-v2 | 2.27 (1.8–2.8) | 144 / 6 336 | 0.85 | +0.43 (+0.08…+0.78) | 0.057 | 17× |
| Parakeet-TDT-0.6B-v3 | 3.05 (1.9–4.5) | 193 / 6 336 | 1.71 | +1.20 (+0.29…+2.44) | **0.030** | **33×** |

### LibriSpeech test-other (300 utterances, 5 783 words; same selection rule)

| run | WER % (95% CI) | errors / words | CER % | Δ WER vs best (95% CI) | RTF | speed |
|---|---|---|---|---|---|---|
| Whisper large-v3 | **3.75** (3.0–4.6) | 217 / 5 783 | **1.55** | – | 0.040 | 25× |
| Canary-1B-v2 | 4.05 (3.4–4.8) | 234 / 5 783 | 1.76 | +0.29 (−0.36…+0.88) | 0.099 | 10× |
| Whisper large-v3-turbo | 4.05 (3.4–4.8) | 234 / 5 783 | 1.77 | +0.29 (−0.30…+0.84) | 0.027 | 37× |
| Parakeet-TDT-0.6B-v3 | 4.12 (3.3–5.1) | 238 / 5 783 | 1.95 | +0.36 (−0.41…+1.30) | **0.025** | **40×** |

### FLEURS en_us (300 sentences, 6 605 words, 57 minutes with the gaps)

| run | WER % (95% CI) | errors / words | CER % | Δ WER vs best (95% CI) | RTF | speed |
|---|---|---|---|---|---|---|
| Canary-1B-v2 | **6.34** (5.3–7.5) | 419 / 6 605 | **3.71** | – | 0.073 | 14× |
| Whisper large-v3 | 6.36 (5.1–7.8) | 420 / 6 605 | 4.35 | +0.02 (−0.92…+1.11) | 0.045 | 22× |
| Parakeet-TDT-0.6B-v3 | 6.56 (5.6–7.6) | 433 / 6 605 | 3.87 | +0.21 (−0.28…+0.67) | **0.022** | **45×** |
| Whisper large-v3-turbo | 7.30 (5.9–8.9) | 482 / 6 605 | 4.74 | +0.95 (−0.06…+2.21) | 0.026 | 38× |

## Real meetings with speakers (AMI)

Four meetings of the AMI test set: ES2004a, IS1009a, TS3003a (scenario meetings) and EN2002b (a free meeting).
`Mix-Headset` audio (the headset microphones summed, 16 kHz), 86.4 minutes in total, 65.8 minutes of reference speech,
13 225 reference words, 4 speakers each (16 in total). Highly spontaneous speech with a lot of overlap.

| run | WER % per 60 s window (95% CI) | WER % whole meetings | Δ WER vs Parakeet (95% CI) | DER % | miss / false alarm / confusion % | DER % no overlap | speakers est. / true | whole transcript | speed |
|---|---|---|---|---|---|---|---|---|---|
| Parakeet-TDT-0.6B-v3 | **22.40** (19.9–24.9) | 22.06 | – | **25.04** | 18.1 / 4.6 / 2.3 | **13.6** | 17 / 16 | **4.8 min** | **18×** |
| Whisper large-v3 | 25.61 (23.0–28.1) | 25.12 | +3.21 (+1.79…+4.54) | 31.48 | 27.9 / 2.0 / 1.6 | 20.5 | 17 / 16 | 7.3 min | 12× |
| Whisper large-v3-turbo | 25.65 (22.9–28.1) | 25.30 | +3.24 (+1.85…+4.60) | 30.79 | 27.5 / 1.6 / 1.7 | 19.7 | 17 / 16 | 5.3 min | 16× |
| Canary-1B-v2 | 26.19 (23.4–29.0) | 25.87 | +3.78 (+2.20…+5.79) | 30.66 | 28.1 / 0.9 / 1.7 | 19.9 | 17 / 16 | 7.1 min | 12× |

Per meeting (whole-meeting WER % / DER %):

| meeting | length | Parakeet | large-v3 | large-v3-turbo | Canary |
|---|---|---|---|---|---|
| ES2004a | 17.5 min | 17.1 / 22.5 | 19.1 / 26.1 | 18.9 / 25.9 | 19.8 / 27.8 |
| IS1009a | 14.0 min | 19.7 / 21.9 | 20.4 / 24.7 | 20.7 / 24.2 | 22.4 / 25.2 |
| TS3003a | 25.1 min | 14.0 / 16.4 | 17.2 / 33.6 | 16.7 / 31.8 | 18.4 / 31.6 |
| EN2002b | 29.8 min | 28.2 / 32.3 | 32.4 / 35.3 | 33.0 / 35.0 | 32.6 / 33.5 |

## What follows from it

- **Clean read English: practically a tie.** On LibriSpeech test-clean Whisper (both) and Canary are at 1.9–2.3 %,
  Parakeet is the weakest (+1.2 points against the best, interval above zero). On test-other and FLEURS the
  differences are within the intervals; 300 utterances cannot separate them. The order of the models is not stable
  between sets, so pick on speed, languages and memory rather than on these differences.
- **Meetings: Parakeet is clearly better here**, 3.2–3.8 points of WER fewer than the other three, intervals
  of the paired differences above zero, and it is the fastest. This is one corpus with four meetings; the
  per-meeting table shows the same order in all four, which is some evidence that it is not an accident of one
  recording. Whisper large-v3 and its turbo variant are indistinguishable; Canary is no better than they are.
- **Speaker attribution works; the DER is mostly missed speech.** Confusion (a word given to the wrong speaker) is
  1.6–2.3 % for all engines, and every engine found 4 speakers in 3 meetings and 5 in ES2004a. The rest of the
  DER is time with reference speech but no transcribed word: overlapped speech (the application gives each word
  one speaker) and, probably, back-channels, fillers and quiet speech that an engine leaves out. In a spot check
  on TS3003a Parakeet produced 2 313 words and large-v3 2 104 (reference: 2 501), which fits Parakeet's lower DER;
  it does not mean its speaker detection is better, since all engines use the same pyannote model. Counting only single-speaker time the DER is 13.6 % (Parakeet) to 20.5 % (large-v3).
- **Time:** an 86-minute meeting set takes 5–7 minutes through the whole pipeline with speakers on this laptop,
  12–18× realtime. The speaker step was not timed separately here; in the 88-minute interview in
  [benchmark-cs.md](benchmark-cs.md) it was about 3 minutes for every engine.
- **Quiet recordings can disappear (found by this benchmark).** FLEURS en_us contains recordings that are 50 dB
  quieter than the rest (peak amplitude 0.002). Joined between normal-level ones, the VAD of every engine skipped
  them, and the first run showed 16.6–17.6 % WER instead of 6.3–7.3 %; 11 utterances were lost in all four
  engines. The tables above are with each FLEURS en utterance peak-normalized before joining; the first,
  uncorrected runs are kept in `benchmarks/results/*-unleveled-*.json`. This was a real defect of the application,
  now fixed by local loudness leveling before the VAD (see [Quiet recordings: before and after](#quiet-recordings-before-and-after)).

## Quiet recordings: before and after

The application now lifts quiet passages itself before the VAD and the recognizers
(`mgtranscriber/leveling.py`, setting `level_audio`, on by default, described in
[ARCHITECTURE.md](ARCHITECTURE.md)). Windowed (2 s) speech level, gain only upwards to −23 dBFS, at most +48 dB,
no gain for silence and stationary noise. The diarization gets the original audio (see below). FLEURS en with
**no** peak normalization in the benchmark (`--no-peak-normalize`), the same 300 sentences:

| run | before: app without leveling | now: app with leveling | peak-normalized benchmark (tables above) |
|---|---|---|---|
| Whisper large-v3-turbo, WER % | 17.55 (18 utterances empty) | **7.37** (6.0–9.0; 2 empty) | 7.30 |
| Parakeet-TDT-0.6B-v3, WER % | 17.23 (18 utterances empty) | **6.77** (5.7–7.9; 0 empty) | 6.56 |

The quiet recordings are back; the remaining difference to the peak-normalized run is 0.1–0.2 points, inside the
interval. Checked for regressions (turbo / Parakeet, same subsets and settings as the tables above):

| set | before (WER %) | with leveling (WER %) |
|---|---|---|
| LibriSpeech test-clean, 300 utt. | 1.85 / 3.05 | 1.82 / 3.09 |
| AMI meetings, WER % (per 60 s window) | 25.7 / 22.4 | 25.6 / 23.1 |
| AMI meetings, DER % | 30.8 / 25.0 | 31.6 / 26.0 |

Leveling is neutral for clean read speech. On AMI the words are within noise (Parakeet +0.7 points); the DER is
about one point higher with leveling; speaker confusion is not the cause (1.7 / 2.2 %), the reason was not
analysed further and may be run-to-run variation. **The diarization
must not get the levelled audio:** in a first run where pyannote received it too, the DER was 34.7 % / 29.5 %
(confusion 2.1 → 9.4 % in meeting TS3003a, a 5th speaker in EN2002b), because leveling evens out the loudness
differences between speakers that the model uses. Results: `benchmarks/results/*-nopeak-*.json` (FLEURS),
`*-leveled-*.json` (LibriSpeech, AMI).

## Limits

- One machine shared with other processes, one run per model: speed numbers are indicative; WER is deterministic
  for a given input but not repeated.
- 300 utterances per read set: intervals are wide (about ±1 point on FLEURS) and many differences are not
  significant. LibriSpeech is read audiobooks; FLEURS is read Wikipedia sentences, so none of it predicts
  noisy or accented audio.
- AMI: four meetings, one microphone mix. Reference words come from the manual annotation (1.6.2); truncated
  words and words without times were dropped, overlapped words are listed in time order. Windows are cut
  by time, so a word at a window border can land in the neighbouring window.
- DER is derived from transcript word times (see Method), which makes it depend on the recognizer as well as on
  the speaker model; with a collar of 0.25 s and a 1 s pause merge. It is not a benchmark of pyannote alone.
- Models: WhisperX runs Whisper with its own VAD and alignment, ONNX engines use Silero VAD. Language was given to
  Whisper and Canary but not to Parakeet. Cloud engines were not measured.
- An earlier, discarded LibriSpeech test-other run (a 198-utterance subset, speaker limit of 6) showed two
  repetition loops of Canary ("S. S. S. …", "Chris? Chris? …") on single utterances, 17.5 % WER for Canary against
  3.7 % for the others. They are not in the final 300-utterance subset, and that run was not saved, so this is
  an observation, not a measurement. It points to a failure mode worth watching for with Canary on noisy speech.

## Data, licenses and citations

| data | license | citation |
|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) `en_us`, test | CC-BY-4.0 | Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, 2022 (arXiv:2205.12446) |
| [LibriSpeech](https://www.openslr.org/12) test-clean, test-other | CC-BY-4.0 | Panayotov, Chen, Povey, Khudanpur, *LibriSpeech: an ASR corpus based on public domain audio books*, ICASSP 2015 |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (audio and manual annotations 1.6.2) | CC BY 4.0 (see `LICENCE.txt` in the annotations archive; the OpenSLR page still lists the older NC-SA terms) | Carletta et al., *The AMI Meeting Corpus: A Pre-announcement*, MLMI 2005 |

Reference and hypothesis text is not included in `benchmarks/results/` (only ids, metrics and error counts); the
audio is not redistributed (`benchmarks/data/` is git-ignored and filled by the script). Models: OpenAI Whisper
large-v3 and large-v3-turbo (via WhisperX), NVIDIA Parakeet-TDT-0.6B-v3 and Canary-1B-v2 (ONNX, via onnx-asr),
pyannote `speaker-diarization-community-1`.

The Czech benchmark, including a test on Czech parliamentary speech with speakers, is in
[benchmark-cs.md](benchmark-cs.md).

## Reproduce

```
uv run python -m benchmarks.bench_cs --dataset librispeech-clean -n 300 --app whisperx:large-v3 --app parakeet --app canary
uv run python -m benchmarks.bench_cs --dataset ami --app whisperx:large-v3-turbo      # downloads 4 meetings, about 170 MB
uv run python -m benchmarks.bench_cs report benchmarks/results/*-ami-*.json
```

Diarization needs a Hugging Face token with access to the pyannote model (`HF_TOKEN`), as in the application.
