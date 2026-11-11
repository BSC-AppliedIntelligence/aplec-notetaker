# Aplec Notetaker

Self-hosted meeting notes for a team. Aplec records Google Meet, Microsoft Teams and
Zoom calls, transcribes them on your own GPU, names the speakers, and writes
structured notes (summary, decisions, action items with owners, open questions)
with the language model of your choice: an open-weights model on the same
machine, or any OpenAI-compatible API, your own server or a hosted one. Audio is
always processed on your machine; with a hosted model, the transcripts go to
that provider.

## What it does

- **Capture without a bot**: a Chrome extension records the meeting tab and your
  microphone (on separate channels) and streams them to your server. Works with
  Google Meet, and with Teams and Zoom joined from the browser. Optionally, a
  **meeting bot** ([Vexa](https://github.com/Vexa-ai/vexa)) joins the meetings on
  your Google Calendar instead.
- **Transcription** with whisper.cpp (large-v3), including meetings that switch
  between languages: each speaker turn is transcribed in its own language, and
  Whisper's habit of inventing a language, or translating speech into another
  one, is checked against a second language-ID model.
- **Named speakers**: pyannote diarization (optionally combined with NVIDIA's
  Nemotron diarizer), matched against the speaking indicators of the meeting
  page. Optional **voiceprints**, enrolled by each person behind a consent
  notice, name people who share a meeting-room microphone.
- **Notes and questions**: a summary, decisions, action items with owners and
  risks, written in the meeting's language or in each user's chosen one. "Ask"
  answers questions about one meeting from its transcript.
- **A web app for the whole team**: Google sign-in, admin approval of new users,
  per-user settings, search, speaker renaming, sharing with people or teams, and
  optional automatic sharing with the colleagues recognized by voice in a
  meeting. Interface in English, Spanish and Catalan.

## Requirements

| | |
| --- | --- |
| GPU | NVIDIA with CUDA. Transcription and diarization always run on it and peak around 8 GB. The default local LLM (Qwen3.8-27B, Q6_K) needs about 25 GB more; with a remote or hosted LLM it is not needed. Developed on an NVIDIA GB10 (ARM64, unified memory). |
| Disk | About 4 GB of models, plus 23 GB for the local LLM. Meeting audio is deleted after 15 days by default; transcripts and notes are kept. |
| Software | Linux, Python 3.12, git, cmake and a C++ toolchain (whisper.cpp is built from source). Docker only for the meeting bot. |
| Accounts | A Hugging Face account that accepted the conditions of `pyannote/speaker-diarization-community-1`, and a Google Cloud OAuth client for sign-in. |

## Install

The full guide, from an empty machine to a working team deployment, is
[docs/INSTALL.md](docs/INSTALL.md). In short:

```bash
git clone https://github.com/BSC-AppliedIntelligence/aplec-notetaker.git && cd aplec-notetaker
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.worker.txt -r requirements.web.txt && pip uninstall -y torchcodec
git clone https://github.com/ggml-org/whisper.cpp vendor/whisper.cpp
(cd vendor/whisper.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j)
./scripts/build_whisper_batch.sh
./scripts/download_models.sh --llm          # drop --llm when the LLM runs elsewhere
cp .env.worker.example .env.worker          # then fill in the secrets (see the guide)
# run the services: deploy/systemd/ has user units, or scripts/run_*.sh by hand
./scripts/package_extension.sh              # the extension, pointed at APLEC_WEB_BASE_URL
```

## How it fits together

| Service | Default port | Role |
| --- | --- | --- |
| Worker (`aplec_worker`) | 18088 | Receives extension uploads and bot webhooks, runs transcription, diarization and the notes, stores every run under `data/meetings/` |
| Web app (`aplec_web`) | 18091 | The site: meetings, notes, transcript, Ask, settings, sharing, admin pages. Also relays the extension's uploads to the worker, so one public hostname serves both |
| llama.cpp server (optional) | 18081 | The local summarizer, over an OpenAI-compatible API; or use your own server or a hosted API |
| Calendar watcher (optional) | none | Sends the bot to the Meet events of each connected Google Calendar |
| Vexa (optional) | 8056, 8057 | The meeting bot, in Docker |

Per meeting: audio → silence check → diarization (who speaks when) → speaker names
(page indicators, then voiceprints) → transcription per speaker turn, language by
language → notes from the transcript → web app.

## Repository layout

```
aplec_worker/   worker: pipeline, ASR, language ID, diarization, voiceprints, notes, capture, bot integration
aplec_web/      web app: sign-in, pages, translations, static assets
extension/          Chrome extension (Manifest V3)
tools/              whisper-batch (keeps whisper.cpp loaded per meeting), Nemotron runner
patches/            whisper.cpp second-language patch, Vexa deployment patch
scripts/            model download, builds, run scripts, extension packaging, replay
deploy/systemd/     example user units
docs/               installation, extension publishing, privacy policy template
```

## License

Apache License 2.0, see [LICENSE](LICENSE). Bundled fonts, icons and scripts keep
their own licenses, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
as do the models Aplec downloads and the projects it builds on (whisper.cpp and
llama.cpp, MIT; Vexa, Apache-2.0).

## Acknowledgements

This work has been financed by the Government of Catalonia through the
[Aina Project](https://projecteaina.cat/).
