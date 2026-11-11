# Third-party notices

Aplec is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). It bundles these
files, each under its own license, whose texts are in [licenses/](licenses/):

| Component | Files | License |
| --- | --- | --- |
| [htmx](https://htmx.org) 2.0.7 | `aplec_web/static/htmx.min.js` | Zero-Clause BSD ([licenses/htmx-0BSD.txt](licenses/htmx-0BSD.txt)) |
| [Inter](https://github.com/rsms/inter), variable font | `aplec_web/static/fonts/` | SIL Open Font License 1.1 ([licenses/Inter-OFL.txt](licenses/Inter-OFL.txt)) |
| [flag-icons](https://github.com/lipis/flag-icons) | `aplec_web/static/flags/` | MIT ([licenses/flag-icons-MIT.txt](licenses/flag-icons-MIT.txt)) |

`patches/` holds changes to [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
(MIT) and [Vexa](https://github.com/Vexa-ai/vexa) (Apache-2.0). Vexa is a git
submodule in `vendor/vexa`; whisper.cpp and llama.cpp (MIT) are cloned and built at
install time. None of them is copied into this repository.

The models are not part of this repository either. Aplec downloads them at install
time or on first use, and each keeps its publisher's terms:

| Model | Used for | License |
| --- | --- | --- |
| OpenAI Whisper large-v3 (GGML conversion, `ggerganov/whisper.cpp`) | transcription | MIT |
| `pyannote/speaker-diarization-community-1` | diarization | CC-BY-4.0; gated, accept its conditions on Hugging Face |
| `speechbrain/lang-id-voxlingua107-ecapa` | language identification | Apache-2.0 |
| `speechbrain/spkrec-ecapa-voxceleb` | voiceprints (default backend) | Apache-2.0 |
| `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM` (optional) | voiceprints | Apache-2.0 |
| `nvidia/Nemotron-3-Diarization` (optional) | second diarizer | OpenMDW-1.1 |
| `unsloth/Qwen3.8-27B-GGUF` (optional, local summarizer) | notes | Apache-2.0 |
