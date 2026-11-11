# Installing Aplec from scratch

This guide takes one Linux machine with an NVIDIA GPU to a deployment your whole
team can use: a site behind HTTPS where people sign in with Google, and a Chrome
extension that records their meetings. The meeting bot, voiceprints and the second
diarizer are optional and come at the end.

Every command runs as an ordinary user from the repository root unless it says
otherwise. Paths in `.env.worker` default to locations inside the checkout, so
nothing needs to live anywhere in particular.

## 1. Host prerequisites

- An NVIDIA driver and the CUDA toolkit (`nvcc`), matching each other.
- Python 3.12 with venv, `git`, `cmake`, a C++ compiler, `curl` and `zip`.
- For voiceprints with the ReDimNet2 model (optional, section 12): the Python
  headers (`python3.12-dev` or equivalent), because `wespeaker` builds `hdbscan`
  from source on some platforms.
- Docker with the compose plugin, only for the meeting bot (section 11).
- No system ffmpeg is needed: the worker uses the binary that `imageio-ffmpeg`
  ships inside the virtualenv.

## 2. Code and Python environment

```bash
git clone https://github.com/BSC-AppliedIntelligence/aplec-notetaker.git
cd aplec-notetaker
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.worker.txt -r requirements.web.txt
pip uninstall -y torchcodec   # see "Known problems"; pip may pull it back with pyannote
```

Check that PyTorch sees the GPU: `python -c 'import torch; print(torch.cuda.is_available())'`
must print `True`. If it does not (a CUDA version or platform the PyPI build does not
cover), install the build for your CUDA from <https://pytorch.org> first, then the
requirements again.

`requirements.worker.lock.txt` is the exact environment Aplec was validated with
(ARM64, CUDA 13); the container image uses it. On other platforms install from
`requirements.worker.txt` as above.

## 3. whisper.cpp and whisper-batch

```bash
git clone https://github.com/ggml-org/whisper.cpp vendor/whisper.cpp
cd vendor/whisper.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ../..
./scripts/build_whisper_batch.sh
```

`whisper-batch` (in `tools/`) keeps the model loaded for a whole meeting instead of
reloading it for every speaker turn, with the same output as `whisper-cli`. The
script also applies `patches/whisper-cpp-second-language.patch`, which lets Aplec
give Whisper a second language and stops it from translating what is said in
that language. If a future whisper.cpp refuses the patch, the build still works
without that feature. After updating whisper.cpp, rerun the script, restart the
worker and check that both engines still agree:

```bash
.venv/bin/python scripts/verify_whisper_batch.py <any recording>
```

## 4. Models

```bash
./scripts/download_models.sh          # Whisper large-v3 for whisper.cpp
./scripts/download_models.sh --llm    # also the local summarizer (section 5, option A)
```

- **Diarization**: `pyannote/speaker-diarization-community-1` is gated. Accept its
  conditions on Hugging Face with your account, create a read token, and put it
  in `.env.worker` as both `FRED_LIVE_HF_TOKEN` and `HF_TOKEN`. The worker
  downloads it on first use.
- **Language ID** (`speechbrain/lang-id-voxlingua107-ecapa`) downloads itself on
  first use into `models/`.
- Voiceprints and the second diarizer have their own downloads (sections 12, 13).

## 5. The language model that writes the notes

Aplec talks to any OpenAI-compatible chat API: a model on this machine, a server you
run elsewhere, or a hosted API. Pick one.

**A. Local llama.cpp** (about 25 GB of GPU memory for the default model):

```bash
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
cd ../..
./scripts/run_llama_server.sh         # serves Qwen3.8-27B on 127.0.0.1:18081
```

The script serves the model downloaded with `--llm`, with its speculative-decoding
drafter when present. Its settings (`FRED_LIVE_LLAMACPP_*`) are read from its own
environment; the example unit in `deploy/systemd` loads them from `.env.worker`.
Keep `FRED_LIVE_LLM_CTX_SIZE` equal to the server's context per slot.

**B. A server you already run** (vLLM, llama.cpp elsewhere, or an Open WebUI
instance, whose base URL ends in `/api`):

```bash
FRED_LIVE_LLM_BASE_URL=https://llm.example.org/api
FRED_LIVE_LLM_MODEL=<the model name on that server>
FRED_LIVE_LLM_API_KEY=<its API key>
```

Qwen3-family models need `FRED_LIVE_LLM_DISABLE_THINKING=true` (the default), or they
spend the whole answer budget reasoning.

**C. A hosted API** (OpenAI, Anthropic, Mistral, Groq, OpenRouter, Google Gemini and
others), through its OpenAI-compatible endpoint:

```bash
FRED_LIVE_LLM_BASE_URL=https://api.openai.com/v1   # the provider's base, usually ending in /v1
FRED_LIVE_LLM_MODEL=<the provider's model name>
FRED_LIVE_LLM_API_KEY=<your API key>
FRED_LIVE_LLM_DISABLE_THINKING=false   # true sends a llama.cpp/vLLM field that hosted APIs refuse
FRED_LIVE_LLM_CTX_SIZE=128000          # the model's context window
```

When a provider's route is not under `/v1` (Gemini's is
`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`), give the
full `.../chat/completions` URL as the base. Aplec sends `temperature` and `max_tokens`
on every call, so models that refuse them, as some reasoning models do, will not work
as they are. The prompts were tuned on Qwen3.8-27B; other capable chat models should
give the same JSON, which Aplec validates and repairs, but test the notes on a few of
your meetings.

With options B and C, the transcripts, the meeting chat and the Ask questions go to
that server, under its terms; audio never does.

**Transcription stays local.** There is no hosted-transcription backend: Aplec
transcribes every speaker turn separately, hundreds of calls per meeting, with its
language set, a second language as a hint and Whisper's own language probabilities
and confidence, which hosted transcription APIs do not expose. Diarization and
voiceprints need the GPU anyway.

## 6. Configuration

```bash
cp .env.worker.example .env.worker
chmod 600 .env.worker
```

Both the worker and the web app read this file. Set at least:

| Setting | Value |
| --- | --- |
| `FRED_LIVE_AUTH_TOKEN` | `openssl rand -hex 24`. Guards the worker's internal API; empty means every internal request is refused |
| `FRED_LIVE_WEB_SESSION_SECRET` | `openssl rand -hex 32`. Signs sign-in sessions and the extension's capture tokens; the web app refuses to start without it |
| `FRED_LIVE_WEB_BASE_URL` | The public URL of your site, e.g. `https://aplec.example.org` |
| `FRED_LIVE_WEB_USERS` | The first administrator: `you@example.org:1:admin` (see section 9) |
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | From section 7 |
| `FRED_LIVE_HF_TOKEN`, `HF_TOKEN` | From section 4 |
| `FRED_LIVE_APP_PORT` | `18088`: the port the web app and the meeting bot expect |

Everything else has working defaults; the file documents each setting.

## 7. Google sign-in

People sign in to the site with their Google account. In the Google Cloud console,
in a project of your own:

1. **OAuth consent screen**: user type Internal if everyone is in your Google
   Workspace organization, External otherwise. Scopes: `openid`, `email` and
   `profile`; add `https://www.googleapis.com/auth/calendar.events` only if you
   will use calendar auto-join (section 11). While an External app is in testing,
   only the test users you list can sign in, and calendar connections expire
   after 7 days; publish the app to lift both.
2. **Credentials → Create OAuth client ID → Web application**, with the
   authorized redirect URI `<FRED_LIVE_WEB_BASE_URL>/auth/google/callback`.
3. Copy the client ID and secret into `.env.worker`.

## 8. Running the services

`deploy/systemd/` has user units for the worker, the web app, the local LLM and
the calendar watcher. They assume the checkout is at `~/fred-notetaker`; edit
`WorkingDirectory` and the paths if it is elsewhere.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/fred-worker.service deploy/systemd/fred-web.service ~/.config/systemd/user/
cp deploy/systemd/fred-llama.service ~/.config/systemd/user/     # option A only
systemctl --user daemon-reload
systemctl --user enable --now fred-worker fred-web               # (+ fred-llama)
loginctl enable-linger "$USER"   # keep them running without an open session (may need an admin)
curl -s http://127.0.0.1:18088/healthz
```

Logs: `journalctl --user -u fred-worker -f`. Without systemd, run
`scripts/run_worker.sh`, `scripts/run_web.sh` and `scripts/run_llama_server.sh`
with the same environment.

The worker binds `0.0.0.0:18088` so the meeting bot's containers can reach it.
If you do not run the bot, set `FRED_LIVE_APP_HOST=127.0.0.1`; otherwise keep the
port closed to everything but the host and its containers.

## 9. Publishing the site

The web app listens on `127.0.0.1:18091`. Put it behind any reverse proxy that
terminates HTTPS (nginx, Caddy, Traefik) on the host name of
`FRED_LIVE_WEB_BASE_URL`, forwarding the usual `X-Forwarded-*` headers. Sign-in
cookies become `Secure` automatically when that URL is https. The site also
relays the extension's uploads (`/capture/*`) to the worker, so this one host name
is all the extension needs.

`Dockerfile.web` builds the web app as a container instead; mount `.env.worker`
at `/app/.env.worker` and `data/` at `/app/data`, and set `FRED_LIVE_WORKER_URL`
to an address where the container can reach the worker.
`docker-compose.gpu.yml` runs the worker, the LLM and the web app all in
containers (build `Dockerfile.worker`, which compiles whisper.cpp; set
`CUDA_ARCH` for your GPU).

**Users.** `FRED_LIVE_WEB_USERS` lists `email:number[:admin]` entries, separated
by commas: the number identifies the person's meetings, and admins see every
meeting. With the meeting bot installed (section 11), new people simply sign in:
their request waits on the **Users** page until an admin approves it, which
creates their bot account too and adds them to `data/users.json`. Without the
bot, add each person to `FRED_LIVE_WEB_USERS` with a number of their own and
restart the web app. Removing someone from either list revokes their session and
their extension token on the next request.

## 10. The browser extension

Build it pointed at your site:

```bash
./scripts/package_extension.sh
```

The script reads `FRED_LIVE_WEB_BASE_URL` from `.env.worker` (or the
environment), writes your site into the extension's permissions and default
server, and puts two zips in `fred_live_web/static/`: `fred-extension.zip`, which
the site offers in **Settings → Browser extension**, and `fred-extension-store.zip`,
the layout the Chrome Web Store accepts. Rerun it whenever the site URL or the
extension changes.

Each person then, in Chrome:

1. Downloads the zip from Settings, unzips it, opens `chrome://extensions`, turns
   on **Developer mode** and chooses **Load unpacked** on the `extension` folder.
2. Opens the extension's **Options**, pastes the capture token shown in Settings
   (the server URL is already filled in), and saves: it should say *connected*.
3. Clicks **Grant microphone access** on the same page. Chrome can only ask for
   the microphone there, never during a recording.
4. Joins a meeting in Chrome and clicks the Aplec icon to record; clicking again,
   or leaving the call, stops it. The notes appear on the site a few minutes later.

Unpacked extensions do not update themselves. To give everyone automatic updates,
publish your build to the Chrome Web Store under your own developer account:
[EXTENSION.md](EXTENSION.md).

## 11. Optional: the meeting bot and calendar auto-join

The bot is [Vexa](https://github.com/Vexa-ai/vexa), pinned as a git submodule and
patched for Aplec (ARM64 builds, speaker events that survive the host ending the
meeting, a persistent signed-in session, delivery to a worker on the same host).

```bash
git submodule update --init vendor/vexa
cd vendor/vexa
git apply ../../patches/vexa-fredlive-deploy.patch
cp deploy/env-example .env
```

In `vendor/vexa/.env` set:

```
BROWSER_IMAGE=fred-live
ADMIN_TOKEN=<openssl rand -hex 24>
ADMIN_API_TOKEN=<the same value>
STORAGE_BACKEND=local
LOCAL_STORAGE_DIR=/data/recordings
LOCAL_STORAGE_VOLUME_SOURCE=<absolute path of this checkout>/recordings
WEBHOOK_ALLOWED_HOSTS=host.docker.internal
DOCKER_GID=<getent group docker | cut -d: -f3>
TRANSCRIPTION_SERVICE_URL=http://transcription-disabled.invalid/v1/audio/transcriptions
```

The last line keeps Vexa in recording-only mode: Aplec transcribes, and no audio
goes to Vexa's hosted transcription service. Then build the bot image and start
Vexa:

```bash
docker build -t fred-live -f services/vexa-bot/Dockerfile ./services/vexa-bot
docker compose --env-file .env -f deploy/compose/docker-compose.yml up -d
cd ../..
```

In `.env.worker`, set `FRED_LIVE_WEBHOOK_SECRET` (`openssl rand -hex 24`), then
register each person with Vexa, starting with yourself:

```bash
python -m fred_live_worker.onboard --email you@example.org --admin
```

This creates their Vexa user and API key, registers the webhook that tells the
worker a meeting ended, and adds them to `data/users.json`. From then on the
**Users** page does the same when an admin approves a request. The keys are kept
in `data/credentials/vexa_keys.json`; copying yours into `.env.worker` as
`VEXA_API_KEY` lets the worker fetch recordings through Vexa's API when it cannot
read them from disk.

**Calendar auto-join.** Each person connects their calendar in Settings (this
needs the `calendar.events` scope from section 7), and chooses which of their
meetings the bot joins. Install and start the watcher:

```bash
cp deploy/systemd/fred-autojoin.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now fred-autojoin
```

**The bot's Google account.** Anonymous bots work, but the host must admit them
by hand and Meet labels them as outsiders. For signed-in bots, create a Google
account for the bot, log it in once through a Vexa browser session (see Vexa's
documentation on browser sessions; Aplec's patch keeps the session alive), and
set `FRED_LIVE_BOT_GOOGLE_ACCOUNT` to its address. The watcher then also invites
that account to each event, so the bot is a guest of the meeting. To stay
anonymous, set `FRED_LIVE_AUTOJOIN_AUTHENTICATED=false` and
`FRED_LIVE_AUTOJOIN_INVITE_BOT=false`.

After any reset of the submodule, apply the patch again and rebuild the bot image.

## 12. Optional: voiceprints

With voiceprints, Aplec names people by their voice: those who share one
microphone in a room, and anyone the page indicators miss. Each person enrolls
in **Settings → Voiceprint**, behind a consent notice; only a numeric template is
kept, never the recording, and they can delete it at any time. Admins invite
people from the **Voiceprints** page but never upload audio for someone else.

The default backend, ECAPA from SpeechBrain, needs nothing else. ReDimNet2
identifies better and is used automatically when installed:

```bash
.venv/bin/pip install "git+https://github.com/wenet-e2e/wespeaker.git" onnxruntime
./scripts/download_models.sh --voiceprints
```

Prints from one backend do not match the other: switching means enrolling again.

## 13. Optional: the second diarizer

NVIDIA's Nemotron 3 Diarization separates voices better in rooms, and Aplec uses it
together with pyannote. It needs a newer `transformers` than the worker, so it gets
its own environment, which reuses the worker's PyTorch:

```bash
./scripts/setup_nemotron.sh     # after the worker's .venv exists
```

Then set `FRED_LIVE_DIARIZATION_NEMOTRON=true` and restart the worker. If the
combination would cost someone their name, or hears fewer voices than the page
shows, Aplec keeps pyannote's result for that meeting.

## 14. Optional: a glossary for names

Speech recognition misspells product names, acronyms and people's names, and the
notes copy what they read. List yours in `data/glossary.txt`, one per line, as
`term | what it is | how it comes out wrong`, for example:

```
Qwen | language model | Cuen, Quen
```

It is added to every prompt that writes notes and to Ask, and read on every call,
so editing it needs no restart.

## 15. Operations

- **Data**: every run is a directory under `data/meetings/<meeting>/<time>/` with
  the payload, the transcript, the notes and the model's raw output;
  `latest.json` mirrors the newest one. A failed run leaves `error.json` with the
  traceback, and its card on the site has a Reprocess button.
- **Retention**: meeting audio is deleted after `FRED_LIVE_RETENTION_DAYS` (15);
  transcripts and notes are kept.
- **Recovery**: on startup the worker reprocesses runs left unfinished, and every
  6 hours it looks for extension recordings that never produced a run.
- **Backups**: `data/` and `.env.worker`; with the bot, also `vendor/vexa/.env` and
  Vexa's Postgres volume. Models and builds can be recreated.
- **Tools**: `python -m fred_live_worker.reprocess --meeting-id <id>` rewrites the
  notes from a stored transcript; `scripts/replay_hook.sh <run>/hook.json` replays a
  whole meeting through a running worker; `python -m fred_live_worker.ask
  --meeting-id <id> "question"` asks from the command line.

## 16. Known problems

- **torchcodec must stay uninstalled.** It arrives with pyannote's dependencies,
  and when it cannot load FFmpeg's shared libraries, `transformers` fails at
  import time. Aplec decodes audio with the ffmpeg binary instead.
  pyannote then prints a warning that torchcodec is not installed: that is
  expected and harmless.
- **The `transformers` ASR backend** uses far more memory than whisper.cpp in
  word-timestamp mode; keep the default `whispercpp` with diarization on.
- **A new Docker network can hide part of your network.** Docker picks the first
  free private range for a network created without a subnet; if that range
  contains a server Aplec talks to (your LLM, for instance), its route is shadowed
  and requests fail with `No route to host`. Give compose projects explicit
  subnets, or set `default-address-pools` in Docker's daemon configuration.
- **Vexa has two kinds of API keys**: user keys (`X-API-Key`, for bots and
  recordings) and the admin token (`X-Admin-API-Key`); they are not
  interchangeable.
- **Bot recordings belong to uid 999** (Vexa's container), so retention deletes
  them through `docker exec` into `vexa-meeting-api-1`
  (`FRED_LIVE_RETENTION_CONTAINER`).
- **The worker reads `.env.worker` from its working directory**: run it from the
  checkout (the scripts and units do).

## 17. Security notes

- The worker's internal API takes one shared token (`FRED_LIVE_AUTH_TOKEN`), and
  ownership of meetings is enforced by the web app. Keep the token secret and the
  worker's port closed to the outside.
- The extension authenticates with a per-person signed token, re-checked against
  the user list on every upload. Bot webhooks are signed with
  `FRED_LIVE_WEBHOOK_SECRET` and must be fresh (5 minutes).
- Meeting content is rendered with autoescaping, external commands run without a
  shell, and recording paths from webhooks must resolve inside the recordings
  directory.
- `data/credentials/` holds calendar refresh tokens and Vexa keys (mode 600): back
  it up as a secret.
