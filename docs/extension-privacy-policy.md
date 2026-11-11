<!-- Template for the privacy policy the Chrome Web Store asks for (docs/EXTENSION.md).
     Adapt it to your deployment, fill in the date and the contact, and host it at a public URL. -->

# Privacy Policy — Aplec: meeting notes without the bot (Chrome extension)

Last updated: <date>

## What this extension is

Aplec is a meeting-recording extension used by a single team with its own
self-hosted Aplec server. It records meetings you are in (with the recording
indicator visible) and uploads the audio to **your team's own server** so it
can produce transcripts and meeting notes. It does not work without such a
server, and the extension's developer operates no server and receives no
data.

## What data it handles

- **Audio**: the meeting tab's audio and, if you grant the microphone
  permission, your own voice. Recording happens only after you explicitly
  start it and is always indicated on the extension icon (REC).
- **Meeting page content**: the names shown on the meeting page's participant
  tiles and who is currently speaking, captured while you record so the notes
  can attribute sentences to speakers.
- **Meeting chat** (Google Meet and Zoom web client): the messages and links
  typed in the meeting's chat panel **after** you start recording, with the
  sender's displayed name, so the notes can include what was shared there.
  Messages written before the recording started are not read, private/direct
  messages are skipped, and the Microsoft Teams chat is not read at all. This
  can be turned off in the extension's Options. No other page content is read.
- **Recording notice**: when you start recording, the extension posts a short
  message in the meeting chat, on your behalf, telling the other participants
  that the meeting is being recorded. Whether the notice was posted is recorded
  with the meeting on your team's server. The notice is on by default and its
  text can be changed, or the notice turned off, in the extension's Options.
- **Configuration**: the server URL and your personal access token, stored
  locally in your browser (`chrome.storage.local`), never synced to Google.

## Where data goes

All recorded audio, speaker events and chat messages are sent **exclusively to the server
URL you configured** — a server operated by your own team, typically
reachable only inside your organization's network or VPN. Nothing is sent to
the extension developer, Google, or any third party. There is no analytics,
no telemetry, no advertising, and no sale of data of any kind.

What happens to the audio after upload is governed by your team's server.
In the standard Aplec deployment the audio is transcribed and its speakers
named on that machine, and old recordings are deleted automatically after a
retention period. The notes are written by an open-weights language model
that runs either on that machine or on another server operated by the same
organization; it receives the transcript, the meeting chat and questions
asked about the meeting, never the audio. Consult your team's administrator
for the details of your deployment.

## Data retention in the browser

Audio is streamed to your server in small chunks during the recording and is
not kept in the browser afterwards. The stored configuration (server URL,
token) persists until you remove it in the extension's Options or uninstall
the extension.

## Your choices

- The microphone permission is optional and granted only on the Options page;
  without it, only the meeting tab's audio (the other participants) is
  recorded.
- You choose when a recording starts and stops; there is no automatic or
  hidden recording, and by default the other participants are told in the
  meeting chat.
- Saving the meeting chat and posting the recording notice can each be turned
  off in the Options page.
- Removing the extension removes all locally stored configuration.
- Your access can be revoked centrally by your team's administrator, which
  immediately disables uploads from your token.

## Contact

This extension is operated privately by the team that runs the corresponding
Aplec server. For any privacy question or request, contact your team's Aplec
administrator at <contact address>.
