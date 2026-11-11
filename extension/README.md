# Aplec: Chrome extension (no-bot capture)

Records **Google Meet, Microsoft Teams (web) and Zoom (web client)** calls in your
browser, the meeting tab and your microphone on separate channels, and streams
them to your team's Aplec server, which runs the same pipeline as bot captures:
diarization, **named speakers** (the extension reads each platform's
active-speaker indicators), multilingual transcription and the notes. No bot
appears in the participant list.

For Teams and Zoom, **join from Chrome**, not the desktop app (Teams: "Continue on
this browser"; Zoom: "Join from your browser"). An extension cannot see desktop
apps.

Build it for your server with `scripts/package_extension.sh` (see
[docs/INSTALL.md](../docs/INSTALL.md), section 10): that writes your site into the
extension's permissions and default server URL.

## Install (once)

1. Chrome or Edge → `chrome://extensions` → enable **Developer mode**.
2. **Load unpacked** → select the packaged `extension/` folder (the zip from
   **Settings → Browser extension** on your Aplec site).
3. Open the extension's **Options**, check the server URL (your Aplec site, already
   filled in) and paste your **capture token** from Settings.
4. Click **Save**: it should say *connected*.
5. Still in Options, click **Grant microphone access** and allow it. Chrome can
   only show this prompt on the Options page, never during a recording; without it
   Aplec records only the other participants, not you.

If your team publishes the extension in the Chrome Web Store
([docs/EXTENSION.md](../docs/EXTENSION.md)), install it from the store link in
Settings instead; it then updates itself.

## Use

- Join the meeting **in Chrome**. Aplec notices the call: the toolbar icon shows an
  amber dot.
- **Click the Aplec icon to start recording** (Chrome requires the click on the
  extension itself to allow tab capture). The badge shows **REC**, and a short
  notice is posted in the meeting chat to tell the others they are being recorded
  (configurable in Options).
- Click the icon again to stop, or leave the call and Aplec stops by itself.
- The notes appear on your Aplec site a few minutes later.
- Recording the **same meeting again within 30 minutes** adds to the same notes
  instead of creating a new meeting.

## Notes and limits

- Meet and Teams report each participant's speaking state; Zoom's web client shows
  one active speaker at a time, so overlapping talkers may lose their names there.
- A meeting you are recording with the extension is not also sent the bot.
- Your microphone is recorded even when the meeting mutes you to the others.
- Saving in Options makes a test call that creates a tiny empty capture on the
  server; these are harmless and never processed.
