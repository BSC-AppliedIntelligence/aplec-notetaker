# Your own browser extension

The extension in `extension/` is the same code for every team; what makes a build
yours is the site it talks to. `scripts/package_extension.sh` writes your
`APLEC_WEB_BASE_URL` into it (the host permission and the default server in
Options), so each deployment packages its own copy.

There are two ways to hand it out.

## Load unpacked (no account needed)

The site offers the zip in **Settings → Browser extension**; each person loads it
in `chrome://extensions` with Developer mode on (steps in
[INSTALL.md](INSTALL.md), section 10). This is enough to start, but unpacked
extensions never update themselves: every new version means everyone unzips and
reloads by hand. The extension checks the version the site advertises and shows
a NEW badge when it is behind.

## Publish to the Chrome Web Store (automatic updates)

An **unlisted** listing stays out of search results (only people with the link
can install it) and gives everyone Chrome's automatic updates.

One-time setup, with a Google account of your team's:

1. Register as a developer at <https://chrome.google.com/webstore/devconsole>
   (a one-time fee).
2. Host a privacy policy at a public URL: the store requires one for extensions
   that capture audio, and its reviewers cannot reach a site on a private network.
   [extension-privacy-policy.md](extension-privacy-policy.md) is a template to adapt
   and publish (a public gist or a GitHub Pages site will do).
3. Prepare at least one screenshot (1280x800 or 640x400), such as the Options
   page and a meeting tab showing the REC badge.

Each release:

```bash
# raise "version" in extension/manifest.json, then:
./scripts/package_extension.sh
# upload aplec_web/static/aplec-extension-store.zip in the developer console:
# your item -> Package -> Upload new package -> Submit for review
```

Upload `aplec-extension-store.zip`, not `aplec-extension.zip`: the store wants
`manifest.json` at the root of the zip, and the script checks that it is. Updates
usually pass review within hours. Once the listing is live, set
`APLEC_EXTENSION_STORE_URL` in `.env.worker` to its link, and Settings hands
out the store link instead of the zip. Installing from the store gives a new
extension ID: people should remove their unpacked copy, grant the microphone
again in Options and paste their token again.

The site advertises the extension version of the build it serves. After publishing
a new version, wait until the store has rolled it out before redeploying the site;
otherwise people with the previous store version are told to update by hand.

### Listing answers

- **Name**: Aplec: meeting notes without the bot
- **Summary**: Records your Google Meet, Teams, and Zoom calls in the browser and
  sends them to your team's self-hosted Aplec for notes.
- **Category**: Productivity → Tools. **Visibility**: Unlisted.
- **Description**: what the extension does (see `extension/README.md`), plus that
  it only works with a self-hosted Aplec server run by the user's own team, and
  that audio goes only to the server the user configures.
- **Single purpose**: record the user's own meeting audio (tab and microphone)
  and upload it to the user's self-hosted meeting-notes server.
- **Permission justifications**:
  - `tabCapture`: record the meeting tab's audio (the other participants).
  - `offscreen`: Manifest V3 needs an offscreen document to run MediaRecorder.
  - `storage`: keep the server URL, the capture token and the session state.
  - `notifications`: recording started or stopped, and errors.
  - `alarms`: a watchdog that finishes an interrupted recording; the update check.
  - `scripting`: inject the meeting-page script into tabs already open when the
    extension is installed or updated.
  - Host permissions: the meeting sites (`meet.google.com`,
    `teams.microsoft.com`, `teams.live.com`, `*.zoom.us`) to detect the call and
    read who is speaking; your Aplec site and localhost, to upload.
  - Microphone: record the user's own voice, granted on the Options page.
- **Data use**: the extension handles audio, website content (participant names
  and who is speaking) and personal communications (the meeting chat after
  recording starts, with sender names, unless turned off in Options). All of it
  goes only to the server the user configures: nothing to the developer or to
  third parties, no sale, no advertising, encrypted in transit. The privacy policy
  you host must say the same; reviewers compare the listing, the policy and what
  the extension does.

## What a user should know

- Speaker names: Meet and Teams report each participant's speaking state; Zoom's
  web client shows one active speaker at a time, so overlapping voices can lose
  their names there. Teams and Zoom must be joined from Chrome, not the desktop
  apps.
- When recording starts, the extension posts a short notice in the meeting chat
  telling the other participants they are being recorded. Its text, and whether
  it is posted, are set in Options.
- Recording the same meeting again within 30 minutes adds to the same notes.
- A meeting being recorded with the extension is not also sent the bot.
