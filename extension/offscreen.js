// Offscreen document: does the actual audio work.
// - tab audio via the streamId minted by the service worker (tabCapture)
// - microphone via getUserMedia (permission must be granted from the OPTIONS
//   page first: Chrome cannot render permission prompts for offscreen docs)
// - both on separate channels (left mic, right tab) -> MediaRecorder (webm/opus,
//   15s timeslice). Never summed here: when another laptop is the room's
//   microphone in the call, the tab brings the room back a fraction of a second
//   after the mic heard it, and only the worker, with the two apart, can use one
// - CRITICAL: tabCapture mutes the tab for the user; we route the tab stream
//   back to the speakers so the meeting stays audible.
//
// Upload pipeline (v0.3.8): chunks go through a pending[] FIFO drained by a
// single-flight loop that RETRIES on failure (VPN drop mid-meeting must not
// punch a hole in the audio; the recording is local, only delivery lags).
// Chunks hold Blobs (Chrome pages large Blobs to disk) and are tagged with
// their recorder's sessionId + partNonce by closure, so a reused document or
// a same-meeting resume can never relabel old audio. The buffer survives ~1h
// of outage (240 chunks at the explicit 64kbps); beyond that the oldest chunk
// is evicted — except seq 0, which carries the webm header without which
// everything after is undecodable server-side.
// All messages are tagged with the sessionId so a stale session can never be
// confused with a new one, and chunk uploads are DRAINED (with retries, up to
// 2 minutes) before "stopped" is signaled so quick stops keep their audio.

let recorder = null;
let sessionId = null;
let contexts = [];
let seq = 0;

const MAX_PENDING_CHUNKS = 240;   // ~1h of 15s chunks
const RETRY_BASE_MS = 5000;
const RETRY_CAP_MS = 25000;       // stays inside the SW's 30s idle window
const STOP_DRAIN_MS = 120000;

let pending = [];                 // [{blob, seq, sid, nonce}]
let drainRunning = false;
let evicted = 0;

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // ONLY answer messages addressed to the offscreen document. runtime
  // messages broadcast to every listener and the first sendResponse wins:
  // answering {} to a content script's call-state poll (field bug) beat the
  // service worker's real answer and made the poll read capturing=false,
  // killing speaker observation seconds into every recording.
  if (msg.type !== "offscreen-start" && msg.type !== "offscreen-stop") return false;
  (async () => {
    if (msg.type === "offscreen-start") {
      try {
        const { micOk, startedAtMs } = await start(msg.streamId, msg.sessionId, msg.partNonce);
        sendResponse({ ok: true, micOk, startedAtMs });
      } catch (e) {
        console.error("offscreen start failed", e);
        sendResponse({ ok: false, error: String(e) });
      }
    } else {
      if (!msg.sessionId || msg.sessionId === sessionId) stop();
      sendResponse({});
    }
  })();
  return true;
});

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function enqueueChunk(item) {
  if (pending.length >= MAX_PENDING_CHUNKS) {
    // Evict the oldest chunk that is NOT a recorder's seq 0 (the webm
    // header); the gap lands at the outage's start, freshest audio survives.
    const idx = pending.findIndex((p) => p.seq !== 0);
    pending.splice(idx === -1 ? 0 : idx, 1);
    evicted += 1;
    if (evicted % 20 === 1) console.warn(`chunk buffer full; evicted ${evicted} so far`);
  }
  pending.push(item);
  drainLoop();
}

async function drainLoop() {
  if (drainRunning) return;         // one loop per document
  drainRunning = true;
  let failures = 0;
  try {
    while (pending.length) {
      const head = pending[0];
      let resp = null;
      try {
        const b64 = await blobToBase64(head.blob);
        resp = await chrome.runtime.sendMessage({
          type: "offscreen-chunk",
          b64,
          seq: head.seq,
          sessionId: head.sid,
          partNonce: head.nonce,
          queued: pending.length,
        });
      } catch (e) {
        resp = null;                // SW asleep/mid-restart: retry
      }
      if (resp && resp.uploaded === true) {
        pending.shift();
        failures = 0;
      } else if (resp && resp.fatal === true) {
        console.warn("dropping chunk (non-retryable)", head.sid, head.seq);
        pending.shift();
        failures = 0;
      } else {
        failures += 1;
        await sleep(Math.min(RETRY_CAP_MS, RETRY_BASE_MS * 2 ** Math.min(failures - 1, 3)));
      }
    }
  } finally {
    drainRunning = false;
  }
}

async function start(streamId, sid, partNonce) {
  sessionId = sid;
  // A fresh session owns this document: chunks a previous session never
  // delivered would only be stale-dropped server-side; do not let them delay
  // live audio behind dead retries.
  pending = pending.filter((p) => p.sid === sid);
  const tabStream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
    video: false,
  });
  // The tab closing ends this track. We record the MERGED stream (which never
  // ends on its own), so without this the recorder would run on silence
  // forever after the tab is gone.
  tabStream.getAudioTracks().forEach((track) => {
    track.onended = () => { console.warn("tab audio track ended; stopping recorder"); stop(); };
  });
  let micStream = null;
  let micOk = false;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    micOk = true;
  } catch (e) {
    console.warn("mic unavailable; recording tab audio only", e);
  }

  const ctx = new AudioContext();
  contexts = [ctx];
  const destination = ctx.createMediaStreamDestination();
  destination.channelCount = 2;
  // Each input of a merger is one channel: a stereo tab is downmixed on the way
  // in, and a missing mic leaves the left channel silent.
  const merger = ctx.createChannelMerger(2);
  merger.connect(destination);

  const tabSource = ctx.createMediaStreamSource(tabStream);
  tabSource.connect(merger, 0, 1);
  tabSource.connect(ctx.destination); // keep the meeting audible to the user

  if (micStream) {
    ctx.createMediaStreamSource(micStream).connect(merger, 0, 0);
  }

  seq = 0;
  evicted = 0;
  // Explicit bitrate: the buffer's time coverage (and the server's storage)
  // must not depend on Chrome's default (~128k). 32k per channel, the same
  // each source had when they went out as one mono mix.
  recorder = new MediaRecorder(destination.stream, {
    mimeType: "audio/webm;codecs=opus",
    audioBitsPerSecond: 64000,
  });
  // handlers close over `sid`/`partNonce`: even if a new session reuses this
  // document, this recorder's chunks and stop signal stay tagged as THEIRS
  recorder.ondataavailable = (event) => {
    if (!event.data || event.data.size === 0) return;
    enqueueChunk({ blob: event.data, seq: seq++, sid, nonce: partNonce });
  };
  recorder.onstop = async () => {
    // Drain with retries for up to 2 minutes, heartbeating the SW every 10s
    // so its 30s force-finish alarm keeps being pushed back and it does not
    // close this document (and destroy the buffer) mid-drain.
    const deadline = Date.now() + STOP_DRAIN_MS;
    const beat = setInterval(() => {
      chrome.runtime.sendMessage(
        { type: "offscreen-draining", sessionId: sid, pending: pending.length }
      ).catch(() => {});
    }, 10000);
    while (pending.length && Date.now() < deadline) {
      drainLoop();
      await sleep(1000);
    }
    clearInterval(beat);
    const undelivered = pending.filter((p) => p.sid === sid).length;
    if (undelivered) {
      console.warn(`giving up on ${undelivered} undelivered chunk(s) for ${sid}`);
      pending = pending.filter((p) => p.sid !== sid);
    }
    try {
      await chrome.runtime.sendMessage({ type: "offscreen-stopped", sessionId: sid, undelivered });
    } catch (e) {}
    tabStream.getTracks().forEach((t) => t.stop());
    micStream?.getTracks().forEach((t) => t.stop());
    contexts.forEach((c) => c.close());
  };
  recorder.start(15000);
  // Speaker-event offsets are computed against this instant, all on the
  // browser's clock; server clocks are never mixed in.
  return { micOk, startedAtMs: Date.now() };
}

function stop() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  recorder = null;
}

function blobToBase64(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result).split(",")[1]);
    reader.readAsDataURL(blob);
  });
}
