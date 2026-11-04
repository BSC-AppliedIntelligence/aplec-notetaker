// Fred capture (MV3 service worker).
// Orchestrates: call detection (from content script) -> badge signal ->
// action-click starts/stops capture (tabCapture requires an extension
// invocation gesture; a page button cannot grant it) -> offscreen document
// records -> chunks upload to the worker -> finish enqueues processing.
//
// Robustness rules learned from field testing:
// - Every offscreen message carries a sessionId; the SW ignores anything from
//   a session that is no longer current, so a quick stop->start can never let
//   the old session's teardown clobber the new one.
// - Capture state lives in chrome.storage.session: MV3 can suspend this
//   worker mid-capture (the offscreen document survives and keeps recording),
//   and the restored state lets late chunks upload instead of being dropped.
// - Action clicks are debounced with a busy flag: start and stop are
//   multi-step async flows and a double-click must not interleave them.

const state = {
  capturing: false,
  sessionId: null,
  tabId: null,
  recordingStartMs: null, // Date.now() at recorder start, browser clock
  lastCallHeartbeatMs: null, // last inCall:true report from the capturing tab
  lastDrainHeartbeatMs: 0, // post-stop buffered-upload drain in progress
  busy: false,
  inCallTabs: new Map(), // tabId -> {platform, meetCode, title}
};

// If the in-call frame dies without saying goodbye (host ends the meeting and
// the platform swaps documents), its 3s heartbeats stop. Chunk messages
// (every 15s) evaluate this: no heartbeat for this long while capturing ->
// the call is gone, stop. Long enough that a laggy tab never false-positives.
const CALL_HEARTBEAT_TIMEOUT_MS = 60_000;

// Supported platforms; the content script's adapter does the in-page work,
// this table covers SW-side URL checks and fallback id extraction.
const PLATFORMS = [
  {
    api: "google_meet",
    matches: (url) => url.startsWith("https://meet.google.com/"),
    idFromUrl: (url) => (url.match(/meet\.google\.com\/([a-z]{3}-[a-z]{4}-[a-z]{3})/) || [])[1] || null,
  },
  {
    api: "teams",
    matches: (url) => url.startsWith("https://teams.microsoft.com/") || url.startsWith("https://teams.live.com/"),
    idFromUrl: (url) => {
      const decoded = decodeURIComponent(url);
      const enterprise = decoded.match(/19:meeting_[A-Za-z0-9_-]+@thread\.v2/);
      if (enterprise) return enterprise[0].slice(0, 120);
      return (decoded.match(/teams\.live\.com\/meet\/(\d{6,16})/) || [])[1] || null;
    },
  },
  {
    api: "zoom",
    matches: (url) => /^https:\/\/([a-z0-9-]+\.)?zoom\.us\//.test(url),
    idFromUrl: (url) => (url.match(/\/wc\/(?:join\/)?(\d{8,12})/) || [])[1] || null,
  },
];

function platformForUrl(url) {
  if (!url) return null;
  return PLATFORMS.find((p) => p.matches(url)) || null;
}

function cleanTabTitle(title) {
  return (title || "")
    .replace(/^Meet\s*[–-]\s*/, "")
    .replace(/^\(\d+\)\s*/, "")
    .replace(/\s*[|]\s*Microsoft Teams.*$/i, "")
    .replace(/\s*[-|]\s*Zoom.*$/i, "")
    .trim();
}

const restored = restoreState();

async function restoreState() {
  try {
    const saved = await chrome.storage.session.get(["capturing", "sessionId", "tabId", "recordingStartMs"]);
    if (saved.sessionId) {
      state.capturing = !!saved.capturing;
      state.sessionId = saved.sessionId;
      state.tabId = saved.tabId ?? null;
      state.recordingStartMs = saved.recordingStartMs ?? null;
    }
  } catch (e) {}
  refreshBadge();
}

async function persistState() {
  try {
    await chrome.storage.session.set({
      capturing: state.capturing,
      sessionId: state.sessionId,
      tabId: state.tabId,
      recordingStartMs: state.recordingStartMs,
    });
  } catch (e) {}
}

// A tab that closes or navigates away never says goodbye through the content
// script (the document is gone), so clean up ourselves: drop it from the
// in-call set (stuck amber badge, field-observed) and stop a capture whose
// tab vanished.
chrome.tabs.onRemoved.addListener(async (tabId) => {
  await restored;
  const wasTracked = state.inCallTabs.delete(tabId);
  if (state.capturing && tabId === state.tabId) await stopCapture("tab closed");
  else if (wasTracked) refreshBadge();
});

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo) => {
  if (!changeInfo.url || platformForUrl(changeInfo.url)) return;
  await restored;
  const wasTracked = state.inCallTabs.delete(tabId);
  if (state.capturing && tabId === state.tabId) await stopCapture("navigated away");
  else if (wasTracked) refreshBadge();
});

// Chrome never (re)injects content scripts into tabs that were already open
// when the extension is installed, updated, or reloaded. Field test: a Meet
// tab from before the reload has no content script, so no call detection, no
// speaker names, no title. Inject into existing Meet tabs ourselves.
chrome.runtime.onInstalled.addListener(async () => {
  try {
    const tabs = await chrome.tabs.query({
      url: [
        "https://meet.google.com/*",
        "https://teams.microsoft.com/*",
        "https://teams.live.com/*",
        "https://*.zoom.us/*",
      ],
    });
    for (const tab of tabs) {
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true }, files: ["content.js"],
        });
      } catch (e) { /* chrome:// error pages etc. */ }
    }
  } catch (e) {}
});

async function cfg() {
  // Token lives in storage.local (this device only); fall back to sync once
  // for users upgrading from a build that stored it there.
  let { workerUrl, captureToken } = await chrome.storage.local.get(["workerUrl", "captureToken"]);
  if (!captureToken && !workerUrl) {
    const sync = await chrome.storage.sync.get(["workerUrl", "captureToken"]);
    if (sync.captureToken || sync.workerUrl) {
      workerUrl = sync.workerUrl;
      captureToken = sync.captureToken;
      try {
        await chrome.storage.local.set({ workerUrl, captureToken });
        await chrome.storage.sync.remove(["workerUrl", "captureToken"]);
      } catch (e) {}
    }
  }
  return { workerUrl: (workerUrl || "http://localhost:18088").replace(/\/$/, ""), captureToken: captureToken || "" };
}

async function api(path, options = {}) {
  const { workerUrl, captureToken } = await cfg();
  const response = await fetch(`${workerUrl}${path}`, {
    ...options,
    // A blackholed VPN (WireGuard-style) HANGS fetches rather than refusing
    // them; without a timeout one hung request freezes the single-flight
    // chunk pipeline with no backoff.
    signal: options.signal || AbortSignal.timeout(20000),
    headers: {
      Authorization: `Bearer ${captureToken}`,
      ...(options.headers || {}),
    },
  });
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}

// --- upload health (persisted: an SW restart mid-outage must not fake a
// fresh transition and re-toast every wake) --------------------------------
async function netState() {
  try {
    const v = await chrome.storage.session.get(["netDegraded", "netFails", "netDelivered"]);
    return { degraded: !!v.netDegraded, fails: v.netFails || 0, delivered: v.netDelivered || 0 };
  } catch (e) { return { degraded: false, fails: 0, delivered: 0 }; }
}

async function recordUploadFailure() {
  const s = await netState();
  const fails = s.fails + 1;
  const becameDegraded = !s.degraded && fails >= 2; // debounce single blips
  await chrome.storage.session.set({ netDegraded: s.degraded || becameDegraded, netFails: fails });
  if (becameDegraded) {
    chrome.notifications.create("fred-net-down", {
      type: "basic", iconUrl: "icons/fred128.png",
      title: "Fred: connection lost",
      message: "Can't reach the server (VPN?). Recording continues; audio is buffered and will catch up.",
    });
  }
}

async function recordUploadSuccess(queuedBefore) {
  const s = await netState();
  if (!s.degraded) {
    if (s.fails) await chrome.storage.session.set({ netFails: 0 });
    return;
  }
  const delivered = s.delivered + 1;
  if (queuedBefore <= 1) {
    // buffer fully drained: recovery is real, not the first lucky packet
    await chrome.storage.session.set({ netDegraded: false, netFails: 0, netDelivered: 0 });
    chrome.notifications.create("fred-net-up", {
      type: "basic", iconUrl: "icons/fred128.png",
      title: "Fred: caught up",
      message: `Connection is back; ${delivered} buffered chunk(s) delivered.`,
    });
  } else {
    await chrome.storage.session.set({ netFails: 0, netDelivered: delivered });
  }
}

function notify(title, message) {
  chrome.notifications.create({ type: "basic", iconUrl: "icons/fred128.png", title, message });
}

function setBadge(text, color) {
  chrome.action.setBadgeText({ text });
  if (color) chrome.action.setBadgeBackgroundColor({ color });
}

function refreshBadge() {
  if (state.busy) setBadge("...", "#E8A33D");
  else if (state.capturing) setBadge("REC", "#D93025");
  else if (state.inCallTabs.size > 0) setBadge("●", "#E8A33D");
  else if (updateAvailableVersion) setBadge("NEW", "#3B82F6");
  else setBadge("");
}

// --- update nudge -----------------------------------------------------------
// Unpacked extensions never auto-update, and a stale content script fails
// QUIETLY (old selectors -> recordings without speaker names). The server
// publishes the downloadable zip's version next to the zip itself
// (/static/fred-extension-version.json, written by package_extension.sh, so
// advertised === downloadable always); we poll it every few hours and badge
// "NEW" + notify once per version. Recording state always outranks the badge.
const UPDATE_CHECK_ALARM = "fred-update-check";
let updateAvailableVersion = null; // cached across SW wakeups via storage
chrome.storage.local.get("updateAvailableVersion").then((v) => {
  updateAvailableVersion = v.updateAvailableVersion || null;
  refreshBadge();
});

function versionNewer(a, b) {
  // true if a > b, comparing dotted numeric segments
  const pa = String(a).split(".").map(Number);
  const pb = String(b).split(".").map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d !== 0) return d > 0;
  }
  return false;
}

async function checkForUpdate() {
  const { workerUrl } = await chrome.storage.local.get("workerUrl");
  if (!workerUrl) return;
  let latest;
  try {
    const r = await fetch(`${workerUrl.replace(/\/$/, "")}/static/fred-extension-version.json`,
                          { cache: "no-store" });
    if (!r.ok) return; // direct-worker dev setups have no /static: stay quiet
    latest = String((await r.json()).version || "");
  } catch (e) {
    return; // offline / VPN off: try again next alarm
  }
  const mine = chrome.runtime.getManifest().version;
  if (latest && versionNewer(latest, mine)) {
    const firstTimeSeeing = updateAvailableVersion !== latest;
    updateAvailableVersion = latest;
    await chrome.storage.local.set({ updateAvailableVersion: latest });
    if (firstTimeSeeing) {
      // Distribution moved to the Chrome Web Store (unlisted), where Chrome
      // updates silently within hours, so the old text — unzip over the folder
      // and hit Reload — told store users to do pointless manual work every
      // time a release landed. Telling which install type this is would need
      // the "management" permission, and adding a permission to a published
      // extension triggers a re-review and a scary prompt, so the message
      // covers both cases instead of guessing.
      notify(
        `Fred extension v${latest} available`,
        `You have v${mine}. If you installed from the Chrome Web Store, Chrome updates it on its own within a few hours — nothing to do. Only a hand-installed zip needs replacing (Settings has the download).`
      );
    }
  } else {
    updateAvailableVersion = null;
    await chrome.storage.local.remove("updateAvailableVersion");
  }
  refreshBadge();
}

chrome.alarms.create(UPDATE_CHECK_ALARM, { delayInMinutes: 1, periodInMinutes: 360 });

// Keep in sync with CHAT_DEFAULTS in content.js and the Options page.
const NOTICE_DEFAULT_TEXT = "🔴 Fred Notetaker is recording this meeting to produce notes. · Fred Notetaker está grabando esta reunión para generar notas.";

// The session a page message belongs to: the live capture of that tab, or the
// one draining toward finish after a stop (closing speaker ENDs, a final chat
// scan, anything requeued during an outage).
async function sessionForTab(tabId) {
  if (state.capturing && tabId === state.tabId && state.sessionId) {
    return { sid: state.sessionId, anchor: state.recordingStartMs };
  }
  try {
    const s = await chrome.storage.session.get(["pendingFinish", "pendingFinishTabId",
                                                "pendingFinishAnchorMs"]);
    if (s.pendingFinish && tabId === s.pendingFinishTabId) {
      return { sid: s.pendingFinish, anchor: s.pendingFinishAnchorMs || null };
    }
  } catch (e) {}
  return null;
}

// Anchor offsets to the recorder start, all on this browser's clock. The
// worker must not derive them from its own clock (skew zeroes them).
function relativeTo(anchor, absoluteMs) {
  return anchor ? Math.max(0, (absoluteMs || Date.now()) - anchor) : undefined;
}

// The recorder's confirmation: what the meeting was told, right when it
// starts. A failure is loud because then nobody was told.
function notifyNoticeOutcome(status, reason) {
  if (status === "sent") {
    notify("Fred is recording", "Participants were notified in the meeting chat.");
  } else if (status === "disabled") {
    notify("Fred is recording", "The automatic chat notice is off in Fred's Options. Let participants know you're recording.");
  } else {
    chrome.notifications.create("fred-notice-failed", {
      type: "basic", iconUrl: "icons/fred128.png", priority: 2,
      title: "Fred is recording — please tell participants",
      message: status === "unsupported"
        ? "This page has no chat Fred can post the notice to."
        : `Fred couldn't post the notice in the meeting chat (${reason || "unknown reason"}).`,
    });
  }
}

// --- messages from content script + offscreen ---
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    await restored;
    if (msg.type === "call-state" && sender.tab) {
      if (msg.inCall) {
        const first = !state.inCallTabs.has(sender.tab.id);
        // Merge across frames: the meeting iframe knows the meet code, but the
        // TOP frame often has the human-readable tab title. Never let a frame
        // null out what another frame provided.
        const prev = state.inCallTabs.get(sender.tab.id) || {};
        state.inCallTabs.set(sender.tab.id, {
          platform: msg.platform || prev.platform,
          meetCode: msg.meetCode || prev.meetCode,
          title: msg.title || prev.title,
        });
        if (state.capturing && sender.tab.id === state.tabId) {
          state.lastCallHeartbeatMs = Date.now();
        }
        if (first && msg.transition && !state.capturing) {
          chrome.notifications.create(`fred-call-${sender.tab.id}`, {
            type: "basic",
            iconUrl: "icons/fred128.png",
            title: "Fred noticed a meeting",
            message: `${msg.title || msg.meetCode || "Google Meet"}. Click the Fred icon to record.`,
            priority: 1,
          });
        }
      } else {
        state.inCallTabs.delete(sender.tab.id);
        // The call ended in the tab we're capturing: stop automatically.
        if (state.capturing && sender.tab.id === state.tabId) await stopCapture("call ended");
      }
      refreshBadge();
      sendResponse({ capturing: state.capturing && sender.tab.id === state.tabId });
    } else if (msg.type === "capture-diagnostics" && sender.tab) {
      // forward the page's self-report to the worker so no-event captures are
      // debuggable from server logs alone (we can't see the user's console)
      console.log("page diagnostics", msg.reason, msg.data);
      if (state.sessionId) {
        try {
          await api(`/capture/${state.sessionId}/events`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ events: [], diagnostics: { reason: msg.reason, ...msg.data } }),
          });
        } catch (e) { console.warn("diagnostics upload failed", e); }
      }
      sendResponse({});
    } else if (msg.type === "speaker-events" && sender.tab) {
      // Post-stop window: the closing SPEAKER_ENDs (and anything requeued
      // during an outage) may arrive after teardown; accept them for the
      // session draining toward finish, from the tab that was captured.
      const target = await sessionForTab(sender.tab.id);
      const sid = target && target.sid;
      if (sid) {
        const events = msg.events.map((e) => ({
          ...e,
          relative_ts_ms: target.anchor ? Math.max(0, (e.absolute_ts_ms || 0) - target.anchor) : undefined,
        }));
        try {
          await api(`/capture/${sid}/events`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ events }),
          });
          sendResponse({ uploaded: true });
          return;
        } catch (e) {
          console.warn("event upload failed; content script requeues", e);
          sendResponse({ uploaded: false });
          return;
        }
      } else {
        console.warn("dropping speaker events", msg.events?.length, {
          capturing: state.capturing,
          tabMatch: sender.tab.id === state.tabId,
          session: state.sessionId,
        });
      }
      // uploaded:true on the drop path: these events can never become valid,
      // so the content script must discard, not requeue forever.
      sendResponse({ uploaded: true });
    } else if (msg.type === "chat-messages" && sender.tab) {
      const target = await sessionForTab(sender.tab.id);
      if (!target) {
        sendResponse({ uploaded: true }); // can never become valid: discard, don't requeue
        return;
      }
      const messages = (msg.messages || []).map((m) => ({
        ...m, relative_ts_ms: relativeTo(target.anchor, m.absolute_ts_ms),
      }));
      try {
        await api(`/capture/${target.sid}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages }),
        });
        sendResponse({ uploaded: true });
      } catch (e) {
        console.warn("chat upload failed; content script requeues", e);
        sendResponse({ uploaded: false });
      }
    } else if (msg.type === "recording-notice" && sender.tab) {
      sendResponse({});
      const target = await sessionForTab(sender.tab.id);
      if (!target) return;
      const body = JSON.stringify({
        status: msg.status, kind: msg.kind || "start", reason: msg.reason || null,
        relative_ts_ms: relativeTo(target.anchor, msg.absolute_ts_ms) ?? null,
      });
      // The audit record is the point: retry through a VPN blip.
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          await api(`/capture/${target.sid}/notice`, {
            method: "POST", headers: { "Content-Type": "application/json" }, body,
          });
          break;
        } catch (e) {
          console.warn("notice record failed", attempt, e);
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
      if ((msg.kind || "start") === "start") notifyNoticeOutcome(msg.status, msg.reason);
    } else if (msg.type === "offscreen-chunk") {
      // Heartbeat staleness check, evaluated on the 15s chunk cadence. Only
      // armed once the call was SEEN at least once this capture: platforms
      // where in-call detection never matches must not get killed at 60s.
      if (
        state.capturing &&
        state.lastCallHeartbeatMs &&
        Date.now() - state.lastCallHeartbeatMs > CALL_HEARTBEAT_TIMEOUT_MS
      ) {
        await stopCapture("call signal lost");
      }
      // Accept chunks for the LIVE session or the one draining toward finish
      // (post-stop retries): the force-finish teardown nulls state.sessionId,
      // and treating those as "stale" would flush the remaining buffer.
      let pendingFinishSid = null;
      try {
        pendingFinishSid = (await chrome.storage.session.get("pendingFinish")).pendingFinish;
      } catch (e) {}
      // binary rides base64 through the message channel (offscreen -> SW -> fetch)
      if (msg.sessionId && (msg.sessionId === state.sessionId || msg.sessionId === pendingFinishSid)) {
        try {
          const bytes = Uint8Array.from(atob(msg.b64), c => c.charCodeAt(0));
          const params = msg.partNonce != null ? `?part=${msg.partNonce}&seq=${msg.seq}` : `?seq=${msg.seq}`;
          await api(`/capture/${msg.sessionId}/chunk${params}`, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: bytes,
          });
          await recordUploadSuccess(msg.queued || 1);
          sendResponse({ uploaded: true });
        } catch (e) {
          // 4xx (except 408/429) will never succeed: session cap reached,
          // token revoked, session gone. Retrying those forever would churn
          // the buffer and lie in the "will catch up" toast.
          const status = Number((String(e).match(/HTTP (\d{3})/) || [])[1] || 0);
          const fatal = status >= 400 && status < 500 && status !== 408 && status !== 429;
          if (fatal) {
            console.warn("chunk upload permanently refused", status, e);
            chrome.notifications.create("fred-upload-fatal", {
              type: "basic", iconUrl: "icons/fred128.png",
              title: "Fred: upload refused",
              message: `The server refused audio (HTTP ${status}). Check your token / session limits.`,
            });
          } else {
            console.warn("chunk upload failed; offscreen will retry", e);
            await recordUploadFailure();
          }
          sendResponse({ uploaded: false, fatal });
        }
      } else {
        console.warn("dropping chunk from stale session", msg.sessionId);
        sendResponse({ uploaded: true }); // flush: it can never become valid
      }
    } else if (msg.type === "offscreen-draining") {
      // Post-stop drain heartbeat: push the force-finish alarm back and keep
      // the offscreen document alive; the 30s alarm reverts to its true
      // purpose (offscreen died SILENTLY).
      let pendingFinishSid = null;
      try {
        pendingFinishSid = (await chrome.storage.session.get("pendingFinish")).pendingFinish;
      } catch (e) {}
      if (msg.sessionId && msg.sessionId === pendingFinishSid) {
        state.lastDrainHeartbeatMs = Date.now();
        chrome.alarms.create(`fred-finish-${msg.sessionId}`, { delayInMinutes: 0.5 });
      }
      sendResponse({});
    } else if (msg.type === "offscreen-stopped") {
      if (msg.undelivered) {
        notify("Fred: some audio didn't upload",
               `${msg.undelivered} chunk(s) could not be delivered before the recording closed.`);
      }
      state.lastDrainHeartbeatMs = 0; // drain is over; the document may close
      await finishSession(msg.sessionId);
      sendResponse({});
    } else if (msg.type === "get-state") {
      sendResponse({ capturing: state.capturing, tabId: state.tabId });
    }
  })();
  return true; // async sendResponse
});

// --- action click: start/stop ---
chrome.action.onClicked.addListener(async (tab) => {
  await restored;
  if (state.busy) return; // a start/stop is already in flight; ignore the click
  state.busy = true;
  refreshBadge();
  try {
    if (state.capturing) await stopCapture("user clicked stop");
    else await startCapture(tab);
  } finally {
    state.busy = false;
    refreshBadge();
  }
});

// The content script does call detection, speaker names, title, and the
// chip. Make SURE it's in the page before recording: ping it, and if nothing
// answers (tab predates an update, SPA navigation, injection missed), inject
// it right now. Deterministic, no guessing about Chrome's injection rules.
async function ensureContentScript(tabId) {
  // allFrames matters: Zoom's web client renders the meeting UI inside an
  // iframe, where a top-frame-only script sees an empty shell (field: a whole
  // meeting with participants_visible=0 and zero speaker events).
  try {
    const r = await chrome.tabs.sendMessage(tabId, { type: "ping" });
    if (r && r.pong) return true;
  } catch (e) {}
  try {
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: true }, files: ["content.js"],
    });
    const r = await chrome.tabs.sendMessage(tabId, { type: "ping" });
    return !!(r && r.pong);
  } catch (e) {
    console.warn("content script injection failed", e);
    return false;
  }
}

async function startCapture(tab) {
  const platform = platformForUrl(tab.url);
  if (!platform) {
    notify("Fred", "Open a Google Meet, Teams, or Zoom (web client) tab first, then click Fred.");
    return;
  }
  const watcherAlive = await ensureContentScript(tab.id);
  if (!watcherAlive) {
    notify("Fred: speaker names may be missing",
      "The page watcher could not be loaded. Recording will work, but refresh the meeting tab if names matter.");
  }
  const info = state.inCallTabs.get(tab.id) || {};
  const tabTitle = cleanTabTitle(tab.title);
  let sessionId;
  let resumed = false;
  // Minted BEFORE the recorder exists and registered with the server in the
  // same start call: uniquely identifies THIS recorder's chunk stream, so the
  // server can drop a previous recorder's retry buffer draining late after a
  // same-meeting resume (it would otherwise corrupt the new part's webm).
  const partNonce = Date.now();
  let noticeEnabled = true;
  let noticeText = NOTICE_DEFAULT_TEXT;
  try {
    const prefs = await chrome.storage.local.get(["noticeEnabled", "noticeText"]);
    noticeEnabled = prefs.noticeEnabled !== false;
    if (typeof prefs.noticeText === "string" && prefs.noticeText.trim()) noticeText = prefs.noticeText.trim().slice(0, 500);
  } catch (e) {}
  try {
    const started = await api("/capture/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        platform: info.platform || platform.api,
        native_meeting_id: info.meetCode || platform.idFromUrl(tab.url),
        title: info.title || tabTitle || null,
        part_nonce: partNonce,
        // So the server knows which build produced a capture. Without it,
        // "has everyone updated?" could only be guessed from how long ago
        // the store published.
        ext_version: chrome.runtime.getManifest().version,
        // left and right of what offscreen.js records, for the server to use
        // them apart (v0.3.14)
        audio_channels: ["mic", "tab"],
        // what this recorder will tell the meeting (the audit starts here;
        // the per-attempt outcome follows on /capture/{id}/notice)
        notice_enabled: noticeEnabled,
        notice_text: noticeEnabled ? noticeText : null,
      }),
    });
    sessionId = started.session_id;
    resumed = !!started.resumed;
  } catch (e) {
    notify("Fred can't reach the worker", String(e).slice(0, 120));
    return;
  }

  try {
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
    await ensureOffscreen();
    const resp = await chrome.runtime.sendMessage({ type: "offscreen-start", streamId, sessionId, partNonce });
    if (!resp || !resp.ok) throw new Error(resp?.error || "offscreen start failed");
    state.capturing = true;
    state.sessionId = sessionId;
    state.tabId = tab.id;
    state.recordingStartMs = resp.startedAtMs || Date.now();
    state.lastCallHeartbeatMs = null; // armed by the first in-call report
    await persistState();
    chrome.tabs.sendMessage(tab.id, { type: "capture-started" }).catch(() => {});
    if (resumed) {
      notify("Same meeting", "Fred will add this recording to the meeting's existing notes.");
    }
    if (resp.micOk === false) {
      notify(
        "Recording WITHOUT your microphone",
        "Only tab audio is captured. Open Fred's Options and click 'Grant microphone access', then restart the recording."
      );
    }
  } catch (e) {
    console.error("start failed", e);
    notify("Fred: recording failed to start", String(e).slice(0, 120));
    try { await chrome.offscreen.closeDocument(); } catch (_) {}
  }
}

async function ensureOffscreen() {
  const has = await chrome.offscreen.hasDocument?.();
  if (!has) {
    await chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["USER_MEDIA"],
      justification: "Record meeting tab audio and microphone for note-taking",
    });
  }
}

async function stopCapture(reason) {
  console.log("stopping capture:", reason);
  const sessionId = state.sessionId;
  if (!sessionId) return;
  if (state.tabId) chrome.tabs.sendMessage(state.tabId, { type: "capture-stopped", reason }).catch(() => {});
  // Safety net: if the offscreen document never confirms (crashed, killed),
  // force-finish after 30s so the session is not stranded. Tab id + recorder
  // anchor ride along so post-stop speaker events (closing SPEAKER_ENDs,
  // outage requeues) can still be accepted and offset correctly.
  try {
    await chrome.storage.session.set({
      pendingFinish: sessionId,
      pendingFinishTabId: state.tabId,
      pendingFinishAnchorMs: state.recordingStartMs,
    });
  } catch (e) {}
  chrome.alarms.create(`fred-finish-${sessionId}`, { delayInMinutes: 0.5 });
  try {
    await chrome.runtime.sendMessage({ type: "offscreen-stop", sessionId });
    // finishSession runs when the offscreen doc confirms the last chunk went out.
  } catch (e) {
    // No offscreen document alive: finish with whatever was already uploaded.
    await finishSession(sessionId);
  }
}

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === UPDATE_CHECK_ALARM) return checkForUpdate();
  if (!alarm.name.startsWith("fred-finish-")) return;
  await restored;
  const sessionId = alarm.name.slice("fred-finish-".length);
  const { pendingFinish } = await chrome.storage.session.get("pendingFinish");
  if (pendingFinish === sessionId) {
    console.warn("offscreen never confirmed stop; force-finishing", sessionId);
    await finishSession(sessionId);
  }
});

const finishedSessions = new Set(); // stop paths can converge (tab close ->
// track-end AND onRemoved); a session must only finish once

async function finishSession(sessionId) {
  if (!sessionId) return;
  // Teardown runs once; the finish POST runs EVERY time. A session the
  // worker reopened (late buffered chunks arrived after a force-finish
  // already processed partial audio) needs this second finish to process the
  // FULL audio — server idempotency makes the repeat free otherwise.
  const firstTime = !finishedSessions.has(sessionId);
  finishedSessions.add(sessionId);
  if (firstTime) {
    const isCurrent = sessionId === state.sessionId;
    if (isCurrent) {
      state.capturing = false;
      state.sessionId = null;
      state.tabId = null;
      state.recordingStartMs = null;
      state.lastCallHeartbeatMs = null;
      await persistState();
      refreshBadge();
    }
  }
  try {
    await chrome.storage.session.remove(
      ["pendingFinish", "pendingFinishTabId", "pendingFinishAnchorMs"]
    );
  } catch (e) {}
  chrome.alarms.clear(`fred-finish-${sessionId}`).catch?.(() => {});
  try {
    const result = await api(`/capture/${sessionId}/finish`, { method: "POST" });
    if (result && result.status === "accepted") {
      notify("Fred is processing", "Notes will appear in Fred Notetaker in a few minutes.");
    }
  } catch (e) {
    if (/no audio/i.test(String(e))) {
      if (firstTime) notify("Fred: nothing recorded", "The recording was too short to save any audio.");
    } else {
      notify("Fred: finish failed", String(e).slice(0, 120));
    }
  }
  // Close the offscreen document only if no newer session is using it and no
  // post-stop drain is still delivering buffered chunks through it.
  const draining = state.lastDrainHeartbeatMs && Date.now() - state.lastDrainHeartbeatMs < 25000;
  if (!state.capturing && !draining) {
    try { await chrome.offscreen.closeDocument(); } catch (e) {}
  }
}
