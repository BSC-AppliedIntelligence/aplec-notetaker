// Aplec content script: call detection, named-speaker events, status chip.
// Multi-platform: Google Meet, Microsoft Teams (web), Zoom (web client).
// Each platform is an ADAPTER (call detection, meeting id, title, speaker
// detection); the engine (event queue, chip, diagnostics, SW messaging) is
// shared. Speaker selectors are ported from the production Vexa bot
// implementations (googlemeet/selectors.ts, msteams/selectors.ts,
// zoom/web/selectors.ts) so captures keep NAMED speakers.
//
// The whole script is guarded: the SW also injects it programmatically into
// tabs that were already open at install/update time (Chrome never does that
// itself), and a second injection into the same world must be a no-op.
(() => {
if (globalThis.__aplecContentLoaded) return;
globalThis.__aplecContentLoaded = true;

// ---------------------------------------------------------------- adapters

const GOOGLE_MEET = {
  api: "google_meet",
  mode: "observe",
  inCallSelectors: [
    'button[aria-label*="Leave call"]', 'button[aria-label*="leave call"]',
    'button[aria-label*="Salir de la llamada"]', 'button[aria-label*="Sair da chamada"]',
    'button[aria-label*="Anruf verlassen"]', 'button[aria-label*="Quitter l\'appel"]',
    '[data-tooltip*="Leave call"]',
  ],
  meetingId() {
    const m = location.pathname.match(/\/([a-z]{3}-[a-z]{4}-[a-z]{3})/);
    return m ? m[1] : null;
  },
  title() {
    // document.title is "Meet – <title or code>"
    return document.title.replace(/^Meet\s*[–-]\s*/, "").trim() || null;
  },
  // Ported verbatim from the production bot (vexa-bot googlemeet/selectors.ts):
  participantSelectors: [
    'div[data-participant-id]', '[data-participant-id]',
    '[aria-label*="participant"]', '[data-self-name]',
    '.participant-tile', '.video-tile',
  ],
  nameSelectors: [
    'span.notranslate', '[data-self-name]', '.zWGUib', '.cS7aqe.N2K3jd', '.XWGOtd',
    '[data-tooltip*="name"]', '[aria-label*="name"]',
    '.participant-name', '.display-name', '.user-name',
  ],
  // Meeting chat. Ported from vexa-bot services/chat.ts (Meet DOM verified
  // there 2024-2026), but preferring LOCALE-FREE signals: teams run Meet
  // in many languages, so aria-labels are only a fallback.
  chat: {
    capture: true,          // Meet in-call chat has no private messages
    openButtonSelectors: [
      'button[aria-label*="Chat with everyone"]', 'button[data-tooltip*="Chat"]',
      'button[aria-label*="chat" i]', 'button[aria-label*="xat" i]',
    ],
    // the chat box is the only textarea in the call UI
    inputSelectors: [
      'textarea[aria-label*="Send a message"]', 'textarea[aria-label*="message" i]', 'textarea',
    ],
    sendButtonSelectors: [
      'button[aria-label*="Send a message"]', 'button[aria-label*="Send" i]',
      'button[aria-label*="Enviar" i]', 'button[aria-label*="Envia" i]',
    ],
    // Meet renders messages ONLY while the chat panel is open (vexa chat.ts).
    scan() {
      const out = [];
      document.querySelectorAll("[data-message-id]").forEach((el) => {
        const key = el.getAttribute("data-message-id") || "";
        // pin-overlay buttons also carry data-message-id
        if (!key || el.classList.contains("VYBDae-Bz112c-LgbsSe") || el.closest(".Sd72u")) return;
        const textEl = el.querySelector(".jO4O1") || el.querySelector(".oIy2qc") || el.querySelector("[data-message-text]");
        if (!textEl) return;
        const clone = textEl.cloneNode(true);
        clone.querySelectorAll(".Sd72u, .VYBDae-Bz112c-LgbsSe, .ne2Ple-oshW8e-V67aGc, .UaaITe").forEach((n) => n.remove());
        const text = (clone.textContent || "").trim();
        if (!text) return;
        // sender header (.poVWob) is a sibling of the message wrapper
        let sender = null;
        let node = el;
        for (let depth = 0; depth < 6 && !sender && node.parentElement; depth++) {
          const parent = node.parentElement;
          for (const sib of parent.children) {
            if (sib === node) continue;
            const s = sib.querySelector(".poVWob");
            if (s && s.textContent.trim()) { sender = s.textContent.trim(); break; }
          }
          node = parent;
        }
        out.push({ key: `meet:${key}`, sender, text, links: linksIn(textEl), isPrivate: false });
      });
      return out;
    },
    selfName() {
      const el = document.querySelector("[data-self-name]");
      return el ? (el.getAttribute("data-self-name") || "").trim() || null : null;
    },
  },
  isSpeaking(el, mutatedClassList) {
    const SPEAKING = [
      'Oaajhc', 'HX2H7', 'wEsLMd', 'OgVli',
      'speaking', 'active-speaker', 'speaker-active', 'speaking-indicator',
      'audio-active', 'mic-active', 'microphone-active', 'voice-active',
    ];
    const SILENCE = [
      'gjg47c', 'silent', 'muted', 'mic-off', 'microphone-off',
      'audio-inactive', 'participant-silent', 'user-silent', 'no-audio',
    ];
    const cl = mutatedClassList || el.classList;
    const speaking =
      SPEAKING.some((c) => cl.contains(c)) ||
      SPEAKING.some((c) => el.querySelector("." + CSS.escape(c)));
    if (speaking) return true;
    if (SILENCE.some((c) => cl.contains(c))) return false;
    return false;
  },
};

const TEAMS = {
  api: "teams",
  mode: "observe",
  // Ported from vexa-bot msteams/selectors.ts (admission indicators):
  inCallSelectors: [
    'button#hangup-button',
    'button[data-tid="hangup-main-btn"]',
    'button[data-tid="hangup-button"]',
    '[role="toolbar"] button[aria-label*="Leave"]',
    'button[aria-label*="Leave"]',
  ],
  meetingId() {
    // Teams v2 shows the thread id ONLY in the join URL and rewrites it away
    // the moment the call starts, so by the time we ask, the id is usually
    // gone (field: a whole Teams capture arrived with an empty meeting id).
    // Remember it per tab — sessionStorage survives the SPA navigation — and
    // keep answering with it. When even the join URL never carried it, the
    // worker falls back to the meeting title (see capture.py resume_key).
    const KEY = "aplec_teams_meeting_id";
    const href = decodeURIComponent(location.href);
    const enterprise = href.match(/19:meeting_[A-Za-z0-9_-]+@thread\.v2/);
    const consumer = href.match(/teams\.live\.com\/meet\/(\d{6,16})/);
    const found = enterprise ? enterprise[0].slice(0, 120) : (consumer ? consumer[1] : null);
    try {
      if (found) sessionStorage.setItem(KEY, found);
      else return sessionStorage.getItem(KEY) || null;
    } catch (e) { /* storage blocked in this frame */ }
    return found;
  },
  title() {
    const t = document.title
      .replace(/^\(\d+\)\s*/, "")
      .replace(/\s*[|]\s*Microsoft Teams.*$/i, "")
      .trim();
    return t && !/^Microsoft Teams$/i.test(t) ? t : null;
  },
  // Ported from vexa-bot msteams/selectors.ts + recording.ts (the bot also
  // scans [role="menuitem"], its most reliable container):
  participantSelectors: [
    '[data-tid*="participant"]', '[aria-label*="participant"]',
    '[data-tid*="roster"]', '[data-tid*="roster-item"]',
    '[data-tid*="video-tile"]', '[data-tid*="videoTile"]',
    '[data-tid*="participant-tile"]', '[data-tid*="participantTile"]',
    '[role="menuitem"]',
  ],
  nameSelectors: [
    '[data-tid*="display-name"]', '[data-tid*="participant-name"]',
    '[data-tid*="user-name"]', '[aria-label*="name"]',
    '.participant-name', '.display-name',
  ],
  // Teams: the recording notice is posted, but the chat is NOT read. The
  // message selectors the bot uses ([data-tid*="chat-pane-message"]) also
  // match PRIVATE conversations when the user opens the Chat app during a
  // call (the meeting shrinks to a mini window and the main pane shows other
  // chats), and without a sample of the real DOM there is no safe way to tell
  // the meeting's chat apart.
  chat: {
    capture: false,
    openButtonSelectors: [
      '#chat-button', 'button[data-tid*="chat-button"]', 'button[data-tid*="chat"]',
      'button[aria-label*="Chat" i]',
    ],
    inputSelectors: [
      '[data-tid="ckeditor"] [contenteditable="true"]', '[contenteditable="true"][data-tid*="message"]',
      '[contenteditable="true"][aria-label*="message" i]', 'div[role="textbox"][contenteditable="true"]',
    ],
    sendButtonSelectors: [
      'button[data-tid="newMessageCommands-send"]', 'button[name="send"]',
      'button[aria-label*="Send" i]', 'button[aria-label*="Enviar" i]',
    ],
    scan() { return []; },
  },
  // Teams' voice-level signal FLICKERS: it pulses with speech energy instead
  // of staying lit through a turn (field: 212 events averaging 1.3s = 7%
  // coverage, under the naming vote's 25% gate). Hold END events open so
  // sub-2s gaps merge into continuous speech intervals.
  endHoldMs: 2000,
  // Teams signal (bot's TeamsSpeakingDetector): a voice-level outline element
  // exists per speaking-capable tile; the vdi-frame-occlusion class on it or
  // any ancestor means SPEAKING.
  isSpeaking(el) {
    const outline = el.querySelector('[data-tid="voice-level-stream-outline"]');
    if (!outline) return false;
    let current = outline;
    while (current) {
      if (current.classList && current.classList.contains("vdi-frame-occlusion")) return true;
      current = current.parentElement;
    }
    return false;
  },
};

const ZOOM = {
  api: "zoom",
  // Zoom web exposes ONE active speaker at a time (no per-tile signal), so
  // the engine polls instead of observing (bot does the same at 250ms).
  mode: "active-speaker-poll",
  pollMs: 400,
  // Ported from vexa-bot zoom/web/selectors.ts (in-meeting indicators):
  inCallSelectors: [
    'button.join-audio-container__btn',
    'button[aria-label="Leave"]',
    'button[class*="footer-button"][aria-label*="Leave"]',
  ],
  meetingId() {
    const m = location.pathname.match(/\/wc\/(?:join\/)?(\d{8,12})/);
    return m ? m[1] : null;
  },
  title() {
    const t = document.title.replace(/\s*[-|]\s*Zoom.*$/i, "").trim();
    return t && !/^Zoom( Meeting)?$/i.test(t) ? t : null;
  },
  activeSpeakerName() {
    const nameFrom = (container) => {
      if (!container) return null;
      const footer = container.querySelector(".video-avatar__avatar-footer");
      if (!footer) return null;
      const span = footer.querySelector("span");
      const text = (span && span.textContent) || footer.innerText || "";
      return text.trim() || null;
    };
    // Layout 1: normal view, dedicated active-speaker container.
    // Layout 2: screen-share view, tile with the --active modifier.
    return (
      nameFrom(document.querySelector(".speaker-active-container__video-frame")) ||
      nameFrom(document.querySelector(".speaker-bar-container__video-frame--active"))
    );
  },
  participantsVisible() {
    return document.querySelectorAll(".video-avatar__avatar").length;
  },
  // Zoom web chat (DOM verified in vexa chat.ts: TipTap/ProseMirror editor,
  // .new-chat-message__container per message, .chat-item__sender above).
  chat: {
    capture: true,
    openButtonSelectors: [
      'button[aria-label*="open the chat panel" i]', 'button[aria-label*="chat panel" i]',
      'button[aria-label*="Chat" i]', 'button[class*="chat-btn"]',
    ],
    inputSelectors: [
      '.chat-rtf-box__editor-outer [contenteditable="true"]', '.tiptap.ProseMirror',
      '[class*="chat-box"] [contenteditable="true"]',
    ],
    sendButtonSelectors: [
      'button[class*="chat-rtf-box__send"]', 'button[aria-label="send"]', 'button[aria-label*="send" i]',
    ],
    scan() {
      const out = [];
      document.querySelectorAll(".new-chat-message__container").forEach((el) => {
        const textEl = el.querySelector(".chat-rtf-box__display") || el.querySelector(".new-chat-message__content");
        const text = ((textEl && textEl.textContent) || "").trim();
        if (!text) return;
        let item = el.parentElement;
        let senderEl = null;
        for (let i = 0; i < 8 && item && !senderEl; i++) {
          senderEl = item.querySelector(".chat-item__sender");
          if (!senderEl) item = item.parentElement;
        }
        const sender = senderEl ? senderEl.textContent.trim() || null : null;
        // Zoom lists direct messages in the same panel. Never capture them:
        // only what was said to everyone belongs in the meeting record.
        // Look at the sender's header line only ("Ana to Everyone" vs "… to
        // Me (Direct Message)"), never the message body: a public message
        // that merely says "privately" must not be dropped.
        const headerEl = senderEl ? (senderEl.closest(".chat-item__chat-info") || senderEl.parentElement) : null;
        const header = headerEl ? (headerEl.textContent || "").replace(text, "").slice(0, 300) : "";
        const isPrivate = !!(
          (item && item.querySelector('[class*="private" i]')) ||
          /direct message|privately|\(privado\)|en privado|mensaje directo|mensagem direta|privadamente|en privat|missatge directe/i.test(header)
        );
        const id = el.getAttribute("id") || el.getAttribute("data-id") || "";
        out.push({ key: id ? `zoom:${id}` : `zoom:${sender}|${text}`, sender, text, links: linksIn(textEl), isPrivate });
      });
      return out;
    },
    selfName() { return null; },
  },
  // Per-layout presence counts: if names fail again, the diagnostics say
  // exactly which container Zoom rendered (speaker view, screen-share bar,
  // gallery) in the frame that has the meeting DOM.
  diagnosticsExtra() {
    const count = (sel) => {
      try { return document.querySelectorAll(sel).length; } catch (e) { return -1; }
    };
    return {
      zoom_speaker_active: count(".speaker-active-container__video-frame"),
      zoom_speaker_bar: count(".speaker-bar-container__video-frame"),
      zoom_bar_active: count(".speaker-bar-container__video-frame--active"),
      zoom_gallery: count(".gallery-video-container__video-frame"),
      zoom_avatars: count(".video-avatar__avatar"),
      zoom_footer_leave: count('button[aria-label="Leave"]'),
    };
  },
};

function pickAdapter() {
  const host = location.hostname;
  if (host === "meet.google.com") return GOOGLE_MEET;
  if (host === "teams.microsoft.com" || host === "teams.live.com") return TEAMS;
  if (host === "zoom.us" || host.endsWith(".zoom.us")) return ZOOM;
  return null;
}

const ADAPTER = pickAdapter();
if (!ADAPTER) return;

// ------------------------------------------------------------------ engine

// Frame-local call state. With all_frames injection (Zoom renders its meeting
// inside an iframe) MANY frames run this script; only the one that actually
// sees the meeting DOM ever reports inCall:true (re-sent every poll = the
// SW's heartbeat), and a frame only reports inCall:false after IT was in the
// call (true -> false transition). Unsolicited "false" from empty shells must
// never reach the SW: it would kill an active capture at record start. The
// host-ended-meeting case (document swap kills the in-call frame before it
// can report false) is covered SW-side by heartbeat timeout.
let inCall = false;
let capturing = false;
let respOffStreak = 0;
let eventQueue = [];
let speakingStates = new Map(); // participantId -> bool
let observer = null;
let activePoll = null;
let lastActiveSpeaker = null;
let chip = null;
let diag = { events_pushed: 0, ids_seen: new Set() };
const lastKnownName = new Map(); // participantId -> display name

function meetingId() {
  try { return ADAPTER.meetingId(); } catch (e) { return null; }
}

function meetingTitle() {
  try { return ADAPTER.title(); } catch (e) { return null; }
}

function isInCall() {
  return ADAPTER.inCallSelectors.some((sel) => {
    try { return document.querySelector(sel); } catch (e) { return false; }
  });
}

// --- participant name resolution (observer mode) ---
function participantName(el) {
  for (const sel of ADAPTER.nameSelectors || []) {
    let node = null;
    try { node = el.querySelector(sel); } catch (e) { continue; }
    const text = node && (node.textContent || node.getAttribute("data-self-name") || "");
    if (text && text.trim().length > 1 && text.trim().length < 80) return text.trim();
  }
  const self = el.getAttribute && el.getAttribute("data-self-name");
  if (self) return self.trim();
  const aria = el.getAttribute && el.getAttribute("aria-label");
  if (aria && aria.length > 1 && aria.length < 80) return aria.trim();
  return null;
}

function participantId(el) {
  return el.getAttribute("data-participant-id") || participantName(el) || "unknown";
}

function participantContainer(node) {
  if (!(node instanceof HTMLElement)) return null;
  for (const sel of ADAPTER.participantSelectors || []) {
    try {
      const container = node.closest(sel);
      if (container) return container;
    } catch (e) { continue; }
  }
  return null;
}

function participantsVisible() {
  if (ADAPTER.participantsVisible) {
    try { return ADAPTER.participantsVisible(); } catch (e) { return 0; }
  }
  const seen = new Set();
  for (const sel of ADAPTER.participantSelectors || []) {
    try { document.querySelectorAll(sel).forEach((el) => seen.add(el)); } catch (e) {}
  }
  return seen.size;
}

// A tile whose name node cannot be read has yielded the participant id itself
// ("spaces/<space>/devices/<n>") or one of its path components ("devices") as a
// name. Worker-side that counts as an extra identity and inflates the
// diarization floor, splitting a real voice in two, so never emit it: prefer the
// last name we did read for this id, else drop the event.
function isIdDerivedName(name, id) {
  if (!id || id.indexOf("/") === -1) return false;
  if (name === id) return true;
  return id.split("/").some((part) => part && part === name);
}

function pushEvent(type, name, id) {
  if (name && isIdDerivedName(name, id)) {
    name = lastKnownName.get(id) || null;
    diag.id_derived_names = (diag.id_derived_names || 0) + 1;
  }
  if (!name) return;
  lastKnownName.set(id, name);
  diag.events_pushed += 1;
  diag.ids_seen.add(id);
  eventQueue.push({
    event_type: type,
    participant_name: name,
    participant_id: id,
    absolute_ts_ms: Date.now(),
  });
}

// --- observer engine (Meet, Teams: per-participant speaking signal) ---
const pendingEnds = new Map(); // participantId -> {timer, name} (endHoldMs merge)

function evaluateContainer(el, mutatedClassList) {
  const id = participantId(el);
  const was = speakingStates.get(id) || false;
  let now = false;
  try { now = !!ADAPTER.isSpeaking(el, mutatedClassList); } catch (e) { now = false; }
  const holdMs = ADAPTER.endHoldMs || 0;
  if (now && !was) {
    speakingStates.set(id, true);
    const pending = pendingEnds.get(id);
    if (pending) {
      // signal resumed within the hold window: same turn continues, no events
      clearTimeout(pending.timer);
      pendingEnds.delete(id);
    } else {
      pushEvent("SPEAKER_START", participantName(el), id);
    }
  } else if (!now && was) {
    speakingStates.set(id, false);
    const name = participantName(el) || lastKnownName.get(id);
    if (holdMs > 0) {
      const timer = setTimeout(() => {
        pendingEnds.delete(id);
        pushEvent("SPEAKER_END", name, id);
      }, holdMs);
      pendingEnds.set(id, { timer, name });
    } else {
      pushEvent("SPEAKER_END", name, id);
    }
  }
}

function startObserverEngine() {
  speakingStates = new Map();
  observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      const container = participantContainer(m.target);
      if (container) evaluateContainer(container, m.target instanceof HTMLElement ? m.target.classList : undefined);
    }
  });
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["class", "style", "aria-hidden", "data-audio-level"],
    subtree: true,
  });
  // periodic full scan as a safety net (bot does the same)
  observer._scan = setInterval(() => {
    const seen = new Set();
    for (const sel of ADAPTER.participantSelectors || []) {
      let els = [];
      try { els = document.querySelectorAll(sel); } catch (e) { continue; }
      els.forEach((el) => {
        const id = participantId(el);
        if (seen.has(id)) return;
        seen.add(id);
        evaluateContainer(el);
      });
    }
  }, 3000);
}

function stopObserverEngine() {
  if (observer) {
    observer.disconnect();
    clearInterval(observer._scan);
    observer = null;
  }
  // flush held ENDs immediately (the hold window must not outlive the capture)
  pendingEnds.forEach(({ timer, name }, id) => {
    clearTimeout(timer);
    if (name) pushEvent("SPEAKER_END", name, id);
  });
  pendingEnds.clear();
  // close any open speaking states; a raw id is useless to the name vote,
  // so only emit when we know the display name
  speakingStates.forEach((speaking, id) => {
    const name = lastKnownName.get(id);
    if (speaking && name) pushEvent("SPEAKER_END", name, id);
  });
  speakingStates = new Map();
}

// --- active-speaker poll engine (Zoom web: one speaker at a time) ---
function startPollEngine() {
  lastActiveSpeaker = null;
  activePoll = setInterval(() => {
    let name = null;
    try { name = ADAPTER.activeSpeakerName(); } catch (e) { name = null; }
    if (name === lastActiveSpeaker) return;
    if (lastActiveSpeaker) pushEvent("SPEAKER_END", lastActiveSpeaker, lastActiveSpeaker);
    if (name) pushEvent("SPEAKER_START", name, name);
    lastActiveSpeaker = name;
  }, ADAPTER.pollMs || 400);
}

function stopPollEngine() {
  clearInterval(activePoll);
  activePoll = null;
  if (lastActiveSpeaker) {
    pushEvent("SPEAKER_END", lastActiveSpeaker, lastActiveSpeaker);
    lastActiveSpeaker = null;
  }
}

function startSpeakerObservation() {
  stopSpeakerObservation();
  if (ADAPTER.mode === "active-speaker-poll") startPollEngine();
  else startObserverEngine();
}

function stopSpeakerObservation() {
  if (ADAPTER.mode === "active-speaker-poll") stopPollEngine();
  else stopObserverEngine();
}

// --- diagnostics: let the WORKER see what this page looks like, so a test
// that yields no speaker events is debuggable from server logs alone ---
function sendDiagnostics(reason) {
  chrome.runtime.sendMessage({
    type: "capture-diagnostics",
    reason,
    data: {
      version: chrome.runtime.getManifest().version,
      platform: ADAPTER.api,
      mode: ADAPTER.mode,
      in_call: !!isInCall(),
      participants_visible: participantsVisible(),
      events_pushed: diag.events_pushed,
      distinct_participants: diag.ids_seen.size,
      queue_length: eventQueue.length,
      url_has_code: !!meetingId(),
      chat_messages: diag.chat_messages || 0,
      chat_queue: chatQueue.length,
      notice_status: lastNoticeStatus,
      doc_title: String(document.title).slice(0, 60),
      // is this the top window or an embedded iframe? (Zoom renders its meeting
      // UI in a frame; a top-only script sees an empty shell)
      is_top_frame: window === window.top,
      frame_host: location.host,
      ...(ADAPTER.diagnosticsExtra ? ADAPTER.diagnosticsExtra() : {}),
    },
  }).catch(() => {});
}

// ------------------------------------------------------- meeting chat engine
// Two features share the meeting's chat panel: the RECORDING NOTICE (posted
// once, when a capture starts) and CHAT CAPTURE (messages and links read from
// the panel). Both run only in the frame that is actually in the call: this
// script lives in every frame (Zoom's meeting is an iframe), and an empty
// frame must neither post nor report a failed notice.
//
// 0.3.12 also re-posted the notice when a participant tile it had not seen
// appeared, for late joiners. Removed in 0.3.13: in Meet a screen share is a
// tile of its own (a new data-participant-id), and tiles appear for other
// reasons too, so the notice went out again with nobody new in the call.

const CHAT_DEFAULTS = {
  noticeEnabled: true,
  noticeText: "🔴 Aplec Notetaker is recording this meeting to produce notes. · Aplec Notetaker está grabando esta reunión para generar notas.",
  chatCapture: true,
};
// the recorder's own messages show under a localized "You"
const SELF_LABELS = /^(you|tú|tu|você|vous|du|jo|yo|eu|ich)$/i;

let chatSettings = { ...CHAT_DEFAULTS };
let chatQueue = [];
let chatSeen = new Set();
let chatScanTimer = null;
let chatFeaturesStarted = false;
let noticeBusy = false;
let lastNoticeStatus = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function linksIn(el) {
  const links = [];
  if (!el) return links;
  el.querySelectorAll("a[href]").forEach((a) => {
    const href = a.href || a.getAttribute("href") || "";
    if (/^https?:\/\//i.test(href) && !links.includes(href)) links.push(href);
  });
  return links;
}

function isVisible(el) {
  return !!(el && el.isConnected && el.getClientRects().length);
}

function firstVisible(selectors) {
  for (const sel of selectors || []) {
    let nodes = [];
    try { nodes = document.querySelectorAll(sel); } catch (e) { continue; }
    for (const n of nodes) if (isVisible(n)) return n;
  }
  return null;
}

async function loadChatSettings() {
  try {
    const v = await chrome.storage.local.get(Object.keys(CHAT_DEFAULTS));
    const text = typeof v.noticeText === "string" ? v.noticeText.trim() : "";
    chatSettings = {
      noticeEnabled: v.noticeEnabled !== false,
      noticeText: (text || CHAT_DEFAULTS.noticeText).slice(0, 500),
      chatCapture: v.chatCapture !== false,
    };
  } catch (e) {
    chatSettings = { ...CHAT_DEFAULTS };
  }
}

async function openChatPanel() {
  const chat = ADAPTER.chat;
  if (firstVisible(chat.inputSelectors)) return true;
  const button = firstVisible(chat.openButtonSelectors);
  if (!button) return false;
  button.click();
  for (let i = 0; i < 20; i++) {   // up to ~4s for the panel to render
    await sleep(200);
    if (firstVisible(chat.inputSelectors)) return true;
  }
  return false;
}

function inputText(input) {
  return input.tagName === "TEXTAREA" || input.tagName === "INPUT" ? input.value : (input.textContent || "");
}

function setInputText(input, text) {
  input.focus();
  if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
    // From the extension's isolated world this IS the native setter, so a
    // framework-controlled field still registers the change on "input".
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    return;
  }
  // Rich editors (Teams' CKEditor, Zoom's ProseMirror) ignore plain DOM writes;
  // insertText goes through their input pipeline the way typing does. (The
  // bot types with Playwright's real keyboard, which an extension cannot do.)
  try {
    document.execCommand("selectAll", false);
    if (!text) { document.execCommand("delete", false); return; }
    if (document.execCommand("insertText", false, text)) return;
  } catch (e) { /* fall through */ }
  input.textContent = text;
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function pressEnter(input) {
  for (const type of ["keydown", "keypress", "keyup"]) {
    input.dispatchEvent(new KeyboardEvent(type, {
      key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true,
    }));
  }
}

// Sent = the box emptied (it may have been re-rendered: look it up again) or
// the message showed up in the chat list.
async function messageWentOut(input, text) {
  for (let i = 0; i < 10; i++) {   // ~2s
    await sleep(200);
    const box = input.isConnected ? input : firstVisible(ADAPTER.chat.inputSelectors);
    if (box && !inputText(box).trim()) return true;
    try {
      if ((ADAPTER.chat.scan() || []).some((m) => m.text === text)) return true;
    } catch (e) {}
  }
  return false;
}

async function postChatMessage(text) {
  const chat = ADAPTER.chat;
  if (!(await openChatPanel())) return { status: "failed", reason: "chat panel not found" };
  const input = firstVisible(chat.inputSelectors);
  if (!input) return { status: "failed", reason: "chat input not found" };
  setInputText(input, text);
  await sleep(250);
  const send = firstVisible(chat.sendButtonSelectors);
  const sendUsable = send && !send.disabled && send.getAttribute("aria-disabled") !== "true";
  if (sendUsable) send.click();
  else pressEnter(input);
  if (await messageWentOut(input, text)) return { status: "sent" };
  pressEnter(input);
  if (await messageWentOut(input, text)) return { status: "sent" };
  // Never leave our text in the box for the user to send by accident later.
  const box = input.isConnected ? input : firstVisible(chat.inputSelectors);
  if (box) setInputText(box, "");
  return { status: "failed", reason: sendUsable ? "message did not go out" : "send button not found" };
}

function reportNotice(status, kind, reason) {
  lastNoticeStatus = status;
  chrome.runtime.sendMessage({
    type: "recording-notice", status, kind, reason: reason || null, absolute_ts_ms: Date.now(),
  }).catch(() => {});
}

async function postNotice() {
  if (noticeBusy) return;
  noticeBusy = true;
  try {
    let result = { status: "failed", reason: "not attempted" };
    // Retry only while the chat UI is still appearing (right after joining).
    // A send that failed will fail again, and each retry would put the text
    // back in the user's message box for several more seconds.
    for (let i = 0; i < 3 && capturing; i++) {
      result = await postChatMessage(chatSettings.noticeText);
      const stillLoading = result.reason === "chat panel not found" || result.reason === "chat input not found";
      if (result.status === "sent" || !stillLoading) break;
      await sleep(3000);
    }
    if (!capturing && result.status !== "sent") return;   // stopped mid-attempt
    reportNotice(result.status, "start", result.reason);
  } finally {
    noticeBusy = false;
  }
}

function isOwnNotice(text) {
  const norm = (s) => String(s || "").replace(/\s+/g, " ").trim();
  return norm(text) === norm(chatSettings.noticeText);
}

// Queue messages not seen before. The baseline pass (capture start) only
// marks what is already in the panel as seen: those were written before the
// recording — and before the notice — so they are not part of the record.
function scanChat(baseline = false) {
  const chat = ADAPTER.chat;
  if (!chat || !chat.capture) return;
  let found = [];
  try { found = chat.scan() || []; } catch (e) { return; }
  let self = null;
  try { self = chat.selfName ? chat.selfName() : null; } catch (e) {}
  for (const m of found) {
    if (!m || !m.text || chatSeen.has(m.key)) continue;
    chatSeen.add(m.key);
    if (baseline || m.isPrivate || isOwnNotice(m.text)) continue;
    let sender = m.sender;
    if (!sender || SELF_LABELS.test(sender)) sender = self || sender || "Unknown";
    chatQueue.push({
      msg_key: m.key, sender, text: m.text.slice(0, 4000),
      links: (m.links || []).slice(0, 20), absolute_ts_ms: Date.now(),
    });
    diag.chat_messages = (diag.chat_messages || 0) + 1;
  }
}

async function maybeStartChatFeatures() {
  if (!capturing || chatFeaturesStarted || !isInCall()) return;
  chatFeaturesStarted = true;
  await loadChatSettings();
  if (!ADAPTER.chat) {
    reportNotice("unsupported", "start", "platform has no chat support");
    return;
  }
  // Capture needs the panel open (Meet renders messages only then); open it
  // and take the baseline BEFORE posting, so our own notice and the earlier
  // history are both excluded.
  if (chatSettings.chatCapture && ADAPTER.chat.capture) {
    await openChatPanel();
    // stopped while the panel was opening: create nothing that outlives it
    if (!capturing || !chatFeaturesStarted) return;
    scanChat(true);
    chatScanTimer = setInterval(() => { if (capturing) scanChat(); }, 2000);
  }
  if (chatSettings.noticeEnabled) await postNotice();
  else reportNotice("disabled", "start", null);
}

function stopChatFeatures() {
  if (chatFeaturesStarted && ADAPTER.chat && ADAPTER.chat.capture && chatSettings.chatCapture) scanChat();
  clearInterval(chatScanTimer);
  chatScanTimer = null;
  chatFeaturesStarted = false;
  chatSeen = new Set();
}

async function flushChat() {
  if (!chatQueue.length) return;
  const messages = chatQueue.splice(0, 200);
  let resp = null;
  try {
    resp = await chrome.runtime.sendMessage({ type: "chat-messages", messages });
  } catch (e) { /* SW asleep/restarting: requeue below */ }
  if (!resp || resp.uploaded !== true) {
    chatQueue.unshift(...messages);
    if (chatQueue.length > 2000) chatQueue.splice(0, chatQueue.length - 2000);
  }
}

// flush speaker events every 5s while capturing. Failed uploads REQUEUE
// (VPN outage must not eat who-spoke-when); the SW answers uploaded:true for
// events that can never become valid, so dead events don't cycle forever.
async function flushEvents() {
  if (!eventQueue.length) return;
  const events = eventQueue.splice(0, 400);
  let resp = null;
  try {
    resp = await chrome.runtime.sendMessage({ type: "speaker-events", events });
  } catch (e) { /* SW asleep/restarting: requeue below */ }
  if (!resp || resp.uploaded !== true) {
    eventQueue.unshift(...events);
    if (eventQueue.length > 3000) eventQueue.splice(0, eventQueue.length - 3000);
  }
}

setInterval(() => {
  if (capturing && eventQueue.length) flushEvents();
  if (capturing && chatQueue.length) flushChat();
}, 5000);

// After a stop, the 5s flush is gated off `capturing`, but the closing
// SPEAKER_ENDs (and anything requeued during an outage) still need a way
// out: bounded retry, 10s cadence for ~2 minutes, matching the SW's
// post-stop acceptance window.
let postStopFlushTimer = null;
function startPostStopFlush() {
  if (postStopFlushTimer) clearInterval(postStopFlushTimer);
  let attempts = 12;
  postStopFlushTimer = setInterval(() => {
    if ((!eventQueue.length && !chatQueue.length) || attempts-- <= 0) {
      clearInterval(postStopFlushTimer);
      postStopFlushTimer = null;
      return;
    }
    flushEvents();
    flushChat();
  }, 10000);
  flushEvents();
  flushChat();
}

// call-state poll every 3s. While in a call we re-send every poll (the SW's
// heartbeat + restarted-SW recovery; transition:false keeps the resend from
// re-firing the notification). A not-in-call frame that was never in the
// call stays silent.
setInterval(async () => {
  const now = isInCall();
  const transition = now !== inCall;
  inCall = now;
  if (!inCall && !transition) return;   // never-in-call frame: stays silent
  try {
    const resp = await chrome.runtime.sendMessage({
      type: "call-state", inCall: !!inCall, transition,
      platform: ADAPTER.api, meetCode: meetingId(), title: meetingTitle(),
    });
    // Only the service worker's answer carries an explicit boolean; anything
    // else (another listener won the response race) is NO information.
    if (!resp || typeof resp.capturing !== "boolean") return;
    // Resync with the SW, but turning observation OFF needs two consecutive
    // polls: a single stale/racing response must not kill speaker capture
    // (field-observed: observation died 1.5s into a recording).
    if (resp.capturing && !capturing) {
      respOffStreak = 0;
      setCapturing(true, "call-state-resp");
    } else if (!resp.capturing && capturing) {
      respOffStreak += 1;
      if (respOffStreak >= 2) { respOffStreak = 0; setCapturing(false, "call-state-resp"); }
    } else {
      respOffStreak = 0;
    }
    // A frame whose call detection came after the capture started (Zoom's
    // meeting iframe can lag the top frame) starts the chat features now.
    if (capturing && inCall) maybeStartChatFeatures();
  } catch (e) { /* SW asleep */ }
}, 3000);

function setCapturing(on, source) {
  const was = capturing;
  capturing = on;
  respOffStreak = 0;
  if (on && !was) {
    diag = { events_pushed: 0, ids_seen: new Set() };
    startSpeakerObservation();
    showChip();
    maybeStartChatFeatures();
    setTimeout(() => { if (capturing) sendDiagnostics("start+8s"); }, 8000);
  } else if (!on && was) {
    stopSpeakerObservation(); // pushes closing SPEAKER_ENDs
    stopChatFeatures();       // final chat scan
    if (eventQueue.length || chatQueue.length) startPostStopFlush();
    hideChip();
    sendDiagnostics(`stop:${source || "?"}`);
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // Same rule as the offscreen doc: only answer messages addressed to the
  // page watcher, or our {} could outrace another listener's real response.
  if (msg.type === "capture-started") { setCapturing(true, "sw:start"); sendResponse({}); return; }
  if (msg.type === "capture-stopped") { setCapturing(false, `sw:${msg.reason || "stop"}`); sendResponse({}); return; }
  if (msg.type === "ping") { sendResponse({ pong: true, inCall: !!isInCall(), platform: ADAPTER.api }); return; }
  return false;
});

// --- status chip ---
function showChip() {
  if (chip) return;
  chip = document.createElement("div");
  chip.textContent = "● Aplec is recording";
  Object.assign(chip.style, {
    position: "fixed", bottom: "88px", right: "16px", zIndex: 999999,
    background: "rgba(20,17,13,.92)", color: "#F4EFE7",
    border: "1px solid #E8A33D", borderRadius: "999px",
    padding: "6px 14px", font: "600 12px system-ui", pointerEvents: "none",
  });
  document.body.appendChild(chip);
}
function hideChip() { chip?.remove(); chip = null; }
})();
