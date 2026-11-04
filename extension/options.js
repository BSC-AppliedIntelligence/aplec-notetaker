const $ = (id) => document.getElementById(id);

// The capture token is a long-lived bearer credential, so it lives in
// chrome.storage.LOCAL (this device only), never chrome.storage.sync (which
// replicates it to Google's servers across all the user's Chrome installs).
// One-time migration: read from local, fall back to a value an older build
// left in sync, then scrub sync.
async function loadConfig() {
  const local = await chrome.storage.local.get(["workerUrl", "captureToken"]);
  if (local.captureToken || local.workerUrl) return local;
  const sync = await chrome.storage.sync.get(["workerUrl", "captureToken"]);
  if (sync.captureToken || sync.workerUrl) {
    await chrome.storage.local.set(sync);
    await chrome.storage.sync.remove(["workerUrl", "captureToken"]);
  }
  return sync;
}

// The production deployment: the web app's hostname proxies /capture/* to the
// worker, so the site URL is the worker URL. Dev setups (SSH tunnel straight
// to the worker) type http://localhost:18088 over it.
const DEFAULT_WORKER_URL = "https://fred.example.org";  // the site, written in by scripts/package_extension.sh

loadConfig().then(({ workerUrl, captureToken }) => {
  $("workerUrl").value = workerUrl || DEFAULT_WORKER_URL;
  $("captureToken").value = captureToken || "";
});

$("save").addEventListener("click", async () => {
  const workerUrl = $("workerUrl").value.trim().replace(/\/$/, "");
  const captureToken = $("captureToken").value.trim();
  await chrome.storage.local.set({ workerUrl, captureToken });
  await chrome.storage.sync.remove(["workerUrl", "captureToken"]);
  $("status").textContent = "checking…";
  try {
    const r = await fetch(`${workerUrl}/capture/start`, {
      method: "POST",
      headers: { Authorization: `Bearer ${captureToken}`, "Content-Type": "application/json" },
      body: JSON.stringify({ platform: "google_meet", native_meeting_id: null, title: "__options_test__" }),
    });
    if (r.status === 201) {
      $("status").textContent = "✅ connected";
      $("status").style.color = "#58C98A";
    } else {
      $("status").textContent = `❌ HTTP ${r.status}`;
      $("status").style.color = "#E5715F";
    }
  } catch (e) {
    $("status").textContent = "❌ unreachable";
    $("status").style.color = "#E5715F";
  }
});

// --- microphone permission (must be granted from an extension PAGE; Chrome
// cannot render permission prompts for the offscreen recording document) ---
function showMicStatus(state) {
  const el = $("micStatus");
  if (state === "granted") { el.textContent = "✅ granted"; el.style.color = "#58C98A"; }
  else if (state === "denied") {
    el.textContent = "❌ blocked. Click the icon left of the address bar and allow the microphone.";
    el.style.color = "#E5715F";
  } else { el.textContent = "not granted yet"; el.style.color = "#AEA492"; }
}

async function refreshMicStatus() {
  try {
    const p = await navigator.permissions.query({ name: "microphone" });
    showMicStatus(p.state);
    p.onchange = () => showMicStatus(p.state);
  } catch (e) { /* permissions API unavailable; button still works */ }
}
refreshMicStatus();

$("grantMic").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop());
    showMicStatus("granted");
  } catch (e) {
    showMicStatus("denied");
  }
});

// --- meeting chat: recording notice + chat capture ---
// Keep in sync with CHAT_DEFAULTS in content.js and NOTICE_DEFAULT_TEXT in background.js.
const CHAT_DEFAULTS = {
  noticeEnabled: true,
  noticeText: "🔴 Fred Notetaker is recording this meeting to produce notes. · Fred Notetaker está grabando esta reunión para generar notas.",
  chatCapture: true,
};

chrome.storage.local.get(Object.keys(CHAT_DEFAULTS)).then((v) => {
  $("noticeEnabled").checked = v.noticeEnabled !== false;
  $("noticeText").value = (typeof v.noticeText === "string" && v.noticeText.trim()) || CHAT_DEFAULTS.noticeText;
  $("chatCapture").checked = v.chatCapture !== false;
  $("noticeText").disabled = !$("noticeEnabled").checked;
});

$("noticeEnabled").addEventListener("change", () => {
  $("noticeText").disabled = !$("noticeEnabled").checked;
});

$("saveChat").addEventListener("click", async () => {
  const noticeText = $("noticeText").value.trim().slice(0, 500) || CHAT_DEFAULTS.noticeText;
  await chrome.storage.local.set({
    noticeEnabled: $("noticeEnabled").checked,
    noticeText,
    chatCapture: $("chatCapture").checked,
  });
  $("noticeText").value = noticeText;
  $("chatStatus").textContent = "✅ saved — applies from the next recording";
  $("chatStatus").style.color = "#58C98A";
});
