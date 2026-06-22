const API = location.protocol === "file:" ? "http://localhost:5000" : "";

const $ = (s) => document.querySelector(s);
const historyEl = $("#history");
const messagesEl = $("#messages");
const contextEl = $("#contextList");
const contextCount = $("#contextCount");
const titleEl = $("#chatTitle");
const inputEl = $("#input");
const toastEl = $("#toast");
const modelSelectEl = $("#modelSelect");

const ASSET = {
  logo: "../images/logo.png",
  opening: "../images/image.png",
  profile: "../images/pp.png",
};

const ICON = {
  trash: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none"><path d="M5 7h14M10 7V5h4v2M8 7l1 12h6l1-12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  doc: `<svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M13 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9l-6-6Z" stroke="currentColor" stroke-width="1.5"/><path d="M13 3v6h6" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>`,
};

let chats = [];          // [{ id, title, ts, unsaved? }]
let activeId = null;
let messages = [];       // active chat's messages: { role, text, route, sources, ts }
let models = [];
let selectedModel = "";

async function loadChats() {
  try {
    const res = await fetch(API + "/chats");
    const rows = await res.json();
    chats = rows.map((c) => ({ id: c.id, title: c.title, ts: Date.parse(c.updated_at || c.created_at) }));
  } catch (e) {
    chats = [];
  }
}

async function loadModels() {
  try {
    const res = await fetch(API + "/models");
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    models = [...(data.models || [])].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
    const saved = localStorage.getItem("documind:model") || "";
    selectedModel = models.includes(saved)
      ? saved
      : models.includes(data.default_model)
        ? data.default_model
        : (models[0] || data.default_model || "");
    renderModelSelect();
  } catch (e) {
    models = [];
    selectedModel = "";
    renderModelSelect("No Ollama models");
  }
}

function currentChat() { return chats.find((c) => c.id === activeId); }

function newChat() {
  const chat = { id: crypto.randomUUID(), title: "New chat", ts: Date.now(), unsaved: true };
  chats.unshift(chat);
  activeId = chat.id;
  messages = [];
  renderAll();
  inputEl.focus();
}

async function setActive(id) {
  activeId = id;
  const chat = currentChat();
  messages = chat && chat.unsaved ? [] : await loadMessages(id);
  renderAll();
  closeNav();
}

async function loadMessages(id) {
  try {
    const res = await fetch(API + "/chats/" + id);
    const rows = await res.json();
    return rows.map((m) => ({ role: m.role, text: m.content, route: m.route, sources: m.sources || [], ts: m.created_at }));
  } catch (e) {
    return [];
  }
}

async function deleteChat(id) {
  const chat = chats.find((c) => c.id === id);
  if (chat && !chat.unsaved) {
    try { await fetch(API + "/chats/" + id, { method: "DELETE" }); } catch (e) {}
  }
  chats = chats.filter((c) => c.id !== id);
  if (activeId === id) {
    if (chats.length) await setActive(chats[0].id);
    else newChat();
  } else {
    renderSidebar();
  }
}

function renderAll() { renderSidebar(); renderHead(); renderMessages(); renderContext(); }

function bucketOf(ts) {
  const now = new Date();
  const day = 86400000;
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (ts >= startToday) return "Today";
  if (ts >= startToday - day) return "Yesterday";
  if (ts >= startToday - 7 * day) return "Previous 7 days";
  return "Older";
}
function fmtTime(ts) { return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
function titleFromText(text, maxLen = 90) {
  const title = text.replace(/\s+/g, " ").trim();
  if (title.length <= maxLen) return title;

  const clipped = title.slice(0, maxLen).replace(/\s+\S*$/, "").trim();
  return (clipped || title.slice(0, maxLen).trim()) + "...";
}

function renderSidebar() {
  const order = ["Today", "Yesterday", "Previous 7 days", "Older"];
  const groups = {};
  [...chats].sort((a, b) => b.ts - a.ts).forEach((c) => {
    (groups[bucketOf(c.ts)] ||= []).push(c);
  });
  historyEl.innerHTML = "";
  order.forEach((label) => {
    const items = groups[label];
    if (!items) return;
    const group = document.createElement("div");
    group.className = "hist-group";
    group.innerHTML = `<p class="hist-label">${label}</p>`;
    items.forEach((c) => {
      const item = document.createElement("button");
      item.className = "hist-item" + (c.id === activeId ? " active" : "");
      item.innerHTML =
        `<span class="hist-title">${escapeHtml(c.title)}</span>` +
        `<span class="hist-time">${fmtTime(c.ts)}</span>` +
        `<span class="hist-del" role="button" aria-label="Delete chat">${ICON.trash}</span>`;
      item.addEventListener("click", (e) => {
        if (e.target.closest(".hist-del")) deleteChat(c.id);
        else setActive(c.id);
      });
      group.appendChild(item);
    });
    historyEl.appendChild(group);
  });
}

function renderHead() { const c = currentChat(); titleEl.textContent = c ? c.title : "New chat"; }

function renderModelSelect(emptyLabel = "No models found") {
  modelSelectEl.innerHTML = "";
  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    modelSelectEl.appendChild(option);
    modelSelectEl.disabled = true;
    $("#model").textContent = "";
    return;
  }

  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelSelectEl.appendChild(option);
  });
  modelSelectEl.disabled = false;
  modelSelectEl.value = selectedModel;
  $("#model").textContent = selectedModel;
}

function renderMessages() {
  messagesEl.innerHTML = "";
  if (!messages.length) { messagesEl.appendChild(welcome()); return; }
  messages.forEach((m) => messagesEl.appendChild(messageEl(m)));
  scrollDown();
}

function welcome() {
  const examples = [
    "Explain LoRA in simple terms",
    "Compare PPO and Soft Actor-Critic",
    "How does retrieval-augmented generation work?",
  ];
  const wrap = document.createElement("div");
  wrap.className = "welcome";
  wrap.innerHTML =
    `<div class="welcome-visual"><img src="${ASSET.opening}" alt="DocuMind" /></div>` +
    `<h2>Ask across your library</h2>` +
    `<p>DocuMind retrieves from your indexed papers and shows where each answer comes from.</p>` +
    `<div class="examples">${examples.map((e) => `<button class="example">${e}</button>`).join("")}</div>`;
  wrap.querySelectorAll(".example").forEach((b) =>
    b.addEventListener("click", () => { inputEl.value = b.textContent; autosize(); inputEl.focus(); })
  );
  return wrap;
}

function renderMarkdown(src) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  let html = "";
  let list = null;
  let code = false;
  let para = [];
  const flush = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const endList = () => { if (list) { html += `</${list}>`; list = null; } };

  for (const raw of src.split("\n")) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      flush(); endList();
      html += code ? "</code></pre>" : "<pre><code>";
      code = !code;
      continue;
    }
    if (code) { html += esc(raw) + "\n"; continue; }
    if (/^#{1,6}\s/.test(line)) {
      flush(); endList();
      const level = line.match(/^#+/)[0].length;
      html += `<h${level}>${inline(line.replace(/^#+\s+/, ""))}</h${level}>`;
      continue;
    }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) { flush(); endList(); html += "<hr>"; continue; }
    const ol = line.match(/^\d+\.\s+(.*)/);
    const ul = line.match(/^[*-]\s+(.*)/);
    if (ol || ul) {
      flush();
      const type = ol ? "ol" : "ul";
      if (list !== type) { endList(); html += `<${type}>`; list = type; }
      html += `<li>${inline((ol || ul)[1])}</li>`;
      continue;
    }
    if (line === "") { flush(); endList(); continue; }
    para.push(line);
  }
  flush(); endList();
  if (code) html += "</code></pre>";
  return html;
}


function messageEl(m) {
  const row = document.createElement("div");
  row.className = `msg ${m.role}`;
  if (m.role === "assistant") {
    row.appendChild(assistantAvatar());
  }
  const col = document.createElement("div");
  col.className = "bubble-col";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (m.role === "assistant") bubble.innerHTML = renderMarkdown(m.text);
  else bubble.textContent = m.text;
  col.appendChild(bubble);

  if (m.role === "assistant" && m.route && m.route !== "error") col.appendChild(footer(m));

  const time = document.createElement("span");
  time.className = "msg-time";
  time.textContent = fmtTime(m.ts);
  col.appendChild(time);

  row.appendChild(col);
  return row;
}

function footer(m) {
  const grounded = m.route === "rag";
  const foot = document.createElement("div");
  foot.className = "msg-foot";

  const pill = document.createElement("span");
  pill.className = "route " + (grounded ? "rag" : "general");
  pill.textContent = grounded ? "Grounded in documents" : "General knowledge";
  foot.appendChild(pill);

  if (grounded && m.sources && m.sources.length) {
    dedupeSources(m.sources).slice(0, 4).forEach((s, i) => {
      const cite = document.createElement("span");
      cite.className = "cite";
      cite.textContent = `${i + 1} · ${shortName(s.source)} p.${s.page}`;
      foot.appendChild(cite);
    });
  }
  return foot;
}

function renderContext() {
  const all = [];
  messages.forEach((m) => {
    if (m.role === "assistant" && m.route === "rag" && m.sources) all.push(...m.sources);
  });
  const items = dedupeSources(all);
  contextCount.textContent = items.length || "";

  if (!items.length) {
    contextEl.innerHTML = `<p class="context-empty">Sources appear here as DocuMind cites them.</p>`;
    return;
  }
  contextEl.innerHTML = "";
  items.forEach((s) => {
    const score = s.score != null ? `<span class="ctx-score">${Number(s.score).toFixed(2)}</span>` : "";
    const card = document.createElement("div");
    card.className = "ctx-card";
    card.innerHTML =
      `<span class="ctx-ico">${ICON.doc}</span>` +
      `<span class="ctx-body"><span class="ctx-name">${escapeHtml(shortName(s.source))}</span>` +
      `<span class="ctx-meta"><span>page ${escapeHtml(String(s.page))}</span>${score}</span></span>`;
    contextEl.appendChild(card);
  });
}

async function send(text) {
  const chat = currentChat();
  messages.push({ role: "user", text, ts: Date.now() });
  if (chat.title === "New chat") chat.title = titleFromText(text);
  chat.unsaved = false;
  chat.ts = Date.now();
  renderSidebar();
  renderHead();
  renderMessages();

  const row = typingEl();
  messagesEl.appendChild(row);
  const bubble = row.querySelector(".bubble");
  scrollDown();

  let acc = "", route = null, sources = [];
  try {
    const res = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, session_id: activeId, model: selectedModel }),
    });
    if (!res.ok || !res.body) throw new Error(res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const data = part.replace(/^data: ?/, "").trim();
        if (!data) continue;
        const evt = JSON.parse(data);
        if (evt.type === "meta") {
          route = evt.route;
          sources = evt.sources || [];
        } else if (evt.type === "token") {
          acc += evt.text;
          bubble.classList.remove("typing");
          bubble.innerHTML = renderMarkdown(acc);
          scrollDown();
        }
      }
    }
  } catch (e) {
    messages.push({ role: "assistant", text: "Couldn't reach the model. Check that the server and Ollama are running, then try again.", route: "error", ts: Date.now() });
    renderMessages();
    return;
  }

  messages.push({ role: "assistant", text: acc, route, sources, ts: Date.now() });
  renderMessages();
  renderContext();
}

async function upload(files) {
  const pdfs = [...files].filter((f) => f.name.toLowerCase().endsWith(".pdf"));
  if (!pdfs.length) { toast("Only PDF files can be indexed."); return; }
  for (const file of pdfs) {
    toast(`Indexing ${file.name}…`, true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(API + "/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error(res.status);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      toast(`Indexed ${data.chunks} chunks from ${data.file}`);
    } catch (e) {
      toast(`Couldn't index ${file.name}.`);
    }
  }
}

async function health() {
  const status = $("#status");
  try {
    const res = await fetch(API + "/health");
    const data = await res.json();
    status.classList.add("online");
    $("#statusText").textContent = "Connected";
    $("#model").textContent = selectedModel || data.model || "";
  } catch (e) {
    status.classList.add("offline");
    $("#statusText").textContent = "Offline";
  }
}

function dedupeSources(list) {
  const seen = new Set();
  const out = [];
  for (const s of list) {
    const key = s.source + "|" + s.page;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}
function shortName(n) { return String(n).replace(/\.pdf$/i, ""); }
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function assistantAvatar() {
  const av = document.createElement("div");
  av.className = "avatar";
  av.innerHTML = `<img src="${ASSET.profile}" alt="" />`;
  return av;
}
function typingEl() {
  const row = document.createElement("div");
  row.className = "msg assistant";
  row.appendChild(assistantAvatar());
  const col = document.createElement("div");
  col.className = "bubble-col";
  col.innerHTML = `<div class="bubble typing"><span></span><span></span><span></span></div>`;
  row.appendChild(col);
  return row;
}
function autosize() { inputEl.style.height = "auto"; inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px"; }

let toastTimer;
function toast(msg, sticky) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  if (!sticky) toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3200);
}

function closeNav() { document.body.classList.remove("nav-open"); }

$("#newChat").addEventListener("click", newChat);

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  autosize();
  send(text);
});
inputEl.addEventListener("input", autosize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
});

const dropzone = $("#dropzone");
$("#fileInput").addEventListener("change", (e) => { upload(e.target.files); e.target.value = ""; });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => { e.preventDefault(); dropzone.classList.remove("drag"); upload(e.dataTransfer.files); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

$("#menuBtn").addEventListener("click", () => document.body.classList.toggle("nav-open"));
$("#scrim").addEventListener("click", closeNav);
modelSelectEl.addEventListener("change", () => {
  selectedModel = modelSelectEl.value;
  localStorage.setItem("documind:model", selectedModel);
  $("#model").textContent = selectedModel;
  toast(`Model switched to ${selectedModel}`);
});

async function init() {
  await Promise.all([loadModels(), loadChats()]);
  if (chats.length) await setActive(chats[0].id);
  else newChat();
  health();
}
init();
