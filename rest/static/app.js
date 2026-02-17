const state = {
  problems: [],
  sessionId: null,
  lastSentCode: "",
  debounceTimer: null,
};

const els = {
  problemSelect: document.getElementById("problem-select"),
  problemMeta: document.getElementById("problem-meta"),
  problemPrompt: document.getElementById("problem-prompt"),
  newSessionBtn: document.getElementById("new-session-btn"),
  modeSelect: document.getElementById("mode-select"),
  forceReactBtn: document.getElementById("force-react-btn"),
  modelInput: document.getElementById("model-input"),
  apiKeyInput: document.getElementById("api-key-input"),
  editor: document.getElementById("code-editor"),
  status: document.getElementById("status"),
  reactionFeed: document.getElementById("reaction-feed"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `HTTP ${response.status}`);
  }
  return response.json();
}

function setStatus(message, tone = "") {
  els.status.className = `status ${tone}`.trim();
  els.status.textContent = message;
}

function currentProblem() {
  const id = els.problemSelect.value;
  return state.problems.find((problem) => problem.id === id);
}

function changedLineCount(oldCode, newCode) {
  const oldLines = oldCode.split("\n");
  const newLines = newCode.split("\n");
  const maxLength = Math.max(oldLines.length, newLines.length);
  let changed = 0;

  for (let i = 0; i < maxLength; i += 1) {
    if ((oldLines[i] || "") !== (newLines[i] || "")) {
      changed += 1;
    }
  }

  return changed;
}

function clearReactions() {
  els.reactionFeed.innerHTML = "";
}

function renderReaction(reaction, source, changedLines) {
  const card = document.createElement("article");
  card.className = "reaction-card";

  const top = document.createElement("div");
  top.className = "reaction-topline";

  const focus = document.createElement("div");
  focus.className = "reaction-focus";
  focus.textContent = reaction.focus;

  const score = document.createElement("div");
  score.className = "reaction-meta";
  score.textContent = `importance ${reaction.importance}/10`;

  top.appendChild(focus);
  top.appendChild(score);
  card.appendChild(top);

  if (reaction.mode === "emoji") {
    const emoji = document.createElement("div");
    emoji.className = "reaction-emoji";
    emoji.textContent = reaction.emoji || "😏";
    card.appendChild(emoji);
  }

  const message = document.createElement("p");
  message.className = "reaction-message";
  message.textContent = reaction.message;
  card.appendChild(message);

  if (reaction.mode === "gif" && reaction.gif_url) {
    const gif = document.createElement("img");
    gif.className = "reaction-gif";
    gif.src = reaction.gif_url;
    gif.alt = reaction.gif_tag || "reaction gif";
    gif.loading = "lazy";
    card.appendChild(gif);
  }

  const meta = document.createElement("div");
  meta.className = "reaction-meta";
  meta.textContent = `${source} | changed lines: ${changedLines}`;
  card.appendChild(meta);

  els.reactionFeed.prepend(card);
}

function updateProblemUI(problem) {
  if (!problem) {
    return;
  }
  els.problemMeta.textContent = `${problem.difficulty} | ${problem.title}`;
  els.problemPrompt.textContent = problem.prompt;
}

async function startNewSession() {
  const problem = currentProblem();
  if (!problem) {
    return;
  }

  const payload = {
    problem_id: problem.id,
    language: "python",
  };

  const data = await api("/api/session/new", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  state.sessionId = data.session_id;
  els.editor.value = data.starter_code || "";
  state.lastSentCode = els.editor.value;
  clearReactions();
  setStatus("Session ready. Reactions fire on meaningful 2+ line updates.", "ok");
}

async function requestReaction(force) {
  const problem = currentProblem();
  if (!problem || !state.sessionId) {
    return;
  }

  const code = els.editor.value;
  const changedLines = changedLineCount(state.lastSentCode, code);
  if (!force && changedLines < 2) {
    setStatus("Waiting for 2+ changed lines.", "warn");
    return;
  }

  const payload = {
    session_id: state.sessionId,
    problem_id: problem.id,
    code,
    language: "python",
    model: els.modelInput.value.trim() || "gpt-4.1-mini",
    reaction_mode: els.modeSelect.value,
    force,
    openai_api_key: els.apiKeyInput.value.trim() || null,
  };

  try {
    const result = await api("/api/react", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    state.lastSentCode = code;

    if (!result.triggered) {
      setStatus(result.reason || "No reaction for this update.", "warn");
      return;
    }

    renderReaction(result.reaction, result.source, result.changed_lines);
    setStatus(`Reaction posted (${result.source}).`, "ok");
  } catch (error) {
    setStatus(`Reaction failed: ${error.message}`, "error");
  }
}

function scheduleReaction() {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => requestReaction(false), 650);
}

async function init() {
  try {
    const data = await api("/api/problems");
    state.problems = data.problems || [];

    els.problemSelect.innerHTML = "";
    state.problems.forEach((problem) => {
      const option = document.createElement("option");
      option.value = problem.id;
      option.textContent = `${problem.title} (${problem.difficulty})`;
      els.problemSelect.appendChild(option);
    });

    updateProblemUI(currentProblem());
    await startNewSession();

    els.problemSelect.addEventListener("change", async () => {
      updateProblemUI(currentProblem());
      await startNewSession();
    });
    els.newSessionBtn.addEventListener("click", startNewSession);
    els.forceReactBtn.addEventListener("click", () => requestReaction(true));
    els.editor.addEventListener("input", scheduleReaction);
  } catch (error) {
    setStatus(`Init failed: ${error.message}`, "error");
  }
}

init();
