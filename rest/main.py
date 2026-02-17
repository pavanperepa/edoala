from __future__ import annotations

import difflib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency check for local setup
    OpenAI = None  # type: ignore[assignment]


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


PROBLEMS: list[dict[str, str]] = [
    {
        "id": "two-sum",
        "title": "Two Sum",
        "difficulty": "Easy",
        "prompt": (
            "Given an array of integers nums and an integer target, return indices of the "
            "two numbers such that they add up to target. Assume exactly one solution and "
            "do not use the same element twice."
        ),
        "starter_code": (
            "def two_sum(nums: list[int], target: int) -> list[int]:\n"
            "    # Write your solution here\n"
            "    pass\n"
        ),
    },
    {
        "id": "valid-parentheses",
        "title": "Valid Parentheses",
        "difficulty": "Easy",
        "prompt": (
            "Given a string containing just characters ()[]{} determine if the input "
            "string is valid. Open brackets must be closed in the correct order."
        ),
        "starter_code": (
            "def is_valid_parentheses(s: str) -> bool:\n"
            "    # Write your solution here\n"
            "    pass\n"
        ),
    },
    {
        "id": "longest-substring",
        "title": "Longest Substring Without Repeating Characters",
        "difficulty": "Medium",
        "prompt": (
            "Given a string s, find the length of the longest substring without "
            "repeating characters."
        ),
        "starter_code": (
            "def length_of_longest_substring(s: str) -> int:\n"
            "    # Write your solution here\n"
            "    pass\n"
        ),
    },
]


GIF_BANK: dict[str, str] = {
    "facepalm": "https://media1.giphy.com/media/TJawtKM6OCKkvwCIqX/giphy.gif",
    "slow-clap": "https://media3.giphy.com/media/nbvFVPiEiJH6JOGIok/giphy.gif",
    "mind-blown": "https://media1.giphy.com/media/OK27wINdQS5YQ/giphy.gif",
    "typing-fury": "https://media0.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
    "suspicious": "https://media3.giphy.com/media/a5viI92PAF89q/giphy.gif",
    "chef-kiss": "https://media4.giphy.com/media/9WXyFIDv2PyBq/giphy.gif",
}

ALLOWED_GIF_TAGS = sorted(GIF_BANK.keys())
IMPORTANT_TOKENS = {
    "for ",
    "while ",
    "if ",
    "elif ",
    "else:",
    "return ",
    "break",
    "continue",
    "try:",
    "except",
    "raise ",
    "stack",
    "queue",
    "dict",
    "set(",
    "sort(",
    "sorted(",
    "heap",
    "binary",
    "left",
    "right",
    "window",
    "pointer",
    "recurs",
    "memo",
}

COMMENT_RE = re.compile(r"^\s*(#|//|/\*|\*|\*/)")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class SessionState(BaseModel):
    session_id: str
    problem_id: str
    language: str
    latest_code: str = ""
    reaction_history: list[dict[str, Any]] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    problem_id: str
    language: str = "python"


class ReactionRequest(BaseModel):
    session_id: str
    problem_id: str
    code: str
    language: str = "python"
    model: str = "gpt-4.1-mini"
    reaction_mode: Literal["auto", "emoji", "gif"] = "auto"
    force: bool = False
    openai_api_key: str | None = None


SESSION_STORE: dict[str, SessionState] = {}
app = FastAPI(title="Live LLM Reactions", version="0.1.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def find_problem(problem_id: str) -> dict[str, str]:
    for problem in PROBLEMS:
        if problem["id"] == problem_id:
            return problem
    raise HTTPException(status_code=404, detail=f"Problem not found: {problem_id}")


def trim_code(code: str, max_lines: int = 120) -> str:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[-max_lines:])


def extract_change_details(old_code: str, new_code: str) -> tuple[int, list[str], list[str]]:
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)

    changed_old: list[str] = []
    changed_new: list[str] = []
    changed_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_count += max(i2 - i1, j2 - j1)
        changed_old.extend(old_lines[i1:i2])
        changed_new.extend(new_lines[j1:j2])

    return changed_count, changed_old, changed_new


def is_trivial_change(changed_old: list[str], changed_new: list[str]) -> bool:
    old_norm = [line.strip() for line in changed_old if line.strip()]
    new_norm = [line.strip() for line in changed_new if line.strip()]

    if old_norm == new_norm:
        return True

    all_lines = old_norm + new_norm
    if not all_lines:
        return True

    return all(COMMENT_RE.match(line) for line in all_lines)


def has_important_signal(changed_new: list[str]) -> bool:
    merged = "\n".join(changed_new).lower()
    return any(token in merged for token in IMPORTANT_TOKENS)


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty model output.")

    if candidate.startswith("{") and candidate.endswith("}"):
        return json.loads(candidate)

    match = JSON_BLOCK_RE.search(candidate)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def build_system_prompt(preferred_mode: str) -> str:
    return f"""
You are a live code reaction engine for competitive-programming practice.
Tone requirements:
- Witty, dry, sarcastic.
- Never toxic, abusive, or personally insulting.
- Keep reactions concise (max 2 sentences).

Reaction policy:
- React only to meaningful code progress or meaningful mistakes:
  algorithm choice, complexity changes, control flow correctness, edge cases, bug risk.
- Ignore minor style, naming, formatting, or cosmetic edits.
- If the change is not meaningful, output should_react=false.

Output strict JSON only with this exact schema:
{{
  "should_react": true,
  "importance": 7,
  "focus": "short phrase for why this matters",
  "mode": "emoji",
  "emoji": "😏",
  "gif_tag": "slow-clap",
  "message": "reaction text"
}}

Rules:
- importance is 1-10.
- mode must be one of ["emoji", "gif"].
- gif_tag must be one of {ALLOWED_GIF_TAGS}.
- If preferred mode is "{preferred_mode}", respect it unless it makes no sense.
""".strip()


def build_user_prompt(
    problem: dict[str, str],
    language: str,
    previous_code: str,
    current_code: str,
    changed_lines_old: list[str],
    changed_lines_new: list[str],
    history: list[dict[str, Any]],
) -> str:
    short_history = history[-4:]
    return (
        f"Problem: {problem['title']} ({problem['difficulty']})\n"
        f"Prompt: {problem['prompt']}\n"
        f"Language: {language}\n\n"
        f"Previous code:\n{trim_code(previous_code)}\n\n"
        f"Current code:\n{trim_code(current_code)}\n\n"
        f"Changed old lines:\n{trim_code(chr(10).join(changed_lines_old), 20)}\n\n"
        f"Changed new lines:\n{trim_code(chr(10).join(changed_lines_new), 20)}\n\n"
        f"Recent reaction history:\n{json.dumps(short_history, ensure_ascii=True)}"
    )


HARDCODED_OPENAI_API_KEY = ""


def get_openai_key(explicit_key: str | None) -> str | None:
    # Keep backend hardcoded key as source of truth; only trust explicit UI input if it looks valid.
    if explicit_key and explicit_key.strip().startswith("sk-"):
        return explicit_key.strip()
    return HARDCODED_OPENAI_API_KEY


def call_openai_reaction(model: str, api_key: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai package is not installed.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
    )

    output_text = getattr(response, "output_text", "") or ""
    if not output_text:
        output_chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", "")
                if text:
                    output_chunks.append(text)
        output_text = "\n".join(output_chunks)

    return parse_json_object(output_text)


def local_fallback_reaction(changed_new: list[str], mode: Literal["auto", "emoji", "gif"]) -> dict[str, Any]:
    merged = "\n".join(changed_new).lower()
    if "while" in merged or "for " in merged:
        focus = "loop logic"
        message = "Ah yes, a loop. Let's hope this one actually converges."
        emoji = "😏"
        tag = "typing-fury"
        importance = 6
    elif "if " in merged or "elif " in merged:
        focus = "branch logic"
        message = "You added branching. One step closer to handling reality."
        emoji = "🧐"
        tag = "suspicious"
        importance = 6
    elif "return " in merged:
        focus = "result path"
        message = "A return statement appears. Revolutionary concept."
        emoji = "👏"
        tag = "slow-clap"
        importance = 5
    else:
        focus = "algorithm direction"
        message = "Interesting move. Not reckless, not brilliant, just interesting."
        emoji = "😬"
        tag = "facepalm"
        importance = 5

    final_mode = "emoji" if mode == "emoji" else "gif" if mode == "gif" else "emoji"
    return {
        "should_react": True,
        "importance": importance,
        "focus": focus,
        "mode": final_mode,
        "emoji": emoji,
        "gif_tag": tag,
        "message": message,
    }


def normalize_reaction_payload(raw: dict[str, Any], reaction_mode: Literal["auto", "emoji", "gif"]) -> dict[str, Any]:
    should_react = bool(raw.get("should_react", False))
    importance = clamp_int(raw.get("importance"), 1, 10, default=5)
    focus = str(raw.get("focus", "logic quality")).strip()[:80] or "logic quality"
    message = str(raw.get("message", "")).strip()[:220]
    if not message:
        message = "Interesting. Bold, but interesting."

    model_mode = str(raw.get("mode", "emoji")).strip().lower()
    if model_mode not in {"emoji", "gif"}:
        model_mode = "emoji"

    if reaction_mode == "emoji":
        final_mode = "emoji"
    elif reaction_mode == "gif":
        final_mode = "gif"
    else:
        final_mode = model_mode

    emoji = str(raw.get("emoji", "😏")).strip()[:4] or "😏"
    gif_tag = str(raw.get("gif_tag", "facepalm")).strip().lower()
    if gif_tag not in GIF_BANK:
        gif_tag = "facepalm"
    gif_url = GIF_BANK[gif_tag]

    return {
        "should_react": should_react,
        "importance": importance,
        "focus": focus,
        "mode": final_mode,
        "emoji": emoji,
        "gif_tag": gif_tag,
        "gif_url": gif_url,
        "message": message,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/problems")
def list_problems() -> dict[str, Any]:
    return {"problems": PROBLEMS}


@app.post("/api/session/new")
def create_session(payload: SessionCreateRequest) -> dict[str, Any]:
    problem = find_problem(payload.problem_id)
    session_id = str(uuid.uuid4())
    session = SessionState(
        session_id=session_id,
        problem_id=payload.problem_id,
        language=payload.language,
        latest_code=problem["starter_code"],
    )
    SESSION_STORE[session_id] = session
    return {
        "session_id": session_id,
        "problem": problem,
        "starter_code": problem["starter_code"],
    }


@app.post("/api/react")
def react_to_code(payload: ReactionRequest) -> dict[str, Any]:
    problem = find_problem(payload.problem_id)
    session = SESSION_STORE.get(payload.session_id)
    if session is None:
        session = SessionState(
            session_id=payload.session_id,
            problem_id=payload.problem_id,
            language=payload.language,
            latest_code=problem["starter_code"],
        )
        SESSION_STORE[payload.session_id] = session

    if session.problem_id != payload.problem_id:
        session.problem_id = payload.problem_id
        session.latest_code = problem["starter_code"]
        session.reaction_history = []

    changed_count, changed_old, changed_new = extract_change_details(session.latest_code, payload.code)

    if not payload.force and changed_count < 2:
        return {
            "triggered": False,
            "reason": "Waiting for at least 2 changed lines.",
            "changed_lines": changed_count,
        }

    if not payload.force and is_trivial_change(changed_old, changed_new):
        session.latest_code = payload.code
        return {
            "triggered": False,
            "reason": "Only cosmetic/comment change detected.",
            "changed_lines": changed_count,
        }

    if not payload.force and changed_count < 4 and not has_important_signal(changed_new):
        session.latest_code = payload.code
        return {
            "triggered": False,
            "reason": "Change looks minor; waiting for something more substantial.",
            "changed_lines": changed_count,
        }

    system_prompt = build_system_prompt(payload.reaction_mode)
    user_prompt = build_user_prompt(
        problem=problem,
        language=payload.language,
        previous_code=session.latest_code,
        current_code=payload.code,
        changed_lines_old=changed_old,
        changed_lines_new=changed_new,
        history=session.reaction_history,
    )

    source = "openai"
    key = get_openai_key(payload.openai_api_key)
    try:
        if key:
            raw_reaction = call_openai_reaction(payload.model, key, system_prompt, user_prompt)
        else:
            source = "local"
            raw_reaction = local_fallback_reaction(changed_new, payload.reaction_mode)
    except Exception as exc:  # pragma: no cover - network/API failure path
        source = "local"
        raw_reaction = local_fallback_reaction(changed_new, payload.reaction_mode)
        raw_reaction["message"] = f"{raw_reaction['message']} (Fallback: {exc})"

    reaction = normalize_reaction_payload(raw_reaction, payload.reaction_mode)
    session.latest_code = payload.code

    if not reaction["should_react"] and not payload.force:
        return {
            "triggered": False,
            "reason": "Model judged this change as not important enough.",
            "changed_lines": changed_count,
            "source": source,
        }

    if payload.force and not reaction["should_react"]:
        reaction["should_react"] = True
        reaction["message"] = "Forced reaction mode: this update is noted."

    event = {
        "at": datetime.now(timezone.utc).isoformat(),
        "importance": reaction["importance"],
        "focus": reaction["focus"],
        "message": reaction["message"],
        "mode": reaction["mode"],
    }
    session.reaction_history.append(event)
    session.reaction_history = session.reaction_history[-8:]

    return {
        "triggered": True,
        "changed_lines": changed_count,
        "reaction": reaction,
        "source": source,
        "session_id": session.session_id,
    }


@app.get("/")
def serve_frontend() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found.")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
