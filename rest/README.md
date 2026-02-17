## Live LLM/SLM Coding Reactions (Local Prototype)

This folder now contains:
- A FastAPI backend for live "important-only" code reactions.
- A static frontend with a Hackerrank-style challenge picker + code editor.
- In-memory session context (no database).

### What it does
- Reacts after meaningful 2+ line changes.
- Keeps context from previous code and recent reactions.
- Avoids reacting to cosmetic edits (formatting/comments) and small noise.
- Supports `emoji` and `gif` reaction styles.
- Uses OpenAI when key is provided, with local fallback reactions if no key exists.

### Run locally
1. Install dependencies:

```powershell
uv sync
```

2. Set key (optional but recommended for best reactions):

```powershell
$env:OPENAI_API_KEY="sk-..."
```

3. Start app:

```powershell
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

4. Open:
- `http://127.0.0.1:8000`

### API overview
- `GET /api/health`
- `GET /api/problems`
- `POST /api/session/new`
- `POST /api/react`

### Notes
- No DB is used. Session memory resets when server restarts.
- For local testing you can also paste the OpenAI key directly in the UI input.
