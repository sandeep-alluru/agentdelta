# OpenAI Integration

## Codex CLI

The `CODEX.md` file at the repo root gives the OpenAI Codex CLI full project context: module map, build commands, invariants, and what not to change. Clone the repo and Codex is immediately project-aware.

## Assistants API / Responses API

`tools/openai-tools.json` contains OpenAI function-calling schemas for `diff_traces`, `inspect_trace`, and `record_snippet`. Paste directly into your assistant definition:

```python
import json
import openai

tools = json.loads(open("tools/openai-tools.json").read())

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Did my agent regress between runs?"}],
    tools=tools,
)
```

## GPT Actions / Custom GPTs

The `openapi.yaml` at repo root is a complete OpenAPI 3.1 spec for a REST wrapper.

### Run the server locally

```bash
pip install "redline[api]"
uvicorn redline.api:app --reload
# Server at http://localhost:8000
# Docs at  http://localhost:8000/docs
```

### Register as a ChatGPT Action

1. Deploy the server to a public URL (Railway, Fly.io, Render, etc.)
2. In ChatGPT → **My GPTs** → **Create** → **Configure** → **Add actions**
3. Import from URL: `https://your-deployment.example.com/openapi.yaml`
4. Set authentication if needed
5. Save and test

### API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/diff` | Compare two traces, returns DiffResult JSON |
| `POST` | `/inspect` | Summarise a single trace |
| `GET` | `/health` | Liveness probe — returns version |

Interactive docs are available at `/docs` (Swagger UI) and `/redoc` when the server is running.
