# choice.cubus.sh

A small FastAPI server that performs single-choice text classification by calling
[GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (`fastino/GLiNER2.5-Decide`) directly.
It loads one fixed model and processes one request at a time using FP16 on CUDA (or FP32 on CPU) with standard attention and a hard 1,024-token prepared input budget.

```text
Client → cubus shared ingress → FastAPI :8000 → GLiNER2.5-Decide / GPU
```

## Repository layout

```text
.
├── app/
│   ├── __init__.py
│   ├── __main__.py          # Server entry point
│   ├── api.py               # Routes, authentication, and request handling
│   ├── schemas.py           # Request/response schemas and input validation
│   ├── model.py             # Model loading, token budget check, and classification
│   └── static/
│       └── index.html       # Public homepage
├── scripts/
│   └── download_model.py    # Build-time model download
├── experiments/             # Isolated precision, format, and model benchmarks
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

The runtime entry point is `python -m app`, run from the repository root or the container's `/app` directory. Docker installs locked dependencies and downloads model files during the build.

For isolated precision, format ablation, and System One model comparisons, see
[the experiment guide](experiments/README.md).

## API

| Method / path | Purpose | Authentication |
|---|---|---|
| `POST /api/classify` | Classify text against 2–16 competing candidate labels | Bearer key |
| `GET /api/healthz` | Readiness after model loading | None |
| `GET /` | Introduction and request example | None |
| `GET /docs` | Swagger UI documentation and interactive requests | None |
| `GET /openapi.json` | OpenAPI schema | None |

```bash
export API_KEY='your-api-key'
curl --fail-with-body --max-time 30 http://127.0.0.1:8000/api/classify \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "I was charged twice for the same order.",
    "task": "department",
    "prompt": "Which team should handle this request?",
    "labels": {
      "billing": "Charges, refunds, and invoices",
      "technical": "Website, login, or technical outages"
    }
  }'
```

The public endpoint is `https://choice.cubus.sh/api/classify`. You can also enter the key itself in **Authorize** at `/docs` to send requests.

### API keys

Configure one or more keys in the server's `API_KEYS` environment variable, separated by commas:

```dotenv
API_KEYS=replace-with-a-random-key,replace-with-another-random-key
```

Replace every placeholder with a separately generated key. Each client sends only its own key as `Authorization: Bearer <key>`; the request examples use the client's `API_KEY` variable for that value. Any configured key grants the same access to `/api/classify`.

- Keys are generated and distributed manually; there are no accounts, login, or key issuance endpoints.
- Surrounding whitespace and empty entries in the list are ignored. At least one real key is required, and placeholder keys beginning with `replace-with-` prevent startup.
- `API_KEYS` takes precedence. The legacy `API_KEY` server setting is used only when `API_KEYS` is absent. An explicitly empty `API_KEYS` does not fall back to `API_KEY`.
- Every configured key is compared with `secrets.compare_digest`, without stopping after a match. Missing or invalid credentials return 401.
- Add, remove, or rotate keys by updating the server environment and recreating the container. Keys have no automatic expiration.
- The homepage, docs, OpenAPI schema, and readiness endpoint remain public.

### Request and response schema

Scores in this response example are illustrative:

```json
{
  "choice": "billing",
  "probabilities": {
    "billing": 0.84,
    "technical": 0.16
  }
}
```

- One model (`fastino/GLiNER2.5-Decide`) is fixed on the server. Requests have no `model` field.
- `text`: original prose, labeled sections, or a JSON object/array serialized as compact JSON up to 2,000 characters.
- `labels`: 2–16 distinct candidate labels. Pass either:
  - a list of label names (e.g., `["negative", "neutral", "positive"]`), or
  - a dictionary mapping label names (1–64 chars) to optional descriptions (1–160 chars).
- `task`: optional task or category group name (1–64 chars, defaults to `"decision"`).
- `prompt`: optional question or classification instruction (1–320 chars).
- For binary or yes/no questions, pass two meaningful competing labels (such as `["no", "yes"]` or `{"safe": "...", "unsafe": "..."}`) rather than a single statement.
- Prepared sequences (text + task + prompt + labels and descriptions) exceeding 1,024 subword tokens are rejected with HTTP 422 rather than silently truncated.
- Invalid keys return 401; invalid or oversized input returns 422.

### Writing effective classification requests

- **Use competing candidates**: even for boolean gates, supply two contrasting options so the softmax classifier compares both hypotheses directly.
- **Include descriptions when boundaries matter**: adding concise label descriptions improves domain-specific routing accuracy.
- **Keep `text` focused**: pass the relevant document, transcript, or command context without unnecessary boilerplate. Objects and arrays are accepted for convenience and serialized as compact JSON text.

## Build and run

The Ubuntu x86_64 server requires an NVIDIA driver and NVIDIA Container Toolkit.

```bash
docker build -t choice-api .
cp .env.example .env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Run the key generation command once per client and save the generated keys as a comma-separated `API_KEYS` value in `.env`. The server refuses to start if no keys are configured or any key is still a placeholder. `.env` is excluded from Git and the build context.

For direct access on the server, bind the port to loopback:

```bash
docker run --rm --name choice-api --gpus all \
  --env-file .env -p 127.0.0.1:8000:8000 choice-api
```

- `DEVICE`: defaults to `cuda`. Set to `cpu` explicitly for CPU execution. CUDA mode refuses to start if no GPU is available.
- `PRECISION`: defaults to `fp16` on `cuda` and `fp32` on `cpu`.
- `MODEL_PATH`: defaults to `/opt/models/gliner2.5-decide`.
- Model weights and tokenizer files are included in the image; there are no external downloads at runtime.
- uv is used only to install locked dependencies during the build. The runtime image does not include uv.
- Uses PyTorch 2.7.1+cu126, GLiNER2 2.1.0, and Transformers 5.16.1. FastAPI, Uvicorn, and all dependencies are pinned in `uv.lock`.
- Model revision: `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`.

## cubus deployment contract

| Setting | Value |
|---|---|
| Dockerfile / build context | Root `Dockerfile` / repository root |
| Listener / intended hostname | `0.0.0.0:8000` / `choice.cubus.sh` |
| Internal network | Shared `web` |
| Proxy metadata | `VIRTUAL_HOST=choice.cubus.sh`, `VIRTUAL_PORT=8000` |
| GPU | GPU allocation through NVIDIA Container Toolkit |
| Required secret | Runtime `API_KEYS` (legacy `API_KEY` accepted when `API_KEYS` is absent) |
| Persistent storage | None; model files are included in the image |
| Writable paths | `/tmp` and temporary/cache files in the container user's home |
| Build outbound access | PyPI, PyTorch wheel repository, Hugging Face, and download CDNs |
| Runtime outbound access | None required by the server; the browser fetches Swagger UI JS and CSS from a CDN |
