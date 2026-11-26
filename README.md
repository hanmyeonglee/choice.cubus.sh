# choice.cubus.sh

A small FastAPI server that performs single-choice text classification by calling
[Amber 150M](https://huggingface.co/MLukauskas/amber-150m) (`MLukauskas/amber-150m`, 151M parameters, `CC-BY-NC-4.0`, English-primary) directly.
It loads one fixed model in an on-demand worker subprocess using FP16 on CUDA (or FP32 on CPU), dynamic micro-batching (`coalesce_ms=5.0`, `max_batch_size=32`, `token_budget=8192` for `<= 2,048` tokens), O(L) banded attention for long sequences (`> 2,048` tokens), and a hard 7,999-token prepared input budget.

```text
Client → cubus shared ingress → FastAPI :8000 → Dynamic Micro-Batch Worker → Amber 150M / GPU
```

## Repository layout

```text
.
├── app/
│   ├── __init__.py
│   ├── __main__.py          # Server entry point (public API + optional Admin API)
│   ├── admin.py             # Optional internal :9090 Admin API server & auth/CIDR guard
│   ├── api.py               # Routes, authentication, and request handling
│   ├── runtime.py           # Thread-safe live runtime settings & worker coordination
│   ├── schemas.py           # Request/response schemas and input validation
│   ├── model.py             # Model loading, token budget check, and classification
│   ├── worker.py            # On-demand inference subprocess and idle GPU eviction
│   └── static/
│       └── index.html       # Public homepage
├── scripts/
│   └── download_model.py    # Build-time model download
├── experiments/             # Isolated precision, format, and model benchmarks
├── Dockerfile
├── compose.service.yml      # Shared service definition extended by cubus.sh
├── docker-compose.yml       # Local loopback wrapper (127.0.0.1:8000)
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
| `GET /api/healthz` | Server readiness, model residency (`model_loaded`), and `model` ID | None |
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

- One model (`MLukauskas/amber-150m`, 151M parameters, `CC-BY-NC-4.0`, English-primary) is fixed on the server. Requests have no `model` field.
- `text`: original prose, labeled sections, or a JSON object/array serialized as compact JSON (defaults to 8,000 characters, configurable 100–16,000 via the Admin API).
- `labels`: 2–16 distinct candidate labels (subject to runtime `max_labels`). Pass either:
  - a list of label names (e.g., `["negative", "neutral", "positive"]`), or
  - a dictionary mapping label names (1–64 chars) to optional descriptions (1–160 chars).
- `task`: optional task or category group name (1–64 chars, defaults to `"decision"`).
- `prompt`: optional question or classification instruction (1–320 chars).
- For binary or yes/no questions, pass two meaningful competing labels (such as `["no", "yes"]` or `{"safe": "...", "unsafe": "..."}`) rather than a single statement.
- Prepared sequences (text + task + prompt + labels and descriptions) exceeding 7,999 subword tokens are rejected with HTTP 422 rather than silently truncated.
- Invalid keys return 401; invalid or oversized input returns 422.
- Amber 150M is English-primary and trained on typed decision tasks; returned scores are softmax distributions over the provided candidates and are not calibrated probabilities of truth.

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
- `IDLE_TIMEOUT_SECONDS`: defaults to `180` (3 minutes). The model loads in a dedicated dynamic micro-batching worker subprocess on the first classification request and terminates after `IDLE_TIMEOUT_SECONDS` of inactivity, releasing 100% of GPU VRAM (including the CUDA context). Set to `0` to pre-load on startup and keep the model resident permanently.
- `MODEL_PATH`: defaults to `/opt/models/amber-150m`.
- Model weights and tokenizer files are included in the image; there are no external downloads at runtime.
- uv is used only to install locked dependencies during the build. The runtime image does not include uv.
- Uses PyTorch 2.7.1+cu126, Transformers 5.16.1, Safetensors 0.8.0, FastAPI 0.143.3, and Uvicorn 0.54.0. All dependencies are pinned in `uv.lock`.
- Model revision: `51ee1bba1f9e71aae4bf65bde1d43216260b6aab`.

### Optional Internal Admin API (`:9090`)

Set `ADMIN_API_TOKEN` (minimum 16 bytes, not a `replace-with-*` placeholder) to enable a separate internal-only Admin API listener for `admin.cubus.sh`. When `ADMIN_API_TOKEN` is unset, the `:9090` listener stays completely disabled.

- `ADMIN_API_TOKEN`: Bearer token required for `/admin/v1/*` requests (compared in constant time via `hmac.compare_digest`).
- `ADMIN_PORT`: internal Admin API port (default `9090`; must be `1..65535` and not `8000`).
- `ADMIN_BIND_CIDR`: optional CIDR (e.g. `172.31.0.0/24` in `cubus.sh`) selecting the exact local interface IP to bind (defaults to `127.0.0.1` when unset; never binds `0.0.0.0` or `::`).
- `ADMIN_ALLOW_CIDR`: optional peer IP CIDR allowlist enforced against the TCP peer IP, ignoring `X-Forwarded-For` (defaults to `ADMIN_BIND_CIDR` when set, or loopback when unset).

Endpoints:

- `GET /admin/v1/manifest`: service capabilities and settings schema (`service: "choice"`, `version: "0.3.0"`, `model: "amber-150m"`, `capabilities: ["settings", "classify"]`).
- `GET /admin/v1/settings`: current runtime values (`accept_requests`, `idle_timeout_seconds`, `max_text_chars`, `max_labels`; never exposes secrets).
- `PATCH /admin/v1/settings`: atomically validates and applies runtime settings live (`accept_requests=false` makes `POST /api/classify` return `503` while `/api/healthz` and the Admin API stay up; `idle_timeout_seconds` dynamically updates worker eviction).
- `POST /admin/v1/classify/batch`: internal batch classification endpoint sharing the single `ModelWorker` instance (up to 256 KiB JSON payload, 1–64 items of 1–4,000 chars each, optional `prompt` up to 512 chars, optional `task` up to 64 chars, `2–16` labels with optional descriptions up to 240 chars). Isolates per-item token-limit/model errors without failing the rest of the batch and stays available even when `accept_requests=false`.

## cubus deployment contract

| Setting | Value |
|---|---|
| Dockerfile / build context | Root `Dockerfile` / repository root |
| Service compose | `compose.service.yml` (`choice` service, extended by `cubus.sh`; exposes `8000` and optional internal `9090`) |
| Listener / intended hostname | `0.0.0.0:8000` / `choice.cubus.sh` (plus optional `ADMIN_BIND_CIDR:9090` internal Admin API) |
| Internal network | Shared `web` (plus optional internal `admin` when Admin API is enabled) |
| Proxy metadata | `VIRTUAL_HOST=choice.cubus.sh`, `VIRTUAL_PORT=8000` |
| GPU | GPU allocation through NVIDIA Container Toolkit |
| Required secret | Runtime `API_KEYS` (legacy `API_KEY` accepted when `API_KEYS` is absent) |
| Persistent storage | None; model files are included in the image |
| Writable paths | `/tmp` and temporary/cache files in the container user's home |
| Build outbound access | PyPI, PyTorch wheel repository, Hugging Face, and download CDNs |
| Runtime outbound access | None required by the server; the browser fetches Swagger UI JS and CSS from a CDN |
