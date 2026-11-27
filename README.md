# choice.cubus.sh

A FastAPI classification server backed directly by
[Amber 150M](https://huggingface.co/MLukauskas/amber-150m) (`MLukauskas/amber-150m`, 151M parameters, `CC-BY-NC-4.0`, English-primary) exposing an industry-standard Jina AI / OpenAI-style `/v1` API.
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
│   ├── api.py               # /v1 routes, DB-backed API key auth, and request handling
│   ├── api_keys.py          # API key generation, SHA-256 hashing, bounded cache & usage batching
│   ├── db.py                # PostgreSQL pool, db-broker bootstrap, and startup DDL migration
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
| `POST /v1/classify` | Classify 1–64 inputs against 2–16 competing candidate labels | Bearer / `X-API-Key` |
| `GET /v1/models` | List available classification models | Bearer / `X-API-Key` |
| `GET /healthz` | Server readiness, model residency (`model_loaded`), and `model` ID | None |
| `GET /` | Introduction and request example | None |
| `GET /docs` | Swagger UI documentation and interactive requests | None |
| `GET /openapi.json` | OpenAPI schema | None |

```bash
export API_KEY='your-api-key'
curl --fail-with-body --max-time 30 http://127.0.0.1:8000/v1/classify \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "amber-150m",
    "input": [
      "I was charged twice for the same order.",
      {"text": "The checkout page returns a 502 error."}
    ],
    "task": "department",
    "prompt": "Which team should handle this request?",
    "labels": {
      "billing": "Charges, refunds, and invoices",
      "technical": "Website, login, or technical outages"
    }
  }'
```

The public endpoints are `https://choice.cubus.sh/v1/classify` and `https://choice.cubus.sh/v1/models`. You can also enter the key itself in **Authorize** at `/docs` to send requests.

### API keys

API keys are 100% PostgreSQL-backed (`.env` `API_KEYS` is not used). Keys are issued, listed, revoked, and deleted through the internal `:9090` Admin API (e.g. from `admin.cubus.sh`):

- Generated keys use the format `chk_` + 32 URL-safe random bytes (~47 characters total). Only the 12-character prefix (`chk_XXXXXXXX`) and SHA-256 hex digest are stored in PostgreSQL; the plaintext key is returned once at creation time.
- Clients authenticate to `POST /v1/classify` and `GET /v1/models` with either `Authorization: Bearer <key>` or `X-API-Key: <key>`. Missing, malformed, unknown, revoked, or expired keys return `401`.
- Verification uses a bounded short-TTL (15 s) in-memory cache of active key hashes (never negative-caching invalid keys) and batches `last_used_at` / `request_count` updates asynchronously (~5 s interval and on graceful shutdown) to avoid adding DB latency to inference requests.
- Revoking or deleting a key via `:9090` invalidates the cache immediately.
- The homepage, docs, OpenAPI schema, and readiness endpoint (`/healthz`) remain public.

### Request and response schema

Scores in this response example are illustrative:

```json
{
  "object": "list",
  "model": "amber-150m",
  "data": [
    {
      "object": "classification",
      "index": 0,
      "prediction": "billing",
      "score": 0.9412,
      "predictions": [
        {"label": "billing", "score": 0.9412},
        {"label": "technical", "score": 0.0588}
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 124,
    "total_tokens": 124
  }
}
```

- `model`: optional string defaulting to `"amber-150m"` (also accepts `"MLukauskas/amber-150m"`). Unknown model names return HTTP `400`.
- `input`: required — accepts a single string, a single `{"text": str | dict | list}` object, or a non-empty list of 1–64 strings and/or `{"text": str | dict | list}` objects. Objects and arrays inside `text` are serialized as compact JSON. Each serialized item defaults to an 8,000-character limit (configurable 100–16,000 via the Admin API), and request bodies are capped at 256 KiB.
- `labels`: 2–16 distinct candidate labels (subject to runtime `max_labels`). Pass either:
  - a list of label names (e.g., `["negative", "neutral", "positive"]`), or
  - a dictionary mapping label names (1–64 chars) to optional descriptions (1–160 chars).
- `task`: optional task or category group name (1–64 chars, defaults to `"decision"`).
- `prompt`: optional question or classification instruction (1–320 chars).
- `data[i].predictions` is sorted descending by `score` (ties preserve requested label order), so `predictions[0].label == prediction` and `predictions[0].score == score`.
- `usage.prompt_tokens` and `usage.total_tokens` report exact prepared subword token counts across the batch.
- All `/v1/*` error responses (400, 401, 413, 422, 499, 503) use the standard OpenAI-style error envelope `{"error": {"message": "...", "type": "...", "code": "..."}}`.
- Prepared sequences (input + task + prompt + labels and descriptions) exceeding 7,999 subword tokens are rejected with HTTP 422 rather than silently truncated.
- Amber 150M is English-primary and trained on typed decision tasks; returned scores are softmax distributions over the provided candidates and are not calibrated probabilities of truth.

### Writing effective classification requests

- **Use competing candidates**: even for boolean gates, supply two contrasting options so the softmax classifier compares both hypotheses directly.
- **Include descriptions when boundaries matter**: adding concise label descriptions improves domain-specific routing accuracy.
- **Keep `input` focused**: pass the relevant document, transcript, or command context without unnecessary boilerplate. Objects and arrays are accepted inside `{"text": ...}` and serialized as compact JSON text.

## Build and run

The Ubuntu x86_64 server requires an NVIDIA driver and NVIDIA Container Toolkit.

```bash
cp .env.example .env
docker compose up --build -d
```

Standalone local `docker-compose.yml` brings up a local `postgres:18.7-alpine3.24` instance alongside `choice` and wires `DATABASE_URL` automatically. Under `cubus.sh`, root `docker-compose.yml` injects `DB_BROKER_URL` so `choice` resolves its isolated `svc_choice` database credentials from `db-broker` at startup and runs the idempotent `api_keys` schema migration automatically.

- `DB_BROKER_URL`: production database credential broker endpoint injected by `cubus.sh` (`POST {}` with bounded exponential backoff + jitter on `503`/network errors; fail-fast on `400`/`403`/`409`). Takes precedence over `DATABASE_URL`.
- `DATABASE_URL`: standalone PostgreSQL DSN used when `DB_BROKER_URL` is unset.
- `DEVICE`: defaults to `cuda`. Set to `cpu` explicitly for CPU execution. CUDA mode refuses to start if no GPU is available.
- `PRECISION`: defaults to `fp16` on `cuda` and `fp32` on `cpu`.
- `IDLE_TIMEOUT_SECONDS`: defaults to `180` (3 minutes). The model loads in a dedicated dynamic micro-batching worker subprocess on the first classification request and terminates after `IDLE_TIMEOUT_SECONDS` of inactivity, releasing 100% of GPU VRAM (including the CUDA context). Set to `0` to pre-load on startup and keep the model resident permanently.
- `MODEL_PATH`: defaults to `/opt/models/amber-150m`.
- Model weights and tokenizer files are included in the image; there are no external downloads at runtime.
- uv is used only to install locked dependencies during the build. The runtime image does not include uv.
- Uses PyTorch 2.7.1+cu126, Transformers 5.16.1, Safetensors 0.8.0, FastAPI 0.143.3, Uvicorn 0.54.0, and `psycopg[binary,pool]` 3.3.9. All dependencies are pinned in `uv.lock`.
- Model revision: `51ee1bba1f9e71aae4bf65bde1d43216260b6aab`.

### Optional Internal Admin API (`:9090`)

Set `ADMIN_API_TOKEN` (minimum 16 bytes, not a `replace-with-*` placeholder) to enable a separate internal-only Admin API listener for `admin.cubus.sh`. When `ADMIN_API_TOKEN` is unset, the `:9090` listener stays completely disabled.

- `ADMIN_API_TOKEN`: Bearer token required for `/admin/v1/*` requests (compared in constant time via `hmac.compare_digest`).
- `ADMIN_PORT`: internal Admin API port (default `9090`; must be `1..65535` and not `8000`).
- `ADMIN_BIND_CIDR`: optional CIDR (e.g. `172.31.0.0/24` in `cubus.sh`) selecting the exact local interface IP to bind (defaults to `127.0.0.1` when unset; never binds `0.0.0.0` or `::`).
- `ADMIN_ALLOW_CIDR`: optional peer IP CIDR allowlist enforced against the TCP peer IP, ignoring `X-Forwarded-For` (defaults to `ADMIN_BIND_CIDR` when set, or loopback when unset).

Endpoints:

- `GET /admin/v1/manifest`: service capabilities and settings schema (`service: "choice"`, `version: "0.3.0"`, `model: "amber-150m"`, `capabilities: ["settings", "classify", "api_keys"]`).
- `GET /admin/v1/settings`: current runtime values (`accept_requests`, `idle_timeout_seconds`, `max_text_chars`, `max_labels`; never exposes secrets).
- `PATCH /admin/v1/settings`: atomically validates and applies runtime settings live (`accept_requests=false` makes `POST /v1/classify` return `503` while `/healthz`, `/v1/models`, and the Admin API stay up; `idle_timeout_seconds` dynamically updates worker eviction).
- `POST /admin/v1/classify`: internal classification endpoint using the same industry-standard request/response schema as `/v1/classify` (`model`, `input` of 1–64 strings or `{"text": ...}` objects up to 16,000 chars each within a 256 KiB payload, optional `prompt` up to 512 chars, optional `task` up to 64 chars, `2–16` labels with optional descriptions up to 240 chars). Shares the single `ModelWorker` instance, isolates per-item token-limit/model errors as `{"object": "classification", "index": i, "error": "..."}` without failing the rest of the batch, and stays available even when `accept_requests=false`.
- `GET /admin/v1/api-keys`: lists API key metadata (`id`, `name`, `owner`, `key_prefix`, effective `status` of `active`/`revoked`/`expired`, `expires_at`, `last_used_at`, `request_count`, `created_by`, `created_at`, `revoked_at`; never returns `key_hash` or plaintext).
- `POST /admin/v1/api-keys`: creates a new API key (`name`, optional `owner`, `expires_in_days` in `30|90|365|null`, optional `created_by`), returning HTTP `201` with one-time `api_key` plaintext and `record`.
- `POST /admin/v1/api-keys/{id}/revoke`: marks a key `revoked`, immediately invalidates the auth cache, and returns the updated record (`404` if not found).
- `DELETE /admin/v1/api-keys/{id}`: deletes a key, immediately invalidates the auth cache, and returns `{"deleted": true, "id": "..."}` (`404` if not found).

## cubus deployment contract

| Setting | Value |
|---|---|
| Dockerfile / build context | Root `Dockerfile` / repository root |
| Service compose | `compose.service.yml` (`choice` service, extended by `cubus.sh`; exposes `8000` and optional internal `9090`) |
| Listener / intended hostname | `0.0.0.0:8000` / `choice.cubus.sh` (plus optional `ADMIN_BIND_CIDR:9090` internal Admin API) |
| Internal network | Shared `web`, `db`, and optional internal `admin` when Admin API is enabled |
| Proxy metadata | `VIRTUAL_HOST=choice.cubus.sh`, `VIRTUAL_PORT=8000` |
| Database | Internal PostgreSQL via `DB_BROKER_URL` (`svc_choice`) in `cubus.sh`, or `DATABASE_URL` standalone |
| GPU | GPU allocation through NVIDIA Container Toolkit |
| Required secret | Optional `ADMIN_API_TOKEN` for `:9090` Admin API (public API keys are stored hashed in PostgreSQL) |
| Persistent storage | PostgreSQL (`api_keys` table); model files are included in the image |
| Writable paths | `/tmp` and temporary/cache files in the container user's home |
| Build outbound access | PyPI, PyTorch wheel repository, Hugging Face, and download CDNs |
| Runtime outbound access | None required by the server; the browser fetches Swagger UI JS and CSS from a CDN |
