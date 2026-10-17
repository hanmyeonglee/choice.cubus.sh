# choice.cubus.sh

A small FastAPI server that evaluates choice and noul questions by calling
[GLiFormer base](https://huggingface.co/knowledgator/gliformer-base-v1) directly.
It loads one model and processes one request at a time using FP32 and eager attention.

```text
Client → cubus shared ingress → FastAPI :8000 → GLiFormer / GPU
```

## Repository layout

```text
.
├── app/
│   ├── __init__.py
│   ├── __main__.py          # Server entry point
│   ├── api.py               # Routes, authentication, and request handling
│   ├── schemas.py           # Request/response schemas and input validation
│   ├── model.py             # Model loading and device configuration
│   └── static/
│       └── index.html       # Public homepage
├── scripts/
│   └── download_model.py    # Build-time model download
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

The runtime entry point is `python -m app`, run from the repository root or the container's `/app` directory. Docker installs dependencies and downloads model files during the build.

## API

| Method / path | Purpose | Authentication |
|---|---|---|
| `POST /api/evaluate` | Evaluate choice and noul questions against state | Bearer key |
| `GET /api/healthz` | Readiness after model loading | None |
| `GET /` | Introduction and request example | None |
| `GET /docs` | Swagger UI documentation and interactive requests | None |
| `GET /openapi.json` | OpenAPI schema | None |

```bash
export API_KEY='your-api-key'
curl --fail-with-body --max-time 30 http://127.0.0.1:8000/api/evaluate \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice for the same order.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Charges and invoices", "technical": "Technical issues"}
      },
      "duplicate_charge": {
        "type": "noul",
        "instructions": "The customer was charged twice for the same order."
      }
    }
  }'
```

The public endpoint is `https://choice.cubus.sh/api/evaluate`. You can also enter the key itself in **Authorize** at `/docs` to send requests.

### API keys

Configure one or more keys in the server's `API_KEYS` environment variable, separated by commas:

```dotenv
API_KEYS=replace-with-a-random-key,replace-with-another-random-key
```

Replace every placeholder with a separately generated key. Each client sends only its own key as `Authorization: Bearer <key>`; the request examples use the client's `API_KEY` variable for that value. Any configured key grants the same access to `/api/evaluate`.

- Keys are generated and distributed manually; there are no accounts, login, or key issuance endpoints.
- Surrounding whitespace and empty entries in the list are ignored. At least one real key is required, and placeholder keys beginning with `replace-with-` prevent startup.
- `API_KEYS` takes precedence. The legacy `API_KEY` server setting is used only when `API_KEYS` is absent. An explicitly empty `API_KEYS` does not fall back to `API_KEY`.
- Every configured key is compared with `secrets.compare_digest`, without stopping after a match. Missing or invalid credentials return 401.
- Add, remove, or rotate keys by updating the server environment and recreating the container. Keys have no automatic expiration.
- The homepage, docs, OpenAPI schema, and readiness endpoint remain public.

### Response

Scores in this response example are illustrative:

```json
{
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.8, "technical": 0.2}
    },
    "duplicate_charge": {"type": "noul", "noul": 0.9}
  }
}
```

- One model is fixed on the server. Requests have no `model` field.
- `questions`: 1–4 per request. Question IDs associate answers with questions and are not passed to the model.
- `instructions`: question or statement to evaluate, up to 320 characters.
- `choice.criteria`: choice name → description string or null. Accepts 2–8 choices, with names up to 64 characters and descriptions up to 160 characters.
- `noul`: accepts only `type` and `instructions`.
- Only choice and noul questions are supported. Responses omit model, usage, and confidence fields.
- Invalid keys return 401; invalid input returns 422.

### State format

**Use a string by default.** Original prose or labeled sections are passed to the model unchanged. Objects and arrays are also accepted for convenience and serialized as JSON text, preserving keys and values. No specific keys or schema are required. The serialized input is limited to 2,000 characters and is never truncated automatically.

| Source material | Suggested starting point |
|---|---|
| Documents, emails, reviews | Original prose |
| Entities with multiple attributes | Short `label: value` entries or shallow JSON |
| Conversations | Text or arrays identifying speakers and message order |
| Event or task histories | Chronological records and current status |

For example, send the following text as a string state. Encode line breaks as `\n` in request JSON.

```text
Customer message:
I was charged twice for the same order.

Payment records:
Two completed payments belong to order A123.

Current status:
No refund has been issued.
```

If the caller already has an object, it can be sent directly:

```json
{
  "state": {
    "customer_message": "I was charged twice for the same order.",
    "payment_records": "Two completed payments belong to order A123.",
    "current_status": "No refund has been issued."
  }
}
```

This example only shows state; an actual request also requires questions.

- Put source material in `state`, and questions and evaluation criteria in each question's `instructions` and `criteria` fields under `questions`.
- Identify who made each statement and distinguish claims from verified facts.
- Preserve relationships between entities, times, units, and negation. Do not fill in unknown information by guessing.
- Include only information relevant to the question and use consistent formats and labels for the same task.

Objects and arrays also reach the model as text. JSON input does not provide a separate structural processing mechanism or guarantee better accuracy. Compare formats using identical facts and questions, and evaluate accuracy or F1 rather than score magnitude. [Related research on input formats](https://arxiv.org/abs/2411.10541) studies GPT models and does not directly establish GLiFormer's format preferences.

GLiFormer's published evaluation results focus on English. If the primary input language is different, evaluate performance in that language separately. See the [model's evaluation scope](https://huggingface.co/knowledgator/gliformer-base-v1#training-and-intended-use).

### Model calls and scores

The model is called directly, with each question evaluated sequentially in a separate forward pass. More questions increase inference time.

- **choice**: passes the question as the class group name and choice names and descriptions as class labels. Raw sigmoid scores are divided by their sum to produce `probabilities`; the highest-scoring choice is returned. An invalid total score returns 503.
- **noul**: returns the raw sigmoid score from evaluating the `instructions` statement as a single class label.

Neither value is calibrated as a probability of correctness. In particular, noul does not use a separately trained truth evaluation model. Callers handle action permissions and allow/block thresholds.

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
- `MODEL_PATH`: defaults to `/opt/models/gliformer-base-v1`.
- Model weights and tokenizer files are included in the image; there are no external downloads at runtime.
- uv is used only to install locked dependencies during the build. The runtime image does not include uv.
- Uses PyTorch 2.7.1+cu126, GLiFormer 0.1.2, GLiNER 0.2.29, and Transformers 5.16.1. FastAPI, Uvicorn, and all dependencies are pinned in `uv.lock`.
- Model revision: `590f9d3f577ea2f6d685aaec84b2d66b6db86b15`.

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

The central Compose configuration supplies build, GPU, network, and proxy settings. Shared ingress forwards every path unchanged to internal port 8000, preserving `/api`. No host port publication is required. The site and API share an origin, so no additional CORS configuration is needed.

Uvicorn processes forwarded headers. The image default `FORWARDED_ALLOW_IPS=*` is intended for internal connections through shared ingress. It can be overridden at runtime with trusted proxy IPs or CIDRs.

The central deployment can use the Python included in the image for its health check:

```bash
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/healthz', timeout=3).close()"
```

Peak VRAM usage and response time on the GTX 1060 3GB still need to be measured on that server. Weight size alone does not guarantee total VRAM usage.

## Verification results (2026-10-17)

- Docker build and offline model loading succeeded.
- Confirmed choice and noul GPU inference for string, object, and array state on an RTX 3060 Laptop without external network access.
- Confirmed question ID matching, choice scores summing to 1, noul values in 0–1, and rejection of the model field and score questions.
- Confirmed the homepage, docs, readiness, OpenAPI, invalid key responses (401), and input limit responses (422).
- Confirmed that the runtime image contains neither Jeff nor Modal. Validation on the actual GTX 1060 3GB remains pending.

These results precede the subsequent documentation, language, repository layout, and multiple API key edits; the image has not been rebuilt for those edits.
