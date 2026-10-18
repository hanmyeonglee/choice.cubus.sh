"""Load the fixed GLiNER2.5-Decide checkpoint and run bounded single-request classification."""

import math
import torch
from gliner2 import AutoExtractor

MAX_TOKENS = 1024
_PRECISIONS = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class InputTooLongError(ValueError):
    """Raised when the prepared text and schema exceed the subword token budget."""


def resolve_dtype(device: str, precision: str | None = None) -> torch.dtype:
    if precision is None or not precision.strip():
        return torch.float16 if device == "cuda" else torch.float32
    key = precision.strip().lower()
    if key not in _PRECISIONS:
        raise ValueError(f"Unsupported PRECISION {precision!r}; expected 'fp16' or 'fp32'.")
    if device == "cpu" and _PRECISIONS[key] == torch.float16:
        raise ValueError("CPU execution requires PRECISION=fp32.")
    return _PRECISIONS[key]


def load_model(path: str, device: str, precision: str | None = None):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable GPU access with the NVIDIA Container Toolkit.")
    dtype = resolve_dtype(device, precision)
    # Load weights on CPU first to avoid holding duplicate copies on the 3 GB GPU.
    model = AutoExtractor.from_pretrained(
        str(path),
        attention_backend="standard",
        local_files_only=True,
        map_location="cpu",
    )
    return model.to(device=device, dtype=dtype).eval()


def classify_text(
    model,
    text: str,
    labels: list[str] | dict[str, str | None],
    task: str = "decision",
    prompt: str | None = None,
) -> tuple[str, dict[str, float]]:
    label_names = list(labels)
    descriptions = (
        {name: desc for name, desc in labels.items() if desc}
        if isinstance(labels, dict)
        else {}
    )
    schema = model.create_schema().classification(
        task,
        label_names,
        multi_label=False,
        class_act="softmax",
        **({"prompt": prompt} if prompt else {}),
    )
    if descriptions:
        schema.schema["classifications"][0]["label_descriptions"] = descriptions

    batch = model.processor.collate_fn_inference(
        [(text, schema)], max_len=None, error_policy="raise"
    )
    tokens = int(batch.attention_mask.sum().item())
    if tokens > MAX_TOKENS:
        raise InputTooLongError(
            f"Prepared input length ({tokens} tokens) exceeds the {MAX_TOKENS}-token limit."
        )

    schema_tokens = batch.schema_tokens_list[0][0]
    encoded_labels = [
        schema_tokens[i + 1]
        for i, token in enumerate(schema_tokens[:-1])
        if token == "[L]"
    ]
    if encoded_labels != label_names:
        raise RuntimeError("Encoded label order differs from the requested label order.")

    target_device = next(model.parameters()).device
    batch = batch.to(target_device)
    encoded = model.encode_tokens(batch)
    _, schemas = model.processor.extract_embeddings_from_batch(
        encoded, batch.input_ids, batch
    )
    computed = model._compute_classification_probabilities(
        batch.original_schemas[0], torch.stack(schemas[0][0]), schema_tokens
    )
    if computed is None:
        raise RuntimeError("Model failed to compute classification probabilities.")
    _, _, probs_tensor = computed
    probs = [float(value) for value in probs_tensor.tolist()]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probs):
        raise RuntimeError("Model returned invalid classification scores.")

    probabilities = dict(zip(label_names, probs, strict=True))
    choice = max(probabilities, key=probabilities.get)
    return choice, probabilities
