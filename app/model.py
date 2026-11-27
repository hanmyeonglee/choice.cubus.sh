"""Load the fixed Amber 150M checkpoint and run bounded uni-encoder classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any
from weakref import WeakSet

import torch
from torch import nn
import torch.nn.functional as F

MAX_TOKENS = 7999
LONG = 2048
BLOCK = 1024
DEFAULT_TOKEN_BUDGET = 8192
# Bound O(B * L^2) transient attention matrix memory on sm_61 (GTX 1060 3 GB) Math SDPA fallback.
MAX_SDPA_ATTN_ELEMENTS = 4 * 1024 * 1024
BANDED_ATTENTION_NAME = "amber_banded"

_PRECISIONS = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

_BANDED_REG_LOCK = Lock()
_REGISTERED_TARGETS: WeakSet[Any] = WeakSet()


class InputTooLongError(ValueError):
    """Raised when the prepared text and schema exceed the subword token budget."""


def resolve_dtype(device: str, precision: str | None = None) -> torch.dtype:
    """Resolve PyTorch dtype from DEVICE and optional PRECISION env values."""
    is_cpu = device.strip().lower() == "cpu"
    if precision is None or not precision.strip():
        return torch.float32 if is_cpu else torch.float16
    key = precision.strip().lower()
    if key not in _PRECISIONS:
        raise ValueError(f"Unsupported PRECISION {precision!r}; expected 'fp16' or 'fp32'.")
    if is_cpu and _PRECISIONS[key] == torch.float16:
        raise ValueError("CPU execution requires PRECISION=fp32.")
    return _PRECISIONS[key]


class Band:
    """Sliding-window attention mask marker for unpadded long-sequence passes."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError(f"Sliding window must be positive, got {window}.")
        self.window = int(window)


def banded_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run O(L) blocked SDPA on sliding-window layers and standard SDPA on global layers."""
    del module, dropout, kwargs
    if isinstance(attention_mask, Band):
        if query.shape[0] != 1:
            raise ValueError("banded_attention requires an unpadded batch size of 1.")
        w, n = attention_mask.window, query.shape[2]
        pos = torch.arange(n, device=query.device)
        out = torch.empty_like(query)
        for s in range(0, n, BLOCK):
            e = min(n, s + BLOCK)
            a, b = max(0, s - w), min(n, e + w)
            near = (pos[s:e, None] - pos[None, a:b]).abs() <= w
            out[:, :, s:e] = F.scaled_dot_product_attention(
                query[:, :, s:e],
                key[:, :, a:b],
                value[:, :, a:b],
                attn_mask=near,
                scale=scaling,
            )
    else:
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=scaling,
        )
    return out.transpose(1, 2).contiguous(), None


def _register_banded_attention() -> None:
    """Register amber_banded with transformers.AttentionInterface once per interface class."""
    from transformers import AttentionInterface

    with _BANDED_REG_LOCK:
        mapping = getattr(AttentionInterface, "_global_mapping", None)
        if isinstance(mapping, dict) and BANDED_ATTENTION_NAME in mapping:
            return
        if AttentionInterface in _REGISTERED_TARGETS and not isinstance(mapping, dict):
            return
        AttentionInterface.register(BANDED_ATTENTION_NAME, banded_attention)
        _REGISTERED_TARGETS.add(AttentionInterface)


def _select_attention_impl(enc_cfg: Any) -> str:
    """Choose SDPA when the encoder architecture supports it, otherwise eager."""
    from transformers import AutoModel

    mapping = getattr(AutoModel, "_model_mapping", None)
    if mapping is None:
        return "eager"
    try:
        model_cls = mapping[type(enc_cfg)]
    except (KeyError, TypeError):
        return "eager"
    return "sdpa" if getattr(model_cls, "_supports_sdpa", False) else "eager"


@dataclass(frozen=True, slots=True)
class Special:
    """Special token IDs required to assemble uni-layout sequences."""

    cls: int
    sep: int
    mask: int
    pad: int

    @classmethod
    def of(cls, tok: Any) -> Special:
        missing = []
        values: dict[str, int] = {}
        for attr, field_name in (
            ("cls_token_id", "cls"),
            ("sep_token_id", "sep"),
            ("mask_token_id", "mask"),
            ("pad_token_id", "pad"),
        ):
            val = getattr(tok, attr, None)
            if isinstance(val, bool) or not isinstance(val, int):
                missing.append(attr)
            else:
                values[field_name] = val
        if missing:
            raise RuntimeError(f"Tokenizer is missing required special token IDs: {', '.join(missing)}.")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Seq:
    """Prepared uni-layout token sequence and option span offsets."""

    ids: list[int]
    markers: list[int]
    spans: list[tuple[int, int]]


@dataclass
class Batch:
    """Padded mini-batch tensors for UniModel.logits."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    seq: torch.Tensor
    marker: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    question: torch.Tensor
    slot: torch.Tensor
    n_questions: int
    max_options: int

    def to(self, device: torch.device | str) -> Batch:
        moved = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in self.__dict__.items()
        }
        return Batch(**moved)


def collate(seqs: list[Seq], owners: list[tuple[int, list[int]]], pad: int) -> Batch:
    """Collate variable-length uni sequences into a padded Batch."""
    if not seqs:
        raise ValueError("Cannot collate an empty sequence list.")
    length = max(len(s.ids) for s in seqs)
    ids = torch.full((len(seqs), length), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), length), dtype=torch.long)
    seq_idx: list[int] = []
    marker: list[int] = []
    start: list[int] = []
    end: list[int] = []
    question: list[int] = []
    slot: list[int] = []

    for i, (s, (qi, slots)) in enumerate(zip(seqs, owners, strict=True)):
        ids[i, : len(s.ids)] = torch.tensor(s.ids, dtype=torch.long)
        mask[i, : len(s.ids)] = 1
        for m, (a, b), k in zip(s.markers, s.spans, slots, strict=True):
            seq_idx.append(i)
            marker.append(m)
            start.append(a)
            end.append(b)
            question.append(qi)
            slot.append(k)

    def _long_tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long)

    return Batch(
        input_ids=ids,
        attention_mask=mask,
        seq=_long_tensor(seq_idx),
        marker=_long_tensor(marker),
        start=_long_tensor(start),
        end=_long_tensor(end),
        question=_long_tensor(question),
        slot=_long_tensor(slot),
        n_questions=max(question) + 1,
        max_options=max(slot) + 1,
    )


class UniModel(nn.Module):
    """Amber uni-encoder model matching the checkpoint's encoder + option head state_dict."""

    def __init__(self, encoder: nn.Module, hidden: int, width: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        resolved_width = width if width is not None else hidden
        if hidden <= 0 or resolved_width <= 0:
            raise ValueError(f"hidden and width must be positive integers (got hidden={hidden}, width={resolved_width}).")
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, resolved_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(resolved_width, 1),
        )

    def logits(self, batch: Batch, masks: dict[str, Any] | None = None) -> torch.Tensor:
        """Compute (n_questions, max_options) logits, -inf for unused option slots."""
        outputs = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask if masks is None else masks,
        )
        h = outputs.last_hidden_state
        cs = F.pad(h.float().cumsum(1), (0, 0, 1, 0))
        lengths = (batch.end - batch.start).clamp(min=1).unsqueeze(-1).float()
        mean = (cs[batch.seq, batch.end] - cs[batch.seq, batch.start]) / lengths
        feats = torch.cat(
            [h[batch.seq, batch.marker].float(), mean, h[batch.seq, 0].float()],
            dim=-1,
        )
        head_dtype = self.head[0].weight.dtype
        score = self.head(feats.to(head_dtype)).squeeze(-1).float()
        out = torch.full((batch.n_questions, batch.max_options), float("-inf"), device=score.device)
        out[batch.question, batch.slot] = score
        return out


@dataclass(frozen=True, slots=True)
class AmberBundle:
    """Loaded Amber 150M model, tokenizer, and attention routing metadata."""

    model: UniModel
    tokenizer: Any
    special: Special
    device: torch.device
    dtype: torch.dtype
    layout: str
    attn_impl: str
    banded: bool
    sliding_window: int


def _snapshot_inv_freq_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot FP32 RoPE *_inv_freq buffers before FP16 downcasting."""
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_buffers):
        return {}
    try:
        entries = list(named_buffers(remove_duplicate=False))
    except TypeError:
        entries = list(named_buffers())
    snapshots: dict[str, torch.Tensor] = {}
    for name, buf in entries:
        if name.endswith("inv_freq") and buf is not None:
            cloned = buf.detach().clone() if hasattr(buf, "detach") else buf
            snapshots[name] = cloned.to(dtype=torch.float32)
    return snapshots


def _restore_inv_freq_buffers(
    model: nn.Module,
    snapshots: dict[str, torch.Tensor],
    target_device: torch.device,
) -> None:
    """Restore snapshotted RoPE *_inv_freq buffers on target_device in FP32."""
    for name, orig_fp32 in snapshots.items():
        if "." in name:
            parent_path, buf_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
        else:
            parent, buf_name = model, name
        restored = orig_fp32.to(device=target_device, dtype=torch.float32)
        buffers = getattr(parent, "_buffers", None)
        if isinstance(buffers, dict) and buf_name in buffers:
            buffers[buf_name] = restored
        else:
            setattr(parent, buf_name, restored)


def load_model(path: str, device: str, precision: str | None = None) -> AmberBundle:
    """Load Amber 150M from a local checkpoint directory strictly offline."""
    clean_device = device.strip()
    target_device = torch.device(clean_device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable GPU access with the NVIDIA Container Toolkit.")
    dtype = resolve_dtype(clean_device, precision)

    model_dir = Path(path)
    uni_config_path = model_dir / "uni_config.json"
    weights_path = model_dir / "model.safetensors"
    encoder_dir = model_dir / "encoder"

    if not uni_config_path.is_file():
        raise RuntimeError(f"Missing Amber config file: {uni_config_path}")
    if not weights_path.is_file():
        raise RuntimeError(f"Missing Amber weights file: {weights_path}")
    if not encoder_dir.is_dir():
        raise RuntimeError(f"Missing Amber encoder directory: {encoder_dir}")

    cfg = json.loads(uni_config_path.read_text(encoding="utf-8"))
    layout = str(cfg.get("layout", ""))
    if layout != "uni":
        raise ValueError(f"Unsupported Amber layout {layout!r}; expected 'uni'.")
    hidden = int(cfg["hidden"])
    width = int(cfg["width"]) if cfg.get("width") is not None else hidden
    dropout = float(cfg.get("dropout", 0.1))

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    enc_cfg = AutoConfig.from_pretrained(str(encoder_dir), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
    sp = Special.of(tok)

    _register_banded_attention()
    attn_impl = _select_attention_impl(enc_cfg)
    encoder = AutoModel.from_config(enc_cfg, attn_implementation=attn_impl)
    model = UniModel(encoder, hidden=hidden, width=width, dropout=dropout)

    # Load weights on CPU first to avoid holding duplicate copies on the 3 GB GPU.
    state_dict = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    inv_freq_snapshots = _snapshot_inv_freq_buffers(model) if dtype == torch.float16 else {}
    model = model.to(device=target_device, dtype=dtype).eval()
    if dtype == torch.float16 and inv_freq_snapshots:
        _restore_inv_freq_buffers(model, inv_freq_snapshots, target_device)
        del inv_freq_snapshots

    base_attn = getattr(encoder.config, "_attn_implementation", None) or attn_impl
    raw_sliding_window = getattr(enc_cfg, "sliding_window", None)
    if raw_sliding_window is None:
        raw_sliding_window = getattr(enc_cfg, "local_attention", 128) // 2
    sliding_window = int(raw_sliding_window)
    if sliding_window <= 0:
        raise ValueError(f"Encoder sliding_window must be positive, got {sliding_window}.")
    banded = getattr(enc_cfg, "model_type", None) == "modernbert" and base_attn == "sdpa" and layout == "uni"

    return AmberBundle(
        model=model,
        tokenizer=tok,
        special=sp,
        device=target_device,
        dtype=dtype,
        layout=layout,
        attn_impl=base_attn,
        banded=banded,
        sliding_window=sliding_window,
    )


def _prepare_item(
    bundle: AmberBundle,
    text: str,
    labels: list[str] | dict[str, str | None],
    task: str,
    prompt: str | None,
) -> tuple[list[str], Seq]:
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-blank string.")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("prompt must be a string or None.")

    if isinstance(labels, Mapping):
        label_names = list(labels.keys())
        option_texts = [
            v.strip() if (isinstance(v, str) and v.strip()) else k
            for k, v in labels.items()
        ]
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes, bytearray)):
        label_names = list(labels)
        option_texts = list(label_names)
    else:
        raise ValueError("labels must be a sequence or mapping of candidate labels.")

    if len(label_names) < 2:
        raise ValueError("At least 2 candidate labels are required.")
    if any(not isinstance(name, str) or not name.strip() for name in label_names):
        raise ValueError("Label names must be non-blank strings.")
    if len(set(label_names)) != len(label_names):
        raise ValueError("Label names must be unique.")

    instructions = (
        prompt.strip()
        if (isinstance(prompt, str) and prompt.strip())
        else f"Classify the text for task '{task.strip()}'."
    )

    encoded = bundle.tokenizer([instructions, text, *option_texts], add_special_tokens=False)["input_ids"]
    q_ids = list(encoded[0])
    s_ids = list(encoded[1])
    opt_ids = [list(item_ids) for item_ids in encoded[2:]]

    total_tokens = 4 + len(q_ids) + len(s_ids) + sum(1 + len(o) for o in opt_ids)
    if total_tokens > MAX_TOKENS:
        raise InputTooLongError(
            f"Prepared input length ({total_tokens} tokens) exceeds the {MAX_TOKENS:,}-token limit."
        )

    sp = bundle.special
    ids = [sp.cls, *q_ids, sp.sep, *s_ids, sp.sep]
    markers: list[int] = []
    spans: list[tuple[int, int]] = []
    for opt in opt_ids:
        markers.append(len(ids))
        ids.append(sp.mask)
        start = len(ids)
        ids.extend(opt)
        spans.append((start, len(ids)))
    ids.append(sp.sep)
    return label_names, Seq(ids=ids, markers=markers, spans=spans)


def classify_batch(
    bundle: AmberBundle,
    items: Sequence[tuple[str, list[str] | dict[str, str | None], str, str | None]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[tuple[str, Any]]:
    """Run micro-batched single-pass classification, isolating per-item validation/length errors."""
    if token_budget <= 0:
        raise ValueError("token_budget must be a positive integer.")
    if not items:
        return []

    results: list[tuple[str, Any] | None] = [None] * len(items)
    prepared: dict[int, tuple[list[str], Seq]] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 4:
            results[idx] = ("error", "Invalid classification item tuple.")
            continue
        text, labels, task, prompt = item
        try:
            prepared[idx] = _prepare_item(bundle, text, labels, task, prompt)
        except InputTooLongError as exc:
            results[idx] = ("input_too_long", str(exc))
        except ValueError as exc:
            results[idx] = ("error", str(exc))

    if not prepared:
        return [
            res if res is not None else ("error", "Item did not produce a classification result.")
            for res in results
        ]

    order = sorted(prepared, key=lambda idx: len(prepared[idx][1].ids))

    def _run_group(group: list[int]) -> None:
        seqs = [prepared[idx][1] for idx in group]
        owners = [(qi, list(range(len(seqs[qi].markers)))) for qi in range(len(group))]
        batch = collate(seqs, owners, bundle.special.pad).to(bundle.device)
        seq_len = int(batch.input_ids.shape[1])

        if bundle.banded and seq_len > LONG:
            _register_banded_attention()
            enc_cfg = bundle.model.encoder.config
            prev_attn = enc_cfg._attn_implementation
            enc_cfg._attn_implementation = BANDED_ATTENTION_NAME
            try:
                logits = bundle.model.logits(
                    batch,
                    masks={
                        "full_attention": None,
                        "sliding_attention": Band(bundle.sliding_window),
                    },
                )
            finally:
                enc_cfg._attn_implementation = prev_attn
        else:
            logits = bundle.model.logits(batch)

        probs_cpu = torch.softmax(logits.float(), dim=-1).cpu()
        for qi, item_idx in enumerate(group):
            label_names, seq = prepared[item_idx]
            n_opts = len(label_names)
            probs = [float(v) for v in probs_cpu[qi, :n_opts].tolist()]
            if len(probs) != n_opts or any(not math.isfinite(v) or not (0.0 <= v <= 1.0) for v in probs):
                results[item_idx] = ("error", "Model returned invalid classification scores.")
                continue
            probabilities = dict(zip(label_names, probs, strict=True))
            choice = max(probabilities, key=probabilities.__getitem__)
            results[item_idx] = ("ok", (choice, probabilities, len(seq.ids)))

    with torch.inference_mode():
        group: list[int] = []
        longest = 0
        for idx in order:
            seq_len = len(prepared[idx][1].ids)
            next_longest = max(longest, seq_len)
            next_count = len(group) + 1
            if group and (
                (bundle.banded and seq_len > LONG)
                or next_count * next_longest > token_budget
                or next_count * next_longest * next_longest > MAX_SDPA_ATTN_ELEMENTS
            ):
                _run_group(group)
                group = []
                longest = 0
            group.append(idx)
            longest = max(longest, seq_len)
            if bundle.banded and seq_len > LONG:
                _run_group(group)
                group = []
                longest = 0
        if group:
            _run_group(group)

    return [res if res is not None else ("error", "Item did not produce a classification result.") for res in results]


def classify_text(
    bundle: AmberBundle,
    text: str,
    labels: list[str] | dict[str, str | None],
    task: str = "decision",
    prompt: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Classify a single request synchronously against competing candidate labels."""
    status, payload = classify_batch(bundle, [(text, labels, task, prompt)])[0]
    if status == "ok":
        return payload[0], payload[1]
    if status == "input_too_long":
        raise InputTooLongError(payload)
    raise RuntimeError(payload)
