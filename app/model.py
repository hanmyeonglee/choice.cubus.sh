"""Load supported classification checkpoints (Amber 150M and BGE-M3 NLI zero-shot) and run bounded inference."""

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

# BGE-M3 zero-shot NLI (XLMRobertaForSequenceClassification, 16 heads) memory bounds for GTX 1060 3 GB.
NLI_MAX_TOKENS = 8192
NLI_MAX_PAIRS_PER_BATCH = 8
NLI_TOKEN_BUDGET = 2048
NLI_MAX_ATTN_ELEMENTS = 2 * 1024 * 1024
NLI_CHUNKED_SDPA_NAME = "choice_chunked_sdpa"
NLI_CHUNKED_SDPA_THRESHOLD = 1024
NLI_SDPA_BLOCK = 256

_PRECISIONS = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

_BANDED_REG_LOCK = Lock()
_REGISTERED_TARGETS: WeakSet[Any] = WeakSet()
_CHUNKED_REGISTERED_TARGETS: WeakSet[Any] = WeakSet()


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


def chunked_sdpa_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run query-blocked full SDPA to bound O(L^2) attention memory on 3 GB GPUs.

    Because transformers 5.x returns attention_mask=None for custom attention implementations
    unregistered in ALL_MASK_ATTENTION_FUNCTIONS, unmasked calls MUST be unpadded B=1 to avoid
    silently attending over pad tokens (and intentionally avoids allocating a 4D LxL mask in VRAM).
    """
    del module, dropout, kwargs
    if attention_mask is None and query.shape[0] != 1:
        raise ValueError("chunked_sdpa_attention requires an unpadded batch size of 1 when attention_mask is None.")
    n = query.shape[2]
    if n <= NLI_SDPA_BLOCK:
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=scaling,
        )
        return out.transpose(1, 2).contiguous(), None

    out = torch.empty_like(query)
    mask_is_4d_full = (
        attention_mask is not None
        and hasattr(attention_mask, "shape")
        and len(attention_mask.shape) == 4
        and attention_mask.shape[2] == n
    )
    for s in range(0, n, NLI_SDPA_BLOCK):
        e = min(n, s + NLI_SDPA_BLOCK)
        block_mask = attention_mask[:, :, s:e, :] if mask_is_4d_full else attention_mask
        out[:, :, s:e] = F.scaled_dot_product_attention(
            query[:, :, s:e],
            key,
            value,
            attn_mask=block_mask,
            scale=scaling,
        )
    return out.transpose(1, 2).contiguous(), None


def _register_chunked_sdpa_attention() -> None:
    """Register choice_chunked_sdpa with transformers.AttentionInterface once per interface class."""
    from transformers import AttentionInterface

    with _BANDED_REG_LOCK:
        mapping = getattr(AttentionInterface, "_global_mapping", None)
        if isinstance(mapping, dict) and NLI_CHUNKED_SDPA_NAME in mapping:
            return
        if AttentionInterface in _CHUNKED_REGISTERED_TARGETS and not isinstance(mapping, dict):
            return
        AttentionInterface.register(NLI_CHUNKED_SDPA_NAME, chunked_sdpa_attention)
        _CHUNKED_REGISTERED_TARGETS.add(AttentionInterface)


def _select_attention_impl(enc_cfg: Any, auto_model_cls: Any | None = None) -> str:
    """Choose SDPA when the encoder architecture supports it, otherwise eager."""
    if auto_model_cls is None:
        from transformers import AutoModel

        auto_model_cls = AutoModel

    mapping = getattr(auto_model_cls, "_model_mapping", None)
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


@dataclass(frozen=True, slots=True)
class NliBundle:
    """Loaded Hugging Face NLI sequence classification model (e.g. bge-m3-zeroshot-v2.0)."""

    model: Any
    tokenizer: Any
    pad_token_id: int
    entailment_idx: int
    max_tokens: int
    device: torch.device
    dtype: torch.dtype
    attn_impl: str
    chunked_sdpa: bool


ModelBundle = AmberBundle | NliBundle


def _resolve_entailment_index(config: Any) -> int:
    """Resolve the 'entailment' class index dynamically from model.config.id2label."""
    id2label = getattr(config, "id2label", None)
    if not isinstance(id2label, Mapping) or not id2label:
        raise RuntimeError("Sequence classification model config is missing id2label mapping.")
    matches: list[int] = []
    for raw_idx, raw_label in id2label.items():
        if isinstance(raw_label, str) and raw_label.strip().lower() == "entailment":
            if isinstance(raw_idx, bool):
                continue
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid id2label index {raw_idx!r}.") from exc
            if idx < 0:
                raise RuntimeError(f"Invalid negative id2label index {idx}.")
            matches.append(idx)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one 'entailment' label in model config id2label, found {id2label!r}."
        )
    return matches[0]


def _resolve_nli_max_tokens(config: Any) -> int:
    """Compute the strict maximum sequence token length supported by an NLI encoder config."""
    raw_max_pos = getattr(config, "max_position_embeddings", None)
    if isinstance(raw_max_pos, int) and not isinstance(raw_max_pos, bool) and raw_max_pos > 0:
        model_type = str(getattr(config, "model_type", "") or "").strip().lower()
        if model_type in {"xlm-roberta", "roberta", "camembert"}:
            raw_pad = getattr(config, "pad_token_id", 1)
            pad_id = raw_pad if (isinstance(raw_pad, int) and not isinstance(raw_pad, bool) and raw_pad >= 0) else 1
            pos_limit = raw_max_pos - pad_id - 1
            if pos_limit > 0:
                return min(NLI_MAX_TOKENS, pos_limit)
        return min(NLI_MAX_TOKENS, raw_max_pos)
    return NLI_MAX_TOKENS


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


def _load_amber_model(model_dir: Path, target_device: torch.device, dtype: torch.dtype) -> AmberBundle:
    """Load Amber 150M from a local checkpoint directory strictly offline."""
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
    attn_impl = _select_attention_impl(enc_cfg, AutoModel)
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


def _load_nli_model(model_dir: Path, target_device: torch.device, dtype: torch.dtype) -> NliBundle:
    """Load a Hugging Face NLI sequence classification model strictly offline with CPU-first FP16 conversion."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing model config file: {config_path}")
    if not any(model_dir.glob("*.safetensors")):
        raise RuntimeError(f"Missing safetensors weights file in: {model_dir}")

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True)
    pad_token_id = getattr(tok, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(cfg, "pad_token_id", None)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise RuntimeError("NLI tokenizer/config is missing a valid non-negative integer pad_token_id.")

    _register_chunked_sdpa_attention()
    attn_impl = _select_attention_impl(cfg, AutoModelForSequenceClassification)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
            attn_implementation=attn_impl,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )

    active_cfg = getattr(model, "config", cfg)
    entailment_idx = _resolve_entailment_index(active_cfg)
    max_tokens = _resolve_nli_max_tokens(active_cfg)

    inv_freq_snapshots = _snapshot_inv_freq_buffers(model) if dtype == torch.float16 else {}
    # Downcast to FP16 on CPU first so large embedding matrices (e.g. 250k x 1024) never occupy FP32 VRAM on a 3 GB GPU.
    if dtype == torch.float16 and target_device.type != "cpu":
        model = model.to(dtype=dtype)
    model = model.to(device=target_device, dtype=dtype).eval()
    if dtype == torch.float16 and inv_freq_snapshots:
        _restore_inv_freq_buffers(model, inv_freq_snapshots, target_device)
        del inv_freq_snapshots

    base_attn = getattr(active_cfg, "_attn_implementation", None) or attn_impl
    return NliBundle(
        model=model,
        tokenizer=tok,
        pad_token_id=pad_token_id,
        entailment_idx=entailment_idx,
        max_tokens=max_tokens,
        device=target_device,
        dtype=dtype,
        attn_impl=base_attn,
        chunked_sdpa=(base_attn == "sdpa"),
    )


def load_model(path: str, device: str, precision: str | None = None) -> ModelBundle:
    """Detect checkpoint kind (Amber UniModel vs HF NLI sequence classifier) and load strictly offline."""
    clean_device = device.strip()
    target_device = torch.device(clean_device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable GPU access with the NVIDIA Container Toolkit.")
    dtype = resolve_dtype(clean_device, precision)

    model_dir = Path(path)
    if (model_dir / "uni_config.json").is_file():
        return _load_amber_model(model_dir, target_device, dtype)
    if (model_dir / "config.json").is_file():
        return _load_nli_model(model_dir, target_device, dtype)
    raise RuntimeError(f"Missing model config file (uni_config.json or config.json) in: {model_dir}")


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


@dataclass(frozen=True, slots=True)
class _NliPair:
    """Single tokenized (premise, hypothesis) pair belonging to one batch item and candidate label."""

    item_idx: int
    label_idx: int
    ids: list[int]


def _build_nli_hypotheses(
    labels: list[str] | dict[str, str | None],
    prompt: str | None,
) -> tuple[list[str], list[str]]:
    if isinstance(labels, Mapping):
        label_names = list(labels.keys())
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes, bytearray)):
        label_names = list(labels)
    else:
        raise ValueError("labels must be a sequence or mapping of candidate labels.")

    if len(label_names) < 2:
        raise ValueError("At least 2 candidate labels are required.")
    if any(not isinstance(name, str) or not name.strip() for name in label_names):
        raise ValueError("Label names must be non-blank strings.")
    if len(set(label_names)) != len(label_names):
        raise ValueError("Label names must be unique.")

    if isinstance(labels, Mapping):
        label_texts = [
            f"{k.strip()}: {v.strip()}" if (isinstance(v, str) and v.strip()) else k.strip()
            for k, v in labels.items()
        ]
    else:
        label_texts = [k.strip() for k in label_names]

    if isinstance(prompt, str) and prompt.strip():
        clean_prompt = prompt.strip()
        hypotheses = [f"{clean_prompt} {lt}" for lt in label_texts]
    else:
        hypotheses = [f"This example is {lt}." for lt in label_texts]
    return label_names, hypotheses


def _prepare_nli_item(
    bundle: NliBundle,
    text: str,
    labels: list[str] | dict[str, str | None],
    task: str,
    prompt: str | None,
) -> tuple[list[str], list[list[int]]]:
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-blank string.")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("prompt must be a string or None.")

    label_names, hypotheses = _build_nli_hypotheses(labels, prompt)
    encoded = bundle.tokenizer(
        [text] * len(hypotheses),
        hypotheses,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    raw_input_ids = encoded["input_ids"]
    pair_ids = [list(seq_ids) for seq_ids in raw_input_ids]
    if len(pair_ids) != len(label_names):
        raise ValueError("Tokenizer returned unexpected number of NLI pairs.")

    longest_pair = max((len(ids) for ids in pair_ids), default=0)
    if longest_pair > bundle.max_tokens:
        raise InputTooLongError(
            f"Prepared input length ({longest_pair} tokens) exceeds the {bundle.max_tokens:,}-token limit."
        )
    return label_names, pair_ids


def _classify_nli_batch(
    bundle: NliBundle,
    items: Sequence[tuple[str, list[str] | dict[str, str | None], str, str | None]],
    token_budget: int = NLI_TOKEN_BUDGET,
    max_pairs_per_batch: int = NLI_MAX_PAIRS_PER_BATCH,
) -> list[tuple[str, Any]]:
    """Run VRAM-bounded NLI zero-shot classification over (text, hypothesis) pairs with FP32 softmax."""
    if token_budget <= 0:
        raise ValueError("token_budget must be a positive integer.")
    if max_pairs_per_batch <= 0:
        raise ValueError("max_pairs_per_batch must be a positive integer.")
    if not items:
        return []

    results: list[tuple[str, Any] | None] = [None] * len(items)
    prepared: dict[int, tuple[list[str], list[list[int]]]] = {}
    all_pairs: list[_NliPair] = []
    item_logits: dict[int, list[float | None]] = {}

    for idx, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 4:
            results[idx] = ("error", "Invalid classification item tuple.")
            continue
        text, labels, task, prompt = item
        try:
            label_names, pair_ids = _prepare_nli_item(bundle, text, labels, task, prompt)
        except InputTooLongError as exc:
            results[idx] = ("input_too_long", str(exc))
        except ValueError as exc:
            results[idx] = ("error", str(exc))
        else:
            prepared[idx] = (label_names, pair_ids)
            item_logits[idx] = [None] * len(label_names)
            for label_idx, ids in enumerate(pair_ids):
                all_pairs.append(_NliPair(item_idx=idx, label_idx=label_idx, ids=ids))

    if not prepared:
        return [
            res if res is not None else ("error", "Item did not produce a classification result.")
            for res in results
        ]

    all_pairs.sort(key=lambda pair: len(pair.ids))

    def _run_nli_group(group: list[_NliPair]) -> None:
        max_len = max(len(pair.ids) for pair in group)
        input_ids = torch.full((len(group), max_len), bundle.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(group), max_len), dtype=torch.long)
        for row_idx, pair in enumerate(group):
            n = len(pair.ids)
            input_ids[row_idx, :n] = torch.tensor(pair.ids, dtype=torch.long)
            attention_mask[row_idx, :n] = 1
        input_ids = input_ids.to(bundle.device, non_blocking=True)
        attention_mask = attention_mask.to(bundle.device, non_blocking=True)

        outputs: Any = None
        logits_tensor: Any = None
        try:
            if bundle.chunked_sdpa and max_len > NLI_CHUNKED_SDPA_THRESHOLD:
                if len(group) != 1:
                    raise RuntimeError("choice_chunked_sdpa requires an unpadded batch size of 1.")
                _register_chunked_sdpa_attention()
                cfg = bundle.model.config
                prev_attn = cfg._attn_implementation
                cfg._attn_implementation = NLI_CHUNKED_SDPA_NAME
                try:
                    outputs = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
                finally:
                    cfg._attn_implementation = prev_attn
            else:
                outputs = bundle.model(input_ids=input_ids, attention_mask=attention_mask)

            logits_tensor = outputs.logits
            rows = logits_tensor.float().cpu().tolist()
        finally:
            del input_ids, attention_mask, outputs, logits_tensor

        if len(rows) != len(group):
            raise RuntimeError("NLI model returned unexpected batch logit count.")
        for row_idx, pair in enumerate(group):
            row = rows[row_idx]
            if not isinstance(row, list) or len(row) <= bundle.entailment_idx:
                raise RuntimeError("NLI model returned logits missing entailment index.")
            item_logits[pair.item_idx][pair.label_idx] = float(row[bundle.entailment_idx])

    with torch.inference_mode():
        group: list[_NliPair] = []
        longest = 0
        for pair in all_pairs:
            seq_len = len(pair.ids)
            next_longest = max(longest, seq_len)
            next_count = len(group) + 1
            if group and (
                next_count > max_pairs_per_batch
                or (bundle.chunked_sdpa and seq_len > NLI_CHUNKED_SDPA_THRESHOLD)
                or next_count * next_longest > token_budget
                or next_count * next_longest * next_longest > NLI_MAX_ATTN_ELEMENTS
            ):
                _run_nli_group(group)
                group = []
                longest = 0
            group.append(pair)
            longest = max(longest, seq_len)
            if (bundle.chunked_sdpa and seq_len > NLI_CHUNKED_SDPA_THRESHOLD) or len(group) >= max_pairs_per_batch:
                _run_nli_group(group)
                group = []
                longest = 0
        if group:
            _run_nli_group(group)

    for item_idx, (label_names, pair_ids) in prepared.items():
        raw_scores = item_logits[item_idx]
        if any(score is None or not math.isfinite(score) for score in raw_scores):
            results[item_idx] = ("error", "Model returned invalid classification scores.")
            continue
        finite_scores = [float(score) for score in raw_scores if score is not None]
        probs_tensor = torch.softmax(torch.tensor([finite_scores], dtype=torch.float32), dim=-1)
        probs = [float(v) for v in probs_tensor.tolist()[0]]
        if len(probs) != len(label_names) or any(not math.isfinite(v) or not (0.0 <= v <= 1.0) for v in probs):
            results[item_idx] = ("error", "Model returned invalid classification scores.")
            continue
        probabilities = dict(zip(label_names, probs, strict=True))
        choice = max(probabilities, key=probabilities.__getitem__)
        total_tokens = sum(len(ids) for ids in pair_ids)
        results[item_idx] = ("ok", (choice, probabilities, total_tokens))

    return [res if res is not None else ("error", "Item did not produce a classification result.") for res in results]


def classify_batch(
    bundle: ModelBundle,
    items: Sequence[tuple[str, list[str] | dict[str, str | None], str, str | None]],
    token_budget: int | None = None,
    *,
    max_pairs_per_batch: int = NLI_MAX_PAIRS_PER_BATCH,
) -> list[tuple[str, Any]]:
    """Run micro-batched classification for Amber or NLI bundles, isolating per-item validation/length errors."""
    if isinstance(bundle, NliBundle):
        effective_nli_budget = NLI_TOKEN_BUDGET if token_budget is None else token_budget
        return _classify_nli_batch(
            bundle,
            items,
            token_budget=effective_nli_budget,
            max_pairs_per_batch=max_pairs_per_batch,
        )

    effective_budget = DEFAULT_TOKEN_BUDGET if token_budget is None else token_budget
    if effective_budget <= 0:
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
                or next_count * next_longest > effective_budget
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
    bundle: ModelBundle,
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
