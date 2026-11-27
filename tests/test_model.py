"""Unit tests for Amber 150M model loading, uni sequence assembly, micro-batching, and banded attention."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest


def _make_fake_ml_modules():
    """Build lightweight pure-Python torch/transformers/safetensors modules for CPU unit tests."""
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    nn_func_mod = types.ModuleType("torch.nn.functional")

    class _FakeDType:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:
            return f"torch.{self.name}"

    class _FakeDevice:
        def __init__(self, spec: object = "cpu") -> None:
            raw = spec.type if isinstance(spec, _FakeDevice) else str(spec)
            self.raw = raw
            self.type = raw.split(":", 1)[0].strip().lower()

        def __str__(self) -> str:
            return self.raw

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeDevice) and self.raw == other.raw

    torch_mod.float16 = _FakeDType("float16")
    torch_mod.float32 = _FakeDType("float32")
    torch_mod.bfloat16 = _FakeDType("bfloat16")
    torch_mod.long = _FakeDType("int64")
    torch_mod.bool = _FakeDType("bool")
    torch_mod.device = _FakeDevice
    torch_mod.__version__ = "2.7.1+cu126-fake"

    cuda_state = {"available": False}
    cuda_mod = types.SimpleNamespace(is_available=lambda: cuda_state["available"])
    torch_mod.cuda = cuda_mod

    @contextmanager
    def _inference_mode():
        yield

    torch_mod.inference_mode = _inference_mode
    torch_mod.no_grad = _inference_mode

    def _infer_shape(data):
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        return (len(data),) + _infer_shape(data[0])

    def _deep_copy(data):
        if isinstance(data, list):
            return [_deep_copy(x) for x in data]
        return data

    def _fill_shape(shape, value):
        if not shape:
            return value
        return [_fill_shape(shape[1:], value) for _ in range(shape[0])]

    def _apply_unary(data, fn):
        if isinstance(data, list):
            return [_apply_unary(x, fn) for x in data]
        return fn(data)

    def _apply_binary(a, b, fn):
        if isinstance(a, list) and isinstance(b, list):
            return [_apply_binary(x, y, fn) for x, y in zip(a, b)]
        if isinstance(a, list):
            return [_apply_binary(x, b, fn) for x in a]
        if isinstance(b, list):
            return [_apply_binary(a, y, fn) for y in b]
        return fn(a, b)

    class _FakeTensor:
        def __init__(self, data, dtype=None, device=None) -> None:
            self._data = _deep_copy(data)
            self.dtype = dtype or torch_mod.float32
            self.device = device if isinstance(device, _FakeDevice) else _FakeDevice(device or "cpu")

        @property
        def shape(self) -> tuple[int, ...]:
            return _infer_shape(self._data)

        def float(self) -> _FakeTensor:
            return _FakeTensor(_apply_unary(self._data, float), dtype=torch_mod.float32, device=self.device)

        def detach(self) -> _FakeTensor:
            return self

        def clone(self) -> _FakeTensor:
            return _FakeTensor(self._data, dtype=self.dtype, device=self.device)

        def to(self, *args, **kwargs) -> _FakeTensor:
            target_device = kwargs.get("device", self.device)
            target_dtype = kwargs.get("dtype", self.dtype)
            for arg in args:
                if isinstance(arg, _FakeDType):
                    target_dtype = arg
                elif isinstance(arg, (_FakeDevice, str)):
                    target_device = _FakeDevice(arg)
            if isinstance(target_device, str):
                target_device = _FakeDevice(target_device)
            data = self._data
            if target_dtype == torch_mod.float16 and self.dtype in (torch_mod.float32, torch_mod.float16):
                def _to_fp16(v):
                    try:
                        return struct.unpack("e", struct.pack("e", float(v)))[0]
                    except OverflowError:
                        return math.copysign(float("inf"), float(v))
                data = _apply_unary(data, _to_fp16)
            return _FakeTensor(data, dtype=target_dtype, device=target_device)

        def cpu(self) -> _FakeTensor:
            return self.to(_FakeDevice("cpu"))

        def tolist(self):
            return _deep_copy(self._data)

        def squeeze(self, dim: int = -1) -> _FakeTensor:
            if dim == -1 and len(self.shape) == 2 and self.shape[1] == 1:
                return _FakeTensor([row[0] for row in self._data], dtype=self.dtype, device=self.device)
            raise NotImplementedError("Only squeeze(-1) on (N, 1) is implemented in fake tensor")

        def unsqueeze(self, dim: int) -> _FakeTensor:
            if dim == -1:
                if len(self.shape) == 1:
                    return _FakeTensor([[x] for x in self._data], dtype=self.dtype, device=self.device)
                if len(self.shape) == 2:
                    return _FakeTensor([[x] for row in self._data for x in row], dtype=self.dtype, device=self.device)
            raise NotImplementedError("Unsupported unsqueeze dim")

        def clamp(self, min: float | None = None) -> _FakeTensor:
            return _FakeTensor(
                _apply_unary(self._data, lambda x: max(min, x) if min is not None else x),
                dtype=self.dtype,
                device=self.device,
            )

        def cumsum(self, dim: int) -> _FakeTensor:
            assert dim == 1 and len(self.shape) == 3
            out = []
            for row in self._data:
                hidden = len(row[0])
                acc = [0.0] * hidden
                row_out = []
                for vec in row:
                    acc = [a + v for a, v in zip(acc, vec)]
                    row_out.append(list(acc))
                out.append(row_out)
            return _FakeTensor(out, dtype=self.dtype, device=self.device)

        def transpose(self, d0: int, d1: int) -> _FakeTensor:
            assert (d0, d1) in {(1, 2), (2, 1)} and len(self.shape) == 4
            b, h, l, d = self.shape
            out = [
                [[self._data[bi][hi][li] for hi in range(h)] for li in range(l)]
                for bi in range(b)
            ]
            return _FakeTensor(out, dtype=self.dtype, device=self.device)

        def contiguous(self) -> _FakeTensor:
            return _FakeTensor(self._data, dtype=self.dtype, device=self.device)

        def abs(self) -> _FakeTensor:
            return _FakeTensor(_apply_unary(self._data, abs), dtype=self.dtype, device=self.device)

        def __sub__(self, other):
            rhs = other._data if isinstance(other, _FakeTensor) else other
            # Handle 2D broadcasting (M, 1) - (1, N)
            if isinstance(rhs, list) and len(self.shape) == 2 and self.shape[1] == 1 and len(_infer_shape(rhs)) == 2 and _infer_shape(rhs)[0] == 1:
                row = rhs[0]
                return _FakeTensor([[a[0] - b for b in row] for a in self._data], dtype=self.dtype, device=self.device)
            return _FakeTensor(_apply_binary(self._data, rhs, lambda x, y: x - y), dtype=self.dtype, device=self.device)

        def __truediv__(self, other):
            rhs = other._data if isinstance(other, _FakeTensor) else other
            return _FakeTensor(_apply_binary(self._data, rhs, lambda x, y: x / y), dtype=self.dtype, device=self.device)

        def __le__(self, other):
            rhs = other._data if isinstance(other, _FakeTensor) else other
            return _FakeTensor(_apply_binary(self._data, rhs, lambda x, y: x <= y), dtype=torch_mod.bool, device=self.device)

        def __getitem__(self, item):
            # 2D fancy index: t[seq_tensor, pos_tensor] or t[seq_tensor, int]
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], _FakeTensor) and isinstance(item[1], _FakeTensor):
                idx0 = item[0]._data
                idx1 = item[1]._data
                return _FakeTensor([_deep_copy(self._data[a][b]) for a, b in zip(idx0, idx1)], dtype=self.dtype, device=self.device)
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], _FakeTensor) and isinstance(item[1], int):
                idx0 = item[0]._data
                col = item[1]
                return _FakeTensor([_deep_copy(self._data[a][col]) for a in idx0], dtype=self.dtype, device=self.device)
            # 2D slice: t[i, :n] or t[i, :k]
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], slice):
                return _FakeTensor(self._data[item[0]][item[1]], dtype=self.dtype, device=self.device)
            # 1D slice on pos: pos[s:e, None] or pos[None, a:b]
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], slice) and item[1] is None:
                return _FakeTensor([[x] for x in self._data[item[0]]], dtype=self.dtype, device=self.device)
            if isinstance(item, tuple) and len(item) == 2 and item[0] is None and isinstance(item[1], slice):
                return _FakeTensor([self._data[item[1]]], dtype=self.dtype, device=self.device)
            # 4D slice: query[:, :, s:e]
            if isinstance(item, tuple) and len(item) == 3 and item[0] == slice(None) and item[1] == slice(None) and isinstance(item[2], slice):
                sl = item[2]
                return _FakeTensor([[head[sl] for head in batch] for batch in self._data], dtype=self.dtype, device=self.device)
            raise NotImplementedError(f"Unsupported __getitem__: {item!r}")

        def __setitem__(self, item, value):
            val = value._data if isinstance(value, _FakeTensor) else value
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], slice):
                start, stop, step = item[1].indices(len(self._data[item[0]]))
                assert step == 1
                if isinstance(val, list):
                    self._data[item[0]][start:stop] = _deep_copy(val)
                else:
                    self._data[item[0]][start:stop] = [val] * (stop - start)
                return
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], _FakeTensor) and isinstance(item[1], _FakeTensor):
                for a, b, v in zip(item[0]._data, item[1]._data, val):
                    self._data[a][b] = v
                return
            if isinstance(item, tuple) and len(item) == 3 and item[0] == slice(None) and item[1] == slice(None) and isinstance(item[2], slice):
                sl = item[2]
                for bi, batch in enumerate(self._data):
                    for hi, head in enumerate(batch):
                        head[sl] = _deep_copy(val[bi][hi])
                return
            raise NotImplementedError(f"Unsupported __setitem__: {item!r}")

    torch_mod.Tensor = _FakeTensor
    torch_mod.is_tensor = lambda obj: isinstance(obj, _FakeTensor)
    torch_mod.tensor = lambda data, dtype=None, device=None: _FakeTensor(data, dtype=dtype, device=device)
    torch_mod.full = lambda shape, fill_value, dtype=None, device=None: _FakeTensor(
        _fill_shape(shape, fill_value), dtype=dtype, device=device
    )
    torch_mod.zeros = lambda shape, dtype=None, device=None: _FakeTensor(
        _fill_shape(shape, 0), dtype=dtype, device=device
    )
    torch_mod.arange = lambda n, device=None: _FakeTensor(list(range(n)), dtype=torch_mod.long, device=device)
    torch_mod.empty_like = lambda t: _FakeTensor(_fill_shape(t.shape, 0.0), dtype=t.dtype, device=t.device)

    def _cat(tensors, dim=0):
        assert dim == -1 and len(tensors[0].shape) == 2
        n = tensors[0].shape[0]
        rows = []
        for i in range(n):
            row = []
            for t in tensors:
                row.extend(t._data[i])
            rows.append(row)
        return _FakeTensor(rows, dtype=tensors[0].dtype, device=tensors[0].device)

    def _softmax(tensor, dim=-1):
        assert dim == -1 and len(tensor.shape) == 2
        out = []
        for row in tensor._data:
            if any(math.isnan(v) or (math.isinf(v) and v > 0) for v in row):
                out.append([float("nan")] * len(row))
                continue
            finite_vals = [v for v in row if not (math.isinf(v) and v < 0)]
            if not finite_vals:
                out.append([float("nan")] * len(row))
                continue
            m = max(finite_vals)
            exps = [0.0 if (math.isinf(v) and v < 0) else math.exp(v - m) for v in row]
            total = sum(exps)
            out.append([e / total for e in exps])
        return _FakeTensor(out, dtype=torch_mod.float32, device=tensor.device)

    torch_mod.cat = _cat
    torch_mod.softmax = _softmax

    class _Module:
        def __init__(self) -> None:
            object.__setattr__(self, "_modules", {})
            object.__setattr__(self, "_buffers", {})
            object.__setattr__(self, "_eval", False)

        def __setattr__(self, name: str, value) -> None:
            if isinstance(value, _Module):
                self._modules[name] = value
            object.__setattr__(self, name, value)

        def register_buffer(self, name: str, tensor: _FakeTensor) -> None:
            self._buffers[name] = tensor

        def __getattr__(self, name: str):
            buffers = self.__dict__.get("_buffers", {})
            if name in buffers:
                return buffers[name]
            modules = self.__dict__.get("_modules", {})
            if name in modules:
                return modules[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

        def named_buffers(self, prefix: str = "", remove_duplicate: bool = True, _seen: set[int] | None = None):
            seen = set() if _seen is None else _seen
            for buf_name, buf in self._buffers.items():
                if buf is not None and remove_duplicate:
                    if id(buf) in seen:
                        continue
                    seen.add(id(buf))
                full = f"{prefix}.{buf_name}" if prefix else buf_name
                yield full, buf
            for mod_name, mod in self._modules.items():
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from mod.named_buffers(prefix=sub_prefix, remove_duplicate=remove_duplicate, _seen=seen)

        def get_submodule(self, target: str):
            cur = self
            for part in target.split("."):
                cur = cur._modules[part]
            return cur

        def to(self, device=None, dtype=None):
            for buf_name, buf in list(self._buffers.items()):
                if buf is not None:
                    self._buffers[buf_name] = buf.to(device=device, dtype=dtype)
            for mod in self._modules.values():
                mod.to(device=device, dtype=dtype)
            return self

        def eval(self):
            self._eval = True
            return self

        def load_state_dict(self, state_dict, strict: bool = True):
            self._loaded_state = (state_dict, strict)
            return ([], [])

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class _Linear(_Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = types.SimpleNamespace(dtype=torch_mod.float32)
            self.bias = types.SimpleNamespace(dtype=torch_mod.float32)

        def to(self, device=None, dtype=None):
            if dtype is not None:
                self.weight.dtype = dtype
                self.bias.dtype = dtype
            return self

        def forward(self, x: _FakeTensor) -> _FakeTensor:
            # Simple deterministic projection over features so scores vary with input features
            rows = []
            for vec in x._data:
                val = sum(vec[: self.out_features]) if self.out_features > 1 else sum(vec)
                if self.out_features == 1:
                    rows.append([val])
                else:
                    rows.append([vec[i % len(vec)] for i in range(self.out_features)])
            return _FakeTensor(rows, dtype=x.dtype, device=x.device)

    class _GELU(_Module):
        def forward(self, x):
            return x

    class _Dropout(_Module):
        def __init__(self, p: float = 0.1) -> None:
            super().__init__()
            self.p = p

        def forward(self, x):
            return x

    class _Sequential(_Module):
        def __init__(self, *layers) -> None:
            super().__init__()
            self._layers = list(layers)

        def __getitem__(self, idx: int):
            return self._layers[idx]

        def to(self, device=None, dtype=None):
            for layer in self._layers:
                layer.to(device=device, dtype=dtype)
            return self

        def forward(self, x):
            out = x
            for layer in self._layers:
                out = layer(out)
            return out

    nn_mod.Module = _Module
    nn_mod.Linear = _Linear
    nn_mod.GELU = _GELU
    nn_mod.Dropout = _Dropout
    nn_mod.Sequential = _Sequential

    def _pad(tensor: _FakeTensor, pad: tuple[int, int, int, int]) -> _FakeTensor:
        assert pad == (0, 0, 1, 0) and len(tensor.shape) == 3
        hidden = tensor.shape[2]
        out = [[[0.0] * hidden] + row for row in tensor._data]
        return _FakeTensor(out, dtype=tensor.dtype, device=tensor.device)

    sdpa_calls: list[dict[str, object]] = []

    def _sdpa(query, key, value, attn_mask=None, scale=None):
        sdpa_calls.append(
            {
                "q_shape": query.shape,
                "k_shape": key.shape,
                "mask_shape": attn_mask.shape if isinstance(attn_mask, _FakeTensor) else None,
                "has_mask": attn_mask is not None,
                "scale": scale,
            }
        )
        # Return query-shaped tensor with deterministic values
        return _FakeTensor(query._data, dtype=query.dtype, device=query.device)

    nn_func_mod.pad = _pad
    nn_func_mod.scaled_dot_product_attention = _sdpa
    nn_mod.functional = nn_func_mod
    torch_mod.nn = nn_mod

    # Fake transformers module
    transformers_mod = types.ModuleType("transformers")
    attn_registry: dict[str, object] = {}
    register_calls: list[str] = []

    class _FakeAttentionInterface:
        _global_mapping = attn_registry

        @classmethod
        def register(cls, key: str, value) -> None:
            register_calls.append(key)
            cls._global_mapping[key] = value

    class _FakeEncConfig:
        model_type = "modernbert"
        hidden_size = 4
        local_attention = 128

    class _FakeRotaryEmb(_Module):
        def __init__(self) -> None:
            super().__init__()
            # Realistic ModernBERT global RoPE frequencies (theta=160000.0): tiny values drift/underflow in FP16!
            global_freqs = [1.0 / (160000.0 ** (i / 64.0)) for i in range(0, 64, 2)]
            local_freqs = [1.0 / (10000.0 ** (i / 64.0)) for i in range(0, 64, 2)]
            global_buf = _FakeTensor(global_freqs, dtype=torch_mod.float32, device="cpu")
            self.register_buffer("inv_freq", global_buf)
            # Aliased buffer pointing to the same tensor object (tests remove_duplicate=False)
            self.register_buffer("original_inv_freq", global_buf)
            self.register_buffer("local_inv_freq", _FakeTensor(local_freqs, dtype=torch_mod.float32, device="cpu"))

    class _FakeEncoder(_Module):
        def __init__(self, config, attn_implementation: str) -> None:
            super().__init__()
            self.config = config
            self.config._attn_implementation = attn_implementation
            self.calls: list[dict[str, object]] = []
            self.rotary_emb = _FakeRotaryEmb()
            self.register_buffer("aux_scale", _FakeTensor([1.25, 2.5], dtype=torch_mod.float32, device="cpu"))

        def forward(self, input_ids, attention_mask=None):
            self.calls.append(
                {
                    "shape": input_ids.shape,
                    "attn_impl": self.config._attn_implementation,
                    "is_mask_dict": isinstance(attention_mask, dict),
                    "mask_keys": sorted(attention_mask.keys()) if isinstance(attention_mask, dict) else None,
                    "sliding_window": attention_mask["sliding_attention"].window if isinstance(attention_mask, dict) else None,
                }
            )
            # Encode each token position deterministically: token_id * 0.01
            b, l = input_ids.shape
            hidden = self.config.hidden_size
            data = [[[float(tok) * 0.01 for _ in range(hidden)] for tok in row] for row in input_ids._data]
            return types.SimpleNamespace(last_hidden_state=_FakeTensor(data, dtype=torch_mod.float32, device=input_ids.device))

    class _FakeAutoConfig:
        calls: list[tuple[str, bool]] = []

        @classmethod
        def from_pretrained(cls, path, local_files_only: bool = False):
            cls.calls.append((str(path), local_files_only))
            if not local_files_only:
                raise AssertionError("Expected local_files_only=True")
            return _FakeEncConfig()

    class _FakeAutoTokenizer:
        calls: list[tuple[str, bool]] = []

        def __init__(self) -> None:
            self.cls_token_id = 101
            self.sep_token_id = 102
            self.mask_token_id = 103
            self.pad_token_id = 0

        @classmethod
        def from_pretrained(cls, path, local_files_only: bool = False):
            cls.calls.append((str(path), local_files_only))
            if not local_files_only:
                raise AssertionError("Expected local_files_only=True")
            return cls()

        def __call__(self, texts: list[str], add_special_tokens: bool = False):
            assert add_special_tokens is False
            # Map each whitespace-separated word to an integer token ID (or repeat tokens for long-input tests)
            out = []
            for text in texts:
                if not text:
                    out.append([])
                else:
                    out.append([1000 + (sum(ord(ch) for ch in tok) % 500) for tok in text.split()])
            return {"input_ids": out}

    class _FakeModelCls:
        _supports_sdpa = True

    class _FakeAutoModel:
        _model_mapping = {_FakeEncConfig: _FakeModelCls}

        @classmethod
        def from_config(cls, cfg, attn_implementation: str = "eager"):
            return _FakeEncoder(cfg, attn_implementation=attn_implementation)

    transformers_mod.AttentionInterface = _FakeAttentionInterface
    transformers_mod.AutoConfig = _FakeAutoConfig
    transformers_mod.AutoTokenizer = _FakeAutoTokenizer
    transformers_mod.AutoModel = _FakeAutoModel

    # Fake safetensors.torch
    safetensors_mod = types.ModuleType("safetensors")
    safetensors_torch_mod = types.ModuleType("safetensors.torch")
    load_file_calls: list[tuple[str, str]] = []

    def _fake_load_file(path: str, device: str = "cpu"):
        load_file_calls.append((path, device))
        return {"encoder.weight": [1.0], "head.0.weight": [1.0]}

    safetensors_torch_mod.load_file = _fake_load_file
    safetensors_mod.torch = safetensors_torch_mod

    modules = {
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torch.nn.functional": nn_func_mod,
        "transformers": transformers_mod,
        "safetensors": safetensors_mod,
        "safetensors.torch": safetensors_torch_mod,
    }
    state = {
        "cuda_state": cuda_state,
        "sdpa_calls": sdpa_calls,
        "register_calls": register_calls,
        "attn_registry": attn_registry,
        "load_file_calls": load_file_calls,
        "AutoConfig": _FakeAutoConfig,
        "AutoTokenizer": _FakeAutoTokenizer,
        "AutoModel": _FakeAutoModel,
        "FakeEncConfig": _FakeEncConfig,
        "FakeModelCls": _FakeModelCls,
        "FakeTensor": _FakeTensor,
    }
    return modules, state


_SAVED_MODULES: dict[str, object] = {}
_FAKE_STATE: dict[str, object] = {}
_MODEL_MOD = None


def setUpModule() -> None:
    global _SAVED_MODULES, _FAKE_STATE, _MODEL_MOD
    fake_modules, fake_state = _make_fake_ml_modules()
    _FAKE_STATE = fake_state
    for name in [*fake_modules.keys(), "app.model"]:
        if name in sys.modules:
            _SAVED_MODULES[name] = sys.modules[name]
    sys.modules.update(fake_modules)
    if "app.model" in sys.modules:
        _MODEL_MOD = importlib.reload(sys.modules["app.model"])
    else:
        _MODEL_MOD = importlib.import_module("app.model")


def tearDownModule() -> None:
    for name in ("app.model", "torch", "torch.nn", "torch.nn.functional", "transformers", "safetensors", "safetensors.torch"):
        sys.modules.pop(name, None)
    sys.modules.update(_SAVED_MODULES)


def _create_mock_model_dir(tmpdir: str, *, layout: str = "uni", hidden: int = 4, width: int = 4) -> Path:
    root = Path(tmpdir)
    (root / "encoder").mkdir(parents=True, exist_ok=True)
    (root / "uni_config.json").write_text(
        json.dumps({"hidden": hidden, "width": width, "dropout": 0.1, "layout": layout}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"fake-safetensors")
    return root


class AmberModelUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        _FAKE_STATE["cuda_state"]["available"] = False  # type: ignore[index]
        _FAKE_STATE["sdpa_calls"].clear()  # type: ignore[attr-defined]
        _FAKE_STATE["load_file_calls"].clear()  # type: ignore[attr-defined]
        _FAKE_STATE["FakeModelCls"]._supports_sdpa = True  # type: ignore[attr-defined]

    def test_resolve_dtype_rules(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        self.assertEqual(str(m.resolve_dtype("cpu", None)), "torch.float32")
        self.assertEqual(str(m.resolve_dtype("cuda", None)), "torch.float16")
        self.assertEqual(str(m.resolve_dtype("cuda", "fp16")), "torch.float16")
        self.assertEqual(str(m.resolve_dtype("cuda", "fp32")), "torch.float32")
        with self.assertRaises(ValueError):
            m.resolve_dtype("cpu", "fp16")
        with self.assertRaises(ValueError):
            m.resolve_dtype("cuda", "bf16")

    def test_load_model_cpu_first_offline_and_no_duplicate_registration(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _create_mock_model_dir(tmpdir)
            bundle1 = m.load_model(str(model_dir), "cpu", "fp32")
            bundle2 = m.load_model(str(model_dir), "cpu", "fp32")

            self.assertEqual(bundle1.layout, "uni")
            self.assertEqual(bundle1.attn_impl, "sdpa")
            self.assertTrue(bundle1.banded)
            self.assertEqual(bundle1.sliding_window, 64)
            self.assertEqual(_FAKE_STATE["load_file_calls"], [  # type: ignore[index]
                (str(model_dir / "model.safetensors"), "cpu"),
                (str(model_dir / "model.safetensors"), "cpu"),
            ])
            # Registered once despite two load_model() calls
            self.assertEqual(_FAKE_STATE["register_calls"], ["amber_banded"])  # type: ignore[index]
            self.assertIn("amber_banded", _FAKE_STATE["attn_registry"])  # type: ignore[operator]
            self.assertEqual(bundle2.model._loaded_state[1], True)

            # CUDA unavailable check
            with self.assertRaises(RuntimeError):
                m.load_model(str(model_dir), "cuda", "fp16")

            # Fallback to eager when _supports_sdpa is False
            _FAKE_STATE["FakeModelCls"]._supports_sdpa = False  # type: ignore[attr-defined]
            eager_bundle = m.load_model(str(model_dir), "cpu", "fp32")
            self.assertEqual(eager_bundle.attn_impl, "eager")
            self.assertFalse(eager_bundle.banded)

    def test_uni_sequence_assembly_option_rendering_and_max_tokens_boundary(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _create_mock_model_dir(tmpdir)
            bundle = m.load_model(str(model_dir), "cpu", "fp32")

            # Option rendering & default prompt
            seen_texts: list[list[str]] = []
            orig_tok = bundle.tokenizer

            def _capturing_tok(texts: list[str], add_special_tokens: bool = False):
                seen_texts.append(list(texts))
                return orig_tok(texts, add_special_tokens=add_special_tokens)

            bundle = m.AmberBundle(
                model=bundle.model,
                tokenizer=_capturing_tok,
                special=bundle.special,
                device=bundle.device,
                dtype=bundle.dtype,
                layout=bundle.layout,
                attn_impl=bundle.attn_impl,
                banded=bundle.banded,
                sliding_window=bundle.sliding_window,
            )

            choice, probs = m.classify_text(
                bundle,
                text="hello world",
                labels={"a": "  alpha description  ", "b": None, "c": "   "},
                task="triage",
                prompt=None,
            )
            self.assertEqual(
                seen_texts[0],
                [
                    "Classify the text for task 'triage'.",
                    "hello world",
                    "alpha description",
                    "b",
                    "c",
                ],
            )
            self.assertIn(choice, {"a", "b", "c"})
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
            self.assertEqual(list(probs.keys()), ["a", "b", "c"])

            # Verify exact [CLS] instructions [SEP] state [SEP] [MASK] opt1 [MASK] opt2 [SEP] layout
            _, seq = m._prepare_item(bundle, "state tok", ["optA", "optB"], "triage", "Choose one")
            # Choose(1)+one(1)=2, state(1)+tok(1)=2, optA(1), optB(1) -> 4 + 2 + 2 + (1+1) + (1+1) = 12
            self.assertEqual(len(seq.ids), 12)
            self.assertEqual(seq.ids[0], 101)  # CLS
            self.assertEqual(seq.ids[3], 102)  # SEP after instructions
            self.assertEqual(seq.ids[6], 102)  # SEP after state
            self.assertEqual(seq.ids[7], 103)  # MASK before optA
            self.assertEqual(seq.ids[9], 103)  # MASK before optB
            self.assertEqual(seq.ids[11], 102)  # trailing SEP
            self.assertEqual(seq.markers, [7, 9])
            self.assertEqual(seq.spans, [(8, 9), (10, 11)])

            # Exactly MAX_TOKENS (7999) succeeds; 8000 raises InputTooLongError without truncation
            # Base overhead for 2 1-token labels + 1-token prompt ("p"): 4 + 1 + 2 + 2 = 9 tokens
            exact_text = " ".join(["w"] * (m.MAX_TOKENS - 9))
            ok_choice, _ = m.classify_text(bundle, text=exact_text, labels=["x", "y"], prompt="p")
            self.assertIn(ok_choice, {"x", "y"})

            too_long_text = " ".join(["w"] * (m.MAX_TOKENS - 8))
            with self.assertRaises(m.InputTooLongError) as cm:
                m.classify_text(bundle, text=too_long_text, labels=["x", "y"], prompt="p")
            self.assertIn("7,999", str(cm.exception))

    def test_batch_splitting_long_banded_routing_error_isolation_and_restoration(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _create_mock_model_dir(tmpdir)
            bundle = m.load_model(str(model_dir), "cpu", "fp32")
            encoder = bundle.model.encoder

            # Mix: two short items (<= 2048), one oversized (> 7999), one long (2100 > 2048)
            short_a = " ".join(["a"] * 40)
            short_b = " ".join(["b"] * 60)
            long_item = " ".join(["c"] * 2100)
            over_item = " ".join(["d"] * 8050)

            res = m.classify_batch(
                bundle,
                [
                    (long_item, ["l1", "l2"], "decision", "p"),
                    (over_item, ["l1", "l2"], "decision", "p"),
                    (short_b, ["l1", "l2"], "decision", "p"),
                    (short_a, ["l1", "l2"], "decision", "p"),
                ],
                token_budget=512,
            )
            self.assertEqual(len(res), 4)
            self.assertEqual(res[0][0], "ok")
            self.assertEqual(len(res[0][1]), 3)
            self.assertEqual(res[0][1][2], 2109)  # 4 + 1 (prompt) + 2100 (text) + 2*(1+1) (labels)
            self.assertEqual(res[1][0], "input_too_long")
            self.assertEqual(res[2][0], "ok")
            self.assertEqual(res[2][1][2], 69)  # 4 + 1 + 60 + 4
            self.assertEqual(res[3][0], "ok")
            self.assertEqual(res[3][1][2], 49)  # 4 + 1 + 40 + 4

            # Verify encoder calls:
            # Call 0: padded micro-batch of the 2 short items (B=2, standard sdpa, mask_keys=None)
            # Call 1: solo unpadded long item (B=1, amber_banded, mask dict with sliding_window=64)
            self.assertEqual(len(encoder.calls), 2)
            self.assertEqual(encoder.calls[0]["shape"][0], 2)
            self.assertEqual(encoder.calls[0]["attn_impl"], "sdpa")
            self.assertFalse(encoder.calls[0]["is_mask_dict"])

            self.assertEqual(encoder.calls[1]["shape"][0], 1)
            self.assertGreater(encoder.calls[1]["shape"][1], m.LONG)
            self.assertEqual(encoder.calls[1]["attn_impl"], "amber_banded")
            self.assertTrue(encoder.calls[1]["is_mask_dict"])
            self.assertEqual(encoder.calls[1]["mask_keys"], ["full_attention", "sliding_attention"])
            self.assertEqual(encoder.calls[1]["sliding_window"], 64)

            # _attn_implementation must be restored after banded pass
            self.assertEqual(encoder.config._attn_implementation, "sdpa")

            # Verify _attn_implementation restoration even when logits() raises!
            orig_logits = bundle.model.logits

            def _failing_logits(batch, masks=None):
                raise RuntimeError("simulated CUDA failure")

            bundle.model.logits = _failing_logits  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                m.classify_batch(bundle, [(long_item, ["l1", "l2"], "decision", "p")])
            self.assertEqual(encoder.config._attn_implementation, "sdpa")
            bundle.model.logits = orig_logits  # type: ignore[method-assign]

            # Verify NaN/Inf logits are rejected
            def _nan_logits(batch, masks=None):
                return _FAKE_STATE["FakeTensor"]([[float("nan"), 0.0]])  # type: ignore[operator]

            bundle.model.logits = _nan_logits  # type: ignore[method-assign]
            nan_res = m.classify_batch(bundle, [(short_a, ["l1", "l2"], "decision", "p")])
            self.assertEqual(nan_res[0][0], "error")
            with self.assertRaises(RuntimeError):
                m.classify_text(bundle, short_a, ["l1", "l2"])

    def test_banded_attention_blocking_and_batch_guard(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        T = _FAKE_STATE["FakeTensor"]  # type: ignore[assignment]
        q = T([[[[0.1, 0.2] for _ in range(2500)]]])  # (1, 1, 2500, 2)
        k = T([[[[0.3, 0.4] for _ in range(2500)]]])
        v = T([[[[0.5, 0.6] for _ in range(2500)]]])

        out, weights = m.banded_attention(None, q, k, v, m.Band(64), scaling=0.5)
        self.assertIsNone(weights)
        self.assertEqual(out.shape, (1, 2500, 1, 2))
        # 2500 tokens with BLOCK=1024 -> 3 block SDPA calls: [0:1024], [1024:2048], [2048:2500]
        self.assertEqual(len(_FAKE_STATE["sdpa_calls"]), 3)  # type: ignore[arg-type]
        self.assertEqual(_FAKE_STATE["sdpa_calls"][0]["q_shape"], (1, 1, 1024, 2))  # type: ignore[index]
        self.assertEqual(_FAKE_STATE["sdpa_calls"][0]["k_shape"], (1, 1, 1088, 2))  # type: ignore[index]
        self.assertTrue(_FAKE_STATE["sdpa_calls"][0]["has_mask"])  # type: ignore[index]

        # Global attention layer (mask=None) uses single plain SDPA call
        _FAKE_STATE["sdpa_calls"].clear()  # type: ignore[attr-defined]
        out_global, _ = m.banded_attention(None, q, k, v, None, scaling=0.5)
        self.assertEqual(out_global.shape, (1, 2500, 1, 2))
        self.assertEqual(len(_FAKE_STATE["sdpa_calls"]), 1)  # type: ignore[arg-type]
        self.assertFalse(_FAKE_STATE["sdpa_calls"][0]["has_mask"])  # type: ignore[index]

        # Band mask rejects batch size > 1
        q_b2 = T([[[[0.1, 0.2] for _ in range(10)]], [[[0.1, 0.2] for _ in range(10)]]])
        with self.assertRaises(ValueError):
            m.banded_attention(None, q_b2, q_b2, q_b2, m.Band(4))

    def test_mixed_option_counts_sdpa_attn_budget_and_edge_validations(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _create_mock_model_dir(tmpdir)
            # Whitespace around device is stripped cleanly
            bundle = m.load_model(str(model_dir), " cpu ", "fp32")
            encoder = bundle.model.encoder

            # Mixed option counts (2 options vs 4 options) inside one micro-batch + 0-token option span
            res_mixed = m.classify_batch(
                bundle,
                [
                    ("alpha beta", ["x", "y"], "decision", "p"),
                    ("gamma delta", ["l1", "l2", "l3", "l4"], "decision", "p"),
                    ("zero opt", {"": "   ", "b": "nonempty"} if False else ["   ", "b"], "decision", "p"),
                ],
                token_budget=8192,
            )
            self.assertEqual(res_mixed[0][0], "ok")
            self.assertEqual(len(res_mixed[0][1][1]), 2)
            self.assertAlmostEqual(sum(res_mixed[0][1][1].values()), 1.0, places=6)
            self.assertEqual(res_mixed[1][0], "ok")
            self.assertEqual(len(res_mixed[1][1][1]), 4)
            self.assertAlmostEqual(sum(res_mixed[1][1][1].values()), 1.0, places=6)
            self.assertEqual(res_mixed[2][0], "error")  # blank label name rejected cleanly

            # 0-token option span directly in _prepare_item / UniModel.logits (start == end)
            orig_tok = bundle.tokenizer
            empty_opt_bundle = m.AmberBundle(
                model=bundle.model,
                tokenizer=lambda texts, add_special_tokens=False: {
                    "input_ids": [[10, 11], [12, 13], [], [14]]
                },
                special=bundle.special,
                device=bundle.device,
                dtype=bundle.dtype,
                layout=bundle.layout,
                attn_impl=bundle.attn_impl,
                banded=bundle.banded,
                sliding_window=bundle.sliding_window,
            )
            res_zero_span = m.classify_batch(empty_opt_bundle, [("ok", ["a", "b"], "decision", "p")])
            self.assertEqual(res_zero_span[0][0], "ok")
            self.assertAlmostEqual(sum(res_zero_span[0][1][1].values()), 1.0, places=6)
            del orig_tok

            # Quadratic SDPA attention budget splits two L=1800 sequences (2 * 1800^2 > 4M) into 2 passes
            encoder.calls.clear()
            seq_1800 = " ".join(["w"] * 1791)
            res_quad = m.classify_batch(
                bundle,
                [
                    (seq_1800, ["a", "b"], "decision", "p"),
                    (seq_1800, ["a", "b"], "decision", "p"),
                ],
                token_budget=8192,
            )
            self.assertEqual([r[0] for r in res_quad], ["ok", "ok"])
            self.assertEqual(len(encoder.calls), 2)
            self.assertEqual(encoder.calls[0]["shape"][0], 1)
            self.assertEqual(encoder.calls[1]["shape"][0], 1)

            # +inf and all -inf logits are rejected as invalid scores
            orig_logits = bundle.model.logits
            bundle.model.logits = lambda batch, masks=None: _FAKE_STATE["FakeTensor"]([[float("inf"), 0.0]])  # type: ignore[method-assign,operator]
            self.assertEqual(m.classify_batch(bundle, [("hi", ["a", "b"], "decision", "p")])[0][0], "error")
            bundle.model.logits = lambda batch, masks=None: _FAKE_STATE["FakeTensor"]([[float("-inf"), float("-inf")]])  # type: ignore[method-assign,operator]
            self.assertEqual(m.classify_batch(bundle, [("hi", ["a", "b"], "decision", "p")])[0][0], "error")
            bundle.model.logits = orig_logits  # type: ignore[method-assign]

            # Eager validation of width <= 0 and sliding_window <= 0
            with self.assertRaises(ValueError):
                m.UniModel(encoder, hidden=4, width=0)
            bad_sw_dir = _create_mock_model_dir(tmpdir + "_badsw")
            orig_local = _FAKE_STATE["FakeEncConfig"].local_attention  # type: ignore[attr-defined]
            _FAKE_STATE["FakeEncConfig"].local_attention = 1  # type: ignore[attr-defined]
            try:
                with self.assertRaises(ValueError):
                    m.load_model(str(bad_sw_dir), "cpu", "fp32")
            finally:
                _FAKE_STATE["FakeEncConfig"].local_attention = orig_local  # type: ignore[attr-defined]

    def test_fp16_load_preserves_modernbert_rope_inv_freq_buffers_in_fp32(self) -> None:
        m = _MODEL_MOD
        assert m is not None
        _FAKE_STATE["cuda_state"]["available"] = True  # type: ignore[index]
        expected_global = [1.0 / (160000.0 ** (i / 64.0)) for i in range(0, 64, 2)]
        expected_local = [1.0 / (10000.0 ** (i / 64.0)) for i in range(0, 64, 2)]

        # Confirm FP16 would actually corrupt/underflow expected_global so this test is meaningful!
        fp16_corrupted = [struct.unpack("e", struct.pack("e", v))[0] for v in expected_global]
        self.assertNotEqual(fp16_corrupted[-1], expected_global[-1])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _create_mock_model_dir(tmpdir)
            bundle = m.load_model(str(model_dir), "cuda", "fp16")

            # Model weights & non-inv_freq buffers are cast to FP16 on CUDA
            self.assertEqual(str(bundle.dtype), "torch.float16")
            self.assertEqual(str(bundle.model.head[0].weight.dtype), "torch.float16")
            self.assertEqual(str(bundle.model.encoder.aux_scale.dtype), "torch.float16")
            self.assertEqual(str(bundle.model.encoder.aux_scale.device), "cuda")

            # RoPE *_inv_freq buffers (including aliased buffers!) MUST remain exact original FP32 on target_device!
            rotary = bundle.model.encoder.rotary_emb
            self.assertEqual(str(rotary.inv_freq.dtype), "torch.float32")
            self.assertEqual(str(rotary.inv_freq.device), "cuda")
            self.assertEqual(rotary.inv_freq.tolist(), expected_global)

            self.assertEqual(str(rotary.original_inv_freq.dtype), "torch.float32")
            self.assertEqual(str(rotary.original_inv_freq.device), "cuda")
            self.assertEqual(rotary.original_inv_freq.tolist(), expected_global)

            self.assertEqual(str(rotary.local_inv_freq.dtype), "torch.float32")
            self.assertEqual(str(rotary.local_inv_freq.device), "cuda")
            self.assertEqual(rotary.local_inv_freq.tolist(), expected_local)


if __name__ == "__main__":
    unittest.main()
