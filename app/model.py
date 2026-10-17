"""Load the fixed GLiFormer checkpoint for single-request FP32 inference."""

import torch
from gliformer import GLiFormer


def load_model(path: str, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable GPU access with the NVIDIA Container Toolkit.")
    # Load weights on CPU first to avoid holding two copies on the small GPU.
    model = GLiFormer.from_pretrained(
        path,
        map_location="cpu",
        local_files_only=True,
        load_tokenizer=True,
        dtype="float32",
        _attn_implementation="eager",
    )
    # The checkpoint can retain its own flash setting despite the loader option.
    for module in model.model.modules():
        if hasattr(module, "attn_kernel"):
            module.attn_kernel = "eager"
    return model.to(device).eval()
