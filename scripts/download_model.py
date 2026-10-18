"""Include a fixed, complete GLiNER2.5-Decide model snapshot in the image at build time."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fastino/GLiNER2.5-Decide",
    revision="9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
    local_dir="/opt/models/gliner2.5-decide",
    allow_patterns=[
        "config.json",
        "encoder_config/config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
    ],
)
