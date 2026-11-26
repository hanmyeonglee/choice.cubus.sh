"""Include a fixed, complete Amber 150M model snapshot in the image at build time."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MLukauskas/amber-150m",
    revision="51ee1bba1f9e71aae4bf65bde1d43216260b6aab",
    local_dir="/opt/models/amber-150m",
    allow_patterns=[
        "uni_config.json",
        "model.safetensors",
        "amber_uni_core.py",
        "amber_uni_engine.py",
        "encoder/config.json",
        "encoder/tokenizer.json",
        "encoder/tokenizer_config.json",
        "README.md",
    ],
)
