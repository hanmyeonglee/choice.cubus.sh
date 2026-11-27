"""Include fixed, complete Amber 150M and BGE-M3 zero-shot model snapshots in the image at build time."""

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

snapshot_download(
    repo_id="MoritzLaurer/bge-m3-zeroshot-v2.0",
    revision="9abf1c8aaeb82a2447809c20753ed0b106b76652",
    local_dir="/opt/models/bge-m3-zeroshot-v2.0",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "README.md",
    ],
)
