"""Include a fixed, complete model snapshot in the image at build time."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="knowledgator/gliformer-base-v1",
    revision="590f9d3f577ea2f6d685aaec84b2d66b6db86b15",
    local_dir="/opt/models/gliformer-base-v1",
    allow_patterns=[
        "gliner_config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
    ],
)
