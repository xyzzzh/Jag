"""Resolve a saved Jag model from a local directory or the Hugging Face Hub."""

from pathlib import Path


def resolve_model_directory(model):
    name = str(model)
    path = Path(name).expanduser()
    if path.is_dir():
        return path
    if path.exists() or path.is_absolute() or name.startswith(("./", "../", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=name,
        allow_patterns=[
            "config.json", "model*.safetensors", "model*.safetensors.index.json",
            "processor_config.json", "preprocessor_config.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "chat_template*", "vocab.json", "merges.txt",
        ],
    ))
