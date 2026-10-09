#!/usr/bin/env python3
"""Export inference assets without optimizer state, credentials or training logs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ASSETS = {
    "config.json", "generation_config.json", "processor_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "chat_template.json", "model.safetensors.index.json",
}


def sanitize_config(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"_name_or_path", "name_or_path", "_commit_hash"}:
                continue
            if key == "base_model_path":
                item = "/models/Qwen3.5-0.8B"
            cleaned[key] = sanitize_config(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_config(item) for item in value]
    return value


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inference_config(config):
    """Export the direct last-token model without changing its tensor weights."""
    config = dict(config)
    if "prediction_mode" in config:
        if (config.get("prediction_mode") != "continuous"
                or config.get("readout") != "last_valid"
                or config.get("freeze_backbone") is not False
                or config.get("backbone_config", {}).get("model_type") != "qwen3_5"):
            raise ValueError("Only the continuous last-token model supports the Jag inference export")
        config["model_type"] = "jag"
        config["architectures"] = ["JagModel"]
        for key in ("prediction_mode", "readout", "readout_description", "freeze_backbone",
                    "num_coordinate_bins", "reg_token", "reg_token_id"):
            config.pop(key, None)
        config["keys_to_ignore_at_inference"] = ["loss_l1", "loss_giou"]
    if config.get("model_type") != "jag":
        raise ValueError("Source is not a compatible Jag model")
    return sanitize_config(config)


def export_model(checkpoint, output, overwrite=False):
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    if output == checkpoint or output in checkpoint.parents or checkpoint in output.parents:
        raise ValueError("Export destination must be separate from the source checkpoint")
    config = inference_config(json.loads((checkpoint / "config.json").read_text()))
    if config.get("stage") != "joint":
        raise ValueError("Export requires a checkpoint from the joint training stage")
    files = sorted(path for path in checkpoint.iterdir() if
                   path.name in ASSETS or path.name.endswith(".safetensors"))
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Model assets must be regular files, not symlinks")
    names = {path.name for path in files}
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("No safetensors weights found")
    if "tokenizer_config.json" not in names or "tokenizer.json" not in names:
        raise ValueError("The checkpoint must include its tokenizer")
    if not {"processor_config.json", "preprocessor_config.json"} & names:
        raise ValueError("The checkpoint must include its image processor")
    if "model.safetensors.index.json" in names:
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        if any(Path(name).name != name or name not in names for name in shards):
            raise ValueError("The checkpoint references missing or unsafe weight shards")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; choose a new path or pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-export-", dir=output.parent))
    backup = None
    try:
        for source in files:
            destination = temporary / source.name
            if source.name == "config.json":
                destination.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
            elif source.name.endswith("config.json"):
                value = sanitize_config(json.loads(source.read_text()))
                destination.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            else:
                shutil.copyfile(source, destination)
        manifest = {
            "schema_version": 1, "model": "Jag",
            "base_model": "Qwen/Qwen3.5-0.8B", "format": "safetensors",
            "files": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                      for path in sorted(temporary.iterdir())},
        }
        (temporary / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-previous-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        temporary.rename(output)
        if backup is not None:
            shutil.rmtree(backup)
        print(json.dumps({"status": "exported", "model": "Jag", "output": str(output),
                          "asset_count": len(manifest["files"])}))
        return manifest
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/models/Jag")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_model(args.checkpoint, args.output, args.overwrite)


if __name__ == "__main__":
    main()
