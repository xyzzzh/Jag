#!/usr/bin/env python3
"""Validate the pinned runtime without loading model weights."""

import argparse
import importlib
from importlib.metadata import version
import json
import sys

PACKAGES = {
    "torch": "2.10.0", "torchvision": "0.25.0", "transformers": "5.9.0",
    "ms-swift": "4.3.2", "evalscope": "1.12.0", "modelscope": "1.37.0",
    "datasets": "4.8.4", "peft": "0.19.0", "accelerate": "1.12.0",
    "trl": "0.29.1", "qwen-vl-utils": "0.0.14", "swanlab": "0.10.1",
    "fla-core": "0.5.2", "flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The pinned environment requires Python 3.12")
    installed = {name: version(name) for name in PACKAGES}
    for name, expected in PACKAGES.items():
        if installed[name].split("+")[0] != expected:
            raise RuntimeError(f"Unexpected version for {name}: {installed[name]}")
    for name in ("torch", "torchvision", "transformers", "swift", "evalscope", "modelscope", "swanlab", "fla", "causal_conv1d"):
        importlib.import_module(name)
    from transformers import Qwen3_5Model, Qwen3_5ForConditionalGeneration
    from swift.trainers import Trainer
    from evalscope.api.model import ModelAPI
    import torch
    result = {"status": "passed", "versions": installed, "cuda_runtime": torch.version.cuda}
    if not args.imports_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; check the NVIDIA container runtime and GPU selection")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Training requires a GPU with BF16 support")
        result["visible_gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
