#!/usr/bin/env python3
"""Evaluate the five supplied RefCOCO-family test splits sequentially."""

import argparse
from pathlib import Path
import subprocess
import sys

SPLITS = [
    ("refcoco", "testA"), ("refcoco", "testB"),
    ("refcocop", "testA"), ("refcocop", "testB"), ("refcocog", "test"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["jag", "base"])
    parser.add_argument("--checkpoint", default="/models/Jag")
    parser.add_argument("--base-model", default="/models/Qwen3.5-0.8B")
    parser.add_argument("--annotations", default="/workspace/datasets/RefCOCO/annotations")
    parser.add_argument("--output", default="/outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Quality-inference request batch; CUDA OOM retries use smaller subbatches")
    parser.add_argument("--gpu-memory-gib", type=float, default=28)
    args, extras = parser.parse_known_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for dataset, subset in SPLITS:
        path = Path(args.annotations) / f"{dataset}_{subset}_eval.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation split: {path}")
    for dataset, subset in SPLITS:
        model_kind = "base" if args.model == "base" else "jag"
        output_name = "base" if args.model == "base" else "jag"
        checkpoint = args.base_model if args.model == "base" else args.checkpoint
        subprocess.run([
            sys.executable, "-m", "jag.evaluate_batched",
            "--model-kind", model_kind, "--checkpoint", checkpoint,
            "--jsonl", str(Path(args.annotations) / f"{dataset}_{subset}_eval.jsonl"),
            "--subset", subset, "--output", str(Path(args.output) / output_name / f"{dataset}-{subset}"),
            "--device", "cuda:0", "--gpu-memory-gib", str(args.gpu_memory_gib),
            "--batch-size", str(args.batch_size), *extras,
        ], check=True)


if __name__ == "__main__":
    main()
