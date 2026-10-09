"""Evaluate the untouched Qwen3.5 multimodal model through EvalScope."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .base_predict import (BaseGroundingPredictor, PARSER_VERSION, PROMPT_TEMPLATE,
                           generation_settings, prompt_sha256)
from .evaluation import JagAPI, checkpoint_manifest, file_sha256, run_evaluation


def base_checkpoint_manifest(checkpoint, jsonl, max_pixels=262144, max_length=2048,
                             max_new_tokens=128, gpu_memory_gib=None, memory_fraction=None):
    manifest = checkpoint_manifest(checkpoint, jsonl, max_pixels, max_length)
    manifest.pop("fingerprint")
    checkpoint = Path(checkpoint).resolve(strict=True)
    for name in ("generation_config.json", "model.safetensors.index.json", "vocab.json", "merges.txt"):
        if (checkpoint / name).exists():
            manifest["files"][name] = file_sha256(checkpoint / name)
    settings = generation_settings(max_new_tokens)
    manifest.update(
        model_kind="qwen35_original_generation", dtype="bfloat16", attn_implementation="sdpa",
        input_template=PROMPT_TEMPLATE, prompt_sha256=prompt_sha256(),
        coordinate_format="normalized_xyxy_1000_generated", prediction_parser=PARSER_VERSION,
        generation_config=settings,
        generation_config_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        gpu_memory_gib=8.0 if gpu_memory_gib is None and memory_fraction is None else gpu_memory_gib,
        memory_fraction=memory_fraction,
        scoring_protocol="jag_refcoco:original-normalized-xyxy-iou:invalid-zero:clamp01:schema-1")
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local original Qwen3.5-0.8B checkpoint")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--gpu-memory-gib", type=float)
    limit.add_argument("--memory-fraction", type=float)
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = argument_parser().parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    manifest = base_checkpoint_manifest(args.model, args.jsonl, args.max_pixels, args.max_length,
                                       args.max_new_tokens, args.gpu_memory_gib, args.memory_fraction)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(args.output) / f"{args.subset}-{manifest['fingerprint'][:12]}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    predictor = BaseGroundingPredictor.from_checkpoint(
        args.model, device=args.device, max_pixels=args.max_pixels, max_length=args.max_length,
        max_new_tokens=args.max_new_tokens, gpu_memory_gib=args.gpu_memory_gib,
        memory_fraction=args.memory_fraction)
    (output / "resource_limits.json").write_text(json.dumps(predictor.memory_status, indent=2) + "\n")
    (output / "resolved_generation_config.json").write_text(
        json.dumps(predictor.generation_config.to_dict(), indent=2) + "\n")
    api = JagAPI(model_name="qwen35-base-" + manifest["fingerprint"][:12], predictor=predictor)
    run_evaluation(args.jsonl, api, output, subset=args.subset, limit=args.limit)
    stats = {"started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
             "wall_seconds": time.monotonic() - started}
    if args.device.startswith("cuda"):
        import torch
        stats.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    (output / "execution_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"status": "completed", "output": str(output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
