"""Run Jag localization evaluation through EvalScope 1.12."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .evaluation import GroundingJevAPI, checkpoint_manifest, run_evaluation
from .resources import configure_cuda_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True, help="Parent directory for a fresh evaluation run")
    parser.add_argument("--subset", default="test")
    parser.add_argument("--processor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--gpu-memory-gib", type=float)
    memory.add_argument("--memory-fraction", type=float)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    resources = configure_cuda_memory(args.device, args.gpu_memory_gib, args.memory_fraction)
    manifest = checkpoint_manifest(args.checkpoint, args.jsonl, args.max_pixels, args.max_length)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(args.output) / f"{args.subset}-{manifest['fingerprint'][:12]}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "resource_limits.json").write_text(json.dumps(resources, indent=2) + "\n")
    api = GroundingJevAPI(model_name="groundingjev-" + manifest["fingerprint"][:12],
                         checkpoint=args.checkpoint, processor_path=args.processor,
                         device=args.device, max_pixels=args.max_pixels, max_length=args.max_length)
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
