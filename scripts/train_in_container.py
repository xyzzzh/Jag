#!/usr/bin/env python3
"""Train both stages, then export the completed model for inference."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/train/jag.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--stage")
    parser.add_argument("--export-output", default="/models/Jag")
    parser.add_argument("--overwrite-export", action="store_true")
    args, extras = parser.parse_known_args()
    configuration = json.loads(Path(args.config).read_text())["training"]
    stage = args.stage or configuration.get("stage", "all")
    help_requested = "--help" in extras or "-h" in extras
    if stage != "head" and not help_requested and Path(args.export_output).exists() and not args.overwrite_export:
        raise FileExistsError("Export directory already exists; choose --export-output or explicitly use --overwrite-export")
    count = int(os.environ.get("JAG_NPROC_PER_NODE", "1"))
    if count not in {1, 2, 4}:
        raise ValueError("Jag training supports 1, 2 or 4 processes")
    overrides = []
    if args.output_dir:
        overrides += ["--output-dir", args.output_dir]
    if args.stage:
        overrides += ["--stage", args.stage]
    command = ["torchrun", "--standalone", f"--nproc_per_node={count}", "train.py",
               "--config", args.config, *overrides, *extras]
    subprocess.run(command, check=True)
    if help_requested:
        return
    if stage == "head":
        return
    output = Path(args.output_dir or configuration.get("output_dir", "/outputs/jag"))
    record = json.loads((output / "joint/completed.json").read_text())
    from export_model import export_model
    export_model(record["checkpoint"], args.export_output, overwrite=args.overwrite_export)


if __name__ == "__main__":
    main()
