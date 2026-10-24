#!/usr/bin/env python3
"""Prepare a reproducible inference workload from test annotations."""

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random


TEST_SPLITS = (("refcoco", "testA"), ("refcoco", "testB"),
               ("refcocop", "testA"), ("refcocop", "testB"),
               ("refcocog", "test"))


def sample_manifest(annotations, samples_per_split=50, warmup_per_split=2, seed=42):
    counts = (samples_per_split, warmup_per_split)
    if any(type(value) is not int or value < 1 for value in counts):
        raise ValueError("Measured and warmup sample counts must be positive integers")
    if type(seed) is not int:
        raise ValueError("The sampling seed must be an integer")
    manifest = {"schema_version": 1,
                "selection": "Uniform random sampling per test split with disjoint warmup rows",
                "seed": seed, "samples_per_split": samples_per_split,
                "warmup_per_split": warmup_per_split, "datasets": []}
    for dataset, split in TEST_SPLITS:
        filename = f"{dataset}_{split}_eval.jsonl"
        contents = (Path(annotations) / filename).read_bytes()
        total = len([line for line in contents.decode().splitlines() if line.strip()])
        requested = samples_per_split + warmup_per_split
        if total < requested:
            raise ValueError(f"Not enough annotation rows in {filename}: need {requested}, found {total}")
        indices = random.Random(seed).sample(range(total), requested)
        manifest["datasets"].append({"dataset": dataset, "split": split,
                                     "sha256": hashlib.sha256(contents).hexdigest(),
                                     "total": total,
                                     "warmup_indices": indices[:warmup_per_split],
                                     "sample_indices": indices[warmup_per_split:]})
    return manifest


def prepare(selection, annotations, output, *, samples_per_split=50, warmup_per_split=2, seed=42):
    manifest = (json.loads(Path(selection).read_text()) if selection is not None else
                sample_manifest(annotations, samples_per_split, warmup_per_split, seed))
    groups = {"cost.jsonl": [], "warmup.jsonl": []}
    for split in manifest["datasets"]:
        filename = f"{split['dataset']}_{split['split']}_eval.jsonl"
        source = Path(annotations) / filename
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != split["sha256"]:
            raise ValueError(f"Annotation identity differs from the recorded selection: {filename}")
        rows = [json.loads(line) for line in contents.decode().splitlines() if line.strip()]
        if len(rows) != split["total"]:
            raise ValueError(f"Annotation sample count differs: {filename}")
        measured, warmup = split["sample_indices"], split["warmup_indices"]
        if set(measured) & set(warmup):
            raise ValueError(f"Warmup overlaps measurement: {filename}")
        for name, indices in (("cost.jsonl", measured), ("warmup.jsonl", warmup)):
            if len(indices) != len(set(indices)):
                raise ValueError(f"Repeated row index in {filename}")
            selected = []
            for index in indices:
                if type(index) is not int or not 0 <= index < len(rows):
                    raise ValueError(f"Invalid row index in {filename}: {index}")
                row = dict(rows[index])
                row["_cost_source"] = {"dataset": split["dataset"], "split": split["split"],
                                       "source_file": filename, "source_row_index": index}
                selected.append(row)
            groups[name].append(selected)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, batches in groups.items():
        rows = [row for group in zip_longest(*batches) for row in group if row is not None]
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (destination / name).write_text(content)
        result[name] = {"samples": len(rows), "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (destination / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", help="Reuse a saved selection manifest instead of sampling")
    parser.add_argument("--samples-per-split", type=int, default=50)
    parser.add_argument("--warmup-per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", default="/workspace/datasets/RefCOCO/annotations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.annotations, args.output,
                             samples_per_split=args.samples_per_split,
                             warmup_per_split=args.warmup_per_split, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
