#!/usr/bin/env python3
"""Recreate the published inference workload from benchmark row indices."""

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


def prepare(selection, annotations, output):
    manifest = json.loads(Path(selection).read_text())
    groups = {"cost.jsonl": [], "warmup.jsonl": []}
    for split in manifest["datasets"]:
        filename = f"{split['dataset']}_{split['split']}_eval.jsonl"
        source = Path(annotations) / filename
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != split["sha256"]:
            raise ValueError(f"Annotation identity differs from the published workload: {filename}")
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
    parser.add_argument("--selection", default="data/cost-selection.json")
    parser.add_argument("--annotations", default="/workspace/datasets/RefCOCO/annotations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.annotations, args.output), indent=2))


if __name__ == "__main__":
    main()
