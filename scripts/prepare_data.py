#!/usr/bin/env python3
"""Validate JSONL annotations and normalize paths for the read-only COCO mount."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

IMAGE_PREFIX = "/workspace/datasets/RefCOCO/train2014"
MAX_UNCOMPRESSED_BYTES = 4 * 1024 ** 3
ARCHIVE_ANNOTATIONS = frozenset({
    "refcoco_train.jsonl", "refcoco_80k_train.jsonl",
    "refcoco_testA_eval.jsonl", "refcoco_testB_eval.jsonl",
    "refcocop_testA_eval.jsonl", "refcocop_testB_eval.jsonl", "refcocog_test_eval.jsonl",
})


def public_filename(name):
    """Accept existing annotation exports while using the public filename convention."""
    if PurePosixPath(name).name != name or not name.endswith(".jsonl") or name.startswith("."):
        raise ValueError("Input annotation names must be ordinary JSONL filenames")
    return name.replace("_iousd", "")


def normalize_row(row, image_root=None):
    if not isinstance(row, dict) or not isinstance(row.get("images"), list) or len(row["images"]) != 1:
        raise ValueError("Each record must contain exactly one image")
    image = row["images"][0]
    if not isinstance(image, str) or not image or "\\" in image:
        raise ValueError("Invalid image path")
    parts = PurePosixPath(image).parts
    if ".." in parts or not parts:
        raise ValueError("Image paths must not contain traversal components")
    name = parts[-1]
    if PurePosixPath(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Expected a JPEG or PNG image filename")
    if len(parts) > 1 and parts[-2] != "train2014":
        raise ValueError("Nested image paths must identify the COCO train2014 directory")
    if image_root is not None and not (image_root / name).is_file():
        raise FileNotFoundError(f"Image missing from the configured host directory: {name}")
    metadata = row.get("additional_paras") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict) or metadata.get("bbox_type", "norm1000") != "norm1000":
        raise ValueError("Annotations must use norm1000 xyxy boxes")
    coordinates = row.get("solution", {}).get("arguments", {}).get("coordinate")
    if (not isinstance(coordinates, list) or len(coordinates) != 4 or
        any(isinstance(x, bool) or not isinstance(x, (int, float)) or
            not math.isfinite(x) or x < 0 or x > 1000 for x in coordinates)):
        raise ValueError("Each target must contain four finite coordinates in [0, 1000]")
    if coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
        raise ValueError("Expected xyxy coordinate order")
    expression = metadata.get("caption")
    if not expression:
        messages = row.get("messages", [])
        if not any(isinstance(message, dict) and message.get("role") == "user" and
                   message.get("content") for message in messages):
            raise ValueError("Each sample requires a caption or a user message")
    result = dict(row)
    result["images"] = [f"{IMAGE_PREFIX}/{name}"]
    return result


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def convert(stream, destination, image_root=None):
    count = 0
    with destination.open("w", encoding="utf-8") as target:
        for count, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line at {destination.name}:{count}")
            try:
                row = normalize_row(json.loads(line), image_root)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Invalid annotation {destination.name}:{count}: {error}") from error
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if not count:
        raise ValueError(f"Empty annotation file: {destination.name}")
    return {"rows": count, "sha256": digest(destination)}


def safe_archive_members(archive):
    members = []
    total = 0
    names = set()
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename or
            (path.parts and ":" in path.parts[0])):
            raise ValueError("Archive contains an unsafe path")
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("Archive must not contain symlinks")
        if entry.is_dir() or path.suffix != ".jsonl":
            continue
        if "__MACOSX" in path.parts or path.name.startswith("."):
            continue
        name = public_filename(path.name)
        if name not in ARCHIVE_ANNOTATIONS:
            continue
        if name in names:
            raise ValueError(f"Archive contains duplicate normalized annotation filenames: {name}")
        names.add(name)
        total += entry.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Annotation archive exceeds the 4 GiB uncompressed limit")
        members.append(entry)
    if not members:
        raise ValueError("No supported RefCOCO annotations found in the archive")
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--jsonl", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("data/refcoco"))
    parser.add_argument("--image-root", type=Path, help="Optional existing host image directory; check every image")
    args = parser.parse_args()
    if args.image_root is not None and not args.image_root.is_dir():
        parser.error("--image-root must identify an existing directory")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jag-data-", dir=output.parent) as temporary:
        stage = Path(temporary)
        records = {}
        if args.archive:
            with zipfile.ZipFile(args.archive) as archive:
                for entry in safe_archive_members(archive):
                    name = public_filename(PurePosixPath(entry.filename).name)
                    with archive.open(entry) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
                        records[name] = convert(stream, stage / name, args.image_root)
        else:
            for source_path in args.jsonl:
                name = public_filename(source_path.name)
                if name in records:
                    parser.error("Input JSONL files must have unique filenames after normalization")
                with source_path.open(encoding="utf-8-sig") as stream:
                    records[name] = convert(stream, stage / name, args.image_root)
        manifest = {"schema_version": 1, "image_prefix": IMAGE_PREFIX,
                    "image_existence_checked": args.image_root is not None, "files": records}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.mkdir(exist_ok=True)
        if any((output / path.name).exists() for path in stage.iterdir()):
            raise FileExistsError("Output already contains prepared files; select a new output directory")
        for path in stage.iterdir():
            shutil.move(str(path), output / path.name)
    print(json.dumps({"status": "prepared", "output": str(output), "files": records}, indent=2))


if __name__ == "__main__":
    main()
