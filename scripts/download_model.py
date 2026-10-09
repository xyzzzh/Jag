#!/usr/bin/env python3
"""Download verified Qwen assets directly from the public ModelScope Hub API.

No ModelScope SDK, credentials, or model code execution is required. The default
model uses the release's committed file manifest. Other explicitly requested
revisions are resolved once and saved before downloading. Existing files are
verified against their content hashes rather than silently following branches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST = Path(__file__).with_name("base_model_manifest.json")
MODEL_ROOT = Path("/models") if Path("/.dockerenv").exists() else PROJECT_ROOT / "models"
OUTPUT_ROOT = Path("/outputs") if Path("/.dockerenv").exists() else PROJECT_ROOT / "outputs"
ENDPOINT = "https://modelscope.cn"
USER_AGENT = "Jag-model-preparation/1.0"
MANIFEST_NAME = ".modelscope-content-manifest.json"
ASSET_NAMES = {
    "LICENSE",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "model.safetensors.index.json",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def request(url: str, headers: dict | None = None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **(headers or {})},
        ),
        timeout=60,
    )


def validate_file(entry: dict) -> None:
    path = entry.get("path", "")
    if (
        not path
        or Path(path).name != path
        or "/" in path
        or "\\" in path
        or path in {".", ".."}
        or not (path in ASSET_NAMES or path.endswith(".safetensors"))
    ):
        raise ValueError(f"Unsupported model asset path: {path!r}")
    if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
        raise ValueError(f"Invalid size for {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
        raise ValueError(f"Missing or invalid SHA256 for {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", entry.get("file_revision", "")):
        raise ValueError(f"Missing or invalid immutable file revision for {path}")


def get_manifest(model_id: str, revision: str, output_dir: Path) -> dict:
    destination = output_dir / MANIFEST_NAME
    if destination.exists():
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        if manifest.get("model_id") != model_id or manifest.get("requested_revision") != revision:
            raise ValueError("Existing content manifest identifies a different model/revision")
        if manifest.get("schema_version") != 1 or manifest.get("source") != ENDPOINT:
            raise ValueError("Unsupported existing content manifest")
    elif model_id == "Qwen/Qwen3.5-0.8B" and revision == "master":
        # The public release records the exact upstream files used by this recipe.
        manifest = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
    else:
        url = (
            f"{ENDPOINT}/api/v1/models/{model_id}/repo/files?"
            + urllib.parse.urlencode({"Revision": revision, "Recursive": "true"})
        )
        with request(url) as response:
            payload = json.load(response)
        if payload.get("Code") != 200 or not payload.get("Success"):
            raise RuntimeError(f"ModelScope manifest request failed: {payload.get('Message')}")
        files = []
        for entry in payload["Data"]["Files"]:
            path = entry["Path"]
            if path not in ASSET_NAMES and not path.endswith(".safetensors"):
                continue
            files.append({
                "path": path,
                "size": entry["Size"],
                "sha256": entry["Sha256"],
                "file_revision": entry["Revision"],
            })
        manifest = {
            "schema_version": 1,
            "source": ENDPOINT,
            "model_id": model_id,
            "requested_revision": revision,
            "fetched_at_utc": utc_now(),
            "listing_url": url,
            "files": sorted(files, key=lambda item: item["path"]),
        }
    files = manifest["files"]
    if model_id == "Qwen/Qwen3.5-0.8B" and revision == "master":
        pinned = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
        if files != pinned["files"]:
            raise ValueError("Existing model manifest differs from the release's pinned base assets; use a new output directory")
    if not files:
        raise ValueError("No model assets found")
    for entry in files:
        validate_file(entry)
    names = {entry["path"] for entry in files}
    if len(names) != len(files):
        raise ValueError("Content manifest contains duplicate paths")
    required = {"config.json", "tokenizer_config.json", "preprocessor_config.json"}
    if not required.issubset(names) or not any(path.endswith(".safetensors") for path in names):
        raise ValueError("Content manifest is missing required model/processor assets")
    if not destination.exists():
        atomic_json(destination, manifest)
    return manifest


def download_file(model_id: str, entry: dict, output_dir: Path, verify_only: bool) -> dict:
    target = output_dir / entry["path"]
    partial = target.with_name(f".{target.name}.part")
    expected_size = entry["size"]
    expected_hash = entry["sha256"]
    if target.is_file() and target.stat().st_size == expected_size and digest(target) == expected_hash:
        print(f"Verified existing {target.name}", flush=True)
        return {**entry, "status": "verified_existing"}
    if verify_only:
        raise RuntimeError(f"Missing or invalid model asset: {target}")
    # This is the URL construction used by ModelScope's official SDK.  Each file
    # uses its committed revision from the listing, never the mutable branch.
    url = (
        f"{ENDPOINT}/api/v1/models/{model_id}/repo?"
        + urllib.parse.urlencode({"Revision": entry["file_revision"], "FilePath": entry["path"]})
    )
    started = time.monotonic()
    for attempt in range(1, 4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                partial.unlink()
                offset = 0
            if offset == expected_size:
                if digest(partial) == expected_hash:
                    os.replace(partial, target)
                    return {**entry, "status": "downloaded", "download_url": url}
                partial.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            print(f"Downloading {target.name}: {offset}/{expected_size} bytes (attempt {attempt})", flush=True)
            with request(url, headers) as response:
                if offset and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab"
                elif response.status == 200:
                    offset, mode = 0, "wb"
                else:
                    raise RuntimeError(f"Unexpected download HTTP status {response.status}")
                received = offset
                last_progress = time.monotonic()
                with partial.open(mode) as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        received += len(chunk)
                        if received > expected_size:
                            raise RuntimeError(f"Response exceeds expected file size: {target.name}")
                        stream.write(chunk)
                        if time.monotonic() - last_progress >= 20:
                            print(f"  {target.name}: {received / expected_size:.1%}", flush=True)
                            last_progress = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f"Incomplete download of {target.name}")
            if digest(partial) != expected_hash:
                partial.unlink()
                raise RuntimeError(f"SHA256 mismatch for {target.name}")
            os.replace(partial, target)
            print(f"Downloaded and verified {target.name}", flush=True)
            return {
                **entry,
                "status": "downloaded",
                "download_url": url,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if attempt == 3:
                raise
            print(f"Retrying {target.name}: {error}", file=sys.stderr, flush=True)
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "Qwen3.5-0.8B")
    parser.add_argument("--report", type=Path, default=OUTPUT_ROOT / "model-download.json")
    parser.add_argument("--verify-only", action="store_true", help="Validate the saved content manifest and local assets")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.model_id):
        parser.error("Invalid ModelScope model ID")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "output_dir": str(output_dir),
        "started_at_utc": utc_now(),
        "status": "running",
        "files": [],
    }
    with (output_dir / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another model preparation process is using this directory", file=sys.stderr)
            return 1
        try:
            if args.verify_only and not (output_dir / MANIFEST_NAME).is_file():
                raise RuntimeError("No saved content manifest; run the downloader first")
            manifest = get_manifest(args.model_id, args.revision, output_dir)
            report["manifest_path"] = str(output_dir / MANIFEST_NAME)
            report["manifest_sha256"] = digest(output_dir / MANIFEST_NAME)
            report["expected_bytes"] = sum(entry["size"] for entry in manifest["files"])
            atomic_json(args.report, report)
            for entry in manifest["files"]:
                report["files"].append(download_file(args.model_id, entry, output_dir, args.verify_only))
                atomic_json(args.report, report)
            report["status"] = "verified"
            report["completed_at_utc"] = utc_now()
            atomic_json(args.report, report)
            print(f"Verified {len(report['files'])} model assets in {output_dir}", flush=True)
            return 0
        except (Exception, KeyboardInterrupt) as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            report["completed_at_utc"] = utc_now()
            atomic_json(args.report, report)
            print(report["error"], file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
