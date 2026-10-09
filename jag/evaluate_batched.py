"""Resumable batched prediction followed by real, full-denominator EvalScope scoring."""

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from PIL import Image

from .data import RefCOCODataset, referring_expression, target_cxcywh
from .evaluation import JagAPI, checkpoint_manifest, file_sha256, run_evaluation
from .evaluate_base import base_checkpoint_manifest


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def image_identity(path):
    """Record bytes even for an undecodable image; missing images remain scored inputs."""
    result = {"sha256": None, "size": None, "error": None}
    try:
        result["sha256"] = file_sha256(path)
        with Image.open(path) as image:
            result["size"] = list(image.size)
    except (OSError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def prepare_records(jsonl, limit=None):
    source = RefCOCODataset(jsonl)
    records, images = [], {}
    count = len(source) if limit is None else min(len(source), limit)
    for index in range(count):
        row = source[index]
        if len(row.get("images", [])) != 1:
            raise ValueError(f"Sample {index} must contain exactly one image")
        path = Path(row["images"][0])
        if not path.is_absolute():
            path = Path(row.get("_jag_source_dir", ".")) / path
        image = str(path)
        if image not in images:
            images[image] = image_identity(image)
        target_cxcywh(row)
        record = {"index": index, "request": {"image": image, "expression": referring_expression(row)},
                  "source_sample_id": row.get("sample_id"),
                  "target": [float(value) / 1000 for value in row["solution"]["arguments"]["coordinate"]],
                  "image_identity": images[image]}
        record["identity_sha256"] = identity_hash(record)
        records.append(record)
    if not records:
        raise ValueError("The evaluation dataset is empty")
    return records, images


def validate_manifest(saved, current):
    for label, value in (("saved", saved), ("current", current)):
        content = {key: item for key, item in value.items() if key != "fingerprint"}
        expected = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if value.get("fingerprint") != expected:
            raise ValueError(f"The {label} evaluation manifest has an invalid fingerprint")
    if saved != current:
        changed = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError(f"Cached evaluation identity differs: {changed}")


def validate_processor(checkpoint, processor):
    if processor is None or Path(processor).resolve() == Path(checkpoint).resolve():
        return
    checked = 0
    for name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json",
                 "tokenizer.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        original, external = Path(checkpoint) / name, Path(processor) / name
        if original.exists() != external.exists():
            raise ValueError(f"External processor differs from the checkpoint: {name}")
        if original.exists():
            checked += 1
            if file_sha256(original) != file_sha256(external):
                raise ValueError(f"External processor differs from the checkpoint: {name}")
    if not checked:
        raise ValueError("Cannot establish processor identity")


def cached_content(row, record, model_name):
    """Validate one cached entry against the exact current dataset request."""
    if type(row.get("index")) is not int or row["index"] != record["index"]:
        raise ValueError("Cached predictions must form the exact contiguous dataset prefix")
    if "prediction_content" in row:
        for key in ("request", "source_sample_id", "target", "image_identity", "identity_sha256"):
            if row.get(key) != record[key]:
                raise ValueError(f"Cached sidecar sample {record['index']} differs in {key}")
        if row.get("schema_version") != 1:
            raise ValueError("Unsupported sidecar schema")
        content = row["prediction_content"]
    else:
        if row.get("model") != model_name:
            raise ValueError("Cached prediction model name differs from its manifest")
        users = [message for message in row.get("messages", []) if message.get("role") == "user"]
        if len(users) != 1 or json.loads(users[0]["content"]) != record["request"]:
            raise ValueError(f"Cached request differs at sample {record['index']}")
        metadata = row.get("metadata") or {}
        for key, expected in (("source_sample_id", record["source_sample_id"]),
                              ("image", record["request"]["image"]),
                              ("image_size", record["image_identity"]["size"]),
                              ("bbox_xyxy_normalized", record["target"])):
            if metadata.get(key) != expected:
                raise ValueError(f"Cached metadata differs at sample {record['index']}: {key}")
        output = row.get("model_output") or {}
        choices = output.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            raise ValueError("Cached output has no unique original completion")
        content = choices[0]["message"].get("content")
        if output.get("model") != model_name:
            raise ValueError("Cached output model differs from its manifest")
    if not isinstance(content, str):
        raise ValueError("Cached completion must be a string")
    # Invalid model completions are deliberately retained verbatim for zero-IoU scoring.
    return content


def load_cache(path, manifest_path, current_manifest, records, model_name):
    saved_manifest = json.loads(Path(manifest_path).read_text())
    validate_manifest(saved_manifest, current_manifest)
    raw = Path(path).read_bytes()
    rows, ignored_tail = [], 0
    lines = raw.splitlines(keepends=True)
    sidecar = None
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                ignored_tail = len(line)
                break
            raise ValueError(f"Malformed complete cached line {index + 1}") from None
        if index >= len(records):
            raise ValueError("Cached output exceeds the dataset length")
        this_sidecar = "prediction_content" in row
        if sidecar is not None and sidecar != this_sidecar:
            raise ValueError("A prediction cache cannot mix formats")
        sidecar = this_sidecar
        content = cached_content(row, records[index], model_name)
        rows.append({**records[index], "schema_version": 1, "prediction_content": content,
                     "origin": "resumed_sidecar" if sidecar else "resumed_evalscope"})
    return rows, {"path": str(Path(path).resolve()), "manifest": str(Path(manifest_path).resolve()),
                  "snapshot_sha256": hashlib.sha256(raw).hexdigest(), "reused_samples": len(rows),
                  "ignored_incomplete_tail_bytes": ignored_tail,
                  "format": "sidecar" if sidecar else "evalscope",
                  "historical_image_byte_hash_available": bool(sidecar),
                  "legacy_identity_checks": ["manifest", "index", "image_path", "expression", "target", "image_size", "sample_id"]}


def append_cache(handle, records):
    payload = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records)
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


class ReplayPredictor:
    """Exact request-matched replay, preserving duplicate request occurrences."""

    def __init__(self, records):
        self.queues = defaultdict(deque)
        self.lock = threading.Lock()
        self.expected = len(records)
        self.consumed = 0
        self.errors = []
        for record in records:
            request = record["request"]
            self.queues[(request["image"], request["expression"])].append(record["prediction_content"])

    def content(self, image, expression):
        with self.lock:
            queue = self.queues.get((image, expression))
            if not queue:
                self.errors.append({"image": image, "expression": expression})
                raise ValueError("EvalScope requested a missing or already consumed prediction")
            self.consumed += 1
            return queue.popleft()

    def assert_complete(self):
        if self.errors or self.consumed != self.expected or any(self.queues.values()):
            raise RuntimeError("EvalScope did not consume each cached prediction exactly once")


class ReplayAPI(JagAPI):
    def generate(self, input, tools=None, tool_choice=None, config=None):
        from evalscope.api.model import ModelOutput
        users = [message for message in input if message.role == "user"]
        if len(users) != 1:
            raise ValueError("Expected one grounding request during replay")
        request = json.loads(users[0].text)
        if set(request) != {"image", "expression"}:
            raise ValueError("Unexpected EvalScope replay request fields")
        content = self.predictor.content(request["image"], request["expression"])
        return ModelOutput.from_content(model=self.model_name, content=content)


def predict_with_oom_backoff(predictor, records, record_oom, depth=0):
    """Yield successful contiguous sub-batches; only genuine CUDA OOM triggers retry."""
    import torch
    started = time.monotonic()
    single_failure = None
    try:
        predictions = predictor.predict_batch([record["request"] for record in records])
    except torch.cuda.OutOfMemoryError as error:
        record_oom({"start_index": records[0]["index"], "request_batch_size": len(records),
                    "depth": depth, "failed_call_seconds": time.monotonic() - started,
                    "error": str(error), "retry": "bisect" if len(records) > 1 else "abort_single_sample",
                    "timestamp": utc_now()})
        if len(records) == 1:
            single_failure = f"CUDA OOM persists for single sample index {records[0]['index']}: {error}"
    else:
        if not isinstance(predictions, list) or len(predictions) != len(records):
            raise ValueError("Batch predictor must preserve every input in order")
        yield records, predictions, {"depth": depth, "prediction_seconds": time.monotonic() - started}
        return
    # Leave the except block before collection, releasing the failed traceback and
    # its tensor references. Never swallow unrelated RuntimeError/processor errors.
    gc.collect()
    torch.cuda.empty_cache()
    if single_failure is not None:
        raise torch.cuda.OutOfMemoryError(single_failure)
    midpoint = len(records) // 2
    yield from predict_with_oom_backoff(predictor, records[:midpoint], record_oom, depth + 1)
    yield from predict_with_oom_backoff(predictor, records[midpoint:], record_oom, depth + 1)


def precompute(predictor, source, cached, cache_path, batch_size, progress_path=None, batch_audit=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    results = list(cached)
    cache_path = Path(cache_path)
    audit_path = cache_path.with_suffix(".batches.json")
    audit = batch_audit if batch_audit is not None else {}
    audit.update(requested_batch_size=batch_size, successful_batches=[], oom_events=[])
    def record_oom(event):
        audit["oom_events"].append(event)
        atomic_json(audit_path, audit)
        print(json.dumps({"event": "cuda_oom_backoff", **event}), flush=True)
    started = time.monotonic()
    with cache_path.open("x", encoding="utf-8") as handle:
        append_cache(handle, cached)
        atomic_json(audit_path, audit)
        for start in range(len(cached), len(source), batch_size):
            batch = source[start:start + batch_size]
            for successful, predictions, timing in predict_with_oom_backoff(predictor, batch, record_oom):
                completed = [{**record, "schema_version": 1, "origin": "batched_inference",
                              "successful_request_batch_size": len(successful),
                              "successful_batch_start_index": successful[0]["index"],
                              "oom_backoff_depth": timing["depth"],
                              "prediction_content": json.dumps(prediction, ensure_ascii=False, allow_nan=False)}
                             for record, prediction in zip(successful, predictions)]
                append_cache(handle, completed)
                results.extend(completed)
                forward_sizes = {prediction.get("inference_batch_size") for prediction in predictions
                                 if isinstance(prediction, dict)} - {None}
                if len(forward_sizes) > 1:
                    raise RuntimeError("Predictor reported inconsistent effective batch sizes")
                audit["successful_batches"].append({"start_index": successful[0]["index"],
                    "request_batch_size": len(successful), "effective_inference_batch_size":
                    next(iter(forward_sizes)) if forward_sizes else None, **timing})
                atomic_json(audit_path, audit)
                elapsed = time.monotonic() - started
                newly_completed = len(results) - len(cached)
                if progress_path:
                    atomic_json(progress_path, {"status": "predicting", "completed": len(results),
                                "total": len(source), "resumed_samples": len(cached), "new_samples": newly_completed,
                                "batch_size": batch_size, "requested_batch_size": batch_size,
                                "last_successful_request_batch_size": len(successful),
                                "oom_events": len(audit["oom_events"]), "elapsed_prediction_seconds": elapsed,
                                "estimated_remaining_seconds": elapsed / newly_completed * (len(source) - len(results)),
                                "updated_at": utc_now()})
                print(json.dumps({"completed": len(results), "total": len(source),
                                  "requested_batch_size": batch_size, "successful_request_batch_size": len(successful),
                                  "new_prediction_seconds": elapsed}), flush=True)
    return results


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=["base", "jag"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-gib", type=float, default=28)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--processor")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume-predictions")
    parser.add_argument("--resume-manifest")
    parser.add_argument("--limit", type=int, help="Explicit smoke-probe subset only; omit for full evaluation")
    return parser


def manifest_for(args):
    if args.model_kind == "base":
        # Preserve the original quality-manifest identity. The actual batch resource
        # budget is recorded independently; it is not an accuracy protocol change.
        return base_checkpoint_manifest(args.checkpoint, args.jsonl, args.max_pixels,
                                        args.max_length, args.max_new_tokens)
    return checkpoint_manifest(args.checkpoint, args.jsonl, args.max_pixels, args.max_length)


def main():
    args = argument_parser().parse_args()
    if bool(args.resume_predictions) != bool(args.resume_manifest):
        raise ValueError("--resume-predictions and --resume-manifest must be provided together")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    started, started_at = time.monotonic(), utc_now()
    manifest = manifest_for(args)
    validate_processor(args.checkpoint, args.processor)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(args.output) / f"{args.subset}-{manifest['fingerprint'][:12]}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "evaluation_manifest.json", manifest)
    model_name = ("qwen35-base-" if args.model_kind == "base" else "jag-") + manifest["fingerprint"][:12]
    try:
        source, images = prepare_records(args.jsonl, args.limit)
        cached, resume = [], None
        if args.resume_predictions:
            cached, resume = load_cache(args.resume_predictions, args.resume_manifest, manifest, source, model_name)
        protocol = {"schema_version": 1, "mode": "batched_prediction_then_evalscope_replay",
                    "batch_size": args.batch_size, "model_kind": args.model_kind,
                    "actual_gpu_memory_gib": args.gpu_memory_gib, "device": args.device,
                    "processor": args.processor, "expected_samples": len(source), "resume": resume,
                    "limit": args.limit, "probe_only": args.limit is not None,
                    "per_sample_content_identity_sha256": identity_hash([row["identity_sha256"] for row in source]),
                    "batch_padding_may_change_floating_point_results": True,
                    "cuda_oom_policy": "recursive_contiguous_bisection; abort if one sample still exhausts memory",
                    "batch_execution_audit": "predictions.cache.batches.json",
                    "execution_time_is_not_single_request_latency": True,
                    "base_manifest_resource_field_is_original_quality_identity": args.model_kind == "base"}
        atomic_json(output / "execution_protocol.json", protocol)
        from .resources import configure_cuda_memory
        did_infer = len(cached) < len(source)
        resources = (configure_cuda_memory(args.device, args.gpu_memory_gib) if did_infer else
                     {"allocator_limit_applied": False, "reason": "all predictions resumed; no model loaded"})
        atomic_json(output / "resource_limits.json", resources)
        predictor = None
        if did_infer:
            if args.model_kind == "base":
                from .base_predict import BaseGroundingPredictor
                predictor = BaseGroundingPredictor.from_checkpoint(
                    args.checkpoint, processor_path=args.processor, device=args.device,
                    max_pixels=args.max_pixels, max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens, gpu_memory_gib=args.gpu_memory_gib)
                atomic_json(output / "resolved_generation_config.json", predictor.generation_config.to_dict())
            else:
                from .predict import GroundingPredictor
                predictor = GroundingPredictor.from_checkpoint(
                    args.checkpoint, processor_path=args.processor, device=args.device,
                    max_pixels=args.max_pixels, max_length=args.max_length)
        prediction_started = time.monotonic()
        batch_audit = {}
        complete = precompute(predictor, source, cached, output / "predictions.cache.jsonl",
                              args.batch_size, output / "progress.json", batch_audit=batch_audit)
        prediction_seconds = time.monotonic() - prediction_started
        del predictor
        gc.collect()
        peak = {}
        if args.device.startswith("cuda") and did_infer:
            import torch
            peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)}
            torch.cuda.empty_cache()
        validate_manifest(manifest, manifest_for(args))
        validate_processor(args.checkpoint, args.processor)
        for image, original in images.items():
            if image_identity(image) != original:
                raise RuntimeError(f"Evaluation image changed during prediction: {image}")
        replay = ReplayPredictor(complete)
        replay_started = time.monotonic()
        run_evaluation(args.jsonl, ReplayAPI(model_name=model_name, predictor=replay), output,
                       subset=args.subset, limit=args.limit)
        replay.assert_complete()
        replay_seconds = time.monotonic() - replay_started
        batch_sizes = [batch["request_batch_size"] for batch in batch_audit["successful_batches"]]
        forward_batch_sizes = [batch["effective_inference_batch_size"] for batch in batch_audit["successful_batches"]]
        stats = {"status": "completed", "started_at": started_at, "finished_at": utc_now(),
                 "wall_seconds": time.monotonic() - started, "prediction_seconds": prediction_seconds,
                 "batch_size": args.batch_size, "resumed_samples": len(cached),
                 "requested_batch_size": args.batch_size, "request_batch_sizes": batch_sizes,
                 "effective_inference_batch_sizes": forward_batch_sizes,
                 "effective_batch_null_means_not_reported_by_predictor": True,
                 "number_of_prediction_batches": len(batch_sizes),
                 "cuda_oom_backoff_count": len(batch_audit["oom_events"]),
                 "precompute_seconds": prediction_seconds, "evalscope_replay_seconds": replay_seconds,
                 "source_samples": len(RefCOCODataset(args.jsonl)), "probe_only": args.limit is not None,
                 "newly_predicted_samples": len(source) - len(cached), "evaluated_samples": len(source),
                 "includes_resumed_predictions": bool(cached),
                 "timing_scope": "this invocation only; prior cached prediction time excluded", **peak}
        atomic_json(output / "execution_stats.json", stats)
        atomic_json(output / "progress.json", {"status": "completed", "completed": len(source),
                    "total": len(source), "finished_at": utc_now()})
        print(json.dumps({"status": "completed", "output": str(output.resolve())}), flush=True)
    except BaseException as error:
        atomic_json(output / "execution_failed.json", {"status": "failed", "started_at": started_at,
                    "failed_at": utc_now(), "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
