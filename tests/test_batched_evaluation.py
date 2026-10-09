"""CPU coverage for identity-safe resume, durable batch progress and real EvalScope replay."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from jag.evaluate_batched import (
    ReplayAPI, ReplayPredictor, cached_content, load_cache, precompute, prepare_records,
    validate_manifest, main,
)
from jag.evaluation import checkpoint_manifest, run_evaluation


def fixture(root, count=3):
    image = root / "image.png"
    Image.new("RGB", (100, 100), "red").save(image)
    rows = [{"images": [str(image)], "sample_id": index,
             "additional_paras": {"caption": "same request" if index < 2 else "another request"},
             "solution": {"arguments": {"coordinate": [100, 100, 600, 600]}}}
            for index in range(count)]
    jsonl = root / "samples.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))
    records, _ = prepare_records(jsonl)
    manifest = {"checkpoint": "test-checkpoint", "jsonl_sha256": "test-jsonl-hash",
                "input_template": "fixed prompt", "files": {"weights": "fixed-hash"}}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return jsonl, records, manifest, manifest_path


def legacy(record, model="test-model", content=None):
    content = content if content is not None else json.dumps({"raw_bbox_xyxy_normalized": [.1, .1, .6, .6]})
    return {"index": record["index"], "model": model,
            "messages": [{"role": "user", "content": json.dumps(record["request"])}],
            "metadata": {"source_sample_id": record["source_sample_id"],
                         "image": record["request"]["image"],
                         "image_size": record["image_identity"]["size"],
                         "bbox_xyxy_normalized": record["target"]},
            "model_output": {"model": model, "choices": [{"message": {"content": content}}]}}


class BatchedEvaluationTests(unittest.TestCase):
    def test_legacy_identity_and_failed_completions_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records, manifest, path = fixture(root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps(legacy(records[0], content="not json")) + "\n")
            cached, audit = load_cache(predictions, path, manifest, records, "test-model")
            self.assertEqual(cached[0]["prediction_content"], "not json")
            self.assertEqual(audit["reused_samples"], 1)
            self.assertFalse(audit["historical_image_byte_hash_available"])
            for field, value in (("expression", "wrong expression"), ("image", "wrong image")):
                wrong = legacy(records[0])
                request = dict(records[0]["request"])
                request[field] = value
                wrong["messages"][0]["content"] = json.dumps(request)
                with self.assertRaisesRegex(ValueError, "Cached request"):
                    cached_content(wrong, records[0], "test-model")
            with self.assertRaisesRegex(ValueError, "model name"):
                cached_content(legacy(records[0], model="wrong"), records[0], "test-model")

    def test_manifest_tampering_and_semantic_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest, _ = fixture(Path(directory))
            validate_manifest(manifest, manifest)
            changed = {**manifest, "input_template": "changed"}
            with self.assertRaisesRegex(ValueError, "invalid fingerprint"):
                validate_manifest(changed, manifest)
            changed.pop("fingerprint")
            changed["fingerprint"] = hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "identity differs"):
                validate_manifest(changed, manifest)

    def test_only_incomplete_last_line_can_be_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records, manifest, path = fixture(root)
            cache = root / "cache.jsonl"
            cache.write_text(json.dumps(legacy(records[0])) + '\n{"index":')
            values, audit = load_cache(cache, path, manifest, records, "test-model")
            self.assertEqual(len(values), 1)
            self.assertGreater(audit["ignored_incomplete_tail_bytes"], 0)
            cache.write_text(json.dumps(legacy(records[1])) + "\n")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_cache(cache, path, manifest, records, "test-model")
            cache.write_text(json.dumps(legacy(records[0])) + '\n{"index":\n')
            with self.assertRaisesRegex(ValueError, "Malformed complete"):
                load_cache(cache, path, manifest, records, "test-model")

    def test_sidecar_rejects_changed_image_bytes(self):
        class Predictor:
            def predict_batch(self, requests):
                return [{"error": "intentional scored failure"} for _ in requests]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl, records, manifest, path = fixture(root)
            cache = root / "cache.jsonl"
            precompute(Predictor(), records, [], cache, 2)
            cached, audit = load_cache(cache, path, manifest, records, "test-model")
            self.assertEqual(len(cached), 3)
            self.assertTrue(audit["historical_image_byte_hash_available"])
            Image.new("RGB", (100, 100), "blue").save(root / "image.png")
            changed, _ = prepare_records(jsonl)
            with self.assertRaisesRegex(ValueError, "image_identity"):
                load_cache(cache, path, manifest, changed, "test-model")

    def test_batch_failure_retains_prefix_and_resume_only_runs_suffix(self):
        class Predictor:
            def __init__(self, fail=False):
                self.calls = []
                self.fail = fail
            def predict_batch(self, requests):
                self.calls.append(requests)
                if self.fail and len(self.calls) == 2:
                    raise RuntimeError("simulated batch OOM")
                return [{"raw_bbox_xyxy_normalized": [.1, .1, .6, .6]} for _ in requests]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records, manifest, path = fixture(root, 5)
            first = root / "first.jsonl"
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                precompute(Predictor(True), records, [], first, 2)
            cached, _ = load_cache(first, path, manifest, records, "test-model")
            self.assertEqual(len(cached), 2)
            predictor = Predictor()
            complete = precompute(predictor, records, cached, root / "second.jsonl", 2)
            self.assertEqual([len(call) for call in predictor.calls], [2, 1])
            self.assertEqual([row["index"] for row in complete], list(range(5)))

    def test_true_cuda_oom_bisects_and_records_actual_successful_batch_sizes(self):
        class Predictor:
            def __init__(self):
                self.calls = []
            def predict_batch(self, requests):
                self.calls.append(len(requests))
                if len(requests) > 2:
                    raise torch.cuda.OutOfMemoryError("synthetic CUDA allocator OOM")
                return [{"echo": request["expression"], "inference_batch_size": len(requests)}
                        for request in requests]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records, _, _ = fixture(root, 8)
            audit, predictor = {}, Predictor()
            with patch("torch.cuda.empty_cache") as clear:
                complete = precompute(predictor, records, [], root / "predictions.cache.jsonl", 8,
                                      batch_audit=audit)
            self.assertEqual(predictor.calls, [8, 4, 2, 2, 4, 2, 2])
            self.assertEqual(clear.call_count, 3)
            self.assertEqual([row["index"] for row in complete], list(range(8)))
            self.assertEqual([batch["request_batch_size"] for batch in audit["successful_batches"]], [2, 2, 2, 2])
            self.assertEqual([batch["effective_inference_batch_size"] for batch in audit["successful_batches"]], [2, 2, 2, 2])
            self.assertEqual(len(audit["oom_events"]), 3)
            self.assertEqual(json.loads((root / "predictions.cache.batches.json").read_text()), audit)
            for record, completed in zip(records, complete):
                self.assertEqual(json.loads(completed["prediction_content"])["echo"], record["request"]["expression"])
                self.assertEqual(completed["successful_request_batch_size"], 2)

    def test_single_sample_oom_is_fatal_after_durable_successful_prefix(self):
        class Predictor:
            def predict_batch(self, requests):
                if len(requests) > 1 or requests[0]["expression"] == "another request":
                    raise torch.cuda.OutOfMemoryError("synthetic irreducible OOM")
                return [{"error": "valid scored model failure"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records, manifest, path = fixture(root, 3)
            cache = root / "predictions.cache.jsonl"
            with patch("torch.cuda.empty_cache"), self.assertRaisesRegex(torch.cuda.OutOfMemoryError, "single sample index 2"):
                precompute(Predictor(), records, [], cache, 3)
            saved, _ = load_cache(cache, path, manifest, records, "test-model")
            self.assertEqual([row["index"] for row in saved], [0, 1])
            audit = json.loads(cache.with_suffix(".batches.json").read_text())
            self.assertEqual(audit["oom_events"][-1]["retry"], "abort_single_sample")
            self.assertEqual(audit["oom_events"][-1]["start_index"], 2)

    def test_non_cuda_runtime_error_is_never_retried(self):
        class Predictor:
            calls = 0
            def predict_batch(self, requests):
                self.calls += 1
                raise RuntimeError("CUDA out of memory text is not the CUDA OOM exception type")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records, _, _ = fixture(root, 3)
            predictor = Predictor()
            with patch("torch.cuda.empty_cache") as clear, self.assertRaises(RuntimeError):
                precompute(predictor, records, [], root / "cache.jsonl", 3)
            self.assertEqual(predictor.calls, 1)
            clear.assert_not_called()

    def test_real_evalscope_replay_preserves_duplicates_and_failure_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl, records, _, _ = fixture(root)
            complete = [{**record, "prediction_content": json.dumps(
                {"error": "scored failure"} if record["index"] == 1 else
                {"raw_bbox_xyxy_normalized": [.1, .1, .6, .6]})} for record in records]
            replay = ReplayPredictor(complete)
            run_evaluation(jsonl, ReplayAPI(model_name="batched-cpu-test", predictor=replay),
                           root / "evalscope", subset="testA")
            replay.assert_complete()
            report = json.loads(next((root / "evalscope").glob("reports/**/*.json")).read_text())
            self.assertEqual(report["num"], 3)
            iou = next(metric for metric in report["metrics"] if metric["identity"]["name"] == "iou")
            self.assertAlmostEqual(iou["score"], .6667, places=4)
            with self.assertRaisesRegex(ValueError, "already consumed"):
                replay.content(**records[0]["request"])
            with self.assertRaises(RuntimeError):
                replay.assert_complete()

    def test_limit_is_explicit_probe_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl, _, _, _ = fixture(Path(directory), 5)
            records, _ = prepare_records(jsonl, limit=2)
            self.assertEqual(len(records), 2)

    def test_full_cache_cli_replays_on_cpu_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl, records, _, _ = fixture(root)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"inert hash-only fixture; must never load")
            (checkpoint / "config.json").write_text("{}")
            manifest = checkpoint_manifest(checkpoint, jsonl, 262144, 2048)
            manifest_path = root / "real-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            model = "jag-" + manifest["fingerprint"][:12]
            cache = root / "legacy.jsonl"
            cache.write_text("".join(json.dumps(legacy(record, model=model)) + "\n" for record in records))
            arguments = ["evaluate_batched", "--model-kind", "jag", "--checkpoint", str(checkpoint),
                         "--jsonl", str(jsonl), "--output", str(root / "runs"), "--device", "cpu",
                         "--resume-predictions", str(cache), "--resume-manifest", str(manifest_path)]
            with patch("sys.argv", arguments):
                main()
            run = next((root / "runs").iterdir())
            stats = json.loads((run / "execution_stats.json").read_text())
            self.assertEqual(stats["evaluated_samples"], 3)
            self.assertEqual(stats["resumed_samples"], 3)
            self.assertEqual(stats["newly_predicted_samples"], 0)
            self.assertEqual(stats["request_batch_sizes"], [])
            self.assertEqual(json.loads((run / "evaluation_manifest.json").read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
