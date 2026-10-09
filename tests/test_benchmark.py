"""CPU-only contracts for synchronized, paired grounding performance reports."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from jag.benchmark import (GpuGuard, TimedModel, argument_parser, latency_summary,
                                   model_weight_statistics, prepare_samples, selection_manifest,
                                   summarize_model, timed_prediction, canonical_gpu_uuid,
                                   discover_gpu, gpu_processes)
from jag.data import RefCOCODataset


class FakeModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, **kwargs):
        self.inputs.append(kwargs)
        return {"raw_bbox_xyxy_normalized": [0, 0, 1, 1]}

    def generate(self, **kwargs):
        self.inputs.append(kwargs)
        return {"raw_bbox_xyxy_normalized": [0, 0, 1, 1], "raw_generation": "[0,0,1000,1000]",
                "generated_token_ids": [1, 2], "generated_tokens": 2, "generation_hit_token_limit": False}


class FakePredictor:
    def __init__(self, model, operation, result=None):
        self.model, self.operation, self.result = model, operation, result

    def predict(self, image, expression):
        kwargs = {"image": image, "expression": expression}
        result = self.model.generate(**kwargs) if self.operation == "generate" else self.model(**kwargs)
        return result if self.result is None else self.result


class BenchmarkTests(unittest.TestCase):
    sample = {"sample_index": 7, "image": "/image.jpg", "expression": "the cat",
              "target_xyxy_normalized": [0, 0, 1, 1], "solution": "DO_NOT_PASS"}

    def test_fixed_selection_disjoint_warmup_and_auditable_digest(self):
        result = selection_manifest(1000)
        expected = random.Random(42).sample(range(1000), 266)
        self.assertEqual(result["warmup_sample_ids"], expected[:10])
        self.assertEqual(result["sample_ids"], expected[10:])
        self.assertFalse(set(result["sample_ids"]) & set(result["warmup_sample_ids"]))
        payload = {key: result[key] for key in ("sample_ids", "warmup_sample_ids")}
        self.assertEqual(result["sample_selection_sha256"],
                         hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        self.assertEqual(result, selection_manifest(1000))
        with self.assertRaises(ValueError):
            selection_manifest(10, samples=10, warmup=1)

    def test_latency_statistics_use_all_samples_and_linear_percentiles(self):
        summary = latency_summary([1, 2, 3, 4])
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertEqual(summary["total_seconds"], .010)
        for invalid in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                latency_summary(invalid)

    def measured(self, key="base", result=None):
        ticks = iter([1.0, 1.1, 1.3, 1.5])
        clock = lambda: next(ticks)
        synchronizations = []
        synchronize = lambda: synchronizations.append(True)
        model = FakeModel()
        wrapper = TimedModel(model, synchronize, clock)
        predictor = FakePredictor(wrapper, "generate" if key == "base" else "forward", result)
        record = timed_prediction(predictor, self.sample, key, synchronize, clock)
        self.assertEqual(len(synchronizations), 4)
        self.assertEqual(model.inputs, [{"image": "/image.jpg", "expression": "the cat"}])
        return record

    def test_wrapper_times_generate_or_forward_without_ground_truth(self):
        for key in ("base", "jag"):
            record = self.measured(key)
            self.assertAlmostEqual(record["end_to_end_ms"], 500)
            self.assertAlmostEqual(record["model_ms"], 200)
            self.assertEqual(record["scores"]["iou"], 1)

    def test_generated_parse_failure_retains_full_latency_and_zero_iou(self):
        failure = {"error": "Malformed generated JSON", "raw_generation": "bad answer",
                   "generated_token_ids": [1, 2, 3], "generated_tokens": 3,
                   "generation_hit_token_limit": True}
        records = [self.measured(), self.measured(result=failure)]
        summary = summarize_model(records)
        self.assertEqual(summary["measured_count"], 2)
        self.assertEqual(summary["latency_ms"]["mean"], 500)
        self.assertEqual(summary["images_per_second"], 2)
        self.assertEqual(summary["subset_quality"]["iou"], .5)
        self.assertEqual(summary["diagnostics"]["invalid_output_count"], 1)
        self.assertEqual(summary["diagnostics"]["zero_iou_count"], 1)
        self.assertEqual(summary["diagnostics"]["generated_tokens"]["total"], 5)
        self.assertEqual(summary["diagnostics"]["token_limit_count"], 1)

    def test_infrastructure_failures_never_become_fast_successful_samples(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected inference failure"):
            self.measured(result={"error": "CUDA out of memory"})
        with self.assertRaisesRegex(RuntimeError, "Unexpected inference failure"):
            self.measured(key="jag", result={"error": "bad forward", "raw_generation": "fake"})

        class BrokenModel(FakeModel):
            def generate(self, **kwargs):
                raise RuntimeError("CUDA failure")

        wrapper = TimedModel(BrokenModel(), lambda: None)
        predictor = FakePredictor(wrapper, "generate")
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            timed_prediction(predictor, self.sample, "base", lambda: None)

    def test_gpu_guard_does_not_confuse_container_pid_with_host_pid(self):
        replies = iter([[], [98765], [98765], [98765, 43210]])
        guard = GpuGuard("GPU-test", query=lambda uuid: next(replies))
        guard.check("before_context")
        guard.check("context_created", allow_own_context=True)
        self.assertEqual(guard.allowed_pid, 98765)
        guard.check("sample_0")
        with self.assertRaisesRegex(RuntimeError, "Competing"):
            guard.check("sample_16")
        self.assertEqual(guard.observations[-1]["unexpected_pids"], [43210])
        with self.assertRaisesRegex(RuntimeError, "Competing"):
            GpuGuard("GPU-test", query=lambda uuid: [123]).check("before_context")
        with self.assertRaisesRegex(RuntimeError, "uniquely"):
            GpuGuard("GPU-test", query=lambda uuid: [123, 456]).check("context", allow_own_context=True)

    def test_torch_bare_uuid_is_normalized_for_nvidia_smi(self):
        bare = "56140a66-16f4-ccc5-2ed2-04d902ebdf02"
        canonical = "GPU-" + bare
        with patch("jag.benchmark.subprocess.run", return_value=SimpleNamespace(
                stdout=json.dumps({"uuid": bare, "device": "cuda:0", "name": "test GPU"}) + "\n")):
            self.assertEqual(discover_gpu("cuda:0")["uuid"], canonical)
        with patch("jag.benchmark.subprocess.run", return_value=SimpleNamespace(stdout="12345\n")) as run:
            self.assertEqual(gpu_processes(bare), [12345])
            self.assertEqual(run.call_args.args[0][1], "--id=" + canonical)
        self.assertEqual(canonical_gpu_uuid(canonical), canonical)
        self.assertEqual(canonical_gpu_uuid("MIG-" + bare), "MIG-" + bare)

    def test_dataset_records_and_image_hashes_capture_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "image.png"
            Image.new("RGB", (100, 100)).save(image)
            source = directory / "data.jsonl"
            row = {"images": [image.name], "sample_id": 123, "additional_paras": {"caption": "the cat"},
                   "solution": {"arguments": {"coordinate": [100, 200, 600, 800]}}}
            source.write_text(json.dumps(row) + "\n")
            result = prepare_samples(RefCOCODataset(source), [0])[0]
            self.assertEqual(result["sample_index"], 0)
            self.assertEqual(result["source_sample_id"], 123)
            self.assertEqual(result["image"], str(image))
            self.assertEqual(result["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(result["target_xyxy_normalized"], [.1, .2, .6, .8])

    def test_actual_weight_dtypes_and_cli_defaults_are_explicit(self):
        model = nn.Sequential(nn.Linear(2, 2).float(), nn.Linear(2, 1).bfloat16())
        weights = model_weight_statistics(model)
        self.assertEqual(weights["parameter_count"], 9)
        self.assertEqual(weights["parameter_bytes"], 30)
        self.assertEqual(weights["parameter_dtypes"]["torch.float32"]["bytes"], 24)
        self.assertEqual(weights["parameter_dtypes"]["torch.bfloat16"]["bytes"], 6)
        args = argument_parser().parse_args(["--base-model", "b", "--checkpoint", "v", "--jsonl", "j", "--output", "o"])
        self.assertEqual((args.samples, args.warmup, args.seed), (256, 10, 42))
        self.assertEqual((args.max_pixels, args.max_length, args.max_new_tokens), (262144, 2048, 128))
        self.assertEqual(args.gpu_memory_gib, 8)


if __name__ == "__main__":
    unittest.main()
