"""CPU scoring, no-decode prediction, and real two-sample EvalScope acceptance."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn

from jag.data import GroundingCollator
from jag.evaluation import JagAPI, run_evaluation, score_prediction
from jag.predict import GroundingPredictor, inference_inputs, prediction_to_result


class RecordingProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(size=None)
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long)}


class ForwardOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, **inputs):
        self.calls += 1
        assert "bbox_targets" not in inputs and "labels" not in inputs
        assert "image_grid_thw" in inputs and "mm_token_type_ids" in inputs
        return SimpleNamespace(logits=torch.tensor([[.5, .5, 1.4, .6]]))

    def generate(self, *args, **kwargs):
        raise AssertionError("Autoregressive generation must never be called")


def record(image, expression, index):
    return {"images": [str(image)], "sample_id": index,
            "messages": [{"role": "user", "content": "<image> ignore this fallback"},
                         {"role": "assistant", "content": "SECRET_GROUND_TRUTH"}],
            "additional_paras": json.dumps({"bbox_type": "norm1000", "caption": expression}),
            "solution": {"arguments": {"coordinate": [100, 100, 600, 600]}}}


class EvaluationTests(unittest.TestCase):
    def test_scoring_failures_clipping_thresholds_and_degenerate_boxes(self):
        target = [0., 0., 1., 1.]
        half = score_prediction({"raw_bbox_xyxy_normalized": [0, 0, .5, 1]}, target)
        self.assertEqual(half["iou"], .5)
        self.assertEqual(half["acc_05"], 1.)
        self.assertEqual(half["acc_075"], 0.)
        clipped = score_prediction({"raw_bbox_xyxy_normalized": [-.2, 0, 1.2, 1]}, target)
        self.assertEqual(clipped["clip_rate"], 1.)
        self.assertEqual(clipped["iou"], 1.)
        for invalid in ["not json", {}, {"error": "missing image"},
                        {"raw_bbox_xyxy_normalized": [0, 0, float("nan"), 1]},
                        {"raw_bbox_xyxy_normalized": [1, 0, 0, 1]},
                        {"raw_bbox_xyxy_normalized": [[0, 0, 1, 1], [0, 0, 1, 1]]}]:
            scores = score_prediction(invalid, target)
            self.assertEqual(scores["iou"], 0.)
            self.assertEqual(scores["invalid_output_rate"], 1.)
        degenerate = score_prediction({"raw_bbox_xyxy_normalized": [0, 0, 0, 1]}, target)
        self.assertEqual(degenerate["degenerate_rate"], 1.)
        self.assertEqual(degenerate["iou"], 0.)

    def test_inference_matches_training_and_uses_one_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (200, 100)).save(image_path)
            processor = RecordingProcessor()
            train = GroundingCollator(processor)([record(image_path, "left cat", 0)])
            train_messages, train_kwargs = processor.calls[-1]
            with Image.open(image_path) as image:
                prediction_inputs = inference_inputs(processor, image.convert("RGB"), "left cat")
            inference_messages, inference_kwargs = processor.calls[-1]
            self.assertEqual(train_kwargs, inference_kwargs)
            self.assertEqual(train_messages[0][0]["content"][1], inference_messages[0][0]["content"][1])
            self.assertNotIn("bbox_targets", prediction_inputs)
            for key, value in prediction_inputs.items():
                torch.testing.assert_close(value, train[key])
            model = ForwardOnlyModel()
            result = GroundingPredictor(model, processor, device="cpu").predict(image_path, "left cat")
            self.assertEqual(model.calls, 1)
            self.assertTrue(result["clipped"])
            torch.testing.assert_close(torch.tensor(result["bbox_xyxy"]), torch.tensor([0., 20., 200., 80.]))
            self.assertEqual(result["image_size"], [200, 100])
            with self.assertRaisesRegex(ValueError, "four finite"):
                prediction_to_result([.5, float("inf"), .2, .2], (200, 100))

    def test_real_evalscope_two_samples_keep_failed_prediction_in_denominator(self):
        class MockPredictor:
            def __init__(self):
                self.requests = []

            def predict(self, image_path, expression):
                self.requests.append((image_path, expression))
                if expression == "failed case":
                    raise OSError("simulated image decode failure")
                return {"raw_bbox_xyxy_normalized": [.1, .1, .6, .6]}

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "source.png"
            Image.new("RGB", (100, 60)).save(image_path)
            records = [record(image_path, "correct case", 11), record(image_path, "failed case", 12)]
            source = directory / "eval.jsonl"
            source.write_text("".join(json.dumps(item) + "\n" for item in records))
            predictor = MockPredictor()
            result = run_evaluation(source, JagAPI("jag-cpu-test", predictor=predictor),
                                    directory / "evalscope", subset="testA")
            self.assertEqual(len(predictor.requests), 2)
            self.assertEqual({request[1] for request in predictor.requests}, {"correct case", "failed case"})
            reports = list((directory / "evalscope").glob("reports/**/*.json"))
            self.assertTrue(reports, f"EvalScope wrote no report; result={result}")
            payload = json.loads(reports[0].read_text())
            self.assertEqual(payload["num"], 2)
            self.assertEqual(payload["primary_metric_identity"]["name"], "accuracy")
            self.assertEqual(payload["primary_metric_identity"]["dimensions"], {"threshold": .5})
            metrics = [metric for metric in payload["metrics"] if metric["identity"]["name"] in
                       {"accuracy", "iou", "invalid_output_rate"}]
            self.assertEqual(len(metrics), 5)
            for metric in metrics:
                self.assertAlmostEqual(metric["score"], .5, places=5)
                self.assertEqual(metric["num"], 2)
            if os.environ.get("JAG_EVAL_TEST_REPORT"):
                report = Path(os.environ["JAG_EVAL_TEST_REPORT"])
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
