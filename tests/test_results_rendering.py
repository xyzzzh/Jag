"""Public tables must preserve unavailable timings and actual batch conditions."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import evaluate_all, render_results


def fixture():
    return {
        "status": "provisional", "quality": {"status": "provisional", "rows": [
            {"dataset": "refcoco", "split": "testA", "samples": 5657, "models": {
                "Qwen3.5-0.8B": {"miou": .75, "iou_at_05": .85},
                "Jag": {"miou": .8, "iou_at_05": .9}}}]},
        "efficiency": {"kind": "evaluation_workflow", "status": "provisional", "samples": 5657,
                       "hardware": "Test GPU", "batch_size": None,
                       "batch_sizes": {"Qwen3.5-0.8B": 64, "Jag": 1},
                       "models": {
                           "Qwen3.5-0.8B": {"mean_ms": None, "samples_per_second": None,
                                            "effective_inference_batch_sizes": [16, 32, 64, None]},
                           "Jag": {"mean_ms": 100., "samples_per_second": 10.,
                                            "effective_inference_batch_sizes": [1]}}},
    }


class ResultsRenderingTests(unittest.TestCase):
    def test_null_timings_are_unavailable_and_mixed_batches_are_explicit(self):
        result = fixture()
        for chinese in (False, True):
            text = render_results.tables(result, chinese=chinese)
            self.assertIn("| Qwen3.5-0.8B | — | — |", text)
            self.assertIn("Qwen3.5-0.8B=64", text)
            self.assertIn("Jag=1", text)
            self.assertIn("16–64+?", text)
            self.assertNotIn("batch size = 1", text)
            self.assertNotIn("batch size = None", text)

    def test_existing_single_request_records_and_formal_benchmark_still_render(self):
        result = fixture()
        result["efficiency"].update(kind="isolated_prediction", status="complete", batch_size=1)
        del result["efficiency"]["batch_sizes"]
        for model in result["efficiency"]["models"].values():
            model.update(mean_ms=100., samples_per_second=10.)
            del model["effective_inference_batch_sizes"]
        text = render_results.tables(result)
        self.assertIn("Latency", text)
        self.assertIn("1.00× inference speedup", text)
        self.assertIn("Qwen3.5-0.8B=1", text)
        self.assertIn("Jag=1", text)
        self.assertNotIn("complete-workflow timing is unavailable", text)

    def test_merge_uses_sample_statistics_not_equal_split_or_rounded_averages(self):
        result = fixture()
        result["quality"]["rows"] = [
            {"dataset": "refcoco", "split": "testA", "samples": 9, "models": {
                label: {"miou": .89, "iou_at_05": .99, "iou_sum": 8.1, "iou_at_05_count": 9}
                for label in render_results.LABELS}},
            {"dataset": "refcoco", "split": "testB", "samples": 1, "models": {
                label: {"miou": .11, "iou_at_05": .01, "iou_sum": .1, "iou_at_05_count": 0}
                for label in render_results.LABELS}},
        ]
        merged, = render_results.dataset_rows(result)
        self.assertEqual(merged["samples"], 10)
        self.assertAlmostEqual(merged["models"]["Jag"]["miou"], .82)
        self.assertAlmostEqual(merged["models"]["Jag"]["iou_at_05"], .9)
        result["quality"]["rows"][1]["models"]["Jag"].pop("iou_sum")
        with self.assertRaisesRegex(ValueError, "original IoU sums"):
            render_results.dataset_rows(result)

    def test_summary_covers_all_datasets_and_details_keep_each_split(self):
        result = fixture()
        result["quality"]["rows"] += [
            {"dataset": "refcocop", "split": "testA", "samples": 4,
             "models": deepcopy(result["quality"]["rows"][0]["models"])},
            {"dataset": "refcocog", "split": "test", "samples": 6,
             "models": deepcopy(result["quality"]["rows"][0]["models"])},
        ]
        summary = render_results.tables(result)
        details = render_results.tables(result, all_splits=True)
        for dataset in ("RefCOCO", "RefCOCO+", "RefCOCOg"):
            self.assertIn(f"| {dataset} |", summary)
        self.assertIn("| RefCOCO testA | 5,657 |", details)
        self.assertIn("| RefCOCO+ testA | 4 |", details)
        self.assertIn("| RefCOCOg test | 6 |", details)

    def test_speedup_is_latency_ratio_only_for_complete_prediction_benchmark(self):
        result = fixture()
        result["efficiency"]["models"]["Qwen3.5-0.8B"]["mean_ms"] = 1600.
        result["efficiency"]["models"]["Jag"]["mean_ms"] = 200.
        self.assertIsNone(render_results.inference_speedup(result["efficiency"]))
        result["efficiency"].update(kind="isolated_prediction", status="complete")
        self.assertEqual(render_results.inference_speedup(result["efficiency"]), 8.)
        text = render_results.tables(result)
        self.assertIn("8.00× inference speedup", text)
        self.assertIn("87.50% lower mean latency", text)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Figure rendering requires the docs extra")
    def test_missing_timings_can_render_figures_without_zero_bars(self):
        for all_missing in (False, True):
            result = deepcopy(fixture())
            if all_missing:
                result["efficiency"]["models"]["Jag"].update(mean_ms=None, samples_per_second=None)
            with TemporaryDirectory() as directory:
                root = Path(directory)
                render_results.figure(result, root)
                svg = (root / "assets/figures/evaluation-results.svg").read_text()
                self.assertIn("Unavailable", svg)
                self.assertIn("Requested batch", svg)
                self.assertTrue((root / "assets/figures/evaluation-results.png").is_file())

    def test_quality_launcher_defaults_to_64_and_uses_batch_evaluator_for_both_models(self):
        with TemporaryDirectory() as directory:
            annotations = Path(directory)
            for dataset, subset in evaluate_all.SPLITS:
                (annotations / f"{dataset}_{subset}_eval.jsonl").write_text("{}\n")
            for model in ("jag", "groundingjev", "base"):
                argv = ["evaluate_all.py", model, "--annotations", str(annotations)]
                with patch("sys.argv", argv), patch.object(evaluate_all.subprocess, "run") as run:
                    evaluate_all.main()
                self.assertEqual(run.call_count, 5)
                for call in run.call_args_list:
                    arguments = call.args[0]
                    self.assertIn("groundingjev.evaluate_batched", arguments)
                    self.assertEqual(arguments[arguments.index("--model-kind") + 1], "base" if model == "base" else "groundingjev")
                    self.assertEqual(arguments[arguments.index("--batch-size") + 1], "64")
                    expected = "/models/Qwen3.5-0.8B" if model == "base" else "/models/Jag"
                    self.assertEqual(arguments[arguments.index("--checkpoint") + 1], expected)


if __name__ == "__main__":
    unittest.main()
