"""CPU contracts for original-model generation, parsing, and EvalScope scoring."""

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from transformers import GenerationConfig

from jag.base_predict import (BaseGroundingPredictor, PROMPT_TEMPLATE,
                                       base_inference_inputs, generation_settings,
                                       parse_generated_box)
from jag.evaluate_base import argument_parser, base_checkpoint_manifest
from jag.evaluation import JagAPI, run_evaluation, score_prediction


class RecordingProcessor:
    def __init__(self, text='{"bbox": [100, 200, 600, 800]}'):
        self.image_processor = SimpleNamespace(size=None)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.calls = []
        self.decode_calls = []
        self.text = text

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": torch.tensor([[10, 11, 12]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    def decode(self, tokens, **kwargs):
        self.decode_calls.append((tokens, kwargs))
        return self.text if kwargs["skip_special_tokens"] else self.text + "<|im_end|>"


class GeneratingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.generation_config = GenerationConfig(eos_token_id=2)
        self.calls = []

    def forward(self, **kwargs):
        raise AssertionError("Baseline must use the original generation model")

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[10, 11, 12, 101, 102, 2]])


class BasePredictionTests(unittest.TestCase):
    def test_parse_single_json_box_normalization_and_fencing(self):
        for payload in ['{"bbox":[100,200,600,800]}', '[100,200,600,800]',
                        '```json\n{"bbox":[100,200,600,800]}\n```']:
            result = parse_generated_box(payload, (200, 100))
            self.assertEqual(result["bbox_xyxy_normalized"], [.1, .2, .6, .8])
            self.assertEqual(result["bbox_xyxy"], [20., 20., 120., 80.])
            self.assertFalse(result["clipped"])
        # Never guess that small numbers already use a different coordinate scale.
        result = parse_generated_box('[0,0,1,1]', (100, 100))
        self.assertEqual(result["bbox_xyxy_normalized"], [0, 0, .001, .001])

    def test_ambiguous_invalid_outputs_are_failures(self):
        invalid = ['', 'prose {"bbox":[0,0,1,1]}', '{"bbox":[0,0,1,1]} trailing',
                   '{"bbox":[0,0,1,1]} {"bbox":[0,0,1,1]}',
                   '{"bbox":[0,0,1,1],"bbox":[0,0,2,2]}',
                   '{"bbox":[0,0,1,1],"other":[1,2,3,4]}', '[[0,0,1,1],[1,1,2,2]]',
                   '{"bbox":[0,0,1,1],"bbox_2d":[0,0,1,1]}',
                   '[{"bbox_2d":[0,0,1,1]},{"bbox_2d":[1,1,2,2]}]',
                   '[{"bbox_2d":[0,0,1,1],"label":42}]',
                   '[{"bbox_2d":[0,0,1,1],"label":"cat","confidence":0.9}]',
                   '[{"bbox_2d":[0,0,1,1],"bbox_2d":[1,1,2,2]}]',
                   '[0,0,1]', '[0,0,true,1]', '[0,0,"1",1]', '[0,0,NaN,1]',
                   '[0,0,Infinity,1]', '[5,0,1,1]', '[0,5,1,1]']
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                parse_generated_box(payload, (100, 100))

    def test_native_qwen_single_box_object_wrappers(self):
        for field in ("bbox", "bbox_2d"):
            for label in ({}, {"label": "person bottom left"}):
                item = {field: [0, 321, 248, 988], **label}
                for wrapper in (item, [item]):
                    for payload in (json.dumps(wrapper), "```json\n" + json.dumps(wrapper) + "\n```"):
                        with self.subTest(payload=payload):
                            result = parse_generated_box(payload, (1000, 1000))
                            self.assertEqual(result["raw_bbox_xyxy_normalized"], [0, .321, .248, .988])
                            self.assertEqual(result["bbox_xyxy"], [0, 321, 248, 988])

    def test_clipping_and_degeneracy_use_same_scorer(self):
        clipped = parse_generated_box('[-100,0,1200,1000]', (100, 60))
        self.assertEqual(clipped["raw_bbox_xyxy_normalized"], [-.1, 0, 1.2, 1])
        self.assertEqual(score_prediction(clipped, [0, 0, 1, 1])["iou"], 1.)
        self.assertEqual(score_prediction(clipped, [0, 0, 1, 1])["clip_rate"], 1.)
        degenerate = parse_generated_box('[100,100,100,600]', (100, 60))
        self.assertEqual(score_prediction(degenerate, [0, 0, 1, 1])["degenerate_rate"], 1.)

    def test_no_ground_truth_argument_prompt_and_identical_image_budget(self):
        self.assertEqual(list(inspect.signature(BaseGroundingPredictor.predict).parameters),
                         ["self", "image_path", "expression"])
        processor = RecordingProcessor()
        inputs = base_inference_inputs(processor, Image.new("RGB", (100, 60)), "left cat")
        messages, kwargs = processor.calls[-1]
        self.assertEqual(len(messages[0]), 1)
        self.assertEqual(messages[0][0]["content"][1]["text"], PROMPT_TEMPLATE.format(expression="left cat"))
        self.assertFalse(kwargs["enable_thinking"])
        self.assertFalse(kwargs["processor_kwargs"]["truncation"])
        self.assertNotIn("bbox_targets", inputs)
        self.assertNotIn("labels", inputs)
        self.assertEqual(processor.image_processor.size, {"shortest_edge": 65536, "longest_edge": 262144})
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            base_inference_inputs(processor, Image.new("RGB", (100, 60)), "cat", max_length=2)

    def test_original_generation_keeps_raw_tokens_and_parse_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (200, 100)).save(image)
            model, processor = GeneratingModel(), RecordingProcessor()
            predictor = BaseGroundingPredictor(model, processor, device="cpu")
            result = predictor.predict(image, "left cat")
            self.assertEqual(result["bbox_xyxy"], [20., 20., 120., 80.])
            self.assertEqual(result["generated_token_ids"], [101, 102, 2])
            self.assertEqual(processor.decode_calls[0][0], [101, 102, 2])
            self.assertIn("<|im_end|>", result["raw_generation_with_special_tokens"])
            self.assertEqual(result["prompt_tokens"], 3)
            self.assertEqual(len(model.calls), 1)
            config = model.calls[-1]["generation_config"]
            self.assertFalse(config.do_sample)
            self.assertEqual(config.max_new_tokens, 128)
            self.assertEqual(config.num_beams, 1)
            processor.text = "bad model response"
            failure = predictor.predict(image, "other cat")
            self.assertIn("error", failure)
            self.assertEqual(failure["raw_generation"], "bad model response")
            self.assertEqual(score_prediction(failure, [0, 0, 1, 1])["invalid_output_rate"], 1.)

    def test_memory_cap_precedes_model_load_and_loads_original_bf16_model(self):
        events = []
        model, processor = GeneratingModel(), RecordingProcessor()
        with patch("jag.base_predict.configure_cuda_memory",
                   side_effect=lambda *args: events.append(("memory", args)) or {"capped": True}), \
             patch("jag.base_predict.Qwen3_5ForConditionalGeneration.from_pretrained",
                   side_effect=lambda *args, **kwargs: events.append(("model", kwargs)) or (model, {})), \
             patch("jag.base_predict.AutoProcessor.from_pretrained", return_value=processor):
            predictor = BaseGroundingPredictor.from_checkpoint("original-qwen", device="cpu", gpu_memory_gib=8)
        self.assertEqual([event[0] for event in events], ["memory", "model"])
        self.assertEqual(events[0][1], ("cpu", 8, None))
        self.assertEqual(events[1][1]["dtype"], torch.bfloat16)
        self.assertTrue(events[1][1]["local_files_only"])
        self.assertEqual(predictor.memory_status, {"capped": True})

    def test_defaults_and_manifest_fingerprints_capture_protocol(self):
        args = argument_parser().parse_args(["--model", "m", "--jsonl", "j", "--output", "o"])
        self.assertEqual((args.max_pixels, args.max_length, args.max_new_tokens), (262144, 2048, 128))
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                generation_settings(value)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "original"
            model.mkdir()
            (model / "weights.safetensors").write_bytes(b"original-model-weights")
            (model / "config.json").write_text('{"model_type":"qwen3_5"}')
            source = directory / "test.jsonl"
            source.write_text('{}\n')
            manifest = base_checkpoint_manifest(model, source)
            fingerprint = manifest.pop("fingerprint")
            self.assertEqual(fingerprint, hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
            self.assertEqual(manifest["model_kind"], "qwen35_original_generation")
            self.assertEqual(manifest["generation_config"]["max_new_tokens"], 128)
            self.assertEqual(manifest["prompt_sha256"], hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest())
            self.assertFalse(manifest["result_cache_enabled"])
            self.assertNotEqual(fingerprint, base_checkpoint_manifest(model, source, max_new_tokens=64)["fingerprint"])

    def test_evalscope_keeps_bad_generations_in_complete_denominator(self):
        class PredictionSequence:
            def __init__(self):
                self.calls = []

            def predict(self, image_path, expression):
                self.calls.append((image_path, expression))
                raw = '{"bbox":[100,100,600,600]}' if expression == "cat" else "nonsense"
                try:
                    return {**parse_generated_box(raw, (100, 100)), "raw_generation": raw}
                except ValueError as error:
                    return {"error": str(error), "raw_generation": raw}

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "image.png"
            Image.new("RGB", (100, 100)).save(image)
            source = directory / "test.jsonl"
            records = [{"images": [str(image)], "additional_paras": {"caption": expression},
                        "messages": [{"role": "assistant", "content": "SECRET_GT"}],
                        "solution": {"arguments": {"coordinate": [100, 100, 600, 600]}}}
                       for expression in ["cat", "bad case"]]
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            predictor = PredictionSequence()
            run_evaluation(source, JagAPI("original-base-cpu-test", predictor=predictor),
                           directory / "eval", subset="testA")
            self.assertEqual({expression for _, expression in predictor.calls}, {"cat", "bad case"})
            report = json.loads(next((directory / "eval").glob("reports/**/*.json")).read_text())
            self.assertEqual(report["num"], 2)
            for metric in report["metrics"]:
                if metric["identity"]["name"] in {"accuracy", "iou", "invalid_output_rate"}:
                    self.assertEqual(metric["num"], 2)
                    self.assertEqual(metric["score"], .5)


if __name__ == "__main__":
    unittest.main()
