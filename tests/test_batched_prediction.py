"""CPU regressions for batch ordering, multimodal padding and generated tails."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn
from transformers import GenerationConfig

from jag.base_predict import BaseGroundingPredictor
from jag.predict import GroundingPredictor


class BatchProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(size=None)
        self.tokenizer = SimpleNamespace(pad_token_id=0, padding_side="right")
        self.calls = []
        self.decode_calls = []
        self.invalid_tokens = set()

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        rows, patches, grids = [], [], []
        for messages in conversations:
            content = messages[0]["content"]
            image, text = content[0]["image"], content[1]["text"]
            length = 12 if "oversized" in text else 6 if "long" in text else 3
            rows.append([10] * (length - 1) + [image.width])
            grid = [1, image.width // 50, 2]
            grids.append(grid)
            patches.extend([[float(image.width), 0, 0]] * (grid[0] * grid[1] * grid[2]))
        maximum = max(map(len, rows))
        padding_side = kwargs["processor_kwargs"].get("padding_side", self.tokenizer.padding_side)
        ids, masks = [], []
        for row in rows:
            padding = [0] * (maximum - len(row))
            ids.append(padding + row if padding_side == "left" else row + padding)
            masks.append([0] * len(padding) + [1] * len(row) if padding_side == "left"
                         else [1] * len(row) + [0] * len(padding))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "pixel_values": torch.tensor(patches), "image_grid_thw": torch.tensor(grids),
                "mm_token_type_ids": torch.tensor(masks)}

    def decode(self, tokens, **kwargs):
        self.decode_calls.append((list(tokens), kwargs))
        if tokens[0] in self.invalid_tokens:
            return "malformed generated response"
        text = '{"bbox": [100, 200, 600, 800]}'
        return text if kwargs["skip_special_tokens"] else text + "<eos>"


def verify_images(inputs):
    widths = inputs["input_ids"][:, -1].tolist()
    patch_offset = 0
    for width, grid in zip(widths, inputs["image_grid_thw"]):
        count = int(grid.prod())
        assert torch.all(inputs["pixel_values"][patch_offset:patch_offset + count, 0] == width)
        patch_offset += count
    assert patch_offset == inputs["pixel_values"].shape[0]
    assert "bbox_targets" not in inputs and "labels" not in inputs
    return widths


class BatchGenerator(nn.Module):
    def __init__(self, eos_token_id=2, pad_token_id=0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.generation_config = GenerationConfig(eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        self.calls = []
        self.failure = None
        self.bad_shape = False

    def generate(self, **inputs):
        self.calls.append(inputs)
        if self.failure is not None:
            raise self.failure
        widths = verify_images(inputs)
        tails = {100: [101, 2], 200: [201, 202, 2], 300: [301, 302, 303, 304]}
        values = [tails[width] for width in widths]
        width = max(map(len, values))
        pad = inputs["generation_config"].pad_token_id
        extra = torch.tensor([value + [pad] * (width - len(value)) for value in values])
        sequences = torch.cat([inputs["input_ids"], extra], dim=1)
        if self.bad_shape:
            sequences = sequences[:1]
        return SimpleNamespace(sequences=sequences)


class BatchRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = []
        self.failure = None
        self.invalid_width = None

    def forward(self, **inputs):
        self.calls.append(inputs)
        if self.failure is not None:
            raise self.failure
        widths = verify_images(inputs)
        values = [[.5, .5, .4, .6] if width != self.invalid_width else [float("nan"), .5, .4, .6]
                  for width in widths]
        return SimpleNamespace(logits=torch.tensor(values))


class BatchedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = {}
        for width in (100, 200, 300):
            path = self.root / f"{width}.png"
            Image.new("RGB", (width, 50)).save(path)
            self.images[width] = path

    def test_base_left_padding_common_prompt_cut_and_eos_tail_counts(self):
        model, processor = BatchGenerator(), BatchProcessor()
        predictor = BaseGroundingPredictor(model, processor, device="cpu", max_new_tokens=4)
        results = predictor.predict_batch([
            {"image": self.images[100], "expression": "short"},
            {"image": self.images[200], "expression": "long expression"},
            {"image": self.images[300], "expression": "short"},
        ])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["attention_mask"].tolist(), [[0, 0, 0, 1, 1, 1], [1] * 6, [0, 0, 0, 1, 1, 1]])
        self.assertEqual([item["generated_token_ids"] for item in results], [[101, 2], [201, 202, 2], [301, 302, 303, 304]])
        self.assertEqual([item["generated_tokens"] for item in results], [2, 3, 4])
        self.assertEqual([item["prompt_tokens"] for item in results], [3, 6, 3])
        self.assertEqual([item["padded_prompt_tokens"] for item in results], [6] * 3)
        self.assertEqual([item["generation_hit_token_limit"] for item in results], [False, False, True])
        self.assertEqual(processor.tokenizer.padding_side, "right")
        self.assertTrue(all("prediction_seconds" not in result for result in results))
        self.assertTrue(all(result["batch_prediction_seconds"] > 0 for result in results))

    def test_base_input_and_parse_failures_preserve_original_result_slots(self):
        model, processor = BatchGenerator(), BatchProcessor()
        processor.invalid_tokens.add(201)
        predictor = BaseGroundingPredictor(model, processor, device="cpu")
        results = predictor.predict_batch([
            (self.images[100], "short"), (self.root / "missing.png", "cat"),
            (self.images[200], "long"), (self.images[300], "  "), (self.images[300], "short"),
        ])
        self.assertEqual(len(results), 5)
        self.assertEqual(["error" in result for result in results], [False, True, True, True, False])
        self.assertEqual(results[2]["raw_generation"], "malformed generated response")
        self.assertEqual(results[4]["generated_token_ids"], [301, 302, 303, 304])
        self.assertEqual(model.calls[0]["input_ids"].shape[0], 3)

    def test_eos_equal_to_padding_keeps_one_terminal_token(self):
        model, processor = BatchGenerator(eos_token_id=[2, 3], pad_token_id=2), BatchProcessor()
        results = BaseGroundingPredictor(model, processor, device="cpu").predict_batch(
            [(self.images[100], "short"), (self.images[200], "long")])
        self.assertEqual(results[0]["generated_token_ids"], [101, 2])
        self.assertEqual(results[1]["generated_token_ids"], [201, 202, 2])

    def test_oversized_sample_rebuilds_multimodal_batch_without_grid_misalignment(self):
        for model_class, predictor_class in ((BatchGenerator, BaseGroundingPredictor), (BatchRegressor, GroundingPredictor)):
            with self.subTest(model=model_class.__name__):
                model, processor = model_class(), BatchProcessor()
                predictor = predictor_class(model, processor, device="cpu", max_length=8)
                results = predictor.predict_batch([(self.images[100], "short"),
                                                   (self.images[200], "oversized"),
                                                   (self.images[300], "long")])
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(len(processor.calls), 2)
                self.assertEqual(["error" in result for result in results], [False, True, False])
                self.assertEqual(model.calls[0]["input_ids"][:, -1].tolist(), [100, 300])
                self.assertEqual(model.calls[0]["image_grid_thw"].tolist(), [[1, 2, 2], [1, 6, 2]])

    def test_grounding_batch_uses_one_forward_and_keeps_invalid_row(self):
        model, processor = BatchRegressor(), BatchProcessor()
        model.invalid_width = 200
        results = GroundingPredictor(model, processor, device="cpu").predict_batch([
            (self.images[100], "short"), (self.images[200], "long"), (self.images[300], "short")])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(["error" in result for result in results], [False, True, False])
        self.assertEqual(results[0]["image_size"], [100, 50])
        self.assertEqual(results[2]["image_size"], [300, 50])
        self.assertTrue(all(item["inference_batch_size"] == 3 for item in results))
        self.assertTrue(all(item["batch_prediction_seconds"] > 0 for item in results))

    def test_batch_singleton_matches_single_predict_and_excludes_ground_truth(self):
        for model_class, predictor_class in ((BatchGenerator, BaseGroundingPredictor), (BatchRegressor, GroundingPredictor)):
            with self.subTest(model=model_class.__name__):
                model, processor = model_class(), BatchProcessor()
                predictor = predictor_class(model, processor, device="cpu")
                single = predictor.predict(self.images[100], "short")
                batch = predictor.predict_batch([{"image": self.images[100], "expression": "short", "solution": "SECRET_GT"}])[0]
                for key, value in single.items():
                    if key != "prediction_seconds":
                        self.assertEqual(batch[key], value)
                self.assertNotIn("SECRET_GT", str(processor.calls))

    def test_empty_and_all_invalid_batches_do_not_invoke_model(self):
        for model_class, predictor_class in ((BatchGenerator, BaseGroundingPredictor), (BatchRegressor, GroundingPredictor)):
            model, processor = model_class(), BatchProcessor()
            predictor = predictor_class(model, processor, device="cpu")
            self.assertEqual(predictor.predict_batch([]), [])
            results = predictor.predict_batch([{}, (self.images[100], None), (17, "cat")])
            self.assertTrue(all("error" in result for result in results))
            self.assertEqual(model.calls, [])

    def test_cuda_oom_and_model_errors_propagate_for_scheduler_handling(self):
        for model_class, predictor_class in ((BatchGenerator, BaseGroundingPredictor), (BatchRegressor, GroundingPredictor)):
            for failure in (torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("model failure")):
                model, processor = model_class(), BatchProcessor()
                model.failure = failure
                predictor = predictor_class(model, processor, device="cpu")
                with self.subTest(model=model_class.__name__, error=type(failure).__name__), self.assertRaises(type(failure)):
                    predictor.predict_batch([(self.images[100], "short")])

    def test_generation_shape_mismatch_never_silently_drops_rows(self):
        model, processor = BatchGenerator(), BatchProcessor()
        model.bad_shape = True
        predictor = BaseGroundingPredictor(model, processor, device="cpu")
        with self.assertRaisesRegex(ValueError, "Generated batch dimensions"):
            predictor.predict_batch([(self.images[100], "short"), (self.images[200], "long")])


if __name__ == "__main__":
    unittest.main()
