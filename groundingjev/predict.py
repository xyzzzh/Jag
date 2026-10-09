"""Predict an original-image box with one backbone forward and no decoding."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

from PIL import Image
import torch
from transformers import AutoProcessor

from .data import GroundingCollator
from .batching import prepare_batch
from .geometry import cxcywh_to_xyxy
from .model import GroundingJevModel


def inference_inputs(processor, image, expression, max_pixels=262144, max_length=2048):
    """Use the training template exactly, without accepting any supervision."""
    GroundingCollator(processor, max_pixels=max_pixels, max_length=max_length)
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A nonempty referring expression is required")
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Locate the object described by: " + expression.strip()},
    ]}]]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": True, "truncation": False})
    length = int(inputs["attention_mask"].sum(dim=-1).max())
    if length > max_length:
        raise ValueError(f"Input length {length} exceeds max_length={max_length}; no truncation")
    return inputs


def prediction_to_result(prediction, image_size):
    """Serialize a normalized cxcywh prediction, preserving raw boxes for audit."""
    prediction = torch.as_tensor(prediction, dtype=torch.float32).detach().cpu()
    if prediction.shape != (4,) or not torch.isfinite(prediction).all():
        raise ValueError("Expected exactly four finite predicted cxcywh coordinates")
    if (prediction[2:] < 0).any():
        raise ValueError("Predicted box has a negative width or height")
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("Original image dimensions must be positive")
    raw = cxcywh_to_xyxy(prediction)
    clipped = raw.clamp(0, 1)
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return {
        "bbox_xyxy": (clipped * scale).tolist(),
        "bbox_xyxy_normalized": clipped.tolist(),
        "raw_bbox_xyxy_normalized": raw.tolist(),
        "raw_bbox_cxcywh_normalized": prediction.tolist(),
        "coordinate_space": "original_image_pixels",
        "image_size": [width, height],
        "clipped": bool((raw != clipped).any()),
        "degenerate": bool((clipped[2:] <= clipped[:2]).any()),
    }


class GroundingPredictor:
    def __init__(self, model, processor, device=None, max_pixels=262144,
                 max_length=2048, use_bf16=True):
        self.model = model
        self.processor = processor
        self.device = torch.device(device or next(model.parameters()).device)
        self.model.to(self.device).eval()
        self.max_pixels = int(max_pixels)
        self.max_length = int(max_length)
        self.use_bf16 = use_bf16
        # EvalScope can dispatch requests in threads; serialize access to the local GPU.
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, checkpoint, processor_path=None, device="cuda",
                        weight_dtype="fp32", **kwargs):
        if weight_dtype not in {"fp32", "bf16"}:
            raise ValueError("weight_dtype must be 'fp32' or 'bf16'")
        checkpoint = Path(checkpoint)
        model, info = GroundingJevModel.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, output_loading_info=True)
        issues = {key: info.get(key) for key in
                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if info.get(key)}
        if issues:
            raise RuntimeError(f"Incomplete Jag model: {issues}")
        # Round only the backbone for deployment; retain the trained head in FP32.
        if weight_dtype == "bf16":
            model.backbone.to(dtype=torch.bfloat16)
        if processor_path is None:
            processor_path = (checkpoint if any((checkpoint / filename).exists() for filename in
                                                ("processor_config.json", "preprocessor_config.json"))
                              else model.config.base_model_path)
        if not processor_path:
            raise ValueError("Specify processor_path; no saved processor or base model path is available")
        processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        return cls(model, processor, device=device, **kwargs)

    def predict(self, image_path, expression):
        with self._lock:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            inputs = inference_inputs(self.processor, image, expression,
                                      self.max_pixels, self.max_length)
            inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                         if self.device.type == "cuda" and self.use_bf16 else nullcontext())
            with torch.inference_mode(), precision:
                outputs = self.model(**inputs)
            if outputs.logits.shape != (1, 4):
                raise ValueError(f"Expected [1, 4] prediction, got {tuple(outputs.logits.shape)}")
            return prediction_to_result(outputs.logits[0], image.size)

    def predict_batch(self, requests):
        """Run one backbone forward for the valid requests, preserving all slots."""
        with self._lock:
            started = time.perf_counter()
            results, valid, inputs = prepare_batch(
                self.processor, requests, "Locate the object described by: {expression}",
                self.max_pixels, self.max_length)
            if not valid:
                return results
            inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                         if self.device.type == "cuda" and self.use_bf16 else nullcontext())
            with torch.inference_mode(), precision:
                outputs = self.model(**inputs)
            if outputs.logits.shape != (len(valid), 4):
                raise ValueError(f"Expected [{len(valid)}, 4] predictions, got {tuple(outputs.logits.shape)}")
            predictions = outputs.logits.detach().float().cpu()
            for prediction, (index, image, _) in zip(predictions, valid):
                try:
                    results[index] = prediction_to_result(prediction, image.size)
                except (ValueError, TypeError, OverflowError) as error:
                    results[index] = {"error": f"{type(error).__name__}: {error}", "image_size": list(image.size)}
            duration = time.perf_counter() - started
            for index, _, _ in valid:
                results[index].update(inference_batch_size=len(valid), batch_prediction_seconds=duration)
            return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--processor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-dtype", choices=("fp32", "bf16"), default="fp32",
                        help="Backbone weight precision; the box head always stays in FP32")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output")
    args = parser.parse_args()
    predictor = GroundingPredictor.from_checkpoint(
        args.checkpoint, processor_path=args.processor, device=args.device,
        max_pixels=args.max_pixels, max_length=args.max_length, weight_dtype=args.weight_dtype)
    result = predictor.predict(args.image, args.expression)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
