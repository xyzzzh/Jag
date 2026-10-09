"""Original Qwen3.5 generation baseline, without an untrained regression head."""

from copy import deepcopy
import hashlib
import json
import math
import re
import threading
import time

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .data import GroundingCollator
from .batching import prepare_batch
from .resources import configure_cuda_memory


PROMPT_TEMPLATE = (
    "Locate the object described by: {expression}\n"
    "Return only one JSON object with the key \"bbox\" and four numbers: "
    "{{\"bbox\": [x1, y1, x2, y2]}}. "
    "The coordinates are the top-left and bottom-right corners of the object's "
    "bounding box, normalized to the range 0 to 1000 relative to the original image. "
    "Do not include explanations or any other objects."
)
PARSER_VERSION = "single-json-box"


def generation_settings(max_new_tokens=128):
    if isinstance(max_new_tokens, bool) or int(max_new_tokens) != max_new_tokens or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return {"max_new_tokens": int(max_new_tokens), "do_sample": False, "num_beams": 1,
            "num_return_sequences": 1, "use_cache": True, "repetition_penalty": 1.0,
            "temperature": None, "top_p": None, "top_k": None, "min_p": None}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are ambiguous")
        result[key] = value
    return result


def parse_generated_box(text, image_size):
    """Accept one JSON bbox/bbox_2d object or four-number array, optionally fenced.

    Never search prose for coordinates, choose among multiple boxes, repair a
    reversed box, or infer a different coordinate scale from the generated values.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model generated no bounding box")
    payload = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", payload, re.IGNORECASE)
    if fenced:
        payload = fenced.group(1).strip()
    box = json.loads(payload, object_pairs_hook=_unique_object)
    if isinstance(box, list) and len(box) == 1 and isinstance(box[0], dict):
        box = box[0]
    if isinstance(box, dict):
        coordinate_keys = set(box) & {"bbox", "bbox_2d"}
        if len(coordinate_keys) != 1:
            raise ValueError("Exactly one bbox or bbox_2d field is required")
        if set(box) - coordinate_keys - {"label"}:
            raise ValueError("Unexpected JSON fields besides one box and optional label")
        if "label" in box and not isinstance(box["label"], str):
            raise ValueError("The optional label must be a string")
        box = box[next(iter(coordinate_keys))]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Expected exactly one four-coordinate bounding box")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        raise ValueError("All box coordinates must be numbers")
    if not all(math.isfinite(value) for value in box):
        raise ValueError("All box coordinates must be finite")
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("Reversed box edges are invalid")
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    raw = [float(value) / 1000.0 for value in box]
    clipped = [min(1.0, max(0.0, value)) for value in raw]
    scale = [width, height, width, height]
    return {
        "bbox_xyxy": [value * multiplier for value, multiplier in zip(clipped, scale)],
        "bbox_xyxy_normalized": clipped,
        "raw_bbox_xyxy_normalized": raw,
        "raw_bbox_xyxy_norm1000": box,
        "coordinate_space": "original_image_pixels",
        "image_size": [width, height],
        "clipped": raw != clipped,
        "degenerate": clipped[2] <= clipped[0] or clipped[3] <= clipped[1],
    }


def base_inference_inputs(processor, image, expression, max_pixels=262144, max_length=2048):
    """The generation model receives only the original image and expression."""
    GroundingCollator(processor, max_pixels=max_pixels, max_length=max_length)
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A nonempty referring expression is required")
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT_TEMPLATE.format(expression=expression.strip())},
    ]}]]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": True, "truncation": False})
    length = int(inputs["attention_mask"].sum(dim=-1).max())
    if length > max_length:
        raise ValueError(f"Input length {length} exceeds max_length={max_length}; no truncation")
    return inputs


class BaseGroundingPredictor:
    def __init__(self, model, processor, device="cuda", max_pixels=262144,
                 max_length=2048, max_new_tokens=128):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor
        self.max_pixels, self.max_length = int(max_pixels), int(max_length)
        self.generation_settings = generation_settings(max_new_tokens)
        self._lock = threading.Lock()
        self.generation_config = deepcopy(model.generation_config)
        self.generation_config.update(**self.generation_settings)
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        if self.generation_config.pad_token_id is None:
            eos = self.generation_config.eos_token_id
            self.generation_config.pad_token_id = eos[0] if isinstance(eos, list) else eos
        self.generation_config.validate()

    @classmethod
    def from_checkpoint(cls, checkpoint, processor_path=None, device="cuda",
                        gpu_memory_gib=None, memory_fraction=None, **kwargs):
        if gpu_memory_gib is None and memory_fraction is None and torch.device(device).type == "cuda":
            gpu_memory_gib = 8.0
        memory_status = configure_cuda_memory(device, gpu_memory_gib, memory_fraction)
        model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True)
        issues = {key: info.get(key) for key in
                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if info.get(key)}
        if issues:
            raise RuntimeError(f"Incomplete original Qwen checkpoint: {issues}")
        processor = AutoProcessor.from_pretrained(processor_path or checkpoint, local_files_only=True)
        predictor = cls(model, processor, device=device, **kwargs)
        predictor.memory_status = memory_status
        return predictor

    def predict(self, image_path, expression):
        with self._lock:
            started = time.perf_counter()
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            inputs = base_inference_inputs(self.processor, image, expression,
                                           self.max_pixels, self.max_length)
            inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = self.model.generate(**inputs, generation_config=self.generation_config)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            if sequences.ndim != 2 or sequences.shape[0] != 1:
                raise ValueError("Expected exactly one generated sequence")
            token_ids = sequences[0, prompt_tokens:].detach().cpu().tolist()
            raw = self.processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            with_special = self.processor.decode(token_ids, skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
            audit = {"raw_generation": raw, "raw_generation_with_special_tokens": with_special,
                     "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                     "prompt_tokens": prompt_tokens,
                     "generation_hit_token_limit": len(token_ids) >= self.generation_settings["max_new_tokens"],
                     "prediction_parser": PARSER_VERSION}
            audit["prediction_seconds"] = time.perf_counter() - started
            try:
                result = parse_generated_box(raw, image.size)
            except (ValueError, TypeError, OverflowError) as error:
                result = {"error": f"{type(error).__name__}: {error}", "image_size": list(image.size)}
            return {**result, **audit}

    def predict_batch(self, requests):
        """Generate one padded batch, preserving one ordered result per request.

        Input and output-parse failures remain in their original slots. Model or
        processor failures, including CUDA OOM, propagate for scheduler handling.
        The single-request ``predict`` timing path remains separate and unchanged.
        """
        with self._lock:
            started = time.perf_counter()
            results, valid, inputs = prepare_batch(
                self.processor, requests, PROMPT_TEMPLATE, self.max_pixels, self.max_length)
            if not valid:
                return results
            prompt_width = int(inputs["input_ids"].shape[1])
            prompt_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
            inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            with torch.inference_mode():
                generated = self.model.generate(**inputs, generation_config=self.generation_config)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            if sequences.ndim != 2 or sequences.shape[0] != len(valid) or sequences.shape[1] < prompt_width:
                raise ValueError("Generated batch dimensions do not match the ordered requests")
            # Every row includes the same padded prompt width. Per-row unpadded
            # lengths must never be used as generation slice offsets.
            token_rows = sequences[:, prompt_width:].detach().cpu().tolist()
            eos = self.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos]) if eos is not None else set()
            for row, ((index, image, _), token_ids) in enumerate(zip(valid, token_rows)):
                # HF generate pads rows that finish before other rows. Preserve
                # the terminating EOS itself and discard only its trailing tail.
                end = next((position + 1 for position, token in enumerate(token_ids) if token in eos_ids), len(token_ids))
                token_ids = token_ids[:end]
                audit = {"generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                         "prompt_tokens": int(prompt_lengths[row]), "padded_prompt_tokens": prompt_width,
                         "generation_hit_token_limit": len(token_ids) >= self.generation_settings["max_new_tokens"],
                         "prediction_parser": PARSER_VERSION, "inference_batch_size": len(valid)}
                try:
                    raw = self.processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    with_special = self.processor.decode(token_ids, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)
                    audit.update(raw_generation=raw, raw_generation_with_special_tokens=with_special)
                    result = parse_generated_box(raw, image.size)
                except (ValueError, TypeError, OverflowError) as error:
                    result = {"error": f"{type(error).__name__}: {error}", "image_size": list(image.size)}
                results[index] = {**result, **audit}
            duration = time.perf_counter() - started
            for index, _, _ in valid:
                # This is a shared batch duration, not a per-request latency.
                results[index]["batch_prediction_seconds"] = duration
            return results


def prompt_sha256():
    return hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
