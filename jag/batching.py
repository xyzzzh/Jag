"""Ordered multimodal input preparation for genuine batched evaluation."""

from os import PathLike

from PIL import Image
import torch

from .data import GroundingCollator


def prepare_batch(processor, requests, template, max_pixels, max_length):
    """Return output slots, valid image records and one left-padded CPU batch.

    A request is ``{"image": path, "expression": text}`` or ``(path, text)``.
    Input failures occupy their original result slots. Processor-wide failures,
    including out-of-memory errors, propagate to the batch scheduler.
    """
    if not isinstance(requests, (list, tuple)):
        raise TypeError("Batch requests must be a list or tuple")
    results, valid = [None] * len(requests), []
    GroundingCollator(processor, max_pixels=max_pixels, max_length=max_length)
    for index, request in enumerate(requests):
        try:
            if isinstance(request, dict):
                path, expression = request["image"], request["expression"]
            elif isinstance(request, (tuple, list)) and len(request) == 2:
                path, expression = request
            else:
                raise TypeError("Each request requires an image path and expression")
            if not isinstance(path, (str, bytes, PathLike)):
                raise TypeError("An image filesystem path is required")
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError("A nonempty referring expression is required")
            with Image.open(path) as source:
                image = source.convert("RGB")
            valid.append((index, image, expression.strip()))
        except (OSError, ValueError, TypeError, KeyError) as error:
            results[index] = {"error": f"{type(error).__name__}: {error}"}

    while valid:
        conversations = [[{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": template.format(expression=expression)},
        ]}] for _, image, expression in valid]
        inputs = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "padding_side": "left", "truncation": False})
        input_ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if (input_ids.ndim != 2 or mask.shape != input_ids.shape
                or input_ids.shape[0] != len(valid)):
            raise ValueError("Processor batch dimensions do not match the ordered requests")
        if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask[:, -1] == 1):
            raise ValueError("Batch inference requires a nonempty left-padded attention mask")
        if mask.shape[1] > 1 and torch.any(mask[:, 1:] < mask[:, :-1]):
            raise ValueError("Processor did not apply left padding")
        lengths = mask.sum(dim=-1).tolist()
        oversized = {row for row, length in enumerate(lengths) if length > max_length}
        if not oversized:
            return results, valid, inputs
        # Rebuild the remaining conversations through the processor so flattened
        # pixel patches and image_grid_thw stay aligned; never slice image tensors
        # as though their first dimension were the text batch dimension.
        for row in oversized:
            index, image, _ = valid[row]
            results[index] = {"error": f"ValueError: Input length {int(lengths[row])} exceeds "
                                      f"max_length={max_length}; no truncation",
                              "image_size": list(image.size)}
        valid = [record for row, record in enumerate(valid) if row not in oversized]
    return results, [], None
