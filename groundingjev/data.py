"""Read original RefCOCO JSONL records and compile image/expression-only inputs."""

from array import array
import json
import os
from pathlib import Path
import re

from PIL import Image
import torch
from torch.utils.data import Dataset

from .geometry import xyxy_to_cxcywh


class RefCOCODataset(Dataset):
    """Keep every source line, with a compact byte-offset index and lazy JSON reads."""

    def __init__(self, path, image_root=None):
        self.path = Path(path).resolve(strict=True)
        self.image_root = str(image_root) if image_root is not None else None
        self.offsets = array("Q")
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                self.offsets.append(offset)
        self._handle = None
        self._pid = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._pid = os.getpid()
        self._handle.seek(self.offsets[index])
        try:
            row = json.loads(self._handle.readline())
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Invalid JSON at {self.path}:{index + 1}") from error
        row["_groundingjev_source_dir"] = self.image_root or str(self.path.parent)
        return row

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_handle=None, _pid=None)
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


def additional_metadata(row):
    metadata = row.get("additional_paras") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError("additional_paras must contain a JSON object")
    return metadata


def referring_expression(row):
    metadata = additional_metadata(row)
    expression = metadata.get("caption")
    if not expression:
        user_messages = [message for message in row.get("messages", []) if message.get("role") == "user"]
        if not user_messages:
            raise ValueError("A referring expression requires a caption or user message")
        content = user_messages[-1]["content"]
        if isinstance(content, list):
            content = "\n".join(item["text"] for item in content if item.get("type") == "text")
        expression = str(content).replace("<image>", "").strip()
        expression = re.sub(
            r"^Please provide the bounding box coordinate of the region this sentence describes:\s*",
            "", expression, flags=re.IGNORECASE)
    expression = str(expression).strip()
    if not expression:
        raise ValueError("Empty referring expression")
    return expression


def target_cxcywh(row):
    metadata = additional_metadata(row)
    if metadata.get("bbox_type", "norm1000") != "norm1000":
        raise ValueError("Expected norm1000 xyxy coordinates in the original JSONL")
    target = torch.tensor(row["solution"]["arguments"]["coordinate"], dtype=torch.float32)
    if target.shape != (4,) or not torch.isfinite(target).all():
        raise ValueError("A bounding box requires four finite coordinates")
    if (target < 0).any() or (target > 1000).any() or (target[2:] < target[:2]).any():
        raise ValueError(f"Invalid norm1000 xyxy bounding box: {target.tolist()}")
    return xyxy_to_cxcywh(target / 1000.0)


class GroundingCollator:
    def __init__(self, processor, max_pixels=262144, max_length=2048):
        self.processor = processor
        self.max_pixels = int(max_pixels)
        self.max_length = int(max_length)
        if self.max_pixels < 1024 or self.max_length < 1:
            raise ValueError("Invalid image or sequence budget")
        # Qwen3.5 represents image budgets as pixel counts in the processor size.
        self.processor.image_processor.size = {
            "shortest_edge": min(65536, self.max_pixels), "longest_edge": self.max_pixels}

    def __call__(self, rows):
        conversations, targets = [], []
        for row in rows:
            images = row.get("images", [])
            if len(images) != 1:
                raise ValueError("Jag expects exactly one original image per sample")
            image_path = Path(images[0])
            if not image_path.is_absolute():
                image_path = Path(row.get("_groundingjev_source_dir", ".")) / image_path
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            expression = referring_expression(row)
            conversations.append([{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Locate the object described by: " + expression},
            ]}])
            targets.append(target_cxcywh(row))
        if not conversations:
            raise ValueError("Cannot collate an empty batch")
        inputs = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "truncation": False},
        )
        lengths = inputs["attention_mask"].sum(dim=-1)
        if (lengths > self.max_length).any():
            raise ValueError(f"Input length {int(lengths.max())} exceeds max_length={self.max_length}; no truncation")
        inputs["bbox_targets"] = torch.stack(targets).float()
        return inputs
