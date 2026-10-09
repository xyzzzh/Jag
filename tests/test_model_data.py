"""CPU checks for box supervision, multimodal batching and complete checkpoints."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from torch import nn
from transformers import Qwen3_5Config

from jag.data import GroundingCollator, RefCOCODataset, referring_expression, target_cxcywh
from jag.geometry import aligned_iou_giou, bbox_loss, cxcywh_to_xyxy, xyxy_to_cxcywh
from jag.model import JagConfig, JagModel, last_valid_pool


def _check_geometry_and_unclipped_boundary_gradients():
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., .25, .25]])
    torch.testing.assert_close(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes)
    iou, giou = aligned_iou_giou(boxes, boxes)
    assert iou.shape == (2,)
    torch.testing.assert_close(iou, torch.ones(2))
    torch.testing.assert_close(giou, torch.ones(2))
    _, disjoint = aligned_iou_giou(boxes[:1], boxes[:1] + 2)
    assert disjoint.item() < 0
    pred = torch.tensor([[.02, .02, .5, .5]], requires_grad=True)
    target = torch.tensor([[.5, .5, .2, .2]])
    loss, l1, giou_loss = bbox_loss(pred, target)
    torch.testing.assert_close(loss, 5 * l1 + 2 * giou_loss)
    torch.testing.assert_close(l1, (pred - target).abs().sum())
    loss.backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad != 0).all()
    degenerate = torch.zeros((1, 4))
    assert torch.isfinite(aligned_iou_giou(degenerate, degenerate)[1]).all()


def _check_pooling_both_padding_sides_and_empty_rejection():
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
    masks = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
    torch.testing.assert_close(last_valid_pool(hidden, masks), torch.stack([hidden[0, 4], hidden[1, 2]]))
    with unittest.TestCase().assertRaisesRegex(ValueError, "no valid tokens"):
        last_valid_pool(hidden, torch.zeros_like(masks))


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.language_model = nn.Embedding(32, 8)
        self.visual = nn.Module()
        self.visual.encoder = nn.Linear(3, 8)
        self.visual.merger = nn.Linear(8, 8)
        self.calls = 0

    def forward(self, input_ids, pixel_values, use_cache, **kwargs):
        assert not use_cache
        self.calls += 1
        hidden = self.language_model(input_ids)
        hidden = hidden + self.visual.merger(self.visual.encoder(pixel_values))[:, None]
        return SimpleNamespace(last_hidden_state=hidden)

    def get_input_embeddings(self):
        return self.language_model


def _check_forward_single_call_fp32_and_exact_trainable_groups(stage):
    backbone = DummyBackbone()
    model = JagModel(JagConfig(head_hidden_size=4, stage=stage), backbone)
    inputs = dict(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
                  attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
                  pixel_values=torch.randn(2, 3), bbox_targets=torch.tensor([[.4, .4, .2, .2]] * 2))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(**inputs)
    assert output.logits.shape == (2, 4) and output.logits.dtype == torch.float32
    assert output.loss.dtype == torch.float32
    assert backbone.calls == 1 and not hasattr(model, "lm_head")
    output.loss.backward()
    assert all(p.grad is not None for p in model.bbox_head.parameters())
    assert all(p.grad is None for p in backbone.visual.encoder.parameters())
    assert not backbone.visual.encoder.training
    for module in (backbone.language_model, backbone.visual.merger):
        assert all(p.requires_grad == (stage == "joint") for p in module.parameters())
        assert all((p.grad is not None) == (stage == "joint") for p in module.parameters())


def sample(image_path, index=0):
    return {"images": [str(image_path)], "sample_id": index,
            "messages": [{"role": "user", "content": "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: left cat."},
                         {"role": "assistant", "content": "SECRET_ASSISTANT_ANSWER"}],
            "solution": {"arguments": {"coordinate": [100, 200, 700, 800]}},
            "additional_paras": json.dumps({"bbox_type": "norm1000", "caption": "left cat"})}


class DummyProcessor:
    def __init__(self, sequence_length=5):
        self.image_processor = SimpleNamespace(size=None)
        self.sequence_length = sequence_length
        self.conversations = None

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations = conversations
        assert kwargs["enable_thinking"] is False and kwargs["processor_kwargs"]["truncation"] is False
        assert kwargs["processor_kwargs"]["padding"] is True and kwargs["add_generation_prompt"] is True
        count = len(conversations)
        return {"input_ids": torch.ones(count, self.sequence_length, dtype=torch.long),
                "attention_mask": torch.ones(count, self.sequence_length, dtype=torch.long),
                "pixel_values": torch.zeros(count, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * count),
                "mm_token_type_ids": torch.zeros(count, self.sequence_length, dtype=torch.long)}


def _check_original_records_no_filtering_and_no_target_leakage(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (80, 40)).save(image_path)
    rows = [sample(image_path, index) for index in range(3)]
    path = tmp_path / "original.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dataset = RefCOCODataset(path)
    assert len(dataset) == 3
    assert [dataset[i]["sample_id"] for i in range(3)] == [0, 1, 2]
    assert dataset[0]["images"] == rows[0]["images"]
    assert dataset[0]["solution"] == rows[0]["solution"]
    processor = DummyProcessor()
    inputs = GroundingCollator(processor)([dataset[0], dataset[2]])
    assert "image_grid_thw" in inputs and "mm_token_type_ids" in inputs
    assert inputs["bbox_targets"].dtype == torch.float32
    torch.testing.assert_close(inputs["bbox_targets"], torch.tensor([[.4, .5, .6, .6]] * 2))
    for conversation in processor.conversations:
        assert [message["role"] for message in conversation] == ["user"]
        assert conversation[0]["content"][1]["text"] == "Locate the object described by: left cat"
    changed = deepcopy(rows[0])
    changed["messages"][1]["content"] = "OTHER_SECRET"
    changed["solution"]["arguments"]["coordinate"] = [200, 300, 800, 900]
    assert referring_expression(changed) == referring_expression(rows[0])
    assert not torch.equal(target_cxcywh(changed), target_cxcywh(rows[0]))
    del changed["additional_paras"]
    assert referring_expression(changed) == "left cat."
    with unittest.TestCase().assertRaisesRegex(ValueError, "no truncation"):
        GroundingCollator(DummyProcessor(10), max_length=9)([rows[0]])


def _check_complete_qwen_checkpoint_round_trip(tmp_path):
    qwen = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, layer_types=["full_attention"],
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                          "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, num_position_embeddings=16),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    model = JagModel(JagConfig(backbone_config=qwen.to_dict(),
                                               head_hidden_size=8, stage="joint"))
    model.eval()
    inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    with torch.no_grad():
        before = model(**inputs).logits
    model.save_pretrained(tmp_path)
    restored, info = JagModel.from_pretrained(tmp_path, output_loading_info=True)
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.eval()
    assert restored.config.stage == "joint"
    assert any(key.startswith("backbone.") for key in restored.state_dict())
    assert any(key.startswith("bbox_head.") for key in restored.state_dict())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        after = restored(**inputs).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)


class ModelDataTests(unittest.TestCase):
    def test_geometry(self):
        _check_geometry_and_unclipped_boundary_gradients()

    def test_pooling(self):
        _check_pooling_both_padding_sides_and_empty_rejection()

    def test_head_stage(self):
        _check_forward_single_call_fp32_and_exact_trainable_groups("head")

    def test_joint_stage(self):
        _check_forward_single_call_fp32_and_exact_trainable_groups("joint")

    def test_data_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_original_records_no_filtering_and_no_target_leakage(Path(directory))

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            _check_complete_qwen_checkpoint_round_trip(Path(directory))


if __name__ == "__main__":
    unittest.main()
