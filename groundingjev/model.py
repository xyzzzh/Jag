"""A pretrained Qwen3.5 multimodal backbone with a continuous bbox head."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, Qwen3_5Config, Qwen3_5Model
from transformers.utils import ModelOutput

from .geometry import bbox_loss


class GroundingJevConfig(PretrainedConfig):
    model_type = "groundingjev"

    def __init__(self, backbone_config=None, head_hidden_size=512, stage="head",
                 l1_weight=5.0, giou_weight=2.0, backbone_attn_implementation="sdpa",
                 base_model_path=None, **kwargs):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config or {}
        self.head_hidden_size = head_hidden_size
        self.stage = stage
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.backbone_attn_implementation = backbone_attn_implementation
        self.base_model_path = base_model_path
        self.coordinate_format = "normalized_cxcywh"
        self.input_template = "Locate the object described by: {expression}"
        self.use_cache = False
        self.keys_to_ignore_at_inference = ["loss_l1", "loss_giou"]


@dataclass
class GroundingJevOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    loss_l1: torch.Tensor | None = None
    loss_giou: torch.Tensor | None = None


def last_valid_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool correctly for both left and right padding, including mixed lengths."""
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("Pooling requires the original [batch, length] attention mask")
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    positions = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("An input sequence has no valid tokens")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class GroundingJevModel(PreTrainedModel):
    config_class = GroundingJevConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]
    _keep_in_fp32_modules = ["bbox_head"]

    def __init__(self, config: GroundingJevConfig, backbone=None):
        super().__init__(config)
        if backbone is None:
            if not config.backbone_config:
                raise ValueError("A Jag model must include its full backbone configuration")
            backbone_config = Qwen3_5Config.from_dict(config.backbone_config)
            backbone_config._attn_implementation = config.backbone_attn_implementation
            backbone = Qwen3_5Model(backbone_config)
        self.backbone = backbone
        hidden_size = backbone.config.text_config.hidden_size
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, config.head_hidden_size),
            nn.GELU(), nn.Linear(config.head_hidden_size, 4), nn.Sigmoid(),
        ).float()
        # Transformers tracks initialized pretrained modules; post_init skips their
        # weights and installs the wrapper's checkpoint/tied-weight bookkeeping.
        self.post_init()
        self.set_stage(config.stage)

    @classmethod
    def from_base(cls, model_dir, stage="head", hidden_dim=512,
                  torch_dtype=torch.float32, attn_implementation="sdpa", **kwargs):
        from transformers import Qwen3_5ForConditionalGeneration

        kwargs.setdefault("local_files_only", True)
        container, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch_dtype, attn_implementation=attn_implementation,
            output_loading_info=True, **kwargs,
        )
        issues = {key: loading_info.get(key) for key in
                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if loading_info.get(key)}
        if issues:
            raise RuntimeError(f"Pretrained backbone was not loaded exactly: {issues}")
        backbone = container.model
        del container
        config = GroundingJevConfig(
            backbone_config=backbone.config.to_dict(), stage=stage,
            head_hidden_size=hidden_dim, backbone_attn_implementation=attn_implementation,
            base_model_path=str(model_dir),
        )
        model = cls(config, backbone=backbone)
        model.pretrained_loading_issues = issues
        return model

    def set_stage(self, stage):
        if stage not in {"head", "joint"}:
            raise ValueError(f"Unknown training stage: {stage!r}")
        self.config.stage = stage
        self.backbone.requires_grad_(False)
        self.bbox_head.requires_grad_(True)
        if stage == "joint":
            self.backbone.language_model.requires_grad_(True)
            self.backbone.visual.merger.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "backbone"):
            if self.config.stage == "head":
                self.backbone.eval()
            else:
                self.backbone.visual.eval()
                self.backbone.visual.merger.train(mode)
        return self

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False})

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None,
                mm_token_type_ids=None, position_ids=None, bbox_targets=None,
                num_items_in_batch=None, return_dict=True, **kwargs):
        # Explicitly exclude trainer metadata and language-generation arguments.
        for key in ("labels", "use_cache", "output_hidden_states", "output_attentions",
                    "past_key_values", "cache_position", "logits_to_keep"):
            kwargs.pop(key, None)
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("input_ids and attention_mask are required")
            attention_mask = torch.ones_like(input_ids)
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                      mm_token_type_ids=mm_token_type_ids, position_ids=position_ids)
        inputs.update(kwargs)
        inputs = {key: value for key, value in inputs.items() if value is not None}
        # Frozen head pretraining need not retain a backbone autograd graph.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.config.stage == "joint"):
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True,
                                   output_hidden_states=False).last_hidden_state
        pooled = last_valid_pool(hidden, attention_mask)
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            logits = self.bbox_head(pooled.float())
            loss = loss_l1 = loss_giou = None
            if bbox_targets is not None:
                loss, loss_l1, loss_giou = bbox_loss(
                    logits, bbox_targets.to(device=logits.device, dtype=torch.float32),
                    self.config.l1_weight, self.config.giou_weight)
        output = GroundingJevOutput(loss=loss, logits=logits,
                                    loss_l1=loss_l1, loss_giou=loss_giou)
        return output if return_dict else output.to_tuple()
