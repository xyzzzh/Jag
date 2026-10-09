"""Continuous box geometry; training boxes are deliberately never clipped."""

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    low, high = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((low + high) / 2, high - low), dim=-1)


def aligned_iou_giou(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """Return corresponding-pair IoU/GIoU (shape [...]), not a B x B matrix."""
    if prediction.shape != target.shape or prediction.shape[-1] != 4:
        raise ValueError("Aligned boxes must have equal shapes ending in four coordinates")
    prediction, target = prediction.float(), target.float()
    intersection_size = (torch.minimum(prediction[..., 2:], target[..., 2:])
                         - torch.maximum(prediction[..., :2], target[..., :2])).clamp_min(0)
    intersection = intersection_size.prod(dim=-1)
    pred_area = (prediction[..., 2:] - prediction[..., :2]).clamp_min(0).prod(dim=-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(dim=-1)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp_min(eps)
    enclosing_size = (torch.maximum(prediction[..., 2:], target[..., 2:])
                      - torch.minimum(prediction[..., :2], target[..., :2])).clamp_min(0)
    enclosing = enclosing_size.prod(dim=-1)
    giou = iou - (enclosing - union) / enclosing.clamp_min(eps)
    return iou, giou


def bbox_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight=5.0, giou_weight=2.0):
    """FP32 batch means with coordinate-summed L1 and un-clipped GIoU."""
    prediction, target = prediction.float(), target.float()
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 4:
        raise ValueError("Expected prediction and target with identical [batch, 4] shapes")
    l1 = (prediction - target).abs().sum(dim=-1).mean()
    _, giou = aligned_iou_giou(cxcywh_to_xyxy(prediction), cxcywh_to_xyxy(target))
    giou_loss = (1 - giou).mean()
    return l1_weight * l1 + giou_weight * giou_loss, l1, giou_loss
