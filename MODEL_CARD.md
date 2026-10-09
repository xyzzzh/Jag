---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: transformers
tags:
  - visual-grounding
  - referring-expression-comprehension
  - qwen3.5
  - modelscope
  - safetensors
---

# Jag: Direct Box Prediction for Efficient Visual Grounding

[Project page](https://xyzzzh.github.io/Jag/) · [Code](https://github.com/xyzzzh/Jag) · [Demo](https://huggingface.co/spaces/xyzzzh/Jag)

Jag adapts Qwen3.5-0.8B for visual grounding through continuous box prediction. A lightweight MLP reads the last valid input-token state and predicts a complete bounding box in one multimodal forward pass, without coordinate-token generation. The approach is inspired by Jev's use of task-specific outputs.

This repository provides the model weights used in the Jag paper, together with the image processor, tokenizer, and inference code. The model outputs one normalized `cxcywh` box for an image and an English referring expression. The prediction script also returns pixel coordinates in the original image.

## Quick start

Use Python 3.12. The complete model is included; inference does not require a separate base-model download.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('xyzzzh/Jag', local_dir='Jag')"
cd Jag
python -m pip install -e .

python -m jag.predict \
  --checkpoint . \
  --image /path/to/image.jpg \
  --expression 'the person wearing a red shirt' \
  --device cuda \
  --weight-dtype bf16
```

The `bbox_xyxy` field contains `[x1, y1, x2, y2]` in original-image pixels. Add `--output prediction.json` to save the result. The `bf16` option stores the backbone in BF16 and keeps the regression head in FP32; use `--weight-dtype fp32` for the full-test accuracy configuration.

## Grounding accuracy

Acc@0.5 (%) on the complete five test splits:

| Model | RefCOCO testA | RefCOCO testB | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg test |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Base | 84.27 | 74.72 | 76.53 | 62.57 | 77.96 |
| NExT-Chat | 89.66 | 77.04 | 83.76 | 66.19 | 79.28 |
| LocateAnything | 93.23 | **89.26** | 88.00 | 79.57 | **88.54** |
| **Jag** | **93.90** | 88.36 | **90.69** | **80.08** | 87.72 |

Jag improves over its base model on every split and achieves the highest accuracy among the compared models on three splits.

## Inference efficiency

Single-request measurements with BF16 backbone weights:

| Model | Mean latency (ms) ↓ | Throughput (samples/s) ↑ | Peak GPU memory (GiB) ↓ | Jag speedup |
| :--- | ---: | ---: | ---: | ---: |
| Base | 1293.01 | 0.77 | **2.25** | 19.60× |
| NExT-Chat | 120.01 | 8.33 | 15.99 | 1.82× |
| LocateAnything | 239.05 | 4.18 | 9.66 | 3.62× |
| Hi-Token | 616.90 | 1.62 | 7.91 | 9.35× |
| **Jag** | **65.97** | **15.16** | 2.52 | — |

Latency covers image loading, preprocessing, model prediction, and output conversion, excluding model loading and warmup. Memory is the sampled peak GPU process usage. See the [evaluation documentation](https://github.com/xyzzzh/Jag/tree/main/evaluation) for measurement settings and complete results.

## Training

Jag uses 321,327 referring-expression training examples. Training first adapts the regression head, then jointly updates the head, language backbone, and visual merger; the remaining vision encoder stays frozen. The objective is `5 × L1 + 2 × (1 − GIoU)`. ModelScope ms-swift supports training, and EvalScope supports evaluation. The repository provides a [Docker environment and reproduction scripts](https://github.com/xyzzzh/Jag#quick-start).

## Intended use

Jag localizes one object described by an English expression in an image. It returns a bounding box rather than a segmentation mask or a list of detections. Its reported evaluation covers the RefCOCO, RefCOCO+, and RefCOCOg benchmarks.

## License and acknowledgments

See [LICENSE](https://github.com/xyzzzh/Jag/blob/main/LICENSE). Jag builds on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and is inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). We thank the authors of Qwen, ModelScope ms-swift, EvalScope, SwanLab, COCO, and the RefCOCO benchmarks. Datasets and third-party software retain their respective licenses.
