# Evaluation results / 评估结果

Accuracy at IoU ≥ 0.5 (%). / IoU ≥ 0.5 时的准确率（%）。

| Model / 模型 | RefCOCO testA | RefCOCO testB | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg test |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Base (Qwen3.5-0.8B) | 84.27 | 74.72 | 76.53 | 62.57 | 77.96 |
| NExT-Chat | 89.66 | 77.04 | 83.76 | 66.19 | 79.28 |
| LocateAnything | 93.23 | **89.26** | 88.00 | 79.57 | **88.54** |
| Jag | **93.90** | 88.36 | **90.69** | **80.08** | 87.72 |

## Dataset summaries / 数据集汇总

| Model | RefCOCO | RefCOCO+ | RefCOCOg |
| :--- | ---: | ---: | ---: |
| Base (Qwen3.5-0.8B) | 79.74 | 70.10 | 77.96 |
| NExT-Chat | 83.68 | 75.67 | 79.28 |
| LocateAnything | **91.35** | 84.12 | **88.54** |
| **Jag** | 91.28 | **85.80** | 87.72 |

RefCOCO and RefCOCO+ are pooled by sample count. / RefCOCO 与 RefCOCO+ 按样本数合并。

## Inference performance / 推理性能

| Model | Latency ↓ (ms) | Throughput ↑ (samples/s) | GPU memory ↓ (GiB) |
| :--- | ---: | ---: | ---: |
| Base | 1293.01 | 0.77 | 2.25 |
| NExT-Chat | 120.01 | 8.33 | 15.99 |
| LocateAnything | 239.05 | 4.18 | 9.66 |
| Hi-Token | 616.90 | 1.62 | 7.91 |
| Jag | 65.97 | 15.16 | 2.52 |

Single-request end-to-end prediction. Loading and warmup are excluded; memory is the sampled peak process memory. / 单请求端到端预测，不含加载与预热；显存为采样进程峰值。

The same Jag weights are used for quality and cost measurements. Quality uses FP32 weights with BF16 autocast; cost uses BF16 backbone weights and an FP32 head. / 质量与成本测试使用相同的 Jag 权重，前者使用 FP32 权重与 BF16 自动混合精度，后者使用 BF16 主干与 FP32 回归头。

[Machine-readable results](results.json) · [Source hashes](provenance.json) · [Reproduction](../docs/evaluation.md)
