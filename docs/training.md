# Training / 训练

The two-stage recipe adapts the regression head for 100 steps, then jointly fine-tunes the language backbone, visual merger, and head for two epochs on 321,327 RefCOCO examples. The remaining visual encoder parameters stay frozen. Both stages use cosine learning-rate decay with warmup and the objective `5 × L1 + 2 × (1 − GIoU)`.

训练采用两阶段方案：先训练回归头 100 步，再基于 321,327 条 RefCOCO 样本，联合微调语言主干、视觉 merger 与回归头两个 epoch，其余视觉编码器参数保持冻结。两个阶段均采用 warmup 与 cosine 学习率衰减，优化目标为 `5 × L1 + 2 × (1 − GIoU)`。

Complete [setup](setup.md) and [data preparation](data.md), then run:

完成[环境配置](setup.md)与[数据准备](data.md)后，执行：

```bash
bash scripts/train.sh
```

Training uses `refcoco_train.jsonl`. Parameters are in [configs/train/jag.json](../configs/train/jag.json). Progress and checkpoints are saved under `outputs/jag`; the completed model is exported to `models/Jag`.

训练使用 `refcoco_train.jsonl`，参数位于 [configs/train/jag.json](../configs/train/jag.json)。进度与检查点保存在 `outputs/jag`，完成后的模型导出到 `models/Jag`。

If you already downloaded `models/Jag`, add `--export-output /models/Jag-trained` to save the new model separately. Pass that model directory with `--checkpoint /models/Jag-trained` to the inference and evaluation scripts.

若已下载 `models/Jag`，添加 `--export-output /models/Jag-trained` 将新模型单独保存；推理和评估脚本可通过 `--checkpoint /models/Jag-trained` 指定新模型。

For SwanLab logging, set `SWANLAB_API_KEY` and run:

如需 SwanLab 日志，设置 `SWANLAB_API_KEY` 后执行：

```bash
bash scripts/train.sh --swanlab
```

See [inference](inference.md) and [evaluation](evaluation.md) for the next steps.

后续操作见[推理](inference.md)与[评估](evaluation.md)。
