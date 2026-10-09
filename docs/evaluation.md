# Evaluation / 评估

After preparing the data and model, run:

准备好数据与模型后，执行：

```bash
bash scripts/evaluate.sh jag
bash scripts/evaluate.sh base
bash scripts/benchmark.sh
```

Quality evaluation covers RefCOCO testA/testB, RefCOCO+ testA/testB, and RefCOCOg test. Reports are saved under `outputs/evaluation`; performance results are saved under `outputs/benchmark`.

质量评估覆盖 RefCOCO testA/testB、RefCOCO+ testA/testB 与 RefCOCOg test。质量报告保存在 `outputs/evaluation`，性能结果保存在 `outputs/benchmark`。

The benchmark reconstructs the recorded workload from [cost-selection.json](../data/cost-selection.json) and measures Base and Jag in sequence. It uses BF16 backbone weights, an FP32 regression head for Jag, and separate warmup examples. The original annotations are required; the selection file contains row indices only.

成本测试根据 [cost-selection.json](../data/cost-selection.json) 重建所测输入，依次测量 Base 和 Jag，使用 BF16 主干权重，Jag 回归头保持 FP32，并使用单独的预热样本。运行时需要原始标注；选择文件仅保存行号。

- **mIoU**: Mean box intersection over union / 平均框交并比。
- **IoU@0.5**: Fraction of expressions with IoU ≥ 0.5 / IoU 不低于 0.5 的描述比例。
- **Prediction latency / 预测延时**: Time to predict one expression / 单条描述的预测耗时。
- **Throughput / 吞吐**: Expressions processed per second / 每秒处理的描述数。
- **Peak process memory / 峰值进程显存**: GPU process memory sampled through NVML / 通过 NVML 采样的 GPU 进程显存。

Invalid predictions remain in the quality denominator. Quality evaluation uses batches; the latency benchmark uses one request at a time. Evaluation-workflow timing includes loading and scoring and is labeled separately from prediction latency.

无效预测保留在质量指标分母中。质量评估采用批量预测，延时测试逐条测量。评估流程耗时包含加载与计分，与预测延时分别标注。

Full-test quality evaluation uses Jag's FP32 weights with BF16 autocast. The released model retains the prompt used during its training. Each comparison model uses its recorded evaluation prompt and image processor; the published comparison evaluates complete models rather than changing only the box prediction head.

完整测试集上的质量评估使用 Jag 的 FP32 权重与 BF16 自动混合精度。发布模型沿用训练时的提示词；对比模型使用各自评测时的提示词与图像处理器，所报告结果比较的是完整模型。

Complete results and their data are in [evaluation/README.md](../evaluation/README.md) and [results.json](../evaluation/results.json).

完整结果与数据见 [evaluation/README.md](../evaluation/README.md) 和 [results.json](../evaluation/results.json)。
