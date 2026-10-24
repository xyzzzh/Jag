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

The benchmark samples shared test inputs and separate warmup examples from the annotations, then measures Base and Jag in sequence. It uses BF16 backbone weights and an FP32 regression head for Jag. The selected row indices are saved with the benchmark outputs.

成本测试从标注中抽取共同的测试输入和独立的预热样本，再依次测量 Base 和 Jag，使用 BF16 主干权重，Jag 回归头保持 FP32。所选样本的行号随测试结果保存。

- **mIoU**: Mean box intersection over union / 平均框交并比。
- **IoU@0.5**: Fraction of expressions with IoU ≥ 0.5 / IoU 不低于 0.5 的描述比例。
- **Prediction latency / 预测延时**: Time to predict one expression / 单条描述的预测耗时。
- **Throughput / 吞吐**: Expressions processed per second / 每秒处理的描述数。
- **Peak process memory / 峰值进程显存**: GPU process memory sampled through NVML / 通过 NVML 采样的 GPU 进程显存。

Invalid predictions remain in the quality denominator. Quality evaluation uses batches; the latency benchmark uses one request at a time. Evaluation-workflow timing includes loading and scoring and is labeled separately from prediction latency.

无效预测保留在质量指标分母中。质量评估采用批量预测，延时测试逐条测量。评估流程耗时包含加载与计分，与预测延时分别标注。

Full-test quality evaluation uses Jag's FP32 weights with BF16 autocast. The released model retains the prompt used during its training. Each comparison model uses its recorded evaluation prompt and image processor; the published comparison evaluates complete models rather than changing only the box prediction head.

完整测试集上的质量评估使用 Jag 的 FP32 权重与 BF16 自动混合精度。发布模型沿用训练时的提示词；对比模型使用各自评测时的提示词与图像处理器，所报告结果比较的是完整模型。

Per-split accuracy and single-request inference results are summarized in the [model card](../MODEL_CARD.md#grounding-accuracy).

各测试划分的准确率和单请求推理结果见[模型卡](../MODEL_CARD.md#grounding-accuracy)。
