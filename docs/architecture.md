# Architecture / 模型结构

The design is inspired by [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): produce task-specific, typed outputs directly. Jag applies this idea to a continuous four-coordinate output space, implemented with Qwen3.5-0.8B and a regression head.

设计受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发：直接预测任务所需的类型化输出。Jag 将这一思路应用于连续的四维坐标输出空间，具体采用 Qwen3.5-0.8B 与坐标回归头实现。

Jag uses the Qwen3.5-0.8B multimodal backbone with a continuous coordinate-regression head. The last valid token's hidden state provides the joint image–expression representation; the head predicts normalized box coordinates without autoregressive coordinate decoding.

Jag 在 Qwen3.5-0.8B 多模态主干上接入连续坐标回归头，以最后一个有效 token 的隐藏状态作为图像与表达的联合表征，直接回归归一化边界框坐标，无需自回归坐标解码。

```text
Image + expression / 图像 + 描述
    → Qwen3.5-0.8B
    → Last valid token / 最后一个有效 token
    → LayerNorm → Linear → GELU → Linear → Sigmoid
    → Normalized [cx, cy, width, height] / 归一化坐标
```

The regression head has a hidden width of 512. Predictions are converted to original-image `xyxy` coordinates. Training combines L1 regression and generalized IoU loss:

回归头的隐藏维度为 512，预测结果转换为原图 `xyxy` 坐标。训练结合 L1 回归与广义 IoU 损失：

```text
loss = 5 × L1 + 2 × (1 − GIoU)
```

Training first adapts the box head, then jointly trains the language backbone, visual merger, and head. The remaining visual encoder parameters stay frozen.

训练先适配定位头，再联合训练语言主干、视觉融合层与定位头，其余视觉编码器参数保持冻结。

See [training](training.md) and [evaluation](evaluation.md).

操作方法见[训练](training.md)与[评估](evaluation.md)。
