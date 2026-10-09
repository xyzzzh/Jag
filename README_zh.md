<div align="center">

# Jag

**Direct Box Prediction for Efficient Visual Grounding**

[项目主页](https://xyzzzh.github.io/Jag/) · [模型](https://huggingface.co/xyzzzh/Jag) · [在线体验](https://huggingface.co/spaces/xyzzzh/Jag)

[English](README.md) · [简体中文](README_zh.md)

</div>

![Jag architecture](assets/architecture.svg)

## 简介

Jag 是一个直接预测连续边界框的紧凑多模态模型。受 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发，Jag 保留 Qwen3.5-0.8B 的图像与语言处理能力，用直接几何预测替代自回归坐标生成。

轻量 MLP 读取最后一个有效输入 token 的表征，一次前向预测归一化 `cxcywh` 坐标。训练使用 L1 和 GIoU 损失，先适配回归头，再联合微调语言主干、视觉 merger 和回归头，其余视觉编码器保持冻结。

训练使用 **ModelScope ms-swift**，评估使用 **EvalScope**，环境使用 **Docker**，可选 **SwanLab** 实时记录训练日志。

## 评估结果

<!-- RESULTS:START -->
IoU ≥ 0.5 时的准确率（%）。RefCOCO 和 RefCOCO+ 按样本数合并 testA、testB；RefCOCOg 使用 test。

| 模型 | RefCOCO | RefCOCO+ | RefCOCOg |
| :--- | ---: | ---: | ---: |
| Base (Qwen3.5-0.8B) | 79.74 | 70.10 | 77.96 |
| NExT-Chat | 83.68 | 75.67 | 79.28 |
| LocateAnything | **91.35** | 84.12 | **88.54** |
| **Jag** | 91.28 | **85.80** | 87.72 |

Jag 在五个测试划分上均超过 Base，并在三个划分上取得所比较模型中的最高准确率。详见[分划分结果](evaluation/README.md)。

单请求端到端推理：

| 模型 | 延时 ↓ (ms) | 吞吐 ↑ (样本/s) | GPU 显存 ↓ (GiB) |
| :--- | ---: | ---: | ---: |
| Base | 1293.01 | 0.77 | 2.25 |
| NExT-Chat | 120.01 | 8.33 | 15.99 |
| LocateAnything | 239.05 | 4.18 | 9.66 |
| Hi-Token | 616.90 | 1.62 | 7.91 |
| Jag | 65.97 | 15.16 | 2.52 |

在所测设置下，Jag 相比 NExT-Chat、LocateAnything、Hi-Token 分别加速 **1.82×、3.62×、9.35×**。耗时不含模型加载与预热；显存为采样得到的进程峰值显存。
<!-- RESULTS:END -->

![单请求延时与显存](docs/site/assets/inference-cost.png)

详见[评估说明](docs/evaluation.md)和[模型卡](MODEL_CARD.md)。

## 快速开始

安装 Docker Compose 和 NVIDIA Container Toolkit，然后克隆仓库：

```bash
git clone https://github.com/xyzzzh/Jag.git
cd Jag
cp .env.example .env
```

### 1. 准备数据与环境

下载并解压 [train2014.zip](https://huggingface.co/datasets/omlab/VLM-R1/blob/main/train2014.zip)，将 `.env` 中的 `REFCOCO_IMAGES_DIR` 设置为解压后的 `train2014` 文件夹，并准备 RefCOCO 标注压缩包：

```bash
python3 scripts/prepare_data.py --archive /path/to/refcoco.zip --output data/refcoco
bash scripts/docker.sh build
bash scripts/docker.sh run --rm jag python scripts/download_model.py
```

最后一条命令下载基础模型。训练使用 `refcoco_train.jsonl`，评估使用[数据说明](docs/data.md)中的五个测试划分。

### 2. 使用模型

下载已发布的权重并预测边界框：

```bash
bash scripts/docker.sh --eval run --rm jag \
  hf download xyzzzh/Jag --local-dir /models/Jag
bash scripts/infer.sh \
  --image /workspace/datasets/RefCOCO/train2014/your_image.jpg \
  --expression 'the person wearing a red shirt' --weight-dtype bf16
```

输出包含原始图像像素坐标下的边界框，也可直接使用[在线 Demo](https://huggingface.co/spaces/xyzzzh/Jag)。

### 3. 训练

```bash
bash scripts/train.sh
```

若 `models/Jag` 已存在，添加 `--export-output /models/Jag-trained` 导出到另一目录。

[训练配置](configs/train/jag.json)使用合并的 RefCOCO 系列训练数据，训练结束后将模型导出到 `models/Jag`。如需 SwanLab，设置 `SWANLAB_API_KEY` 并添加 `--swanlab`。详见[训练说明](docs/training.md)。

### 4. 评估

```bash
bash scripts/evaluate.sh jag
bash scripts/evaluate.sh base
bash scripts/benchmark.sh
```

质量与推理性能测试详见[评估说明](docs/evaluation.md)。

## 参与贡献

欢迎改进模型、训练和评估。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 引用

引用信息见[英文文档](README.md#citation)。

## 许可证与致谢

代码采用 [Apache 2.0](LICENSE)，模型权重和数据集遵循各自许可证。感谢 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[ms-swift](https://github.com/modelscope/ms-swift)、[EvalScope](https://github.com/modelscope/evalscope)、[SwanLab](https://github.com/SwanHubX/SwanLab) 和 [COCO](https://cocodataset.org/)/[RefCOCO](https://github.com/lichengunc/refer) 作者。
