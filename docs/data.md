# Data preparation / 数据准备

Download and extract [train2014.zip](https://huggingface.co/datasets/omlab/VLM-R1/blob/main/train2014.zip), and obtain the RefCOCO JSONL annotation files.

下载并解压 [train2014.zip](https://huggingface.co/datasets/omlab/VLM-R1/blob/main/train2014.zip)，并准备 RefCOCO JSONL 标注文件。

```bash
python3 scripts/prepare_data.py --archive /path/to/refcoco.zip --output data/refcoco
```

Alternatively, import existing JSONL files:

也可以导入已有 JSONL：

```bash
python3 scripts/prepare_data.py --jsonl /path/to/annotations/*.jsonl --output data/refcoco
```

Training uses `refcoco_train.jsonl` with 321,327 expressions.

训练使用 `refcoco_train.jsonl`，共 321,327 条描述。

Evaluation files / 评测文件：

| Split / 划分 | File / 文件 | Samples / 样本 |
|---|---|---:|
| RefCOCO testA | `refcoco_testA_eval.jsonl` | 5,657 |
| RefCOCO testB | `refcoco_testB_eval.jsonl` | 5,095 |
| RefCOCO+ testA | `refcocop_testA_eval.jsonl` | 5,726 |
| RefCOCO+ testB | `refcocop_testB_eval.jsonl` | 4,889 |
| RefCOCOg test | `refcocog_test_eval.jsonl` | 9,602 |

Set `REFCOCO_IMAGES_DIR` in `.env` to the extracted `train2014` folder. The preparation script maps image paths to `/workspace/datasets/RefCOCO/train2014` for Docker. Images stay in the original folder.

将 `.env` 中的 `REFCOCO_IMAGES_DIR` 设置为解压后的 `train2014` 文件夹。准备脚本将标注内的图像路径映射为 Docker 中的 `/workspace/datasets/RefCOCO/train2014`，图片仍保存在原文件夹。

Each JSONL record contains an image path, an expression, and a target `xyxy` box normalized to 0–1000. See [the synthetic format example](../data/examples/synthetic.jsonl). [The data manifest](../data/manifest.json) records the annotation identities used by the training and evaluation setup.

每条 JSONL 包含图像路径、目标描述，以及按 0–1000 归一化的目标 `xyxy` 框。格式见[人工示例](../data/examples/synthetic.jsonl)，训练与评估所用标注的身份信息见[数据清单](../data/manifest.json)。
