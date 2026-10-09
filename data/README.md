# Dataset files / 数据文件

Download images from [train2014.zip](https://huggingface.co/datasets/omlab/VLM-R1/blob/main/train2014.zip). Prepare the annotation archive with:

从 [train2014.zip](https://huggingface.co/datasets/omlab/VLM-R1/blob/main/train2014.zip) 下载图像，使用以下命令准备标注：

```bash
python3 scripts/prepare_data.py --archive /path/to/refcoco.zip --output data/refcoco
```

Training uses `refcoco_train.jsonl` with 321,327 expressions.

训练使用 `refcoco_train.jsonl`，共 321,327 条描述。

Evaluation files / 评测文件：

| Split / 划分 | File / 文件 |
|---|---|
| RefCOCO testA | `refcoco_testA_eval.jsonl` |
| RefCOCO testB | `refcoco_testB_eval.jsonl` |
| RefCOCO+ testA | `refcocop_testA_eval.jsonl` |
| RefCOCO+ testB | `refcocop_testB_eval.jsonl` |
| RefCOCOg test | `refcocog_test_eval.jsonl` |

Set `REFCOCO_IMAGES_DIR` in `.env` to the extracted `train2014` folder. See [data preparation](../docs/data.md), [annotation manifest](manifest.json), and [synthetic format example](examples/synthetic.jsonl).

将 `.env` 中的 `REFCOCO_IMAGES_DIR` 设置为解压后的 `train2014` 文件夹。详见[数据准备](../docs/data.md)、[标注清单](manifest.json)与[人工格式示例](examples/synthetic.jsonl)。

Images and annotations are obtained separately and retain their respective licenses.

图像与标注需另行获取，并遵循各自许可证。
