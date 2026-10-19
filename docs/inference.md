# Inference / 推理

Try the [online demo](https://huggingface.co/spaces/xyzzzh/Jag), or use the published model on your own image. No training data or separate base model is needed.

可直接使用[在线 Demo](https://huggingface.co/spaces/xyzzzh/Jag)，也可使用已发布模型对自己的图像进行推理，无需训练数据或单独下载基础模型。

## Python

From the repository root, create a Python 3.12 environment and run:

在仓库根目录中创建 Python 3.12 环境并执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m jag.predict \
  --checkpoint xyzzzh/Jag \
  --image /path/to/image.jpg \
  --expression 'the person wearing a red shirt' \
  --device cuda --weight-dtype bf16 \
  --output prediction.json
```

Replace the image path and expression with your example. The model downloads automatically on first use and is cached for later calls. Use `--checkpoint /path/to/Jag` to load a local model directory instead. The JSON field `bbox_xyxy` contains `[x1, y1, x2, y2]` in original-image pixels.

替换图像路径与目标描述即可使用。首次运行自动下载模型，后续使用缓存；也可通过 `--checkpoint /path/to/Jag` 加载本地模型目录。JSON 字段 `bbox_xyxy` 为原图像素坐标下的 `[x1, y1, x2, y2]`。

Use `--weight-dtype bf16` for the inference-cost setting; full-test quality evaluation uses `--weight-dtype fp32` with BF16 autocast.

`--weight-dtype bf16` 对应推理成本测试设置；完整测试集上的质量评估使用 `--weight-dtype fp32` 与 BF16 自动混合精度。

## Docker

For an existing [Docker environment](setup.md), download the published model and run the inference launcher:

已配置好 [Docker 环境](setup.md)时，可下载发布模型并使用推理脚本：

```bash
bash scripts/docker.sh --eval run --rm jag \
  hf download xyzzzh/Jag --local-dir /models/Jag
bash scripts/infer.sh \
  --image /workspace/datasets/RefCOCO/train2014/your_image.jpg \
  --expression 'the person wearing a red shirt' --weight-dtype bf16 \
  --output /outputs/prediction.json
```

The launcher loads `models/Jag` and uses container paths for the image and output.

脚本加载 `models/Jag`，图像与输出位置使用容器内路径。
