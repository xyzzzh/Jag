# Inference / 推理

After Docker setup, download the published [Jag weights](https://huggingface.co/xyzzzh/Jag) to skip training:

完成 Docker 环境配置后，可下载已发布的 [Jag 权重](https://huggingface.co/xyzzzh/Jag)，跳过训练：

```bash
bash scripts/docker.sh --eval run --rm jag \
  hf download xyzzzh/Jag --local-dir /models/Jag
```

After downloading or training, run:

下载或训练完成后，执行：

```bash
bash scripts/infer.sh \
  --image /workspace/datasets/RefCOCO/train2014/your_image.jpg \
  --expression 'the person wearing a red shirt' --weight-dtype bf16
```

Replace the filename and expression with your example. The command loads `models/Jag`. The JSON field `bbox_xyxy` contains `[x1, y1, x2, y2]` in original-image pixels. Add `--output /outputs/prediction.json` to save the result.

替换图像文件名与目标描述即可使用。命令加载 `models/Jag`，返回的 JSON 字段 `bbox_xyxy` 是原图像素坐标下的 `[x1, y1, x2, y2]`。添加 `--output /outputs/prediction.json` 可保存结果。

Use `--weight-dtype bf16` for the inference-cost setting; quality evaluation uses the default FP32 weights with BF16 autocast.

`--weight-dtype bf16` 对应推理成本测试设置；质量评估默认使用 FP32 权重与 BF16 自动混合精度。
