# Environment setup / 环境配置

Install Docker Compose and NVIDIA Container Toolkit, then prepare the images and annotations using [data preparation](data.md).

安装 Docker Compose 和 NVIDIA Container Toolkit，并按[数据准备](data.md)准备图像与标注。

```bash
cp .env.example .env
bash scripts/docker.sh build
bash scripts/docker.sh run --rm jag python scripts/download_model.py
```

Set `REFCOCO_IMAGES_DIR` in `.env` to the extracted `train2014` directory. The download command saves the Qwen3.5-0.8B base model to `models/Qwen3.5-0.8B`.

将 `.env` 中的 `REFCOCO_IMAGES_DIR` 设置为解压后的 `train2014` 目录。下载命令将 Qwen3.5-0.8B 基础模型保存到 `models/Qwen3.5-0.8B`。

The Docker dependency files define the environment. Training outputs are saved in `outputs/jag`; exported models are saved in `models/Jag`.

Docker 依赖文件定义运行环境。训练产物保存在 `outputs/jag`，导出模型保存在 `models/Jag`。
