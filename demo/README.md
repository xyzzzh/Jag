---
title: Jag
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12.12
app_file: app.py
pinned: false
license: apache-2.0
models:
  - xyzzzh/Jag
short_description: Direct box prediction for efficient visual grounding.
---

# Jag demo

Jag predicts a continuous bounding box in one forward pass through a compact multimodal model. The demo accepts an image and an English referring expression, and visualizes the predicted box.

Model: [xyzzzh/Jag](https://huggingface.co/xyzzzh/Jag).

To deploy, upload the contents of this directory to a Gradio Space and select **ZeroGPU** hardware. The app downloads the pinned model revision automatically. No access token or training dependencies are needed.

[Project page](https://xyzzzh.github.io/Jag/) · [Code](https://github.com/xyzzzh/Jag) · [Model](https://huggingface.co/xyzzzh/Jag)
