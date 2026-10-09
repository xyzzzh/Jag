"""Jag's Gradio Space, using shared ZeroGPU compute."""

# Import before torch/model code so ZeroGPU can initialize CUDA emulation.
import spaces

import os
from pathlib import Path
import sys

import gradio as gr
from huggingface_hub import snapshot_download

from rendering import predict_and_draw


MODEL_ID = "xyzzzh/Jag"
MODEL_REVISION = "da168ef40db7fc5c9a3b6aaee50554ea9d28f256"
checkpoint = os.environ.get("JAG_MODEL_DIR") or snapshot_download(
    MODEL_ID,
    revision=MODEL_REVISION,
    allow_patterns=[
        "config.json", "model.safetensors", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "jag/*.py",
    ],
)
sys.path.insert(0, str(Path(checkpoint).resolve()))

from jag.predict import GroundingPredictor


# ZeroGPU requires the model's CUDA placement during module initialization.
predictor = GroundingPredictor.from_checkpoint(
    checkpoint, device="cuda", max_pixels=262144, max_length=2048, weight_dtype="bf16",
)


@spaces.GPU(duration=20)
def ground(image_path, expression):
    try:
        return predict_and_draw(predictor, image_path, expression)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error


with gr.Blocks(title="Jag", css=".gradio-container {max-width: 1100px !important}") as app:
    gr.Markdown(
        "# Jag\n"
        "Direct box prediction for efficient visual grounding. "
        "[Model](https://huggingface.co/xyzzzh/Jag)"
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Image", type="filepath", sources=["upload"], height=420)
            expression = gr.Textbox(
                label="Referring expression (English)", placeholder="the person on the left",
                max_lines=3,
            )
            run = gr.Button("Run inference", variant="primary")
        with gr.Column():
            result = gr.Image(label="Predicted bounding box", type="pil", height=420)
    run.click(ground, inputs=[image, expression], outputs=result, api_name="ground")
    expression.submit(ground, inputs=[image, expression], outputs=result)

app.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    app.launch()
