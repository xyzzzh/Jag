"""Validate a demo request and draw its predicted box on the uploaded image."""

import math

from PIL import Image, ImageDraw


def predict_and_draw(predictor, image_path, expression):
    if not image_path:
        raise ValueError("Upload an image first.")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Enter an object description.")
    expression = expression.strip()
    if len(expression) > 1000:
        raise ValueError("Use an object description of at most 1,000 characters.")
    prediction = predictor.predict(image_path, expression)
    box = prediction["bbox_xyxy"]
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("The model did not return a valid bounding box.")
    with Image.open(image_path) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width = max(3, round(min(canvas.size) / 160))
    draw.rectangle(box, outline="#04A4FF", width=width)
    return canvas
