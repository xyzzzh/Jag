#!/usr/bin/env python3
"""Draw Jag's qualitative framework in compact ACL figure style.

Architecture sources: groundingjev/model.py and docs/architecture.md.
This schematic encodes no experimental measurements.
"""

import argparse
from pathlib import Path


COLORS = {
    "dark": "#222222", "muted": "#6B7280", "grid": "#D8DEE6",
    "base": "#C7CDD4", "base_edge": "#8F98A3",
    "method": "#2F6F9F", "method_edge": "#254B63",
    "gain": "#D55E00", "light": "#F6F8FA",
}


def render(root):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "jag-architecture",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(7.0, 2.15))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.025, top=0.975)
    ax.set(xlim=(0, 7), ylim=(0, 2.15))
    ax.axis("off")

    ax.text(0.02, 2.03, "Jag", weight="bold", fontsize=9.0,
            color=COLORS["dark"], ha="left", va="center")
    ax.text(6.98, 2.03, "One forward pass", fontsize=8.0,
            color=COLORS["muted"], ha="right", va="center")

    bottom, height, middle = 0.77, 0.96, 1.25
    boxes = [
        (0.02, 1.16, "Image +\nexpression", "Input", False),
        (1.45, 1.55, "Qwen3.5-0.8B", "Multimodal backbone", True),
        (3.27, 0.95, "Last valid\ntoken", "Hidden state", False),
        (4.49, 1.14, "Box head", "MLP + sigmoid", True),
        (5.90, 1.08, "Bounding box", "Original-image xyxy", True),
    ]
    for x, width, title, subtitle, primary in boxes:
        ax.add_patch(Rectangle(
            (x, bottom), width, height, linewidth=0.75,
            edgecolor=COLORS["method_edge"] if primary else COLORS["base_edge"],
            facecolor=COLORS["light"] if primary else "white",
        ))
        ax.text(x + width / 2, 1.40, title, fontsize=7.8, weight="bold",
                color=COLORS["method"] if primary else COLORS["dark"],
                ha="center", va="center", linespacing=1.18)
        ax.text(x + width / 2, 1.02, subtitle, fontsize=6.6,
                color=COLORS["muted"], ha="center", va="center")
    for current, following in zip(boxes, boxes[1:]):
        left = current[0] + current[1]
        right = following[0]
        ax.add_patch(FancyArrowPatch(
            (left + 0.025, middle), (right - 0.025, middle),
            arrowstyle="-|>", mutation_scale=7.5, linewidth=0.75,
            color=COLORS["method_edge"], shrinkA=0, shrinkB=0,
        ))

    ax.text(5.06, 0.59, r"Normalized $(c_x,c_y,w,h)$", fontsize=6.5,
            color=COLORS["muted"], ha="center", va="center")
    ax.plot([0.02, 6.98], [0.39, 0.39], color=COLORS["grid"], linewidth=0.55)
    ax.text(0.02, 0.18, "Training", fontsize=7.1, weight="bold",
            color=COLORS["dark"], va="center")
    ax.text(0.69, 0.18, "Head warmup → joint fine-tuning", fontsize=7.1,
            color=COLORS["method"], va="center")
    ax.text(6.98, 0.18, r"Objective: $5\,\mathcal{L}_{\mathrm{L1}} + 2\,\mathcal{L}_{\mathrm{GIoU}}$",
            fontsize=7.2, color=COLORS["dark"], ha="right", va="center")

    output = root / "assets"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        metadata = None
        if extension == "pdf":
            metadata = {"Title": "Jag architecture", "CreationDate": None,
                        "ModDate": None}
        elif extension == "svg":
            metadata = {"Title": "Jag architecture", "Date": None,
                        "Description": "An image and expression enter Qwen3.5-0.8B. "
                        "The last valid hidden state feeds a box regression head, "
                        "producing a bounding box in one forward pass."}
        target = output / f"architecture.{extension}"
        fig.savefig(target, dpi=300,
                    bbox_inches="tight", pad_inches=0.02, metadata=metadata)
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    render(args.root.resolve())


if __name__ == "__main__":
    main()
