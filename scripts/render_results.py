#!/usr/bin/env python3
"""Render the public tables and figure from evaluation/results.json."""

import argparse
import json
import math
from pathlib import Path


LABELS = ("Qwen3.5-0.8B", "Jag")
DATASETS = {"refcoco": "RefCOCO", "refcocop": "RefCOCO+", "refcocog": "RefCOCOg"}
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def percentage(value):
    return f"{value * 100:.2f}"


def batch_description(efficiency, chinese=False, compact=False):
    """Describe recorded request/effective batches without inventing a shared size."""
    requested, observed = [], []
    for label in LABELS:
        model = efficiency["models"][label]
        name = "Base" if compact and label == "Qwen3.5-0.8B" else label
        size = efficiency.get("batch_sizes", {}).get(label)
        if size is None:
            size = model.get("requested_batch_size", model.get("batch_size", efficiency.get("batch_size")))
        requested.append(f"{name}={size if size is not None else ('未记录' if chinese else 'unrecorded')}")
        actual = model.get("effective_inference_batch_sizes")
        if actual is not None:
            known = sorted({size for size in actual if type(size) is int and size > 0})
            value = str(known[0]) if len(known) == 1 else f"{known[0]}–{known[-1]}" if known else "?"
            if known and any(size is None for size in actual):
                value += "+?"
            observed.append(f"{name}={value}")
    text = ("请求 batch：" if chinese else "Requested batch: ") + ", ".join(requested)
    if observed:
        text += ("；实际 batch：" if chinese else "; effective batch: ") + ", ".join(observed)
    return text


def measured_number(value, decimals=2):
    return "—" if value is None else f"{value:.{decimals}f}"


def dataset_rows(result):
    """Pool split statistics, never average two rounded split scores equally."""
    grouped = {}
    for row in result["quality"]["rows"]:
        grouped.setdefault(row["dataset"], []).append(row)
    output = []
    for dataset, rows in grouped.items():
        total = sum(row["samples"] for row in rows)
        if total <= 0:
            raise ValueError("Dataset summaries require a positive sample count")
        models = {}
        for label in LABELS:
            if not all(label in row["models"] for row in rows):
                raise ValueError(f"Missing {label} split in {dataset}")
            values = [row["models"][label] for row in rows]
            if len(rows) == 1:
                models[label] = {key: values[0][key] for key in ("miou", "iou_at_05")}
            else:
                if not all("iou_sum" in value and "iou_at_05_count" in value for value in values):
                    raise ValueError("Merged splits require original IoU sums and hit counts")
                for row, value in zip(rows, values):
                    if not (0 <= value["iou_sum"] <= row["samples"] and
                            type(value["iou_at_05_count"]) is int and
                            0 <= value["iou_at_05_count"] <= row["samples"]):
                        raise ValueError("Invalid original-sample statistics")
                models[label] = {
                    "miou": math.fsum(value["iou_sum"] for value in values) / total,
                    "iou_at_05": sum(value["iou_at_05_count"] for value in values) / total,
                }
        output.append({"dataset": dataset, "samples": total,
                       "splits": [row["split"] for row in rows], "models": models})
    return output


def inference_speedup(efficiency):
    if efficiency["kind"] != "isolated_prediction" or efficiency["status"] != "complete":
        return None
    baseline = efficiency["models"][LABELS[0]].get("mean_ms")
    grounding = efficiency["models"][LABELS[1]].get("mean_ms")
    if any(value is None or not math.isfinite(value) or value <= 0 for value in (baseline, grounding)):
        return None
    return baseline / grounding


def metric_table(rows, metric, chinese=False, all_splits=False):
    title = "mIoU" if metric == "miou" else "Acc@0.5"
    heading = "测试划分" if chinese and all_splits else "数据集" if chinese else "Split" if all_splits else "Dataset"
    gain = "提升（百分点）" if chinese else "Gain (points)"
    count = " | 样本数" if chinese else " | Samples"
    lines = [f"### {title} ↑ (%)", "",
             f"| {heading}" + (count if all_splits else "") + f" | {LABELS[0]} | **Jag** | {gain} |",
             "| :---" + (" | ---:" if all_splits else "") + " | ---: | ---: | ---: |"]
    for row in rows:
        base, model = (row["models"][label][metric] for label in LABELS)
        baseline, proposed = percentage(base), percentage(model)
        if base > model:
            baseline = f"**{baseline}**"
        else:
            proposed = f"**{proposed}**"
        label = DATASETS[row["dataset"]] + (f" {row['split']}" if all_splits else "")
        cells = [label] + ([f"{row['samples']:,}"] if all_splits else [])
        cells += [baseline, proposed, f"{(model - base) * 100:+.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def tables(result, chinese=False, all_splits=False):
    quality = result["quality"]
    efficiency = result["efficiency"]
    provisional = quality["status"] != "complete"
    lines = [
        ("**暂定结果，完整评测后更新。**" if chinese else
         "**Provisional results; to be updated after full evaluation.**")
        if provisional else
        ("完整测试集结果。" if chinese else "Full test-set results."), "",
    ]
    rows = quality["rows"] if all_splits else dataset_rows(result)
    if not all_splits:
        lines += [("RefCOCO 与 RefCOCO+ 按全部样本合并 testA、testB；RefCOCOg 使用 test。" if chinese else
                   "RefCOCO and RefCOCO+ pool all testA and testB samples; RefCOCOg uses test."), ""]
    for metric in ("miou", "iou_at_05"):
        lines += metric_table(rows, metric, chinese, all_splits) + [""]
    formal = efficiency["kind"] == "isolated_prediction" and efficiency["status"] == "complete"
    if formal:
        heading = "推理性能" if chinese else "Inference performance"
        latency = "延时" if chinese else "Latency"
        throughput = "吞吐" if chinese else "Throughput"
        note = ("端到端推理耗时，不含模型加载与预热。" if chinese else
                "End-to-end prediction time, excluding model loading and warmup.")
    else:
        heading = "评估流程效率（暂定）" if chinese else "Evaluation workflow efficiency (provisional)"
        latency = "每样本均摊评估耗时" if chinese else "Amortized evaluation time"
        throughput = "评估流程吞吐" if chinese else "Evaluation throughput"
        note = ("暂定流程耗时；正式推理性能结果待更新。" if chinese else
                "Provisional workflow timings; inference performance results are pending.")
    ratio = inference_speedup(efficiency)
    speedup_header = "加速比 ↑" if chinese else "Speedup ↑"
    lines += ["### " + heading, "",
              (f"| 模型 | {latency} ↓ (ms) | {throughput} ↑ (样本/s) | {speedup_header} |" if chinese else
               f"| Model | {latency} ↓ (ms) | {throughput} ↑ (samples/s) | {speedup_header} |"),
              "| :--- | ---: | ---: | ---: |"]
    for label in LABELS:
        value = efficiency["models"][label]
        multiplier = "—" if ratio is None else "1.00×" if label == LABELS[0] else f"**{ratio:.2f}×**"
        delay = measured_number(value.get("mean_ms"))
        rate = measured_number(value.get("samples_per_second"), 3)
        if label == LABELS[1] and ratio is not None and ratio > 1:
            delay, rate = f"**{delay}**", f"**{rate}**" if rate != "—" else rate
        lines.append(f"| {label} | {delay} | {rate} | {multiplier} |")
    if ratio is not None:
        reduction = (1 - 1 / ratio) * 100
        lines += ["", (f"**推理加速 {ratio:.2f}×，平均延时降低 {reduction:.2f}%。**" if chinese else
                       f"**{ratio:.2f}× inference speedup, with {reduction:.2f}% lower mean latency.**")]
    if any(efficiency["models"][label].get("mean_ms") is None for label in LABELS):
        lines += ["", ("“—”表示完整耗时暂不可用。" if chinese else
                       "“—” means complete timing is unavailable.")]
    lines += ["", note, "",
              batch_description(efficiency, chinese=chinese)]
    return "\n".join(lines)


def update_block(path, content, start=START, end=END):
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one results block in {path.name}")
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    path.write_text(left + start + "\n" + content + "\n" + end + right, encoding="utf-8")


def figure(result, root):
    """Render pooled results in the serif, gray/blue/orange paper style."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"dark": "#222222", "muted": "#6B7280", "grid": "#D8DEE6",
              "base": "#C7CDD4", "base_edge": "#8F98A3", "method": "#2F6F9F",
              "method_edge": "#254B63", "gain": "#D55E00"}
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "jag-paper-figures",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.linewidth": .6, "xtick.major.width": .5, "ytick.major.width": .5,
        "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.7,
        "ytick.labelsize": 7.0, "text.color": colors["dark"],
        "axes.labelcolor": colors["dark"], "axes.edgecolor": colors["base_edge"],
    })
    rows = dataset_rows(result)
    efficiency = result["efficiency"]
    formal = efficiency["kind"] == "isolated_prediction" and efficiency["status"] == "complete"
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.90), gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.125, right=.965, bottom=.15, top=.86, wspace=.64, hspace=.80)
    for axis in axes.flat:
        axis.set_axisbelow(True)
        axis.grid(axis="x", color=colors["grid"], linewidth=.45)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_linewidth(.55)
        axis.tick_params(direction="out", length=2.3, pad=2)
        axis.tick_params(axis="y", length=0)

    for panel, metric in enumerate(("miou", "iou_at_05")):
        axis = axes[0, panel]
        for index, row in enumerate(rows):
            base, method = [row["models"][label][metric] * 100 for label in LABELS]
            y = len(rows) - index - 1
            axis.plot([base, method], [y, y], color=colors["base_edge"], linewidth=1.3, zorder=2)
            axis.scatter([base], [y], s=25, color=colors["base"], edgecolor=colors["base_edge"], linewidth=.6, zorder=3)
            axis.scatter([method], [y], s=25, color=colors["method"], edgecolor=colors["method_edge"], linewidth=.6, zorder=3)
            axis.annotate(f"{base:.2f}", (base, y), xytext=(0, -9), textcoords="offset points",
                          ha="center", color=colors["muted"], fontsize=6.1)
            axis.annotate(f"{method:.2f}", (method, y), xytext=(0, 6), textcoords="offset points",
                          ha="center", color=colors["method"], fontsize=6.1)
            axis.text(1.02, y, f"{method - base:+.2f}", transform=axis.get_yaxis_transform(),
                      ha="left", va="center", fontsize=6.4, color=colors["gain"])
        axis.text(1.02, 1.06, "Δ (points)", transform=axis.transAxes, color=colors["gain"], fontsize=6.4)
        axis.set_yticks(range(len(rows)), [DATASETS[row["dataset"]] for row in reversed(rows)])
        axis.set_ylim(-.65, len(rows) - .40)
        axis.set_xlim(55, 100)
        axis.set_xticks([60, 70, 80, 90, 100])
        metric_name = "mIoU" if metric == "miou" else "Acc@0.5"
        axis.set_xlabel(f"{metric_name} (%) ↑", labelpad=3)
        axis.text(0, 1.19, f"({'ab'[panel]}) {metric_name}", transform=axis.transAxes,
                  fontsize=8.4, fontweight="bold", ha="left")

    for panel, metric in enumerate(("mean_ms", "samples_per_second")):
        axis = axes[1, panel]
        values = [efficiency["models"][label].get(metric) for label in LABELS]
        available = [value for value in values if value is not None]
        maximum = max(available or [1])
        for index, value in enumerate(values):
            y = 1 - index
            if value is None:
                axis.text(.05, y, "Unavailable", transform=axis.get_yaxis_transform(),
                          va="center", color=colors["muted"], fontsize=6.5)
                continue
            axis.barh(y, value, height=.45,
                      color=colors["base" if index == 0 else "method"],
                      edgecolor=colors["base_edge" if index == 0 else "method_edge"], linewidth=.5)
            axis.text(value + maximum * .025, y, f"{value:.2f}" if panel == 0 else f"{value:.3f}",
                      va="center", fontsize=6.7, color=colors["dark"])
        axis.set_xlim(0, maximum * 1.34)
        axis.set_ylim(-.52, 1.66)
        axis.set_yticks([1, 0], ["Base", "Jag"])
        title = ("Prediction latency" if panel == 0 else "Prediction throughput") if formal else (
            "Workflow time" if panel == 0 else "Workflow throughput")
        axis.text(0, 1.16, f"({'cd'[panel]}) {title}", transform=axis.transAxes,
                  fontsize=8.4, fontweight="bold", ha="left")
        axis.set_xlabel("Mean latency (ms) ↓" if panel == 0 else "Samples / s ↑", labelpad=3)
        if (ratio := inference_speedup(efficiency)) is not None:
            note = f"{ratio:.2f}× speedup" if panel == 0 else f"{ratio:.2f}× throughput"
            axis.text(.98, .89, note, transform=axis.transAxes, ha="right", va="center",
                      fontsize=7.2, fontweight="bold", color=colors["gain"])

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                      markerfacecolor=colors[fill], markeredgecolor=colors[edge],
                      markeredgewidth=.6, label=name)
               for fill, edge, name in [("base", "base_edge", "Base"),
                                        ("method", "method_edge", "Jag")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.035),
               ncol=2, frameon=False, fontsize=7.3, handletextpad=.3, columnspacing=1.5)
    caption = ("testA + testB pooled by sample; RefCOCOg: test. "
               "Prediction excludes loading and warmup." if formal else
               "Provisional workflow timings; prediction benchmark pending.")
    fig.text(.125, .027, caption, fontsize=6.2, color=colors["muted"])
    fig.text(.125, -.005, batch_description(efficiency, compact=True), fontsize=5.9, color=colors["muted"])
    output = root / "assets/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        target = output / f"evaluation-results.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Date": None} if extension == "svg" else None
        fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=.02, metadata=metadata)
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)
    source = root / "evaluation/figure-source-values.csv"
    source.parent.mkdir(exist_ok=True)
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["dataset", "split", "samples", "model", "miou", "iou_at_05", "latency_ms", "samples_per_second"])
        for row in rows:
            for label in LABELS:
                value = row["models"][label]
                writer.writerow([row["dataset"], "+".join(row["splits"]), row["samples"], label,
                                 repr(value["miou"]), repr(value["iou_at_05"]), "", ""])
        for label in LABELS:
            value = efficiency["models"][label]
            writer.writerow([efficiency.get("dataset", ""), efficiency.get("split", ""),
                             efficiency.get("samples", ""), label, "", "",
                             value.get("mean_ms"), value.get("samples_per_second")])


def main():
    parser = argparse.ArgumentParser(description="Render figures from the public evaluation values")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "evaluation/results.json").read_text(encoding="utf-8"))
    figure(result, root)


if __name__ == "__main__":
    main()
