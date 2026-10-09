"""EvalScope model and benchmark adapters for single-box RefCOCO grounding."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from evalscope import TaskConfig, run_task
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.model import ModelAPI, ModelOutput
from evalscope.api.registry import register_benchmark, register_model_api
from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS, MetricEntry

from .data import RefCOCODataset, referring_expression, target_cxcywh


BENCHMARK_NAME = "groundingjev_refcoco"
THRESHOLDS = (0.5, 0.75, 0.9)
ACCURACY_KEYS = {0.5: "acc_05", 0.75: "acc_075", 0.9: "acc_09"}

# Declare our additional output-status ratios in EvalScope's metric catalog;
# they are diagnostics, never candidates for the benchmark's primary metric.
for _name, _label in (("invalid_output_rate", "Invalid output rate"),
                      ("clip_rate", "Clipped box rate"), ("degenerate_rate", "Degenerate box rate")):
    METRIC_DEFINITIONS.setdefault(_name, MetricEntry(
        baseline="diagnostic.parse_status.ratio", metric_name=_label, display_name=_label))


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A prediction must contain exactly one four-coordinate box")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("Box coordinates must be numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Box coordinates must be finite")
    if result[2] < result[0] or result[3] < result[1]:
        raise ValueError("Reversed box edges are invalid")
    return result


def score_prediction(prediction, target_xyxy):
    """All inputs produce a score; malformed predictions count as zero-IoU failures."""
    target = _box(target_xyxy)
    values = {"iou": 0.0, **{key: 0.0 for key in ACCURACY_KEYS.values()},
              "invalid_output_rate": 0.0, "clip_rate": 0.0, "degenerate_rate": 0.0}
    try:
        prediction = json.loads(prediction) if isinstance(prediction, str) else prediction
        if not isinstance(prediction, dict) or prediction.get("error"):
            raise ValueError("Prediction failed or is not a structured box result")
        raw = _box(prediction.get("raw_bbox_xyxy_normalized", prediction.get("bbox_xyxy_normalized")))
        clipped = [min(1.0, max(0.0, coordinate)) for coordinate in raw]
        values["clip_rate"] = float(raw != clipped)
        values["degenerate_rate"] = float(clipped[2] <= clipped[0] or clipped[3] <= clipped[1])
        intersection = (max(0.0, min(clipped[2], target[2]) - max(clipped[0], target[0]))
                        * max(0.0, min(clipped[3], target[3]) - max(clipped[1], target[1])))
        area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
        union = area + target_area - intersection
        values["iou"] = intersection / union if union > 0 else 0.0
        for threshold, key in ACCURACY_KEYS.items():
            values[key] = float(values["iou"] >= threshold)
    except (ValueError, TypeError, KeyError, OverflowError):
        values["invalid_output_rate"] = 1.0
    return values


@register_model_api("groundingjev")
class GroundingJevAPI(ModelAPI):
    """EvalScope's `generate` interface performs a normal model forward only."""

    def __init__(self, model_name="groundingjev", predictor=None, checkpoint=None,
                 processor_path=None, device="cuda", max_pixels=262144, max_length=2048, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        if predictor is None:
            from .predict import GroundingPredictor
            predictor = GroundingPredictor.from_checkpoint(
                checkpoint or model_name, processor_path=processor_path, device=device,
                max_pixels=max_pixels, max_length=max_length)
        self.predictor = predictor

    def generate(self, input, tools=None, tool_choice=None, config=None):
        try:
            messages = [message for message in input if message.role == "user"]
            if len(messages) != 1:
                raise ValueError("Expected one grounding request")
            request = json.loads(messages[0].text)
            if set(request) != {"image", "expression"}:
                raise ValueError("Grounding model input must contain only image and expression")
            result = self.predictor.predict(request["image"], request["expression"])
            content = json.dumps(result, ensure_ascii=False, allow_nan=False)
        except Exception as error:
            # Keep this as a normal completion. ModelOutput.error/Score error status
            # would let a framework skip a failed prediction and change the denominator.
            content = json.dumps({"error": f"{type(error).__name__}: {error}"}, ensure_ascii=False)
        return ModelOutput.from_content(model=self.model_name, content=content)


@register_benchmark(BenchmarkMeta(
    name=BENCHMARK_NAME,
    pretty_name="Jag RefCOCO",
    description="One image and referring expression to a continuous bounding box.",
    dataset_id="local_refcoco_jsonl",
    subset_list=["test"],
    eval_split="test",
    few_shot_num=0,
    few_shot_mode="disabled",
    metric_list=["iou", "accuracy"],
    primary_metric=MetricSelector(name="accuracy", dimensions={"threshold": 0.5}),
    extra_params={"jsonl": None, "subset": "test"},
))
class GroundingRefCOCOAdapter(DefaultDataAdapter):
    def load(self):
        path = self.extra_params.get("jsonl")
        if not path:
            raise ValueError("dataset_args.groundingjev_refcoco.extra_params.jsonl is required")
        source = RefCOCODataset(path)
        count = len(source)
        if self.limit is not None:
            count = min(count, int(count * self.limit) if isinstance(self.limit, float) else int(self.limit))
        if count < 1:
            raise ValueError("The selected evaluation dataset has no records")
        subset = self.extra_params.get("subset", "test")
        samples = []
        for index in range(count):
            record = source[index]
            record["_groundingjev_sample_index"] = index
            samples.append(self.record_to_sample(record))
        return DatasetDict({subset: MemoryDataset(samples, name=subset, location=str(source.path))}), None

    def record_to_sample(self, record):
        if len(record.get("images", [])) != 1:
            raise ValueError("Evaluation records require exactly one source image")
        image = Path(record["images"][0])
        if not image.is_absolute():
            image = Path(record.get("_groundingjev_source_dir", ".")) / image
        expression = referring_expression(record)
        target_cxcywh(record)  # Validate the annotation without a floating-point round trip.
        target = [float(value) / 1000.0 for value in record["solution"]["arguments"]["coordinate"]]
        image_size, image_error = None, None
        try:
            with Image.open(image) as source:
                image_size = list(source.size)
        except (OSError, ValueError) as error:
            # Let inference yield a scored failure instead of dropping a missing image.
            image_error = f"{type(error).__name__}: {error}"
        request = json.dumps({"image": str(image), "expression": expression}, ensure_ascii=False)
        return Sample(
            id=record.get("_groundingjev_sample_index"),
            input=[ChatMessageUser(content=request)], target=json.dumps(target),
            metadata={"source_sample_id": record.get("sample_id"), "image": str(image),
                      "image_size": image_size, "image_error": image_error,
                      "bbox_xyxy_normalized": target},
        )

    def extract_answer(self, prediction, task_state):
        return prediction

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        values = score_prediction(original_prediction, task_state.metadata["bbox_xyxy_normalized"])
        return Score(value=values, prediction=original_prediction,
                     extracted_prediction=filtered_prediction, main_score_name="acc_05")

    def aggregate_scores(self, sample_scores):
        if not sample_scores:
            raise ValueError("Cannot aggregate an empty evaluation")
        count = len(sample_scores)
        def mean(key):
            return sum(float(sample.score.value.get(key, 1.0 if key == "invalid_output_rate" else 0.0))
                       for sample in sample_scores) / count
        metrics = [AggScore(metric_name="iou", aggregation="mean", score=mean("iou"), num=count)]
        metrics.extend(AggScore(metric_name="accuracy", aggregation="mean", dimensions={"threshold": threshold},
                                score=mean(key), num=count) for threshold, key in ACCURACY_KEYS.items())
        metrics.extend(AggScore(metric_name=key, aggregation="mean", score=mean(key), num=count)
                       for key in ("invalid_output_rate", "clip_rate", "degenerate_rate"))
        return metrics


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(checkpoint, jsonl, max_pixels, max_length):
    checkpoint = Path(checkpoint).resolve(strict=True)
    files = sorted(checkpoint.glob("*.safetensors"))
    files += [checkpoint / name for name in ("config.json", "preprocessor_config.json", "processor_config.json",
                                            "tokenizer_config.json", "tokenizer.json", "chat_template.jinja")
              if (checkpoint / name).exists()]
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Checkpoint contains no complete safetensors model weights")
    manifest = {"checkpoint": str(checkpoint), "files": {path.name: file_sha256(path) for path in files},
                "jsonl": str(Path(jsonl).resolve(strict=True)), "jsonl_sha256": file_sha256(jsonl),
                "max_pixels": max_pixels, "max_length": max_length,
                "input_template": "Locate the object described by: {expression}",
                "enable_thinking": False, "coordinate_format": "normalized_cxcywh",
                "result_cache_enabled": False}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def run_evaluation(jsonl, model_api, output_dir, subset="test", limit=None):
    """Enter the actual EvalScope scheduling, scoring, aggregation and report path."""
    task = TaskConfig(
        model=model_api, model_id=model_api.model_name,
        datasets=[BENCHMARK_NAME],
        dataset_args={BENCHMARK_NAME: {"subset_list": [subset],
                                      "extra_params": {"jsonl": str(Path(jsonl).resolve()), "subset": subset}}},
        eval_batch_size=1, generation_config={"batch_size": 1}, limit=limit,
        work_dir=str(output_dir), no_timestamp=True, use_cache=None,
        ignore_errors=False, collect_perf=False, seed=42,
    )
    return run_task(task)
