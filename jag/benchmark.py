"""Measure paired grounding latency on one idle GPU, using fixed test samples."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import threading

import torch

from .base_predict import BaseGroundingPredictor
from .data import RefCOCODataset, referring_expression, target_cxcywh
from .evaluate_base import base_checkpoint_manifest
from .evaluation import checkpoint_manifest, file_sha256, score_prediction
from .predict import GroundingPredictor
from .resources import configure_cuda_memory


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def selection_manifest(total, samples=256, warmup=10, seed=42):
    """Select once, before inference; warmup and measured rows are disjoint."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (total, samples, warmup, seed)):
        raise ValueError("Sample counts and seed must be integers")
    if samples < 1 or warmup < 1 or total < samples + warmup:
        raise ValueError("Need positive samples and warmup, and enough distinct source records")
    selected = random.Random(seed).sample(range(total), samples + warmup)
    payload = {"sample_ids": selected[warmup:], "warmup_sample_ids": selected[:warmup]}
    return {**payload, "sample_selection_sha256": hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def latency_summary(values_ms):
    values = [float(value) for value in values_ms]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Latencies must be finite and positive, without omitted samples")
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {"mean": statistics.fmean(values), "p50": percentile(.5), "p95": percentile(.95),
            "total_seconds": sum(values) / 1000.0}


class TimedModel:
    """Time the actual forward or generate call, leaving predictor inputs intact."""

    def __init__(self, model, synchronize, clock=time.perf_counter):
        self.model = model
        self.synchronize = synchronize
        self.clock = clock
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _invoke(self, operation, function, *args, **kwargs):
        self.synchronize()
        started = self.clock()
        result = function(*args, **kwargs)
        self.synchronize()
        self.calls.append({"operation": operation, "milliseconds": (self.clock() - started) * 1000.0})
        return result

    def __call__(self, *args, **kwargs):
        return self._invoke("forward", self.model, *args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._invoke("generate", self.model.generate, *args, **kwargs)


def timed_prediction(predictor, sample, model_key, synchronize, clock=time.perf_counter):
    """Runtime exceptions abort the benchmark; validly generated bad boxes remain."""
    wrapper = predictor.model
    if not isinstance(wrapper, TimedModel):
        raise TypeError("The predictor model must be instrumented with TimedModel")
    wrapper.calls.clear()
    synchronize()
    started = clock()
    # Only these two public inference inputs cross the model boundary.
    prediction = predictor.predict(sample["image"], sample["expression"])
    synchronize()
    elapsed = (clock() - started) * 1000.0
    expected_operation = "generate" if model_key == "base" else "forward"
    if len(wrapper.calls) != 1 or wrapper.calls[0]["operation"] != expected_operation:
        raise RuntimeError(f"Expected one timed {expected_operation} call")
    if not isinstance(prediction, dict):
        raise RuntimeError("Predictor returned no structured result")
    # Base predictor returns parse failures with the already produced text and
    # token IDs. Infrastructure/model failures must never appear as fast samples.
    if prediction.get("error") and (model_key != "base" or "raw_generation" not in prediction
                                     or "generated_token_ids" not in prediction):
        raise RuntimeError(f"Unexpected inference failure: {prediction.get('error')}")
    model_ms = wrapper.calls[0]["milliseconds"]
    latency_summary([elapsed, model_ms])
    if model_ms > elapsed:
        raise RuntimeError("Model duration cannot exceed enclosing end-to-end duration")
    return {"end_to_end_ms": elapsed, "model_ms": model_ms, "prediction": prediction,
            "scores": score_prediction(prediction, sample["target_xyxy_normalized"])}


def model_weight_statistics(model):
    elements, dtype_bytes = Counter(), Counter()
    for parameter in model.parameters():
        name = str(parameter.dtype)
        elements[name] += parameter.numel()
        dtype_bytes[name] += parameter.numel() * parameter.element_size()
    return {"parameter_count": sum(elements.values()), "parameter_bytes": sum(dtype_bytes.values()),
            "parameter_dtypes": {name: {"elements": elements[name], "bytes": dtype_bytes[name]}
                                 for name in sorted(elements)},
            "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())}


def summarize_model(records):
    if not records:
        raise ValueError("No measured predictions")
    latency = latency_summary([record["end_to_end_ms"] for record in records])
    model_latency = latency_summary([record["model_ms"] for record in records])
    scores = records[0]["scores"].keys()
    if any(record["scores"].keys() != scores for record in records):
        raise ValueError("Inconsistent score schema")
    quality = {key: statistics.fmean(record["scores"][key] for record in records) for key in scores}
    predictions = [record["prediction"] for record in records]
    generated = [prediction["generated_tokens"] for prediction in predictions if "generated_tokens" in prediction]
    diagnostics = {
        "invalid_output_count": sum(record["scores"]["invalid_output_rate"] > 0 for record in records),
        "zero_iou_count": sum(record["scores"]["iou"] == 0 for record in records),
        "token_limit_count": sum(bool(prediction.get("generation_hit_token_limit")) for prediction in predictions),
        "empty_generation_count": sum("raw_generation" in prediction and not prediction["raw_generation"].strip()
                                      for prediction in predictions),
        "generated_tokens": ({"total": sum(generated), "mean": statistics.fmean(generated),
                              "min": min(generated), "max": max(generated)} if generated else None),
    }
    return {"measured_count": len(records), "latency_ms": latency, "model_latency_ms": model_latency,
            "images_per_second": 1000.0 / latency["mean"], "subset_quality": quality,
            "diagnostics": diagnostics}


def prepare_samples(source, indices):
    result = []
    for index in indices:
        row = source[index]
        if len(row.get("images", [])) != 1:
            raise ValueError(f"Sample {index} must contain exactly one image")
        path = Path(row["images"][0])
        if not path.is_absolute():
            path = Path(row.get("_jag_source_dir", ".")) / path
        target_cxcywh(row)
        result.append({"sample_index": index, "source_sample_id": row.get("sample_id"),
                       "image": str(path.resolve(strict=True)), "image_sha256": file_sha256(path),
                       "expression": referring_expression(row), "target_xyxy_normalized": [
                           float(value) / 1000.0 for value in row["solution"]["arguments"]["coordinate"]]})
    return result


def canonical_gpu_uuid(value):
    """Torch exposes a bare UUID; nvidia-smi requires the GPU-/MIG- prefix."""
    value = str(value).strip()
    if not value:
        raise ValueError("GPU UUID must not be empty")
    return value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value


def discover_gpu(device):
    """Resolve CUDA mapping in a short-lived process so the parent stays idle."""
    if not str(device).startswith("cuda"):
        raise ValueError("The formal performance benchmark requires one CUDA GPU")
    script = (
        "import json,sys,torch; d=torch.device(sys.argv[1]); "
        "p=torch.cuda.get_device_properties(d); "
        "print(json.dumps({'device':str(d),'name':p.name,'uuid':str(p.uuid),"
        "'total_memory_bytes':p.total_memory,'compute_capability':[p.major,p.minor],"
        "'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda}))"
    )
    result = subprocess.run([sys.executable, "-c", script, str(device)], capture_output=True, text=True,
                            check=True, timeout=120)
    gpu = json.loads(result.stdout.strip().splitlines()[-1])
    gpu["uuid"] = canonical_gpu_uuid(gpu["uuid"])
    return gpu


def gpu_processes(uuid):
    result = subprocess.run(
        ["nvidia-smi", "--id=" + canonical_gpu_uuid(uuid), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not value.isdigit() for value in values):
        raise RuntimeError("Cannot audit active compute processes on the benchmark GPU")
    return sorted({int(value) for value in values})


class GpuGuard:
    """Reject competing compute jobs, including jobs appearing during a run."""

    def __init__(self, uuid, query=gpu_processes):
        self.uuid, self.query = uuid, query
        self.allowed_pid = None
        self.observations = []

    def check(self, stage, allow_own_context=False):
        pids = self.query(self.uuid)
        if allow_own_context:
            if self.allowed_pid is not None or len(pids) != 1:
                raise RuntimeError("Cannot uniquely identify the benchmark CUDA process")
            self.allowed_pid = pids[0]
        unexpected = [pid for pid in pids if pid != self.allowed_pid]
        self.observations.append({"stage": stage, "utc": utc_now(), "pids": pids,
                                  "unexpected_pids": unexpected})
        if unexpected:
            raise RuntimeError(f"Competing compute process detected on benchmark GPU: {unexpected}")


class NvmlMemorySampler:
    """Measure process/context and whole-device memory; these are sampled peaks."""
    def __init__(self, uuid, pid, interval_ms):
        import pynvml
        self.nvml = pynvml
        self.uuid, self.pid = uuid, pid
        self.interval = interval_ms/1000
        self.stop_event = threading.Event()
        self.samples, self.error = [], None
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)

    def sample(self):
        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        if sorted(process.pid for process in processes) != [self.pid]:
            raise RuntimeError('Benchmark GPU compute processes changed during NVML sampling')
        process_bytes = processes[0].usedGpuMemory
        unavailable = getattr(self.nvml, 'NVML_VALUE_NOT_AVAILABLE', 2**64-1)
        if process_bytes is None or process_bytes >= unavailable:
            raise RuntimeError('NVML did not expose this process GPU memory')
        self.samples.append({'seconds_from_start': time.perf_counter()-self.started,
                             'process_used_bytes': int(process_bytes),
                             'device_used_bytes': int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)})

    def poll(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as error:
            self.error = error
            self.stop_event.set()

    def __enter__(self):
        self.started = time.perf_counter()
        try:
            self.sample()
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
        except Exception:
            self.nvml.nvmlShutdown()
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.stop_event.set()
        self.thread.join()
        try:
            if self.error is None:
                self.sample()
        finally:
            self.nvml.nvmlShutdown()
        if exception_type is None and self.error is not None:
            raise RuntimeError('NVML sampling failed') from self.error

    def report(self):
        gaps = [right['seconds_from_start']-left['seconds_from_start']
                for left, right in zip(self.samples, self.samples[1:])]
        process_peak = max(sample['process_used_bytes'] for sample in self.samples)
        device_peak = max(sample['device_used_bytes'] for sample in self.samples)
        return {'method': 'NVML periodic sampling', 'scope': 'after_warmup_b1_measurement_window',
                'gpu_uuid': self.uuid, 'host_process_pid': self.pid,
                'requested_interval_ms': self.interval*1000, 'sample_count': len(self.samples),
                'max_observed_interval_ms': max(gaps, default=0)*1000,
                'baseline_process_used_bytes': self.samples[0]['process_used_bytes'],
                'peak_process_used_bytes': process_peak, 'peak_process_used_gib': process_peak/2**30,
                'peak_device_used_bytes': device_peak, 'peak_device_used_gib': device_peak/2**30,
                'peak_is_sampled': True, 'samples': self.samples}


def benchmark_model(model_key, args, warmup_samples, measured_samples, guard):
    guard.check(model_key + ":before_load")
    resources = configure_cuda_memory(args.device, memory_gib=args.gpu_memory_gib)
    if model_key == "base":
        predictor = BaseGroundingPredictor.from_checkpoint(
            args.base_model, device=args.device, max_pixels=args.max_pixels, max_length=args.max_length,
            max_new_tokens=args.max_new_tokens, gpu_memory_gib=args.gpu_memory_gib)
    else:
        predictor = GroundingPredictor.from_checkpoint(
            args.checkpoint, device=args.device, max_pixels=args.max_pixels, max_length=args.max_length,
            use_bf16=True, weight_dtype=args.weight_dtype)
    gc.collect()
    torch.cuda.empty_cache()
    synchronize = lambda: torch.cuda.synchronize(args.device)
    weights = model_weight_statistics(predictor.model)
    predictor.model = TimedModel(predictor.model, synchronize)
    try:
        for sample in warmup_samples:
            timed_prediction(predictor, sample, model_key, synchronize)
        guard.check(model_key + ":after_warmup")
        synchronize()
        baseline_allocated = torch.cuda.memory_allocated(args.device)
        baseline_reserved = torch.cuda.memory_reserved(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        records = []
        with NvmlMemorySampler(guard.uuid, guard.allowed_pid, args.nvml_interval_ms) as sampler:
            for position, sample in enumerate(measured_samples):
                if position % 16 == 0:
                    guard.check(f"{model_key}:sample_{position}")
                if sampler.error is not None:
                    raise RuntimeError("GPU memory sampling failed") from sampler.error
                records.append(timed_prediction(predictor, sample, model_key, synchronize))
        guard.check(model_key + ":after_measurement")
        synchronize()
        summary = summarize_model(records)
        summary.update(
            weights=weights, resource_limits=resources, nvml_memory=sampler.report(),
            autocast_dtype="bfloat16" if model_key == "jag" else None,
            allocated_bytes_after_warmup=baseline_allocated,
            reserved_bytes_after_warmup=baseline_reserved,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
        )
        if model_key == "base":
            summary["resolved_generation_config"] = predictor.generation_config.to_dict()
        return summary, records
    finally:
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        synchronize()


def render_markdown(report):
    base, jag = report["models"]["base"], report["models"]["jag"]
    rows = [
        "# Base 与 Jag 配对性能测评", "",
        f"同一张 {report['gpu']['name']}（{report['gpu']['uuid']}）顺序运行，batch=1。"
        f"固定抽取 {report['dataset']['num_samples']} 条测试样本，"
        f"每个模型先预热 {report['settings']['warmup']} 条独立样本。", "",
        "| 指标 | 原始 Qwen3.5-0.8B | Jag |", "|---|---:|---:|",
    ]
    for title, field, subfield in [
        ("端到端平均延迟（ms）", "latency_ms", "mean"),
        ("端到端 P50（ms）", "latency_ms", "p50"),
        ("端到端 P95（ms）", "latency_ms", "p95"),
        ("模型调用平均延迟（ms）", "model_latency_ms", "mean"),
        ("模型调用 P50（ms）", "model_latency_ms", "p50"),
        ("模型调用 P95（ms）", "model_latency_ms", "p95"),
    ]:
        rows.append(f"| {title} | {base[field][subfield]:.3f} | {jag[field][subfield]:.3f} |")
    if all("nvml_memory" in model for model in (base, jag)):
        rows.append(
            f"| 峰值 GPU 进程显存（GiB） | {base['nvml_memory']['peak_process_used_gib']:.3f} | "
            f"{jag['nvml_memory']['peak_process_used_gib']:.3f} |")
    rows.extend([
        f"| 顺序处理速度（images/s） | {base['images_per_second']:.3f} | {jag['images_per_second']:.3f} |",
        f"| 峰值 allocated 显存（GiB） | {base['peak_allocated_bytes'] / 1024**3:.3f} | {jag['peak_allocated_bytes'] / 1024**3:.3f} |",
        f"| 峰值 reserved 显存（GiB） | {base['peak_reserved_bytes'] / 1024**3:.3f} | {jag['peak_reserved_bytes'] / 1024**3:.3f} |",
        f"| 本子集 mIoU | {base['subset_quality']['iou']:.4f} | {jag['subset_quality']['iou']:.4f} |",
        f"| 本子集 Acc@0.5 | {base['subset_quality']['acc_05']:.2%} | {jag['subset_quality']['acc_05']:.2%} |",
        f"| 无效输出数 | {base['diagnostics']['invalid_output_count']} | {jag['diagnostics']['invalid_output_count']} |",
        "", f"端到端平均延迟比（base ÷ jag）：{report['speedups']['end_to_end_mean']:.3f}×；"
        f"模型调用平均延迟比：{report['speedups']['model_only_mean']:.3f}×。", "",
        "端到端计时包含图片打开、预处理、输入传输、模型调用、解码及框结果处理。"
        "模型计时覆盖 base 的完整 generate 和 jag 的一次 forward。计时边界均同步 CUDA；"
        "权重加载、预热、进程检查及指标计算不计入延迟。速度按 1000 / 平均毫秒计算，表示 batch=1 顺序推理。", "",
        "base 使用 BF16 权重、固定提示词、关闭 thinking、贪心生成，最多 128 个新 token；"
        "Jag 权重精度和 BF16 自动混合精度设置记录在 benchmark.json 中。两者图像预算为 262144、最大输入长度为 2048。"
        "图片在计时前统一校验 SHA，系统文件缓存可能已预热。固定顺序为 base 然后 jag，未进行多轮顺序交叉实验。", "",
        "无效框生成的完整延迟保留在分母中，并作为零 IoU；运行错误使性能测评失败。"
        "这里的 IoU 仅描述固定性能子集；正式质量结论使用完整测试集评测。", "",
        "样本索引、图片 SHA、原始预测、逐样本配对延迟、权重与数据指纹、显存及 GPU 进程审计见 benchmark.json。", "",
    ])
    return "\n".join(rows)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--warmup-jsonl", help="Separate preselected warmup data")
    parser.add_argument("--source-order", action="store_true", help="Evaluate every prepared row in source order")
    parser.add_argument("--weight-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--nvml-interval-ms", type=float, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-gib", type=float, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main():
    args = argument_parser().parse_args()
    if not math.isfinite(args.nvml_interval_ms) or args.nvml_interval_ms <= 0:
        raise ValueError("NVML interval must be positive and finite")
    if args.source_order != bool(args.warmup_jsonl):
        raise ValueError("Use --source-order and --warmup-jsonl together")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("benchmark.json", "benchmark.md", "benchmark_started.json")):
        raise FileExistsError("Benchmark output already contains a run; use a fresh output directory")
    started = utc_now()
    (output / "benchmark_started.json").write_text(json.dumps({"started_at": started, "arguments": vars(args)}, indent=2) + "\n")
    guard = None
    try:
        source = RefCOCODataset(args.jsonl)
        if args.source_order:
            warmup_source = RefCOCODataset(args.warmup_jsonl)
            if not len(source) or not len(warmup_source):
                raise ValueError("Prepared measurement and warmup data must both be nonempty")
            selection = {"sample_ids": list(range(len(source))),
                         "warmup_sample_ids": list(range(len(warmup_source))),
                         "selection": "preselected_source_order",
                         "warmup_jsonl_sha256": file_sha256(args.warmup_jsonl)}
            measured = prepare_samples(source, selection["sample_ids"])
            warmup = prepare_samples(warmup_source, selection["warmup_sample_ids"])
            identity = lambda item: (item["image"], item["expression"],
                                     tuple(item["target_xyxy_normalized"]))
            if {identity(item) for item in measured} & {identity(item) for item in warmup}:
                raise ValueError("Warmup and measured samples overlap")
            args.samples, args.warmup = len(measured), len(warmup)
        else:
            selection = selection_manifest(len(source), args.samples, args.warmup, args.seed)
            measured = prepare_samples(source, selection["sample_ids"])
            warmup = prepare_samples(source, selection["warmup_sample_ids"])
        manifests = {
            "base": base_checkpoint_manifest(args.base_model, args.jsonl, args.max_pixels, args.max_length,
                                             args.max_new_tokens, args.gpu_memory_gib),
            "jag": checkpoint_manifest(args.checkpoint, args.jsonl, args.max_pixels, args.max_length),
        }
        gpu = discover_gpu(args.device)
        guard = GpuGuard(gpu["uuid"])
        guard.check("before_cuda_context")
        configure_cuda_memory(args.device, memory_gib=args.gpu_memory_gib)
        probe = torch.empty(1, device=args.device)
        torch.cuda.synchronize(args.device)
        guard.check("own_cuda_context", allow_own_context=True)
        del probe
        torch.cuda.empty_cache()
        models, paired = {}, [dict(sample) for sample in measured]
        for model_key in ("base", "jag"):
            summary, records = benchmark_model(model_key, args, warmup, measured, guard)
            models[model_key] = {**summary, "manifest": manifests[model_key]}
            for sample, record in zip(paired, records):
                sample[model_key] = record
            # Preserve raw completed work on failure, without publishing success.
            (output / f"{model_key}_measurements.json").write_text(
                json.dumps({"model": model_key, "summary": models[model_key], "records": records},
                           indent=2, allow_nan=False) + "\n")
        guard.check("all_models_finished")
        if file_sha256(args.jsonl) != manifests["base"]["jsonl_sha256"]:
            raise RuntimeError("Dataset changed during the benchmark")
        for sample in measured + warmup:
            if file_sha256(sample["image"]) != sample["image_sha256"]:
                raise RuntimeError("An input image changed during the benchmark")
        report = {
            "schema_version": 1, "status": "completed", "started_at": started, "finished_at": utc_now(),
            "dataset": {"jsonl": str(Path(args.jsonl).resolve()),
                        "jsonl_sha256": manifests["base"]["jsonl_sha256"], "num_source_records": len(source),
                        "num_samples": args.samples, **selection},
            "settings": {"max_pixels": args.max_pixels, "max_length": args.max_length,
                         "max_new_tokens": args.max_new_tokens, "batch_size": 1, "warmup": args.warmup,
                         "seed": args.seed, "jag_weight_dtype": args.weight_dtype, "execution_order": ["base", "jag"],
                         "timing_method": "cuda-synchronized-wall-clock", "model_loading_excluded": True,
                         "warmup_excluded": True, "e2e_includes_preprocessing": True,
                         "image_files_prehashed_before_timing": True, "gpu_guard_interval_samples": 16,
                         "percentile_method": "linear interpolation at p*(n-1)"},
            "gpu": gpu,
            "conditions": {"serial": True, "same_gpu": True, "other_gpu_processes_detected": False,
                           "allowed_gpu_pid": guard.allowed_pid, "observations": guard.observations},
            "models": models, "paired_samples": paired,
            "speedups": {
                "end_to_end_mean": models["base"]["latency_ms"]["mean"] / models["jag"]["latency_ms"]["mean"],
                "model_only_mean": models["base"]["model_latency_ms"]["mean"] / models["jag"]["model_latency_ms"]["mean"],
            },
        }
        (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output / "benchmark.md").write_text(render_markdown(report))
        print(json.dumps({"status": "completed", "output": str(output.resolve())}))
    except Exception as error:
        failure = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "started_at": started, "finished_at": utc_now(),
                   "gpu_observations": guard.observations if guard else []}
        (output / "benchmark_failed.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
