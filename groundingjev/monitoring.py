"""Rank-zero SwanLab reporting and measured training-duration estimates."""

from collections import deque
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time

from transformers import TrainerCallback


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def public_config(value):
    if isinstance(value, dict):
        return {str(k): public_config(v) for k, v in value.items()
                if not any(s in str(k).lower() for s in ("api_key", "password", "secret", "credential", "hub_token"))}
    if isinstance(value, (list, tuple)):
        return [public_config(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def read_swanlab_api_key():
    """Resolve an opt-in credential without persisting it in the run config."""
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    key_file = os.environ.get("SWANLAB_API_KEY_FILE")
    if not key and key_file:
        key = Path(key_file).read_text().strip()
    if not key:
        raise RuntimeError("Set SWANLAB_API_KEY or SWANLAB_API_KEY_FILE when enabling --swanlab")
    return key


class MonitoringSession:
    """One cloud experiment across both stages; credentials never enter config."""

    def __init__(self, output_dir, run_config, enabled=False, project="Jag",
                 experiment_name=None, calibration_path=None):
        self.output_dir = Path(output_dir)
        self.config = public_config(run_config)
        self.is_main = int(os.environ.get("RANK", "0")) == 0
        self.enabled = bool(enabled) and self.is_main
        self.run = None
        self._secret = None
        self.stages = {}
        self.progress = {}
        self.started = time.monotonic()
        self.calibration = {}
        if calibration_path:
            self.calibration = json.loads(Path(calibration_path).read_text())
        if not self.is_main:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            self._connect(project, experiment_name)

    def _connect(self, project, experiment_name):
        import swanlab

        self._secret = read_swanlab_api_key()
        record = self.output_dir / "swanlab-run.json"
        prior = json.loads(record.read_text()) if record.exists() else {}
        try:
            if not swanlab.login(api_key=self._secret, save=False, timeout=30):
                raise RuntimeError("SwanLab authentication did not succeed")
            settings = swanlab.Settings(
                interactive=False,
                core=swanlab.Settings.Core(record_interval=1.5),
                probe=swanlab.Settings.Probe(git=False, monitor=True, monitor_interval=10),
            )
            self.run = swanlab.init(
                mode="online", project=project, public=False,
                name=experiment_name or self.output_dir.name,
                log_dir=str(self.output_dir / "swanlog"),
                config=self.config, settings=settings,
                id=prior.get("id"), resume="must" if prior.get("id") else "never",
            )
            atomic_json(record, {"id": self.run.id, "url": self.run.url, "path": self.run.path,
                                 "project": project, "mode": "online"})
            print(f"SwanLab live dashboard: {self.run.url}", flush=True)
        except Exception as error:
            detail = str(error).replace(self._secret, "[redacted]")
            raise RuntimeError(f"SwanLab online initialization failed: {detail}") from None

    def callbacks(self, stage, step_offset=0, future_joint_steps=0):
        if not self.is_main:
            return []
        return [TrainingProgressCallback(self, stage, step_offset, future_joint_steps)]

    def calibrated_seconds_per_step(self, stage, effective_batch):
        record = self.calibration.get("stages", {}).get(stage, {})
        duration = record.get("seconds_per_step")
        prior_batch = record.get("effective_batch_size", effective_batch)
        if duration is None or not prior_batch:
            return None
        return float(duration) * effective_batch / prior_batch

    def record(self, progress, metrics, global_step):
        self.progress = progress
        atomic_json(self.output_dir / "progress.json", progress)
        with (self.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"timestamp": progress["timestamp"], "step": global_step,
                                     **metrics}, allow_nan=False) + "\n")
        if self.run is not None:
            self.run.log(metrics, step=global_step)

    def finish(self, status="success", error=None):
        if not self.is_main:
            return
        if error is not None and self._secret:
            error = str(error).replace(self._secret, "[redacted]")
        self.progress.update(status="completed" if status == "success" else "failed",
                             finished_at=datetime.now(timezone.utc).isoformat(), error=error)
        atomic_json(self.output_dir / "progress.json", self.progress)
        atomic_json(self.output_dir / "calibration.json", {
            "stages": self.stages, "source": "measured_optimizer_steps",
            "wall_seconds": time.monotonic() - self.started,
        })
        if self.run is not None:
            self.run.finish(state="success" if status == "success" else "crashed", error=error)


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, session, stage, step_offset=0, future_joint_steps=0):
        self.session = session
        self.stage = stage
        self.step_offset = int(step_offset)
        self.future_joint_steps = int(future_joint_steps)
        self.durations = deque(maxlen=50)
        self.last_time = None
        self.last_step = None
        self.effective_batch = None
        self.latest = {}
        self.stage_start = None
        self.observed_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
        self.last_time = self.stage_start = time.monotonic()
        self.last_step = state.global_step
        self.latest = self._progress(state)
        self.session.record(self.latest, {f"{self.stage}/planned_steps": state.max_steps},
                            self.step_offset + state.global_step)

    def _progress(self, state):
        duration = statistics.median(self.durations) if self.durations else None
        if duration is None:
            duration = self.session.calibrated_seconds_per_step(self.stage, self.effective_batch)
        current_remaining = (state.max_steps - state.global_step) * duration if duration is not None else None
        future_remaining = 0.0
        if self.future_joint_steps:
            future_rate = self.session.calibrated_seconds_per_step("joint", self.effective_batch)
            future_remaining = self.future_joint_steps * future_rate if future_rate is not None else None
        total_remaining = (current_remaining + future_remaining
                           if current_remaining is not None and future_remaining is not None else None)
        now = datetime.now(timezone.utc)
        return {
            "status": "running", "timestamp": now.isoformat(), "stage": self.stage,
            "stage_step": state.global_step, "stage_total_steps": state.max_steps,
            "global_step": self.step_offset + state.global_step, "epoch": state.epoch,
            "effective_batch_size": self.effective_batch,
            "seconds_per_step": duration,
            "samples_per_second": self.effective_batch / duration if duration else None,
            "stage_remaining_seconds": current_remaining,
            "eta_seconds": total_remaining,
            "estimated_finish_utc": (now + timedelta(seconds=total_remaining)).isoformat()
            if total_remaining is not None else None,
            "estimate_basis": "rolling_median_measured_steps" if self.durations else "preflight_calibration",
            "observed_steps": self.observed_steps,
            "swanlab_url": self.session.run.url if self.session.run is not None else None,
        }

    def on_step_end(self, args, state, control, **kwargs):
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        now = time.monotonic()
        steps = state.global_step - self.last_step
        if steps > 0:
            self.observed_steps += steps
            # The first optimizer step includes lazy kernel initialization.
            if self.observed_steps > 1:
                self.durations.append((now - self.last_time) / steps)
        self.last_step, self.last_time = state.global_step, now
        self.latest = self._progress(state)
        metrics = {f"{self.stage}/step": state.global_step,
                   f"{self.stage}/epoch": float(state.epoch or 0)}
        for field in ("seconds_per_step", "samples_per_second", "eta_seconds", "stage_remaining_seconds"):
            if self.latest[field] is not None:
                metrics[f"timing/{field}"] = self.latest[field]
        if torch.cuda.is_initialized():
            metrics["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        self.session.record(self.latest, metrics, self.step_offset + state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics = {}
        for name, value in (logs or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                metrics[f"{self.stage}/{name}"] = value
        if metrics:
            self.session.record(self._progress(state), metrics, self.step_offset + state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        measured = self._progress(state)
        self.session.stages[self.stage] = {
            "seconds_per_step": measured["seconds_per_step"],
            "effective_batch_size": self.effective_batch,
            "observed_steps": self.observed_steps,
            "wall_seconds": time.monotonic() - self.stage_start,
            "completed_steps": state.global_step,
        }
        atomic_json(self.session.output_dir / "calibration.json", {
            "stages": self.session.stages, "source": "measured_optimizer_steps"})
