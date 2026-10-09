#!/usr/bin/env python3
"""Two-stage Jag training with the ModelScope ms-swift Trainer."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import AutoProcessor, set_seed
from transformers.trainer_utils import get_last_checkpoint

from jag.data import GroundingCollator, RefCOCODataset
from jag.model import JagModel
from jag.recipes import load_recipe, recipe_fingerprint, stage_schedule, validate_recipe
from jag.trainer import (
    GroundingTrainer, attach_swift_metadata, make_training_arguments, write_json,
)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Complete training JSON; explicit CLI arguments override its training section")
    p.add_argument("--model", default="/models/Qwen3.5-0.8B")
    p.add_argument("--train-jsonl", default="/workspace/datasets/RefCOCO/annotations/refcoco_train.jsonl")
    p.add_argument("--output-dir", default="/outputs/jag")
    p.add_argument("--stage", choices=["all", "head", "joint"], default="all")
    p.add_argument("--init-checkpoint", help="Weights-only stage initialization; never restores an optimizer")
    p.add_argument("--resume-from-checkpoint", help="Exact stage checkpoint path, or auto")
    p.add_argument("--microbatch", type=int, choices=[1, 2, 4], default=2)
    p.add_argument("--effective-batch-size", type=int, default=96)
    p.add_argument("--head-steps", type=int, default=100)
    p.add_argument("--joint-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=6696, help="Joint-stage optimizer update budget")
    p.add_argument("--limit-samples", type=int, default=None, help="Explicit training subset for wiring checks only")
    p.add_argument("--expected-samples", type=int, default=321327)
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--calibration-path")
    p.add_argument("--run-name", default="Jag")
    p.add_argument("--recipe-file", help="Optional optimizer/schedule JSON overriding the config recipe")
    return p


def parse_args(argv=None):
    """Read one reproducible config, validating its values before CLI overrides."""
    p = parser()
    initial, _ = p.parse_known_args(argv)
    recipe = None
    if initial.config:
        document = json.loads(Path(initial.config).read_text())
        if (not isinstance(document, dict) or set(document) != {"schema_version", "training", "recipe"}
                or type(document["schema_version"]) is not int or document["schema_version"] != 1):
            p.error("Config requires schema_version=1, training and recipe sections")
        values = document["training"]
        if not isinstance(values, dict):
            p.error("Config training section must be an object")
        allowed = {action.dest: action for action in p._actions
                   if action.dest not in {"help", "config", "recipe_file"}}
        if unknown := set(values) - set(allowed):
            p.error(f"Unknown training config fields: {sorted(unknown)}")
        nullable = {"init_checkpoint", "resume_from_checkpoint", "limit_samples", "calibration_path"}
        for name, value in values.items():
            action = allowed[name]
            if value is None and name in nullable:
                continue
            if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
                valid = type(value) is bool
            elif action.type is int:
                valid = type(value) is int
            elif action.type is float:
                valid = type(value) in {int, float} and math.isfinite(value)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid or (action.choices and value not in action.choices):
                p.error(f"Invalid training config value for {name}: {value!r}")
        recipe = validate_recipe(document["recipe"])
        p.set_defaults(**values)
    args = p.parse_args(argv)
    args.config_recipe = recipe
    return args


def resolve_resume(args):
    value = args.resume_from_checkpoint
    if value != "auto":
        return Path(value).resolve() if value else None
    for stage in (["joint", "head"] if args.stage == "all" else [args.stage]):
        directory = Path(args.output_dir) / stage
        if directory.is_dir() and (last := get_last_checkpoint(str(directory))):
            return Path(last)
    raise FileNotFoundError("No existing stage checkpoint found for automatic resume")


def validate_resume(checkpoint, config):
    required = ["config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt",
                "sampler_state.json", "jag_run_config.json"]
    world_size = config["world_size"]
    required += (["rng_state.pth"] if world_size == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world_size)])
    absent = [name for name in required if not (checkpoint / name).is_file()]
    if absent:
        raise ValueError(f"Incomplete resumable checkpoint {checkpoint}: missing {absent}")
    previous = json.loads((checkpoint / "jag_run_config.json").read_text())
    prior_recipe = previous.get("training_recipe", load_recipe())
    if recipe_fingerprint(prior_recipe) != recipe_fingerprint(config.get("training_recipe", load_recipe())):
        raise ValueError("Exact resume requires the same optimizer and scheduler recipe")
    for key in ("dataset_sha256", "dataset_samples", "limit_samples", "world_size", "microbatch",
                "effective_batch_size", "gradient_accumulation_steps", "seed", "max_pixels", "max_length",
                "head_steps", "joint_epochs", "max_steps", "used_samples", "framework", "master_dtype",
                "autocast_dtype", "head_learning_rate", "joint_learning_rates", "head_scheduler",
                "joint_scheduler", "joint_warmup_ratio", "optimizer", "adam_betas", "adam_epsilon",
                "weight_decay", "max_grad_norm", "sampler"):
        if previous.get(key) != config.get(key):
            raise ValueError(f"Exact resume requires unchanged {key}: {previous.get(key)!r} != {config.get(key)!r}")
    sampler = json.loads((checkpoint / "sampler_state.json").read_text())
    for key, expected in {"sampler": config["sampler"], "world_size": world_size,
                          "microbatch": config["microbatch"], "data_seed": config["seed"],
                          "dataset_size": config["used_samples"],
                          "gradient_accumulation_steps": config["gradient_accumulation_steps"]}.items():
        if sampler.get(key) != expected:
            raise ValueError(f"Checkpoint sampler {key} does not match this run")


def train_stage(args, stage, model, processor, dataset, config, callbacks=None, resume=None):
    stage_dir = Path(args.output_dir) / stage
    if resume is None and (stage_dir / "trainer_state.json").exists():
        raise FileExistsError(f"Existing stage output requires --resume-from-checkpoint: {stage_dir}")
    model.set_stage(stage)
    if resume is not None:
        saved_state = json.loads((resume / "trainer_state.json").read_text())
        target_steps = args.head_steps if stage == "head" else config["planned_joint_steps"]
        if saved_state["global_step"] >= target_steps:
            # Calling HF train() at an already completed mid-epoch max_steps
            # boundary can consume one further batch before its stop callback.
            # The model was restored by run(); keep its final weights intact.
            if int(os.environ.get("RANK", "0")) == 0:
                write_json(stage_dir / "completed.json", {
                    "stage": stage, "global_step": saved_state["global_step"],
                    "checkpoint": str(resume), "already_completed": True,
                })
            return resume, saved_state["global_step"]
    model.float()  # FP32 master weights; SWIFT/Accelerate executes BF16 autocast.
    collator = GroundingCollator(processor, max_pixels=args.max_pixels, max_length=args.max_length)
    template = attach_swift_metadata(model, processor, args.model, args.max_length)
    arguments = make_training_arguments(
        stage_dir, microbatch=args.microbatch, accumulation=config["gradient_accumulation_steps"],
        stage=stage, head_steps=args.head_steps, joint_epochs=args.joint_epochs, max_steps=args.max_steps,
        use_cpu=args.use_cpu, workers=args.workers, save_steps=args.save_steps,
        logging_steps=args.logging_steps, seed=args.seed, resume=str(resume) if resume else None,
        recipe=config["training_recipe"],
        planned_steps=args.head_steps if stage == "head" else config["planned_joint_steps"],
    )
    trainer = GroundingTrainer(
        model=model, args=arguments, template=template, train_dataset=dataset,
        data_collator=collator, callbacks=callbacks or [], run_config={**config, "active_stage": stage},
        learning_rates={**config["joint_learning_rates"],
                        "head": config["head_learning_rate"] if stage == "head" else config["joint_learning_rates"]["head"]},
    )
    if trainer.is_world_process_zero():
        write_json(stage_dir / "run_config.json", {**config, "active_stage": stage})
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    last = stage_dir / f"checkpoint-{trainer.state.global_step}"
    trainer.accelerator.wait_for_everyone()
    if not (last / "trainer_state.json").is_file():
        trainer._save_checkpoint(trainer.model, trial=None)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        write_json(stage_dir / "completed.json", {
            "stage": stage, "global_step": trainer.state.global_step,
            "checkpoint": str(last), "metrics": result.metrics,
        })
    steps = trainer.state.global_step
    # All stages run in one process group. Release optimizer/accelerator state
    # before building the joint optimizer with the newly unfrozen parameters.
    trainer.accelerator.free_memory()
    del trainer
    gc.collect()
    if torch.cuda.is_available() and not args.use_cpu:
        torch.cuda.empty_cache()
    return last, steps


def run(args):
    from jag.monitoring import MonitoringSession

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {1, 2, 4}:
        raise ValueError("This training recipe supports one, two, or four GPUs")
    divisor = args.microbatch * world_size
    if args.effective_batch_size < divisor or args.effective_batch_size % divisor:
        raise ValueError("Effective batch size must be a positive multiple of microbatch × world size")
    if args.head_steps <= 0 or args.joint_epochs <= 0:
        raise ValueError("Training stage lengths must be positive")
    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("Training requires the selected CUDA GPU(s)")
    if not args.use_cpu:
        if world_size == 1 and torch.cuda.device_count() != 1:
            raise ValueError("Single-process training requires exactly one visible GPU; set CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    set_seed(args.seed)
    recipe = (load_recipe(args.recipe_file) if args.recipe_file else
              getattr(args, "config_recipe", None) or load_recipe())
    source = RefCOCODataset(args.train_jsonl)
    if len(source) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} source rows, found {len(source)}")
    dataset = Subset(source, range(min(args.limit_samples, len(source)))) if args.limit_samples else source
    config = {key: value for key, value in vars(args).items() if key != "config_recipe"}
    config.update(world_size=world_size, dataset_samples=len(source), used_samples=len(dataset),
                  dataset_sha256=file_hash(args.train_jsonl), framework="ms-swift==4.3.2",
                  gradient_accumulation_steps=args.effective_batch_size // divisor,
                  master_dtype="float32", autocast_dtype="bfloat16" if not args.use_cpu else "float32",
                  training_recipe=recipe, recipe_sha256=recipe_fingerprint(recipe),
                  head_learning_rate=recipe["head_learning_rate"], joint_learning_rates=recipe["joint_learning_rates"],
                  head_scheduler=recipe["head_scheduler"], head_warmup_steps=recipe["head_warmup_steps"],
                  joint_scheduler=recipe["joint_scheduler"], joint_warmup_ratio=recipe["joint_warmup_ratio"],
                  optimizer="AdamW", adam_betas=recipe["adam_betas"], adam_epsilon=recipe["adam_epsilon"],
                  weight_decay=recipe["weight_decay"], max_grad_norm=recipe["max_grad_norm"],
                  sampler="jag.trainer.GroundingBatchSampler",
                  validation_dataset=None, evaluation_policy="Evaluation samples are not used for gradient updates.")
    joint_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(math.ceil(len(dataset) / divisor) / config["gradient_accumulation_steps"]) * args.joint_epochs)
    config["planned_joint_steps"] = joint_steps
    stage_schedule(recipe, "head", args.head_steps)
    config["joint_warmup_steps"] = stage_schedule(recipe, "joint", joint_steps)["warmup_steps"]
    resume = resolve_resume(args)
    resume_stage = None
    if resume:
        validate_resume(resume, config)
        resume_stage = json.loads((resume / "config.json").read_text())["stage"]
        if args.stage != "all" and resume_stage != args.stage:
            raise ValueError("Requested stage does not match the resumable checkpoint")
    session = MonitoringSession(
        output_dir=args.output_dir, run_config=config, enabled=args.swanlab,
        project="Jag", experiment_name=args.run_name,
        calibration_path=args.calibration_path,
    )
    try:
        stages = ["head", "joint"] if args.stage == "all" else [args.stage]
        if resume_stage == "joint":
            stages = ["joint"]
        processor_path = resume or args.init_checkpoint or args.model
        processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        initial = resume or (Path(args.init_checkpoint) if args.init_checkpoint else None)
        if initial:
            model = JagModel.from_pretrained(initial, dtype=torch.float32, local_files_only=True)
        else:
            model = JagModel.from_base(args.model, stage=stages[0], torch_dtype=torch.float32)
        offset = args.head_steps if resume_stage == "joint" and args.stage == "all" else 0
        for stage in stages:
            checkpoint, steps = train_stage(
                args, stage, model, processor, dataset, config,
                callbacks=session.callbacks(stage=stage, step_offset=offset,
                                            future_joint_steps=joint_steps if stage == "head" and "joint" in stages else 0),
                resume=resume if resume_stage == stage else None,
            )
            offset += steps
            if stage == "head" and "joint" in stages:
                # The optimizer and scheduler are rebuilt by a new SWIFT Trainer;
                # the trained model itself retains exactly the saved head weights.
                model.set_stage("joint")
        session.finish(status="success")
    except Exception as error:
        session.finish(status="failed", error=str(error))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
