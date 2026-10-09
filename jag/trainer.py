"""Grounding regression implemented through the ms-swift Trainer lifecycle."""

import json
import math
from functools import partial
from pathlib import Path

import torch
from swift.dataloader import DataLoaderShard
from swift.model import ModelInfo, ModelMeta
from swift.model.model_arch import MultiModelKeys
from swift.optimizers import OptimizerCallback, optimizers_map
from swift.template import Template, TemplateMeta
from swift.trainers import Trainer, TrainingArguments
from transformers import GenerationConfig
from transformers.trainer_utils import seed_worker

from .recipes import load_recipe, recipe_fingerprint, stage_schedule, validate_recipe


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class GroundingTemplate(Template):
    """SWIFT lifecycle adapter; GroundingCollator supplies the encoded batch.

    No generative Qwen template hook is used: the backbone injects visual tokens
    exactly once.  SWIFT's causal_lm task name is only infrastructure metadata;
    GroundingTrainer supplies the independent bbox regression objective.
    """

    def pre_forward_hook(self, model, args, kwargs):
        return args, kwargs


def attach_swift_metadata(model, processor, model_dir, max_length=2048):
    metadata = ModelMeta(
        model_type="jag", model_groups=[], template="jag",
        is_multimodal=True, task_type="causal_lm",
        architectures=["JagModel"], additional_saved_files=[],
    )
    metadata.model_arch = MultiModelKeys(
        arch_name="jag", language_model="backbone.language_model",
        vision_tower="backbone.visual", aligner="backbone.visual.merger",
        embedding="backbone.language_model.embed_tokens",
    )
    information = ModelInfo(
        model_type="jag", model_dir=str(model_dir), torch_dtype=torch.float32,
        max_model_len=max_length, quant_method=None, quant_bits=None,
        is_multimodal=True, config=model.config, task_type="causal_lm",
    )
    for target in (model, processor):
        target.model_meta = metadata
        target.model_info = information
        target.model_dir = str(model_dir)
    tokenizer = getattr(processor, "tokenizer", processor)
    model.generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    template = GroundingTemplate(
        processor, TemplateMeta(
            template_type="jag", prefix=[], prompt=["{{QUERY}}"],
            chat_sep=[], suffix=[],
        ), max_length=max_length, remove_unused_columns=False,
        padding_free=False, sequence_parallel_size=1, enable_thinking=False,
    )
    template.mode = "train"
    return template


def parameter_groups(model, learning_rates, weight_decay):
    groups, audit = {}, []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("bbox_head."):
            owner = "head"
        elif name.startswith("backbone.visual.merger."):
            owner = "merger"
        elif name.startswith("backbone.language_model."):
            owner = "language"
        else:
            raise ValueError(f"Unclassified trainable parameter: {name}")
        if parameter.dtype != torch.float32:
            raise ValueError(f"Trainable master parameter must be FP32: {name} ({parameter.dtype})")
        decay = parameter.ndim >= 2 and not name.endswith(".bias")
        key = (owner, decay)
        if key not in groups:
            groups[key] = {
                "params": [], "lr": learning_rates[owner],
                "weight_decay": weight_decay if decay else 0.0,
                "group_name": owner, "decay": decay,
            }
        groups[key]["params"].append(parameter)
        audit.append({"name": name, "group": owner, "numel": parameter.numel(),
                      "dtype": str(parameter.dtype), "weight_decay": weight_decay if decay else 0.0,
                      "learning_rate": learning_rates[owner]})
    expected = {"head"} if model.config.stage == "head" else {"head", "merger", "language"}
    if {owner for owner, _ in groups} != expected:
        raise ValueError(f"Incorrect trainable parameter groups for {model.config.stage}")
    actual = [id(parameter) for group in groups.values() for parameter in group["params"]]
    if len(actual) != len(set(actual)):
        raise ValueError("Optimizer contains duplicate parameters")
    return list(groups.values()), audit


class GroundingOptimizer(OptimizerCallback):
    def create_optimizer(self, model=None):
        model = model or self.trainer.model
        groups, audit = parameter_groups(model, self.trainer.group_learning_rates, self.args.weight_decay)
        if self.trainer.is_world_process_zero():
            write_json(Path(self.args.output_dir) / "trainable_parameters.json", audit)
        return torch.optim.AdamW(
            groups, lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon,
        )

    def create_optimizer_and_scheduler(self, num_training_steps):
        trainer = self.trainer
        if trainer.optimizer is None:
            trainer.optimizer = self.create_optimizer()
        trainer._optimizer_ori = trainer.optimizer
        trainer.lr_scheduler = self.create_scheduler(num_training_steps, trainer.optimizer)


optimizers_map["jag"] = GroundingOptimizer


class GroundingBatchSampler:
    """Shuffle every source row, pad across ranks, and preserve epoch on resume.

    Only the minimum number of repeated rows needed for equal rank lengths is
    added. When row count is not divisible by world size, only the required padding rows repeat.
    Keeping the skip offset in this sampler avoids wrappers losing set_epoch.
    """

    def __init__(self, dataset_size, batch_size, num_replicas=1, rank=0,
                 seed=42, shuffle=True, skip_batches=0):
        if dataset_size <= 0 or batch_size <= 0 or num_replicas <= 0:
            raise ValueError("Dataset, batch and world sizes must be positive")
        if not 0 <= rank < num_replicas or skip_batches < 0:
            raise ValueError("Invalid rank or skipped batch count")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.skip_batches = skip_batches
        self.epoch = 0
        self.samples_per_rank = math.ceil(dataset_size / num_replicas)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return max(0, math.ceil(self.samples_per_rank / self.batch_size) - self.skip_batches)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = (torch.randperm(self.dataset_size, generator=generator).tolist()
                   if self.shuffle else list(range(self.dataset_size)))
        missing = self.samples_per_rank * self.num_replicas - len(indices)
        if missing:
            indices += (indices * math.ceil(missing / len(indices)))[:missing]
        local = indices[self.rank::self.num_replicas]
        for start in range(self.skip_batches * self.batch_size, len(local), self.batch_size):
            yield local[start:start + self.batch_size]


class GroundingTrainer(Trainer):
    """Use SWIFT optimization, BF16 AMP, DDP, sampler, and checkpoint handling."""

    def __init__(self, *args, data_collator, callbacks=None, learning_rates=None,
                 run_config=None, **kwargs):
        self.grounding_collator = data_collator
        self.grounding_callbacks = list(callbacks or [])
        self.run_config = run_config or {}
        training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
        recipe = getattr(training_args, "_jag_recipe", None)
        stage = getattr(training_args, "_jag_stage", None)
        if recipe is None or stage not in {"head", "joint"}:
            raise ValueError("GroundingTrainer requires explicit args from make_training_arguments(..., stage=...)")
        recipe = validate_recipe(recipe)
        configured_model = kwargs.get("model", args[0] if args else None)
        if getattr(getattr(configured_model, "config", None), "stage", None) != stage:
            raise ValueError("Training arguments and model must declare the same head/joint stage")
        configured_recipe = self.run_config.get("training_recipe")
        if configured_recipe is not None and recipe_fingerprint(configured_recipe) != recipe_fingerprint(recipe):
            raise ValueError("Training arguments and run configuration use different recipes")
        self.group_learning_rates = (dict(learning_rates) if learning_rates is not None else
                                     dict(recipe["joint_learning_rates"]))
        if learning_rates is None and stage == "head":
            self.group_learning_rates["head"] = recipe["head_learning_rate"]
        self.loss_statistics = None
        super().__init__(*args, **kwargs)
        # HF 5.9 then divides exactly once by current_gradient_accumulation_steps.
        # We separately account for unequal final microbatch sizes below.
        self.model_accepts_loss_kwargs = False
        self.label_names = ["bbox_targets"]

    def _get_callbacks(self, args):
        return super()._get_callbacks(args) + self.grounding_callbacks

    def _get_data_collator(self, args, template):
        return self.grounding_collator

    def get_train_dataloader(self, skip_batches=0):
        if self.train_dataset is None:
            raise ValueError("Training requires a dataset")
        args = self.args
        sampler = GroundingBatchSampler(
            len(self.train_dataset), self._train_batch_size,
            num_replicas=args.world_size, rank=args.process_index,
            seed=args.data_seed, shuffle=args.train_dataloader_shuffle,
            skip_batches=skip_batches,
        )
        return DataLoaderShard(
            self.train_dataset, device=self.accelerator.device,
            batch_sampler=sampler, collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            persistent_workers=args.dataloader_persistent_workers,
            prefetch_factor=args.dataloader_prefetch_factor,
            worker_init_fn=partial(seed_worker, num_workers=args.dataloader_num_workers,
                                   rank=args.process_index),
        )

    def _prepare_gradient_checkpointing(self, model):
        # The project's model already forwards checkpointing to Qwen3.5. Avoid
        # generic SWIFT patching of generative wrapper layouts or the frozen ViT.
        model.config.use_cache = False
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.backbone.visual.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_disable()
        self.args.gradient_checkpointing = False

    def _get_num_items_in_batch(self, batch_samples, device):
        if not batch_samples:
            return None
        count = torch.tensor(sum(batch["bbox_targets"].shape[0] for batch in batch_samples),
                             device=device, dtype=torch.float32)
        return self.accelerator.reduce(count, reduction="sum")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "bbox_targets" not in inputs or "labels" in inputs:
            raise ValueError("Grounding regression requires bbox_targets and no language labels")
        outputs = model(**inputs)
        loss = outputs.loss
        batch_size = inputs["bbox_targets"].shape[0]
        if loss is None or not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("Grounding loss is absent or non-finite")
        if model.training:
            statistics = torch.stack((outputs.loss_l1.detach() * batch_size,
                                      outputs.loss_giou.detach() * batch_size,
                                      loss.detach().new_tensor(batch_size)))
            if self.loss_statistics is None:
                self.loss_statistics = statistics
            else:
                self.loss_statistics += statistics
        if num_items_in_batch is not None:
            # DDP averages rank gradients; HF divides by the actual accumulation
            # window length. Compensate both to average individual samples,
            # including a smaller last microbatch and the last partial window.
            loss = loss * (batch_size * self.current_gradient_accumulation_steps
                           * self.accelerator.num_processes / num_items_in_batch)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self.loss_statistics is not None:
            sums = self.accelerator.reduce(self.loss_statistics, reduction="sum")
            logs["loss_l1"] = (sums[0] / sums[2]).item()
            logs["loss_giou"] = (sums[1] / sums[2]).item()
            self.loss_statistics = None
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    logs[f"lr_{group['group_name']}"] = group["lr"]
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = Path(output_dir or self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_model(str(output_dir), state_dict)
        self.template.processor.save_pretrained(output_dir)
        torch.save(self.args, output_dir / "training_args.bin")
        write_json(output_dir / "jag_run_config.json", self.run_config)

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        super()._save_checkpoint(model, trial, *args, **kwargs)
        if self.is_world_process_zero():
            path = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            write_json(path / "sampler_state.json", {
                "sampler": "jag.trainer.GroundingBatchSampler", "data_seed": self.args.data_seed,
                "epoch": self.state.epoch, "global_step": self.state.global_step,
                "world_size": self.args.world_size,
                "microbatch": self.args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "dataset_size": len(self.train_dataset),
                "resume": "Reconstruct seeded epoch permutation, then restore the Trainer batch offset and RNG.",
            })


def make_training_arguments(output_dir, *, microbatch, accumulation, stage, head_steps=100,
                            joint_epochs=2, max_steps=-1, bf16=True, use_cpu=False,
                            workers=2, save_steps=250, logging_steps=5, seed=42, resume=None,
                            recipe=None, planned_steps=None):
    recipe = recipe or load_recipe()
    total_steps = planned_steps or (head_steps if stage == "head" else max_steps)
    schedule = stage_schedule(recipe, stage, total_steps) if total_steps > 0 else None
    warmup = ({"warmup_steps": schedule["warmup_steps"]} if schedule is not None else
              {"warmup_ratio": recipe["joint_warmup_ratio"]})
    arguments = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=joint_epochs, max_steps=head_steps if stage == "head" else max_steps,
        learning_rate=recipe["head_learning_rate"] if stage == "head" else recipe["joint_learning_rates"]["language"],
        optimizer="jag", optim="adamw_torch", weight_decay=recipe["weight_decay"],
        adam_beta1=recipe["adam_betas"][0], adam_beta2=recipe["adam_betas"][1],
        adam_epsilon=recipe["adam_epsilon"],
        lr_scheduler_type=recipe[f"{stage}_scheduler"], **warmup,
        max_grad_norm=recipe["max_grad_norm"], bf16=bf16 and not use_cpu, use_cpu=use_cpu,
        gradient_checkpointing=stage == "joint", vit_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}, use_cache=False,
        eval_strategy="no", save_strategy="steps", save_steps=save_steps,
        save_total_limit=2, save_only_model=False,
        logging_steps=logging_steps, logging_first_step=True, report_to=[],
        dataloader_num_workers=workers, dataloader_pin_memory=not use_cpu,
        dataloader_persistent_workers=workers > 0, dataloader_drop_last=False,
        remove_unused_columns=False, label_names=["bbox_targets"],
        seed=seed, data_seed=seed, ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False, check_model=False, average_tokens_across_devices=False,
        train_dataloader_shuffle=True, group_by_length=False,
        resume_from_checkpoint=resume, ignore_data_skip=False,
        restore_callback_states_from_checkpoint=True, disable_tqdm=True,
    )
    arguments._jag_recipe = validate_recipe(recipe)
    arguments._jag_stage = stage
    return arguments
