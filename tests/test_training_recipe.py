"""Exercise the published schedule, optimizer ratios, config overrides and resume."""

from contextlib import chdir, redirect_stderr
from copy import deepcopy
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import get_scheduler

from groundingjev.recipes import default_config_path, load_recipe, recipe_fingerprint, stage_schedule
from groundingjev.trainer import GroundingTrainer, make_training_arguments
from train import parse_args


CONFIG = Path(__file__).resolve().parents[1] / "configs/train/jag.json"


def scheduler(recipe, stage, total):
    spec = stage_schedule(recipe, stage, total)
    rates = ([recipe["head_learning_rate"]] if stage == "head" else
             [recipe["joint_learning_rates"][key] for key in ("language", "merger", "head")])
    optimizer = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate}
                                   for rate in rates])
    schedule = get_scheduler(spec["scheduler"], optimizer,
                             num_warmup_steps=spec["warmup_steps"], num_training_steps=total)
    return optimizer, schedule


def advance(optimizer, schedule, count):
    rates = {0: schedule.get_last_lr()}
    for step in range(1, count + 1):
        optimizer.step()
        schedule.step()
        rates[step] = schedule.get_last_lr()
    return rates


class RecipeTests(unittest.TestCase):
    def test_config_location_prefers_checkout_then_installed_data_and_ignores_cwd(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "checkout/groundingjev/recipes.py"
            checkout = root / "checkout/configs/train/jag.json"
            installed = root / "prefix/share/jag/configs/train/jag.json"
            rogue = root / "cwd/configs/train/jag.json"
            for path in (checkout, installed, rogue):
                path.parent.mkdir(parents=True)
                path.write_text(CONFIG.read_text())
            with patch("groundingjev.recipes.__file__", str(package)), \
                    patch("groundingjev.recipes.sysconfig.get_path", return_value=str(root / "prefix")), \
                    chdir(root / "cwd"):
                self.assertEqual(default_config_path(), checkout)
                checkout.unlink()
                self.assertEqual(default_config_path(), installed)
                self.assertEqual(load_recipe(), json.loads(installed.read_text())["recipe"])

    def test_single_published_config_is_default_recipe(self):
        args = parse_args(["--config", str(CONFIG)])
        self.assertEqual(args.config_recipe, load_recipe())
        self.assertEqual(args.expected_samples, 321327)
        self.assertEqual(args.output_dir, "/outputs/jag")
        self.assertFalse(args.swanlab)

    def test_published_recipe_and_plain_cli_match_the_released_training_run(self):
        recipe = load_recipe()
        self.assertEqual(recipe_fingerprint(recipe), "0e99ad2864804062e7bb6b02af41a6127d1cb04784e0ce456d85775996008502")
        self.assertEqual(stage_schedule(recipe, "head", 100),
                         {"scheduler": "cosine", "warmup_steps": 10, "learning_rate": .0005})
        self.assertEqual(stage_schedule(recipe, "joint", 6696),
                         {"scheduler": "cosine", "warmup_steps": 335, "learning_rate": .00001})
        args = parse_args([])
        expected = json.loads(CONFIG.read_text())["training"]
        for key, value in expected.items():
            self.assertEqual(getattr(args, key), value, key)
        self.assertIsNone(args.config_recipe)
        self.assertIsNone(args.recipe_file)

    def test_direct_trainer_uses_stage_specific_recipe_and_requires_explicit_factory_arguments(self):
        recipe = load_recipe()
        with TemporaryDirectory() as directory:
            for stage, steps in (("head", 100), ("joint", 6696)):
                args = make_training_arguments(directory, microbatch=2, accumulation=12, stage=stage,
                                               planned_steps=steps, use_cpu=True, workers=0)
                model = SimpleNamespace(config=SimpleNamespace(stage=stage))
                with patch("groundingjev.trainer.Trainer.__init__", return_value=None):
                    trainer = GroundingTrainer(model=model, args=args, data_collator=lambda rows: rows)
                rates = dict(recipe["joint_learning_rates"])
                if stage == "head":
                    rates["head"] = recipe["head_learning_rate"]
                self.assertEqual(trainer.group_learning_rates, rates)
                self.assertEqual(args.lr_scheduler_type.value, "cosine")
                self.assertEqual(args.get_warmup_steps(steps), 10 if stage == "head" else 335)
                self.assertEqual(args.weight_decay, .01)
                self.assertEqual((args.adam_beta1, args.adam_beta2, args.adam_epsilon), (.9, .999, 1e-8))
                self.assertEqual(args.max_grad_norm, 1.)
                self.assertEqual(args.gradient_checkpointing, stage == "joint")
                self.assertFalse(args.vit_gradient_checkpointing)
                self.assertFalse(args.use_cache)
                self.assertFalse(args.lr_scheduler_kwargs)
            with self.assertRaisesRegex(ValueError, "make_training_arguments"):
                GroundingTrainer(model=model, data_collator=lambda rows: rows)
            with self.assertRaisesRegex(ValueError, "make_training_arguments"):
                GroundingTrainer(model=model, args=SimpleNamespace(), data_collator=lambda rows: rows)
            with self.assertRaisesRegex(ValueError, "same head/joint stage"):
                GroundingTrainer(model=SimpleNamespace(config=SimpleNamespace(stage="head")), args=args,
                                 data_collator=lambda rows: rows)
            changed = {**recipe, "head_learning_rate": recipe["head_learning_rate"] * 2}
            with self.assertRaisesRegex(ValueError, "different recipes"):
                GroundingTrainer(model=model, args=args, data_collator=lambda rows: rows,
                                 run_config={"training_recipe": changed})

    def test_explicit_cli_overrides_config_and_unknown_fields_fail(self):
        args = parse_args(["--config", str(CONFIG), "--microbatch", "1", "--output-dir", "/tmp/run", "--swanlab"])
        self.assertEqual(args.microbatch, 1)
        self.assertEqual(args.output_dir, "/tmp/run")
        self.assertTrue(args.swanlab)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = json.loads(CONFIG.read_text())
            invalid_values = [{"microbatch": 3}, {"microbatch": True}, {"microbatch": 2.5},
                              {"swanlab": "false"}, {"max_pixels": None}, {"typo_epochs": 2}]
            for values in invalid_values:
                document = deepcopy(original)
                document["training"].update(values)
                path.write_text(json.dumps(document))
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--config", str(path)])

    def test_actual_joint_schedule_warms_up_then_decays_with_group_ratios(self):
        recipe = load_recipe()
        total = 6696
        warmup = math.ceil(total * recipe["joint_warmup_ratio"])
        opt, sched = scheduler(recipe, "joint", total)
        points = advance(opt, sched, total)
        self.assertEqual(stage_schedule(recipe, "joint", total)["warmup_steps"], warmup)
        for group, name in enumerate(("language", "merger", "head")):
            self.assertAlmostEqual(points[warmup][group], recipe["joint_learning_rates"][name])
            self.assertEqual(points[total][group], 0)
        self.assertLess(points[900][0], points[500][0])
        if warmup > 20:
            self.assertGreater(points[20][0], points[10][0])

    def test_actual_swift_arguments_use_published_recipe(self):
        recipe = load_recipe()
        with TemporaryDirectory() as directory:
            args = make_training_arguments(directory, microbatch=2, accumulation=12,
                                           stage="joint", planned_steps=6696,
                                           recipe=recipe, use_cpu=True, workers=0)
            self.assertEqual(args.get_warmup_steps(6696), stage_schedule(recipe, "joint", 6696)["warmup_steps"])
            self.assertEqual(args.lr_scheduler_type.value, recipe["joint_scheduler"])
            self.assertEqual(args.learning_rate, recipe["joint_learning_rates"]["language"])
            self.assertEqual(args.weight_decay, recipe["weight_decay"])

    def test_resume_preserves_next_scheduler_steps_for_all_groups(self):
        recipe = load_recipe()
        opt, sched = scheduler(recipe, "joint", 6696)
        advance(opt, sched, 123)
        saved_opt, saved_sched = deepcopy(opt.state_dict()), deepcopy(sched.state_dict())
        expected = advance(opt, sched, 10)
        restored_opt, restored_sched = scheduler(recipe, "joint", 6696)
        restored_opt.load_state_dict(saved_opt)
        restored_sched.load_state_dict(saved_sched)
        self.assertEqual(advance(restored_opt, restored_sched, 10), expected)

    def test_recipe_validation_rejects_invalid_optimizer_values(self):
        original = load_recipe()
        renamed = {**original, "name": "display-only-name"}
        self.assertEqual(recipe_fingerprint(original), recipe_fingerprint(renamed))
        changed = {**original, "head_learning_rate": original["head_learning_rate"] * 2}
        self.assertNotEqual(recipe_fingerprint(original), recipe_fingerprint(changed))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            for invalid in ({"typo_learning_rate": 1e-5}, {"head_learning_rate": -1},
                            {"joint_learning_rates": {"language": 1e-5}},
                            {"head_learning_rate": float("nan")}, {"joint_warmup_ratio": 1}):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_recipe(path)


if __name__ == "__main__":
    unittest.main()
