"""Validated optimizer and scheduler settings for Jag."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sysconfig


RECIPE_FIELDS = {
    "schema_version", "name", "head_learning_rate", "head_scheduler", "head_warmup_steps",
    "joint_learning_rates", "joint_scheduler", "joint_warmup_ratio", "weight_decay",
    "max_grad_norm", "adam_betas", "adam_epsilon",
}


def default_config_path():
    """Resolve the checkout or installed-wheel config without consulting cwd."""
    root = Path(__file__).resolve().parents[1]
    prefix = Path(sysconfig.get_path("data"))
    candidates = (
        root / "configs/train/jag.json",
        root / "configs/train/groundingjev.json",
        prefix / "share/jag/configs/train/jag.json",
        prefix / "share/groundingjev/configs/train/groundingjev.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Jag training configuration is missing from the checkout or installation")


def validate_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Training recipe must be a JSON object")
    if set(recipe) != RECIPE_FIELDS:
        raise ValueError(f"Recipe fields differ from schema: {set(recipe) ^ RECIPE_FIELDS}")
    if type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ValueError("Unsupported training recipe schema")
    if not isinstance(recipe["name"], str) or not recipe["name"].strip():
        raise ValueError("Recipe requires a nonempty name")
    def number(value, name, minimum=0, inclusive=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (value < minimum if inclusive else value <= minimum)):
            raise ValueError(f"Invalid {name}: {value!r}")
    number(recipe["head_learning_rate"], "head_learning_rate")
    rates = recipe["joint_learning_rates"]
    if not isinstance(rates, dict) or set(rates) != {"language", "merger", "head"}:
        raise ValueError("Joint recipe requires language, merger, and head learning rates")
    for name, value in rates.items():
        number(value, f"joint_learning_rates.{name}")
    for name in ("head_scheduler", "joint_scheduler"):
        if recipe[name] not in {"constant", "constant_with_warmup", "cosine"}:
            raise ValueError(f"Unsupported scheduler {recipe[name]}")
    if type(recipe["head_warmup_steps"]) is not int or recipe["head_warmup_steps"] < 0:
        raise ValueError("head_warmup_steps must be a nonnegative integer")
    number(recipe["joint_warmup_ratio"], "joint_warmup_ratio", inclusive=True)
    if recipe["joint_warmup_ratio"] >= 1:
        raise ValueError("joint_warmup_ratio must be less than one")
    if recipe["head_scheduler"] == "constant" and recipe["head_warmup_steps"]:
        raise ValueError("A constant scheduler cannot have warmup; use constant_with_warmup")
    if recipe["joint_scheduler"] == "constant" and recipe["joint_warmup_ratio"]:
        raise ValueError("A constant scheduler cannot have warmup; use constant_with_warmup")
    number(recipe["weight_decay"], "weight_decay", inclusive=True)
    number(recipe["max_grad_norm"], "max_grad_norm")
    number(recipe["adam_epsilon"], "adam_epsilon")
    betas = recipe["adam_betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("adam_betas must contain two values")
    for beta in betas:
        number(beta, "adam_beta", inclusive=True)
        if beta >= 1:
            raise ValueError("Adam beta must be less than one")
    return deepcopy(recipe)


def load_recipe(path=None):
    """Use the single published configuration, with an optional recipe override."""
    recipe = validate_recipe(json.loads(default_config_path().read_text())["recipe"])
    if path is None:
        return recipe
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Training recipe must be a JSON object")
    unknown = set(value) - RECIPE_FIELDS
    if unknown:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    recipe.update(value)
    return validate_recipe(recipe)


def stage_schedule(recipe, stage, total_steps):
    if stage not in {"head", "joint"} or type(total_steps) is not int or total_steps <= 0:
        raise ValueError("A stage schedule requires a valid stage and positive optimizer-step count")
    warmup = (recipe["head_warmup_steps"] if stage == "head" else
              math.ceil(total_steps * recipe["joint_warmup_ratio"]))
    if warmup > total_steps:
        raise ValueError("Warmup must not exceed the training stage")
    return {"scheduler": recipe[f"{stage}_scheduler"], "warmup_steps": warmup,
            "learning_rate": recipe["head_learning_rate"] if stage == "head" else
            recipe["joint_learning_rates"]["language"]}


def recipe_fingerprint(recipe):
    # The descriptive label is not an optimizer setting.
    functional = {key: value for key, value in recipe.items() if key != "name"}
    return hashlib.sha256(json.dumps(functional, sort_keys=True, allow_nan=False).encode()).hexdigest()
