#!/usr/bin/env python3
"""Portable Docker Compose launcher with explicit physical GPU selection."""

import grp
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read_environment(path):
    values = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env assignment on line {number}")
            name, value = line.split("=", 1)
            name = name.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid .env key on line {number}")
            # Literal values only: no shell execution or variable interpolation.
            tokens = shlex.split(value, comments=True)
            if len(tokens) > 1:
                raise ValueError(f"Quote values containing spaces on .env line {number}")
            values[name] = tokens[0] if tokens else ""
    values.update(os.environ)  # Shell overrides take precedence over .env.
    return values


def launch(arguments):
    environment = read_environment(ROOT / ".env")
    evaluation = bool(arguments and arguments[0] == "--eval")
    if evaluation:
        arguments = arguments[1:]
    if not arguments:
        raise ValueError("Usage: bash scripts/docker.sh [--eval] <docker compose arguments>")
    training_ids = environment.get("GPU_IDS", "0").split(",")
    ids = ([environment.get("EVAL_GPU", training_ids[0])] if evaluation else training_ids)
    ids = [value.strip() for value in ids]
    if not ids or len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+|MIG-[A-Za-z0-9/-]+)", value)
        for value in ids
    ):
        raise ValueError("GPU_IDS/EVAL_GPU must specify unique physical GPU IDs or UUIDs")
    environment["JAG_PROJECT_DIR"] = str(ROOT)
    environment.setdefault("JAG_UID", str(os.getuid()))
    environment.setdefault("JAG_GID", str(os.getgid()))
    environment.setdefault("JAG_NPROC_PER_NODE", str(len(training_ids)))
    defaults = {
        "REFCOCO_ANNOTATIONS_DIR": "data/refcoco",
        "REFCOCO_IMAGES_DIR": "data/coco/train2014",
        "JAG_MODELS_DIR": "models",
        "JAG_OUTPUTS_DIR": "outputs",
        "JAG_CACHE_DIR": ".cache",
    }
    for name, default in defaults.items():
        path = Path(environment.get(name, default)).expanduser()
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        environment[name] = str(path)
        if name != "REFCOCO_IMAGES_DIR":
            path.mkdir(parents=True, exist_ok=True)
    # Build/config do not require the COCO mount. Runtime does.
    if arguments[0] in {"run", "up", "create"} and not Path(environment["REFCOCO_IMAGES_DIR"]).is_dir():
        raise FileNotFoundError("Set REFCOCO_IMAGES_DIR in .env to your existing COCO train2014 directory")
    specification = {"services": {"jag": {"deploy": {"resources": {
        "reservations": {"devices": [{"driver": "nvidia", "device_ids": ids,
                                       "capabilities": ["gpu"]}]}
    }}}}}
    descriptor, filename = tempfile.mkstemp(prefix="jag-compose-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(specification, stream)
        command = ["docker", "compose", "--project-directory", str(ROOT),
                   "-f", str(ROOT / "docker/compose.yaml"), "-f", filename, *arguments]
        if arguments[0] != "config":
            access = subprocess.run(["docker", "info"], env=environment,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if access.returncode:
                try:
                    group = grp.getgrnam("docker")
                    import pwd
                    member = pwd.getpwuid(os.getuid()).pw_name in group.gr_mem
                except KeyError:
                    member = False
                if member and group.gr_gid not in os.getgroups():
                    # shlex.join uses POSIX quoting, including multiline arguments.
                    command = ["sg", "docker", "-c", shlex.join(command)]
        return subprocess.run(command, env=environment, cwd=ROOT).returncode
    finally:
        Path(filename).unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(launch(sys.argv[1:]))
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
