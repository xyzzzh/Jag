#!/usr/bin/env python3
"""Render the author-supplied Jag framework PDF for the README."""

import argparse
from pathlib import Path
import subprocess


def render(root):
    source = root / "assets" / "architecture.pdf"
    if not source.is_file():
        raise FileNotFoundError(source)
    subprocess.run(
        ["pdftoppm", "-f", "1", "-l", "1", "-r", "300",
         "-singlefile", "-png", str(source), str(source.with_suffix(""))],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    render(args.root.resolve())


if __name__ == "__main__":
    main()
