"""Load a TOML config into a plain dict.

Usage:
    from llmbr.config import load_config
    cfg = load_config("configs/small.toml")
    cfg["model"]["emb_dim"]  # -> 384
"""

import tomllib  # standard library since Python 3.11
from pathlib import Path


def load_config(path: str | Path = "configs/small.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
