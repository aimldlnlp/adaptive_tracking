from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_config(config_path: str | Path) -> ConfigDict:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config: ConfigDict = yaml.safe_load(handle)
    config["config_path"] = str(path)
    config["project_root"] = str(path.parent.parent.resolve())
    return config


def get_project_root(config: ConfigDict) -> Path:
    return Path(config["project_root"]).resolve()


def get_output_dir(config: ConfigDict, *parts: str) -> Path:
    project_cfg = config.get("project", {})
    output_subdir = str(project_cfg.get("output_subdir", "")).strip().strip("/")
    base = get_project_root(config).joinpath("outputs")
    if output_subdir:
        base = base.joinpath(output_subdir)
    return base.joinpath(*parts)


def get_config_name(config: ConfigDict) -> str:
    return Path(config["config_path"]).stem


def merge_config(base: ConfigDict, overrides: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)
    _merge_into(merged, overrides)
    return merged


def _merge_into(target: ConfigDict, updates: ConfigDict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_into(target[key], value)
            continue
        target[key] = deepcopy(value)
