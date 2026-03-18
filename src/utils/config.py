from __future__ import annotations

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
    return get_project_root(config).joinpath("outputs", *parts)


def get_config_name(config: ConfigDict) -> str:
    return Path(config["config_path"]).stem
