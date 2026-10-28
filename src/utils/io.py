from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(payload: Any, path: str | Path) -> None:
    target = Path(path).resolve()
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path).resolve()
    ensure_dir(target.parent)
    frame.to_csv(target, index=False)


def save_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path).resolve()
    ensure_dir(target.parent)
    np.savez_compressed(target, **arrays)


def load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path).resolve(), allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def write_text(path: str | Path, text: str) -> None:
    target = Path(path).resolve()
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def list_npz_files(path: str | Path) -> list[Path]:
    return sorted(Path(path).resolve().glob("*.npz"))


def flatten_dict_rows(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.json_normalize(list(rows), sep=".")
