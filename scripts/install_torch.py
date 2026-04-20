from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cuda_env import audit_cuda_environment, detect_system_cuda


PYTORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", *sorted(PYTORCH_INDEX)], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-cuda-if-available", action="store_true")
    parser.add_argument("--force-reinstall", action="store_true")
    args = parser.parse_args()

    system_info = detect_system_cuda()
    target_mode = system_info.recommended_channel if args.mode == "auto" else args.mode
    if target_mode not in PYTORCH_INDEX:
        raise SystemExit(f"Unsupported mode: {target_mode}")

    current = audit_cuda_environment()
    current_torch = current["torch"]
    current_system = current["system"]
    if (
        current_torch["installed"]
        and current_torch["cuda_available"] == (target_mode != "cpu")
        and _channel_matches(current_torch["torch_cuda"], target_mode)
        and not args.force_reinstall
    ):
        print(f"torch already matches target mode {target_mode}")
        return

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio",
            "triton",
            "cuda-toolkit",
            "cuda-bindings",
            "nvidia-cudnn-cu13",
            "nvidia-cusparselt-cu13",
            "nvidia-nccl-cu13",
            "nvidia-nvshmem-cu13",
        ],
        check=False,
    )
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--index-url",
        PYTORCH_INDEX[target_mode],
        "torch",
    ]
    _run(install_cmd)

    refreshed = _audit_fresh()
    diagnosis = refreshed["diagnosis"]
    require_cuda = args.require_cuda or (
        args.require_cuda_if_available and current_system["recommended_channel"] != "cpu"
    )
    print(f"target_mode: {target_mode}")
    print(f"diagnosis: {diagnosis['status']}")
    print(f"reason: {diagnosis['reason']}")
    if require_cuda and diagnosis["status"] != "cuda_ready":
        raise SystemExit("CUDA installation did not become ready.")


def _channel_matches(torch_cuda: str | None, target_mode: str) -> bool:
    if target_mode == "cpu":
        return torch_cuda in (None, "")
    if not torch_cuda:
        return False
    normalized = torch_cuda.replace(".", "")
    return normalized == target_mode.replace("cu", "")


def _run(command: list[str], check: bool = True) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=check)


def _audit_fresh() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/check_cuda_env.py"), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


if __name__ == "__main__":
    main()
