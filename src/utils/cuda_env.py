from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SystemCudaInfo:
    gpu_present: bool
    gpu_name: str | None
    driver_version: str | None
    max_cuda: str | None
    recommended_channel: str


@dataclass
class TorchCudaInfo:
    installed: bool
    torch_version: str | None
    torch_cuda: str | None
    cuda_available: bool
    device_count: int
    error: str | None


def detect_system_cuda() -> SystemCudaInfo:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return SystemCudaInfo(
            gpu_present=False,
            gpu_name=None,
            driver_version=None,
            max_cuda=None,
            recommended_channel="cpu",
        )

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    gpu_name = None
    driver_version = None
    if rows:
        parts = [part.strip() for part in rows[0].split(",")]
        if parts:
            gpu_name = parts[0]
        if len(parts) > 1:
            driver_version = parts[1]

    try:
        header = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        header = ""
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", header)
    max_cuda = match.group(1) if match else None
    return SystemCudaInfo(
        gpu_present=bool(rows),
        gpu_name=gpu_name,
        driver_version=driver_version,
        max_cuda=max_cuda,
        recommended_channel=recommend_pytorch_channel(max_cuda),
    )


def inspect_torch_cuda() -> TorchCudaInfo:
    try:
        import torch
    except ImportError:
        return TorchCudaInfo(
            installed=False,
            torch_version=None,
            torch_cuda=None,
            cuda_available=False,
            device_count=0,
            error="torch not installed",
        )

    torch_version = getattr(torch, "__version__", None)
    torch_cuda = getattr(torch.version, "cuda", None)
    device_count = 0
    cuda_available = False
    error = None
    try:
        device_count = int(torch.cuda.device_count())
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available and device_count > 0:
            torch.cuda.get_device_properties(0)
    except Exception as exc:  # pragma: no cover - environment dependent
        error = str(exc)
        cuda_available = False
    return TorchCudaInfo(
        installed=True,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        device_count=device_count,
        error=error,
    )


def recommend_pytorch_channel(max_cuda: str | None) -> str:
    version = _parse_version(max_cuda)
    if version >= (12, 8):
        return "cu128"
    if version >= (12, 6):
        return "cu126"
    if version >= (12, 4):
        return "cu124"
    return "cpu"


def audit_cuda_environment() -> dict[str, Any]:
    system_info = detect_system_cuda()
    torch_info = inspect_torch_cuda()
    diagnosis = diagnose_cuda_mismatch(system_info, torch_info)
    return {
        "python": sys.version.split()[0],
        "system": asdict(system_info),
        "torch": asdict(torch_info),
        "diagnosis": diagnosis,
    }


def diagnose_cuda_mismatch(system_info: SystemCudaInfo, torch_info: TorchCudaInfo) -> dict[str, Any]:
    if not system_info.gpu_present:
        return {
            "status": "cpu_only",
            "reason": "No NVIDIA GPU detected via nvidia-smi.",
            "action": "Install CPU-only torch or run scripts/install_torch.py --mode cpu.",
        }
    if not torch_info.installed:
        return {
            "status": "torch_missing",
            "reason": "PyTorch is not installed in this environment.",
            "action": f"Run scripts/install_torch.py --mode {system_info.recommended_channel}.",
        }
    if torch_info.cuda_available:
        return {
            "status": "cuda_ready",
            "reason": "PyTorch CUDA runtime initialized successfully.",
            "action": "No action required.",
        }

    expected_channel = system_info.recommended_channel
    installed_cuda = torch_info.torch_cuda or "cpu"
    if expected_channel != "cpu" and installed_cuda.startswith("13"):
        return {
            "status": "driver_too_old_for_installed_torch",
            "reason": (
                f"Installed torch targets CUDA {installed_cuda}, but the system driver exposes CUDA {system_info.max_cuda}."
            ),
            "action": f"Reinstall a matching torch wheel with scripts/install_torch.py --mode {expected_channel}.",
        }
    return {
        "status": "cuda_unavailable",
        "reason": torch_info.error or "PyTorch could not initialize CUDA.",
        "action": f"Run scripts/check_cuda_env.py and reinstall torch for {expected_channel} if needed.",
    }


def _parse_version(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    match = re.search(r"(\d+)(?:\.(\d+))?", value)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def audit_as_json() -> str:
    return json.dumps(audit_cuda_environment(), indent=2, sort_keys=True)
