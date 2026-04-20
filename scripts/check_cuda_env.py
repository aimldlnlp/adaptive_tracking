from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cuda_env import audit_cuda_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args()

    payload = audit_cuda_environment()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        system = payload["system"]
        torch = payload["torch"]
        diagnosis = payload["diagnosis"]
        print(f"gpu_present: {system['gpu_present']}")
        print(f"gpu_name: {system['gpu_name']}")
        print(f"driver_version: {system['driver_version']}")
        print(f"system_cuda: {system['max_cuda']}")
        print(f"recommended_channel: {system['recommended_channel']}")
        print(f"torch_installed: {torch['installed']}")
        print(f"torch_version: {torch['torch_version']}")
        print(f"torch_cuda: {torch['torch_cuda']}")
        print(f"cuda_available: {torch['cuda_available']}")
        print(f"device_count: {torch['device_count']}")
        print(f"diagnosis: {diagnosis['status']}")
        print(f"reason: {diagnosis['reason']}")
        print(f"action: {diagnosis['action']}")

    if args.fail_if_not_ready and payload["diagnosis"]["status"] != "cuda_ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
