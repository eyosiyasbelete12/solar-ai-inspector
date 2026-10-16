"""Environment + configuration sanity check.

Run from the repository root:
    python -m scripts.check_setup

Exit code 0 = everything needed for the next step is in place, 1 = something is missing.
"""
from __future__ import annotations

import importlib
import sys

from common.config import PROJECT_ROOT, get_device, load_yaml
from common.datasets import DATASETS_CONFIG, load_dataset_specs, validate_spec

# (label, [import names to try], required?)
PACKAGES = [
    ("numpy", ["numpy"], True),
    ("pyyaml", ["yaml"], True),
    ("opencv", ["cv2"], True),
    ("pillow", ["PIL"], True),
    ("scikit-learn", ["sklearn"], True),
    ("matplotlib", ["matplotlib"], True),
    ("torch", ["torch"], True),
    ("torchvision", ["torchvision"], True),
    ("ultralytics", ["ultralytics"], True),
    ("fastapi", ["fastapi"], False),        # needed from Step 6
    ("uvicorn", ["uvicorn"], False),        # needed from Step 6
    ("python-multipart", ["python_multipart", "multipart"], False),  # needed from Step 6
]

CONFIGS = {
    "configs/detector.yaml": ["seed", "dataset", "model", "train", "inference"],
    "configs/classifier.yaml": ["seed", "dataset", "class_names", "model", "train", "inference"],
    "configs/datasets.yaml": ["datasets"],
}

problems: list[str] = []


def line(label: str, value: str, status: str) -> None:
    print(f"  {label:<26}{value:<32}{status}")


def check_python() -> None:
    v = sys.version_info
    ok = v >= (3, 10)
    line("python", f"{v.major}.{v.minor}.{v.micro}", "OK" if ok else "FAIL (need >= 3.10)")
    if not ok:
        problems.append("Python 3.10 or newer is required")


def check_packages() -> None:
    for label, names, required in PACKAGES:
        module = None
        for name in names:
            try:
                module = importlib.import_module(name)
                break
            except ImportError:
                continue
        if module is None:
            status = "MISSING" if required else "missing (needed later)"
            line(label, "-", status)
            if required:
                problems.append(f"{label} is not installed")
        else:
            line(label, str(getattr(module, "__version__", "installed")), "OK")


def check_device() -> None:
    device = get_device("auto")
    detail = device
    try:
        import torch

        if device == "cuda":
            detail = f"cuda ({torch.cuda.get_device_name(0)})"
        elif not torch.cuda.is_available():
            detail = f"{device} (no CUDA-enabled PyTorch/GPU found)"
    except ImportError:
        detail = "cpu (PyTorch not installed)"
    line("compute device", detail, "OK")


def check_configs() -> None:
    for path, required_keys in CONFIGS.items():
        try:
            cfg = load_yaml(path)
        except Exception as e:  # noqa: BLE001 - report any parse problem
            line(path, "-", f"FAIL ({e})")
            problems.append(f"{path} could not be read")
            continue
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            line(path, "-", f"FAIL (missing keys: {', '.join(missing)})")
            problems.append(f"{path} is missing keys: {', '.join(missing)}")
        else:
            line(path, "-", "OK")


def check_datasets() -> None:
    try:
        specs = load_dataset_specs(DATASETS_CONFIG)
    except Exception as e:  # noqa: BLE001
        print(f"  could not read dataset registry: {e}")
        problems.append("dataset registry is invalid")
        return
    if not specs:
        print("  No datasets registered yet (expected at Step 1; done in Step 2).")
    for key, spec in specs.items():
        issues = validate_spec(spec)
        print(f"  {key} [{spec.task}]: {len([i for i in issues if i.level == 'error'])} error(s), "
              f"{len([i for i in issues if i.level == 'warning'])} warning(s)")
        for issue in issues:
            print(f"      {issue}")

    # A dataset that a config actually points at must be error-free.
    for cfg_path in ("configs/detector.yaml", "configs/classifier.yaml"):
        try:
            key = load_yaml(cfg_path).get("dataset")
        except Exception:  # noqa: BLE001 - already reported in check_configs
            continue
        if key is None:
            print(f"  {cfg_path}: dataset not set yet")
        elif key not in specs:
            print(f"  {cfg_path}: dataset '{key}' is not in the registry")
            problems.append(f"{cfg_path} references unknown dataset '{key}'")
        elif any(i.level == "error" for i in validate_spec(specs[key])):
            problems.append(f"dataset '{key}' (used by {cfg_path}) has errors")


def check_dirs() -> None:
    for d in ("data", "models", "results"):
        ok = (PROJECT_ROOT / d).is_dir()
        line(d + "/", "-", "OK" if ok else "MISSING")
        if not ok:
            problems.append(f"directory {d}/ is missing")


def main() -> int:
    print("Solar AI Inspector - setup check")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}\n")
    print("Environment")
    check_python()
    check_packages()
    check_device()
    print("\nDirectories")
    check_dirs()
    print("\nConfig files")
    check_configs()
    print("\nDatasets")
    check_datasets()
    print("\n" + "=" * 60)
    if problems:
        print(f"RESULT: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("Fix these (see README 'Installation'), then re-run.")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
