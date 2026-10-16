"""Dataset registry.

Datasets are described in configs/datasets.yaml (name, source URL, license, image
directory, annotation directory, class names, ...). Nothing here downloads data:
you obtain each dataset yourself after reading its license, then register it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from common.config import load_yaml, resolve_path

DATASETS_CONFIG = "configs/datasets.yaml"

TASKS = ("detection", "classification")
DETECTION_FORMATS = ("yolo", "coco", "voc")
CLASSIFICATION_FORMATS = ("folders",)  # one sub-folder per class
GROUP_BY_OPTIONS = ("none", "subdirectory", "filename_regex", "manifest_csv")
CLASSIFIER_CLASSES = ("clean", "dusty")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetSpec:
    key: str
    task: str
    name: str
    source_url: str
    license: str
    license_verified: bool
    image_dir: Path
    annotation_dir: Optional[Path]
    annotation_format: str
    class_names: list[str]
    # classification only: map source folder/label names -> "clean" / "dusty"
    class_map: dict[str, str] = field(default_factory=dict)
    # how related images (same scene / flight / sequence) are grouped before splitting:
    #   none            -> every image is its own group (only safe if images are independent)
    #   subdirectory    -> each immediate sub-folder name under a class folder is one group
    #   filename_regex  -> group_arg is a regex whose first capture group is the group id
    #   manifest_csv    -> group_arg is a CSV path with columns: filename,group
    group_by: str = "none"
    group_arg: Optional[str] = None
    # What do the labels actually depict? (e.g. "dust", "dust + bird droppings", "any dirt")
    label_semantics: str = ""
    notes: str = ""


@dataclass
class Issue:
    level: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


def _is_placeholder(value: Any) -> bool:
    return value is None or not str(value).strip() or "TODO" in str(value).upper()


def _spec_from_entry(key: str, entry: dict[str, Any]) -> DatasetSpec:
    missing = [k for k in ("name", "task", "image_dir", "class_names") if k not in entry]
    if missing:
        raise ValueError(f"Dataset '{key}' is missing required field(s): {', '.join(missing)}")
    ann = entry.get("annotation_dir")
    task = entry["task"]
    default_fmt = "folders" if task == "classification" else "yolo"
    return DatasetSpec(
        key=key,
        task=task,
        name=str(entry["name"]),
        source_url=str(entry.get("source_url") or ""),
        license=str(entry.get("license") or ""),
        license_verified=bool(entry.get("license_verified", False)),
        image_dir=resolve_path(entry["image_dir"]),
        annotation_dir=resolve_path(ann) if ann else None,
        annotation_format=str(entry.get("annotation_format", default_fmt)),
        class_names=[str(c) for c in entry["class_names"]],
        class_map={str(k): str(v) for k, v in (entry.get("class_map") or {}).items()},
        group_by=str(entry.get("group_by", "none")),
        group_arg=entry.get("group_arg"),
        label_semantics=str(entry.get("label_semantics") or ""),
        notes=str(entry.get("notes") or ""),
    )


def load_dataset_specs(config_path: str | Path = DATASETS_CONFIG) -> dict[str, DatasetSpec]:
    """Read every dataset entry from the registry file. Returns {key: DatasetSpec}."""
    raw = load_yaml(config_path)
    entries = raw.get("datasets") or {}
    if not isinstance(entries, dict):
        raise ValueError("'datasets' in the registry must be a mapping of key -> entry")
    return {key: _spec_from_entry(key, entry) for key, entry in entries.items()}


def get_dataset(key: str, config_path: str | Path = DATASETS_CONFIG) -> DatasetSpec:
    specs = load_dataset_specs(config_path)
    if key not in specs:
        available = ", ".join(sorted(specs)) or "(none registered yet)"
        raise KeyError(f"Dataset '{key}' not found in {config_path}. Registered: {available}")
    return specs[key]


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_spec(spec: DatasetSpec, check_files: bool = True) -> list[Issue]:
    """Return a list of problems with a dataset entry (errors block use, warnings do not)."""
    issues: list[Issue] = []
    err = lambda m: issues.append(Issue("error", m))  # noqa: E731
    warn = lambda m: issues.append(Issue("warning", m))  # noqa: E731

    if spec.task not in TASKS:
        err(f"task must be one of {TASKS}, got '{spec.task}'")
        return issues

    allowed_formats = DETECTION_FORMATS if spec.task == "detection" else CLASSIFICATION_FORMATS
    if spec.annotation_format not in allowed_formats:
        err(f"annotation_format for a {spec.task} dataset must be one of {allowed_formats}, "
            f"got '{spec.annotation_format}'")

    if not spec.class_names:
        err("class_names is empty")

    if spec.task == "detection" and spec.annotation_dir is None:
        err("annotation_dir is required for detection datasets")

    if spec.task == "classification":
        mapped = {spec.class_map.get(c, c) for c in spec.class_names}
        if mapped != set(CLASSIFIER_CLASSES):
            err(f"classification classes must map to exactly {CLASSIFIER_CLASSES}; "
                f"got {sorted(mapped)}. Use class_map to rename source labels.")
        bad_targets = set(spec.class_map.values()) - set(CLASSIFIER_CLASSES)
        if bad_targets:
            err(f"class_map targets must be 'clean' or 'dusty', got {sorted(bad_targets)}")
        if _is_placeholder(spec.label_semantics):
            warn("label_semantics is empty/TODO: state what the labels really depict "
                 "(dust? any dirt? bird droppings?) before claiming the model detects dust.")

    if spec.group_by not in GROUP_BY_OPTIONS:
        err(f"group_by must be one of {GROUP_BY_OPTIONS}, got '{spec.group_by}'")
    elif spec.group_by in ("filename_regex", "manifest_csv") and not spec.group_arg:
        err(f"group_arg is required when group_by='{spec.group_by}'")
    elif spec.group_by == "filename_regex" and spec.group_arg:
        try:
            if re.compile(spec.group_arg).groups < 1:
                err("group_arg regex needs at least one capture group, e.g. '^(flight\\d+)_'")
        except re.error as e:
            err(f"group_arg is not a valid regex: {e}")
    elif spec.group_by == "none":
        warn("group_by='none': images are treated as independent. If any come from the same "
             "scene/flight/sequence, splits will leak. Set group_by once you know.")

    if _is_placeholder(spec.source_url):
        warn("source_url is empty/TODO")
    if _is_placeholder(spec.license):
        warn("license is empty/TODO")
    elif not spec.license_verified:
        warn(f"license '{spec.license}' has not been verified (set license_verified: true "
             "only after reading the license text at the source)")

    if check_files:
        if not spec.image_dir.is_dir():
            err(f"image_dir does not exist: {spec.image_dir}")
        elif count_images(spec.image_dir) == 0:
            err(f"no images found under image_dir: {spec.image_dir}")
        if spec.annotation_dir is not None and not spec.annotation_dir.is_dir():
            err(f"annotation_dir does not exist: {spec.annotation_dir}")

    return issues
