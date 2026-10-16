"""Train the solar panel detector (Ultralytics YOLO).

Run from the repository root:
    python -m training.train_detector [--data PATH] [--config configs/detector.yaml]
"""
from __future__ import annotations
import argparse, shutil, sys
from common.config import get_device, load_yaml, resolve_path, set_seed

DEFAULT_DATA_YAML = "/content/data/yolo_detector/data.yaml"  # where Cell 14 wrote it

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--data", default=None, help="Path to data.yaml; defaults to the Colab dataset from Cell 14")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_yaml = args.data or cfg.get("data_yaml") or DEFAULT_DATA_YAML
    print(f"Data config: {data_yaml}")

    t = cfg["train"]
    set_seed(cfg.get("seed", 42), deterministic=t.get("deterministic", False))
    device = get_device(t.get("device", "auto"))
    print(f"Using device: {device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: run `pip install -q ultralytics` first.")
        return 1

    model = YOLO(cfg["model"])
    model.train(
        data=data_yaml, epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
        lr0=t["lr0"], patience=t["patience"], workers=t["workers"], device=device,
        seed=cfg.get("seed", 42), project=str(resolve_path(t["output_dir"])),
        name=t["run_name"], exist_ok=True,
    )

    metrics = model.val(data=data_yaml, split="val")
    print("\n=== Validation metrics ===")
    print(f"precision : {metrics.box.mp:.4f}")
    print(f"recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")

    src_best = resolve_path(t["output_dir"]) / t["run_name"] / "weights" / "best.pt"
    dst_best = resolve_path("models/detector/best.pt")
    dst_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_best, dst_best)
    print(f"\nWeights copied to: {dst_best}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
