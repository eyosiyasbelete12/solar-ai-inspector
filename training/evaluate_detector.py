"""Evaluate the trained solar panel detector on the held-out test set (Blackwood field).

Run from the repository root:
    python -m training.evaluate_detector
"""
from __future__ import annotations
import argparse, glob, json, random, sys
import cv2
import yaml as _yaml
import matplotlib.pyplot as plt
from common.config import get_device, load_yaml, resolve_path

DEFAULT_DATA_YAML = "/content/data/yolo_detector/data.yaml"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--n-examples", type=int, default=8)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_yaml = args.data or cfg.get("data_yaml") or DEFAULT_DATA_YAML
    weights = args.weights or str(resolve_path(cfg["inference"]["weights"]))
    conf = cfg["inference"].get("conf_threshold", 0.25)
    iou = cfg["inference"].get("iou_threshold", 0.5)
    out_dir = resolve_path(cfg["train"]["output_dir"]) / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg["train"].get("device", "auto"))

    print(f"Weights : {weights}")
    print(f"Data    : {data_yaml}")
    print(f"Device  : {device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: run `pip install -q ultralytics` first.")
        return 1

    with open(data_yaml) as f:
        dcfg = _yaml.safe_load(f)
    base = dcfg.get("path", "")
    test_rel = dcfg.get("test", "test/images")
    test_dir = test_rel if test_rel.startswith("/") else f"{base}/{test_rel}"
    test_images = sorted(glob.glob(f"{test_dir}/*"))
    n_instances = 0
    for p in test_images:
        label_path = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        with open(label_path) as f:
            n_instances += sum(1 for line in f if line.strip())

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split="test", conf=conf, iou=iou, device=device)

    report = {
        "split": "test (Blackwood field - never used in training or checkpoint selection)",
        "n_images": len(test_images),
        "n_instances": n_instances,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "conf_threshold": conf,
        "iou_threshold": iou,
    }
    print(f"\n=== TEST SET metrics: {report['n_images']} images, {report['n_instances']} instances ===")
    for k in ("precision", "recall", "mAP50", "mAP50-95"):
        print(f"{k:10s}: {report[k]:.4f}")

    report_path = out_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics saved to: {report_path}")

    random.seed(0)
    sample = random.sample(test_images, min(args.n_examples, len(test_images)))
    results = model.predict(sample, conf=conf, iou=iou, device=device, verbose=False)

    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flat if n > 1 else [axes]
    for ax, res in zip(axes, results):
        img = res.plot()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(res.path.split("/")[-1], fontsize=8)
        ax.axis("off")
    for ax in list(axes)[n:]:
        ax.axis("off")
    plt.tight_layout()
    examples_path = out_dir / "example_predictions.png"
    plt.savefig(examples_path, dpi=100)
    print(f"Example predictions saved to: {examples_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
