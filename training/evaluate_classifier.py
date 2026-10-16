"""Evaluate the trained clean/dusty classifier on the held-out test set.

Run from the repository root:
    python -m training.evaluate_classifier
"""
from __future__ import annotations
import argparse, json, random, sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from common.config import get_device, load_yaml, resolve_path
from training.train_classifier import ManifestDataset, build_transforms, build_model, DEFAULT_MANIFEST

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--n-examples", type=int, default=8)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = args.manifest or cfg.get("manifest") or DEFAULT_MANIFEST
    class_names = cfg["class_names"]
    weights_path = args.weights or str(resolve_path(cfg["inference"]["weights"]))
    device = get_device(cfg["train"].get("device", "auto"))

    print(f"Manifest : {manifest}")
    print(f"Weights  : {weights_path}")
    print(f"Device   : {device}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_name"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    eval_tfm = build_transforms(checkpoint["image_size"], train=False)
    test_ds = ManifestDataset(manifest, "test", class_names, eval_tfm)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["train"]["num_workers"])

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(lbls.tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    print(f"\n=== TEST SET metrics: {len(all_labels)} images ===")
    print(f"Overall accuracy: {acc:.4f}\n")
    for i, name in enumerate(class_names):
        print(f"  {name:8s}: precision={precision[i]:.4f} recall={recall[i]:.4f} f1={f1[i]:.4f} support={support[i]}")
    print(f"\nConfusion matrix (rows=true, cols=predicted), order={class_names}:")
    print(cm)

    out_dir = resolve_path(cfg["train"]["output_dir"]) / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_test_images": len(all_labels),
        "accuracy": acc,
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_order": class_names,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics saved to: {out_dir / 'metrics.json'}")

    rows = test_ds.rows
    correct_idx = [i for i in range(len(rows)) if class_names[all_preds[i]] == rows[i]["label"]]
    incorrect_idx = [i for i in range(len(rows)) if class_names[all_preds[i]] != rows[i]["label"]]
    random.seed(0)
    n_each = args.n_examples // 2
    sample_idx = (random.sample(correct_idx, min(n_each, len(correct_idx)))
                  + random.sample(incorrect_idx, min(n_each, len(incorrect_idx))))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, idx in zip(axes.flat, sample_idx):
        row = rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        true_label, pred_label = row["label"], class_names[all_preds[idx]]
        conf = max(all_probs[idx])
        mark = "CORRECT" if true_label == pred_label else "WRONG"
        ax.imshow(img)
        ax.set_title(f"{mark}: true={true_label} pred={pred_label} ({conf:.2f})", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    examples_path = out_dir / "example_predictions.png"
    plt.savefig(examples_path, dpi=100)
    print(f"Example predictions saved to: {examples_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
