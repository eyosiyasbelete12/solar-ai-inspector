"""Combined inference: detect panels, crop each from the ORIGINAL high-res image,
classify each crop as clean/dusty.

Usage (from the repository root):
    python -m inference.pipeline --image path/to/photo.jpg
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from common.config import get_device, load_yaml, resolve_path
from training.train_classifier import build_model, build_transforms

ROOT = Path(__file__).resolve().parents[1]

def pick(configured, fallback):
    # use the configured weights if present, else the ones shipped in pretrained/
    return str(configured) if Path(configured).exists() else str(ROOT / fallback)

def load_classifier(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_name"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    transform = build_transforms(checkpoint["image_size"], train=False)
    return model, transform, checkpoint["class_names"]

def run_pipeline(image_path, detector, classifier, classifier_transform, class_names,
                  device, det_conf, det_iou, crop_padding):
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h_img, w_img = orig.shape[:2]

    results = detector.predict(str(image_path), conf=det_conf, iou=det_iou, device=device, verbose=False)[0]

    panels = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        det_conf_i = float(box.conf[0])
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * crop_padding, bh * crop_padding
        cx1, cy1 = max(int(x1 - pad_x), 0), max(int(y1 - pad_y), 0)
        cx2, cy2 = min(int(x2 + pad_x), w_img), min(int(y2 + pad_y), h_img)
        crop = orig[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = classifier_transform(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(classifier(tensor), dim=1)[0]
            cls_idx = int(probs.argmax())
            cls_conf = float(probs[cls_idx])

        panels.append({
            "id": i + 1,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "crop_size_px": [cx2 - cx1, cy2 - cy1],
            "detection_confidence": round(det_conf_i, 4),
            "condition": class_names[cls_idx].upper(),
            "classification_confidence": round(cls_conf, 4),
        })

    return {
        "image": str(image_path),
        "image_size_px": [w_img, h_img],
        "panels_detected": len(panels),
        "clean": sum(1 for p in panels if p["condition"] == "CLEAN"),
        "dusty": sum(1 for p in panels if p["condition"] == "DUSTY"),
        "panels": panels,
    }, orig

def draw_annotated(orig, result):
    img = orig.copy()
    colors = {"CLEAN": (0, 200, 0), "DUSTY": (0, 0, 230)}
    for p in result["panels"]:
        x1, y1, x2, y2 = p["bbox"]
        color = colors.get(p["condition"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"#{p['id']} {p['condition']} {p['classification_confidence']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--classifier-config", default="configs/classifier.yaml")
    parser.add_argument("--out-dir", default="results/inference")
    args = parser.parse_args()

    det_cfg = load_yaml(args.detector_config)
    cls_cfg = load_yaml(args.classifier_config)
    device = get_device(det_cfg["train"].get("device", "auto"))

    detector = YOLO(pick(resolve_path(det_cfg["inference"]["weights"]), "pretrained/detector_best.pt"))
    classifier, cls_transform, class_names = load_classifier(
        pick(resolve_path(cls_cfg["inference"]["weights"]), "pretrained/classifier_best.pt"), device)

    result, orig = run_pipeline(
        args.image, detector, classifier, cls_transform, class_names, device,
        det_conf=det_cfg["inference"]["conf_threshold"],
        det_iou=det_cfg["inference"]["iou_threshold"],
        crop_padding=det_cfg["inference"].get("crop_padding", 0.02),
    )

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    with open(out_dir / f"{stem}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"), draw_annotated(orig, result))

    print(f"Panels detected: {result['panels_detected']}")
    print(f"Clean: {result['clean']}  Dusty: {result['dusty']}")
    print(f"Results JSON : {out_dir / f'{stem}_results.json'}")
    print(f"Annotated img: {out_dir / f'{stem}_annotated.jpg'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
