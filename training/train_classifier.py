from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "/content/drive/MyDrive/solar-ai-inspector-work/data_cache/classifier_split_manifest.csv"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DEFAULTS = dict(class_names=["clean", "dusty"], seed=42,
    model=dict(name="efficientnet_b0", pretrained=True, image_size=224),
    train=dict(epochs=15, batch_size=32, learning_rate=3e-4, weight_decay=1e-4, num_workers=2),
    inference=dict(weights="models/classifier/best.pt"))

def load_cfg(path):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    p = ROOT / path
    if p.exists():
        import yaml
        user = yaml.safe_load(open(p)) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict): cfg[k].update(v)
            else: cfg[k] = v
    return cfg

class RandomLowRes:
    """Simulate small aerial crops (~20-64 px) that get upscaled to model input."""
    def __init__(self, p=0.5, lo=20, hi=64): self.p, self.lo, self.hi = p, lo, hi
    def __call__(self, img):
        if random.random() > self.p: return img
        s = random.randint(self.lo, self.hi)
        return img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BICUBIC)

class ManifestDataset(Dataset):
    def __init__(self, manifest_path, split, class_names, transform):
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        with open(manifest_path) as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split='{split}' in {manifest_path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(Image.open(r["path"]).convert("RGB")), self.class_to_idx[r["label"]]

def build_transforms(size, train, lowres_aug=True):
    if train:
        steps = ([RandomLowRes()] if lowres_aug else []) + [
            transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        return transforms.Compose(steps)
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def build_model(name, num_classes, pretrained):
    w = "DEFAULT" if pretrained else None
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); return m
    if name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        m = getattr(models, name)(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    raise ValueError(f"Unknown model: {name}")

def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0); correct += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return tot_loss / n, correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--no-lowres-aug", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    manifest = a.manifest or DEFAULT_MANIFEST
    names, t, m = cfg["class_names"], cfg["train"], cfg["model"]
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Manifest:", manifest, "| Device:", device, "| Classes:", names)

    tr = ManifestDataset(manifest, "train", names, build_transforms(m["image_size"], True, not a.no_lowres_aug))
    va = ManifestDataset(manifest, "val", names, build_transforms(m["image_size"], False))
    tl = DataLoader(tr, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], pin_memory=True)
    vl = DataLoader(va, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], pin_memory=True)

    counts = [sum(1 for r in tr.rows if r["label"] == c) for c in names]
    w = torch.tensor([sum(counts) / (len(names) * c) for c in counts], dtype=torch.float32).to(device)
    print("Train counts:", dict(zip(names, counts)), "| weights:", w.tolist())

    model = build_model(m["name"], len(names), m.get("pretrained", True)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])

    best, best_path = -1.0, ROOT / cfg["inference"]["weights"]
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, t["epochs"] + 1):
        trl, tra = run_epoch(model, tl, device, crit, opt)
        val, vaa = run_epoch(model, vl, device, crit)
        sched.step()
        print(f"Epoch {ep:2d}/{t['epochs']}  train_loss={trl:.4f} acc={tra:.4f}  val_loss={val:.4f} acc={vaa:.4f}")
        if vaa > best:
            best = vaa
            torch.save({"model_state_dict": model.state_dict(), "class_names": names,
                        "model_name": m["name"], "image_size": m["image_size"]}, best_path)
            print(f"  -> best, saved to {best_path}")
    print(f"\nBest val acc: {best:.4f}\nWeights: {best_path}")

if __name__ == "__main__":
    sys.exit(main())
