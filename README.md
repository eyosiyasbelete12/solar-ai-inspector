# Solar AI Inspector

Upload an aerial (drone) image of a solar farm and get back how many panels are CLEAN and how many are DUSTY.

## How it works
1. **Panel detector** - YOLO11n (one class, solar_panel) trained on the Duke UAV solar PV dataset.
2. **Clean/dusty classifier** - efficientnet_b0 (ImageNet-pretrained, fine-tuned) trained on 3,176 images: 2,562 Kaggle dust images + 614 PV Soiling images.
3. **Inference** - detect panels, crop each one from the original image, classify each crop, and save an annotated image plus a JSON report.

## Quick start (uses the weights in the pretrained folder)

    pip install -r requirements.txt ultralytics
    python -m inference.pipeline --image path/to/aerial_photo.jpg

Output: results/inference/NAME_annotated.jpg and NAME_results.json (panel count, clean count, dusty count, per-panel boxes and confidences).
An example is in reports/inference_example.jpg.

## Training

    python -m training.train_detector
    python -m training.evaluate_detector
    python -m training.train_classifier
    python -m training.evaluate_classifier

## Data preparation
- **Detector:** segmentation masks are converted to YOLO boxes (connected components, minimum area 20 px). Split by flight zone: zone B = validation, other zones = train, Blackwood field = test.
- **Classifier:** near-duplicate images are grouped using image embeddings (cosine similarity > 0.98) and split 70/15/15 by group (2,217 train / 461 val / 498 test), so duplicates never cross splits. Class weights balance clean vs dusty.

## Results
**Detector** (held-out Blackwood test set, 84 images, 420 panels): precision 0.639, recall 0.598, mAP50 0.523, mAP50-95 0.232.

**Classifier** (held-out test set, 498 images): accuracy 0.896.

| | precision | recall | f1 | support |
|---|---|---|---|---|
| clean | 0.902 | 0.895 | 0.898 | 256 |
| dusty | 0.889 | 0.897 | 0.893 | 242 |

Confusion matrix (rows = true, cols = predicted; clean, dusty): [[229, 27], [25, 217]].
More in the reports folder.

## Limitations
- The classifier was trained on close-up soiling photos, while panels in high-altitude drone images are only a few tens of pixels wide. Training uses a low-resolution augmentation to help, but there are no clean/dusty labels on aerial images, so end-to-end accuracy on aerial photos has not been measured.
- The detector's recall is about 60% on the held-out field, so some panels will be missed.

## Datasets (both CC BY 4.0)
- UAV-based solar photovoltaic detection dataset (figshare 18093890)
- Image-based identification of PV soiling (figshare 26977123)
- Kaggle solar panel dust dataset - check its license before redistributing
