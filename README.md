


Solar AI Inspector

Drone-Based Solar Panel Inspection Using Computer Vision

Solar energy systems need regular maintenance to continue operating reliably. One challenge is soiling: dust and other particles accumulate on solar panels and can reduce the sunlight reaching their cells, potentially lowering electricity generation. Identifying affected panels is especially challenging across large solar installations, where inspecting every panel individually can require considerable time and effort.

Solar AI Inspector is a computer-vision project exploring how drones and artificial intelligence could make solar-panel inspection faster and more scalable. Given an aerial image of a solar farm, the system detects individual panels, examines each panel's image, and classifies it as clean or dusty. It then produces an annotated image and a report summarizing the predictions.

Why This Project Matters

Solar installations can contain hundreds or thousands of panels spread across large areas. An inspection approach that helps maintenance teams identify panels that may need attention could make it easier to prioritize follow-up inspections rather than examine every panel manually.

This matters in places where solar energy supports homes, businesses, farms, and essential services. In Ethiopia, expanding access to dependable energy is an important motivation for exploring practical renewable-energy technologies. I want to investigate whether accessible AI tools can help make maintaining solar infrastructure more efficient.

The system does not measure electricity output or prove how much energy is lost to dust. Instead, it focuses on a more immediate task: identifying panels that appear dusty in images and flagging them for further inspection.

Who Could Benefit?

- Solar-farm operators: Screen large installations and identify panels that may warrant closer inspection.
- Maintenance technicians: Use image-based predictions to help prioritize site visits and cleaning checks.
- Renewable-energy organizations: Explore ways to make visual inspection more efficient, particularly where installations cover large areas.
- Researchers and developers: Build on the project to investigate aerial computer vision, image quality, and automated solar-panel maintenance.

These are potential beneficiaries; the project's practical value still needs to be validated through further testing.

What Makes the Approach Interesting?

The project combines two computer-vision tasks that serve different purposes:

1. Panel detection: Locate individual solar panels in a drone image.
2. Soiling classification: Examine each detected panel and predict whether it appears clean or dusty.

An important question is whether a classifier trained on close-up images can work reliably when panels occupy far fewer pixels in aerial photographs. Solar AI Inspector therefore provides a starting point for investigating the limitations of drone-based inspection, not simply producing clean/dusty predictions.

Current Status and Limitations

The detector and classifier have been trained and evaluated separately on held-out datasets. The detector achieved a recall of 0.598 on its held-out field test, while the classifier achieved 89.6% accuracy on its held-out test set.

These results are encouraging starting points, but they do not establish the accuracy of the complete system on drone imagery. The classifier was trained on close-up soiling images, and end-to-end performance on aerial images has not yet been measured. The detector can also miss panels, so some panels may not receive a classification.

The next steps are to evaluate the integrated pipeline on suitable aerial imagery, investigate how image resolution and viewing conditions affect performance, and determine where the system needs improvement.

Long-Term Vision

I hope to develop this project into a practical, affordable tool that helps make solar-panel inspection more accessible. Starting with publicly available datasets, I want to investigate its potential relevance to solar installations in Ethiopia and, eventually, other regions facing similar inspection challenges.

The broader goal is to explore how AI can support the maintenance of renewable-energy infrastructure. Solar AI Inspector is an early step toward that goal, and its future usefulness will depend on rigorous testing, improvement, and validation in real-world conditions.

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
