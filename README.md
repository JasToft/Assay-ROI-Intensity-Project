# Lateral Flow Assay Signal Classifier

A lightweight Python pipeline that extracts time-series signals from synthetic lateral flow assay videos and trains a binary classifier to predict positive or negative results.

---

## Files

| File | What it does |
|---|---|
| `generate_video.py` | Generates synthetic test videos simulating strip darkening |
| `extract_signal.py` | Extracts and normalizes ROI intensity signals from videos |
| `train_classifier.py` | Trains and evaluates a binary classifier on the extracted signals |

---

## How to Run

**1. Generate synthetic videos**
```bash
python generate_video.py
```
Creates `pos1.mp4`, `pos2.mp4`, `neg1.mp4`, `neg2.mp4`.

**2. Extract signals**
```bash
python extract_signal.py
```
You will be prompted to draw a region of interest (ROI) on the first frame. The same ROI is reused for all videos. Outputs one `*_signal.csv` per video.

**3. Train the classifier**
```bash
python ML_pipeline.py
```
Prints per-sample predictions and a classification report.

---

## Features Used

Each video is represented by 4 numbers extracted from its normalized signal:

- **Peak intensity** — how dark the strip gets at its darkest
- **Total change** — difference in intensity from first to last frame
- **Mean intensity** — average darkness across all frames
- **Max slope** — steepest single-frame increase in darkness

---

## Model & Evaluation

A Logistic Regression classifier is trained on these features. Because the dataset is small, evaluation uses **leave-one-out cross-validation** — each sample is predicted by a model that was trained without it, giving an honest performance estimate.

---

## Dependencies

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib
```
