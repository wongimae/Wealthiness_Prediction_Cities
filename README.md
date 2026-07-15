# README

## Overview
This project implements a Multi-gate Mixture-of-Experts (MMOE) model with Swin Transformer for fine-grained image-based wealthiness classification, trained on pairwise street-view comparisons from Place Pulse 2.0. The pipeline spans raw data assembly (`prepare_data.py`), TrueSkill-based label scoring and dataset construction (`data.py`), model definition and training (`model.py`, `train.py`), evaluation (`analyze.py`), and inference on new images (`predict.py`).

---

## Pipeline

**1. Environment setup**
```bash
conda activate torchenv   # or: pip install -r requirements.txt in a fresh venv
```

**2. Get the raw dataset**
```bash
curl -L "https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1" -o data/place-pulse-2.0.zip
unzip data/place-pulse-2.0.zip -d data/
```
Produces `data/place-pulse-2.0/` with `votes.tsv`, `locations.tsv`, `images/`, etc.

**3. Build the training dataframe**
```bash
python prepare_data.py --pp_dir data/place-pulse-2.0 --out_path data/wealthy_votes.pkl
```
Filters `votes.tsv` to the "wealthy" study, joins with `locations.tsv` (lat/lon) and image filenames → `data/wealthy_votes.pkl` (raw pairwise votes, ready for TrueSkill scoring).

**4. Train**
```bash
python train.py --epochs 10 --train_batch_size 8 --learning_rate 2e-5
```
Internally: `data.py` runs the TrueSkill algorithm on the votes to get continuous wealthiness scores, buckets them into 3 classes, splits train/val/test, and `model.py` trains the MMoE+Swin classifier with class-weighted loss, early stopping, and epoch checkpoints saved to `runs/wealthiness/`. Final best model → `runs/wealthiness/final`.

**5. Evaluate / diagnose**
```bash
python analyze.py
```
Loads `runs/wealthiness/final`, runs it on the held-out test set, prints a confusion matrix + per-class precision/recall/F1 — use this to sanity-check the model isn't collapsing to one class before trusting it.

**6. Run inference on new images**
```bash
python predict.py <directory_or_image_or_glob> --checkpoint runs/wealthiness/final \
  --output_csv predictions.csv --batch_size 32
```
Applies the same CLAHE preprocessing used in training, outputs `file_name, class, lat, lon` per image (lat/lon parsed from filename).

---

## File Descriptions

### 0. Place Pulse 2.0 Dataset
Raw votes, locations, and images (~2.64 GB) are mirrored on Dropbox (sourced from the `aleksandrskoselevs/place-pulse-dataset` GitHub repo):

```bash
curl -L "https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1" -o data/place-pulse-2.0.zip
unzip data/place-pulse-2.0.zip -d data/
```

This extracts to `data/place-pulse-2.0/`, containing `votes.tsv`, `locations.tsv`, `places.tsv`, `studies.tsv`, `qscores.tsv`, and `images/`. Run `prepare_data.py` afterward to build the wealthy-study votes dataframe used for training.

### 1. `prepare_data.py`
- **Purpose:** Builds the raw pairwise-vote dataframe from the Place Pulse 2.0 dump.
- **Key Features:**
  - Filters `votes.tsv` to the "wealthy" comparison study (`WEALTHY_STUDY_ID`).
  - Joins votes to image file paths and (lat, lon) locations via `locations.tsv` and the `images/` directory.
  - Writes `data/wealthy_votes.pkl` (pickle, not CSV, since locations are tuples).

### 2. `data.py`
- **Purpose:** Turns pairwise votes into a labeled image dataset ready for training.
- **Key Features:**
  - **TrueSkill algorithm** (`update_trueskill_scores` and friends): Converts pairwise win/loss/draw comparisons into a continuous per-image score, weighted by temporal decay (`compute_temporal_decay`) and spatial influence (`compute_spatial_influence`).
  - **`preprocess_csv`:** Runs the TrueSkill pass over all votes, then buckets scores into 2 or 3 wealthiness classes by mean/std thresholds.
  - **`RegressionDataset`:** PyTorch `Dataset` that loads images, applies CLAHE contrast enhancement, and runs the train/val/test transform pipeline.
  - **`build_dataset_and_feacture`:** End-to-end entry point — loads the pickle, scores/labels it, and returns train/val/test `Dataset`s plus the model's feature extractor.

### 3. `model.py`
- **Purpose:** Defines the MMOE model and its Hugging Face `Trainer`-based training loop.
- **Key Features:**
  - **`MMOEModel`:** MMOE architecture (5 experts, 3 gated task towers by default) on top of a pre-trained Swin Transformer backbone.
  - **`build_classification_trainer`:** Builds inverse-frequency class weights (so training can't collapse to the majority class), then configures a `Trainer` with cosine LR scheduling, fp16, early stopping (patience 3), and epoch checkpointing.
  - **`CheckpointProgressCallback`:** Prints plain-text (non-`\r`) progress toward the next checkpoint, so progress is readable via `tail -f` on a log file.

### 4. `train.py`
- **Purpose:** CLI entry point that wires `data.py` and `model.py` together, trains, and evaluates on the held-out test set. See Pipeline step 4 for usage.

### 5. `analyze.py`
- **Purpose:** Loads a trained checkpoint (`runs/wealthiness/final` by default) and reports a confusion matrix plus per-class precision/recall/F1 on the test set — use before trusting a model to confirm it isn't collapsing to one class.

### 6. `predict.py`
- **Purpose:** Runs inference on new images (a file, directory, or glob pattern), applying the same CLAHE preprocessing used in training. Outputs `file_name, class, lat, lon` per image, either printed or written to a CSV (lat/lon parsed from the filename).

### 7. `requirements.txt`
- **Purpose:** Pinned-free dependency list (`torch`, `torchvision`, `transformers`, `accelerate`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `opencv-python`, `Pillow`, `tqdm`, `geopy`) for setting up a fresh venv per Pipeline step 1.

