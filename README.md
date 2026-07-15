# README

## Overview
This project implements a Multi-gate Mixture-of-Experts (MMOE) model with Swin Transformer for fine-grained image-based wealthiness classification. It includes two Python modules: `data.py` for data preprocessing and dataset creation, and `model.py` for the implementation of the MMOE model and its training pipeline.

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

### 1. `data.py`
- **Purpose:** Prepares the dataset and processes comparisons to compute TrueSkill scores.
- **Key Features:**
  - **`RegressionDataset`:** A custom PyTorch dataset to handle image preprocessing and loading.
  - **TrueSkill Algorithm:** Updates image scores based on temporal decay and spatial influence.
  - **`preprocess_csv`:** Processes input CSV data to compute wealthiness labels.
  - **Transforms:** Includes CLAHE preprocessing and feature extraction using a pre-trained model.

### 2. `model.py`
- **Purpose:** Defines the MMOE model and integrates it with the Swin Transformer for multi-task classification.
- **Key Features:**
  - **`MMOEModel`:** Implements the MMOE architecture with expert and gate networks.
  - **`build_classification_trainer`:** Configures and returns a Hugging Face `Trainer` for training the model.
  - **Base Model:** Utilizes a pre-trained Swin Transformer for feature extraction.

