import argparse
import csv
import glob
import os
import re

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from data import apply_clahe, build_transform
from model import MMOEModel

CLASS_NAMES = {0: "Impoverished", 1: "Middle", 2: "Affluent"}

LAT_LON_RE = re.compile(r"(-?\d+\.\d{4,})_(-?\d+\.\d{4,})")


def extract_lat_lon(filename):
    match = LAT_LON_RE.search(os.path.basename(filename))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def load_model(checkpoint, model_name, device):
    base_model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=3, ignore_mismatched_sizes=True
    )
    model = MMOEModel(base_model=base_model, num_experts=5, num_tasks=3, hidden_dim=512)

    if os.path.isfile(checkpoint):
        weights_path = checkpoint
    else:
        weights_path = os.path.join(checkpoint, "model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(checkpoint, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights found at {checkpoint}")

    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


IMAGE_EXTENSIONS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")


def collect_image_paths(patterns):
    paths = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            for ext in IMAGE_EXTENSIONS:
                paths.extend(glob.glob(os.path.join(pattern, ext)))
        else:
            matches = glob.glob(pattern)
            paths.extend(matches if matches else [pattern])
    return sorted(paths)


def preprocess_image(path, transform):
    img = Image.open(path).convert("RGB")
    img = apply_clahe(img)
    return transform(img)


def predict_batch(model, transform, image_paths, device):
    pixel_values = torch.stack([preprocess_image(p, transform) for p in image_paths]).to(device)
    with torch.no_grad():
        logits = model(pixel_values)["logits"]
        preds = logits.argmax(dim=-1).cpu()
    return [CLASS_NAMES[int(p)] for p in preds]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image file paths, directories, or glob patterns")
    parser.add_argument("--checkpoint", default="runs/wealthiness/final",
                         help="Path to a trained checkpoint directory (or a weights file directly)")
    parser.add_argument("--model_name", default="microsoft/swin-large-patch4-window12-384-in22k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_csv", default=None,
                         help="If set, write file_name,class,lat,lon rows here instead of printing per image")
    return parser.parse_args()


def main():
    args = parse_args()

    image_paths = collect_image_paths(args.images)
    print(f"Found {len(image_paths)} images")

    feature_extractor = AutoImageProcessor.from_pretrained(args.model_name)
    _, _, test_transform = build_transform(None, feature_extractor.image_mean, feature_extractor.image_std)

    model = load_model(args.checkpoint, args.model_name, args.device)

    rows = []
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i:i + args.batch_size]
        labels = predict_batch(model, test_transform, batch_paths, args.device)
        for path, label in zip(batch_paths, labels):
            lat, lon = extract_lat_lon(path)
            rows.append((os.path.basename(path), label, lat, lon))
            if args.output_csv is None:
                print(f"{path}: {label} (lat={lat}, lon={lon})")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "class", "lat", "lon"])
            writer.writerows(rows)
        print(f"Wrote {len(rows)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
