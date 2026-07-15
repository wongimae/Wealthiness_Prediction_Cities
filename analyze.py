import multiprocessing as mp
mp.set_start_method("fork", force=True)

import argparse

from sklearn.metrics import confusion_matrix, classification_report

from data import build_dataset_and_feacture
from model import build_classification_trainer

args = argparse.Namespace(
    root=".", csv_path="data/wealthy_votes.pkl", num_classes=3,
    model_name="microsoft/swin-large-patch4-window12-384-in22k",
    log_dir="runs/wealthiness", learning_rate=2e-5,
    train_batch_size=8, eval_batch_size=8, epochs=10,
)
(train_set, val_set, test_set), fe = build_dataset_and_feacture(args)
trainer, model = build_classification_trainer(args, train_set, val_set, fe)

import torch, safetensors.torch
state_dict = safetensors.torch.load_file("runs/wealthiness/final/model.safetensors")
model.load_state_dict(state_dict)

output = trainer.predict(test_set)
preds = output.predictions.argmax(axis=-1)
labels = output.label_ids

print("Prediction distribution:", {int(c): int((preds == c).sum()) for c in [0, 1, 2]})
print("True label distribution:", {int(c): int((labels == c).sum()) for c in [0, 1, 2]})
print()
print(confusion_matrix(labels, preds))
print()
print(classification_report(labels, preds, target_names=["Impoverished", "Middle", "Affluent"]))
