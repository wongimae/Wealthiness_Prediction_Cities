import multiprocessing as mp
mp.set_start_method("fork", force=True)

import argparse

from data import build_dataset_and_feacture
from model import build_classification_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--csv_path", default="data/wealthy_votes.pkl")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--model_name", default="microsoft/swin-large-patch4-window12-384-in22k")
    parser.add_argument("--log_dir", default="runs/wealthiness")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    (train_set, val_set, test_set), feature_extractor = build_dataset_and_feacture(args)
    trainer, model = build_classification_trainer(args, train_set, val_set, feature_extractor)
    trainer.train()
    trainer.save_model(args.log_dir + "/final")
    metrics = trainer.evaluate(test_set)
    print(metrics)


if __name__ == "__main__":
    main()
