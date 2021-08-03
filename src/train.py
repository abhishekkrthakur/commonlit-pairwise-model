import argparse
import os
import pandas as pd
import tez
import transformers

from dataset import CommonlitDataset
from model import CommonlitModel
import numpy as np
import random
import torch
import itertools
from torch.utils.data import Sampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str, default="microsoft/deberta-large")
    parser.add_argument("--learning_rate", type=float, default=1, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--epochs", type=int, default=200, required=False)
    parser.add_argument("--max_len", type=int, default=256, required=False)
    parser.add_argument("--output_folder", type=str, default="../models/")
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--num_samples", type=int, required=True, default=5000)
    return parser.parse_args()


class RandomSampler(Sampler):
    # via Adam Montgomerie
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer " "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)
    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(
        args.output_folder,
        f"{args.model.replace('/',':')}__fold_{args.fold}.bin",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    df = pd.read_csv("../input/train_folds.csv")

    # base string is excerpt where target is 0 in the dataframe
    base_string = df.loc[df.target == 0, "excerpt"].values[0]

    # create dictionary out of excerpt and target columns from dataframe
    target_dict = dict(zip(df.excerpt.values.tolist(), df.target.values.tolist()))
    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)
    training_pairs = list(itertools.combinations(df_train.excerpt.values.tolist(), 2))

    # randomize training_pairs
    random.shuffle(training_pairs)
    validation_pairs = [(base_string, k) for k in df_valid.excerpt.values.tolist()]
    train_dataset = CommonlitDataset(
        excerpts=training_pairs,
        target_dict=target_dict,
        tokenizer=tokenizer,
        max_len=args.max_len,
        num_samples=args.num_samples,
    )
    valid_dataset = CommonlitDataset(
        excerpts=validation_pairs,
        target_dict=target_dict,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    if args.learning_rate == 1:
        n_train_steps = int(args.num_samples / args.batch_size * 20)
    else:
        n_train_steps = int(args.num_samples / args.batch_size * args.epochs)
    model = CommonlitModel(
        model_name=args.model,
        num_train_steps=n_train_steps,
        learning_rate=args.learning_rate,
        steps_per_epoch=args.num_samples / args.batch_size,
    )
    es = tez.callbacks.EarlyStopping(
        monitor="valid_rmse",
        model_path=output_path,
        save_weights_only=True,
        patience=args.early_stopping_patience,
    )
    train_sampler = RandomSampler(train_dataset, num_samples=args.num_samples)
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_sampler=train_sampler,
        train_bs=args.batch_size,
        valid_bs=64,
        device="cuda",
        epochs=args.epochs,
        callbacks=[es],
        fp16=True,
        train_shuffle=False,
    )
