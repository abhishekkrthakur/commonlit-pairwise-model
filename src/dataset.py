from numpy import random
import torch
import numpy as np
import psutil


class CommonlitDataset:
    def __init__(self, excerpts, target_dict, error_dict, tokenizer, max_len, num_samples=None):
        self.excerpts = excerpts
        self.target_dict = target_dict
        self.error_dict = error_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_samples = num_samples
        self.count = 0

    def __len__(self):
        if self.num_samples is None:
            return len(self.excerpts)
        return self.num_samples

    def __getitem__(self, item):
        if self.num_samples is not None:
            self.count += 1
            if self.count >= self.num_samples / psutil.cpu_count():
                self.count = 0
                random.shuffle(self.excerpts)

        text1 = str(self.excerpts[item][1])
        text2 = str(self.excerpts[item][0])
        target = [
            self.target_dict[text2],
            self.target_dict[text1],
        ]

        inputs1 = self.tokenizer(text1, max_length=self.max_len, padding="max_length", truncation=True)
        inputs2 = self.tokenizer(text2, max_length=self.max_len, padding="max_length", truncation=True)

        ids1 = inputs1["input_ids"]
        mask1 = inputs1["attention_mask"]

        ids2 = inputs2["input_ids"]
        mask2 = inputs2["attention_mask"]

        return {
            "ids1": torch.tensor(ids1, dtype=torch.long),
            "mask1": torch.tensor(mask1, dtype=torch.long),
            "ids2": torch.tensor(ids2, dtype=torch.long),
            "mask2": torch.tensor(mask2, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.float),
        }
