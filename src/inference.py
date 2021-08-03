import pandas as pd
import tez
import transformers
import numpy as np
import torch
from sklearn import metrics
import transformers
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler

from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class AttentionHead(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        output = self.dropout(context_vector)
        return output


class CommonlitModel(tez.Model):
    def __init__(self, model_name, num_train_steps, steps_per_epoch, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"
        hidden_dropout_prob: float = 0.0
        layer_norm_eps: float = 1e-7
        config = transformers.AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
            }
        )
        self.transformer = transformers.AutoModel.from_pretrained(model_name, config=config)
        self.attention = AttentionHead(in_size=config.hidden_size, hidden_size=config.hidden_size)
        self.regressor = nn.Linear(config.hidden_size * 2, 2)

    def forward(self, ids1, mask1, ids2, mask2, targets=None):
        output1 = self.transformer(ids1, mask1)
        output2 = self.transformer(ids2, mask2)
        output1 = self.attention(output1.last_hidden_state)
        output2 = self.attention(output2.last_hidden_state)
        output = torch.cat((output1, output2), dim=1)
        output = self.regressor(output)
        return output, 0, {}


class CommonlitDataset:
    def __init__(self, excerpts, target_dict, tokenizer, max_len, num_samples=None):
        self.excerpts = excerpts
        self.target_dict = target_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_samples = num_samples

    def __len__(self):
        return len(self.excerpts)

    def __getitem__(self, item):
        text1 = str(self.excerpts[item][1])
        text2 = str(self.excerpts[item][0])
        target = [self.target_dict[text2], self.target_dict[text1]]
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


if __name__ == "__main__":
    max_len = 256
    model_name = "microsoft/deberta-large"
    df = pd.read_csv("../input/train_folds.csv")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    base_string = df.loc[df.target == 0, "excerpt"].values[0]
    target_dict = dict(zip(df.excerpt.values.tolist(), df.target.values.tolist()))

    scores = []
    final_preds = []
    final_targets = []
    for _fold in range(5):
        df_valid = df[df.kfold == _fold].reset_index(drop=True)
        validation_pairs = [(base_string, k) for k in df_valid.excerpt.values.tolist()]

        valid_dataset = CommonlitDataset(
            excerpts=validation_pairs,
            target_dict=target_dict,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        model = CommonlitModel(
            model_name=model_name,
            num_train_steps=1,
            learning_rate=1,
            steps_per_epoch=1,
        )

        model.load(
            f"../pairwise_model/{model_name.replace('/',':')}__fold_{_fold}.bin",
            weights_only=True,
        )

        temp_final_preds = []
        for p in tqdm(model.predict(valid_dataset, batch_size=64, n_jobs=-1)):
            temp_preds = p[:, 1].tolist()
            temp_final_preds.extend(temp_preds)

        score = np.sqrt(metrics.mean_squared_error(df_valid.target.values, temp_final_preds))
        scores.append(score)
        final_targets.extend(df_valid.target.values.tolist())
        final_preds.extend(temp_final_preds)

    print(scores)
    print(np.mean(scores))
    print(np.sqrt(metrics.mean_squared_error(final_targets, final_preds)))
