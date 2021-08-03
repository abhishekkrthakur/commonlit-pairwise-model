from sklearn import metrics
import transformers
from transformers import AdamW
import torch
import torch.nn as nn
import numpy as np
import tez
from torch.optim.lr_scheduler import LambdaLR


def custom_scheduler(optimizer, steps_per_epoch, lr_min, lr_max):
    def lr_lambda(current_step):
        current_step = current_step % steps_per_epoch
        if current_step < steps_per_epoch / 2:
            y = ((lr_max - lr_min) / (steps_per_epoch / 2)) * current_step + lr_min
        else:
            y = (-1.0 * (lr_max - lr_min) / (steps_per_epoch / 2)) * current_step + lr_max + (lr_max - lr_min)
        return y

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


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

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        sch = custom_scheduler(
            self.optimizer,
            self.steps_per_epoch,
            1e-6,
            1.5e-5,
        )
        return sch

    def loss(self, outputs, targets):
        return torch.sqrt(nn.MSELoss()(outputs, targets))

    def monitor_metrics(self, outputs, targets):
        outputs = outputs.cpu().detach().numpy()[:, 1].ravel()
        targets = targets.cpu().detach().numpy()[:, 1].ravel()
        mse = metrics.mean_squared_error(targets, outputs)
        rmse = np.sqrt(mse)
        return {"rmse": rmse, "mse": mse}

    def forward(self, ids1, mask1, ids2, mask2, targets=None):
        output1 = self.transformer(ids1, mask1)
        output2 = self.transformer(ids2, mask2)
        output1 = self.attention(output1.last_hidden_state)
        output2 = self.attention(output2.last_hidden_state)
        output = torch.cat((output1, output2), dim=1)
        output = self.regressor(output)
        # output = self.regressor(output.pooler_output)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc