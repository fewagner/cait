# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core.lightning import LightningModule
import numpy as np


# ------------------------------------------------------
# MODEL
# ------------------------------------------------------

class LSTMModule(LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, seq_steps, device, nmbr_out, label_keys,
                 feature_keys, lr, is_classifier=True, is_binary=False, one_hot=True, down=1, down_keys=None,
                 norm_vals=None, offset_keys=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_steps = seq_steps
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size * self.seq_steps, nmbr_out)
        self.device = device
        self.label_keys = label_keys
        self.feature_keys = feature_keys
        self.lr = lr
        self.is_classifier = is_classifier
        self.is_binary = is_binary
        self.one_hot = one_hot
        self.down = down  # just store as info for later
        self.down_keys = down_keys
        self.offset_keys = offset_keys
        self.norm_vals = norm_vals  # just store as info for later

    def forward(self, x):
        batchsize = x.size(0)

        x = x.view(batchsize, self.seq_steps, self.input_size)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(self.device)

        # Forward propagate LSTM
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(batchsize, self.seq_steps * self.hidden_size)

        out = self.fc1(out)

        if self.is_classifier:
            out = F.softmax(out, dim=-1)

        return out

    def loss_function(self, logits, labels):
        if self.is_classifier:
            return F.nll_loss(logits, labels)
        else:
            return F.mse_loss(logits, labels)

    def training_step(self, batch, batch_idx):

        x = torch.cat(tuple([batch[k] for k in self.feature_keys]), dim=1)
        y = torch.cat(tuple([batch[k] for k in self.label_keys]), dim=1)
        if self.one_hot:
            y = F.one_hot(y)

        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):

        x = torch.cat(tuple([val_batch[k] for k in self.feature_keys]), dim=1)
        y = torch.cat(tuple([val_batch[k] for k in self.label_keys]), dim=1)
        if self.one_hot:
            y = F.one_hot(y)

        logits = self.forward(x)
        loss = self.loss_function(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):

        x = torch.cat(tuple([batch[k] for k in self.feature_keys]), dim=1)
        y = torch.cat(tuple([batch[k] for k in self.label_keys]), dim=1)
        if self.one_hot:
            y = F.one_hot(y)

        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self, lr=None):
        if lr is None:
            lr = self.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def predict(self, sample, is_batch=False):

        # if no batch make batch size 1
        if not is_batch:
            for k in sample.keys():
                sample[k] = sample[k].reshape(1, -1)

        # remove offset
        if self.offset_keys is not None:
            for key in self.offset_keys:
                sample[key] = (sample[key].transpose - np.mean(sample[key][:, :int(len(sample[key]) / 8)],
                                                               axis=1)).transpose()

        # normalize
        if self.norm_vals is not None:
            for key in self.norm_vals.keys():
                mean, std = self.norm_vals[key]
                sample[key] = (sample[key] - mean) / std

        # downsample
        if self.down_keys is not None:
            for key in self.down_keys:
                sample[key] = np.mean(sample[key].
                                      reshape(-1, int(len(sample[key]) / self.down), self.down),
                                      axis=1)

        # to tensor
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])

        # put features together
        x = torch.cat(tuple([sample[k] for k in self.feature_keys]), dim=1)
        out = self(x)

        # put the decision rule
        if self.is_classifier:
            out = torch.argmax(x, dim=1)  # give back the label with highest value

        return out
