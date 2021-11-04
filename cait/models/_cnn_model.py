import numpy as np
import cait as ai
import os
from pytorch_lightning import Trainer
from torchvision import transforms
import h5py
from cait.datasets import RemoveOffset, Normalize, DownSample, ToTensor, CryoDataModule
from cait.models import LSTMModule, nn_predict
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F
import torch
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn


class CNNModule(LightningModule):
    """
    TODO
    """
    def __init__(self, input_size, nmbr_out,
                 label_keys, feature_keys, lr, device_name='cpu', down=1, down_keys=None,
                 norm_vals=None, offset_keys=None, weight_decay=1e-5,
                 norm_type='minmax', lr_scheduler=True, kernelsize=8):
        super().__init__()

        assert np.isclose(input_size % 2, 0), 'Input size must be power of 2!'
        self.conv1 = nn.Conv1d(1, 50, kernelsize, 4)
        self.conv2 = nn.Conv1d(50, 10, kernelsize, 4)
        self.intermed_size = int(np.floor(127 * input_size / 8192) * 10)
        self.fc1 = nn.Linear(self.intermed_size, 200)
        self.fc2 = nn.Linear(200, nmbr_out)

        # save pars
        self.save_hyperparameters()

        self.input_size = input_size
        self.nmbr_out = nmbr_out

        self.device_name = device_name
        self.label_keys = label_keys
        self.feature_keys = feature_keys
        self.lr = lr
        self.weight_decay = weight_decay
        self.down = down  # just store as info for later
        self.down_keys = down_keys
        self.offset_keys = offset_keys
        self.norm_vals = norm_vals  # just store as info for later
        self.norm_type = norm_type
        self.lr_scheduler = lr_scheduler

    def forward(self, x):
        """
        TODO
        """
        bs = x.size()[0]
        x = x.view(bs, 1, -1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(bs, self.intermed_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def loss_function(self, logits, labels):
        return F.nll_loss(logits, labels.long())

    def training_step(self, batch, batch_idx):

        x = torch.cat(tuple([batch[k] for k in self.feature_keys]), dim=1)
        if len(self.label_keys) == 1:
            y = batch[self.label_keys[0]]
        else:
            y = torch.cat(tuple([batch[k] for k in self.label_keys]), dim=1)

        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):

        x = torch.cat(tuple([val_batch[k] for k in self.feature_keys]), dim=1)
        if len(self.label_keys) == 1:
            y = val_batch[self.label_keys[0]]
        else:
            y = torch.cat(tuple([val_batch[k] for k in self.label_keys]), dim=1)

        logits = self.forward(x)
        loss = self.loss_function(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):

        x = torch.cat(tuple([batch[k] for k in self.feature_keys]), dim=1)
        if len(self.label_keys) == 1:
            y = batch[self.label_keys[0]]
        else:
            y = torch.cat(tuple([batch[k] for k in self.label_keys]), dim=1)

        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self, lr=None, weight_decay=None):
        if lr is None:
            lr = self.lr
        if weight_decay is None:
            weight_decay = self.weight_decay
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if self.lr_scheduler:
            lambda1 = lambda epoch: 0.95 ** epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def predict(self, sample):
        """
        TODO
        """

        # if no batch make batch size 1
        for k in sample.keys():
            if len(sample[k].shape) < 2:
                sample[k] = sample[k].reshape(1, -1)

        # remove offset
        # if self.offset_keys is not None:
        #     for key in self.offset_keys:
        #         sample[key] = (sample[key].transpose() - np.mean(sample[key][:, :int(len(sample[key]) / 8)],
        #                                                        axis=1)).transpose()

        # normalize
        if self.norm_vals is not None:
            if self.norm_type == 'z':
                for key in self.norm_vals.keys():
                    mean, std = self.norm_vals[key]
                    sample[key] = (sample[key] - mean) / std
            elif self.norm_type == 'minmax':
                for key in self.norm_vals.keys():
                    min, max = self.norm_vals[key]
                    sample[key] = (sample[key] - min) / (max - min)
            elif self.norm_type == 'indiv_minmax':
                for key in self.norm_vals.keys():
                    min, max = np.min(sample[key], axis=1, keepdims=True), np.max(sample[key], axis=1, keepdims=True)
                    sample[key] = (sample[key] - min) / (max - min)
            else:
                raise NotImplementedError('This normalization type is not implemented.')

        # downsample
        if self.down_keys is not None:
            for key in self.down_keys:
                sample[key] = np.mean(sample[key].
                                      reshape(len(sample[key]), int(len(sample[key][1]) / self.down), self.down),
                                      axis=2)

        # to tensor
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key]).float()

        # put features together
        x = torch.cat(tuple([sample[k] for k in self.feature_keys]), dim=1)
        x = x.to(self.device_name)
        out = self(x).detach()

        # put the decision rule
        out = torch.argmax(out, dim=1)  # give back the label with highest value

        return out