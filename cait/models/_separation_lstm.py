from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class SeparationLSTM(LightningModule):
    """TODO"""

    def __init__(self, nmbr_pileup, label_keys, input_size, hidden_size, num_layers, seq_steps,
                 feature_keys, lr, device_name='cpu', down=1, down_keys=None,
                 norm_vals=None, offset_keys=None, weight_decay=1e-5,
                 norm_type='minmax', lr_scheduler=True):
        super().__init__()
        self.save_hyperparameters()

        self.n_channels = 1
        self.n_classes = nmbr_pileup
        self.nmbr_pileup = nmbr_pileup
        self.bilinear = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_steps = seq_steps

        self.lr = lr
        self.feature_keys = feature_keys
        self.label_keys = label_keys
        self.device_name = device_name
        self.down = down
        self.down_keys = down_keys
        self.norm_vals = norm_vals
        self.offset_keys = offset_keys
        self.weight_decay = weight_decay
        self.norm_type = norm_type
        self.lr_scheduler = lr_scheduler

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.nmbr_pileup * self.input_size)

    def forward(self, x):

        bs = x.size(0)
        x = x.view(bs, self.seq_steps, self.input_size)

        x, _ = self.lstm(x)

        x = x.contiguous().view(bs * self.seq_steps, self.hidden_size)
        x = self.fc(x)
        x = x.view(bs, self.seq_steps * self.input_size, self.nmbr_pileup)
        x = torch.transpose(x, 1, 2)

        return x

    def loss_function(self, y_hat, x):
        loss = F.mse_loss(y_hat, x, reduction='mean')
        return loss

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

        # if no batch make batch size 1
        for k in sample.keys():
            if len(sample[k].shape) < 2:
                sample[k] = sample[k].reshape(1, -1)

        # normalize
        if self.norm_vals is not None:
            if self.norm_type == 'z':
                for key in self.norm_vals.keys():
                    if key in self.feature_keys:
                        mean, std = self.norm_vals[key]
                        sample[key] = (sample[key] - mean) / std
            elif self.norm_type == 'minmax':
                for key in self.norm_vals.keys():
                    if key in self.feature_keys:
                        min, max = self.norm_vals[key]
                        sample[key] = (sample[key] - min) / (max - min)
            else:
                raise NotImplementedError('This normalization type is not implemented.')

        # downsample
        if self.down_keys is not None:
            for key in self.down_keys:
                s = sample[key].shape
                l = len(s)
                if l == 2:
                    sample[key] = np.mean(sample[key].
                                          reshape(s[0], -1, self.down),
                                          axis=2)
                elif l == 3:
                    sample[key] = np.mean(sample[key].
                                          reshape(s[0], s[1], -1, self.down),
                                          axis=3)
                else:
                    raise NotImplemented

        # to tensor
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key]).float()

        # put features together
        x = torch.cat(tuple([sample[k] for k in self.feature_keys]), dim=1)
        x = x.to(self.device_name)
        out = self(x).detach()

        # de-normalize
        if self.norm_vals is not None:
            if self.norm_type == 'z':
                for key in self.norm_vals.keys():
                    if key in self.label_keys:
                        mean, std = self.norm_vals[key]
                        out = out * std + mean
            elif self.norm_type == 'minmax':
                for key in self.norm_vals.keys():
                    if key in self.label_keys:
                        min, max = self.norm_vals[key]
                        out = out * (max - min) + min
            else:
                raise NotImplementedError('This normalization type is not implemented.')

        # upsample
        out = torch.repeat_interleave(out, self.down, dim=2)

        return out