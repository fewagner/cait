import numpy as np
import torch.nn.functional as F
import torch
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn


class CNNModule(LightningModule):
    """
    Lightning module for the training of a CNN model for classification.

    :param input_size: The number of features that get passed to the CNN as one sample.
    :type input_size: int
    :param nmbr_out: The number of output nodes the last linear layer has.
    :type nmbr_out: int
    :param device_name: The device on that the NN is trained.
    :type device_name: string, either 'cpu' or 'cude'
    :param label_keys: The keys of the dataset that are used as labels.
    :type label_keys: list of strings
    :param feature_keys: The keys of the dataset that are used as nn inputs.
    :type feature_keys: list of strings
    :param lr: The learning rate for the neural network training.
    :type lr: float between 0 and 1
    :param down: The downsample factor of the training data set, if one is applied.
    :type down: int
    :param down_keys: The keys of the data that is to downsample (usually the event time series).
    :type down_keys: list of string
    :param norm_vals: The keys of this dictionary get scaled in the sample with (x - mu)/sigma.
    :type norm_vals: dictionary, every enty is a list of 2 ints (mean, std)
    :param offset_keys: The keys in the sample from that we want to subtract the baseline offset level.
    :type offset_keys: list of strings
    :param weight_decay: The weight decay parameter for the optimizer.
    :type weight_decay: float
    :param norm_type: Either 'z' (mu=0, sigma=1) or 'minmax' (min=0, max=1). The type of normalization.
    :type norm_type: string
    :param lr_scheduler: If true, a learning rate scheduler is used.
    :type lr_scheduler: bool
    :param kernelsize: The size of the kernels used for the CNN.
    :type kernelsize: int
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

    def get_prob(self, sample):
        """
        Get the outputs for all classes, before the decision rule is applied.
        """

        # to tensor
        for key in sample.keys():
            try:
                sample[key] = torch.from_numpy(sample[key]).float()
            except:
                pass

        # if no batch make batch size 1
        for k in sample.keys():
            if len(sample[k].shape) < 2:
                sample[k] = sample[k].reshape(1, -1)

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
                    min, max = torch.min(sample[key], dim=1, keepdim=True), torch.max(sample[key], dim=1, keepdim=True)
                    sample[key] = (sample[key] - min.values) / (max.values - min.values)
            else:
                raise NotImplementedError('This normalization type is not implemented.')

        # downsample
        if self.down_keys is not None:
            for key in self.down_keys:
                sample[key] = torch.mean(sample[key].
                                      reshape(len(sample[key]), int(len(sample[key][1]) / self.down), self.down),
                                      dim=2)

        # put features together
        x = torch.cat(tuple([sample[k] for k in self.feature_keys]), dim=1)
        x = x.to(self.device_name)
        out = self(x).detach()

        return out


    def predict(self, sample):
        """
        Predict the class for a sample.
        """

        out = self.get_prob(sample)
        out = torch.argmax(out, dim=1)  # give back the label with highest value

        return out