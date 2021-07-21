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
    """
    Lightning module for the training of an LSTM model for classification or regression.
    For classification, the classes need to get one hot encoded, best with the corresponding transform.

    :param input_size: The number of features that get passed to the LSTM in one time step.
    :type input_size: int
    :param hidden_size: The number of nodes in the hidden layer of the lstm.
    :type hidden_size: int
    :param num_layers: The number of LSTM layers.
    :type num_layers: int
    :param seq_steps: The number of time steps.
    :type seq_steps: int
    :param device_name: The device on that the NN is trained.
    :type device_name: string, either 'cpu' or 'cude'
    :param nmbr_out: The number of output nodes the last linear layer after the lstm has.
    :type nmbr_out: int
    :param label_keys: The keys of the dataset that are used as labels.
    :type label_keys: list of strings
    :param feature_keys: The keys of the dataset that are used as nn inputs.
    :type feature_keys: list of strings
    :param lr: The learning rate for the neural network training.
    :type lr: float between 0 and 1
    :param is_classifier: If true, the output of the nn gets an additional softmax activation.
    :type is_classifier: bool
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
    :param bidirectional: If true, a bidirectional LSTM is used.
    :type bidirectional: bool
    :param norm_type: Either 'z' (mu=0, sigma=1) or 'minmax' (min=0, max=1). The type of normalization.
    :type norm_type: string
    :param lr_scheduler: If true, a learning rate scheduler is used.
    :type lr_scheduler: bool
    :param indiv_norm: If true, every event is divide by its maximal value before passing into the network.
    :type indiv_norm: bool
    :param attention: If activated, an attention layer is added before passing into the model.
    :type attention: bool
    """
    def __init__(self, input_size, hidden_size, num_layers, seq_steps, nmbr_out, label_keys,
                 feature_keys, lr, device_name='cpu', is_classifier=True, down=1, down_keys=None,
                 norm_vals=None, offset_keys=None, weight_decay=1e-5, bidirectional=False,
                 norm_type='minmax', lr_scheduler=True, indiv_norm=False, attention=False):

        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_steps = seq_steps
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        inp = (1 + int(bidirectional)) * self.hidden_size * self.seq_steps + int(indiv_norm)
        #print('Dim Input: ', inp)
        self.fc1 = nn.Linear(inp, nmbr_out)
        self.nmbr_out = nmbr_out
        self.device_name = device_name
        self.label_keys = label_keys
        self.feature_keys = feature_keys
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_classifier = is_classifier
        self.down = down  # just store as info for later
        self.down_keys = down_keys
        self.offset_keys = offset_keys
        self.norm_vals = norm_vals  # just store as info for later
        self.bidirectional = bidirectional
        self.norm_type = norm_type
        self.lr_scheduler = lr_scheduler
        self.indiv_norm = indiv_norm
        if attention:
            self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=1)
        else:
            self.attention = None

    def forward(self, x):
        """
        The forward pass in the neural network.

        :param x: The input features.
        :type x: torch tensor of size (batchsize, nmbr_features)
        :return: The ouput of the neural network.
        :rtype: torch tensor of size (batchsize, nmbr_outputs)
        """
        batchsize = x.size(0)

        if self.indiv_norm:
            max_vals = torch.max(x, dim=1).values.view(batchsize, 1)
            x = x/(max_vals + 1e-6)

        x = x.view(batchsize, self.seq_steps, self.input_size)

        # attention
        if self.attention is not None:
            att = x.permute(1, 0, 2)
            att, _ = self.attention(att, att, att)
            x = att.permute(1, 0, 2)
            #x = (x + att)/2

        # Set initial hidden and cell states
        h0 = torch.zeros((1 + int(self.bidirectional))*self.num_layers, batchsize, self.hidden_size).to(self.device_name)
        c0 = torch.zeros((1 + int(self.bidirectional))*self.num_layers, batchsize, self.hidden_size).to(self.device_name)

        # Forward propagate LSTM
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(batchsize, (1 + int(self.bidirectional)) * self.seq_steps * self.hidden_size)

        if self.indiv_norm:
            out = torch.cat((out, max_vals), dim=1)

        #print('Dim Out: ', out.shape)

        out = self.fc1(out)

        if self.is_classifier:
            out = F.log_softmax(out, dim=-1)

        #print('Dim Out: ', out.shape)

        return out

    def loss_function(self, logits, labels):
        if self.is_classifier:
            return F.nll_loss(logits, labels.long())
        else:
            return F.mse_loss(logits, labels, reduction='mean')

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
            lambda1 = lambda epoch: 0.95**epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def predict(self, sample):
        """
        Give a prediction for incoming data array or batch of arrays, does all essential transforms.

        :param sample: The features for one (1D case) or more (2D case) samples.
        :type sample: 1D numpy array or batch of arrays, i.e. then 2D array
        :return: The prediction.
        :rtype: torch tensor of size (batchsize - 1 if no batch, nn_output_size)
        """

        # if no batch make batch size 1
        for k in sample.keys():
            if len(sample[k].shape) < 2:
                sample[k] = sample[k].reshape(1, -1)

        # remove offset
        if self.offset_keys is not None:
            for key in self.offset_keys:
                sample[key] = (sample[key].transpose() - np.mean(sample[key][:, :int(len(sample[key]) / 8)],
                                                               axis=1)).transpose()

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
        if self.is_classifier:
            out = torch.argmax(out, dim=1)  # give back the label with highest value

        return out
