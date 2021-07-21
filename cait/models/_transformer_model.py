# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core.lightning import LightningModule
import numpy as np
import math


# ------------------------------------------------------
# MODEL
# ------------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModule(LightningModule):
    """
    Lightning module for the training of an Transformer Encoder model for classification or regression.
    For classification, the classes need to get one hot encoded, best with the corresponding transform.

    :param input_size: The number of features that get passed to the Model in one time step.
    :type input_size: int
    :param d_model: The dimensions of the model.
    :type d_model: int
    :param number_heads: The number of heads for the attention layer.
    :type number_heads: int
    :param dim_feedforward: The dimensions in the feed forward net.
    :type dim_feedforward: int
    :param hidden_size: The number of nodes in the hidden layer of the lstm.
    :type hidden_size: int
    :param num_layers: The number of LSTM layers.
    :type num_layers: int
    :param seq_steps: The number of time steps.
    :type seq_steps: int
    :param device_name: The device on that the NN is trained. Either 'cpu' or 'cuda'.
    :type device_name: string
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
    :param dropout: The share of weights that is set to zero in the dropout layer.
    :type dropout: float
    :param norm_type: Either 'z' (mu=0, sigma=1) or 'minmax' (min=0, max=1). The type of normalization.
    :type norm_type: string
    :param pos_enc: If true, we include a positional encoding layer.
    :type pos_enc: bool
    :param lr_scheduler: If true, a learning rate scheduler is used.
    :type lr_scheduler: bool
    """

    def __init__(self, input_size, d_model, number_heads, dim_feedforward, num_layers, nmbr_out,
                 seq_steps, device_name, label_keys, feature_keys, lr, is_classifier,
                 down, down_keys, offset_keys, norm_vals, weight_decay=1e-5, dropout=0.5,
                 norm_type='minmax', pos_enc=True, lr_scheduler=True):

        super().__init__()
        self.save_hyperparameters()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'
        if pos_enc:
            self.pos_encoding = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoding = nn.Identity()
        encoder_layers = TransformerEncoderLayer(d_model, number_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.input_embedding = nn.Linear(input_size, d_model)
        self.input_size = input_size
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.seq_steps = seq_steps
        self.d_model = d_model
        self.number_heads = number_heads
        self.nmbr_out = nmbr_out
        self.device_name = device_name
        self.decoder = nn.Linear(seq_steps*d_model, nmbr_out)

        self.label_keys = label_keys
        self.feature_keys = feature_keys
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_classifier = is_classifier
        self.down = down  # just store as info for later
        self.down_keys = down_keys
        self.offset_keys = offset_keys
        self.norm_vals = norm_vals  # just store as info for later
        self.norm_type = norm_type
        self.lr_scheduler = lr_scheduler

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        """
        The forward pass in the neural network

        :param x: the input features
        :type x: torch tensor of size (batchsize, nmbr_features)
        :return: the ouput of the neural network
        :rtype: torch tensor of size (batchsize, nmbr_outputs)
        """
        batchsize = src.size(0)

        src = src.view(batchsize, self.seq_steps, self.input_size)
        src = src.permute(1, 0, 2)  # now (seq_len, batch, features)

        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        out = self.transformer_encoder(src, src_mask)

        out = out.permute(1, 0, 2)  # now (batch, seq_len, features)
        out = out.reshape(batchsize, self.seq_steps * self.d_model)

        out = self.decoder(out)

        if self.is_classifier:
            out = F.log_softmax(out, dim=-1)

        #out = out.permute(1, 0)  # now (batch, outs)

        return out

    def loss_function(self, logits, labels):
        """
        Calculates the loss value, for classfiers the negative log likelihood, for regressors the MSE.

        :param logits: The output values of the neural network.
        :type logits: float
        :param labels: The labels, e.g. the objective values or classes.
        :type labels: float
        :return: The loss value
        :rtype: float
        """
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
            w = 2
            def lambdarate(epoch):
                if epoch == 0:
                    rate = 1/w
                else:
                    rate = min(1/math.sqrt(epoch), epoch / math.sqrt(w ** 3))
                return rate

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdarate)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def predict(self, sample):
        """
        Give a prediction for incoming data array or batch of arrays, does all essential transforms

        :param sample: the features for one (1D case) or more (2D case) samples
        :type sample: 1D numpy array or batch of arrays, i.e. then 2D array
        :return: the prediction
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
