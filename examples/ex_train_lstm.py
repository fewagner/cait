"""
Train an LSTM to classify a HDF5 dataset with included labels
"""

# imports
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai
from cait.datasets import CryoDataModule
from cait.datasets import Normalize, RemoveOffset, ToTensor, DownSample
from cait.models import LSTMModule
from pytorch_lightning import Trainer
from torchvision import transforms
import h5py

# some parameters
# nmbr_gpus = ... uncommment and put in trainer to use GPUs
path_h5 = 'toy_data/Run35_DetF/bck_001-P_Ch26-L_Ch27.h5'
type = 'events'
keys = ['event', 'labels']
channel_indices = [[0], [0]]
feature_keys = ['event']
label_keys = ['labels']
norm_vals = {'event': [0, 1]}
down_keys = ['event']
down = 1
input_size = 8

# create the transforms
transforms = transforms.Compose([RemoveOffset(keys=feature_keys),
                                 Normalize(norm_vals=norm_vals),
                                 DownSample(keys=down_keys, down=down),
                                 ToTensor()])

# create data module
dm = CryoDataModule(hdf5_path=path_h5,
                    type=type,
                    keys=keys,
                    channel_indices=channel_indices,
                    transform=transforms)

dm.prepare_data(val_size=0.2,
                test_size=0.2,
                batch_size=16,
                dataset_size=None,
                only_idx=None,
                shuffle_dataset=True,
                random_seed=42,
                feature_keys=feature_keys,
                label_keys=label_keys,
                one_hot=True)

dm.setup()

# create lstm clf
lstm = LSTMModule(input_size=input_size,
                  hidden_size=input_size * 10,
                  num_layers=2,
                  seq_steps=int(dm.dims()[1] / input_size),  # downsampling is already considered in dm
                  device='cpu',
                  nmbr_out=dm.label_dims()[1],  # this is the number of labels
                  lr=1e-3,
                  label_keys=label_keys,
                  feature_keys=feature_keys,
                  is_classifier=True,
                  down=down,
                  down_keys=feature_keys,
                  norm_vals=feature_keys,
                  offset_keys=feature_keys)

# create instance of Trainer
trainer = Trainer()  # keyword gpus=nmbr_gpus for GPU Usage

# all training happens here
trainer.fit(model=lstm, datamodule=dm)

# run test set
result = trainer.test()
print(result)

# model can be saved and loaded with instane of ModelHandler that uses pickle


# predictions with the model are made that way
f = h5py.File(dm.hdf5_path, 'r')
x = {keys[0]: f[type][keys[0]][channel_indices[0]]}  # array of shape: (nmbr_events, nmbr_features)
prediction = lstm.predict(x)

# predictions can be saved with instance of EvaluationTools
print(prediction)
