# -----------------------------------------------------
# IMPORTS
# -------------------------------------------------

import pickle

import h5py
import numpy as np
import tsfel
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

from .features._ts_feat import calc_ts_features
from .datasets._rf_dataset import get_rf_dataset

# -----------------------------------------------------
# ML MODULE
# -------------------------------------------------

class ModelHandler:

    def __init__(self, run, module, model_type, record_length, nmbr_channels,
                 sample_frequency=25000, down=1,
                 model1=None, model2=None, model3=None,
                 classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        self.run = run
        self.module = module
        self.model_type = model_type
        self.down = down
        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.models = [model1, model2, model3]
        self.nmbr_channels = nmbr_channels
        self.scalers = [None, None, None]
        self.classes = classes
        self.info = ''

        if self.model_type != 'rf':
            raise NotImplementedError('Other models than Randfom Forest are not yet implemented.')

        print('MLModel Instance created.')

    def add_classifier(self, channel, model=None, down=1):

        # check if downsample rate matches
        if down != self.down:
            raise ValueError('Downsample rate must match.')

        # add the model
        if model != None:
            self.models[channel] = model
        elif self.model_type == 'rf':
            self.models[channel] = RandomForestClassifier()

    def add_data(self, data_path, bcks,
                 module_channels, bck_naming='bck', ):

        if len(module_channels) != self.nmbr_channels:
            raise ValueError('Channel list must have same legth as nmbr_channels!')

        self.module_channels = module_channels
        self.data_path = data_path
        self.bcks = bcks
        self.bck_naming = bck_naming

        if self.nmbr_channels == 2:

            self.paths_h5 = [
                '{}/run{}_{}/{}_{}-P_Ch{}-L_Ch{}.h5'.format(data_path, self.run, self.module,
                                                            bck_naming, nmbr, module_channels[0],
                                                            module_channels[1])
                for
                nmbr in bcks]

        else:
            raise NotImplementedError('Only for 2 channels implemented.')

        print('Added paths of hdf5 bck files.')


    def train_rf(self,
                 channel,
                 test_size=0.3,
                 random_seed=None,
                 include_ts_features=True):

        print('Start training model for channel {}.'.format(channel))
        if self.model_type != 'rf':
            raise AttributeError('This instance has another classifier!')

        X_train, X_test, y_train, y_test, scaler = get_rf_dataset(paths_h5=self.paths_h5,
                                                          channel=channel,
                                                          include_ts_features=include_ts_features,
                                                          random_seed=random_seed,
                                                          test_size=test_size)

        self.models[channel].fit(X_train, y_train)
        self.scalers[channel] = scaler
        self.random_seed = random_seed
        self.test_size = test_size

        print('Training Score: ', self.models[channel].score(X_train, y_train))
        print('Test Score: ', self.models[channel].score(X_test, y_test))

    def scores_rf(self, channel, include_ts_features=True):

        if self.model_type != 'rf':
            raise AttributeError('This instance has another classifier!')

        X_train, X_test, y_train, y_test, scaler = get_rf_dataset(paths_h5=self.paths_h5,
                                                                  channel=channel,
                                                                  include_ts_features=include_ts_features,
                                                                  random_seed=self.random_seed,
                                                                  test_size=self.test_size,
                                                                  scalers=self.scalers)

        print('Training Score: ', self.models[channel].score(X_train, y_train))
        print('Test Score: ', self.models[channel].score(X_test, y_test))

    def add_scaler(self, scaler, channel):
        self.scalers[channel] = scaler
        print('Added scaler for channel {}.'.format(channel))

    def save(self, path):
        self.info = input('Write info about this instance: ')
        path_model = '{}/{}_run{}_{}'.format(path, self.model_type, self.run, self.module)
        pickle.dump(self, open(path_model, 'wb'))
        print('Save Model to {}.'.format(path_model))

    def _pred_rf(self, x):
        raise NotImplementedError('Prediction not yet implemented.')

    def _pred_lstm(self, x):
        raise NotImplementedError('Prediction not yet implemented.')

    def predict(self, x):

        # somehow handle input error if channel nmbr no match
        if len(x.shape) != 3:
            raise ValueError('Input Dimension must be [nmbr_channels, nmbr_events, record_length/down].')
        elif x.shape[0] != self.nmbr_channels:
            raise ValueError('The number of channels in the input does not match the number of channels in the models.')

        if self.model_type == 'rf':
            return self._pred_rf(x)
        elif self.model_type == 'lstm':
            return self._pred_lstm(x)

    def __call__(self, x):
        return self.predict(x)
