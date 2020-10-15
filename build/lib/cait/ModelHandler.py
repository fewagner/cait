# -----------------------------------------------------
# IMPORTS
# -------------------------------------------------

import pickle

# -----------------------------------------------------
# ML MODULE
# -------------------------------------------------

class MLModel:

    def __init__(self, run, module, model_type, record_length, down, model1=None, model2=None, model3=None):
        self.run = run
        self.module = module
        self.model_type = model_type
        self.down = down
        self.models = [model1, model2, model3]
        self.nmbr_channels = sum(x is not None for x in self.models)
        self.scalers = [None, None, None]

        print('MLModel Instance created.')

    def add_model(self, model, channel, down):
        if down != self.down:
            raise ValueError('Downsample rate must match.')
        self.model[channel] = model
        if model != None:
            self.nmbr_channels += 1
        else:
            self.nmbr_channels -= 1

    def add_scaler(self, scaler, channel):
        self.scalers[channel] = scaler
        print('Added scaler for channel {}.'.format(channel))

    def save_model(self, path):
        path_model = '{}/{}_{}_{}'.format(path, self.model_type, self.run, self.module)
        pickle.dump(self, open(path_model, 'wb'))
        print('Save Model to {}.'.format(path_model))

    def _pred_rf(self, x):
        return 0

    def _pred_lstm(self, x):
        return 0

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
