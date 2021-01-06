# -----------------------------------------------------
# IMPORTS
# -------------------------------------------------

import pickle

# -----------------------------------------------------
# ML MODULE
# -------------------------------------------------

class ModelHandler:
    """
    Wrapper class to store ML models and scalers
    """

    def __init__(self, run, module, model_type, record_length, nmbr_channels,
                 sample_frequency=25000, down=1,
                 model1=None, model2=None, model3=None,
                 classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
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

        print('MLModel Instance created.')


    def add_model(self, channel, model, down=1):

        # check if downsample rate matches
        if down != self.down:
            raise ValueError('Downsample rate must match.')

        # add the model
        self.models[channel] = model


    def get_model(self, channel):
        return self.models[channel]


    def add_scaler(self, scaler, channel):
        self.scalers[channel] = scaler
        print('Added scaler for channel {}.'.format(channel))


    def save(self, path, name_app=''):
        self.info = input('Write info about this instance: ')
        path_model = '{}/{}_run{}_{}_{}'.format(path, self.model_type, self.run, self.module, name_app)
        pickle.dump(self, open(path_model, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('Save Model to {}.'.format(path_model))
