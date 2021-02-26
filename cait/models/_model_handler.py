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

    def __init__(self, model_type, run: str = '01', module: str = 'Test', record_length: int = 16384,
                 sample_frequency: int = 25000, nmbr_channels=2, down: int = 1,
                 model1=None, model2=None, model3=None,
                 classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
        """
        TODO

        :param run:
        :type run:
        :param module:
        :type module:
        :param model_type:
        :type model_type:
        :param record_length:
        :type record_length:
        :param nmbr_channels:
        :type nmbr_channels:
        :param sample_frequency:
        :type sample_frequency:
        :param down:
        :type down:
        :param model1:
        :type model1:
        :param model2:
        :type model2:
        :param model3:
        :type model3:
        :param classes:
        :type classes:
        """
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
        """
        TODO

        :param channel:
        :type channel:
        :param model:
        :type model:
        :param down:
        :type down:
        :return:
        :rtype:
        """

        # check if downsample rate matches
        if down != self.down:
            raise ValueError('Downsample rate must match.')

        # add the model
        self.models[channel] = model

    def get_model(self, channel):
        """
        TODO

        :param channel:
        :type channel:
        :return:
        :rtype:
        """
        return self.models[channel]

    def add_scaler(self, scaler, channel):
        """
        TODO

        :param scaler:
        :type scaler:
        :param channel:
        :type channel:
        :return:
        :rtype:
        """
        self.scalers[channel] = scaler
        print('Added scaler for channel {}.'.format(channel))

    def save(self, path, name_app=None):
        """
        TODO

        :param path:
        :type path:
        :param name_app:
        :type name_app:
        :return:
        :rtype:
        """

        self.info = input('Write info about this instance: ')
        path_model = '{}/{}_run{}_{}'.format(path, self.model_type, self.run, self.module)
        if name_app is not None:
            path_model += '{}'.format(name_app)
        pickle.dump(self, open(path_model, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('Save Model to {}.'.format(path_model))
