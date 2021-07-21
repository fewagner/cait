# -----------------------------------------------------
# IMPORTS
# -------------------------------------------------

import pickle


# -----------------------------------------------------
# ML MODULE
# -------------------------------------------------

class ModelHandler:
    """
    Wrapper class to store ML models and scalers, as they are typically used by the Scikit-Learn Library.

    :param model_type: The type of the model, e.g. RF or SVM.
    :type model_type: string
    :param nmbr_channels: The number of channels of the detector module.
    :type nmbr_channels: int
    :param record_length: The number of samples per record window.
    :type record_length: int
    :param sample_frequency: THe sample frequency of the measurement.
    :type sample_frequency: int
    :param run: The number of the run of the experiment.
    :type run: string
    :param module: The name of the detector module.
    :type module: string
    :param info: Additional information about this model, e.g. training set size.
    :type info: string
    """

    def __init__(self, model_type: str = None, nmbr_channels: int = None,
                 record_length: int = 16384,
                 sample_frequency: int = 25000,
                 run: str = None, module: str = None,
                 info:str = None):
        self.run = run
        self.module = module
        self.model_type = model_type
        self.record_length = record_length
        self.sample_frequency = sample_frequency
        self.models = {}
        self.nmbr_channels = nmbr_channels
        self.scalers = {}
        self.info = info

        print('MLModel Instance created.')

    # setters

    def add_model(self, model, channel):
        """
        Add a model to the model handler.

        :param channel: The channel number that the model corresponds to.
        :type channel: int
        :param model: The model that you want to store.
        :type model: object
        """

        # add the model
        self.models[channel] = model
        print('Added model for channel {}.'.format(channel))

    def add_scaler(self, scaler, channel):
        """
        Add a scaler to the model handler.

        :param channel: The channel number that the model corresponds to.
        :type channel: int
        :param model: The model that you want to store.
        :type model: object
        """
        self.scalers[channel] = scaler
        print('Added scaler for channel {}.'.format(channel))

    # getters

    def get_model(self, channel):
        """
        Return a model from the model handler.

        :param channel: The number of the channel from that we want the model.
        :type channel: int
        :return: The model.
        :rtype: object
        """
        return self.models[channel]

    def get_scaler(self, channel):
        """
        Return a scaler from the model handler.

        :param channel: The number of the channel from that we want the scaler.
        :type channel: int
        :return: The scaler.
        :rtype: object
        """
        return self.scalers[channel]

    def save(self, path, info=None):
        """
        Save the model handler with pickle.

        :param path: The full path to save the model handler.
        :type path: string
        :param info: Additional information about this model, e.g. training set size.
        :type info: string
        """

        self.info = info
        pickle.dump(self, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('Save Model to {}.'.format(path))
