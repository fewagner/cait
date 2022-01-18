import h5py
from tqdm.auto import tqdm
import pickle
import cait as ai
import numpy as np
import os
from ..resources import change_channel

def nn_predict(h5_path: str,
               feature_channel: int,
               model: object = None,
               ptl_module: object = None,
               ptl_ckp_path: str = None,
               group_name: str = 'events',
               prediction_name: str = 'prediction',
               keys: list = ['event'],
               chunk_size: int = 50,
               no_channel_idx_in_pred: bool = False,
               use_prob=False,
               ):
    """
    Add predictions from a PyTorch model to the HDF5 set.

    :param h5_path: The path to the HDF5 file.
    :type h5_path: string
    :param feature_channel: The channel of the detector module on that we make the predictions.
    :type feature_channel: int
    :param model: A trained PyTorch Lightning module or Pytorch model.
    :type model: object
    :param ptl_module: A Pytorch lightning model class.
    :type ptl_module: object
    :param ptl_ckp_path: A path to the checkpoint from where we load the PyTorch lightning model parameters.
    :type ptl_ckp_path: string
    :param group_name: The name of the group within the HDF5 file.
    :type group_name: string
    :param prediction_name: The name of the prediction that is saved to the HDF5 set.
    :type prediction_name: string
    :param keys: The keys from the HDF5 set that are included as features into every sample handed to the neural network.
    :type keys: list
    :param chunk_size: The size of the chunks to predict at once.
    :type chunk_size: int
    :param no_channel_idx_in_pred: If True, then we assume that there is no channel index in the data set from the HDF5
        file.
    :type no_channel_idx_in_pred: bool
    :param use_prob: Include the probabilities corresponding to all classes, instead of the prediction for one class.
    :type use_prob: bool
    """

    assert np.logical_xor(model is None, ptl_module is None or ptl_ckp_path is None), \
        'You need provide either model, or ptl_module path and ptl_model_path!'
    assert np.logical_xor(model is None, ptl_module is None), \
        'You cannot provide both a model and a checkpoint to laod a model from!'

    if model is None:
        model = ptl_module.load_from_checkpoint(ptl_ckp_path)

    model = change_channel(model, feature_channel)

    with h5py.File(h5_path, 'r+') as f_handle:

        nmbr_channels = f_handle[group_name][keys[0]].shape[0]
        nmbr_events = f_handle[group_name][keys[0]].shape[1]

        # add predictions to the h5 file
        data = f_handle.require_group(group_name)
        if no_channel_idx_in_pred:
            if use_prob:
                data.require_dataset(name=prediction_name,
                                     shape=(nmbr_events, model.nmbr_out),
                                     dtype=float)
            else:
                data.require_dataset(name=prediction_name,
                                     shape=(nmbr_events),
                                     dtype=float)
        else:
            if use_prob:
                data.require_dataset(name=prediction_name,
                                     shape=(nmbr_channels, nmbr_events, model.nmbr_out),
                                     dtype=float)
            else:
                data.require_dataset(name=prediction_name,
                                     shape=(nmbr_channels, nmbr_events),
                                     dtype=float)

        count = 0
        with tqdm(total=nmbr_events) as pbar:
            while count < nmbr_events - chunk_size:

                # make input data
                x = {}
                for k in keys:
                    x[k + '_ch' + str(feature_channel)] = f_handle[group_name][k][feature_channel,
                                                          count:count + chunk_size]  # array of shape: (nmbr_events, nmbr_features)

                # make predictions
                if use_prob:
                    prediction = model.get_prob(x).numpy().reshape(-1, model.nmbr_out)
                else:
                    prediction = model.predict(x).numpy().reshape(-1)

                if no_channel_idx_in_pred:
                    data[prediction_name][count:count + chunk_size] = prediction
                else:
                    data[prediction_name][feature_channel, count:count + chunk_size] = prediction

                count += chunk_size
                pbar.update(chunk_size)
            # make input data
            x = {}
            for k in keys:
                x[k + '_ch' + str(feature_channel)] = f_handle[group_name][k][feature_channel,
                                                      count:count + chunk_size]  # array of shape: (nmbr_events, nmbr_features)

            # make predictions
            if use_prob:
                prediction = model.get_prob(x).numpy().reshape(-1, model.nmbr_out)
            else:
                prediction = model.predict(x).numpy().reshape(-1)

            if no_channel_idx_in_pred:
                data[prediction_name][count:] = prediction
            else:
                data[prediction_name][feature_channel, count:] = prediction
            pbar.update(chunk_size)

        data[prediction_name].attrs.create(name='unlabeled', data=0)
        data[prediction_name].attrs.create(name='Event_Pulse', data=1)
        data[prediction_name].attrs.create(name='Test/Control_Pulse', data=2)
        data[prediction_name].attrs.create(name='Noise', data=3)
        data[prediction_name].attrs.create(name='Squid_Jump', data=4)
        data[prediction_name].attrs.create(name='Spike', data=5)
        data[prediction_name].attrs.create(name='Early_or_late_Trigger', data=6)
        data[prediction_name].attrs.create(name='Pile_Up', data=7)
        data[prediction_name].attrs.create(name='Carrier_Event', data=8)
        data[prediction_name].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
        data[prediction_name].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
        data[prediction_name].attrs.create(name='Decaying_Baseline', data=11)
        data[prediction_name].attrs.create(name='Temperature_Rise', data=12)
        data[prediction_name].attrs.create(name='Stick_Event', data=13)
        data[prediction_name].attrs.create(name='Sawtooth_Cycle', data=14)
        data[prediction_name].attrs.create(name='Human_Disturbance', data=15)
        data[prediction_name].attrs.create(name='Large_Sawtooth', data=16)
        data[prediction_name].attrs.create(name='Cosinus_Tail', data=17)
        data[prediction_name].attrs.create(name='Light_only_Event', data=18)
        data[prediction_name].attrs.create(name='Ring_Light_Event', data=19)
        data[prediction_name].attrs.create(name='Sharp_Light_Event', data=20)
        data[prediction_name].attrs.create(name='unknown/other', data=99)

        print('{} written to file {}.'.format(prediction_name, h5_path))


def mh_predict(h5_path: str,
               feature_channel: int,
               group_name: str = 'events',
               prediction_name: str = 'prediction',
               model_handler: object = None,
               mh_path: str = None,
               which_data: str = 'mainpar',
               ):
    """
    Add predictions from a Scikit-Learn model to the HDF5 set.

    :param h5_path: The path to the HDF5 file.
    :type h5_path: string
    :param feature_channel: The channel of the detector module on that we make the predictions.
    :type feature_channel: int
    :param group_name: The name of the group within the HDF5 file.
    :type group_name: string
    :param prediction_name: The name of the prediction that is saved to the HDF5 set.
    :type prediction_name: string
    :param model_handler: A model handler with that we want to make predictions.
    :type model_handler: object
    :param mh_path: A path to load a model handler from.
    :type mh_path: string
    :param which_data: Used for the evaluation tools instance, to tell which data is used for the prediction.
    :type which_data: string
    """

    if model_handler is None and mh_path is None:
        raise KeyError("You need provide either model_handler or mh_path!")

    if model_handler is None:
        model_handler = pickle.load(open(mh_path, 'rb+'))

    et_pred = ai.EvaluationTools()

    et_pred.add_events_from_file(file=h5_path,
                                 channel=feature_channel,
                                 which_data=which_data,
                                 )

    et_pred.set_scaler(model_handler.get_scaler(feature_channel))

    predictions = model_handler.get_model(feature_channel).predict(et_pred.features)

    with h5py.File(h5_path, 'r+') as f_handle:

        nmbr_channels = f_handle[group_name]['mainpar'].shape[0]
        nmbr_events = f_handle[group_name]['mainpar'].shape[1]

        data = f_handle[group_name]

        data.require_dataset(prediction_name,
            shape=(nmbr_channels, nmbr_events),
            dtype=float)

        data[prediction_name][:] = predictions

        data[prediction_name].attrs.create(name='unlabeled', data=0)
        data[prediction_name].attrs.create(name='Event_Pulse', data=1)
        data[prediction_name].attrs.create(name='Test/Control_Pulse', data=2)
        data[prediction_name].attrs.create(name='Noise', data=3)
        data[prediction_name].attrs.create(name='Squid_Jump', data=4)
        data[prediction_name].attrs.create(name='Spike', data=5)
        data[prediction_name].attrs.create(name='Early_or_late_Trigger', data=6)
        data[prediction_name].attrs.create(name='Pile_Up', data=7)
        data[prediction_name].attrs.create(name='Carrier_Event', data=8)
        data[prediction_name].attrs.create(name='Strongly_Saturated_Event_Pulse', data=9)
        data[prediction_name].attrs.create(name='Strongly_Saturated_Test/Control_Pulse', data=10)
        data[prediction_name].attrs.create(name='Decaying_Baseline', data=11)
        data[prediction_name].attrs.create(name='Temperature_Rise', data=12)
        data[prediction_name].attrs.create(name='Stick_Event', data=13)
        data[prediction_name].attrs.create(name='Sawtooth_Cycle', data=14)
        data[prediction_name].attrs.create(name='Human_Disturbance', data=15)
        data[prediction_name].attrs.create(name='Large_Sawtooth', data=16)
        data[prediction_name].attrs.create(name='Cosinus_Tail', data=17)
        data[prediction_name].attrs.create(name='Light_only_Event', data=18)
        data[prediction_name].attrs.create(name='Ring_Light_Event', data=19)
        data[prediction_name].attrs.create(name='Sharp_Light_Event', data=20)
        data[prediction_name].attrs.create(name='unknown/other', data=99)

    print('Added Predictions: {}.'.format(prediction_name))
