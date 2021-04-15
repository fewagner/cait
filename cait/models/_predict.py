import h5py
from tqdm.auto import tqdm
import pickle
import cait as ai


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
               ):
    """
    TODO

    :param h5_path:
    :type h5_path:
    :param feature_channel:
    :type feature_channel:
    :param model:
    :type model:
    :param ptl_module:
    :type ptl_module:
    :param ptl_ckp_path:
    :type ptl_ckp_path:
    :param group_name:
    :type group_name:
    :param prediction_name:
    :type prediction_name:
    :param keys:
    :type keys:
    :param chunk_size:
    :type chunk_size:
    :param no_channel_idx_in_pred:
    :type no_channel_idx_in_pred:
    """

    if model is None and (ptl_module is None or ptl_ckp_path is None):
        raise KeyError('You need provide either model or ptl_module path and ptl_model_path!')

    if model is None:
        model = ptl_module.load_from_checkpoint(ptl_ckp_path)

    with h5py.File(h5_path, 'r+') as f_handle:

        nmbr_channels = len(f_handle[group_name][keys[0]])
        nmbr_events = len(f_handle[group_name][keys[0]][0])

        # add predictions to the h5 file
        data = f_handle.require_group(group_name)
        if no_channel_idx_in_pred:
            data.require_dataset(name=prediction_name,
                                 shape=(nmbr_events),
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
                prediction = model.predict(x).numpy()
                if no_channel_idx_in_pred:
                    data[prediction_name][count:count + chunk_size] = prediction.reshape(-1)
                else:
                    data[prediction_name][feature_channel, count:count + chunk_size] = prediction.reshape(-1)

                count += chunk_size
                pbar.update(chunk_size)
            # make input data
            x = {}
            for k in keys:
                x[k + '_ch' + str(feature_channel)] = f_handle[group_name][k][feature_channel,
                                                      count:count + chunk_size]  # array of shape: (nmbr_events, nmbr_features)

            # make predictions
            prediction = model.predict(x).numpy()

            if no_channel_idx_in_pred:
                data[prediction_name][count:] = prediction.reshape(-1)
            else:
                data[prediction_name][feature_channel, count:] = prediction.reshape(-1)
            pbar.update(chunk_size)

        print('{} written to file {}.'.format(prediction_name, h5_path))


def mh_predict(h5_path: str,
               feature_channel: int,
               type:str = 'events',
               model_type: str = 'model',
               model_handler: object = None,
               mh_path: str = None,
               which_data: str = 'mainpar',
               ):
    """
    TODO

    :param h5_path:
    :type h5_path:
    :param feature_channel:
    :type feature_channel:
    :param type:
    :type type:
    :param model_type:
    :type model_type:
    :param model_handler:
    :type model_handler:
    :param mh_path:
    :type mh_path:
    :param which_data:
    :type which_data:
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

        nmbr_channels = len(f_handle[type]['mainpar'])
        nmbr_events = len(f_handle[type]['mainpar'][0])

        data = f_handle[type]

        data.require_dataset(
            "{}_predictions".format(model_type),
            shape=(nmbr_channels, nmbr_events),
            dtype=float)
        data["{}_predictions".format(model_type)][feature_channel, ...] = predictions

        data["{}_predictions".format(model_type)].attrs.create(
            name='unlabeled', data=0)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Event_Pulse', data=1)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Test/Control_Pulse', data=2)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Noise', data=3)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Squid_Jump', data=4)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Spike', data=5)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Early_or_late_Trigger', data=6)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Pile_Up', data=7)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Carrier_Event', data=8)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Strongly_Saturated_Event_Pulse', data=9)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Strongly_Saturated_Test/Control_Pulse', data=10)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Decaying_Baseline', data=11)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Temperature_Rise', data=12)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Stick_Event', data=13)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Sawtooth_Cycle', data=14)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Human_Disturbance', data=15)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Large_Sawtooth', data=16)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Cosinus_Tail', data=17)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Light_only_Event', data=18)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Ring_Light_Event', data=19)
        data["{}_predictions".format(model_type)].attrs.create(
            name='Sharp_Light_Event', data=20)
        data["{}_predictions".format(model_type)].attrs.create(
            name='unknown/other', data=99)

        print('Added {} Predictions.'.format(model_type))
