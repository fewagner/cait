import h5py
from tqdm.notebook import tqdm


def nn_predict(h5_path,
               model,
               feature_channel,
               group_name='events',
               prediction_name='prediction',
               keys=['event'],
               chunk_size=50,
               no_channel_idx_in_pred=False,
               ):
    # TODO

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
                print('Events predicted: {}%'.format(100 * count / nmbr_events))

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
