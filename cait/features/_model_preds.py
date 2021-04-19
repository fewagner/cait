

def model_predict(f_handle,
                 model,
                 tpa,
                 nmbr_channels,
                 nmbr_events,
                 group_name,
                 key_name,
                 chunk_size,
                 channel):
    """
    Include predictions from a model to a HDF5 file
    TODO

    :param f_handle:
    :type f_handle:
    :param model:
    :type model:
    :param tpa:
    :type tpa:
    :param nmbr_channels:
    :type nmbr_channels:
    :param nmbr_events:
    :type nmbr_events:
    :param group_name:
    :type group_name:
    :param key_name:
    :type key_name:
    :param chunk_size:
    :type chunk_size:
    :param channel:
    :type channel:
    :return:
    :rtype:
    """

    # add predictions to the h5 file
    data = f_handle.require_group(group_name)
    if tpa:
        data.require_dataset(name=key_name,
                             shape=(nmbr_events),
                             dtype=float)
    else:
        data.require_dataset(name=key_name,
                             shape=(nmbr_channels, nmbr_events),
                             dtype=float)

    count = 0
    while count < nmbr_events - chunk_size:
        print('count: ', count)

        # make input data
        x = {'event_ch0': f_handle[group_name]['event'][channel,
                          count:count + chunk_size]}  # array of shape: (nmbr_events, nmbr_features)

        # make predictions
        prediction = model.predict(x).numpy()
        if tpa:
            data[key_name][count:count + chunk_size] = prediction.reshape(-1)
        else:
            data[key_name][channel, count:count + chunk_size] = prediction.reshape(-1)

        count += chunk_size
    # make input data
    x = {'event_ch0': f_handle[group_name]['event'][channel, count:]}  # array of shape: (nmbr_events, nmbr_features)

    # make predictions
    prediction = model.predict(x).numpy()

    if tpa:
        data[key_name][count:] = prediction.reshape(-1)
    else:
        data[key_name][channel, count:] = prediction.reshape(-1)