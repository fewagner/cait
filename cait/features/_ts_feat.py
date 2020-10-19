# ---------------------------------------------------------
# IMPORT
# ---------------------------------------------------------

import numpy as np
import tsfel as ts

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def calc_ts_features(events, mainpar, nmbr_channels,
                     nmbr_events, record_length,
                     down=1, sample_frequency=25000, scaler=None):

    # downsample
    events = events.reshape(nmbr_channels, nmbr_events, int(record_length/down), down)
    events = np.mean(events, axis=3)

    # remove offset
    events = events[:, :, :] - np.mean(events[:, :, :int(1000 / down)], axis=2)[..., np.newaxis]

    # reshape mainpars
    mainpar = mainpar.reshape(nmbr_channels, nmbr_events, -1)

    # calc features
    cfg_file = ts.get_features_by_domain()

    features = []
    for i in range(nmbr_channels):
        features.append(ts.time_series_features_extractor(cfg_file,
                                                          events[i].reshape((-1)),
                                                          fs=int(sample_frequency / down),
                                                          window_splitter=True,
                                                          window_size=int(record_length/down)))

        # add mainpar
        features[i]['Pulse Height'] = mainpar[i, :, 0]
        features[i]['Onset'] = mainpar[i, :, 1]
        features[i]['Rise Time'] = mainpar[i, :, 2]
        features[i]['Max Time'] = mainpar[i, :, 3]
        features[i]['Decay Start'] = mainpar[i, :, 4]
        features[i]['Half Time'] = mainpar[i, :, 5]
        features[i]['End Time'] = mainpar[i, :, 6]
        features[i]['Offset'] = mainpar[i, :, 7]
        features[i]['Linear Drift'] = mainpar[i, :, 8]
        features[i]['Quadratic Drift'] = mainpar[i, :, 9]

        # remove inf indices
        inf_indices = features[i].index[np.isinf(features[i]).any(1)]
        print('INF Indices channel {}: {}'.format(i, inf_indices))

        # scale
        features[i] = features[i].to_numpy()
        features[i][inf_indices] = 0
        if scaler:
            features[i] = scaler.transform(features[i])

    return features