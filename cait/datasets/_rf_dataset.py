# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------

def get_rf_dataset(paths_h5,
                   channel,
                   include_ts_features,
                   random_seed,
                   test_size,
                   scalers = None):
    """
    DEPRICATED IN THE CURRENT VERSION
    """

    if include_ts_features:
        f = h5py.File(paths_h5[0], 'r')
        length_features = len(f['events']['ts_features'][0, 0])

    mainpar = np.empty((0, 10))
    if include_ts_features:
        features = np.empty((0, length_features))
    labels = np.empty((0))

    for path in paths_h5:
        f = h5py.File(path, 'r')
        mainpar = np.concatenate((mainpar, np.array(f['events']['mainpar'])[channel]), axis=0)
        labels = np.concatenate((labels, np.array(f['events']['labels'])[channel]), axis=0)
        if include_ts_features:
            features = np.concatenate((features, np.array(f['events']['ts_features'])[channel]), axis=0)
        f.close()

    mainpar = mainpar.reshape((-1, 10))
    if include_ts_features:
        features = features.reshape((-1, length_features))
    features = np.concatenate((mainpar, features), axis = 1)
    labels = labels.reshape((-1))

    unique, counts = np.unique(labels, return_counts=True)
    print('Label Counts: ', np.asarray((unique, counts)).T)

    if scalers is None:
        scaler = StandardScaler()
    else:
        scaler = scalers[channel]

    X = scaler.fit_transform(features)
    print('Features scaled.')

    np.random.seed(seed=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)

    return X_train, X_test, y_train, y_test, scaler