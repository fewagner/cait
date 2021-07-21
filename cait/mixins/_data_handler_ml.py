# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

import numpy as np
import h5py
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

# -----------------------------------------------------------
# CLASS
# -----------------------------------------------------------

class MachineLearningMixin(object):
    """
    A Mixin Class to the DataHandler Class with methods for machine learning predictions.
    """

    def apply_pca(self, nmbr_components=2, type='events', down=1, batchsize=500, fit_idx=None):
        """
        Apply a principal component analysis to the data matrix and store projections, reconstruction error and
        components.

        This method was used in "Pulse Shape Discrimination in CUPID-Mo using Principal Component Analysis
        (doi: 10.1088/1748-0221/16/03/P03032)"
        for the dicrimination of similar signal types.

        :param nmbr_components: The number of components we want to store and project to.
        :type nmbr_components: int
        :param type: The group name in the HDF5 file, either 'events' or 'testpulses'.
        :type type: string
        :param down: We downsample the events by this factor.
        :type down: int
        :param batchsize: In the incremental PCA calculate an estimation of the eigenvectors on subsets of the data set
            with the size batchsize.
        :type batchsize: int
        :param fit_idx: We use only these indices for the fit, but calculate projections for all events.
        :type fit_idx: list of ints
        """

        with h5py.File(self.path_h5, 'r+') as f:

            def downsample(X):
                X_down = np.empty([len(X), int(len(X[0]) / down)])
                for i, x in enumerate(X):
                    X_down[i] = np.mean(x.reshape(int(X.shape[1] / down), down), axis=1)
                return X_down

            def remove_offset(X):
                for x in iter(X):
                    x -= np.mean(x[:int(self.record_length / 8)], axis=0, keepdims=True)
                return X

            rem_off = FunctionTransformer(remove_offset)
            downsam = FunctionTransformer(downsample)
            pca = IncrementalPCA(n_components=nmbr_components, batch_size=batchsize)
            pipe = Pipeline([('downsample', downsam),
                             ('remove_offset', rem_off),
                             ('PCA', pca)])

            if 'pca_projection' in f[type]:
                print('Overwrite old pca projections')
                del f[type]['pca_projection']
            if 'pca_components' in f[type]:
                print('Overwrite old pca components')
                del f[type]['pca_components']
            pca_projection = f[type].create_dataset(name='pca_projection',
                                                    shape=(
                                                        self.nmbr_channels, len(f[type]['event'][0]), nmbr_components),
                                                    dtype=float)
            pca_error = f[type].require_dataset(name='pca_error',
                                                shape=(self.nmbr_channels, len(f[type]['event'][0])),
                                                dtype=float)
            pca_components = f[type].create_dataset(name='pca_components',
                                                    shape=(
                                                        self.nmbr_channels, nmbr_components,
                                                        len(f[type]['event'][0, 0])),
                                                    dtype=float)

            for c in range(self.nmbr_channels):
                print('Channel ', c)
                print('Fitting ...')
                if fit_idx is None:
                    X = f[type]['event'][c]
                else:
                    X = f[type]['event'][c, fit_idx]
                pipe.fit(X)

                print('Explained Variance: ', pipe['PCA'].explained_variance_ratio_)
                print('Singular Values: ', pipe['PCA'].singular_values_)

                # save the transformed events and their error
                print('Saving predictions ...')
                for i, ev in tqdm(enumerate(f[type]['event'][c])):
                    preprocessed_ev = pipe['downsample'].transform(ev.reshape(1, -1))
                    preprocessed_ev = pipe['remove_offset'].transform(preprocessed_ev)
                    transformed_ev = pipe['PCA'].transform(preprocessed_ev)
                    pca_projection[c, i] = transformed_ev

                    pca_error[c, i] = np.mean(
                        (pipe['PCA'].inverse_transform(transformed_ev) - preprocessed_ev) ** 2)

                # save the principal components
                for i in range(nmbr_components):
                    # create the unit vector in the transformed pca
                    transformed = np.zeros(nmbr_components)
                    transformed[i] = 1
                    comp = pca.inverse_transform(transformed.reshape(1, -1))
                    pca_components[c, i, :] = comp.reshape(-1).repeat(down, axis=0)
