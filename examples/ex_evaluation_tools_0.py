"""

"""

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, mixture
import cait as ai

if __name__ == '__main__':
    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------

    et = ai.EvaluationTools()
    np.random.seed(seed=1)  # fixing random seed

    path = 'toy_data/run35_DetF/'
    plot_dir = 'plots/'
    fnames = ['bck_001-P_Ch26-L_Ch27-all-all']
    # fnames = ['bck_001-P_Ch26-L_Ch27-all-all',
    #           'bck_031-P_Ch26-L_Ch27-all-all',
    #           'bck_064-P_Ch26-L_Ch27-all-all']

    filepath = ['{}{}.h5'.format(path, f) for f in fnames]

    pl_channel = 0

    et.add_events_from_file(file=filepath[0],
                            channel=pl_channel,
                            which_data='mainpar'
                            )

    et.split_test_train(test_size=0.60)

    # print('------------RFC------------')
    clf_rf = RandomForestClassifier(criterion='entropy', max_depth=7)
    clf_rf.fit(et.get_train_features(), et.get_train_label_nbrs())
    # print(clf_rf.score(e.features, e.label_nbrs))
    # print(clf_rf.score(e.get_train_features(), e.get_train_label_nbrs()))
    # print(clf_rf.score(e.get_test_features(), e.get_test_label_nbrs()))
    et.add_prediction('RFC', clf_rf.predict(et.features), true_labels=True)
    # print('---------------------------')

    # print('------------SVM------------')
    clf_svm = svm.SVC()
    clf_svm.fit(et.get_train_features(), et.get_train_label_nbrs())
    # print(clf_svm.score(e.features, e.label_nbrs))
    # print(clf_svm.score(e.get_train_features(), e.get_train_label_nbrs()))
    # print(clf_svm.score(e.get_test_features(), e.get_test_label_nbrs()))

    svm_pred = clf_svm.predict(et.features)
    et.add_prediction('SVM', svm_pred, true_labels=True)
    # print('---------------------------')

    # -------- Bayesian Gaussian Mixture Model --------
    bgmm = mixture.BayesianGaussianMixture(n_components=len(np.unique(et.label_nbrs)),
                                           covariance_type='full').fit(et.features)
    bgmm_pred = bgmm.predict(et.features)
    et.add_prediction('BGMM', bgmm_pred)

    gmm = mixture.GaussianMixture(n_components=len(np.unique(et.label_nbrs)),
                                  covariance_type='full').fit(et.features)
    gmm_pred = gmm.predict(et.features)
    et.add_prediction('GMM', gmm_pred)

    et.save_plot_dir = plot_dir
    et.save_pgf = False

    et.plt_pred_with_tsne(['RFC'], plt_what='all', verb=True)
    et.plt_pred_with_pca(['RFC'], plt_what='all', verb=True)
