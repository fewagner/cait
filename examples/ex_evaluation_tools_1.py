# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, mixture
import cait as ai


et = ai.EvaluationTools()
np.random.seed(seed=1) # fixing random seed


path = './toy_data/'
plot_dir = './plots/'
fnames = ['bck_001-P_Ch26-L_Ch27-all-all',
          'bck_031-P_Ch26-L_Ch27-all-all',
          'bck_064-P_Ch26-L_Ch27-all-all']
filepath = ['{}{}.h5'.format(path, f) for f in fnames]

pl_channel = 0

et.add_events_from_file(file = filepath[0],
                        channel= pl_channel,
                        which_data = 'mainpar'
                        )
et.add_events_from_file(file = filepath[1],
                        channel= pl_channel,
                        which_data = 'mainpar'
                        )
et.split_test_train(test_size=0.80)


print('------------RFC------------')
clf_rf = RandomForestClassifier(criterion='entropy', max_depth=7)
clf_rf.fit(et.get_train_features(), et.get_train_label_nbrs())
print(clf_rf.score(et.features, et.label_nbrs))
print(clf_rf.score(et.get_train_features(), et.get_train_label_nbrs()))
print(clf_rf.score(et.get_test_features(), et.get_test_label_nbrs()))
et.add_prediction('RFC', clf_rf.predict(et.features), true_labels=True)
print('---------------------------')


print('------------GBC------------')
clf_gb = GradientBoostingClassifier(n_estimators=15, max_depth=7, learning_rate=0.1)
clf_gb.fit(et.get_train_features(), et.get_train_label_nbrs())
# print(clf_rf.score(e.features, e.label_nbrs))
print(clf_gb.score(et.get_train_features(), et.get_train_label_nbrs()))
print(clf_gb.score(et.get_test_features(), et.get_test_label_nbrs()))
et.add_prediction('GBC', clf_gb.predict(et.features), true_labels=True)
print('---------------------------')


print('------------SVM------------')
clf_svm = svm.SVC()
clf_svm.fit(et.get_train_features(), et.get_train_label_nbrs())
# print(clf_svm.score(e.features, e.label_nbrs))
print(clf_svm.score(et.get_train_features(), et.get_train_label_nbrs()))
print(clf_svm.score(et.get_test_features(), et.get_test_label_nbrs()))
svm_pred = clf_svm.predict(et.features)
et.add_prediction('SVM', svm_pred, true_labels=True)
print('---------------------------')



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

# e.plot_labels_distribution(figsize=(7,4))

figsize = (9,6)

# e.plt_pred_with_tsne([], plt_what='all', figsize=(9,5), verb=True)
# e.plt_pred_with_tsne(['GMM'], plt_what='all', figsize=figsize, verb=True)
# e.plt_pred_with_tsne(['BGMM'], plt_what='all', figsize=figsize, verb=True)
# e.plt_pred_with_tsne(['GMM','BGMM'], plt_what='all', verb=True)
# e.plt_pred_with_tsne(['GMM','BGMM'], plt_what='all', plt_labels=False, verb=True)
# e.plt_pred_with_tsne(['SVM'], plt_what='test', verb=True)
# e.plt_pred_with_tsne(['RFC'], plt_what='test', verb=True)
# e.plt_pred_with_tsne(['SVM', 'RFC'], plt_what='test', verb=True)
# e.plt_pred_with_tsne(['BGMM','RFC'], plt_what='test', verb=True)
# e.plt_pred_with_tsne(['GBC','RFC'], plt_what='test', verb=True)
# exit()

figsize = (8,5)
# e.confusion_matrix_pred('GMM', what='all', rotation_xticklabels=0, force_xlabelnbr=True, figsize=figsize)
# e.confusion_matrix_pred('BGMM', what='all', rotation_xticklabels=0, force_xlabelnbr=True, figsize=figsize)
# e.confusion_matrix_pred('SVM', what='test', rotation_xticklabels=0, force_xlabelnbr=True, figsize=figsize)
et.confusion_matrix_pred('RFC', what='test', rotation_xticklabels=0, force_xlabelnbr=True, figsize=figsize)
#
exit()


figsize = (6,4)
# e.pulse_height_histogram(extend_plot=True, figsize=None, verb=True)
# e.events_saturated_histogram(figsize=figsize, bins=100, verb=True)
# e.events_saturated_histogram(figsize=figsize, bins=100, ylog=True, verb=True)

# exit()


# e.correctly_labeled_per_mv('RFC', what='test', verb=True)
et.correctly_labeled_events_per_pulse_height('RFC', what='test', bin_size=7, figsize=figsize, verb=True)

exit()




et.plt_pred_with_tsne(['GMM'], plt_what='test', verb=True)
et.plt_pred_with_tsne(['BGMM'], plt_what='test', verb=True)
et.plt_pred_with_tsne(['GMM', 'BGMM'], plt_what='test', verb=True)
et.plt_pred_with_tsne(['SVM'], plt_what='test', verb=True)
et.plt_pred_with_tsne(['RFC'], plt_what='test', verb=True)
et.plt_pred_with_tsne(['SVM', 'RFC'], plt_what='test', verb=True)
et.plt_pred_with_tsne(['BGMM', 'RFC'], plt_what='test', verb=True)

et.confusion_matrix_pred('RFC', what='test', rotation_xticklabels=35)
et.confusion_matrix_pred('BGMM')




# exit()


# import ipdb; ipdb.set_trace()
