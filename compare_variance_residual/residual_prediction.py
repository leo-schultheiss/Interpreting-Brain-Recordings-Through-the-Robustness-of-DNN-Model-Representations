import logging
import os

import numpy as np
from himalaya.ridge import RidgeCV
from ridge_utils.util import make_delayed
from sklearn.compose import ColumnTransformer
from voxelwise_tutorials.delayer import Delayer

from compare_variance_residual.common_utils.feature_utils import load_subject_fmri, load_low_level_textual_features, \
    get_prediction_path
from compare_variance_residual.common_utils.npp import zscore
from compare_variance_residual.common_utils.ridge import bootstrap_ridge
from plotting.feature_comparison import language_model

logging.basicConfig(level=logging.DEBUG)


def cross_val_ridge(train_features, train_data, n_splits=10,
                    lambdas=np.array([10 ** i for i in range(-6, 10)]),
                    method='plain',
                    do_plot=False):
    ridge_1 = dict(plain=ridge_by_lambda,
                   svd=ridge_by_lambda_svd,
                   kernel_ridge=kernel_ridge_by_lambda,
                   kernel_ridge_svd=kernel_ridge_by_lambda_svd,
                   ridge_sk=ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain=ridge,
                   svd=ridge_svd,
                   kernel_ridge=kernel_ridge,
                   kernel_ridge_svd=kernel_ridge_svd,
                   ridge_sk=ridge_sk)[method]

    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        #         print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn], train_data[trn],
                       train_features[val], train_data[val],
                       lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost, aspect='auto')
        r_cv += cost
    #         if icv%3 ==0:
    #             print(icv)
    #         print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv, aspect='auto', cmap='RdBu_r');

    argmin_lambda = np.argmin(r_cv, axis=0)
    weights = np.zeros((train_features.shape[1], train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:, idx_vox] = ridge_2(train_features, train_data[:, idx_vox], lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])


def residuals_textual(data_dir, target_data, low_level_feature, layer):
    trim = 5

    source_train, source_val = load_low_level_textual_features(data_dir)
    sourcedata_train = np.vstack(
        [zscore(source_train[story][low_level_feature][5 + trim:-trim]) for story in source_train.keys()])
    sourcedata_test = np.vstack(
        [zscore(source_val[story][low_level_feature][5 + trim:-trim]) for story in source_val.keys()])

    y_train = np.vstack([zscore(target_data.item()[story][layer][5 + trim:-trim]) for story in
                         list(target_data.item().keys())[:-1]])
    y_test = np.vstack([zscore(target_data.item()[story][layer][5 + trim:-trim]) for story in
                        list(target_data.item().keys())[-1:]])

    model = RidgeCV(cv=10, alphas=np.logspace(-1, 4, 5))
    model.fit(sourcedata_train, y_train)

    y_pred = model.predict(sourcedata_test)
    Rstim = y_test - y_pred

    y_pred = model.predict(sourcedata_train)
    Pstim = y_train - y_pred

    return Rstim, Pstim


def predict_residual(data_dir, subject_num, modality, layer, feature_filename, low_level_feature, number_of_delays=4):
    stimulus_features = np.load(feature_filename, allow_pickle=True)  # This file contains already downsampled data
    Rstim, Pstim = residuals_textual(data_dir, stimulus_features, low_level_feature, layer)

    # Delay stimuli
    delays = range(1, number_of_delays + 1)
    ct = ColumnTransformer(
        [("context_without_residual", Delayer(delays), slice(0, Rstim.shape[1] - 1))]
    )
    zRresp, zPresp = load_subject_fmri(data_dir, subject_num, modality)

    corr, coef, alphas = bootstrap_ridge(Rstim, zRresp, Pstim, zPresp, ct)

    save_file = get_prediction_path(language_model, "residual", modality, subject_num, low_level_feature, layer)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(save_file)
    np.save(os.path.join(str(save_file), "layer_" + str(layer)), corr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute residuals from brain predictions")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="../data")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--layer", help="Layer of natural language model to use for semantic representation of words",
                        type=int, default=9)
    parser.add_argument("--feature_filename", help="Choose feature", type=str, default="../bert_base20.npy")
    parser.add_argument("--low_level_feature", help="Low level feature to use. Possible options:\n"
                                                    "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    args = parser.parse_args()
    print(args)

    predict_residual()
