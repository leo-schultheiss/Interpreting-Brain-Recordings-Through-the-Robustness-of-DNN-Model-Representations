import logging
import os

import numpy as np
from himalaya.ridge import RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from voxelwise_tutorials.delayer import Delayer

from compare_variance_residual.common_utils.feature_utils import load_subject_fmri, load_low_level_textual_features, \
    get_prediction_path
from compare_variance_residual.common_utils.npp import zscore
from compare_variance_residual.common_utils.ridge import bootstrap_ridge

logging.basicConfig(level=logging.DEBUG)



def residuals_textual(data_dir, target_data, low_level_feature, layer):
    trim = 5

    source_train, source_val = load_low_level_textual_features(data_dir)
    sourcedata_train = np.vstack(
        [zscore(source_train[story][low_level_feature][5 + trim:-trim]) for story in source_train.keys()])
    sourcedata_test = np.vstack(
        [zscore(source_val[story][low_level_feature][5 + trim:-trim]) for story in source_val.keys()])
    sourcedata_train = np.nan_to_num(sourcedata_train)
    sourcedata_test = np.nan_to_num(sourcedata_test)
    print("stim test shape: ", sourcedata_test.shape)
    print("stim train shape: ", sourcedata_train.shape)

    y_train = np.vstack([zscore(target_data.item()[story][layer][5 + trim:-trim]) for story in
                         list(target_data.item().keys())[:-1]])
    y_test = np.vstack([zscore(target_data.item()[story][layer][5 + trim:-trim]) for story in
                        list(target_data.item().keys())[-1:]])
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)
    print("resp test shape: ", y_test.shape)
    print("resp train shape: ", y_train.shape)

    model = RidgeCV(cv=KFold(n_splits=10), alphas=np.logspace(-1, 4, 5))
    model.fit(sourcedata_train, y_train)

    y_pred = model.predict(sourcedata_train)
    Rstim = y_train - y_pred

    y_pred = model.predict(sourcedata_test)
    Pstim = y_test - y_pred

    return Rstim, Pstim


def predict_residual(data_dir, subject_num, modality, layer, feature_filename, low_level_feature, number_of_delays=4):
    downsampled_stimulus_features = np.load(feature_filename, allow_pickle=True)
    Rstim, Pstim = residuals_textual(data_dir, downsampled_stimulus_features, low_level_feature, layer)

    # Delay stimuli
    delays = range(1, number_of_delays + 1)
    ct = ColumnTransformer(
        [("context_without_residual", Delayer(delays), slice(0, Rstim.shape[1] - 1))]
    )
    zRresp, zPresp = load_subject_fmri(data_dir, subject_num, modality)

    corr, coef, alphas = bootstrap_ridge(Rstim, zRresp, Pstim, zPresp, ct)

    save_file = get_prediction_path("bert", "residual", modality, subject_num, low_level_feature, layer)
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
    parser.add_argument("--feature_filename", help="Choose feature", type=str, default="../bert_downsampled_data.npy")
    parser.add_argument("--low_level_feature", help="Low level feature to use. Possible options:\n"
                                                    "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    args = parser.parse_args()
    print(args)

    predict_residual(args.data_dir, args.subject_num, args.modality, args.layer, args.feature_filename,
                     args.low_level_feature)
