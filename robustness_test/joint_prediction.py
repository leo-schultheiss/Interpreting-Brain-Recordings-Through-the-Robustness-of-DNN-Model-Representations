import logging
import os.path

import numpy as np
from himalaya.ridge import GroupRidgeCV
from ridge_utils.util import make_delayed

from common_utils.npp import zscore
from common_utils.training_utils import load_subject_fmri, \
    load_low_level_textual_features, load_downsampled_context_representations, get_prediction_path
from plotting.display_rois_variance import language_model, subject

logging.basicConfig(level=logging.DEBUG)


def predict_joint_model(data_dir, feature_filename, subject_num, modality, layer, textual_features, output_dir):
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)
    Rstim, Pstim = [], []
    # join input features (context representations and low-level textual features)
    low_level_train, low_level_val = load_low_level_textual_features(data_dir)
    for feature in textual_features.split(","):
        if feature == "semantic":
            training_stim, prediction_stim = load_downsampled_context_representations(data_dir, feature_filename, layer)
        elif feature in low_level_train['story_01'].keys():
            training_stim = (
                np.vstack([zscore(low_level_train[story][feature][5 + 5:-5]) for story in low_level_train.keys()]))
            prediction_stim = (
                np.vstack([zscore(low_level_val[story][feature][5 + 5:-5]) for story in low_level_val.keys()]))
        else:
            raise ValueError(f"Textual feature {feature} not found in the dataset")

        print("training_stim.shape: ", training_stim.shape)
        print("prediction_stim.shape: ", prediction_stim.shape)
        Rstim.append(training_stim)
        Pstim.append(prediction_stim)


    # Delay stimuli to account for hemodynamic lag
    numer_of_delays = 4
    delays = range(1, numer_of_delays + 1)
    for feature in len(Rstim):
        Rstim[feature] = make_delayed(Rstim[feature], delays)
        Pstim[feature] = make_delayed(Pstim[feature], delays)

    model = GroupRidgeCV(groups="input", cv=5, random_state=12345)
    model.fit(Rstim, Rresp)
    voxelwise_correlations = model.score(Pstim, Presp)

    # save voxelwise correlations and predictions
    output_file = get_prediction_path(language_model, "joint", modality, subject, textual_features, layer)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_file, voxelwise_correlations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict fMRI data using joint model")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="../data")
    parser.add_argument("-c", "--feature_filename",
                        help="File with context representations from LM for each story", type=str,
                        default="../bert_base20.npy")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="listening")
    parser.add_argument("-l", "--layer", help="layer of the language model to use as input", type=int, default=9)
    parser.add_argument("--textual_features",
                        help="Comma separated, textual feature to use as input. Possible options include:\n"
                             "semantic, letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="semantic,letters")
    parser.add_argument("--output_dir", help="Output directory", type=str, default="../bert-joint-predictions")
    args = parser.parse_args()
    print(args)

    predict_joint_model(args.data_dir, args.feature_filename, args.subject_num, args.modality, args.layer,
                        args.textual_features, args.output_dir)
