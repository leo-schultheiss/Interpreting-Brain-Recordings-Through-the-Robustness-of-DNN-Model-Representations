import os.path

import himalaya
import numpy as np
from himalaya.ridge import GroupRidgeCV
from ridge_utils.util import make_delayed

from compare_variance_residual.common_utils.feature_utils import load_subject_fmri, load_downsampled_context_representations, \
    get_prediction_path, load_z_low_level_feature
from compare_variance_residual.common_utils.ridge import GROUP_CV_SOVER_PARAMS


def predict_joint_model(data_dir, feature_filename, language_model, subject_num, modality, layer, textual_features,
                        number_of_delays=4):
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)
    Rstim, Pstim = [], []
    # join input features (context representations and low-level textual features)
    for feature in textual_features.split(","):
        if feature == "semantic":
            training_stim, prediction_stim = load_downsampled_context_representations(data_dir, feature_filename, layer)
        elif feature in ['letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std']:
            training_stim, prediction_stim = load_z_low_level_feature(data_dir, feature)
        else:
            raise ValueError(f"Textual feature {feature} not found in the dataset")
        print("training_stim.shape: ", training_stim.shape)
        print("prediction_stim.shape: ", prediction_stim.shape)
        Rstim.append(training_stim)
        Pstim.append(prediction_stim)

    # Delay stimuli to account for hemodynamic lag
    delays = range(1, number_of_delays + 1)
    for feature in range(len(Rstim)):
        Rstim[feature] = make_delayed(Rstim[feature], delays)
        Pstim[feature] = make_delayed(Pstim[feature], delays)

    # Fit model
    model = GroupRidgeCV(groups="input", random_state=12345, solver_params=(GROUP_CV_SOVER_PARAMS))
    model.fit(Rstim, Rresp)
    voxelwise_correlations = model.score(Pstim, Presp)

    # save voxelwise correlations
    output_file = get_prediction_path(language_model, "joint", modality, subject_num, textual_features, layer)
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
    parser.add_argument("--language_model", help="Language model, where the features are extracted from", type=str,
                        default="bert")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("-l", "--layer", help="layer of the language model to use as input", type=int, default=9)
    parser.add_argument("--textual_features",
                        help="Comma separated, textual feature to use as input. Possible options include:\n"
                             "semantic, letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="semantic,letters")
    args = parser.parse_args()
    print(args)

    predict_joint_model(args.data_dir, args.feature_filename, args.language_model, args.subject_num, args.modality,
                        args.layer, args.textual_features)
    print("All done!")
