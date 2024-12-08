import logging
import argparse
import os

import h5py
from ridge_utils.ridge import bootstrap_ridge

from common_utils.residuals_text_speech import *
from compare_variance_residual.common_utils.feature_utils import load_subject_fmri

logging.basicConfig(level=logging.DEBUG)


# These files contain low-level textual and speech features
def load_low_level_textual_features():
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File(f'{data_dir}/features_trn_NEW.hdf', 'r+')
    base_features_val = h5py.File(f'{data_dir}/features_val_NEW.hdf', 'r+')
    return base_features_train, base_features_val


def load_low_level_speech_features(lowlevelfeature):
    # 'diphone', 'powspec', 'triphone'
    if lowlevelfeature in ['diphone', 'powspec', 'triphone']:
        df = h5py.File(f'{data_dir}/features_matrix.hdf')
        base_features_train = df[lowlevelfeature + '_train']
        base_features_val = df[lowlevelfeature + '_test']
    elif lowlevelfeature in 'articulation':
        base_features_train = np.load('data/articulation_train.npy')
        base_features_val = np.load('data/articulation_test.npy')
    return base_features_train, base_features_val


def load_low_level_visual_features():
    stimulus_data_file = np.load('m_ll.npz', allow_pickle=True)
    stimulus_data_file = {key: stimulus_data_file[key].item() for key in stimulus_data_file}
    train_matrix = stimulus_data_file['train']['7']  # (3737, 6555) matrix of (TRs, feature_dims) for train stories (train stories ordered in alphabetical order)
    test_matrix = stimulus_data_file['test']['7']  # (291, 6555) matrix of (TRs, feature_dims) for test story
    return train_matrix, test_matrix


trim = 5
data_dir = '../data/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute residuals from brain predictions")
    parser.add_argument("data_dir", help="Choose data directory", type=str, default="../data")
    parser.add_argument("subjectNum", help="Choose subject", type=int)
    parser.add_argument("featurename", help="Choose feature", type=str)
    parser.add_argument("modality", help="Choose modality", type=str)
    parser.add_argument("dirname", help="Choose Directory", type=str)
    parser.add_argument("layernum", help="Choose Layer Num", type=int)
    parser.add_argument("lowlevelfeature", help="Choose low-level feature name", type=str)
    args = parser.parse_args()

    stimulus_features = np.load(args.featurename, allow_pickle=True)  # This file contains already downsampled data

    if args.lowlevelfeature in ['letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std']:
        base_features_train, base_features_val = load_low_level_textual_features()
        residual_features = residuals_textual(base_features_train, base_features_val, stimulus_features,
                                              args.lowlevelfeature)
    elif args.lowlevelfeature in ['powspec', 'diphone', 'triphone', 'articulation']:
        base_features_train, base_features_val = load_low_level_speech_features(args.lowlevelfeature)
        residual_features = residuals_phones(base_features_train, base_features_val, stimulus_features,
                                             args.lowlevelfeature)
    elif args.lowlevelfeature in ['motion']:
        base_features_train, base_features_val = load_low_level_visual_features()
        residual_features = residuals_visual(base_features_train, base_features_val, stimulus_features,
                                             args.lowlevelfeature)

    # Delay stimuli
    from common_utils.statistical_analysis import make_delayed

    ndelays = 6
    delays = range(1, ndelays + 1)

    print("FIR model delays: ", delays)

    delRstim = []
    for layer in np.arange(12):
        delRstim.append(make_delayed(np.array(residual_features[args.lowlevelfeature][0][layer]), delays))

    delPstim = []
    for layer in np.arange(12):
        delPstim.append(make_delayed(np.array(residual_features[args.lowlevelfeature][1][layer]), delays))

    # Print the sizes of these matrices
    print("delRstim shape: ", delRstim[0].shape)
    print("delPstim shape: ", delPstim[0].shape)

    subject = '0' + str(args.subjectNum)

    nboots = 5  # Number of cross-validation runs.
    chunklen = 40  #
    nchunks = 20
    main_dir = os.path.join(args.dirname, args.modality, subject)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    for layer in np.arange(args.layernum, 12):
        zRresp, zPresp = load_subject_fmri(data_dir, subject, args.modality)
        alphas = np.logspace(1, 3,
                             10)  # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
        all_corrs = []
        wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(delRstim[layer]), zRresp,
                                                             np.nan_to_num(delPstim[layer]), zPresp,
                                                             alphas, nboots, chunklen, nchunks,
                                                             singcutoff=1e-10, single_alpha=True)
        pred = np.dot(np.nan_to_num(delPstim[layer]), wt)

        print("pred has shape: ", pred.shape)
        voxelwise_correlations = np.zeros((zPresp.shape[1],))  # create zero-filled array to hold correlations
        for vi in range(zPresp.shape[1]):
            voxelwise_correlations[vi] = np.corrcoef(zPresp[:, vi], pred[:, vi])[0, 1]
        print(voxelwise_correlations)

        np.save(os.path.join(str(main_dir), "layer_" + str(layer)), voxelwise_correlations)
