import os

from compare_variance_residual.common_utils.feature_utils import get_prediction_path
import numpy as np

def ssc(data: np.array):
    """
    Calculate the signed squared correlation of a matrix
    :param data: np.array
    :return: np.array
    """
    # return data ** 2
    return (data ** 2) * np.sign(data)


def variance_partitioning(language_model, modality, subject, low_level_feature, layer):
    # load numpy correlation data
    model_a_path = get_prediction_path(language_model, "semantic", modality, subject, low_level_feature, layer)
    model_b_path = get_prediction_path(language_model, "low-level", modality, subject, low_level_feature, layer)
    joint_model_path = get_prediction_path(language_model, "joint", modality, subject, low_level_feature, layer)
    model_a = np.load(model_a_path, allow_pickle=True)
    model_b = np.load(model_b_path, allow_pickle=True)
    joint_model = np.load(joint_model_path, allow_pickle=True)

    # remove nan values
    model_a = np.nan_to_num(model_a)
    model_b = np.nan_to_num(model_b)
    joint_model = np.nan_to_num(joint_model)

    # estimate the explained variance of each model using signed squared correlation
    squared_intersection = ssc(model_a) + ssc(model_b) - ssc(joint_model)
    squared_variance_a_minus_b = ssc(model_a) - squared_intersection
    squared_variance_b_minus_a = ssc(model_b) - squared_intersection

    # take roots of the squared values
    intersection = np.sqrt(squared_intersection)
    variance_a_minus_b = np.sqrt(squared_variance_a_minus_b)
    variance_b_minus_a = np.sqrt(squared_variance_b_minus_a)

    # output directory
    output_dir = os.path.join(f"{language_model}-variance-partitioning", modality, f"{subject:02}", low_level_feature, str(layer))
    # save the results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "intersection.npy"), intersection)
    np.save(os.path.join(output_dir, "semantic_minus_low.npy"), variance_a_minus_b)
    np.save(os.path.join(output_dir, "low_minus_semantic.npy"), variance_b_minus_a)

def variance_partitioning_3d(language_model, modality, subject, low_level_features, layer):
    # single model correlations
    low_level_correlations = []
    for low_level_feature in low_level_features.split(","):
        model_path = get_prediction_path(language_model, "low-level", modality, subject, low_level_feature, layer)
        model = np.load(model_path, allow_pickle=True)
        model = np.nan_to_num(model)
        low_level_correlations.append(model)
    semantic_correlation = np.load(get_prediction_path(language_model, "semantic", modality, subject, low_level_features, layer), allow_pickle=True)
    # double model correlations
    joint_correlations_bi = []
    for low_level_feature in low_level_features.split(","):
        model_path = get_prediction_path(language_model, "joint", modality, subject, "semantic," + low_level_feature, layer)
        model = np.load(model_path, allow_pickle=True)
        model = np.nan_to_num(model)
        joint_correlations_bi.append(model)
    model_path = get_prediction_path(language_model, "joint", modality, subject, low_level_features, layer)
    model = np.load(model_path, allow_pickle=True)
    model = np.nan_to_num(model)
    joint_correlations_bi.append(model)
    # triple model correlations
    joint_correlation_tri = np.load(get_prediction_path(language_model, "joint", modality, subject, low_level_features, layer), allow_pickle=True)

    # estimate the explained variance of each model using signed squared correlation
    squared_intersection = ssc(joint_correlation_tri)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Perform variance partitioning on two models")
    parser.add_argument("--language_model", help="Language model e.g. bert, gpt", type=str, default="bert")
    parser.add_argument("--modality", help="Modality of subject recording, reading or listening", type=str, default="listening")
    parser.add_argument("--subject", help="Subject e.g. 1, 2, 3", type=int, default=1)
    parser.add_argument("--low_level_feature", help="Low level feature e.g. letters, phonemes...", type=str, default="phonemes")
    parser.add_argument("--layer", help="Layer of the model", type=int, default=9)
    args = parser.parse_args()
    print(args)

    if args.low_level_feature.__contains__(","):
        variance_partitioning_3d(args.language_model, args.modality, args.subject, args.low_level_feature, args.layer)
    else:
        variance_partitioning(args.language_model, args.modality, args.subject, args.low_level_feature, args.layer)
