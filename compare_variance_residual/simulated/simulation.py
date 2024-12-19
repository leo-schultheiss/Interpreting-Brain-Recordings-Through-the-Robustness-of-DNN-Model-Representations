import numpy as np
from himalaya.backend import get_backend


def generate_dataset(n_targets=500,
                     n_samples_train=1000, n_samples_test=400,
                     noise=0, unique_contributions=None,
                     n_features_list=None, random_state=None, random_distribution=np.random.randn):
    """Utility to generate dataset.

    Parameters
    ----------
    n_targets : int
        Number of targets.
    n_samples_train : int
        Number of samples in the training set.
    n_samples_test : int
        Number of sample in the testing set.
    noise : float > 0
        Scale of the Gaussian white noise added to the targets.
    unique_contributions : list of floats
        Proportion of the target variance explained by each feature space.
    n_features_list : list of int of length (n_features, ) or None
        Number of features in each kernel. If None, use 1000 features for each.
    random_state : int, or None
        Random generator seed use to generate the true kernel weights.

    Returns
    -------
    Xs_train : array of shape (n_feature_spaces, n_samples_train, n_features)
        Training features.
    Xs_test : array of shape (n_feature_spaces, n_samples_test, n_features)
        Testing features.
    Y_train : array of shape (n_samples_train, n_targets)
        Training targets.
    Y_test : array of shape (n_samples_test, n_targets)
        Testing targets.
    kernel_weights : array of shape (n_targets, n_features)
        Kernel weights in the prediction of the targets.
    n_features_list : list of int of length (n_features, )
        Number of features in each kernel.
    """
    from himalaya.utils import check_random_state

    backend = get_backend()

    if unique_contributions is None:
        unique_contributions = [0.5, 0.5]

    if n_features_list is None:
        n_features_list = np.full(len(unique_contributions), fill_value=1000)

    rng = check_random_state(random_state)

    # Then, generate a random dataset, using the arbitrary scalings.
    Xs_train, Xs_test = [], []
    Xs_unique_test, Xs_unique_train = [], []
    Y_train, Y_test = None, None

    # Generate a shared component Z
    Z_train = rng.randn(n_samples_train, 1)
    Z_test = rng.randn(n_samples_test, 1)

    for ii, unique_contribution in enumerate(unique_contributions):
        n_features = n_features_list[ii]
        weights = rng.randn(n_features, n_targets) / n_features

        # generate random features
        X_train = rng.randn(n_samples_train, n_features)
        X_test = rng.randn(n_samples_test, n_features)

        # compute unique feature contribution, before adding shared component
        X_unique_train = unique_contribution * X_train @ weights
        X_unique_test = unique_contribution * X_test @ weights
        Xs_unique_train.append(X_unique_train)
        Xs_unique_test.append(X_unique_test)

        # add shared component
        shared_contribution = (1 - np.sum(unique_contributions)) / len(unique_contributions)
        print("shared_contribution", shared_contribution)
        X_train = shared_contribution * Z_train + unique_contribution * X_train
        X_test = shared_contribution * Z_test + unique_contribution * X_test

        # demean
        X_train -= X_train.mean(0)
        X_test -= X_test.mean(0)

        Xs_train.append(X_train)
        Xs_test.append(X_test)

        if ii == 0:
            Y_train = X_train @ weights
            Y_test = X_test @ weights
        else:
            Y_train += X_train @ weights
            Y_test += X_test @ weights

    std = Y_train.std(0)[None]
    Y_train /= std
    Y_test /= std

    Y_train += rng.randn(n_samples_train, n_targets) * noise
    Y_test += rng.randn(n_samples_test, n_targets) * noise
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    Xs_train = backend.asarray(Xs_train, dtype="float32")
    Xs_test = backend.asarray(Xs_test, dtype="float32")
    Y_train = backend.asarray(Y_train, dtype="float32")
    Y_test = backend.asarray(Y_test, dtype="float32")
    # Xs_unique_train = backend.asarray(Xs_unique_train, dtype="float32")
    # Xs_unique_test = backend.asarray(Xs_unique_test, dtype="float32")

    return Xs_train, Xs_test, Y_train, Y_test, n_features_list, Xs_unique_train, Xs_unique_test
