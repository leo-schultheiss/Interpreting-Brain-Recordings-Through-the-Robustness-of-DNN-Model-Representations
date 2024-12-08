import itertools as itools
import logging

import himalaya
import numpy as np
from himalaya.ridge import GroupRidgeCV
from ridge_utils.utils import counter
from sklearn.model_selection import BaseCrossValidator

ridge_logger = logging.getLogger("ridge_corr")


class WholeDatasetSplitter(BaseCrossValidator):
    """Yields the whole dataset as training and testing set.
    We use this since himalaya only provides a cross-validation solver for group ridge regression. Instead, we want to be able to use bootstrap sampling."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        yield indices, indices


def gen_temporal_chunk_splits(num_splits: int, num_examples: int, chunk_len: int, num_chunks: int, seed=42):
    rng = np.random.RandomState(seed)
    all_indexes = range(num_examples)
    index_chunks = list(zip(*[iter(all_indexes)] * chunk_len))
    splits_list = []
    for _ in range(num_splits):
        rng.shuffle(index_chunks)
        tune_indexes_ = list(itools.chain(*index_chunks[:num_chunks]))
        train_indexes_ = list(set(all_indexes) - set(tune_indexes_))
        splits_list.append((train_indexes_, tune_indexes_))
    return splits_list


def bootstrap_ridge(stim_train, resp_train, stim_test, resp_test, alphas, nboots, chunklen, nchunks, joined=None,
                    single_alpha=False, use_corr=True, n_iter=1, n_targets_batch=10, n_targets_batch_refit=10,
                    n_alphas_batch=5, logger=ridge_logger, random_state=12345):
    """From https://github.com/csinva/fmri/blob/master/neuro/encoding/ridge.py
    Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.

    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist
    will be given the same regularization parameter (the one that is the best on average).

    Parameters
    ----------
    stim_train : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    resp_train : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    stim_test : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    resp_test : array_like, shape (TP, M)
        Test responses with TP time points and M different responses. Each response should be Z-scored across
        time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this
        product should be about 20 percent of the total length of the training data.
    joined : None or list of array_like indices, default None
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    n_iter : int, default 1
        Number of feature-space weights combination to search. If an array is given, the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution.
    n_targets_batch : int, default 10
        Number of targets to use in each batch. If None, all targets are used.
    n_targets_batch_refit : int, default 10
        Number of targets to use in each batch for refit. If None, all targets are used.
    n_alphas_batch : int, default 5
        Number of alphas to try in each batch.
    logger : logging.Logger, default logging.Logger("bootstrap_ridge")
    random_state : int, default 12345
        Random seed for reproducibility.

    Returns
    -------
    wt : array_like, shape (N, M)
        If [return_wt] is True, regression weights for N features and M responses. If [return_wt] is False, [].
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = resp_train.shape
    splits = gen_temporal_chunk_splits(
        nboots, nresp, chunklen, nchunks)
    valinds = [splits[1] for splits in splits]

    correlation_matrices = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.debug("Selecting held-out test set..")
        train_indexes_, tune_indexes_ = splits[bi]

        # Select data
        stim_train_ = stim_train[train_indexes_, :]
        stim_test_ = stim_train[tune_indexes_, :]
        resp_train_ = resp_train[train_indexes_, :]
        resp_test_ = resp_train[tune_indexes_, :]

        # Run ridge regression using this test set
        logger.info(f"Running ridge regression on bootstrap sample {bi}")
        correlation_matrix_, model_best_alphas = group_ridge(stim_train_, stim_test_, resp_train_, resp_test_, alphas,
                                                             n_iter, n_targets_batch, n_targets_batch_refit,
                                                             random_state, n_alphas_batch, use_corr)
        # print some statistics
        logger.debug(
            f"Mean correlation: {correlation_matrix_.mean()}, max correlation: {correlation_matrix_.max()}, min correlation: {correlation_matrix_.min()}, best alphas: {model_best_alphas}")
        correlation_matrices.append(correlation_matrix_)

    # Find best alphas
    if nboots > 0:
        all_correlation_matrices = np.dstack(correlation_matrices)
    else:
        all_correlation_matrices = None

    if single_alpha:
        logger.debug("Finding single best alpha..")
        if nboots == 0:
            if len(alphas) == 1:
                bestalphaind = 0
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = all_correlation_matrices.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
        bestalpha = alphas[bestalphaind]
        valphas = bestalpha
        logger.debug("Best alpha = %0.3f" % bestalpha)
    else:
        if nboots == 0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")
        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = all_correlation_matrices.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = all_correlation_matrices[:, jl, :].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
        logger.debug("Best alphas = %s" % valphas)

    logger.info("Calculating overall correlation based on optimal alphas")
    # get correlations for prediction dataset directly
    corrs, model_best_alphas = group_ridge(stim_train, stim_test, resp_train, resp_test, valphas, n_iter,
                                           n_targets_batch, n_targets_batch_refit, random_state, n_alphas_batch,
                                           use_corr)
    logger.debug(
        f"Mean correlation: {corrs.mean()}, max correlation: {corrs.max()}, min correlation: {corrs.min()}, best alphas: {model_best_alphas}")
    return [], corrs, valphas, all_correlation_matrices, valinds


def group_ridge(stim_train, stim_test, resp_train, resp_test, alphas, n_iter, n_targets_batch, n_targets_batch_refit,
                random_state, n_alphas_batch, use_corr=True):
    GROUP_CV_SOLVER_PARAMS = dict(alphas=alphas,
                                  score_func=himalaya.scoring.correlation_score, local_alpha=False,
                                  progress_bar=True, n_iter=n_iter, n_targets_batch=n_targets_batch,
                                  n_targets_batch_refit=n_targets_batch_refit, n_alphas_batch=n_alphas_batch)

    # create "fake" cross validation splitter that returns whole dataset since we don't want to do cross validation
    cv = WholeDatasetSplitter()
    model = GroupRidgeCV(cv=cv, groups=None, random_state=random_state, solver_params=GROUP_CV_SOLVER_PARAMS)
    model.fit(stim_train, resp_train)
    predictions = model.predict(stim_test)
    if use_corr:
        score = np.array([np.corrcoef(resp_test[:, ii], predictions[:, ii].ravel())[0, 1]
                          for ii in range(resp_test.shape[1])])
    else:
        # compute R^2
        score = np.array([1 - np.mean((resp_test[:, ii] - predictions[:, ii].ravel()) ** 2) / np.var(resp_test[:, ii])
                          for ii in range(resp_test.shape[1])])
    return score, model.best_alphas_
