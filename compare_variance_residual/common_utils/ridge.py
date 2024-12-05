import itertools as itools
import logging
import time

import himalaya
import numpy as np
from himalaya.ridge import GroupRidgeCV
from ridge_utils.utils import counter


GROUP_CV_SOLVER_PARAMS = dict(n_iter=1, n_targets_batch=100, n_targets_batch_refit=100, alphas=np.logspace(0, 4, 10),
                              score_func=himalaya.scoring.correlation_score, progress_bar=True)

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


def bootstrap_ridge(
        stim_train, resp_train, stim_test, resp_test, alphas, nboots, chunklen, nchunks,
        joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
        use_corr=True, return_wt=True, logger=logging.Logger("bootstrap_ridge")):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
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
    singcutoff : float, default 1e-10
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean, default False
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    return_wt : boolean, default True
        If True, this function will compute and return the regression weights after finding the best
        alpha parameter for each voxel. However, for very large models this can lead to memory issues.
        If false, this function will _not_ compute weights, but will still compute prediction performance
        on the prediction dataset (Pstim, Presp).
    logger : logging.Logger, default logging.Logger("bootstrap_ridge")

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

        # get indices for training / testing
        # all_indexes = range(nresp)
        # index_chunks = list(zip(*[iter(all_indexes)]*chunklen))
        # random.shuffle(index_chunks)
        # tune_indexes_ = list(itools.chain(*index_chunks[:nchunks]))
        # train_indexes_ = list(set(all_indexes)-set(tune_indexes_))
        # valinds.append(tune_indexes_)
        train_indexes_, tune_indexes_ = splits[bi]

        # Select data
        stim_train_ = stim_train[train_indexes_, :]
        stim_test_ = stim_train[tune_indexes_, :]
        resp_train_ = resp_train[train_indexes_, :]
        resp_test_ = resp_train[tune_indexes_, :]

        # Run ridge regression using this test set
        model = GroupRidgeCV(groups=None, random_state=12345, solver_params=GROUP_CV_SOLVER_PARAMS)
        model.fit(stim_train_, resp_train_)
        correlation_matrix_ = model.score(stim_test_, resp_test_)

        # correlation_matrix_ = _ridge_corr(
        #     stim_train_, stim_test_, resp_train_, resp_test_, alphas,
        #     corrmin=corrmin, singcutoff=singcutoff,
        #     normalpha=normalpha, use_corr=use_corr,
        #     logger=logger)
        correlation_matrices.append(correlation_matrix_)

    # Find best alphas
    if nboots > 0:
        all_correlation_matrices = np.dstack(correlation_matrices)
    else:
        all_correlation_matrices = None

    if not single_alpha:
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
    else:
        logger.debug("Finding single best alpha..")
        if nboots == 0:
            if len(alphas) == 1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = all_correlation_matrices.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]

        valphas = np.array([bestalpha]*nvox)
        logger.debug("Best alpha = %0.3f" % bestalpha)

    if return_wt:
        # Find weights
        logger.debug(
            "Computing weights for each response using entire training set..")
        wt = ridge(stim_train, resp_train, valphas,
                   singcutoff=singcutoff, normalpha=normalpha)

        # Predict responses on prediction set
        logger.debug("Predicting responses for predictions set..")
        pred = np.dot(stim_test, wt)

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        if use_corr:
            corrs = np.nan_to_num(np.array([np.corrcoef(resp_test[:, ii], nnpred[:, ii].ravel())[0, 1]
                                            for ii in range(resp_test.shape[1])]))
        else:
            residual_variance = (resp_test-pred).var(0)
            residual_sum_of_squares = 1 - \
                (residual_variance / resp_test.var(0))
            corrs = np.sqrt(np.abs(residual_sum_of_squares)) * \
                np.sign(residual_sum_of_squares)

        return wt, corrs, valphas, all_correlation_matrices, valinds
    else:
        # get correlations for prediction dataset directly
        corrs = ridge_corr_pred(
            stim_train, stim_test, resp_train, stim_test, valphas,
            normalpha=normalpha, use_corr=use_corr, logger=logger, singcutoff=singcutoff)

        return [], corrs, valphas, all_correlation_matrices, valinds
