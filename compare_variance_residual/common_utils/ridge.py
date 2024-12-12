import itertools as itools
import logging

from himalaya.scoring import correlation_score, r2_score
import numpy as np
from himalaya.ridge import GroupRidgeCV, RidgeCV
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import make_pipeline

ridge_logger = logging.getLogger("ridge_corr")


class TemporalChunkSplitter(BaseCrossValidator):
    def __init__(self, num_splits, chunk_len, num_chunks, seed=42):
        self.num_splits = num_splits
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.num_splits

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.seed)
        num_examples = len(X)
        all_indexes = range(num_examples)
        index_chunks = list(zip(*[iter(all_indexes)] * self.chunk_len))

        # creates bootstrapped, random (with replacement) index slices
        for _ in range(self.num_splits):
            rng.shuffle(index_chunks)
            tune_indexes = list(itools.chain(*index_chunks[:self.num_chunks]))
            train_indexes = list(set(all_indexes) - set(tune_indexes))
            yield train_indexes, tune_indexes


def bootstrap_ridge(stim_train, resp_train, stim_test, resp_test, ct, alphas=np.logspace(-5, 15, 20), nboots=15,
                    chunklen=40, nchunks=20, single_alpha=True, singcutoff=1e-10,use_corr=True, n_iter=20, n_targets_batch=None,
                    n_targets_batch_refit=None, n_alphas_batch=10, logger=ridge_logger, random_state=42):
    """Adapted from https://github.com/csinva/fmri/blob/master/neuro/encoding/ridge.py to use himalaya
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
    ct : ColumnTransformer or ColumnTransformerNoStack with a list of transformers with entries (feature_name, transformer, columns)
        This estimator allows different columns or column subsets of the input to be transformed separately.
        The columns represent the indices of the separate feature groups fit in the banded ridge regression.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int, default 15
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int, default 40
        On each sample, the training data is broken into chunks of this length. This should be a few times
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int, default 10
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this
        product should be about 20 percent of the total length of the training data.
    single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    singcutoff : float, default 1e-10
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    n_iter : int, default 1
        Number of feature-space weights combination to search. If an array is given,
        the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution.
    n_targets_batch : int, default 10
        Number of targets to use in each batch. If None, all targets are used. Only used to constrain memeory usage.
    n_targets_batch_refit : int, default 10
        Number of targets to use in each batch for refit. If None, all targets are used. Only used to constrain memeory usage.
    n_alphas_batch : int, default 5
        Number of alphas to try in each batch. Only used to constrain memeory usage.
    logger : logging.Logger, default logging.Logger("bootstrap_ridge")
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    score : array_like, shape (M,)
        Validation set score. Either correlations if useCorrs or R^2.
        Predicted responses for the validation set are obtained using the regression weights: pred = np.dot(Pstim, wt),
        and then the correlation between each predicted response and each
        column in Presp is found.
    weights : array_like, shape (N, M)
        The weights for each feature and response. These are the weights that are used to predict the responses.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    """
    columns = dict((t[0], (t[2].start, t[2].stop)) for t in ct.transformers)
    logger.debug(f"Performing bootstrap ridge regression with temporal chunking on groups (columns/bands): {columns} "
                 f"on training set of shape: {stim_train.shape}, and on test set of shape: {stim_test.shape}")

    score_func = correlation_score if use_corr else r2_score
    common_solver_params = dict(score_func=score_func, local_alpha=not single_alpha,
                                n_targets_batch_refit=n_targets_batch_refit, n_targets_batch=n_targets_batch,
                                n_alphas_batch=n_alphas_batch)
    cv = TemporalChunkSplitter(nboots, chunklen, nchunks, random_state)
    if len(ct.transformers) == 1:  # use normal Ridge regression
        logger.info("Using single solver")
        single_solver_params = dict()
        solver_params = {**common_solver_params, **single_solver_params}
        model = RidgeCV(cv=cv, alphas=alphas, solver_params=solver_params)
    else:  # use banded ridge regression
        logger.info("Using group solver")
        group_solver_params = dict(n_iter=n_iter, alphas=alphas, progress_bar=True)
        solver_params = {**common_solver_params, **group_solver_params}
        model = GroupRidgeCV(cv=cv, groups="input", random_state=random_state, solver_params=solver_params)

    pipeline = make_pipeline(
        # scaler,
        ct,
        model
    )
    pipeline.fit(stim_train, resp_train)
    predictions = pipeline.predict(stim_test)

    if use_corr:  # compute pearson correlation
        evaluation_type = "correlation"
        score = np.array([np.corrcoef(resp_test[:, ii], predictions[:, ii].ravel())[0, 1]
                          for ii in range(resp_test.shape[1])])
    else:  # compute R^2
        evaluation_type = "R^2"
        score = np.array([1 - np.mean((resp_test[:, ii] - predictions[:, ii].ravel()) ** 2) / np.var(resp_test[:, ii])
                          for ii in range(resp_test.shape[1])])

    score = np.nan_to_num(score)
    logger.debug(
        f"Mean {evaluation_type}: {score.mean()}, max {evaluation_type}: {score.max()}, min {evaluation_type}: {score.min()}, best alpha(s): {model.best_alphas_}")

    return score, model.coef_, model.best_alphas_
