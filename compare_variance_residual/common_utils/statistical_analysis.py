import numpy as np
import scipy.linalg
from matplotlib.pyplot import figure, show


def best_corr_vec(wvec, vocab, SU, n=10):
    """Returns the [n] words from [vocab] most similar to the given [wvec], where each word is represented
    as a row in [SU].  Similarity is computed using correlation."""
    wvec = wvec - np.mean(wvec)
    nwords = len(vocab)
    corrs = np.nan_to_num([np.corrcoef(wvec, SU[wi, :] - np.mean(SU[wi, :]))[1, 0] for wi in range(nwords - 1)])
    scorrs = np.argsort(corrs)
    words = list(reversed([(corrs[i], vocab[i]) for i in scorrs[-n:]]))
    return words


def get_word_prob():
    """Returns the probabilities of all the words in the mechanical turk video labels.
    """
    import constants as c
    import cPickle
    data = cPickle.load(open(c.datafile))  # Read in the words from the labels
    wordcount = dict()
    totalcount = 0
    for label in data:
        for word in label:
            totalcount += 1
            if word in wordcount:
                wordcount[word] += 1
            else:
                wordcount[word] = 1

    wordprob = dict([(word, float(wc) / totalcount) for word, wc in wordcount.items()])
    return wordprob


def best_prob_vec(wvec, vocab, space, wordprobs):
    """Orders the words by correlation with the given [wvec], but also weights the correlations by the prior
    probability of the word appearing in the mechanical turk video labels.
    """
    words = best_corr_vec(wvec, vocab, space, n=len(vocab))  ## get correlations for all words
    ## weight correlations by the prior probability of the word in the labels
    weightwords = []
    for wcorr, word in words:
        if word in wordprobs:
            weightwords.append((wordprobs[word] * wcorr, word))

    return sorted(weightwords, key=lambda ww: ww[0])


def find_best_words(vectors, vocab, wordspace, actual, display=True, num=15):
    cwords = []
    for si in range(len(vectors)):
        cw = best_corr_vec(vectors[si], vocab, wordspace, n=num)
        cwords.append(cw)
        if display:
            print("Closest words to scene %d:" % si)
            print([b[1] for b in cw])
            print("Actual words:")
            print(actual[si])
            print("")
    return cwords


def find_best_stims_for_word(wordvector, decstims, n):
    """Returns a list of the indexes of the [n] stimuli in [decstims] (should be decoded stimuli)
    that lie closest to the vector [wordvector], which should be taken from the same space as the
    stimuli.
    """
    scorrs = np.array([np.corrcoef(wordvector, ds)[0, 1] for ds in decstims])
    scorrs[np.isnan(scorrs)] = -1
    return np.argsort(scorrs)[-n:][::-1]


def princomp(x, use_dgesvd=False):
    """Does principal components analysis on [x].
    Returns coefficients, scores and latent variable values.
    Translated from MATLAB princomp function.  Unlike the matlab princomp function, however, the
    rows of the returned value 'coeff' are the principal components, not the columns.
    """

    n, p = x.shape
    # cx = x-np.tile(x.mean(0), (n,1)) ## column-centered x
    cx = x - x.mean(0)
    r = np.min([n - 1, p])  ## maximum possible rank of cx

    if use_dgesvd:
        from svd_dgesvd import svd_dgesvd
        U, sigma, coeff = svd_dgesvd(cx, full_matrices=False)
    else:
        U, sigma, coeff = np.linalg.svd(cx, full_matrices=False)

    sigma = np.diag(sigma)
    score = np.dot(cx, coeff.T)
    sigma = sigma / np.sqrt(n - 1)

    latent = sigma ** 2

    return coeff, score, latent


def eigprincomp(x, npcs=None, norm=False, weights=None):
    """Does principal components analysis on [x].
    Returns coefficients (eigenvectors) and eigenvalues.
    If given, only the [npcs] greatest eigenvectors/values will be returned.
    If given, the covariance matrix will be computed using [weights] on the samples.
    """
    n, p = x.shape
    # cx = x-np.tile(x.mean(0), (n,1)) ## column-centered x
    cx = x - x.mean(0)
    r = np.min([n - 1, p])  ## maximum possible rank of cx

    xcov = np.cov(cx.T)
    if norm:
        xcov /= n

    if npcs is not None:
        latent, coeff = scipy.linalg.eigh(xcov, eigvals=(p - npcs, p - 1))
    else:
        latent, coeff = np.linalg.eigh(xcov)

    ## Transpose coeff, reverse its rows
    return coeff.T[::-1], latent[::-1]


def weighted_cov(x, weights=None):
    """If given [weights], the covariance will be computed using those weights on the samples.
    Otherwise the simple covariance will be returned.
    """
    if weights is None:
        return np.cov(x)
    else:
        w = weights / weights.sum()  ## Normalize the weights
        dmx = (x.T - (w * x).sum(1)).T  ## Subtract the WEIGHTED mean
        wfact = 1 / (1 - (w ** 2).sum())  ## Compute the weighting factor
        return wfact * np.dot(w * dmx, dmx.T.conj())  ## Take the weighted inner product


def test_weighted_cov():
    """Runs a test on the weighted_cov function, creating a dataset for which the covariance is known
    for two different populations, and weights are used to reproduce the individual covariances.
    """
    T = 1000  ## number of time points
    N = 100  ## A signals
    M = 100  ## B signals
    snr = 5  ## signal to noise ratio

    ## Create the two datasets
    siga = np.random.rand(T)
    noisea = np.random.rand(T, N)
    respa = (noisea.T + snr * siga).T

    sigb = np.random.rand(T)
    noiseb = np.random.rand(T, M)
    respb = (noiseb.T + snr * sigb).T

    ## Compute self-covariance matrixes
    cova = np.cov(respa)
    covb = np.cov(respb)

    ## Compute the full covariance matrix
    allresp = np.hstack([respa, respb])
    fullcov = np.cov(allresp)

    ## Make weights that will recover individual covariances
    wta = np.ones([N + M, ])
    wta[N:] = 0

    wtb = np.ones([N + M, ])
    wtb[:N] = 0

    recova = weighted_cov(allresp, wta)
    recovb = weighted_cov(allresp, wtb)

    return locals()


def fixPCs(orig, new):
    """Finds and fixes sign-flips in PCs by finding the coefficient with the greatest
    magnitude in the [orig] PCs, then negating the [new] PCs if that coefficient has
    a different sign.
    """
    flipped = []
    for o, n in zip(orig, new):
        maxind = np.abs(o).argmax()
        if o[maxind] * n[maxind] > 0:
            ## Same sign, no need to flip
            flipped.append(n)
        else:
            ## Different sign, flip
            flipped.append(-n)

    return np.vstack(flipped)
