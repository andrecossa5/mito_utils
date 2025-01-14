"""
Moduele for kBET metric (Buttner et al., 2018). See Cellula.
"""

from joblib import cpu_count, parallel_backend, Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import binom
from sklearn.metrics import normalized_mutual_info_score



##


def chunker(n):
    """
    Create an np.array of starting indeces for parallel computation.
    """
    n_jobs = cpu_count()
    starting_indeces = np.zeros(n_jobs + 1, dtype=int)
    quotient = n // n_jobs
    remainder = n % n_jobs

    for i in range(n_jobs):
        starting_indeces[i+1] = starting_indeces[i] + quotient + (1 if i < remainder else 0)

    return starting_indeces


##


def kbet_one_chunk(index, batch, null_dist):
    """
    kBET calculation for a single index chunk.
    """
    dof = null_dist.size-1
    n = index.shape[0]
    k = index.shape[1]-1
    results = np.zeros((n, 2))

    for i in range(n):
        observed_counts = (
            pd.Series(batch[index[i,:]]).value_counts(sort=False).values
        )
        expected_counts = null_dist * k
        stat = np.sum(
            np.divide(
            np.square(np.subtract(observed_counts, expected_counts)),
                expected_counts,
            )
        )
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


##


def kbet(index, batch, alpha=0.05, only_score=True):
    """
    Computes the kBET metric to assess batch effects for an index matrix of a KNN graph.

    Parameters
    ----------
    index : numpy.ndarray
        An array of shape (n_cells, n_neighbors) containing the indices of the k nearest neighbors for each cell.
    batch : pandas.Series
        A categorical pandas Series of length n_cells indicating the batch for each cell.
    alpha : float, optional (default : 0.05)
        The significance level of the test.
    only_score : bool, optional (default : True)
        Whether to return only the accept rate or the full kBET results.

    Returns
    -------
    float or tuple of floats
        If only_score is True, returns the accept rate of the test as a float between 0 and 1.
        If only_score is False, returns a tuple of three floats: the mean test statistic, the mean p-value, and the
        accept rate.
    """
 
    # Compute null batch distribution
    batch = batch.astype('category')
    null_dist = batch.value_counts(normalize=True, sort=False).values 

    # Parallel computation of kBET metric (pegasus code)
    starting_idx = chunker(len(batch))
    n_jobs = cpu_count()

    with parallel_backend("loky", inner_max_num_threads=1):
        kBET_arr = np.concatenate(
            Parallel(n_jobs=n_jobs)(
                delayed(kbet_one_chunk)(
                    index[starting_idx[i] : starting_idx[i + 1], :], 
                    batch, 
                    null_dist
                )
                for i in range(n_jobs)
            )
        )
        
    # Gather results 
    stat_mean, pvalue_mean = kBET_arr.mean(axis=0)
    accept_rate = (kBET_arr[:,1] >= alpha).sum() / len(batch)

    if only_score:
        return accept_rate
    else:
        return (stat_mean, pvalue_mean, accept_rate)


##


def NN_entropy(index, labels):
    """
    Calculate the median (over cells) lentiviral-labels Shannon Entropy, given an index matrix of a KNN graph.
    """
    SH = []
    for i in range(index.shape[0]):
        freqs = labels[index[i, :]].value_counts(normalize=True).values
        SH.append(-np.sum(freqs * np.log(freqs + 0.00001))) # Avoid 0 division
    
    return np.median(SH)


##


def NN_purity(index, labels):
    """
    Calculate the median purity of cells neighborhoods.
    """
    kNN_purities = []
    n_cells = index.shape[0]
    k = index.shape[1]-1

    for i in range(n_cells):
        l = labels[i]
        idx = index[i, 1:]
        l_neighbors = labels[idx]
        kNN_purities.append(np.sum(l_neighbors == l) / k)
    
    return np.median(kNN_purities)


##


def binom_sum(x, k=2):
    return binom(x, k).sum()


##


def custom_ARI(g1, g2):
    """
    Compute scib modified ARI.
    """

    # Contingency table
    n = len(g1)
    contingency = pd.crosstab(g1, g2)

    # Calculate and rescale ARI
    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))
    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)


##