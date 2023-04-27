"""
Module to create custom distance function among cell AF profiles.
"""

import numpy as np
from scipy.spatial.distance import sqeuclidean, cosine, correlation
from scipy.spatial.distance import euclidean as euclidean_std
from sklearn.metrics import pairwise_distances


##

def euclidean_nans(x, y):
    ix = np.where(~np.isnan(x))[0]
    iy = np.where(~np.isnan(y))[0]
    idx = list(set(ix) & set(iy))
    x_ = x[idx]
    y_ = y[idx]
    return euclidean_std(x_, y_)


##


def sqeuclidean_nans(x, y):
    ix = np.where(~np.isnan(x))[0]
    iy = np.where(~np.isnan(y))[0]
    idx = list(set(ix) & set(iy))
    x_ = x[idx]
    y_ = y[idx]
    return sqeuclidean(x_, y_)


##


def correlation_nans(x, y):
    ix = np.where(~np.isnan(x))[0]
    iy = np.where(~np.isnan(y))[0]
    idx = list(set(ix) & set(iy))
    x_ = x[idx]
    y_ = y[idx]
    return correlation(x_, y_)


## 


def cosine_nans(x, y):
    ix = np.where(~np.isnan(x))[0]
    iy = np.where(~np.isnan(y))[0]
    idx = list(set(ix) & set(iy))
    x_ = x[idx]
    y_ = y[idx]
    return cosine(x_, y_)


##


def pair_d(X, **kwargs):
    """
    Function for calculating pairwise distances within the row vectors of a matrix X.
    """
    # Get kwargs
    try:
        metric = kwargs['metric']
    except:
        metric = 'euclidean'
    try:
        ncores = kwargs['ncores']
    except:
        ncores = 8
    try:
        nans = kwargs['nans']
    except:
        nans = False

    print(f'pair_d arguments: metric={metric}, ncores={ncores}, nans={nans}')

    # Compute D
    if not nans:
        D = pairwise_distances(X, metric=metric, n_jobs=ncores)
    else:
        print(f'Custom, nan-robust {metric} metric will be used here...')
        if metric == 'euclidean':
            D = pairwise_distances(X, metric=euclidean_nans, n_jobs=ncores, force_all_finite=False)
        elif metric == 'sqeuclidean':
            D = pairwise_distances(X, metric=sqeuclidean_nans, n_jobs=ncores, force_all_finite=False)
        elif metric == 'correlation':
            D = pairwise_distances(X, metric=correlation_nans, n_jobs=ncores, force_all_finite=False)
        elif metric == 'cosine':
            D = pairwise_distances(X, metric=cosine_nans, n_jobs=ncores, force_all_finite=False)

    return D


##


























