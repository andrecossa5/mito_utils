"""
Module to create custom distance function among cell AF profiles.
"""

import numpy as np
import pandas as pd

from anndata import AnnData
from itertools import product
from scipy.spatial.distance import sqeuclidean, cosine, correlation
from scipy.spatial.distance import euclidean as euclidean_std
from sklearn.metrics import pairwise_distances, recall_score, precision_score, auc
from joblib import Parallel, delayed, parallel_backend, cpu_count

from .utils import rescale


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


def ludwig_one_pair(X, cov, i, j, ncov):
    
    x = X[i,:]
    y = X[j,:]
    cx = cov[i,:]
    cy = cov[j,:]

    test = (cx>ncov) & (cy>ncov)
    n = test.sum()
    x = x[test]
    y = y[test]
    d = np.sum(np.sqrt(np.abs(x-y))) / n
    
    return (i, j, d)


##


def ludwig_distances(X, cov, n_jobs=None, **kwargs):

    n = X.shape[0]
    try:
        ncov = kwargs['ncov'] 
    except:
        ncov = 100
    
    iterable = [ (i, j) for i, j in product(range(n), range(n)) if i<j ]
    ncores = cpu_count() if n_jobs is None else n_jobs
    
    with parallel_backend("loky"):
        results = Parallel(n_jobs=ncores)(
            delayed(ludwig_one_pair)(X, cov, i, j, ncov)
            for i,j in iterable
        )

    D = np.zeros((n,n))
    for i, j, value in results:
        D[i, j] = value
        D[j, i] = value
        
    return D


##


def prep_X_cov(a):
    
    # Find recurrent position
    var_pos = a.var_names.map(lambda x: x.split('_')[0])
    var_pos_counts = var_pos.value_counts()
    recurrent_pos = var_pos_counts.loc[lambda x: x>1].index
    
    # Choose for each recurrent position the variant seen most diffusively and 
    # at the highest mean AF
    var_to_exclude = []
    for x in recurrent_pos:
        multiple_at_pos = a.var_names[a.var_names.str.match(x)]
        top_seen_rank = np.sum(a[:, multiple_at_pos].X>0, axis=0).argsort()+1
        top_af_rank = np.nanmean(a[:, multiple_at_pos].X, axis=0).argsort()+1   
        rank = (top_seen_rank + top_af_rank) / 2
        retain = multiple_at_pos[rank.argmax()]
        for var in multiple_at_pos:
            if var != retain:
                var_to_exclude.append(var)
    
    # Define final variants and positions to retain
    final_vars = a.var_names[~a.var_names.isin(var_to_exclude)]
    final_pos = final_vars.map(lambda x: x.split('_')[0])
    
    # Prep AF and coverage arrays
    X = a[:, final_vars].X
    cov = a.uns['per_position_coverage'].loc[:, final_pos].values
    
    assert X.shape[1] == cov.shape[1]
    
    return X, cov

    
##


def pair_d(a, **kwargs):
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
    
    # Prep afm matrix (optionally coverage matrix also)
    if metric == 'ludwig2019':
        if not isinstance(a, AnnData):
            raise TypeError(f'a must be an AnnData to use metric {metric}')
        else:
            X, cov = prep_X_cov(a)
    else:
        X = a
    
    # Compute D
    if not nans:
        
        if metric in ['jaccard', 'matching']:
            X = np.where(X>0, 1, 0)
            D = pairwise_distances(X, metric=metric, n_jobs=ncores)
        elif metric == 'ludwig2019':
            D = ludwig_distances(X, cov, n_jobs=ncores, **kwargs)
        else:
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


def evaluate_metric_with_gt(a, metric, labels, **kwargs):
    
    print(f'Computing distances with metric {metric}...')
    
    final = {}
    D = pair_d(a, metric=metric)

    for alpha in np.linspace(0,1,10):
        
        print(f'Computing together/separate cell pairs: alpha {alpha:.2f}...')

        p_list = []
        gt_list = []

        for i in range(D.shape[0]):
            x = rescale(D[i,:])
            p_list.append(np.where(x<=alpha, 1, 0))
            c = labels.cat.codes.values[i]
            gt_list.append(np.where(labels.cat.codes.values==c, 1, 0))

        predicted = np.concatenate(p_list)
        gt = np.concatenate(gt_list)
        p = precision_score(gt, predicted)
        r = recall_score(gt, predicted)

        final[alpha] = (p, r)

    df = pd.DataFrame(final).T.reset_index(drop=True)
    df.columns = ['precision', 'recall']
    auc_score = auc(df['recall'], df['precision'])

    return auc_score


##



























