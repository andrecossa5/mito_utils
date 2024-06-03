"""
Module to create custom distance function among cell AF profiles.
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from itertools import product
from sklearn.metrics import pairwise_distances, recall_score, precision_score, auc
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS
from joblib import Parallel, delayed, parallel_backend, cpu_count
from anndata import AnnData
from .utils import rescale


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



def pair_d(data, metric='jaccard', t=.01, weights=None, scale=False, ncores=8, metric_kwargs={}):
    """
    Function for calculating (possibly weighted) pairwise distances among cells 
    """
    print(f'pair_d arguments: metric={metric}, t={t:.2f}, ncores={ncores}.')
    
    if metric == 'ludwig2019':
        if isinstance(data, AnnData):
            X, cov = prep_X_cov(a)
            D = ludwig_distances(X, cov, n_jobs=ncores, **metric_kwargs)
        else:
            raise TypeError('data must be of type AnnData to use metric ludwig2019.')
    else:
        if isinstance(data, AnnData):
            X = data.X
        else:
            X = data
        if metric in PAIRWISE_BOOLEAN_FUNCTIONS:
            X = np.where(X>=t,1,0)
        if weights is not None:
            weights = np.array(weights)
            X = X * weights
        if scale:
            X = pp.scale(X)
        D = pairwise_distances(X, metric=metric, n_jobs=ncores, force_all_finite=False)

    return D


##


def evaluate_metric_with_gt(a, metric, labels, **kwargs):
    
    print(f'Computing distances with metric {metric}...')
    
    final = {}
    D = pair_d(a, metric=metric, **kwargs)

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



























