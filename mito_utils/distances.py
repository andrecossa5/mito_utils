"""
Module to create custom distance function among cell AF profiles.
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from scipy.sparse import issparse, csc_matrix, coo_matrix
from scipy.spatial.distance import jaccard
from sklearn.metrics import pairwise_distances, recall_score, precision_score, auc
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS
from .filters import *
from .preprocessing import *
from .utils import rescale
from .stats_utils import *
from .kNN import *


##


def _custom_MI_TO_jaccard(x,y):
    mask = (x!=-1) & (y!=-1)
    return jaccard(x[mask], y[mask])


##


def _get_X(a, X, AD, DP, scale=True):
    """
    Get continuous character matrix according to given arguments.
    """
    if a is not None:
        X = a.X if not issparse(a.X) else a.X.toarray()
    elif X is not None:
        pass
    elif AD is not None and DP is not None:
        X = AD / (DP+.0000001)
    else:
        raise TypeError('Provide one between: a, X or AD and DP arguments')
    if scale:
        X = pp.scale(X)
    return X


##


def get_AD_DP(afm, to='coo'):
    """
    From a given AFM matrix, find the AD and DP parallel matrices.
    """
    # Get DP counts
    DP = afm.uns['per_position_coverage'].T.values
    if to == 'coo':
        DP = coo_matrix(DP)
    else:
        DP = csc_matrix(DP)
        
    # Get AD counts
    sites = afm.uns['per_position_coverage'].columns
    variants = afm.var_names
    
    # Check consistency sites/variants remaining after previous filters
    test_1 = variants.map(lambda x: x.split('_')[0]).unique().isin(sites).all()
    test_2 = sites.isin(variants.map(lambda x: x.split('_')[0]).unique()).all()
    assert test_1 and test_2
    
    # Get alternative allele variants
    ad_vars = []
    only_sites_names = variants.map(lambda x: x.split('_')[0])
    for x in sites:
        test = only_sites_names == x
        site_vars = variants[test]
        if site_vars.size == 1:
            ad_vars.append(site_vars[0])
        else:
            cum_sum = afm[:, site_vars].layers['coverage'].sum(axis=0)
            idx_ad = np.argmax(cum_sum)
            ad_vars.append(site_vars[idx_ad])
            
    # Get AD counts
    AD = afm[:, ad_vars].layers['coverage'].T
    if to == 'coo':
        AD = coo_matrix(AD)
    else:
        AD = csc_matrix(AD)
    
    return AD, DP, ad_vars


##


def _get_AD_DP(a, X, AD, DP):
    """
    Private method to handle multiple inputs and retrieve (if possible), AD, DP matrices.
    """
    if a is not None:
        AD, DP, _ = get_AD_DP(a)
        AD = AD.A.T
        DP = DP.A.T
    elif AD is not None and DP is not None:
        pass
    else:
        raise TypeError('Provide one between: 1) a, and 2) AD and DP.')
    return AD, DP


##


def genotype_MI_TO(AD, DP, t_prob=.75, t_vanilla=.05, min_AD=2, debug=False):
    """
    Single-cell MT-SNVs genotyping with binomial mixtures posterior probabilities thresholding (Kwock et al., 2022) 
    """
    X = np.zeros(AD.shape)
    for idx in range(AD.shape[1]):
        X[:,idx] = genotype_mix(AD[:,idx], DP[:,idx], t_prob=t_prob, t_vanilla=t_vanilla, min_AD=min_AD, debug=debug)
    return X


##


def genotype_MI_TO_smooth(AD, DP, t_prob=.75, k=10, gamma=.25, n_samples=100, min_AD=2):
    """
    Single-cell MT-SNVs genotyping with binomial mixtures posterior probabilities thresholding (readapted from  MQuad, Kwock et al., 2022)
    and kNN smoothing (readapted from Phylinsic, Liu et al., 2022).
    """

    # kNN 
    L = []
    for _ in range(1,n_samples+1):

        print(f'Resampling AD counts for kNN calculations: sample {_}/{n_samples}')
        AD_sample = np.zeros(AD.shape)
        for idx in range(AD.shape[1]):
            model = MixtureBinomial(n_components=2, tor=1e-20)
            params = model.fit((AD[:,idx], DP[:,idx]), max_iters=500, early_stop=True)
            AD_sample[:,idx] = model.sample(DP[:,idx])

        D = compute_distances(AD=AD_sample, DP=DP, metric='custom_MI_TO_jaccard', 
                              bin_method='MI_TO', binarization_kwargs={'t_prob':t_prob, 'min_AD':min_AD}, verbose=False)
        L.append(D)
    D = np.mean(np.stack(L, axis=0), axis=0)
    index, _, _ = kNN_graph(D=D, k=k, from_distances=True)

    # Compute osteriors
    P0 = np.zeros(AD.shape)
    P1 = np.zeros(AD.shape)
    for idx in range(P0.shape[1]):
        p = get_posteriors(AD[:,idx], DP[:,idx])
        P0[:,idx] = p[:,0]
        P1[:,idx] = p[:,1]

    # Smooth posteriors
    P0_smooth = np.zeros(P0.shape)
    P1_smooth = np.zeros(P1.shape)
    for i in range(index.shape[0]):
        neighbors = index[i,1:]
        P0_smooth[i,:] = (1-gamma) * P0[i,:] + gamma * (P0[neighbors,:].mean(axis=0))
        P1_smooth[i,:] = (1-gamma) * P1[i,:] + gamma * (P1[neighbors,:].mean(axis=0))

    # Assign final genotypes
    tests = [ 
        (P1_smooth>t_prob) & (P0_smooth<(1-t_prob)) & (AD>=min_AD), 
        (P1_smooth<(1-t_prob)) & (P0_smooth>t_prob) 
    ]
    X = np.select(tests, [1,0], default=-1)

    return X


##


def call_genotypes(a=None, X=None, AD=None, DP=None, bin_method='vanilla', t_prob=.75, t_vanilla=.01,
                   k=10, gamma=.25, n_samples=100, min_AD=1):
    """
    Call genotypes using simple thresholding or th MI_TO binomial mixtures approachm (w/i or w/o kNN smoothing).
    """
    AD, DP = _get_AD_DP(a, X, AD, DP)

    if bin_method == 'vanilla':
        X = np.where((AD/(DP+.0000001)>t_vanilla) & (AD>=min_AD),1,0)
    elif bin_method == 'MI_TO':
        X = genotype_MI_TO(AD, DP, t_prob=t_prob, min_AD=min_AD)
    elif bin_method == 'MI_TO_smooth':
        X = genotype_MI_TO_smooth(AD, DP, t_prob=t_prob, k=k, gamma=gamma, n_samples=n_samples, min_AD=min_AD)

    return X


##


def preprocess_feature_matrix(X=None, a=None, AD=None, DP=None, metric='euclidean', binary=True,
                              bin_method='vanilla', weights=None, scale=False, binarization_kwargs={}, verbose=False):
    """
    Preprocess a feature matrix for some distance computations.
    """
    
    discrete_metrics = PAIRWISE_BOOLEAN_FUNCTIONS + ['custom_MI_TO_jaccard']
    continuous_metrics = list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) + ['correlation', 'sqeuclidean']

    if metric in continuous_metrics or not binary:
        binary = False
        bin_method = None
    elif metric in discrete_metrics or binary:
        binary = True
    
    if verbose:
        print(f'Preprocess feature matrix. Metric: {metric}, bin: {binary}, bin_method: {bin_method}')

    # Get raw AF matrix
    X_raw = _get_X(a, X, AD, DP, scale=False)
    
    # Get rescaled or binarized AF matrix
    if binary:
        X = call_genotypes(a=a, X=X, AD=AD, DP=DP, bin_method=bin_method, **binarization_kwargs)
        metric = _custom_MI_TO_jaccard if metric == 'custom_MI_TO_jaccard' else metric
    elif not binary and scale:
        X = pp.scale(X)

    # Weight different features if necessary
    if weights is not None:
        weights = np.array(weights)
        X = X * weights if not binary else np.round(X * weights)

    return metric, X_raw, X


##


def compute_distances(
    X=None, a=None, AD=None, DP=None, 
    metric='euclidean', binary=True, bin_method=None,
    weights=None, scale=True, ncores=8, 
    metric_kwargs={}, binarization_kwargs={}, verbose=False, return_matrices=False
    ):
    """
    Calculates pairwise cell- (or sample-) by-cells distances in some character space (e.g., MT-SNVs mutation space).

    Args:
        X (np.array, optional): A cell x character matrix. Default: None.
        a (AnnData, optional): An annotated cell x character matrix. Defaults: None.
        AD (np.array, optional): A cell x site alternative allele reads (or UMIs) count matrix. Default: None.
        DP (np.array, optional): A cell x site total reads (or UMIs) count matrix. Default: None.
        metric ((str, callable), optional): distance metric. Default: 'euclidean'.
        binary (bool, True): specifies wheater the user-provided metric requires discretization of input character matrices or not.
            To use only in case metric is a callable. Default: True.
        bin_method (str, optional): method to binarise the provided character matrix, if the chosen metric involves comparison 
            of discrete character vectors. Default: None.
        weights (np.array, optional): A vector of character weights to differentially weight each feature at distance computation. Default: None.
        scale (bool, optional): scales each character matrix column (i.e., characters) if X, or a are provided as character matrices 
            and metric does not involve binarization. Default: True.
        ncores (int, optional): n processors for parallel computation. Default: 8.
        metric_kwargs (dict, optional): **kwargs of the metric function. Default: {}.
        binarization_kwargs (dict, optional): **kwargs of the discretization function. Default: {}.

    Returns:
        np.array: n x n pairwise distance matrix.
    """

    if verbose:
        print(f'compute_distances with: metric={metric}, bin_method={bin_method}, ncores={ncores}.')
    
    metric, X_raw, X = preprocess_feature_matrix(
        X=X, a=a, AD=AD, DP=DP, metric=metric, binary=binary, bin_method=bin_method, 
        weights=weights, scale=scale, binarization_kwargs=binarization_kwargs
    )
    D = pairwise_distances(X, metric=metric, n_jobs=ncores, force_all_finite=False, **metric_kwargs)

    if return_matrices:
        return X_raw, X, D
    else:
        return D
    

##


def distance_AUPRC(D, labels):
    """
    Uses a n x n distance matrix D as a binary classifier for a set of labels  (1,...,n). 
    Reports Area Under Precision Recall Curve.
    """

    final = {}
    for alpha in np.linspace(0,1,10):
 
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



























