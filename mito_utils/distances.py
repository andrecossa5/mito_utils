"""
Module to create custom distance function among cell AF profiles.
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard
from sklearn.metrics import pairwise_distances, recall_score, precision_score, auc
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS
from mito_utils.filters import *
from mito_utils.preprocessing import *
from mito_utils.utils import rescale
from mito_utils.stats_utils import *
from mito_utils.kNN import *


##


def custom_MI_TO_jaccard(x,y):
    mask = (x!=-1) & (y!=-1)
    return jaccard(x[mask], y[mask])


##


discrete_metrics = PAIRWISE_BOOLEAN_FUNCTIONS 
continuous_metrics = list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) + ['correlation', 'sqeuclidean']
custom_discrete_metrics = {'custom_MI_TO_jaccard':custom_MI_TO_jaccard}


##


def genotype_mixtures(AD, DP, t_prob=.75, t_vanilla=.001, min_AD=2, debug=False):
    """
    Single-cell MT-SNVs genotyping with binomial mixtures posterior probabilities thresholding (Kwock et al., 2022).
    """
    X = np.zeros(AD.shape)
    for idx in range(AD.shape[1]):
        X[:,idx] = genotype_mix(AD[:,idx], DP[:,idx], t_prob=t_prob, t_vanilla=t_vanilla, min_AD=min_AD, debug=debug)
    return X


##


def genotype_MI_TO(AD, DP, t_prob=.75, t_vanilla=0, min_AD=2, min_cell_prevalence=.1, debug=False):
    """
    Hybrid genotype calling strategy: if a mutation has prevalence (AD>=min_AD and AF>=t_vanilla) >= min_cell_prevalence,
    use probabilistic modeling as in 'bin_mixtures'. Else, use simple tresholding as in 'vanilla' method.
    """
    X = np.zeros(AD.shape)
    n_binom = 0
    
    for idx in range(AD.shape[1]):
        test = (AD[:,idx]/(DP[:,idx]+.0000001)>t_vanilla)
        prevalence = test.sum() / test.size
        if prevalence >= min_cell_prevalence:
            X[:,idx] = genotype_mix(AD[:,idx], DP[:,idx], t_prob=t_prob, t_vanilla=t_vanilla, min_AD=min_AD, debug=debug)
            n_binom += 1
        else:
            X[:,idx] = np.where(test & (AD[:,idx]>=min_AD), 1, 0)

    logging.info(f'n MT-SNVs genotyped with binomial mixtures: {n_binom}')

    return X


##


def genotype_mixtures_smooth(AD, DP, t_prob=.75, k=10, gamma=.25, n_samples=100, min_AD=2):
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


def call_genotypes(afm, bin_method='vanilla', t_vanilla=.0, min_AD=2, t_prob=.75, min_cell_prevalence=.1, k=10, gamma=.25, n_samples=100):
    """
    Call genotypes using simple thresholding or th MI_TO binomial mixtures approachm (w/i or w/o kNN smoothing).
    """

    assert 'AD' in afm.layers and 'DP' in afm.layers

    X = afm.X.A.copy()
    AD = afm.layers['AD'].A.copy()
    DP = afm.layers['DP'].A.copy()
    
    if bin_method == 'vanilla':
        X = np.where((X>=t_vanilla) & (AD>=min_AD), 1, 0)
    elif bin_method == 'bin_mixtures':
        X = genotype_mixtures(AD, DP, t_prob=t_prob, min_AD=min_AD)
    elif bin_method == 'bin_mixtures_smooth':
        X = genotype_mixtures_smooth(AD, DP, t_prob=t_prob, k=k, gamma=gamma, 
                                     n_samples=n_samples, min_AD=min_AD)
    elif bin_method == 'MI_TO':
        X = genotype_MI_TO(AD, DP, t_prob=t_prob, t_vanilla=t_vanilla, 
                           min_AD=min_AD, min_cell_prevalence=min_cell_prevalence)
    else:
        raise ValueError("""
                Provide one of the following genotype calling methods: 
                vanilla, bin_mixtures, bin_mixtures_smooth, MI_TO
                """
            )

    afm.layers['bin'] = csr_matrix(X)
    afm.uns['genotyping'] = {
        'bin_method':bin_method, 
        'binarization_kwargs': {
            't_prob':t_prob, 't_vanilla':t_vanilla, 
            'min_AD':min_AD, 'min_cell_prevalence':min_cell_prevalence
        }
    }


##


def preprocess_feature_matrix(
    afm, distance_key='distances', metric='jaccard', bin_method=None, binarization_kwargs={}, precomputed=False
    ):
    """
    Preprocess a feature matrix for distancea computations.
    """
    
    layer = 'bin'
    logging.info('Updating afm.uns.genotyping')
    afm.uns['genotyping'].update({'bin_method':bin_method, 'binarization_kwargs':binarization_kwargs})
    afm.uns['distance_calculations'] = {}

    if metric in continuous_metrics:
        logging.info(f'Preprocess feature matrix: metric={metric}')
        layer = 'scaled'
        afm.layers['scaled'] = csr_matrix(pp.scale(afm.X.A))
        
    elif metric in discrete_metrics or metric in custom_discrete_metrics:
        logging.info(f'Preprocess feature matrix: metric={metric}, bin_method={bin_method}')
        if 'bin' in afm.layers and precomputed:
            pass
        else:
            call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)
        layer = 'bin'

    else:
        raise ValueError(f'Specify for a valid metric in {continuous_metrics}, {discrete_metrics} or {custom_discrete_metrics}')

    afm.uns['distance_calculations'][distance_key] = {'metric':metric}
    afm.uns['distance_calculations'][distance_key]['layer'] = layer


##


def _get_metric_and_layer(afm, distance_key):
    layer = afm.uns['distance_calculations'][distance_key]['layer']
    metric = afm.uns['distance_calculations'][distance_key]['metric']
    metric = metric if metric not in custom_discrete_metrics else custom_discrete_metrics[metric]
    return layer, metric


##


def compute_distances(
    afm, distance_key='distances', metric='jaccard', precomputed_bin=False,
    bin_method='vanilla', ncores=1,  metric_kwargs={}, binarization_kwargs={}
    ):
    """
    Calculates pairwise cell--cell (or sample-) distances in some character space (e.g., MT-SNVs mutation space).

    Args:
        afm (AnnData): An annotated cell x character matrix with .X slot and bin or scaled layers.
        distance_key (str, optional): Key in .obsp at which the new distances will be stored. Default: distances.
        metric ((str, callable), optional): distance metric. Default: 'jaccard'.
        bin_method (str, optional): method to binarize the provided character matrix, if the chosen metric 
            involves comparison of discrete character vectors. Default: vanilla.
        ncores (int, optional): n processors for parallel computation. Default: 8.
        metric_kwargs (dict, optional): **kwargs of the metric function. Default: {}.
        binarization_kwargs (dict, optional): **kwargs of the discretization function. Default: {}.

    Returns:
        Updates inplace .obsp slot in afm AnnData with key 'distances'.
    """
    
    preprocess_feature_matrix(
        afm, distance_key=distance_key, metric=metric, precomputed=precomputed_bin,
        bin_method=bin_method, binarization_kwargs=binarization_kwargs
    )

    layer, metric = _get_metric_and_layer(afm, distance_key)
    X = afm.layers[layer].A.copy()

    logging.info(f'Compute distances: ncores={ncores}')
    D = pairwise_distances(X, metric=metric, n_jobs=ncores, force_all_finite=False, **metric_kwargs)
    afm.obsp[distance_key] = csr_matrix(D)
    
    afm.uns['distance_calculations'][distance_key].update(metric_kwargs)
    

##


def distance_AUPRC(D, labels):
    """
    Uses a n x n distance matrix D as a binary classifier for a set of labels  (1,...,n). 
    Reports Area Under Precision Recall Curve.
    """

    labels = pd.Categorical(labels) 

    final = {}
    for alpha in np.linspace(0,1,10):
 
        p_list = []
        gt_list = []

        for i in range(D.shape[0]):
            x = rescale(D[i,:])
            p_list.append(np.where(x<=alpha, 1, 0))
            c = labels.codes[i]
            gt_list.append(np.where(labels.codes==c, 1, 0))

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



























