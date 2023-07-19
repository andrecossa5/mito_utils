"""
Utilities for the iterative scheme of clonal selection.
"""

import os
import pickle
import numpy as np
import pandas as pd
from kneed import KneeLocator
from vireoSNP import BinomMixtureVB

from .utils import *
from .preprocessing import *
from .clustering import *
from .distances import *
from .it_diagnostics import *


##


def subset_one(afm, partition):
    """
    Given some afm, the labels dictionary produced across iterations, 
    an iteration number and one of its partitions, subset the afm to its cells.
    """
    cells = afm.obs.query('it == @partition').index
    a = afm[cells,:].copy()
    a.uns['per_position_coverage'] = a.uns['per_position_coverage'].loc[cells]
    a.uns['per_position_quality'] = a.uns['per_position_quality'].loc[cells]

    return a


##


def subset_afm(afm):

    afm = afm[afm.obs['it'] != 'poor quality', :]
    L = [ subset_one(afm, p) for p in afm.obs['it'].unique() ]

    return L


##


def vireo_wrapper(afm, min_n_clones=2, max_n_clones=None, 
                n_max_mut=True, p_treshold=.85, random_seed=1234):
    """
    Given an AFM (cells x MT-variants), this function uses the vireoSNP method to return a set of 
    putatite MT-clones labels.
    """
    afm = nans_as_zeros(afm)

    # Get AD, DP
    AD, DP, _ = get_AD_DP(afm, to='csc')
    # Find the max_n_clones to test
    if max_n_clones is None:
        if n_max_mut:
            max_n_clones = afm.shape[1]
        else:
            max_n_clones = 15 if afm.shape[1] > 15 else afm.shape[1]
    # Here we go
    range_clones = range(min_n_clones, max_n_clones+1)
    _ELBO_mat = []
    for k in range_clones:
        print(f'Clone n: {k}')
        _model = BinomMixtureVB(n_var=AD.shape[0], n_cell=AD.shape[1], n_donor=k)
        _model.fit(AD, DP, min_iter=30, max_iter=500, max_iter_pre=250, n_init=50, random_seed=random_seed)
        _ELBO_mat.append(_model.ELBO_inits)

    x = range_clones
    y = np.median(_ELBO_mat, axis=1)
    knee = KneeLocator(x, y).find_knee()[0]
    n_clones = knee

    # Refit with optimal n_clones
    _model = BinomMixtureVB(n_var=AD.shape[0], n_cell=AD.shape[1], n_donor=n_clones)
    _model.fit(AD, DP, min_iter=30, n_init=50, max_iter=500, max_iter_pre=250, random_seed=random_seed)
    
    # Clonal assignment probabilites --> to crisp labels
    clonal_assignment = _model.ID_prob
    df_ass = pd.DataFrame(
        clonal_assignment, 
        index=afm.obs_names, 
        columns=range(clonal_assignment.shape[1])
    )

    # Define labels
    labels = []
    for i in range(df_ass.shape[0]):
        cell_ass = df_ass.iloc[i, :]
        try:
            labels.append(np.where(cell_ass>p_treshold)[0][0])
        except:
            labels.append('unassigned')

    return pd.Series(labels, index=afm.obs_names)


##


def one_it_iterative_splitting(
        afm, mode='iterative_splitting', rank_by='log2_perc_ratio', **kwargs
    ):
    """
    One iteration of variants selection, vireo clustering and testing exclusive variants.
    """
    try:
        a_cells, a = filter_cells_and_vars(
            afm, filtering='MQuad', min_cov_treshold=50, nproc=1, path_=os.getcwd(),
            filter_cells=False if 'it' in afm.obs else True
        )
    except:
        print(f'No elbow found!')
        a_cells = None
        a = None
    
    if a_cells is not None:

        try:
            labels = vireo_wrapper(a) 
            if 'it' in a_cells.obs.columns:
                labels = pd.Series(
                    [ '.'.join([str(x), str(y)]) for x,y in zip(a_cells.obs['it'], labels) ],
                    index=a_cells.obs_names
                )
            else:
                labels = pd.Series(labels, index=a_cells.obs_names)
            a.obs['it'] = labels
            a_cells.obs['it'] = labels

            if mode == 'iterative_splitting':
                exclusive_variants = { 
                    c : rank_clone_variants(
                        a, var='it', group=c, 
                        rank_by=rank_by,
                        #**kwargs
                    ) \
                    for c in a.obs['it'].unique()
                }
                assigned, unassigned = test_partitions_variants(exclusive_variants)  
                a_cells.obs.loc[a_cells.obs['it'].isin(unassigned), 'it'] = 'unassigned'
                test = len(assigned)>0

            elif mode == 'distances':
                pass
                # test = test_partitions_distances(a)  

            variants = a.var_names
            
        except:
            print('Problem with this partition...')
            labels = [ 'unassigned' for x in range(a.obs.shape[0])]
            labels = pd.Series(labels, index=a_cells.obs_names)
            test = False
            variants = a.var_names
    
    else:
        labels = None
        test = None, 
        variants = None
        
    return labels, test, variants, a_cells


##


