"""
Module for clustring affinity matrices derived from some cell-to-cell similarity metric defined over 
the MT variants space.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import leidenalg
from scipy.special import binom


##


def leiden_clustering(A, res=0.5):
    """
    Compute leiden clustering, at some resolution.
    """
    g = sc._utils.get_igraph_from_adjacency(A, directed=True)
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res,
        seed=1234
    )
    labels = np.array(part.membership)

    return labels


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


# rank_clone_variants(a, var='it', group=3, rank_by=rank_by)
def rank_clone_variants(
    a, var='GBC', group=None,
    filter_vars=True, rank_by='log2_perc_ratio', 
    min_clone_perc=.5, max_perc_rest=.2, min_perc_all=.1, log2_min_perc_ratio=.2
    ):
    """
    Rank a clone variants.
    """
    test = a.obs[var] == group
    AF_clone = np.nanmean(a.X[test,:], axis=0)
    AF_rest = np.nanmean(a.X[~test, :], axis=0)
    log2FC = np.log2(AF_clone+1)-np.log2(AF_rest+1)
    perc_all = np.sum(a.X>0, axis=0) / a.shape[0]
    perc_clone = np.sum(a.X[test,:]>0, axis=0) / a[test,:].shape[0]
    perc_rest = np.sum(a.X[~test,:]>0, axis=0) / a[~test,:].shape[0]
    perc_ratio = np.log2(perc_clone+1) - np.log2(perc_rest+1)
    df_vars = pd.DataFrame({
        'median_AF_clone' : AF_clone,
        'median_AF_rest' : AF_rest,
        'log2FC': log2FC, 
        'perc_clone': perc_clone, 
        'perc_rest': perc_rest, 
        'log2_perc_ratio': perc_ratio,
        'perc_all' : perc_all,
        },
        index=a.var_names
    )
    df_vars['n_cells_clone'] = np.sum(test)

    # Filter variants
    if filter_vars:
        if rank_by == 'log2_perc_ratio':
            test = f'log2_perc_ratio >= @log2_min_perc_ratio & perc_clone >= @min_clone_perc'
            df_vars = df_vars.query(test)
        elif rank_by == 'custom_perc_tresholds':
            test = f'perc_rest <= @max_perc_rest & perc_clone >= @min_clone_perc'
            df_vars = df_vars.query(test)
            df_vars.shape
    else:
        print('Returning all variants, ranked...')

    # Sort
    df_vars = df_vars.sort_values('log2_perc_ratio', ascending=False)

    return df_vars