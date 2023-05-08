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


def rank_clone_variants(a, clone, by='perc_ratio', min_clone_perc=0.5, max_perc_rest=0.1):
    """
    Rank a clone variants.
    """
    test = a.obs['GBC'] == clone
    log2FC = np.log2((np.nanmean(a.X[test, :], axis=0) + 0.000001) / np.nanmean(a.X[~test, :], axis=0))
    perc_clone = np.sum(a.X[test,:]>0, axis=0) / a[test,:].shape[0] + 0.000001
    perc_rest = np.sum(a.X[~test,:]>0, axis=0) / a[~test,:].shape[0]
    perc_ratio = perc_clone / perc_rest
    
    df_vars = pd.DataFrame({
        'log2FC':log2FC, 
        'perc_clone':perc_clone, 
        'perc_rest':perc_rest, 
        'perc_ratio':perc_ratio,
        },
        index=a.var_names
    )
    df_vars = df_vars.query(f'perc_rest < @max_perc_rest and perc_clone > @min_clone_perc')
    
    return df_vars.sort_values(by)