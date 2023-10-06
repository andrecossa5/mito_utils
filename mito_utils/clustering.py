"""
Module for clustring affinity matrices derived from some cell-to-cell similarity metric defined over 
the MT variants space.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import leidenalg
from scipy.special import binom
from scipy.stats import fisher_exact
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.sandbox.stats.multicomp import multipletests


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


##


def compute_clonal_fate_bias(df, state_column, clone_column, target_state):
    """
    Compute -log10(FDR) Fisher's exact test: clonal fate biases towards some target_state.
    """

    n = df.shape[0]
    clones = np.sort(df[clone_column].unique())

    target_ratio_array = np.zeros(clones.size)
    oddsratio_array = np.zeros(clones.size)
    pvals = np.zeros(clones.size)

    # Here we go
    for i, clone in enumerate(clones):

        test_clone = df[clone_column] == clone
        test_state = df[state_column] == target_state

        clone_size = test_clone.sum()
        clone_state_size = (test_clone & test_state).sum()
        target_ratio = clone_state_size / clone_size
        target_ratio_array[i] = target_ratio
        other_clones_state_size = (~test_clone & test_state).sum()

        # Fisher
        oddsratio, pvalue = fisher_exact(
            [
                [clone_state_size, clone_size - clone_state_size],
                [other_clones_state_size, n - other_clones_state_size],
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    # Correct pvals --> FDR
    pvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]

    # Results
    results = pd.DataFrame({
        'perc_in_target_state' : target_ratio_array,
        'odds_ratio' : oddsratio_array,
        'FDR' : pvals,
        'fate_bias' : -np.log10(pvals) 
    }).sort_values('fate_bias', ascending=False)

    return results


##


def fast_hclust_distance(D):
    
    D[np.isnan(D)] = 0                               
    D[np.diag_indices(D.shape[0])] = 0
    linkage_matrix = linkage(squareform(D), method='weighted')
    order = leaves_list(linkage_matrix)

    return order