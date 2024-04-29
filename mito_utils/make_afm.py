"""
Module to make_afm: create a clean allelic frequency matrix.
"""

from itertools import product
import gc
import re
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from mquad.mquad import *
from scipy.sparse import csr_matrix


##


def mask_layer(A, base='C', strand='fw', layer='counts', min_qual=30):
    """
    Clean (base-specific, stranded) UMI counts.
    Set adata layer entries (i.e., cells x sites stranded allele consensus UMI counts) to 0,
    if the corresponding (average, across all consensus UMIs) base calling quality is <30.
    """
    logging.info(f'Filtering counts for {base} base and {strand} strand...')
    
    X = A.layers[f'{base}_{layer}_{strand}'].A
    mask = A.layers[f'{base}_qual_{strand}'].A < min_qual

    if layer == 'counts':
        i_counts = X.sum()
    i_entries = (X>0).sum()

    X[mask] = 0
    assert np.all((X==0) == mask)

    if layer == 'counts':
        o_counts = X.sum()
        logging.info(f'Retaining {o_counts/i_counts*100:.2f}% (consensus) UMI counts')    
    o_entries = (X>0).sum()

    logging.info(f'Retaining {o_entries/i_entries*100:.2f}% entries')

    return X, mask


##


def clean_BC_quality(A):
    """
    Clean A.layers cell x site x allele UMI counts, qualities and coverage.
    """
    cleaned_coverage = np.zeros(A.shape)
    combos = list(product(['A', 'C', 'T', 'G'], ['fw', 'rev']))
    for base, strand in combos:
        X, mask = mask_layer(A, base=base, strand=strand, layer='counts')
        qual = A.layers[f'{base}_qual_{strand}'].A
        qual[mask] = 0 
        A.layers[f'{base}_counts_{strand}'] = csr_matrix(X)         # Cleaned counts
        A.layers[f'{base}_qual_{strand}'] = csr_matrix(qual)        # Cleaned qualities
        cleaned_coverage += X
    A.layers['cov'] = csr_matrix(cleaned_coverage)                  # Cleaned cell x site coverage

    return A


##


def create_one_base_tables(A, base, only_variants=True):
    """
    For one of the 4 possible DNA bases creates:
    * df_x: the allelic frequency (AF) table for that base (i.e., a cell x site, with values in [0,1])
    * df_qual: the average base-calling quality table for that base (i.e., cell x site, No value should be below 30, if already cleaned UMI counts)
    * df_cov: UMI counts table for that base (i.e., cell x site, with integers)
    """

    # AF
    cov = A.layers[f'{base}_counts_fw'].A + A.layers[f'{base}_counts_rev'].A
    X = cov / (A.layers['cov'].A + 0.000001)

    # Calculate the average quality (across both strands) for each base-site combination
    q = A.layers[f'{base}_qual_fw'].A + A.layers[f'{base}_qual_rev'].A
    m = np.where(A.layers[f'{base}_qual_fw'].A>0, 1, 0) + np.where(A.layers[f'{base}_qual_rev'].A>0, 1, 0)
    qual = np.round(q / (m + 0.000001))

    # Re-format
    A.var['wt_allele'] = A.var['wt_allele'].str.capitalize()
    variant_names = A.var.index + '_' + A.var['wt_allele'] + f'>{base}'
    df_x = pd.DataFrame(X, index=A.obs_names, columns=variant_names)
    df_qual = pd.DataFrame(qual, index=A.obs_names, columns=variant_names)
    df_cov = pd.DataFrame(cov, index=A.obs_names, columns=variant_names)
    gc.collect()

    if only_variants:
        test = (A.var['wt_allele'] != base).values
        return df_cov.loc[:, test], df_x.loc[:, test], df_qual.loc[:, test]
    else:
        return df_cov, df_x, df_qual


##


def format_matrix(A, cbc_gbc_df=None, with_GBC=True, only_variants=True):
    """
    Create a full cell x variant AFM from the original maegatk output. 
    Add lentiviral clones' labels to resulting .obs if necessary.
    """

    # Add labels to .obs
    if with_GBC and cbc_gbc_df is not None:
        A.obs['GBC_Set'] = pd.Categorical(cbc_gbc_df['GBC_set'])

    # For each position and cell, compute each base AF and quality tables
    A_cov, A_x, A_qual = create_one_base_tables(A, 'A', only_variants=only_variants)
    C_cov, C_x, C_qual = create_one_base_tables(A, 'C', only_variants=only_variants)
    T_cov, T_x, T_qual = create_one_base_tables(A, 'T', only_variants=only_variants)
    G_cov, G_x, G_qual = create_one_base_tables(A, 'G', only_variants=only_variants)

    # Concat all of them in three complete coverage, AF and quality matrices, for each variant from the ref
    cov = pd.concat([A_cov, C_cov, T_cov, G_cov], axis=1)
    X = pd.concat([A_x, C_x, T_x, G_x], axis=1)
    qual = pd.concat([A_qual, C_qual, T_qual, G_qual], axis=1)

    # Reorder columns...
    variants = X.columns.map(lambda x: x.split('_')[0]).astype('int').values
    idx = np.argsort(variants)
    cov = cov.iloc[:, idx]
    X = X.iloc[:, idx]
    qual = qual.iloc[:, idx]

    # Create the per site quality matrix
    quality = np.zeros(A.shape)
    n_times = np.zeros(A.shape)
    for k in A.layers:
        if bool(re.search('qual', k)):
            quality += A.layers[k].A
            r, c = np.nonzero(A.layers[k].A)
            n_times[r, c] += 1
    quality = np.round(quality / (n_times + 0.0000001))

    # Create AnnData with variants and sites matrices
    afm = anndata.AnnData(X=X, obs=A.obs)
    afm.layers['coverage'] = cov
    afm.layers['quality'] = qual

    # Per site slots, in 'uns'. Each matrix is a ncells x nsites matrix
    afm.uns['per_position_coverage'] = pd.DataFrame(
        A.layers['cov'].A, index=afm.obs_names, columns=A.var_names
    )
    afm.uns['per_position_quality'] = pd.DataFrame(
        quality, index=afm.obs_names, columns=A.var_names
    )
    gc.collect()
    
    return afm


##