"""
Module to make_afm: create a clean allelic frequency matrix.
"""

from itertools import product
import gc
import re
import numpy as np
import pandas as pd
import anndata
from mquad.mquad import *


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
    ref_col = 'wt_allele' if 'wt_allele' in A.var.columns else 'ref'
    assert ref_col in A.var.columns
    A.var[ref_col] = A.var[ref_col].str.capitalize()
    variant_names = A.var.index + '_' + A.var[ref_col] + f'>{base}'
    df_x = pd.DataFrame(X, index=A.obs_names, columns=variant_names)
    df_qual = pd.DataFrame(qual, index=A.obs_names, columns=variant_names)
    df_cov = pd.DataFrame(cov, index=A.obs_names, columns=variant_names)
    gc.collect()

    if only_variants:
        test = (A.var[ref_col] != base).values
        return df_cov.loc[:, test], df_x.loc[:, test], df_qual.loc[:, test]
    else:
        return df_cov, df_x, df_qual


##


def format_matrix(A, only_variants=True):
    """
    Create a full cell x variant AFM from the original AnnData storing all dataset tables.
    """
    
    # Clones to categorical, if present
    if 'GBC' in A.obs.columns:
        A.obs['GBC'] = pd.Categorical(A.obs['GBC'])

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
    afm = anndata.AnnData(X=X, obs=A.obs, dtype=np.float32)
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



