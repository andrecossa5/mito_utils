"""
Module to preprocess AFMs: reformat original AFM; filter variants/cells.
"""

import sys
import gc
import re
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from mquad.mquad import *
from scipy.sparse import coo_matrix, csc_matrix
from pegasus.tools.hvf_selection import fit_loess


##


filtering_options = [
    'CV',
    'ludwig2019', 
    'velten2021', 
    'miller2022', 
    'seurat', 
    'pegasus', 
    'MQuad', 
    #'DADApy',
    'density'
]


##


def create_one_base_tables(A, base, only_variants=True):
    """
    Create a full cell x variant AFM from the original maegatk output, and one of the 4 bases,
    create the allelic frequency df for that base, a cell x {i}_{ref}>{base} table.
    """
    # base = 'A'
    cov = A.layers[f'{base}_counts_fw'].A + A.layers[f'{base}_counts_rev'].A
    X = cov / A.layers['coverage'].A
    q = A.layers[f'{base}_qual_fw'].A + A.layers[f'{base}_qual_rev'].A
    m = np.where(A.layers[f'{base}_qual_fw'].A > 0, 1, 0) + np.where(A.layers[f'{base}_qual_rev'].A > 0, 1, 0)
    qual = q / m

    A.var['wt_allele'] = A.var['wt_allele'].str.capitalize()
    variant_names = A.var.index + '_' + A.var['wt_allele'] + f'>{base}'
    df_cov = pd.DataFrame(cov, index=A.obs_names, columns=variant_names)
    df_x = pd.DataFrame(X, index=A.obs_names, columns=variant_names)
    df_qual = pd.DataFrame(qual, index=A.obs_names, columns=variant_names)
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
    Add lentiviral clones' labels to resulting .obs.
    """

    # Add labels to .obs
    if with_GBC and cbc_gbc_df is not None:
        A.obs['GBC'] = pd.Categorical(cbc_gbc_df['GBC'])

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

    # Create the per position quality matrix
    quality = np.zeros(A.shape)
    n_times = np.zeros(A.shape)
    for k in A.layers:
        if bool(re.search('qual', k)):
            quality += A.layers[k].A
            r, c = np.nonzero(A.layers[k].A)
            n_times[r, c] += 1
    quality = quality / n_times

    # Create AnnData with variants and sites matrices
    afm = anndata.AnnData(X=X, obs=A.obs)
    afm.layers['coverage'] = cov
    afm.layers['quality'] = qual

    # Per site slots, in 'uns'. Each matrix is a ncells x nsites matrix
    afm.uns['per_position_coverage'] = pd.DataFrame(
        A.layers['coverage'].A, index=afm.obs_names, columns=A.var_names
    )
    afm.uns['per_position_quality'] = pd.DataFrame(
        quality, index=afm.obs_names, columns=A.var_names
    )
    gc.collect()
    
    return afm


##


def read_one_sample(path_data, sample=None, only_variants=True, with_GBC=False):
    """
    Read and format one sample AFM.
    """
    A = sc.read(os.path.join(path_data, sample, 'AFM.h5ad'))
    barcodes = pd.read_csv(os.path.join(path_data, sample, 'barcodes.txt'), index_col=0)

    # Filter cell barcodes
    valid_cbcs = set(A.obs_names) & set(barcodes.index)
    # GBC info
    if with_GBC:
        cbc_gbc_df = pd.read_csv(
            os.path.join(path_data, sample, 'cells_summary_table.csv'), 
            index_col=0
        )
        valid_cbcs = valid_cbcs & set(cbc_gbc_df.index)
    
    # Subset
    cells = list(valid_cbcs)
    A = A[cells, :].copy()
    if with_GBC:
        cbc_gbc_df = cbc_gbc_df.loc[cells, :]
    
    # Format
    A.layers['coverage'] = A.layers['cov']
    afm = format_matrix(
        A, 
        cbc_gbc_df=cbc_gbc_df if with_GBC else None, 
        only_variants=only_variants,
        with_GBC=with_GBC
    )
    afm.obs = afm.obs.assign(sample=sample)

    return afm


##


def read_all_samples(path_data, sample_list=None):
    """
    Read and format all samples AFMs. 
    """
    ORIG = {}
    for sample in sample_list:
        orig = sc.read(os.path.join(path_data, sample, 'AFM.h5ad'))
        orig.obs = orig.obs.assign(sample=sample)
        ORIG[sample] = orig
        meta_vars = orig.var
    orig = anndata.concat(ORIG.values(), axis=0)
    orig.var = meta_vars
    del ORIG
    afm = format_matrix(orig, with_clones=False)

    return afm


##


def create_blacklist_table(path_main):
    """
    Creates a summary stat table to exclude further variants, common to more samples.
    """
    afm = read_all_samples(path_main)
    DFs = []
    for x in afm.obs['sample'].unique():
        cells = afm.obs.query('sample == @x').index
        x_sample = afm[cells, :].X.copy()
        cov_sample = afm[cells, :].layers['coverage'].copy()
        n_positives = np.sum(x_sample > 0, axis=0)
        perc_positives = n_positives / len(cells)
        n_cells_more_than_5_umis = np.sum(cov_sample > 5, axis=0)
        df_sample = pd.DataFrame({
            'n_pos' : n_positives,
            'perc_pos' : perc_positives,
            'n_cells_more_than_5_umis' : n_cells_more_than_5_umis,
            'sample' : [x] * afm.shape[1]
        })
        df_sample.index = afm.var_names
        DFs.append(df_sample)

    df = pd.concat(DFs)

    return df


##


def nans_as_zeros(afm):
    """
    Fill nans with zeros. Technical holes are considered as biologically meaningul zeroes.
    """
    X_copy = afm.X.copy()
    X_copy[np.isnan(X_copy)] = 0
    afm.X = X_copy

    return afm


##


def filter_cells_coverage(afm, mean_coverage=50):
    """
    Simple filter to subset an AFM only for cells with at least <median_coverage> median site coverage. 
    """
    test_cells = np.nanmean(afm.uns['per_position_coverage'].values, axis=1) > mean_coverage
    filtered = afm[test_cells, :].copy()
    filtered.uns['per_position_coverage'] = filtered.uns['per_position_coverage'].loc[test_cells, :]
    filtered.uns['per_position_quality'] = filtered.uns['per_position_quality'].loc[test_cells, :]

    return filtered


##


def remove_excluded_sites(afm):
    """
    Remove sites belonging to non-selected AFM variants.
    """
    cells = afm.obs_names
    sites_retained = afm.var_names.map(lambda x: x.split('_')[0]).unique()
    afm.uns['per_position_coverage'] = afm.uns['per_position_coverage'].loc[cells, sites_retained]
    afm.uns['per_position_quality'] = afm.uns['per_position_quality'].loc[cells, sites_retained]

    return afm


##


def remove_from_blacklist(variants, df, sample='MDA_clones'):
    """
    Remove variants share by other samples.
    """
    df_sample = df.loc[variants,:].query('sample == @sample')
    test = df_sample['perc_pos'].values > 0.01
    other_samples = [ x for x in df['sample'].unique() if x != sample]
    for x in other_samples:
        df_ = df.loc[variants,:].query('sample == @x')
        test &= ~(df_['n_pos'].values > 10)

    return variants[test]


##


def filter_baseline(afm):
    """
    Baseline filter, applied on all variants, before any method-specific solution.
    This is a very mild filter to exclude all variants that will be pretty much impossible
    to use by any method due to:
    * extremely low coverage at which the variant site have been observed across the population
    * extremely low quality at which the variant site have been observed across the population
    * Too less cells in which the variant have been detected with AF >1% 
    """
    # Test 1: variants whose site has been covered by nUMIs >= 10 (mean, over all cells)
    test_sites = pd.Series(
        np.mean(afm.uns['per_position_coverage'], axis=0) > 10,
        index=afm.uns['per_position_coverage'].columns
    )
    sites = test_sites[test_sites].index
    test_vars_site_coverage = (
        afm.var_names
        .map(lambda x: x.split('_')[0] in sites)
        .to_numpy(dtype=bool)
    )
    # Test 2-4: 
    # variants with quality > 20 (mean, over all cells); 
    # variants seen in at least 3 cells;
    # variants with AF>0.01 in at least 3 cells;
    test_vars_quality = np.nanmean(afm.layers['quality'], axis=0) > 20
    test_vars_coverage = np.sum(afm.layers['coverage'] > 0, axis=0) > 3
    test_vars_AF = np.sum(afm.X > 0.01, axis=0) > 3

    # Filter vars and sites
    test_vars = test_vars_site_coverage & test_vars_quality & test_vars_coverage & test_vars_AF 
    filtered = afm[:, test_vars].copy()
    filtered = remove_excluded_sites(filtered)

    return filtered


##


def filter_CV(afm, n=1000):
    """
    Filter variants based on their coefficient of variation.
    """
    # Create test
    CV = np.nanmean(afm.X, axis=0) / np.nanvar(afm.X, axis=0)
    idx_vars = np.argsort(CV)[::-1][:n]

    # Filter vars and sites
    filtered = afm[:, idx_vars].copy()
    filtered = remove_excluded_sites(filtered)

    return filtered


##


def filter_ludwig2019(afm, mean_AF=0.5, mean_qual=20):
    """
    Filter variants based on fixed tresholds adopted in Ludwig et al., 2019, 
    in the experiment without ATAC-seq reference, Fig.7.
    """
    # Create test
    test_vars_het = np.nanmean(afm.X, axis=0) > mean_AF # highly heteroplasmic variants
    test_vars_qual = np.nanmean(afm.layers['quality'], axis=0) > mean_qual # high quality vars
    test_vars = test_vars_het & test_vars_qual

    # Filter vars and sites
    filtered = afm[:, test_vars].copy()
    filtered = remove_excluded_sites(filtered)

    return filtered


##


def filter_velten2021(afm, blacklist=None, sample=None, mean_AF=0.1, min_cell_perc=0.2):
    """
    Filter variants based on fixed tresholds adopted in Ludwig et al., 2021.
    """
    # Site covered by at least 5 UMIs (median) in 20 cells
    test_sites = pd.Series(
        np.sum(afm.uns['per_position_coverage'] > 5, axis=0) > 20,
        index=afm.uns['per_position_coverage'].columns
    )
    sites = test_sites[test_sites].index
    test_vars_site_coverage = afm.var_names.map(lambda x: x.split('_')[0] in sites).to_numpy(dtype=bool)

    # Min avg heteroplasmy and min cell prevalence
    test_vars_het = np.nanmean(afm.X, axis=0) > mean_AF
    test_vars_exp = (np.sum(afm.X > 0, axis=0) / afm.shape[0]) > min_cell_perc
    test_vars = test_vars_site_coverage & test_vars_het & test_vars_exp
    candidate_vars = afm.var_names[test_vars]

    # Remove from blacklist.
    if blacklist is not None:
        passing_vars = remove_from_blacklist(candidate_vars, blacklist, sample=sample)

    # Filter vars and sites
    filtered = afm[:, passing_vars].copy()
    filtered = remove_excluded_sites(filtered)

    return filtered

##


def filter_miller2022(afm, mean_coverage=100, mean_qual=30, 
    perc_1=0.01, perc_99=0.1): 
    """
    Filter variants based on adaptive adopted in Miller et al., 2022.
    """
    # Site covered by at least (median) 10 UMIs
    test_sites = pd.Series(
       np.mean(afm.uns['per_position_coverage'], axis=0) > mean_coverage,
        index=afm.uns['per_position_coverage'].columns
    )
    sites = test_sites[test_sites].index
    test_vars_site_coverage = afm.var_names.map(lambda x: x.split('_')[0] in sites).to_numpy(dtype=bool)

    # Avg quality and percentile heteroplasmy tests
    test_vars_qual = np.nanmean(afm.layers['quality'], axis=0) > mean_qual
    test_vars_het = (np.percentile(afm.X, q=1, axis=0) < perc_1) & (np.percentile(afm.X, q=99, axis=0) > perc_99)
    test_vars = test_vars_site_coverage & test_vars_qual & test_vars_het
    candidate_vars = afm.var_names[test_vars]

    # Filter vars and sites
    filtered = afm[:, candidate_vars].copy()
    filtered = remove_excluded_sites(filtered)

    return filtered


##


def filter_seurat(afm, nbins=50, n=1000, log=True):
    """
    Filter with the scanpy/seurat flavour, readapted from scanpy.
    """
    # Calc stats
    X = afm.X
    X[np.isnan(X)] = 0
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    dispersion = np.full(X.shape[1], np.nan)
    idx_valid = (mean > 0.0) & (var > 0.0)
    dispersion[idx_valid] = var[idx_valid] / mean[idx_valid] 

    # Optional, put them in the log space
    if log:
        mean = np.log1p(mean) 
        dispersion = np.log(dispersion)
    
    # Bin dispersion values, subtract from each value the bin dispersion mean and divide by the bin std
    df = pd.DataFrame({"log_dispersion": dispersion, "bin": pd.cut(mean, bins=nbins)})
    log_disp_groups = df.groupby("bin")["log_dispersion"]
    log_disp_mean = log_disp_groups.mean()
    log_disp_std = log_disp_groups.std(ddof=1)
    log_disp_zscore = (
        df["log_dispersion"].values - log_disp_mean.loc[df["bin"]].values
    ) / log_disp_std.loc[df["bin"]].values
    log_disp_zscore[np.isnan(log_disp_zscore)] = 0.0

    # Rank, order and slice first n. Subset AFM
    hvf_rank = np.full(X.shape[1], -1, dtype=int)
    ords = np.argsort(log_disp_zscore)[::-1]
    hvf_rank[ords[:n]] = range(n)
    select = np.where(hvf_rank != -1)[0]
    filtered = afm[:, select].copy()
    filtered = remove_excluded_sites(filtered) # Remove sites

    return filtered


##


def filter_pegasus(afm, span=0.02, n=1000):
    """
    Filter with the method implemented in pegasus.
    """
    # Get means and vars
    X = afm.X
    X[np.isnan(X)] = 0
    mean = X.mean(axis=0)
    var = X.var(axis=0)

    # Fit var to the mean with loess regression, readjusting the span
    span_value = span
    while True:
        lobj = fit_loess(mean, var, span=span_value, degree=2)
        if lobj is not None:
            break
        span_value += 0.01

    # Create two ranks
    rank1 = np.zeros(mean.size, dtype=int)
    rank2 = np.zeros(mean.size, dtype=int)
    delta = var - lobj.outputs.fitted_values # obs variance - fitted one
    fc = var / lobj.outputs.fitted_values # obs variance / fitted one
    rank1[np.argsort(delta)[::-1]] = range(mean.size) # Rank in desc order
    rank2[np.argsort(fc)[::-1]] = range(mean.size)

    # Rank according to the sum of the two ranks, and filter AFM.
    hvf_rank = rank1 + rank2
    hvf_index = np.zeros(mean.size, dtype=bool)
    hvf_index[np.argsort(hvf_rank)[:n]] = True
    filtered = afm[:, hvf_index].copy()
    filtered = remove_excluded_sites(filtered) # Remove sites

    return filtered


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


def filter_Mquad(afm, nproc=8, minDP=10, minAD=1, minCell=3, path_=None):
    """
    Filter variants using the Mquad method.
    """
    AD, DP, ad_vars = get_AD_DP(afm, to='coo')

    # Select variants
    assert DP.shape == AD.shape
    M = Mquad(AD=AD, DP=DP)
    df = M.fit_deltaBIC(out_dir=path_, nproc=nproc, minDP=minDP, minAD=minAD)
    best_ad, best_dp = M.selectInformativeVariants(
        min_cells=minCell, out_dir=path_, tenx_cutoff=None, export_heatmap=False, export_mtx=False
    )
    selected_idx = M.final_df.index.to_list()
    selected_vars = [ ad_vars[i] for i in selected_idx ]
    # Subset matrix
    filtered = afm[:, selected_vars].copy()
    filtered = remove_excluded_sites(filtered) # Remove sites

    return filtered


##


def filter_DADApy(afm):
    """
    Filter using DADApy.
    """

    return 'Not implemented yet...'


##


def filter_density(afm, density=0.5, steps=np.Inf):
    """
    Jointly filter cells and variants based on the iterative filtering algorithm 
    adopted by Moravec et al., 2022.
    """
    # Get AF matrix, convert into a df
    logger = logging.getLogger("mito_benchmark")
    X_bool = np.where(~np.isnan(afm.X), 1, 0)

    # Check initial density not already above the target one
    d0 = X_bool.sum() / X_bool.size
    if d0 >= density:
        logger.info(f'Density is already more than the desired target: {d0}')
        return afm
        
    else:
        print(f'Initial density: {d0}')

    # Iteratively remove lowest density rows/cols, until desired density is reached
    densities = []
    i = 0
    while i < steps:

        print(f'Step {i}:')
        rowsums = X_bool.sum(axis=1)
        colsums = X_bool.sum(axis=0)
        d = X_bool.sum() / X_bool.size
        densities.append(d)
        print(f'Density: {d}')

        if d >= density or (len(rowsums) == 0 or len(colsums) == 0):
            break

        rowmin = rowsums.min()
        colmin = colsums.min()
        if rowmin <= colmin:
            lowest_density_rows = np.where(rowsums == rowmin)[0]
            X_bool = X_bool[ [ i for i in range(X_bool.shape[0]) if i not in lowest_density_rows ], :]
            afm = afm[ [ i for i in range(afm.shape[0]) if i not in lowest_density_rows ], :].copy()
        else:
            lowest_density_cols = np.where(colsums == colmin)[0]
            X_bool = X_bool[:, [ i for i in range(X_bool.shape[1]) if i not in lowest_density_cols ] ]
            afm = afm[:, [ i for i in range(afm.shape[1]) if i not in lowest_density_cols ] ].copy()
        i += 1

    afm = remove_excluded_sites(afm) 
    
    return afm


##


def filter_cells_and_vars(
    afm, blacklist=None, sample=None, filtering=None, min_cell_number=0, 
    filter_cells=True, min_cov_treshold=None, variants=None, cells=None, 
    nproc=8, path_=None, n=1000):
    """
    Filter cells and vars from an afm.
    """ 
    logger = logging.getLogger("mito_benchmark")
    logger.info(f'Filter cells and variants for the original AFM...')

    if filtering in filtering_options and filtering != 'density':

        # Cells
        logger.info(f'Feature selection method: {filtering}')

        if filter_cells:
            logger.info(f'Filtering cells with >{min_cov_treshold} coverage')
            a_cells = filter_cells_coverage(afm, mean_coverage=min_cov_treshold)
            logger.info(f'Original AFM n cells: {afm.shape[0]}')
            logger.info(f'Filtered AFM n cells: {a_cells.shape[0]}')
            logger.info(f'Removed n {afm.shape[0]-a_cells.shape[0]} cells')
        else:
            logger.info(f'No cell filtering according to MT-coverage. Already done?')
            a_cells = afm.copy()

        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logger.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            test = a_cells.obs['GBC'].isin(clones_to_retain)
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
            a_cells = a_cells[test, :].copy()
            logger.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logger.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
       
        # Variants
        a_cells = filter_baseline(a_cells)
        if filtering == 'CV':
            a = filter_CV(a_cells, n=100)
        elif filtering == 'ludwig2019':
            a = filter_ludwig2019(a_cells)
        elif filtering == 'velten2021':
            a = filter_velten2021(a_cells, blacklist=blacklist, sample=sample)
        elif filtering == 'miller2022':
            a = filter_miller2022(a_cells)
        elif filtering == 'seurat':
            a = filter_seurat(a_cells, n=n)
        elif filtering == 'pegasus':
            a = filter_pegasus(a_cells, n=n)
        elif filtering == 'MQuad':
            a = filter_Mquad(a_cells, nproc=nproc, path_=path_)
        elif filtering == 'DADApy':
            a = filter_DADApy(a_cells)

    elif filtering == 'density':
        a_cells = filter_cells_coverage(afm, mean_coverage=min_cov_treshold)
        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logger.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            cells_to_retain = a_cells.obs.query('GBC in @clones_to_retain').index
            a_cells = a_cells[cells_to_retain, :].copy()
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[cells_to_retain, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[cells_to_retain, :]
            logger.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logger.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
        a_cells = filter_baseline(a_cells)
        a = filter_density(a_cells)

    elif filtering == 'LINEAGE_prep':
        a_cells = filter_cells_coverage(afm, mean_coverage=min_cov_treshold)
        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logger.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            test = a_cells.obs['GBC'].isin(clones_to_retain)
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
            a_cells = a_cells[test, :].copy()
            logger.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logger.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
        a = a_cells.copy()

    elif cells is None and variants is not None:
        a_cells = filter_cells_coverage(afm, mean_coverage=min_cov_treshold)
        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logger.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            test = a_cells.obs['GBC'].isin(clones_to_retain)
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
            a_cells = a_cells[test, :].copy()
            logger.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logger.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
        a = a_cells[:, variants].copy()
        a = remove_excluded_sites(a)

    elif cells is not None and variants is None:
        a_cells = afm[cells, :].copy()
        a_cells = remove_excluded_sites(a_cells)
        a = a_cells

    elif cells is not None and variants is not None:
        a_cells = afm[cells, variants].copy()
        a_cells = remove_excluded_sites(a_cells)
        a = a_cells
    
    else:
        raise ValueError(
                    f'''The provided filtering method {filtering} is not supported.
                        Choose another one...'''
                )

    logger.info(f'Filtered feature matrix contains {a.shape[0]} cells and {a.shape[1]} variants.')

    return a_cells, a


##



def summary_stats_vars(afm, variants=None):
    """
    Calculate the most important summary stats for a bunch of variants, collected for
    a set of cells.
    """
    if variants is not None:
        test = afm.var_names.isin(variants)
        density = (~np.isnan(afm[:, test].X)).sum(axis=0) / afm.shape[0]
        median_vafs = np.nanmedian(afm[:, test].X, axis=0)
        median_coverage_var = np.nanmedian(afm[:, test].layers['coverage'], axis=0)
        fr_positives = np.sum(afm[:, test].X > 0, axis=0) / afm.shape[0]
        var_names = afm.var_names[test]
    else:
        density = (~np.isnan(afm.X)).sum(axis=0) / afm.shape[0]
        median_vafs = np.nanmedian(afm.X, axis=0)
        median_coverage_var = np.nanmedian(afm.layers['coverage'], axis=0)
        fr_positives = np.sum(afm.X > 0, axis=0) / afm.shape[0]
        var_names = afm.var_names

    df = pd.DataFrame(
        {   
            'density' : density,
            'median_coverage' : median_coverage_var,
            'median_AF' : median_vafs,
            'fr_positives' : fr_positives
        }, index=var_names
    )

    return df
