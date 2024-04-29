"""
Module to preprocess AFMs: reformat original AFM; filter variants/cells.
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from mquad.mquad import *
from scipy.sparse import coo_matrix, csc_matrix
from pegasus.tools.hvf_selection import fit_loess
from itertools import chain


##


filtering_options = [
    'CV',
    'ludwig2019', 
    'velten2021', 
    'miller2022', 
    'weng2024',
    'seurat', 
    'pegasus', 
    'MQuad', 
    'MQuad_optimized',
    'density'
]


##


def nans_as_zeros(afm):
    """
    Fill nans with zeros.
    """
    X_copy = afm.X.copy()
    X_copy[np.isnan(X_copy)] = 0
    afm.X = X_copy
    return afm


##


def filter_sites(afm):
    """
    Filter sites info belonging only to selected AFM variants.
    """
    cells = afm.obs_names
    sites_retained = afm.var_names.map(lambda x: x.split('_')[0]).unique()
    afm.uns['per_position_coverage'] = afm.uns['per_position_coverage'].loc[cells, sites_retained]
    afm.uns['per_position_quality'] = afm.uns['per_position_quality'].loc[cells, sites_retained]

    return afm


##


def filter_CV(afm, n=1000):
    """
    Filter variants based on their coefficient of variation.
    """
    CV = np.mean(afm.X, axis=0) / np.var(afm.X, axis=0)
    idx_vars = np.argsort(CV)[::-1][:n]
    filtered = afm[:, idx_vars].copy()
    filtered = filter_sites(filtered)

    return filtered


##


def filter_ludwig2019(afm, mean_AF=0.5, mean_qual=20):
    """
    Filter variants based on fixed tresholds adopted in Ludwig et al., 2019, 
    in the experiment without ATAC-seq reference, Fig.7.
    """
    test_vars_het = np.mean(afm.X, axis=0) > mean_AF                        # high average AF variants
    test_vars_qual = np.mean(afm.layers['quality'], axis=0) > mean_qual     # high average quality variants
    test_vars = test_vars_het & test_vars_qual
    filtered = afm[:, test_vars].copy()
    filtered = filter_sites(filtered)

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
    filtered = filter_sites(filtered)

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
    filtered = filter_sites(filtered) # Remove sites

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
    delta = var - lobj.outputs.fitted_values          # obs variance - fitted one
    fc = var / lobj.outputs.fitted_values             # obs variance / fitted one
    rank1[np.argsort(delta)[::-1]] = range(mean.size) # Rank in desc order
    rank2[np.argsort(fc)[::-1]] = range(mean.size)

    # Rank according to the sum of the two ranks, and filter AFM.
    hvf_rank = rank1 + rank2
    hvf_index = np.zeros(mean.size, dtype=bool)
    hvf_index[np.argsort(hvf_rank)[:n]] = True
    filtered = afm[:, hvf_index].copy()
    filtered = filter_sites(filtered)        # Remove sites

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


def fit_MQuad_mixtures(afm, n=None, path_=None, nproc=1, minDP=10, minAD=1, with_M=False):
    """
    Filter variants using the Mquad method.
    """
    # Prefilter again, if still too much
    if n is not None:
        afm = filter_pegasus(afm, n=n)  
        afm = filter_sites(afm)

    AD, DP, ad_vars = get_AD_DP(afm, to='coo')

    # Fit models
    M = Mquad(AD=AD, DP=DP)
    path_ = os.getcwd() if path_ is None else path_
    df = M.fit_deltaBIC(out_dir=path_, nproc=nproc, minDP=minDP, minAD=minAD)
    df.index = ad_vars
    df['deltaBIC_rank'] = df['deltaBIC'].rank(ascending=False)

    if with_M:
        return df.sort_values('deltaBIC', ascending=False), M, ad_vars
    else:
        return df.sort_values('deltaBIC', ascending=False)
    

##


def filter_Mquad(afm, nproc=8, minDP=10, minAD=1, minCell=3, path_=None, n=None):
    """
    Filter variants using the Mquad method.
    """
    df, M, ad_vars = fit_MQuad_mixtures(
        afm, n=n, path_=path_, nproc=nproc, minDP=minDP, minAD=minAD, with_M=True, 
    )
    best_ad, best_dp = M.selectInformativeVariants(
        min_cells=minCell, out_dir=path_, tenx_cutoff=None,
        export_heatmap=False, export_mtx=False
    )
    selected_idx = M.final_df.index.to_list()
    selected_vars = [ ad_vars[i] for i in selected_idx ]

    # Subset matrix
    filtered = afm[:, selected_vars].copy()
    filtered = filter_sites(filtered) 

    # Write df
    os.system(f'rm {os.path.join(path_, "*BIC*")}')
    df.to_csv(os.path.join(path_, 'MQuad_stats.csv'))

    return filtered


##


def z_test_vars_into_bins(delta, alpha=0.05):
    """
    Perform a z-test on each deltaBIC value within a bin.
    """
    delta = delta.loc[lambda x: ~x.isna()]
    z_scores = ( delta-delta.mean() ) / ( delta.std() / (delta.shape[0] ** 0.5) )
    critical_z_value = stats.norm.ppf(1-alpha)
    variants = z_scores.loc[lambda x: x>critical_z_value].index.to_list()

    return variants
    

##


def calc_median_AF_in_positives(afm, candidates):

    median_af = []
    for x in candidates:
        idx = afm[:, x].X.toarray().flatten()>0
        median_af.append(np.median(afm[idx, x].X, axis=0)[0])

    return np.array(median_af)
    

##


def filter_from_bins(df, afm, min_f, max_f, treshold_AF=.05, n_bins=10):
        
    bins = np.linspace(min_f, max_f, n_bins+1)
    df['bin'] = pd.cut(
        df['fr_pos_cells'], bins=bins, include_lowest=True, labels=False
    )
    var_l = df.groupby('bin').apply(lambda x: z_test_vars_into_bins(x['deltaBIC']))
    candidates = list(chain.from_iterable(var_l))
    test = calc_median_AF_in_positives(afm, candidates)>=treshold_AF
    filtered_candidates = pd.Series(candidates)[test].to_list()
    
    return filtered_candidates


##


def filter_Mquad_optimized(afm, nproc=8, minDP=10, minAD=1, path_=None,
    split_t=.2, treshold_AF_high=.05, treshold_AF_low=.025, n_bins=10
    ):
    """
    Filter variants using the MQuad method, optimized version.
    """
    df, M, ad_vars = fit_MQuad_mixtures(
        afm, path_=path_, nproc=nproc, minDP=minDP, minAD=minAD, with_M=True, 
    )

    # Split into high and low
    df['fr_pos_cells'] = df['num_cells_nonzero_AD'] / afm.shape[0]
    df['fr_pos_cells'] = df['fr_pos_cells'].astype('float')
    df['deltaBIC'] = df['deltaBIC'].astype('float')

    # Get vars
    high_vars = filter_from_bins(
        df.query('fr_pos_cells>=@split_t').copy(), 
        afm, split_t, 1, treshold_AF=treshold_AF_high, n_bins=n_bins
    )
    low_vars = filter_from_bins(
        df.query('fr_pos_cells<@split_t').copy(), 
        afm, 0, split_t, treshold_AF=treshold_AF_low, n_bins=n_bins
    )

    # Subset matrix
    filtered = afm[:, high_vars+low_vars].copy()
    filtered = filter_sites(filtered) # Remove sites

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

    # Check initial density not already above the target one
    X_bool = np.where(afm.X>0, 1, 0)
    d0 = X_bool.sum() / X_bool.size
    if d0 >= density:
        logging.info(f'Density is already more than the desired target: {d0}')
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

    afm = filter_sites(afm) 
    
    return afm


##


def make_vars_df(afm):
    """
    Compute vars_df as in in Weng et al., 2024, and Miller et al. 2022 before.
    """

    # Initialize vars_df

    # vars.tib <- tibble(var = rownames(af.dm),
    #                    mean_af = rowMeans(af.dm),
    #                    mean_cov = rowMeans(assays(maegtk)[["coverage"]])[as.numeric(cutf(rownames(af.dm), d = "_"))],
    #                    quality = qual.num)

    afm.uns['per_position_coverage'].mean(axis=0).mean() # Good mean coverage
    mean_sites = afm.uns['per_position_coverage'].mean(axis=0)
    vars_df = pd.DataFrame(
        np.column_stack([
            afm.X.mean(axis=0),
            afm.var_names.map(lambda x: mean_sites[x.split('_')[0]]),
            np.ma.getdata(np.ma.mean(np.ma.masked_less_equal(afm.layers['quality'], 0), axis=0))
        ]),
        columns=['mean_af', 'mean_cov', 'quality'],
        index=afm.var_names
    )

    # Calculate the number of cells that exceed VAF thresholds 0, 1, 5, 10, 50 as in Weng et al., 2024

    # vars.tib <- vars.tib %>%
    #     mutate(n0 = apply(af.dm, 1, function(x) sum(x == 0))) %>%  # NEGATIVE CELLS
    #     mutate(n1 = apply(af.dm, 1, function(x) sum(x > 1))) %>%
    #     mutate(n5 = apply(af.dm, 1, function(x) sum(x > 5))) %>%
    #     mutate(n10 = apply(af.dm, 1, function(x) sum(x > 10))) %>%
    #     mutate(n50 = apply(af.dm, 1, function(x) sum(x > 50)))
    # Variant_CellN<-apply(af.dm,1,function(x){length(which(x>0))})
    # vars.tib<-cbind(vars.tib,Variant_CellN)

    vars_df = (
        vars_df
        .assign(
            n0 = lambda x: np.sum(afm.X==0, axis=0),                # NEGATIVE CELLS
            n1 = lambda x: np.sum(afm.X>.01, axis=0),
            n5 = lambda x: np.sum(afm.X>.05, axis=0),
            n10 = lambda x: np.sum(afm.X>.1, axis=0),
            n50 = lambda x: np.sum(afm.X>.5, axis=0),
            Variant_CellN = lambda x: np.sum(afm.X>0, axis=0),
        )
    )

    return vars_df


##


def filter_baseline_old(afm):
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
    # variants with quality >= 30 (mean, over all cells); 
    # variants seen in at least 2 cells;
    # variants with AF>0.01 in at least 2 cells;
    test_vars_quality = np.nanmean(afm.layers['quality'], axis=0) > 30
    test_vars_coverage = np.sum(afm.layers['coverage']>0, axis=0) > 2
    test_vars_AF = np.sum(afm.X > 0.01, axis=0) > 2

    # Filter vars and sites
    test_vars = test_vars_site_coverage & test_vars_quality & test_vars_coverage & test_vars_AF 
    filtered = afm[:, test_vars].copy()
    filtered = filter_sites(filtered)

    return filtered


##


def filter_baseline(afm, min_site_cov=5, min_var_quality=30, min_n_positive=2):
    """
    Compute summary stats and filter baseline variants.
    """
    vars_df = make_vars_df(afm)
    variants = (
        vars_df.loc[
            (vars_df['mean_cov']>min_site_cov) & \
            (vars_df['quality']>min_var_quality) & \
            (vars_df['Variant_CellN']>=min_n_positive) 
        ]
        .index
    )
    
    # Subset matrix
    filtered = afm[:, variants].copy()
    filtered = filter_sites(filtered) 

    return filtered


##


def filter_weng2024(
    afm, vars_df, 
    min_site_cov=5, min_var_quality=30, min_frac_negative=.9, min_n_positive=2,
    low_af=.1, high_af=.5, min_frac_cells_below_low_af=.1, min_n_cells_above_high_af=2
    ):
    """
    Calculate vars_df and select MT-vars, as in in Weng et al., 2024, and Miller et al. 2022 before.
    """

    # Filter Weng et al., 2024

    # vars_filter.tib <- vars.tib %>% filter(mean_cov > 5, quality >= 30, n0 > 0.9*ncol(af.dm),Variant_CellN>=2)

    ## Apply the same filter as in MAESTER
    # IsInfo<-function(x){
    # total<-length(x)
    # if(length(which(x<10))/total>0.1 & length(which(x>50))>10){
    #     return("Variable")
    # }else{
    #     return("Non")
    # }
    # }
    # Variability<-apply(af.dm,1,IsInfo) %>% data.frame(Info=.)

    # vars_filter.tib<-Tomerge_v2(vars_filter.tib,Variability) 

    vars_df = (
        vars_df.loc[
            (vars_df['mean_cov']>min_site_cov) & \
            (vars_df['quality']>min_var_quality) & \
            (vars_df['n0']>min_frac_negative) & \
            (vars_df['Variant_CellN']>=min_n_positive) 
        ]
    )

    # Detect "Variable" variants as in MAESTER

    # IsInfo<-function(x){
    # total<-length(x)
    # if(length(which(x<10))/total>0.1 & length(which(x>50))>10){
    #     return("Variable")
    # }else{
    #     return("Non")
    # }
    # }
    # Variability<-apply(af.dm,1,IsInfo) %>% data.frame(Info=.)

    test_low_homoplasy = (afm.X<low_af).sum(axis=0)/afm.shape[0] > min_frac_cells_below_low_af
    test_detection_evidence = (afm.X>high_af).sum(axis=0) > min_n_cells_above_high_af
    test = test_low_homoplasy & test_detection_evidence
    vars_df['weng_2024'] = pd.Series(np.where(test, 'Variable', 'Non'), index=afm.var_names).loc[vars_df.index]

    # Subset matrix
    filtered = afm[:, vars_df['weng_2024'].loc[lambda x: x=='Variable'].index].copy()
    filtered = filter_sites(filtered) 

    return filtered


##