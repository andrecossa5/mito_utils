"""
Module to preprocess AFMs: reformat original AFM; filter variants/cells.
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from mquad.mquad import *
from scipy.sparse import coo_matrix, csc_matrix
from itertools import chain


##


filtering_options = [
    'baseline',
    'CV',
    'ludwig2019', 
    'velten2021', 
    'miller2022', 
    'weng2024',
    'seurat', 
    'MQuad', 
    'MQuad_optimized',
    'density',
    'MI_TO'
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


def mask_mt_sites(afm):
    """
    Function to mask all sites outside of known MT-genes bodies.
    """

    # All expressed mitochondrial genes with their start and end positions
    all_mt_genes_positions = [
        ["MT-ND1", 3307, 4262], ["MT-ND2", 4470, 5511], ["MT-CO1", 5904, 7445],
        ["MT-CO2", 7586, 8269], ["MT-ATP8", 8366, 8572], ["MT-ATP6", 8527, 9207],
        ["MT-CO3", 9207, 9990], ["MT-ND3", 10059, 10404], ["MT-ND4L", 10470, 10766],
        ["MT-ND4", 10760, 12137], ["MT-ND5", 12337, 14148], ["MT-ND6", 14149, 14673],
        ["MT-CYB", 14747, 15887], ["MT-TF", 577, 647], ["MT-TV", 1602, 1670],
        ["MT-TL1", 3230, 3304], ["MT-TI", 4263, 4331], ["MT-TQ", 4329, 4400],
        ["MT-TM", 4402, 4469], ["MT-TW", 5512, 5579], ["MT-TA", 5587, 5655],
        ["MT-TN", 5657, 5729], ["MT-TC", 5761, 5826], ["MT-TY", 5826, 5891],
        ["MT-TS1", 7518, 7585], ["MT-TD", 7513, 7585], ["MT-TK", 8295, 8364],
        ["MT-TG", 9991, 10058], ["MT-TR", 10405, 10469], ["MT-TH", 12138, 12206],
        ["MT-TS2", 12207, 12265], ["MT-TL2", 12266, 12336], ["MT-TE", 14674, 14742],
        ["MT-TT", 15888, 15953], ["MT-TP", 15956, 16023], ["12S rRNA", 648, 1601],
        ["16S rRNA", 1671, 3229]
    ]

    # Here we go
    sites = afm.uns['per_position_coverage'].columns
    mask = []
    for x in sites:
        x = int(x)
        t = [ x>=start and x<=end for _, start, end in all_mt_genes_positions ]
        if any(t):
            mask.append(True)
        else:
            mask.append(False)

    return np.array(mask)


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


def filter_cell_clones(afm, min_cell_number=10):
    """
    Filter only cells from clones (i.e., GBC column in afm.obs) with at least
    min_cell_number cells.
    """
    print(f'Filtering cells from clones with >={min_cell_number} cells')
          
    cell_counts = afm.obs.groupby('GBC').size()
    clones_to_retain = cell_counts[cell_counts>=min_cell_number].index 
    test = afm.obs['GBC'].isin(clones_to_retain)
    afm.uns['per_position_coverage'] = afm.uns['per_position_coverage'].loc[test, :]
    afm.uns['per_position_quality'] = afm.uns['per_position_quality'].loc[test, :]
    afm = afm[test, :].copy()

    print(f'Removed other {afm-afm.shape[0]} cells')
    print(f'Retaining {afm.obs["GBC"].unique().size} clones for the analysis.')
          
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

    mean_sites = afm.uns['per_position_coverage'].mean(axis=0)
    vars_df = pd.DataFrame(
        np.column_stack([
            afm.X.mean(axis=0),
            afm.var_names.map(lambda x: mean_sites[x.split('_')[0]]),
            np.ma.getdata(np.ma.mean(np.ma.masked_less_equal(afm.layers['quality'].astype(np.int16), 0), axis=0)),
            np.ma.getdata(np.ma.mean(np.ma.masked_less_equal(afm.X.astype(np.float16), 0), axis=0))
        ]),
        columns=['mean_af', 'mean_cov', 'quality', 'median_af_in_positives'],
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


def filter_baseline(afm, min_site_cov=5, min_var_quality=30, min_n_positive=2, only_genes=True):
    """
    Compute summary stats and filter baseline variants.
    """
    # Remove variants outside gene sites
    if only_genes:
        test = mask_mt_sites(afm)
        filtered_sites = afm.uns['per_position_coverage'].columns[test]
        test = afm.var_names.map(lambda x: x.split('_')[0]).isin(filtered_sites)
        filtered_vars = afm.var_names[test]
        afm = afm[:,filtered_vars].copy()
        afm = filter_sites(afm)
    # Basic filter as Weng et al., 2024
    vars_df = make_vars_df(afm)
    variants = (
        vars_df.loc[
            (vars_df['mean_cov']>=min_site_cov) & \
            (vars_df['quality']>=min_var_quality) & \
            (vars_df['Variant_CellN']>=min_n_positive) 
        ]
        .index
    )
    # Subset matrix
    filtered = afm[:, variants].copy()
    filtered = filter_sites(filtered) 

    return filtered


##


def filter_CV(afm, n_top=1000):
    """
    Filter variants based on their coefficient of variation.
    """
    CV = (np.std(afm.X, axis=0) / np.mean(afm.X, axis=0)) * 100
    idx_vars = np.argsort(CV)[::-1][:n_top]
    filtered = afm[:, idx_vars].copy()
    filtered = filter_sites(filtered)

    return filtered


##


def filter_ludwig2019(afm, mean_af=0.5, mean_qual=20):
    """
    Filter variants based on fixed tresholds adopted in Ludwig et al., 2019, 
    in the experiment without ATAC-seq reference, Fig.7.
    """
    test_vars_het = np.mean(afm.X, axis=0) >= mean_af                        # high average AF variants
    test_vars_qual = np.mean(afm.layers['quality'], axis=0) >= mean_qual     # high average quality variants
    test_vars = test_vars_het & test_vars_qual
    filtered = afm[:, test_vars].copy()
    filtered = filter_sites(filtered)

    return filtered


##


def filter_miller2022(afm, mean_cov=100, mean_qual=20, 
    perc_1=0.01, perc_99=0.1): 
    """
    Filter variants based on adaptive adopted in Miller et al., 2022.
    """
    # Site covered by at least (median) 10 UMIs
    test_sites = pd.Series(
       np.mean(afm.uns['per_position_coverage'], axis=0) >= mean_cov,
        index=afm.uns['per_position_coverage'].columns
    )
    sites = test_sites[test_sites].index
    test_vars_site_coverage = afm.var_names.map(lambda x: x.split('_')[0] in sites).to_numpy(dtype=bool)

    # Avg quality and percentile heteroplasmy tests
    test_vars_qual = np.nanmean(afm.layers['quality'], axis=0) >= mean_qual
    test_vars_het = (np.percentile(afm.X, q=1, axis=0) < perc_1) & (np.percentile(afm.X, q=99, axis=0) > perc_99)
    test_vars = test_vars_site_coverage & test_vars_qual & test_vars_het
    candidate_vars = afm.var_names[test_vars]

    # Filter vars and sites
    filtered = afm[:, candidate_vars].copy()
    filtered = filter_sites(filtered)

    return filtered


##


def filter_seurat(afm, nbins=50, n_top=1000, log=True):
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
    hvf_rank[ords[:n_top]] = range(n_top)
    select = np.where(hvf_rank != -1)[0]
    filtered = afm[:, select].copy()
    filtered = filter_sites(filtered) # Remove sites

    return filtered


##


# To much problem with sc-misc installation
# def filter_pegasus(afm, span=0.02, n=1000):
#     """
#     Filter with the method implemented in pegasus.
#     """
#     # Get means and vars
#     X = afm.X
#     X[np.isnan(X)] = 0
#     mean = X.mean(axis=0)
#     var = X.var(axis=0)
# 
#     # Fit var to the mean with loess regression, readjusting the span
#     span_value = span
#     while True:
#         lobj = fit_loess(mean, var, span=span_value, degree=2)
#         if lobj is not None:
#             break
#         span_value += 0.01
# 
#     # Create two ranks
#     rank1 = np.zeros(mean.size, dtype=int)
#     rank2 = np.zeros(mean.size, dtype=int)
#     delta = var - lobj.outputs.fitted_values          # obs variance - fitted one
#     fc = var / lobj.outputs.fitted_values             # obs variance / fitted one
#     rank1[np.argsort(delta)[::-1]] = range(mean.size) # Rank in desc order
#     rank2[np.argsort(fc)[::-1]] = range(mean.size)
# 
#     # Rank according to the sum of the two ranks, and filter AFM.
#     hvf_rank = rank1 + rank2
#     hvf_index = np.zeros(mean.size, dtype=bool)
#     hvf_index[np.argsort(hvf_rank)[:n]] = True
#     filtered = afm[:, hvf_index].copy()
#     filtered = filter_sites(filtered)        # Remove sites
# 
#     return filtered


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


def fit_MQuad_mixtures(afm, n_top=None, path_=None, nproc=1, minDP=10, minAD=1, with_M=False):
    """
    Filter variants using the Mquad method.
    """
    # Prefilter again, if still too much
    if n_top is not None:
        afm = filter_seurat(afm, n_top=n_top)  
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


def filter_Mquad(afm, nproc=8, minDP=10, minAD=1, minCell=3, path_=None, n_top=None):
    """
    Filter variants using the Mquad method.
    """
    df, M, ad_vars = fit_MQuad_mixtures(
        afm, n_top=n_top, path_=path_, nproc=nproc, minDP=minDP, minAD=minAD, with_M=True, 
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


def filter_density(afm, density=0.1, t=.01, steps=np.Inf):
    """
    Jointly filter cells and variants based on the iterative filtering algorithm 
    adopted by Moravec et al., 2022.
    """

    # Check initial density not already above the target one
    X_bool = np.where(afm.X>=t, 1, 0)
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

        rowsums = X_bool.sum(axis=1)
        colsums = X_bool.sum(axis=0)
        d = X_bool.sum() / X_bool.size
        densities.append(d)
        print(f'Step {i}: density {d:.2f}')

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


# def filter_baseline_old(afm):
#     """
#     Baseline filter, applied on all variants, before any method-specific solution.
#     This is a very mild filter to exclude all variants that will be pretty much impossible
#     to use by any method due to:
#     * extremely low coverage at which the variant site have been observed across the population
#     * extremely low quality at which the variant site have been observed across the population
#     * Too less cells in which the variant have been detected with AF >1% 
#     """
#     # Test 1: variants whose site has been covered by nUMIs >= 10 (mean, over all cells)
#     test_sites = pd.Series(
#         np.mean(afm.uns['per_position_coverage'], axis=0) > 10,
#         index=afm.uns['per_position_coverage'].columns
#     )
#     sites = test_sites[test_sites].index
#     test_vars_site_coverage = (
#         afm.var_names
#         .map(lambda x: x.split('_')[0] in sites)
#         .to_numpy(dtype=bool)
#     )
#     # Test 2-4: 
#     # variants with quality >= 30 (mean, over all cells); 
#     # variants seen in at least 2 cells;
#     # variants with AF>0.01 in at least 2 cells;
#     test_vars_quality = np.nanmean(afm.layers['quality'], axis=0) > 30
#     test_vars_coverage = np.sum(afm.layers['coverage']>0, axis=0) > 2
#     test_vars_AF = np.sum(afm.X > 0.01, axis=0) > 2
# 
#     # Filter vars and sites
#     test_vars = test_vars_site_coverage & test_vars_quality & test_vars_coverage & test_vars_AF 
#     filtered = afm[:, test_vars].copy()
#     filtered = filter_sites(filtered)
# 
#     return filtered


##


def filter_weng2024(
    afm, 
    min_site_cov=5, 
    min_var_quality=30, 
    min_frac_negative=.9,
    min_n_positive=2,
    low_confidence_af=.1, 
    high_confidence_af=.5, 
    min_prevalence_low_confidence_af=.1, 
    min_cells_high_confidence_af=2,
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
    
    vars_df = make_vars_df(afm)
    vars_df = (
        vars_df.loc[
            (vars_df['mean_cov']>min_site_cov) & \
            (vars_df['quality']>=min_var_quality) & \
            (vars_df['n0']>min_frac_negative*afm.shape[0]) & \
            (vars_df['Variant_CellN']>=min_n_positive) 
        ]
    )

    # Detect "Variable" variants as in MAESTER

    # IsInfo<-function(x){
    # total<-length(x)
    # if(length(
        # which(x<10))/total>0.1        # Test1 : low prevalence of minimal detection.
        # & 
        # length(which(x>50))>10)       # Test2 : enough cells with confident detection.
        # {
    #     return("Variable")            
    # }else{
    #     return("Non")
    # }
    # }
    # Variability<-apply(af.dm,1,IsInfo) %>% data.frame(Info=.)

    t1 = (afm.X<low_confidence_af).sum(axis=0)/afm.shape[0] > min_prevalence_low_confidence_af
    t2 = (afm.X>high_confidence_af).sum(axis=0) > min_cells_high_confidence_af
    test = t1 & t2
    vars_df['weng_2024'] = (
        pd.Series(
        np.where(test, 'Variable', 'Non'), index=afm.var_names)
        .loc[vars_df.index]
    )

    # Subset matrix
    filtered = afm[:, vars_df['weng_2024'].loc[lambda x: x=='Variable'].index].copy()
    filtered = filter_sites(filtered) 

    return filtered


##


def filter_MI_TO(
    afm, 
    min_site_cov=25, 
    min_var_quality=30, 
    min_frac_negative=.9,
    min_n_positive=2,
    af_confident_detection=.01,
    min_n_confidently_detected=5,
    min_median_af=.01
    ):
    """
    Custom filter.
    """
    
    vars_df = make_vars_df(afm)
    n_confidently_detected = f'n{int(af_confident_detection*100)}'
    vars_df = (
        vars_df.loc[
            (vars_df['mean_cov']>min_site_cov) & \
            (vars_df['quality']>=min_var_quality) & \
            (vars_df['n0']>min_frac_negative*afm.shape[0]) & \
            (vars_df['Variant_CellN']>=min_n_positive) & \
            (vars_df[n_confidently_detected]>=min_n_confidently_detected) & \
            (vars_df['median_af_in_positives']>=min_median_af)     
        ]
    )

    # Subset matrix
    filtered = afm[:, vars_df.index].copy()
    filtered = filter_sites(filtered) 

    return filtered


##