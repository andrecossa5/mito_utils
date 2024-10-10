"""
Module to preprocess AFMs: reformat original AFM; filter variants/cells.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
from mquad.mquad import *
from mito_utils.distances import *
from mito_utils.make_afm import mask_mt_sites


##


filtering_options = [
    'baseline',
    'CV',
    'miller2022', 
    'weng2024',
    'MQuad', 
    'MI_TO',
    'GT_enriched'
    # 'ludwig2019', 
    # 'velten2021', 
    # 'seurat', 
    # 'MQuad_optimized',
    # 'density',
    # 'GT_stringent'
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


def filter_cells_with_at_least_one(a, bin_method='vanilla', binarization_kwargs={}):

    X = call_genotypes(a=a, bin_method=bin_method, **binarization_kwargs)
    a = a[a.obs_names[X.sum(axis=1)>=1],:]
    a.uns['per_position_coverage'] = a.uns['per_position_coverage'].loc[a.obs_names,:]
    a.uns['per_position_quality'] = a.uns['per_position_quality'].loc[a.obs_names,:]

    return a


##


def filter_cell_clones(afm, column='GBC', min_cell_number=10):
    """
    Filter only cells from afm.obs.<column> categories with >= min_cell_number cells.
    """
    
    logging.info(f'Filtering cells from clones with >={min_cell_number} cells')
    
    n0 = afm.shape[0]
    cell_counts = afm.obs.groupby(column).size()
    clones_to_retain = cell_counts[cell_counts>=min_cell_number].index 
    test = afm.obs[column].isin(clones_to_retain)
    afm = afm[test,:].copy()

    logging.info(f'Removed other {n0-afm.shape[0]} cells')
    logging.info(f'Retaining {afm.obs[column].unique().size} discrete categories (i.e., {column}) for the analysis.')
          
    return afm


##


def annotate_vars(afm, overwrite=False):
    """
    Annotate MT-SNVs properties as in in Weng et al., 2024, and Miller et al. 2022 before.
    Create vars_df and update .var.
    """

    if 'mean_af' in afm.var.columns:
        if not overwrite:
            return
        else:
            logging.info('Re-annotate variants in afm')
            afm.var = afm.var.iloc[:,:3].copy()

    # Initialize vars_df

    # vars.tib <- tibble(var = rownames(af.dm),
    #                    mean_af = rowMeans(af.dm),
    #                    mean_cov = rowMeans(assays(maegtk)[["coverage"]])[as.numeric(cutf(rownames(af.dm), d = "_"))],
    #                    quality = qual.num)

    afm.var['mean_af'] = afm.X.A.mean(axis=0)
    afm.var['mean_cov'] = afm.layers['site_coverage'].A.mean(axis=0)
    afm.var['quality'] = np.nanmean(np.where(afm.layers['qual'].A>0, afm.layers['qual'].A, np.nan), axis=0)

    # Calculate the number of cells that exceed VAF thresholds 0, 1, 5, 10, 50 as in Weng et al., 2024

    # vars.tib <- vars.tib %>%
    #     mutate(n0 = apply(af.dm, 1, function(x) sum(x == 0))) %>%  # NEGATIVE CELLS
    #     mutate(n1 = apply(af.dm, 1, function(x) sum(x > 1))) %>%
    #     mutate(n5 = apply(af.dm, 1, function(x) sum(x > 5))) %>%
    #     mutate(n10 = apply(af.dm, 1, function(x) sum(x > 10))) %>%
    #     mutate(n50 = apply(af.dm, 1, function(x) sum(x > 50)))
    # Variant_CellN<-apply(af.dm,1,function(x){length(which(x>0))})
    # vars.tib<-cbind(vars.tib,Variant_CellN)

    afm.var['n0'] = np.sum(afm.X.A==0, axis=0)              # NEGATIVE CELLS
    afm.var['n1'] = np.sum(afm.X.A>.01, axis=0)
    afm.var['n2'] = np.sum(afm.X.A>.02, axis=0)
    afm.var['n5'] = np.sum(afm.X.A>.05, axis=0)
    afm.var['n10'] = np.sum(afm.X.A>.1, axis=0)
    afm.var['n50'] = np.sum(afm.X.A>.5, axis=0)
    afm.var['Variant_CellN'] = np.sum(afm.X.A>0, axis=0)

    # Add mean AF, AD and DP in +cells
    afm.var['median_af_in_positives'] = np.nanmean(np.where(afm.X.A>0, afm.X.A, np.nan), axis=0)
    afm.var['mean_AD_in_positives'] = np.nanmean(
        np.where(afm.X.A>0, afm.layers['AD'].A, np.nan), axis=0
    )
    afm.var['mean_DP_in_positives'] = np.nanmean(
        np.where(afm.X.A>0, afm.layers['DP'].A, np.nan), axis=0
    )



##


def filter_baseline(afm, min_site_cov=5, min_var_quality=30, min_n_positive=2, only_genes=True):
    """
    Compute summary stats and filter baseline variants.
    """

    if only_genes:
        test_sites = mask_mt_sites(afm.var['pos'])
        afm = afm[:,test_sites].copy()

    # Basic filter as in Weng et al., 2024
    test_baseline = (
        (afm.var['mean_cov']>=min_site_cov) & \
        (afm.var['quality']>=min_var_quality) & \
        (afm.var['Variant_CellN']>=min_n_positive) 
    )
    afm = afm[:,test_baseline].copy()

    # Exclude sites with more than one alt alleles observed
    var_sites = afm.var_names.map(lambda x: x.split('_')[0])
    test = var_sites.value_counts()[var_sites]==1
    afm = afm[:,afm.var_names[test]].copy()

    # Exclude variants sites not observed in any cells and vice versa
    afm = afm[np.sum(afm.X.A>0, axis=1)>0,:].copy()
    afm = afm[:,np.sum(afm.X.A>0, axis=0)>0].copy()

    return afm


##


def filter_CV(afm, n_top=1000):
    """
    Filter variants based on their coefficient of variation (CV).
    """
    CV = (np.std(afm.X.A, axis=0)**2 / np.mean(afm.X.A, axis=0))
    idx_vars = np.argsort(CV)[::-1][:n_top]
    afm = afm[:,idx_vars].copy()
    return afm


##


def filter_miller2022(afm, min_site_cov=100, min_var_quality=30, p1=1, p2=99, perc1=0.01, perc2=0.1): 
    """
    Filter variants based on adaptive tresholds adopted in Miller et al., 2022.
    """

    test = (
        (afm.var['mean_cov']>=min_site_cov) & \
        (afm.var['quality']>=min_var_quality) & \
        ((np.percentile(afm.X.A, q=p1, axis=0) < perc1) & \
         (np.percentile(afm.X.A, q=p2, axis=0) > perc2))
    )
    afm = afm[:,test].copy()

    return afm


##


def fit_MQuad_mixtures(afm, n_top=25, path_=None, ncores=8, minDP=10, minAD=1, with_M=False):
    """
    Filter variants using the Mquad method.
    """
    # Prefilter again, if still too much
    if n_top is not None:
        afm = filter_CV(afm, n_top=1000)  

    # Fit models
    M = Mquad(AD=afm.layers['AD'].T, DP=afm.layers['DP'].T)
    path_ = os.getcwd() if path_ is None else path_
    df = M.fit_deltaBIC(out_dir=path_, nproc=ncores, minDP=minDP, minAD=minAD)
    df.index = afm.var_names
    df['deltaBIC_rank'] = df['deltaBIC'].rank(ascending=False)

    if with_M:
        return df.sort_values('deltaBIC', ascending=False), M
    else:
        return df.sort_values('deltaBIC', ascending=False)
    

##


def filter_MQuad(afm, nproc=8, minDP=10, minAD=1, minCell=3, path_=None, n_top=None):
    """
    Filter variants using the MQuad (Kwock 2022 et al.,) method.
    """

    _, M = fit_MQuad_mixtures(
        afm, n_top=n_top, path_=path_, nproc=nproc, minDP=minDP, minAD=minAD, with_M=True
    )
    _, _ = M.selectInformativeVariants(
        min_cells=minCell, out_dir=path_, tenx_cutoff=None,
        export_heatmap=False, export_mtx=False
    )
    idx = M.final_df.index.to_list()
    selected = [ afm.var_names[i] for i in idx ]
    afm = afm[:,selected].copy()
    afm.var['deltaBIC'] = M.final_df['deltaBIC']

    os.system(f'rm {os.path.join(path_, "*BIC*")}')

    return afm


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
    
    annotate_vars(afm, overwrite=True)
    test = (
        (afm.var['mean_cov']>min_site_cov) & \
        (afm.var['quality']>=min_var_quality) & \
        (afm.var['n0']>min_frac_negative*afm.shape[0]) & \
        (afm.var['Variant_CellN']>=min_n_positive) 
    )
    afm = afm[:,test].copy()

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

    t1 = (afm.X.A<low_confidence_af).sum(axis=0)/afm.shape[0] > min_prevalence_low_confidence_af
    t2 = (afm.X.A>high_confidence_af).sum(axis=0) > min_cells_high_confidence_af
    test = t1 & t2
    afm = afm[:,test].copy() 

    return afm


##


def filter_MI_TO(
    afm, 
    min_cov=10,
    min_var_quality=30,
    min_frac_negative=0.2,
    min_n_positive=5,
    af_confident_detection=.01,
    min_n_confidently_detected=2,
    min_mean_AD_in_positives=1.5,
    min_mean_DP_in_positives=25
    ):
    """
    MI_TO filter.
    """
    annotate_vars(afm, overwrite=True)
    afm.var['n_confidently_detected'] = np.sum(afm.X.A>=af_confident_detection, axis=0)

    test = (
        (afm.var['mean_cov']>=min_cov) & \
        (afm.var['quality']>=min_var_quality) & \
        (afm.var['n0']>=min_frac_negative*afm.shape[0]) & \
        (afm.var['Variant_CellN']>=min_n_positive) & \
        (afm.var['n_confidently_detected']>=min_n_confidently_detected) & \
        (afm.var['mean_AD_in_positives']>=min_mean_AD_in_positives) & \
        (afm.var['mean_DP_in_positives']>=min_mean_DP_in_positives) 
    )
    afm = afm[:,test].copy()

    return afm

    

##


def compute_lineage_biases(afm, lineage_column, target_lineage, bin_method='MI_TO', binarization_kwargs={}, alpha=.05):
    """
    Compute -log10(FDR) Fisher's exact test: lineage biases of some mutation.
    """

    if lineage_column not in afm.obs.columns:
        raise ValueError(f'{lineage_column} not present in cell metadata!')
        
    n = afm.shape[0]
    muts = afm.var_names

    target_ratio_array = np.zeros(muts.size)
    oddsratio_array = np.zeros(muts.size)
    pvals = np.zeros(muts.size)
    call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)
    G = afm.layers['bin'].A.copy()

    # Here we go
    for i in range(muts.size):

        test_mut = G[:,i] == 1
        test_lineage = afm.obs[lineage_column] == target_lineage
        mut_size = test_mut.sum()
        mut_lineage_size = (test_mut & test_lineage).sum()
        target_ratio = mut_lineage_size / mut_size
        target_ratio_array[i] = target_ratio
        other_mut_lineage_size = (~test_mut & test_lineage).sum()

        # Fisher
        oddsratio, pvalue = fisher_exact(
            [
                [mut_lineage_size, mut_size - mut_lineage_size],
                [other_mut_lineage_size, n - other_mut_lineage_size],
            ],
            alternative='greater',
        )
        oddsratio_array[i] = oddsratio
        pvals[i] = pvalue

    # Correct pvals --> FDR
    pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]

    # Results
    results = (
        pd.DataFrame({
            'perc_in_target_lineage' : target_ratio_array,
            'odds_ratio' : oddsratio_array,
            'FDR' : pvals,
            'lineage_bias' : -np.log10(pvals) 
        }, index=muts
        )
        .sort_values('lineage_bias', ascending=False)
    )

    return results


##


def filter_GT_enriched(afm, lineage_column=None, fdr_treshold=.1, n_enriched_groups=2, 
                       bin_method='MI_TO', binarization_kwargs={}):

    """
    Given an afm matrix with a <cov> columns in .obs wich correspond to 
    ground truth clones, filter cells and vars.
    """

    if lineage_column is not None and lineage_column in afm.obs.columns:
        pass
    else:
        raise ValueError(f'{lineage_column} not available in afm.obs!')
    
    L = []
    lineages = afm.obs[lineage_column].dropna().unique()
    for target_lineage in lineages:
        print(f'Computing variants enrichment for lineage {target_lineage}...')
        res = compute_lineage_biases(afm, lineage_column, target_lineage, 
                                    bin_method=bin_method, binarization_kwargs=binarization_kwargs)
        L.append(res['FDR']<=fdr_treshold)
    
    df_enrich = pd.concat(L, axis=1)
    df_enrich.columns = lineages
    test = df_enrich.apply(lambda x: np.sum(x>0)>0 and np.sum(x>0)<=n_enriched_groups, axis=1)
    vois = df_enrich.loc[test].index.unique()
    id_lineages = df_enrich.loc[test].sum(axis=0).loc[lambda x: x>0].index.to_list()
    cells = afm.obs[lineage_column].loc[lambda x: x.isin(id_lineages)].index
    afm = afm[cells, vois].copy()

    return afm 


##


# ############################### Deprecated filters
# 
# def filter_ludwig2019(afm, mean_af=0.5, min_var_quality=20):
#     """
#     Filter variants based on fixed tresholds adopted in Ludwig et al., 2019, 
#     in the experiment without ATAC-seq reference, Fig.7.
#     """
#     test_vars_het = np.mean(afm.X, axis=0) >= mean_af                               # high average AF variants
#     test_vars_qual = np.mean(afm.layers['quality'], axis=0) >= min_var_quality      # high average quality variants
#     test_vars = test_vars_het & test_vars_qual
#     filtered = afm[:, test_vars].copy()
#     filtered = filter_sites(filtered)
# 
#     return filtered
# 
# 
# def z_test_vars_into_bins(delta, alpha=0.05):
#     """
#     Perform a z-test on each deltaBIC value within a bin.
#     """
#     delta = delta.loc[lambda x: ~x.isna()]
#     z_scores = ( delta-delta.mean() ) / ( delta.std() / (delta.shape[0] ** 0.5) )
#     critical_z_value = stats.norm.ppf(1-alpha)
#     variants = z_scores.loc[lambda x: x>critical_z_value].index.to_list()
# 
#     return variants
#     
# 
# ##
# 
# 
# def calc_median_AF_in_positives(afm, candidates):
# 
#     median_af = []
#     for x in candidates:
#         idx = afm[:, x].X.toarray().flatten()>0
#         median_af.append(np.median(afm[idx, x].X, axis=0)[0])
# 
#     return np.array(median_af)
#     
# 
# ##
# 
# 
# def filter_from_bins(df, afm, min_f, max_f, treshold_AF=.05, n_bins=10):
#         
#     bins = np.linspace(min_f, max_f, n_bins+1)
#     df['bin'] = pd.cut(
#         df['fr_pos_cells'], bins=bins, include_lowest=True, labels=False
#     )
#     var_l = df.groupby('bin').apply(lambda x: z_test_vars_into_bins(x['deltaBIC']))
#     candidates = list(chain.from_iterable(var_l))
#     test = calc_median_AF_in_positives(afm, candidates)>=treshold_AF
#     filtered_candidates = pd.Series(candidates)[test].to_list()
#     
#     return filtered_candidates
# 
# 
# ##
# 
# 
# def filter_Mquad_optimized(afm, nproc=8, minDP=10, minAD=1, path_=None,
#     split_t=.2, treshold_AF_high=.05, treshold_AF_low=.025, n_bins=10
#     ):
#     """
#     Filter variants using the MQuad method, optimized version.
#     """
#     df, M, ad_vars = fit_MQuad_mixtures(
#         afm, path_=path_, nproc=nproc, minDP=minDP, minAD=minAD, with_M=True, 
#     )
# 
#     # Split into high and low
#     df['fr_pos_cells'] = df['num_cells_nonzero_AD'] / afm.shape[0]
#     df['fr_pos_cells'] = df['fr_pos_cells'].astype('float')
#     df['deltaBIC'] = df['deltaBIC'].astype('float')
# 
#     # Get vars
#     high_vars = filter_from_bins(
#         df.query('fr_pos_cells>=@split_t').copy(), 
#         afm, split_t, 1, treshold_AF=treshold_AF_high, n_bins=n_bins
#     )
#     low_vars = filter_from_bins(
#         df.query('fr_pos_cells<@split_t').copy(), 
#         afm, 0, split_t, treshold_AF=treshold_AF_low, n_bins=n_bins
#     )
# 
#     # Subset matrix
#     filtered = afm[:, high_vars+low_vars].copy()
#     filtered = filter_sites(filtered) # Remove sites
# 
#     return filtered
# 
# 
# ##
# 
# 
# def filter_DADApy(afm):
#     """
#     Filter using DADApy.
#     """
#     return 'Not implemented yet...'
# 
#
##
#
#  
# def filter_density(afm, density=0.1, t=.01, steps=np.Inf):
#     """
#     Jointly filter cells and variants based on the iterative filtering algorithm 
#     adopted by Moravec et al., 2022.
#     """
# 
#     # Check initial density not already above the target one
#     X_bool = np.where(afm.X>=t, 1, 0)
#     d0 = X_bool.sum() / X_bool.size
#     if d0 >= density:
#         logging.info(f'Density is already more than the desired target: {d0}')
#         return afm
#         
#     else:
#         print(f'Initial density: {d0}')
# 
#     # Iteratively remove lowest density rows/cols, until desired density is reached
#     densities = []
#     i = 0
#     while i < steps:
# 
#         rowsums = X_bool.sum(axis=1)
#         colsums = X_bool.sum(axis=0)
#         d = X_bool.sum() / X_bool.size
#         densities.append(d)
#         print(f'Step {i}: density {d:.2f}')
# 
#         if d >= density or (len(rowsums) == 0 or len(colsums) == 0):
#             break
# 
#         rowmin = rowsums.min()
#         colmin = colsums.min()
#         if rowmin <= colmin:
#             lowest_density_rows = np.where(rowsums == rowmin)[0]
#             X_bool = X_bool[ [ i for i in range(X_bool.shape[0]) if i not in lowest_density_rows ], :]
#             afm = afm[ [ i for i in range(afm.shape[0]) if i not in lowest_density_rows ], :].copy()
#         else:
#             lowest_density_cols = np.where(colsums == colmin)[0]
#             X_bool = X_bool[:, [ i for i in range(X_bool.shape[1]) if i not in lowest_density_cols ] ]
#             afm = afm[:, [ i for i in range(afm.shape[1]) if i not in lowest_density_cols ] ].copy()
#         i += 1
# 
#     afm = filter_sites(afm) 
#     
#     return afm
# 
# 
# ##
# 
# 
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
# 
# 
# ##
# 
# 
# def filter_seurat(afm, nbins=50, n_top=1000, log=True):
#     """
#     Filter with the scanpy/seurat flavour, readapted from scanpy.
#     """
#     # Calc stats
#     X = afm.X
#     X[np.isnan(X)] = 0
#     mean = X.mean(axis=0)
#     var = X.var(axis=0)
#     dispersion = np.full(X.shape[1], np.nan)
#     idx_valid = (mean > 0.0) & (var > 0.0)
#     dispersion[idx_valid] = var[idx_valid] / mean[idx_valid] 
# 
#     # Optional, put them in the log space
#     if log:
#         mean = np.log1p(mean) 
#         dispersion = np.log(dispersion)
#     
#     # Bin dispersion values, subtract from each value the bin dispersion mean and divide by the bin std
#     df = pd.DataFrame({"log_dispersion": dispersion, "bin": pd.cut(mean, bins=nbins)})
#     log_disp_groups = df.groupby("bin")["log_dispersion"]
#     log_disp_mean = log_disp_groups.mean()
#     log_disp_std = log_disp_groups.std(ddof=1)
#     log_disp_zscore = (
#         df["log_dispersion"].values - log_disp_mean.loc[df["bin"]].values
#     ) / log_disp_std.loc[df["bin"]].values
#     log_disp_zscore[np.isnan(log_disp_zscore)] = 0.0
# 
#     # Rank, order and slice first n. Subset AFM
#     hvf_rank = np.full(X.shape[1], -1, dtype=int)
#     ords = np.argsort(log_disp_zscore)[::-1]
#     hvf_rank[ords[:n_top]] = range(n_top)
#     select = np.where(hvf_rank != -1)[0]
#     filtered = afm[:, select].copy()
#     filtered = filter_sites(filtered) # Remove sites
# 
#     return filtered
# 
# 
# ##
# 
# 
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
# 
# 
# ##
# 
# 
# def filter_sites(afm):
#     """
#     Filter sites info belonging only to selected AFM variants.
#     """
#     cells = afm.obs_names
#     sites_retained = afm.var_names.map(lambda x: x.split('_')[0]).unique()
#     afm.uns['per_position_coverage'] = afm.uns['per_position_coverage'].loc[cells, sites_retained]
#     afm.uns['per_position_quality'] = afm.uns['per_position_quality'].loc[cells, sites_retained]
# 
#     return afm


###############################