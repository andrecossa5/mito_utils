"""
Module to create and process AFMs.
"""

from scipy.special import binom
from scipy.stats import fisher_exact
from statsmodels.sandbox.stats.multicomp import multipletests
from mito_utils.filters import *
from mito_utils.make_afm import *
from mito_utils.distances import *
from mito_utils.phylo import *


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



def read_one_sample(path_data, sample='MDA_clones', only_variants=True, with_GBC=False, nmads=3, mean_coverage=25):
    """
    Read and format one sample AFM. Path data should be folder with a <sample> subfolder, storing:
    * 'AFM.h5ad', the maegatk output produced by mito_preprocessing Nextflow pipeline
    * 'barcodes.txt', a list of good quality (expression QC) cell barcodes.
    *  (Optional) 'cells_summary_table.csv', a table of storing cell clone assignments (if lentivirally barcoded cells).
    """

    print(f'Create the full cell x MT-SNV Allele Frequency Matrix (AFM)...')

    # Read maegatk output
    A = sc.read(os.path.join(path_data, sample, 'AFM.h5ad'))
    barcodes = pd.read_csv(os.path.join(path_data, sample, 'barcodes.txt'), index_col=0, header=None)

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
    print(f'Valid cells: {len(valid_cbcs)}')
    cells = list(valid_cbcs)
    A = A[cells, :].copy()
    if with_GBC:
        cbc_gbc_df = cbc_gbc_df.loc[cells, :]

    # Clean UMI counts for good enough (average) Base-Calling quality
    A = clean_BC_quality(A)

    # Filter cells with too high or low mean coverage across MT-genome
    x = A.layers['cov'].A.mean(axis=1)
    median = np.median(x)
    MAD = np.median(np.abs(x-median))
    test = (x<=median+nmads*MAD) & (x>=mean_coverage)  # Test
    A = A[test,:].copy()                               # Filtering 
    print(f'Filtered cells (i.e., mean MT-genome coverage >={mean_coverage} and <={median+nmads*MAD:.2f}): {A.shape[0]}')

    # Format into a complete afm
    afm = format_matrix(
        A, 
        cbc_gbc_df=cbc_gbc_df if with_GBC else None, 
        only_variants=only_variants,
        with_GBC=with_GBC
    )
    afm.obs = afm.obs.assign(sample=sample)
    
    if with_GBC:
        afm.obs['GBC'] = afm.obs['GBC_Set']
        afm.obs = afm.obs.drop(columns=['GBC_Set'])

    return afm


##


def compute_metrics_raw(afm):
    """
    Compute raw dataset metrics.
    """
    d = {}
    # Mean n consensus MT UMIs per cell
    d['median_cell_cov'] = afm.uns['per_position_coverage'].mean(axis=1).median()
    # Mean n consensus MT UMIs per site
    d['median_site_cov'] = afm.uns['per_position_coverage'].mean(axis=0).median()
    # Compute spec/aspec median site coverage per cell
    test_sites = mask_mt_sites(afm)
    aspec = np.median(afm.uns['per_position_coverage'].loc[:,~test_sites].values)
    spec = np.median(afm.uns['per_position_coverage'].loc[:,test_sites].values)
    d['log10_specific_vs_aspecific_signal'] = np.log10(spec/(aspec+0.000001))
    # To df
    df = pd.Series(d).T.to_frame('value').reset_index().rename(columns={'index':'metric'})

    return df


##


def compute_metrics_filtered(a, spatial_metrics=True, t=.01):
    """
    Compute additional metrics on selected feature space.
    """

    # Binarize
    X_bin = np.where(a.X>=t,1,0)
    d = {}

    # n cells and vars
    d['n_cells'] = X_bin.shape[0]
    d['n_vars'] = X_bin.shape[1]
    # n cells per var and n vars per cell (mean, median, std)
    d['median_n_vars_per_cell'] = np.median(X_bin.sum(axis=1))
    d['mean_n_vars_per_cell'] = np.mean(X_bin.sum(axis=1))
    d['std_n_vars_per_cell'] = np.std(X_bin.sum(axis=1))
    d['mean_n_cells_per_var'] = np.mean(X_bin.sum(axis=0))
    d['median_n_cells_per_var'] = np.median(X_bin.sum(axis=0))
    d['std_n_cells_per_var'] = np.std(X_bin.sum(axis=0))
    # AFM sparseness and genotypes uniqueness
    d['density'] = X_bin.sum() / np.product(X_bin.shape)
    seqs = AFM_to_seqs(a)
    unique_genomes_occurrences = pd.Series(seqs).value_counts(normalize=True)
    d['genomes_redundancy'] = 1-(unique_genomes_occurrences.size / X_bin.shape[0])
    d['median_genome_prevalence'] = unique_genomes_occurrences.median()
    # Mutational spectra
    class_annot = a.var_names.map(lambda x: x.split('_')[1]).value_counts().astype('int')
    d = pd.concat([pd.Series(d), class_annot])

    # Spatial metrics
    if spatial_metrics:
        # Cell connectedness
        D = pairwise_distances(X_bin, metric=lambda x, y: np.sum(np.logical_and(x, y)))
        cell_conn = np.ma.masked_equal(D, np.diag(D)).mean(axis=1).data
        d['median_connectedness'] = np.median(cell_conn)
        d['mean_connectedness'] = np.mean(cell_conn)
        # Baseline tree internal nodes mutations support
        tree = build_tree(a, t=t)
        tree_collapsed = tree.copy()
        tree_collapsed.collapse_mutationless_edges(True)
        d['frac_supported_nodes'] = len(tree_collapsed.internal_nodes) / len(tree.internal_nodes)

    # to df
    df = pd.Series(d).T.to_frame('value').reset_index().rename(columns={'index':'metric'})

    return df


##


def compute_lineage_biases(a, lineage_column, target_lineage, t=.01):
    """
    Compute -log10(FDR) Fisher's exact test: lineage biases of some mutation.
    """

    n = a.shape[0]
    muts = a.var_names

    target_ratio_array = np.zeros(muts.size)
    oddsratio_array = np.zeros(muts.size)
    pvals = np.zeros(muts.size)

    # Here we go
    for i, mut in enumerate(muts):

        test_mut = a[:,mut].X.flatten() >= t 
        test_lineage = a.obs[lineage_column] == target_lineage

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
    pvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]

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


def filter_cells_and_vars(
    afm, filtering=None, min_cell_number=0, cells=None, variants=None, nproc=8, filtering_kwargs={},
    spatial_metrics=False, path_priors=None, lineage_column=None, fit_mixtures=False,
    ):
    """
    Filter cells and vars from an afm.
    """ 

    # General dataset QC
    print('Compute general dataset metrics...')
    dataset_df = compute_metrics_raw(afm)

    # Variants metrics
    print('Compute vars_df as in Weng et al., 2024')
    vars_df = make_vars_df(afm)

    print(f'Filter MT-SNVS from the full AFM...')
    if filtering in filtering_options:

        # n cells
        print(f'Feature selection method: {filtering}')
        print(f'Original AFM n cells: {afm.shape[0]} and {afm.shape[1]} MT-SNVs.')
        a_cells = afm.copy()
        n_cells = afm.shape[0]
        
        # Cells from clone with at least min_cell_number cells, if necessary
        if min_cell_number > 0:
            a_cells = filter_cell_clones(afm, min_cell_number=min_cell_number)
       
        # Filter MT-SNVs
        a_cells = filter_baseline(a_cells)
        if filtering == 'baseline':
            a = a_cells.copy()
        if filtering == 'CV':
            a = filter_CV(a_cells, **filtering_kwargs)
        elif filtering == 'seurat':
            a = filter_seurat(a_cells, **filtering_kwargs)
        elif filtering == 'ludwig2019':
            a = filter_ludwig2019(a_cells, **filtering_kwargs)
        elif filtering == 'miller2022':
            a = filter_miller2022(a_cells, **filtering_kwargs)
        elif filtering == 'weng2024':
            a = filter_weng2024(a_cells, **filtering_kwargs)
        elif filtering == 'MQuad':
            a = filter_Mquad(a_cells, nproc=nproc, **filtering_kwargs)
        elif filtering == 'MQuad_optimized':
            a = filter_Mquad_optimized(a_cells, nproc=nproc, **filtering_kwargs)
        elif filtering == 'DADApy':
            a = filter_DADApy(a_cells)
        elif filtering == 'density':
            a = filter_density(a_cells, **filtering_kwargs)

    # elif filtering == 'LINEAGE_prep':
    # 
    #     # n cells
    #     print(f'Feature selection method: {filtering}')
    #     print(f'Original AFM n cells: {afm.shape[0]}')
    #     a_cells = afm.copy()
    # 
    #     if min_cell_number > 0:
    #         n_cells = a_cells.shape[0]
    #         print(f'Filtering cells from clones with >{min_cell_number} cells')
    #         cell_counts = a_cells.obs.groupby('GBC').size()
    #         clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
    #         test = a_cells.obs['GBC'].isin(clones_to_retain)
    #         a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
    #         a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
    #         a_cells = a_cells[test, :].copy()
    #         print(f'Removed other {n_cells-a_cells.shape[0]} cells')
    #         print(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
    #     a = a_cells.copy()

    elif cells is None and variants is not None:

        # n cells
        print(f'Original AFM n cells: {afm.shape[0]} and {afm.shape[1]} MT-SNVs.')
        a_cells = afm.copy()
        # Cells from clone with at least min_cell_number cells, if necessary
        if min_cell_number > 0:
            a_cells = filter_cell_clones(afm, min_cell_number=min_cell_number)
        a = a_cells[:, variants].copy()
        a = filter_sites(a)

    elif cells is not None and variants is None:
        print(f'Original AFM n cells: {afm.shape[0]} and {afm.shape[1]} MT-SNVs.')
        a_cells = afm[cells, :].copy()
        a_cells = filter_sites(a_cells)
        a = a_cells.copy()

    elif cells is not None and variants is not None:
        print(f'Original AFM n cells: {afm.shape[0]} and {afm.shape[1]} MT-SNVs.')
        a_cells = afm[cells, variants].copy()
        a_cells = filter_sites(a_cells)
        a = a_cells.copy()
    
    else:
        raise ValueError(
                f'''The provided filtering method {filtering} is not supported.
                    Choose another one...'''
            )

    # Final dataset and filtered MT-SNVs metrics to evalutate the selected MT-SNVs space quality
    print(f'Filtered AFM contains {a.shape[0]} cells and {a.shape[1]} MT-SNVs.')
    if a.shape[1] == 0:
        raise ValueError('No variant selected! Change filtering method!!')
    
    # Dataset
    dataset_df = pd.concat([
        dataset_df, 
        compute_metrics_filtered(a, spatial_metrics=spatial_metrics)
    ])

    # Compute last metrics for filtered variants
    vars_df['filtered'] = vars_df.index.isin(a.var_names)
    filtered_vars_df = vars_df.loc[a.var_names]

    # Lineage bias
    if lineage_column is not None:
        lineages = a.obs[lineage_column].dropna().unique()
        for target_lineage in lineages:
            res = compute_lineage_biases(a, lineage_column, target_lineage)
            test = filtered_vars_df.index.isin(res.query('FDR<=0.1').index)
            filtered_vars_df[f'enriched_{target_lineage}'] = test

    # Bimodal mixture modelling deltaBIC (MQuad-like)
    if fit_mixtures:
        filtered_vars_df = (
            filtered_vars_df
            .join(
                fit_MQuad_mixtures(a)
                .dropna()
                [['deltaBIC']]
            )
        )

    # Add priors from external data sources
    priors = pd.read_csv(path_priors, index_col=0)
    vars_df['prior'] = priors.iloc[:,0]
    filtered_vars_df['prior'] = vars_df['prior'].loc[filtered_vars_df.index]

    # Add all filtered variants metadata to afm
    assert all(a.var_names == filtered_vars_df.index)
    a.var = filtered_vars_df

    return vars_df, dataset_df, a


##


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


def summary_stats_vars(afm, variants=None):
    """
    Calculate the most important summary stats for a bunch of variants, collected for
    a set of cells.
    """
    if variants is not None:
        test = afm.var_names.isin(variants)
        density = (~np.isnan(afm[:, test].X)).sum(axis=0) / afm.shape[0]
        median_vafs = np.nanmedian(afm[:, test].X, axis=0)
        mean_vafs = np.nanmean(afm[:, test].X, axis=0)
        var_vafs = np.nanvar(afm[:, test].X, axis=0)
        vmr_vafs = (var_vafs + 0.000001) / mean_vafs
        median_coverage_var = np.nanmedian(afm[:, test].layers['coverage'], axis=0)
        fr_positives = np.sum(afm[:, test].X > 0, axis=0) / afm.shape[0]
        var_names = afm.var_names[test]
    else:
        density = (~np.isnan(afm.X)).sum(axis=0) / afm.shape[0]
        median_vafs = np.nanmedian(afm.X, axis=0)
        mean_vafs = np.nanmean(afm.X, axis=0)
        var_vafs = np.nanvar(afm.X, axis=0)
        vmr_vafs = (var_vafs + 0.000001) / mean_vafs
        median_coverage_var = np.nanmedian(afm.layers['coverage'], axis=0)
        fr_positives = np.sum(afm.X > 0, axis=0) / afm.shape[0]
        var_names = afm.var_names

    df = pd.DataFrame(
        {   
            'density' : density,
            'median_coverage' : median_coverage_var,
            'median_AF' : median_vafs,
            'VMR_AF' : vmr_vafs,
            'fr_positives' : fr_positives,
        }, index=var_names
    )
    df['VMR_rank'] = df['VMR_AF'].rank(ascending=False).astype('int')

    return df


##


def filter_afm_with_gt(afm, t=.75, rest=.25):
    """
    Given an afm matrix with a <cov> columns in .obs wich correspond to 
    ground truth clones, filter cells and vars.
    """
    a_cells = filter_baseline(a_cells)
    gt_l = [
        rank_clone_variants(
            a_cells, var='GBC', group=g, rank_by='custom_perc_tresholds',
            min_clone_perc=t, max_perc_rest=rest
        ).assign(clone=g)
        for g in a_cells.obs['GBC'].unique()
    ]
    df_gt = pd.concat(gt_l).join(summary_stats_vars(a_cells))

    vois_df = (
        df_gt
        .query('n_cells_clone>=@min_cells_clone')
        .sort_values('log2_perc_ratio', ascending=False)
        .loc[:, 
            [
                'median_AF_clone', 'median_AF_rest', 'perc_clone', 
                'perc_rest', 'log2_perc_ratio', 'n_cells_clone', 'clone'
            ]
        ]
    )
    vois = vois_df.index.unique()
    cells = a_cells.obs['GBC'].loc[lambda x: x.isin(vois_df['clone'])].index
    d_metrics, a = filter_cells_and_vars(afm, cells=cells, variants=vois)

    return d_metrics, a


##