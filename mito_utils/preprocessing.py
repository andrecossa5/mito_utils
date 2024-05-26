"""
Module to create and process AFMs.
"""

from mito_utils.filters import *
from mito_utils.make_afm import *
from mito_utils.distances import *
from mito_utils.phylo import *


##


##

patterns = [ 'A>C', 'T>G', 'A>T', 'A>G', 'G>A', 'C>G', 'C>A', 'T>A', 'G>C', 'G>T', 'N>T', 'C>T', 'T>C' ]
transitions = [pattern for pattern in patterns if pattern in ['A>G', 'G>A', 'C>T', 'T>C']]
transversions = [pattern for pattern in patterns if pattern not in transitions and 'N' not in pattern]

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



def read_one_sample(path_data, sample='MDA_clones', cell_file='barcodes.txt', only_variants=True, with_GBC=False, nmads=3, mean_coverage=25):
    """
    Read and format one sample AFM. Path data should be folder with a <sample> subfolder, storing:
    * 'AFM.h5ad', the maegatk output produced by mito_preprocessing Nextflow pipeline
    * 'barcodes.txt', a list of good quality (expression QC) cell barcodes.
    *  (Optional) 'cells_summary_table.csv', a table of storing cell clone assignments (if lentivirally barcoded cells).
    """

    print(f'Create the full cell x MT-SNV Allele Frequency Matrix (AFM)...')

    # Read maegatk output
    A = sc.read(os.path.join(path_data, sample, 'AFM.h5ad'))
    barcodes = pd.read_csv(os.path.join(path_data, sample, cell_file), index_col=0, header=None)

    # Filter cell barcodes
    valid_cbcs = set(A.obs_names) & set(barcodes.index)

    # GBC info
    if with_GBC:
        cbc_gbc_df = pd.read_csv(
            os.path.join(path_data, sample, 'cells_summary_table.csv'), 
            index_col=0
        )
        valid_cbcs = set(valid_cbcs) & set(cbc_gbc_df.index)
    
    # Subset
    print(f'Valid cells: {len(valid_cbcs)}')
    cells = list(valid_cbcs)
    A = A[cells, :].copy()

    # Clean UMI counts for good enough (average) Base-Calling quality
    A = clean_BC_quality(A)

    # Filter cells with too high or low mean coverage across MT-genome
    x = A.layers['cov'].A.mean(axis=1)
    median = np.median(x)
    MAD = np.median(np.abs(x-median))
    test = (x<=median+nmads*MAD) & (x>=mean_coverage)  # Test
    A = A[test,:].copy()                               # Filtering 
    if with_GBC:
        cbc_gbc_df = cbc_gbc_df.loc[A.obs_names, :]
        
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


def compute_metrics_filtered(a, spatial_metrics=True, weights=None, tree_kwargs={}):
    """
    Compute additional metrics on selected feature space.
    """

    # Binarize
    t = .05 if 't' not in tree_kwargs else tree_kwargs['t']
    X_bin = np.where(a.X>=t,1,0).astype(np.int8)
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
    class_annot.index = class_annot.index.map(lambda x: f'mut_class_{x}')
    n_transitions = class_annot.loc[class_annot.index.str.contains('|'.join(transitions))].sum()
    n_transversions = class_annot.loc[class_annot.index.str.contains('|'.join(transversions))].sum()
    # % lineage-biased mutations
    perc_lineage_biased_df = a.var.loc[:,a.var.columns.str.startswith('enrich')].any(axis=1).sum() / a.var.shape[0]
    # Collect
    d = pd.concat([
        pd.Series(d), 
        class_annot,
        pd.Series({'transitions_vs_transversions_ratio':n_transitions/n_transversions}),
        pd.Series({'perc_lineage_biased_muts':perc_lineage_biased_df}),
    ])

    # Spatial metrics
    if spatial_metrics:
        # Cell connectedness
        D = pairwise_distances(X_bin, metric=lambda x, y: np.sum(np.logical_and(x, y)))
        n_shared_muts = np.ma.masked_equal(D, np.diag(D)).mean(axis=1).data
        cell_conn = (D>0).sum(axis=1)-1
        d['median_n_shared_muts'] = np.median(n_shared_muts)
        d['median_connectedness'] = np.median(cell_conn)
        # Baseline tree internal nodes mutations support
        tree = build_tree(a, weights=weights, **tree_kwargs)
        tree_collapsed = tree.copy()
        tree_collapsed.collapse_mutationless_edges(True)
        d['frac_supported_nodes'] = len(tree_collapsed.internal_nodes) / len(tree.internal_nodes)

    # to df
    df = pd.Series(d).T.to_frame('value').reset_index().rename(columns={'index':'metric'})

    return df


##


def filter_cells_and_vars(
    afm, sample_name=None,
    filtering=None, min_cell_number=0, cells=None, variants=None, nproc=8, 
    max_AD_counts=1,
    af_confident_detection=.05,
    filtering_kwargs={}, tree_kwargs={},
    spatial_metrics=False, 
    path_priors=None, max_prior=1,
    fit_mixtures=False, only_positive_deltaBIC=False,
    lineage_column=None, with_clones_df=False, 
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
            a = filter_Mquad(a_cells, nproc=nproc, path_=os.getcwd(), **filtering_kwargs)
        elif filtering == 'MQuad_optimized':
            a = filter_Mquad_optimized(a_cells, nproc=nproc, **filtering_kwargs)
        elif filtering == 'DADApy':
            a = filter_DADApy(a_cells)
        elif filtering == 'density':
            a = filter_density(a_cells, **filtering_kwargs)
        elif filtering == 'MI_TO':
            a = filter_MI_TO(a_cells, **filtering_kwargs)
        elif filtering == 'GT_stringent':
            a, clones_df = filter_GT_stringent(a_cells, lineage_column=lineage_column, **filtering_kwargs)
        elif filtering == 'GT_enriched':
            a, clones_df = filter_GT_enriched(a_cells, lineage_column=lineage_column, **filtering_kwargs)

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
    
    # Retrieve af_confident_detection treshold, if present, else use default 0.01.
    if 'af_confident_detection' in filtering_kwargs:
        af_confident_detection = filtering_kwargs['af_confident_detection']
    elif 't' in tree_kwargs:
        af_confident_detection = tree_kwargs['t']

    # Filter cells with at leaast one muts above af_confident_detection
    a = filter_cells_with_at_least_one(a, t=af_confident_detection)

    # Final dataset and filtered MT-SNVs metrics to evalutate the selected MT-SNVs space quality
    print(f'Filtered AFM contains {a.shape[0]} cells and {a.shape[1]} MT-SNVs.')
    if a.shape[1] == 0:
        raise ValueError('No variant selected! Change filtering method!!')

    # Compute last metrics for filtered variants
    vars_df['filtered'] = vars_df.index.isin(a.var_names)
    filtered_vars_df = vars_df.loc[a.var_names]

    # Add priors from external data sources
    with_priors = path_priors is not None and sample_name is not None
    if with_priors:
        priors = pd.read_csv(path_priors, index_col=0)
        vars_df['prior'] = priors.loc[:,priors.columns!=sample_name].mean(axis=1)
        filtered_vars_df['prior'] = vars_df['prior'].loc[filtered_vars_df.index]

    # Lineage bias
    if lineage_column is not None and filtering != 'GT_enriched':
        lineages = a.obs[lineage_column].dropna().unique()
        for target_lineage in lineages:
            res = compute_lineage_biases(a, lineage_column, target_lineage, t=af_confident_detection)
            filtered_vars_df[f'lineage_bias_{target_lineage}'] = res['lineage_bias']
            test = filtered_vars_df.index.isin(res.query('FDR<=0.1').index)
            filtered_vars_df[f'enriched_{target_lineage}'] = test

    # Bimodal mixture modelling deltaBIC (MQuad-like)
    if fit_mixtures:
        filtered_vars_df = filtered_vars_df.join(fit_MQuad_mixtures(a).dropna()[['deltaBIC']])

    # Add all filtered variants metadata to afm
    assert all(a.var_names == filtered_vars_df.index)
    a.var = filtered_vars_df

    # Last (optional filters):
    if fit_mixtures and only_positive_deltaBIC:
        a = a[:,a.var['deltaBIC']>0].copy()
    if with_priors and max_prior<1:
        a = a[:,a.var['prior']<max_prior].copy()
    if max_AD_counts>1:
        a = filter_sites(a)
        AD, _, _ = get_AD_DP(a)
        test_max_ad_alleles = np.max(AD.A.T, axis=0)>=max_AD_counts
        a = a[:,test_max_ad_alleles].copy()

    # Final fixes
    a = filter_cells_with_at_least_one(a, t=af_confident_detection)
    a = filter_sites(a)
    print(f'Last filters: filtered AFM contains {a.shape[0]} cells and {a.shape[1]} MT-SNVs.')
    
    # Last dataset stats
    dataset_df = pd.concat([
        dataset_df, 
        compute_metrics_filtered(
            a, spatial_metrics=spatial_metrics, 
            weights=1-a.var['prior'].values if with_priors else None, 
            tree_kwargs=tree_kwargs
        )
    ])

    if filtering in ['GT_stringent', 'GT_enriched'] and with_clones_df:
        return dataset_df, a, clones_df
    else:
        return dataset_df, a


##


