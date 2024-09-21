"""
Module to create and process AFMs.
"""

import scanpy as sc
from mito_utils.filters import *
from mito_utils.make_afm import *
from mito_utils.distances import *
from mito_utils.phylo import *


##


##------------------------------------------------------------------------------------------------------##
patterns = [ 'A>C', 'T>G', 'A>T', 'A>G', 'G>A', 'C>G', 'C>A', 'T>A', 'G>C', 'G>T', 'N>T', 'C>T', 'T>C' ]
transitions = [pattern for pattern in patterns if pattern in ['A>G', 'G>A', 'C>T', 'T>C']]
transversions = [pattern for pattern in patterns if pattern not in transitions and 'N' not in pattern]
##------------------------------------------------------------------------------------------------------##


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


def read_one_sample(path_afm, path_meta, sample='MDA_clones', only_variants=True, nmads=5, mean_coverage=25):
    """
    Read and format a complete Allele Frequency Matrix (follows the logic of Miller et al., 2022). 

    Args:
        path_afm (str): path to AFM.h5ad, the collection of cell x site feature table output of mito_preprocessing
        path_meta (str): path to cells_meta.csv, cells_metadata for all valid CBs
        sample (str, optional): sample name. Defaults to 'MDA_clones'.
        only_variants (bool, optional): Return AF of all possible site-bases or only variants from the rRCS MT-reference. Defaults to True.
        nmads (int, optional): n Minimum Absolute Deviations to filter cells with high MT-library UMI counts. Defaults to 5.
        mean_coverage (int, optional): minimum mean consensus (at least 3-supporting-reads) UMI coverage across MT-genome, per cell. Defaults to 25.

    Returns:
        AnnData: The complete, annotated Allelic Frequency Matrix of the sample.
    """

    print(f'Create the full cell x MT-SNV Allele Frequency Matrix (AFM)...')

    # Read mito_preprocessing output and filter only good quality CBs 
    A = sc.read(path_afm)
    meta = pd.read_csv(path_meta, index_col=0).query('sample==@sample')
    valid_cbcs = set(A.obs_names) & set(meta.index)    # Filter cells with good transcriptional profile (and possibly clonal annotation)

    print(f'Valid CBs: {len(valid_cbcs)}')

    cells = list(valid_cbcs)
    A = A[cells,:].copy()
    A.obs = meta.loc[cells]                            # Add meta info
    x = A.layers['cov'].A.mean(axis=1)                 # Mean MT genome coverage 
    median = np.median(x)
    MAD = np.median(np.abs(x-median))
    test = (x<=median+nmads*MAD) & (x>=mean_coverage)  
    A = A[test,:].copy()                               # Filter cells with too high or low mean coverage across MT-genome
        
    print(f'Filtered cells (i.e., mean MT-genome coverage >={mean_coverage} and <={median+nmads*MAD:.2f}): {A.shape[0]}')

    # Format into a complete afm
    afm = format_matrix(A, only_variants=only_variants)
    afm.obs = afm.obs.assign(
        sample=sample, 
        nUMIs_MAESTER=afm.layers['coverage'].sum(axis=1),
        mean_nUMIs_MAESTER=afm.layers['coverage'].mean(axis=1),
    )

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
    afm, 
    lineage_column=None, min_cell_number=0, cells=None,
    filtering='MI_TO', filtering_kwargs={}, max_AD_counts=1, af_confident_detection=.01, variants=None,
    spatial_metrics=False, tree_kwargs={}, nproc=8,
    fit_mixtures=False, only_positive_deltaBIC=False,  
    path_priors=None, max_prior=1, path_dbSNP=None, path_REDIdb=None, compute_enrichment=False
    ):
    """
    Filter an Allele Frequency Matrix (AFM) for downstream analysis.
    This function implements different strategies to subset the detected cells and MT-SNVs (mitochondrial single nucleotide variants) to those that exhibit
    optimal properties for single-cell lineage tracing (scLT). The user can tune filtering method defaults via the `filtering_kwargs` argument. 
    Pre-computed sets of cells and variants can be selected without relying on any specific method (the function ensures integrity of the AFM `AnnData` object after subsetting).

    Parameters
    ----------
    afm : AnnData
        The AFM to subset. The AFM should be an `AnnData` object with slots as in the `read_one_sample` output.
    lineage_column : str, optional
        Categorical label used with `min_cell_number` and `compute_enrichment` arguments. Cells from lineages with fewer than `min_cell_number` cells are discarded. Default is `None`.
    min_cell_number : int, optional
        Cell filter. Minimum number of cells required for a certain lineage to be included in the final data. Default is `0`.
    cells : list, optional
        Pre-computed list of cells to subset. Default is `None`.
    filtering : str, optional
        Filtering method to use. Default is `'MI_TO'`.

        The following filtering strategies are implemented (tunable filtering kwargs that can be passed with the `filtering_kwargs` argument are highlighted as `{"kwarg": default_value}`):

        1. **'baseline'**: Baseline filter to retain only MT-SNV candidates showing minimal conditions to not be considered technical artifacts or sequencing errors. Filters MT-SNVs with:
            - Position in MT-gene bodies (`only_genes`: `True`)
            - Mean site coverage ≥ 5 (`min_site_cov`: `5`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - Number of positive cells ≥ 2 (`min_n_positive`: `2`)
            This filter is applied before every other one to exclude "definitely garbage" MT-SNV candidates.

        2. **'CV'**: Filters the top `n_top` MT-SNVs by Coefficient of Variation (CV). (`n_top`: `100`)

        3. **'seurat'**: Adapts the highly variable genes (HVG) selection procedure in `scanpy.pp.highly_variable_genes` with the 'seurat' flavor. (`n_top`: `1000`)

        4. **'ludwig2019'**: Filter adapted from Ludwig et al., 2019, experiment without ATAC-seq reference, Fig. 7. Filters MT-SNVs with:
            - Mean allele frequency (AF) ≥ 0.5 (`mean_af`: `0.5`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 20 (`min_var_quality`: `20`)

        5. **'miller2022'**: Filter adapted from Miller et al., 2022. Filters MT-SNVs with:
            - Mean site coverage ≥ 100 (`min_site_cov`: `100`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - 1st percentile AF value ≤ 0.01 (`perc_1`: `0.01`)
            - 99th percentile AF value ≥ 0.1 (`perc_99`: `0.1`)

        6. **'weng2024'**: Filter adapted from Weng et al., 2024 MAESTER data analysis. Filters MT-SNVs with:
            - Mean site coverage ≥ 5 (`min_site_cov`: `5`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - Fraction of negative cells (i.e., 0 ALT UMIs) ≥ 0.9 (`min_frac_negative`: `0.9`)
            - Number of positive cells ≥ 2 (`min_n_positive`: `2`)
            - Enough prevalence of minimal detection (`low_confidence_af`: `0.1`, `min_prevalence_low_confidence_af`: `0.1`)
            - Enough evidence of high-AF detection events (`high_confidence_af`: `0.5`, `min_cells_high_confidence_af`: `2`)

        7. **'MQuad'**: Filter from Kwock et al., 2022.

        8. **'MQuad_optimized'**: Filter from Kwock et al., 2022, with some twists to handle high- and low-prevalence MT-SNVs separately (`split_t`: `0.2`)

        9. **'density'**: Removes rows and columns recursively until a desired density is matched (`density`: `0.1`, `t`: `0.01`)

        10. **'MI_TO'**: Default filter, integrating aspects of 'miller2022' and 'weng2024'. Filters MT-SNVs with:
            - Mean site coverage ≥ 10 (`min_site_cov`: `10`)
            - Mean MT-SNV consensus UMI base sequencing quality ≥ 30 (`min_var_quality`: `30`)
            - Fraction of negative cells ≤ 0.2 (`min_frac_negative`: `0.2`)
            - Number of positive cells ≥ 5 (`min_n_positive`: `5`)
            - Enough evidence of high-AF detection events (`af_confident_detection`: `0.05`, `min_n_confidently_detected`: `3`)
            - Minimum median AF in positive cells > 0.01 (`min_median_af`: `0.01`)

        11. **'GT_stringent'**: [Description missing]

        12. **'GT_enriched'**: [Description missing]

    filtering_kwargs : dict, optional
        Keyword arguments for the selected filtering method. Default is `{}`.
    max_AD_counts : int, optional
        Site/variant filter. The minimum number of consensus UMIs supporting the MT-SNV ALT allele required for at least one cell across the dataset. Default is `1` (i.e., no filter).
    af_confident_detection : float, optional
        Cell filter. The minimum AF threshold at which at least one of the filtered MT-SNVs needs to be detected in a cell to retain the cell in the final dataset. It may be passed as a key-value pair in `filtering_kwargs`. Default is `0.01`.
    variants : list, optional
        Pre-computed list of variants to subset. Default is `None`.
    spatial_metrics : bool, optional
        If `True`, compute a list of "spatial" metrics for retained MT-SNVs, including [details missing]. Default is `False`.
    tree_kwargs : dict, optional
        Tree building keyword arguments that can be passed to `mito_utils.phylo.build_tree` when `spatial_metrics=True`. Default is `{}`.
    nproc : int, optional
        Number of cores to use by `mito_utils.phylo.build_tree` and `sklearn.metrics.pairwise_distances` when `spatial_metrics=True`. Default is `8`.
    fit_mixtures : bool, optional
        If `True`, fit MQuad (Kwock et al., 2022) binomial mixtures and calculate each variant's (passing baseline filters) delta BIC. Default is `False`.
    only_positive_deltaBIC : bool, optional
        Site/variant filter. Irrespective of the filtering strategy, retain only variants with positive delta BIC (estimated with MQuad). Default is `False`.
    path_priors : str, optional
        Path to a `.csv` file `pos,sample1,sample2,...,sampleN`, storing the mean (across sample cells) AF of each possible MT-SNV (3 x 16,569, rCRS MT-genome sequence). This table is used to compute AF priors (i.e., average AF of an MT-SNV across samples) for each MT-SNV. Default is `None`.
    max_prior : float, optional
        Site/variant filter. Threshold on the maximum AF prior value (calculated from `path_priors`) allowed for a bona fide somatic MT-SNV. Default is `1` (i.e., no filter).
    path_dbSNP : str, optional
        Path to a `.txt` tab-separated file with MT-SNVs "COMMON" variants from the dbSNP database. Required fields: `'pos'`, `'ALT'`, `'REF'`. All of these variants will be discarded from the final dataset. Can be found at `<path_main from zenodo folder>/data/MI_TO_bench/miscellanea`. Default is `None`.
    path_REDIdb : str, optional
        Path to a `.txt` tab-separated file with common MT-RNA edits from the REDIdb database. Required fields: `'Position'`, `'Ref'`, `'Ed'`, `'nSamples'`. All of these RNA-editing sites will be discarded from the final dataset. Can be found at `<path_main from zenodo folder>/data/MI_TO_bench/miscellanea`. Default is `None`.
    compute_enrichment : bool, optional
        If `True`, compute MT-SNV enrichment in individual lineages from `lineage_column`. Default is `False`.

    Returns
    -------
    afm_filtered : AnnData
        Filtered Allelic Frequency Matrix.
    additional_info : tuple
        A tuple containing:
            - dataset_df : pandas.DataFrame
                Reporting stats of the filtered dataset.
            - clones_df : pandas.DataFrame (if `compute_enrichment=True`)
                Reporting stats of individual lineages (i.e., categories from `lineage_column`) MT-SNV enrichment.
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
        if min_cell_number > 0 and lineage_column is not None:
            a_cells = filter_cell_clones(afm, cat_label=lineage_column, min_cell_number=min_cell_number)
       
        # Filter MT-SNVs, baseline
        a_cells = filter_baseline(a_cells)
        var_sites = a_cells.var_names.map(lambda x: x.split('_')[0])
        test = var_sites.value_counts()[var_sites]==1
        a_cells = a_cells[:,a_cells.var_names[test]].copy()
        a_cells = filter_sites(a_cells)

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
            a = filter_GT_stringent(a_cells, lineage_column=lineage_column, **filtering_kwargs)
        elif filtering == 'GT_enriched':
            a = filter_GT_enriched(a_cells, lineage_column=lineage_column, **filtering_kwargs)

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
    
    # Filter common SNVs and possible RNA-edits
    if path_dbSNP is not None:
        if os.path.exists(path_dbSNP):
            common = pd.read_csv(path_dbSNP, index_col=0, sep='\t')
            common = common['pos'].astype('str') + '_' + common['REF'] + '>' + common['ALT'].map(lambda x: x.split('|')[0])
            common = common.to_list()
            variants = a.var_names[~a.var_names.isin(common)]
            a = a[:,variants].copy()
            a = filter_sites(a)
    if path_REDIdb is not None:
        if os.path.exists(path_REDIdb):
            edits = pd.read_csv(path_REDIdb, index_col=0, sep='\t')
            edits = edits.query('nSamples>100')
            edits = edits['Position'].astype('str') + '_' + edits['Ref'] + '>' + edits['Ed']
            edits = edits.to_list()
            variants = a.var_names[~a.var_names.isin(edits)]
            a = a[:,variants].copy()
            a = filter_sites(a)
    
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
    if path_priors is not None:
        if os.path.exists(path_priors) and 'sample' in a.obs:
            sample_name = a.obs['sample'].unique()[0]
            priors = pd.read_csv(path_priors, index_col=0)
            vars_df['prior'] = priors.loc[:,priors.columns!=sample_name].mean(axis=1)
            filtered_vars_df['prior'] = vars_df['prior'].loc[filtered_vars_df.index]

    # Bimodal mixture modelling deltaBIC (MQuad-like)
    if fit_mixtures:
        filtered_vars_df = filtered_vars_df.join(fit_MQuad_mixtures(a).dropna()[['deltaBIC']])

    # Add all filtered variants metadata to afm
    assert all(a.var_names == filtered_vars_df.index)
    a.var = filtered_vars_df

    # Last (optional filters):
    if fit_mixtures and only_positive_deltaBIC:
        a = a[:,a.var['deltaBIC']>0].copy()
    if 'prior' in a.var.columns and max_prior<1:
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
    
    # Lineage bias
    if lineage_column in a.obs.columns and compute_enrichment:
        lineages = a.obs[lineage_column].dropna().unique()
        for target_lineage in lineages:
            res = compute_lineage_biases(a, lineage_column, target_lineage, t=af_confident_detection)
            a.var[f'FDR_{target_lineage}'] = res['FDR']
            a.var[f'odds_ratio_{target_lineage}'] = res['odds_ratio']

    # Last dataset stats
    dataset_df = pd.concat([
        dataset_df, 
        compute_metrics_filtered(
            a, spatial_metrics=spatial_metrics, 
            weights=1-a.var['prior'].values if 'prior' in a.var.columns else None, 
            tree_kwargs=tree_kwargs
        )
    ])
    
    return a, dataset_df


##


