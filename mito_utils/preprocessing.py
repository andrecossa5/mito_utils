"""
Module to create and process AFMs.
"""

import scanpy as sc
from igraph import Graph
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


def make_AFM(path_afm, path_meta=None, cell_file=None, sample='MDA_clones', only_variants=True, cell_filter='filter1', 
                    nmads=5, mean_cov_all=25, median_cov_target=30, min_perc_covered_sites=.75, is_covered_treshold=10):
    """
    Read and format a complete Allele Frequency Matrix (follows the logic of Miller et al., 2022). 

    Args:
        path_afm (str): path to AFM.h5ad, the collection of cell x site feature table output of mito_preprocessing
        path_meta (str, optional): path to cells_meta.csv, cells_metadata for all valid CBs
        cell_file (str, optional): path <txt.> file containing desired subsets of valid CBs present in cells metadata.
        sample (str, optional): sample name. Defaults to 'MDA_clones'.
        only_variants (bool, optional): Return AF of all possible site-bases or only variants from the rRCS MT-reference. Defaults to True.
        cell_filter (str, optional): cell filter.
            1. **'filter1'**: Filter cells based on mean MT-genome coverage (all sites).
            2. **'filter2'**: Filter cells based on median target MT-sites coverage and min % of target sites covered.
        nmads (int, optional): n Minimum Absolute Deviations to filter cells with high MT-library UMI counts. Defaults to 5.
        mean_coverage (int, optional): minimum mean consensus (at least 3-supporting-reads) UMI coverage across MT-genome, per cell. Defaults to 25.
        median_cov_target (int, optional): minimum median UMI coverage at target MT-sites. Defaults to 30.
        min_perc_covered_sites (float, optional): minimum fraction of MT target sites covered. Defaults to .75.
        is_covered_treshold (int, optional): minimum n UMIs to consider a site covered. Default to 10.

    Returns:
        AnnData: The complete, annotated Allelic Frequency Matrix of the sample.
    """

    print(f'Create the full cell x MT-SNV Allele Frequency Matrix (AFM)...')

    # Read mito_preprocessing output and filter only good quality CBs 
    A = sc.read(path_afm)

    if path_meta is not None:
        meta = pd.read_csv(path_meta, index_col=0).query('sample==@sample')
        valid_cbcs = set(A.obs_names) & set(meta.index)    
    else:
        meta = None
        valid_cbcs = set(A.obs_names)
    
    if cell_file is not None: 
        valid_cbcs = set(valid_cbcs) & set(pd.read_csv(cell_file, header=None)[0]) 

    print(f'Valid CBs: {len(valid_cbcs)}')

    cells = list(valid_cbcs)
    A = A[cells,:].copy()
    A.obs = meta.loc[cells] if meta is not None else A.obs      # Add meta info
        
    if cell_filter == 'filter1':

        x = A.layers['cov'].A.mean(axis=1)       
        median = np.median(x)
        MAD = np.median(np.abs(x-median))
        test = (x<=median+nmads*MAD) & (x>=mean_cov_all)  
        A = A[test,:].copy()                                      

        print(f'Filtered cells (i.e., mean MT-genome coverage >={mean_cov_all} and <={median+nmads*MAD:.2f}): {A.shape[0]}')

    elif cell_filter == 'filter2':

        test_sites =  mask_mt_sites(A, raw_matrix=True)
        test1 = np.median(A.layers['cov'].A[:,test_sites], axis=1) >= median_cov_target
        test2 = np.sum(A.layers['cov'].A[:,test_sites]>=is_covered_treshold, axis=1) >= min_perc_covered_sites
        A = A[(test1) & (test2),:].copy()                                  

        print(f'Filtered cells (i.e., median target MT-genome coverage >={median_cov_target} and fraction covered sites >={min_perc_covered_sites}: {A.shape[0]}')

    # Format a complete AFM
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
    test_sites = mask_mt_sites(afm)
 
    # n consensus UMIs per target site per cell
    d['median_site_cov'] = afm.uns['per_position_coverage'].loc[:,test_sites].median(axis=0).median()
    # Compute spec/aspec median site coverage per cell
    aspec = np.median(afm.uns['per_position_coverage'].loc[:,~test_sites].values)
    spec = np.median(afm.uns['per_position_coverage'].loc[:,test_sites].values)
    d['log10_specific_vs_aspecific_signal'] = np.log10(spec/(aspec+0.000001))

    # To df
    df = pd.Series(d).T.to_frame('value').reset_index().rename(columns={'index':'metric'})

    return df


##


def compute_connectivity_metrics(X):
    """
    Calculate the connectivity metrics presented in Weng et al., 2024.
    """

    # Create connectivity graph
    A = np.dot(X, X.T)
    np.fill_diagonal(A, 0)
    A.diagonal()
    g = Graph.Adjacency((A>0).tolist(), mode='undirected')
    edges = g.get_edgelist()
    weights = [A[i][j] for i, j in edges]
    g.es['weight'] = weights

    # Calculate metrics
    average_degree = sum(g.degree()) / g.vcount()                       # avg_path_length
    if g.is_connected():
        average_path_length = g.average_path_length()
    else:
        largest_component = g.clusters().giant()
        average_path_length = largest_component.average_path_length()   # avg_path_length
    transitivity = g.transitivity_undirected()                          # transitivity
    components = g.clusters()
    largest_component_size = max(components.sizes())
    proportion_largest_component = largest_component_size / g.vcount()  # % cells in largest subgraph

    return average_degree, average_path_length, transitivity, proportion_largest_component


##


def compute_metrics_filtered(a, spatial_metrics=True, weights=None, bin_method='MI_TO', 
                            binarization_kwargs={}, tree_kwargs={}):
    """
    Compute additional metrics on selected MT-SNVs feature space.
    """

    # Binarize
    X_bin = call_genotypes(a=a, bin_method=bin_method, **binarization_kwargs)
    d = {}

    # n cells and vars
    d['n_cells'] = X_bin.shape[0]
    d['n_vars'] = X_bin.shape[1]
    # n cells per var and n vars per cell (mean, median, std)
    d['median_n_vars_per_cell'] = np.median((X_bin>0).sum(axis=1))
    d['mean_n_vars_per_cell'] = np.mean((X_bin>0).sum(axis=1))
    d['std_n_vars_per_cell'] = np.std((X_bin>0).sum(axis=1))
    d['mean_n_cells_per_var'] = np.mean((X_bin>0).sum(axis=0))
    d['median_n_cells_per_var'] = np.median((X_bin>0).sum(axis=0))
    d['std_n_cells_per_var'] = np.std((X_bin>0).sum(axis=0))
    # AFM sparseness and genotypes uniqueness
    d['density'] = (X_bin>0).sum() / np.product(X_bin.shape)
    seqs = AFM_to_seqs(a, bin_method=bin_method, binarization_kwargs=binarization_kwargs)
    unique_genomes_occurrences = pd.Series(seqs).value_counts(normalize=True)
    d['genomes_redundancy'] = 1-(unique_genomes_occurrences.size / X_bin.shape[0])
    d['median_genome_prevalence'] = unique_genomes_occurrences.median()
    # Mutational spectra
    class_annot = a.var_names.map(lambda x: x.split('_')[1]).value_counts().astype('int')
    class_annot.index = class_annot.index.map(lambda x: f'mut_class_{x}')
    n_transitions = class_annot.loc[class_annot.index.str.contains('|'.join(transitions))].sum()
    n_transversions = class_annot.loc[class_annot.index.str.contains('|'.join(transversions))].sum()
    # % lineage-biased mutations
    freq_lineage_biased_muts = (a.var.loc[:,a.var.columns.str.startswith('FDR')]<=.1).any(axis=1).sum() / a.shape[1]

    # Collect
    d = pd.concat([
        pd.Series(d), 
        class_annot,
        pd.Series({'transitions_vs_transversions_ratio':n_transitions/n_transversions}),
        pd.Series({'freq_lineage_biased_muts':freq_lineage_biased_muts}),
    ])

    # Spatial metrics
    if spatial_metrics:
        
        # Cell connectedness
        average_degree, average_path_length, transitivity, proportion_largest_component = compute_connectivity_metrics(X_bin)
        d['average_degree'] = average_degree
        d['average_path_length'] = average_path_length
        d['transitivity'] = transitivity
        d['proportion_largest_component'] = proportion_largest_component

        # Baseline tree internal nodes mutations support
        tree = build_tree(a=a, weights=weights, bin_method=bin_method, binarization_kwargs=binarization_kwargs, **tree_kwargs)
        tree_collapsed = tree.copy()
        tree_collapsed.collapse_mutationless_edges(True)
        d['frac_supported_nodes'] = len(tree_collapsed.internal_nodes) / len(tree.internal_nodes)

    # to df
    df = pd.Series(d).T.to_frame('value').reset_index().rename(columns={'index':'metric'})

    return df, coo_matrix(X_bin.T)


##


def filter_AFM(
    afm, 
    lineage_column=None, min_cell_number=0, cells=None,
    filtering='MI_TO', filtering_kwargs={}, max_AD_counts=1, variants=None,
    spatial_metrics=False, tree_kwargs={}, nproc=8,
    fit_mixtures=False, only_positive_deltaBIC=False,  
    path_priors=None, max_prior=1, path_dbSNP=None, path_REDIdb=None, 
    compute_enrichment=False, bin_method='vanilla', binarization_kwargs={}, 
    return_X_bin=False
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
            - Fraction of negative cells >= 0.2 (`min_frac_negative`: `0.2`)
            - Number of positive cells ≥ 3 (`min_n_positive`: `3`)
            - Enough evidence of high-AF detection events (`af_confident_detection`: `0.05`, `min_n_confidently_detected`: `2`)
            - Minimum median AF in positive cells > 0.01 (`min_median_af`: `0.01`)

        11. **'GT_enriched'**: Filter with availabe lineage cell annotations (i.e., Ground Truth lentiviral clones.). The AF matrix is binarized, and each variant-lineage enrichment is tested. If a variant is significantly enriched in less than `n_enriched_groups` lineages (assumed independent), is retained.
            - afm.obs column for lineage annotation ('lineage_column': None)
            - FDR threshold for Fisher's Exact test ('fdr_treshold' : .1) 
            - Max number of lineages the MT-SNV can be enriched for ('n_enriched_groups' : 2) 
            - Binarization strategy ('bin_method' : 'MI_TO')
            - **kwargs for bianrization ('binarization_kwargs' : {})

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
    bin_method : str, oprional 
        Binarization strategy used for i) compute the final dataset statistics (i.e., mutation number, connectivity ecc) and ii) lineage enrichment. Default is `MI_TO`
    binarization_kwargs : dict, optional
        Binarization strategy **kwargs (see mito_utils.distances.call_genotypes). Default is `{}`
    return_X_bin : bool, optional
        Return the binarized and filtered AF matrix. Default is False

    Returns
    -------
    afm_filtered : AnnData
        Filtered Allelic Frequency Matrix.
    dataset_stats : pandas.DataFrame
        Reporting stats of the filtered dataset.
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
        print(f'AFM after baseline filters n cells: {a_cells.shape[0]} and {a_cells.shape[1]} MT-SNVs.')

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
        # elif filtering == 'DADApy':
        #     a = filter_DADApy(a_cells)
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

    # Filter cells with at leaast one muts above af_confident_detection
    a = filter_cells_with_at_least_one(a, bin_method=bin_method, binarization_kwargs=binarization_kwargs)

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
        a = filter_sites(a)
    if 'prior' in a.var.columns and max_prior<1:
        a = a[:,a.var['prior']<max_prior].copy()
        a = filter_sites(a)
    if max_AD_counts>1:
        AD, _, _ = get_AD_DP(a)
        test_max_ad_alleles = np.max(AD.A.T, axis=0)>=max_AD_counts
        a = a[:,test_max_ad_alleles].copy()
        a = filter_sites(a)

    # Final fixes
    a = filter_cells_with_at_least_one(a, bin_method=bin_method, binarization_kwargs=binarization_kwargs)
    a = filter_sites(a)
    print(f'Last filters: filtered AFM contains {a.shape[0]} cells and {a.shape[1]} MT-SNVs.')
    
    # Lineage bias
    if lineage_column in a.obs.columns and compute_enrichment:
        lineages = a.obs[lineage_column].dropna().unique()
        for target_lineage in lineages:
            res = compute_lineage_biases(a, lineage_column, target_lineage, 
                                        bin_method=bin_method, binarization_kwargs=binarization_kwargs)
            a.var[f'FDR_{target_lineage}'] = res['FDR']
            a.var[f'odds_ratio_{target_lineage}'] = res['odds_ratio']

    # Last dataset stats 
    final_metrics, X_bin = compute_metrics_filtered(
        a, spatial_metrics=spatial_metrics, 
        weights=1-a.var['prior'].values if 'prior' in a.var.columns else None, 
        tree_kwargs=tree_kwargs,
        bin_method=bin_method, 
        binarization_kwargs=binarization_kwargs
    )
    dataset_df = pd.concat([dataset_df,final_metrics])
    
    if return_X_bin:
        return a, dataset_df, X_bin
    else:
        return a, dataset_df


##


