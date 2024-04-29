"""
Module to create and process AFMs.
"""

from mito_utils.filters import *
from mito_utils.make_afm import *


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



def read_one_sample(path_data, sample='MDA_clones', only_variants=True, with_GBC=True, nmads=3, mean_coverage=25):
    """
    Read and format one sample AFM. Path data should be folder with a <sample> subfolder, storing:
    * 'AFM.h5ad', the maegatk output produced by mito_preprocessing Nextflow pipeline
    * 'barcodes.txt', a list of good quality (expression QC) cell barcodes.
    *  (Optional) 'cells_summary_table.csv', a table of storing cell clone assignments (if lentivirally barcoded cells).
    """

    logging.info(f'Filter MT-SNVS from the full AFM...')

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
    logging.info(f'Valid cells: {len(valid_cbcs)}')
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
    logging.info(f'Cells with MT-genome coverage >={mean_coverage} and <={median+nmads*MAD}: {A.shape[0]}')

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
    

def filter_cells_and_vars(
    afm, cells=None, variants=None, filtering=None, min_cell_number=0,  nproc=8, 
    path_=None, n=1000, prefilter_MQuad=False, **kwargs
    ):
    """
    Filter cells and vars from an afm.
    """ 
    logging.info(f'Filter MT-SNVS from the full  AFM...')

    if filtering in filtering_options and filtering != 'density':

        # n cells
        logging.info(f'Feature selection method: {filtering}')
        logging.info(f'Original AFM n cells: {afm.shape[0]}')
        a_cells = afm.copy()

        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logging.info(f'Filtering cells from clones with >={min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>=min_cell_number].index 
            test = a_cells.obs['GBC'].isin(clones_to_retain)
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
            a_cells = a_cells[test, :].copy()
            logging.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logging.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
       
        # Variants
        a_cells = filter_baseline(a_cells)
        if filtering == 'CV':
            a = filter_CV(a_cells, n=100)
        elif filtering == 'ludwig2019':
            a = filter_ludwig2019(a_cells)
        elif filtering == 'miller2022':
            a = filter_miller2022(a_cells)
        elif filtering == 'weng2024':
            vars_df = make_vars_df(a_cells)
            a = filter_weng2024(a_cells, vars_df)
        elif filtering == 'seurat':
            a = filter_seurat(a_cells, n=n)
        elif filtering == 'pegasus':
            a = filter_pegasus(a_cells, n=n)
        elif filtering == 'MQuad':
            n = None if not prefilter_MQuad else n
            a = filter_Mquad(a_cells, nproc=nproc, path_=path_, n=n)
        elif filtering == 'MQuad_optimized':
            a = filter_Mquad_optimized(a_cells, nproc=nproc, path_=path_, **kwargs)
        elif filtering == 'DADApy':
            a = filter_DADApy(a_cells)

    elif filtering == 'density':
        
        # n cells
        logging.info(f'Feature selection method: {filtering}')
        logging.info(f'Original AFM n cells: {afm.shape[0]}')
        a_cells = afm.copy()

        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logging.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            cells_to_retain = a_cells.obs.query('GBC in @clones_to_retain').index
            a_cells = a_cells[cells_to_retain, :].copy()
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[cells_to_retain, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[cells_to_retain, :]
            logging.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logging.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
        a_cells = filter_baseline(a_cells)
        a = filter_density(a_cells)

    elif filtering == 'LINEAGE_prep':

        # n cells
        logging.info(f'Feature selection method: {filtering}')
        logging.info(f'Original AFM n cells: {afm.shape[0]}')
        a_cells = afm.copy()

        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logging.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            test = a_cells.obs['GBC'].isin(clones_to_retain)
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
            a_cells = a_cells[test, :].copy()
            logging.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logging.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
        a = a_cells.copy()

    elif cells is None and variants is not None:

        # n cells
        logging.info(f'Original AFM n cells: {afm.shape[0]}')
        a_cells = afm.copy()
        
        if min_cell_number > 0:
            n_cells = a_cells.shape[0]
            logging.info(f'Filtering cells from clones with >{min_cell_number} cells')
            cell_counts = a_cells.obs.groupby('GBC').size()
            clones_to_retain = cell_counts[cell_counts>min_cell_number].index 
            test = a_cells.obs['GBC'].isin(clones_to_retain)
            a_cells.uns['per_position_coverage'] = a_cells.uns['per_position_coverage'].loc[test, :]
            a_cells.uns['per_position_quality'] = a_cells.uns['per_position_quality'].loc[test, :]
            a_cells = a_cells[test, :].copy()
            logging.info(f'Removed other {n_cells-a_cells.shape[0]} cells')
            logging.info(f'Retaining {a_cells.obs["GBC"].unique().size} clones for the analysis.')
        a = a_cells[:, variants].copy()
        a = filter_sites(a)

    elif cells is not None and variants is None:
        logging.info(f'Original AFM n cells: {afm.shape[0]}')
        a_cells = afm[cells, :].copy()
        a_cells = filter_sites(a_cells)
        a = a_cells

    elif cells is not None and variants is not None:
        logging.info(f'Original AFM n cells: {afm.shape[0]}')
        a_cells = afm[cells, variants].copy()
        a_cells = filter_sites(a_cells)
        a = a_cells
    
    else:
        raise ValueError(
                    f'''The provided filtering method {filtering} is not supported.
                        Choose another one...'''
                )

    logging.info(f'Filtered feature matrix contains {a.shape[0]} cells and {a.shape[1]} variants.')

    return a_cells, a


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
    a_cells, a = filter_cells_and_vars(afm, cells=cells, variants=vois)

    return a_cells, a


##