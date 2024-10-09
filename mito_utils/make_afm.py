"""
Module to make_afm: create a clean allelic frequency matrix as AnnData object.
"""

import os
import warnings
from scipy.io import mmread
from scipy.sparse import csr_matrix
from anndata import AnnData
from .utils import *
from .positions import *
warnings.filterwarnings("ignore")


##


def mask_mt_sites(site_list):
    """
    Function to mask all sites outside of known MT-genes bodies.
    """

    mask = []
    for pos in site_list:
        pos = int(pos)
        t = [ pos>=start and pos<=end for _, start, end in MAESTER_genes_positions ]
        if any(t):
            mask.append(True)
        else:
            mask.append(False)

    return np.array(mask)


##


def read_from_AD_DP(path_ch_matrix, path_meta, sample=None, pp_method=None, cell_col='cell', char_col='MUT'):
    """
    Create AFM as as AnnData object from a path_ch_matrix folder with AD, DP tables. AD and DP columns must have:
    1) <cell_col> column; 2) <char> column; 3) AD/DP columns, respectively.
    N.B. <char> columns must be formatted in "pos>_ref>alt" fashion and cell_meta index must be in {CB}_{sample} format.
    """

    AD = pd.read_csv(os.path.join(path_ch_matrix, 'AD.csv.gz'), index_col=0)
    DP = pd.read_csv(os.path.join(path_ch_matrix, 'DP.csv.gz'), index_col=0)
    cell_meta = pd.read_csv(path_meta, index_col=0)
    AD = AD.pivot(index=cell_col, columns=char_col, values='AD').fillna(0)
    DP = DP.pivot(index=cell_col, columns=char_col, values='DP').fillna(0)

    cells = list(set(cell_meta.index) & set(DP.index))
    AD = AD.loc[cells].copy()
    DP = DP.loc[cells].copy()
    cell_meta = cell_meta.loc[cells].copy()

    assert (AD.index == DP.index).all()
    assert (AD.index == cell_meta.index).all()

    cell_meta = AD.index.to_series().to_frame().assign(sample=sample)[['sample']]
    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]
    AF = csr_matrix(np.divide(AD,(DP+.00000001)).values.astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)

    afm = AnnData(X=AF, obs=cell_meta, var=char_meta, layers={'AD':AD, 'DP':DP}, uns={'pp_method':pp_method})
    sorted_vars = afm.var['pos'].sort_values().index
    afm = afm[:,sorted_vars].copy()

    return afm


##


def read_from_cellsnp(path_ch_matrix, path_meta, sample=None, pp_method='cellsnp-lite'):
    """
    Create AFM as as AnnData object from cellsnp output tables. The path_ch_matrix folder must contain the four default output from cellsnp-lite:
    * 1: 'cellSNP.tag.AD.mtx.gz'
    * 2: 'cellSNP.tag.AD.mtx.gz'
    * 3. 'cellSNP.base.vcf.gz'
    * 4: 'cellSNP.samples.tsv.gz'
    N.B. cell_meta index must be in {CB}_{sample} format.
    """

    path_AD = os.path.join(path_ch_matrix, 'cellSNP.tag.AD.mtx.gz')
    path_DP = os.path.join(path_ch_matrix, 'cellSNP.tag.AD.mtx.gz')
    path_vcf = os.path.join(path_ch_matrix, 'cellSNP.base.vcf.gz')
    path_cells = os.path.join(path_ch_matrix, 'cellSNP.samples.tsv.gz')

    cells = [ f'{x}_{sample}' for x in pd.read_csv(path_cells, header=None)[0].to_list() ]
    vcf = pd.read_csv(path_vcf, sep='\t', skiprows=1)
    variants = vcf['POS'].astype(str) + '_' + vcf['REF'] + '>' + vcf['ALT']
    AD = pd.DataFrame(mmread(path_AD).A.T, index=cells, columns=variants)
    DP = pd.DataFrame(mmread(path_DP).A.T, index=cells, columns=variants)
    cell_meta = pd.read_csv(path_meta, index_col=0)

    cells = list(set(cell_meta.index) & set(DP.index))
    AD = AD.loc[cells].copy()
    DP = DP.loc[cells].copy()
    cell_meta = cell_meta.loc[cells].copy()

    assert (AD.index == DP.index).all()
    assert (AD.index == cell_meta.index).all()

    cell_meta = AD.index.to_series().to_frame().assign(sample=sample)[['sample']]
    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]
    AF = csr_matrix(np.divide(AD,(DP+.00000001)).values.astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)

    afm = AnnData(X=AF, obs=cell_meta, var=char_meta, layers={'AD':AD, 'DP':DP}, uns={'pp_method':pp_method})
    sorted_vars = afm.var['pos'].sort_values().index
    afm = afm[:,sorted_vars].copy()


    return afm


##


def sparse_from_long(df, covariate, nrow, ncol, cell_order):
    """
    Make a long df a sparse matrix more efficiently.
    """

    df['code'] = pd.Categorical(df['cell'], categories=cell_order).codes
    sparse_matrix = csr_matrix(
        (df[covariate].values, (df['code'].values, df['pos'].values-1)), 
        shape=(nrow, ncol)
    )

    return sparse_matrix


##


def read_from_scmito(path_ch_matrix, path_meta, sample=None, pp_method='maegatk'):
    """
    Create AFM as as AnnData object from cellsnp output tables. 
    The path_ch_matrix folder must contain the default output from mito_preprocessing/maegatk:
    * 1: 'A|C|T|G.txt.gz'
    * 2: 'coverage.txt.gz'
    * 3. 'refAllele.txt'
    Additional outputs from mito_preprocessing can be used for separate analyses.
    N.B. cell_meta index must be in {CB}_{sample} format. mito_preprocessing and maegatk CBs are
    plain, not in {CB}_{sample} format.
    """

    # Metrics
    metrics = {}

    # Process each base table
    path_A = os.path.join(path_ch_matrix, 'A.txt.gz')
    path_C = os.path.join(path_ch_matrix, 'C.txt.gz')
    path_T = os.path.join(path_ch_matrix, 'T.txt.gz')
    path_G = os.path.join(path_ch_matrix, 'G.txt.gz')
    path_cov = os.path.join(path_ch_matrix, 'coverage.txt.gz')
    path_refAllele = os.path.join(path_ch_matrix, 'refAllele.txt')
    ref_allele = pd.read_csv(path_refAllele, header=None)
    ref = { pos:ref for pos,ref in zip(ref_allele[0], ref_allele[1]) }

    L = []
    for base, path_base in zip(['A', 'C', 'T', 'G'], [path_A, path_C, path_T, path_G]):

        logging.info(f'Process table: {base}')
        base_df = pd.read_csv(path_base, header=None)

        if pp_method == 'maegatk':
            base_df.columns = ['pos', 'cell', 'count_fw', 'qual_fw', 'count_rev', 'qual_rev']
        elif pp_method == 'mito_preprocessing':
            base_df.columns = ['pos', 'cell', 
                               'count_fw', 'qual_fw', 'cons_fw', 'gs_fw', 
                               'count_rev', 'qual_rev', 'cons_rev', 'gs_rev']
            
        base_df['counts'] = base_df['count_fw'] + base_df['count_rev']
        qual = base_df[['qual_fw', 'qual_rev']].values
        base_df['qual'] = np.nanmean(np.where(qual>0,qual,np.nan), axis=1)
        L.append(base_df[['pos', 'cell', 'counts', 'qual']].assign(base=base))
    
    # Concat in long format
    logging.info(f'Format all basecalls in a long table')
    long = pd.concat(L)
    long['cell'] = long['cell'].map(lambda x: f'{x}_{sample}')

    # Annotate ref and alt base calls
    long['ref'] = long['pos'].map(ref)
    # s = long.groupby(['cell', 'pos', 'base']).size()
    # assert all(s == 1)
    metrics['total_basecalls'] = long.shape[0]
    logging.info(f'Total basecalls: {long.shape[0]}')

    # Filter only variant basecalls
    logging.info(f'Filter variant allele basecalls')
    long = long.query('base!=ref').copy()
    long['nunique'] = long.groupby(['cell', 'pos'])['base'].transform('nunique')
    long = (
        long.query('nunique==1')
        .drop(columns=['nunique'])
        .rename(columns={'counts':'AD', 'base':'alt'})
        .copy()
    )

    # s = long.groupby(['cell', 'pos']).size()
    # assert all(s == 1)
    metrics['variant_basecalls'] = long.shape[0]
    logging.info(f'Unique variant basecalls: {long.shape[0]}')
 
    # Filter basecalls of annotated cells only (i.e., we have cell metadata)
    logging.info(f'Filter for annotated cells (i.e., sample CBs in cell_meta)')
    cell_meta = pd.read_csv(path_meta, index_col=0)
    cells = list(set(cell_meta.index) & set(long['cell'].unique()))
    long = long.query('cell in @cells').copy()
    metrics['variant_basecalls_for_annot_cells'] = long.shape[0]
    logging.info(f'Unique variant basecalls for annotated cells: {long.shape[0]}')
 
    # Add site coverage
    logging.info(f'Retrieve cell-site total coverage')
    cov = pd.read_csv(path_cov, header=None)
    cov.columns = ['pos', 'cell', 'DP']
    cov['cell'] = cov['cell'].map(lambda x: f'{x}_{sample}')
    long = long.merge(cov, on=['pos', 'cell'], how='left')
  
    # Matrices
    logging.info(f'Format AD/DP/qual matrices')
    long['mut'] = long['pos'].astype(str) + '_' + long['ref'] + '>' + long['alt']
    AD = long.pivot(index='cell', columns='mut', values='AD').fillna(0)
    DP = long.pivot(index='cell', columns='mut', values='DP').fillna(0)
    qual = long.pivot(index='cell', columns='mut', values='qual').fillna(0)
 
    assert (AD.index.value_counts()==1).all()
 
    # Ensure common cell index for each matrix
    AD = AD.loc[cells].copy()
    DP = DP.loc[cells].copy()
    qual = qual.loc[cells].copy()
    cell_meta = cell_meta.loc[cells].copy()
 
    # At least one unique variant basecall for each cell
    assert (np.sum(DP>0, axis=1)>0).all()
    assert (np.sum(DP>0, axis=1)>0).all() 
    assert (AD.index == DP.index).all()
    assert (AD.columns == DP.columns).all()
    assert (AD.index == cell_meta.index).all()
 
    # Char and cell metadata
    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]
 
    """
    We have selected relevant info (alt and ref counts, quality and allelic frequency) 
    about all interesting variant basecalls in the data. We have just excluded:
        1- basecalls of un-annotated cells
        2- basecalls for which more than one alternative allele has been observed (same cell, same site).
    """
 
    # To sparse and AnnData
    logging.info('Build AnnData object')
    AF = csr_matrix(np.divide(AD.values,(DP.values+.00000001)).astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)
    qual = csr_matrix(qual.values).astype(np.int16)
    afm = AnnData(
        X=AF, obs=cell_meta, var=char_meta, 
        layers={'AD':AD, 'DP':DP, 'qual':qual}, 
        uns={'pp_method':pp_method, 'raw_basecalls_metrics':metrics}
    )
    sorted_vars = afm.var['pos'].sort_values().index
    assert sorted_vars.size == afm.shape[1]
    afm = afm[:,sorted_vars].copy()
 
    # Add complete site coverage info
    logging.info('Add site-coverage matrix and cell-coverage metrics')
    cov = cov.pivot(index='cell', columns='pos', values='DP').fillna(0)
    cov = cov.loc[cells]
    mapping = afm.var['pos'].to_dict()
    df_ = pd.DataFrame({ mut : cov[mapping[mut]].values for mut in mapping }, index=cells)
    assert all(df_.columns == afm.var_names)
    afm.layers['site_coverage'] = csr_matrix(df_.values)
    afm.obs['mean_site_coverage'] = cov.mean(axis=1)   
    test_sites = mask_mt_sites(range(cov.shape[1]))
    afm.obs['median_target_site_coverage'] = cov.loc[:,test_sites].median(axis=1)
    afm.obs['median_untarget_site_coverage'] = cov.loc[:,~test_sites].median(axis=1)
    afm.obs['frac_target_site_covered'] = np.sum(cov.loc[:,test_sites]>0, axis=1) / test_sites.sum()

    return afm


##


def make_afm(path_ch_matrix, path_meta, sample=None, pp_method='mito_preprocessing'):
    """
    
    Creates an annotated Allele Frequency Matrix from different preprocessing pipelines outputs.

    Args:
        path_ch_matrix (str): Path to folder with necessary data.
        path_meta (str): 
        sample (str, optional): Sample name to append at preprocessed CBs. Defaults to None.
        pp_method (str, optional): Preprocessing method (i.e., mito_preprocessing, maegatk, cellsnp-lite, freebayes, samtools). Defaults to 'mito_preprocessing'.
        **kwargs
    """

    if os.path.exists(path_ch_matrix) and os.path.exists(path_meta):

        logging.info(f'Allele Frequency Matrix from {pp_method} output')
        T = Timer()
        T.start()

        if pp_method in ['samtools', 'freebayes']:
            afm = read_from_AD_DP(path_ch_matrix, path_meta, sample, pp_method)
        elif pp_method == 'cellsnp-lite':
            afm = read_from_cellsnp(path_ch_matrix, path_meta, sample, pp_method)
        elif pp_method in ['mito_preprocessing', 'maegatk']:
            afm = read_from_scmito(path_ch_matrix, path_meta, sample, pp_method)
        
        logging.info(f'Allele Frequency Matrix: cell x char {afm.shape}. {T.stop()}')

        return afm

    else:
        raise ValueError('Specify good path_ch_matrix and path_meta! ')


##


##################################### DEPRECATED FUNCTIONS

# def create_one_base_tables(A, base, only_variants=True):
#     """
#     For one of the 4 possible DNA bases creates:
#     * df_x: the allelic frequency (AF) table for that base (i.e., a cell x site, with values in [0,1])
#     * df_qual: the average base-calling quality table for that base (i.e., cell x site, No value should be below 30, if already cleaned UMI counts)
#     * df_cov: UMI counts table for that base (i.e., cell x site, with integers)
#     """
# 
#     # AF
#     cov = A.layers[f'{base}_counts_fw'].A + A.layers[f'{base}_counts_rev'].A
#     X = cov / (A.layers['cov'].A + 0.000001)
# 
#     # Calculate the average quality (across both strands) for each base-site combination
#     q = A.layers[f'{base}_qual_fw'].A + A.layers[f'{base}_qual_rev'].A
#     m = np.where(A.layers[f'{base}_qual_fw'].A>0, 1, 0) + np.where(A.layers[f'{base}_qual_rev'].A>0, 1, 0)
#     qual = np.round(q / (m + 0.000001))
# 
#     # Re-format
#     ref_col = 'wt_allele' if 'wt_allele' in A.var.columns else 'ref'
#     assert ref_col in A.var.columns
#     A.var[ref_col] = A.var[ref_col].str.capitalize()
#     variant_names = A.var.index + '_' + A.var[ref_col] + f'>{base}'
#     df_x = pd.DataFrame(X, index=A.obs_names, columns=variant_names)
#     df_qual = pd.DataFrame(qual, index=A.obs_names, columns=variant_names)
#     df_cov = pd.DataFrame(cov, index=A.obs_names, columns=variant_names)
#     gc.collect()
# 
#     if only_variants:
#         test = (A.var[ref_col] != base).values
#         return df_cov.loc[:, test], df_x.loc[:, test], df_qual.loc[:, test]
#     else:
#         return df_cov, df_x, df_qual


##


# def format_matrix(A, only_variants=True):
#     """
#     Create a full cell x variant AFM from the original AnnData storing all dataset tables.
#     """
#     
#     # Clones to categorical, if present
#     if 'GBC' in A.obs.columns:
#         A.obs['GBC'] = pd.Categorical(A.obs['GBC'])
# 
#     # For each position and cell, compute each base AF and quality tables
#     A_cov, A_x, A_qual = create_one_base_tables(A, 'A', only_variants=only_variants)
#     C_cov, C_x, C_qual = create_one_base_tables(A, 'C', only_variants=only_variants)
#     T_cov, T_x, T_qual = create_one_base_tables(A, 'T', only_variants=only_variants)
#     G_cov, G_x, G_qual = create_one_base_tables(A, 'G', only_variants=only_variants)
# 
#     # Concat all of them in three complete coverage, AF and quality matrices, for each variant from the ref
#     cov = pd.concat([A_cov, C_cov, T_cov, G_cov], axis=1)
#     X = pd.concat([A_x, C_x, T_x, G_x], axis=1)
#     qual = pd.concat([A_qual, C_qual, T_qual, G_qual], axis=1)
# 
#     # Reorder columns...
#     variants = X.columns.map(lambda x: x.split('_')[0]).astype('int').values
#     idx = np.argsort(variants)
#     cov = cov.iloc[:, idx]
#     X = X.iloc[:, idx]
#     qual = qual.iloc[:, idx]
# 
#     # Create the per site quality matrix
#     quality = np.zeros(A.shape)
#     n_times = np.zeros(A.shape)
#     for k in A.layers:
#         if bool(re.search('qual', k)):
#             quality += A.layers[k].A
#             r, c = np.nonzero(A.layers[k].A)
#             n_times[r, c] += 1
#     quality = np.round(quality / (n_times + 0.0000001))
# 
#     # Create AnnData with variants and sites matrices
#     afm = anndata.AnnData(X=X, obs=A.obs, dtype=np.float32)
#     afm.layers['coverage'] = cov
#     afm.layers['quality'] = qual
# 
#     # Per site slots, in 'uns'. Each matrix is a ncells x nsites matrix
#     afm.uns['per_position_coverage'] = pd.DataFrame(
#         A.layers['cov'].A, index=afm.obs_names, columns=A.var_names
#     )
#     afm.uns['per_position_quality'] = pd.DataFrame(
#         quality, index=afm.obs_names, columns=A.var_names
#     )
#     gc.collect()
#     
#     return afm

#####################################