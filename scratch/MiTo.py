"""
WTF...
"""

import os
from itertools import product
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *
from mito_utils.metrics import *
from mito_utils.dimred import *
from mito_utils.phylo import *
matplotlib.use('macOSX')


##


# Args
os.chdir('/Users/IEO5505/Desktop/MI_TO/mito_utils/scratch')
path_dbSNP = '/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/miscellanea/dbSNP_MT.txt'
path_REDIdb = '/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/miscellanea/REDIdb_MT.txt'


d = {}

t_prob = [.5,.6,.7,.8,.9]
min_cell_prevalence = [.1,.15,.2]

for x,y in product(t_prob, min_cell_prevalence):


    afm = sc.read('afm.h5ad')
    afm = filter_cells(afm, cell_filter='None')
    afm_raw = afm.copy()

    afm = filter_afm(
        afm,
        min_cell_number=10,
        lineage_column='GBC',
        filtering='MiTo',
        filtering_kwargs={
            'min_cov' : 10,
            'min_var_quality' : 30,
            'min_frac_negative' : .2,
            'min_n_positive' : 5,
            'af_confident_detection' : .02,
            'min_n_confidently_detected' : 2,
            'min_mean_AD_in_positives' : 1.2,       # 1.25,
            'min_mean_DP_in_positives' : 5
        },
        binarization_kwargs={
            't_prob':x, 't_vanilla':.0, 'min_AD':2, 'min_cell_prevalence':y
        },
        bin_method='MiTo',
        path_dbSNP=path_dbSNP, 
        path_REDIdb=path_REDIdb,
        max_AD_counts=2
    )

    compute_distances(afm, precomputed=True)
    tree = build_tree(afm, precomputed=True)
    tree, _,_ = MiToTreeAnnotator(tree)
    df = tree.cell_meta.dropna()
    ari = custom_ARI(df['GBC'], df['MT_clone'])
    nmi = normalized_mutual_info_score(df['GBC'], df['MT_clone'])

    d[(x,y)] = [ari,nmi,afm.shape]



pd.Series(d).sort_values()


