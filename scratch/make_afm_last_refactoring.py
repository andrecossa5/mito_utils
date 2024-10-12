"""
Coding game.
"""

import os
from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.make_afm import read_from_cellsnp
from mito_utils.phylo import *


sample ='MDA_PT'
path_dbSNP = '/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/miscellanea/dbSNP_MT.txt'
path_REDIdb = '/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/miscellanea/REDIdb_MT.txt'
path_meta = f'/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/cells_meta.csv'

# path_afm = f'/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/AFMs/mito_preprocessing/{sample}/afm.h5ad'
# afm = sc.read(path_afm)

path_ch_matrix = f'/Users/IEO5505/Desktop/MI_TO/MI_TO_analysis_repro/data/MI_TO_bench/AFMs/samtools/{sample}'
pp_method = 'samtools'

# make_afm(path_ch_matrix, path_meta=path_meta, sample=sample, pp_method='cellsnp-lite')

afm = read_from_AD_DP(path_ch_matrix, path_meta, sample, 'samtools')

afm = filter_cells(afm, cell_filter='no filter')
filter_afm(afm, filtering=None, bin_method='vanilla') #, return_tree=True, spatial_metrics=True)




