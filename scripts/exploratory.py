"""
Exploratory analysis and checks on the three original and formatted AFMs.
"""

import os
import sys
import scanpy as sc
from mito_utils.utils.helpers import Timer
from mito_utils.plotting.plotting_base import *
from mito_utils.plotting.colors import *
from mito_utils.preprocessing.preprocessing import *
from mito_utils.plotting.diagnostic_plots import *
from mito_utils.plotting.heatmaps_plots import *


##

# path_main = '/Users/IEO5505/Desktop/example_mito_exploratory'
# n=2
# matplotlib.use('macOSX')
# afm = AFMs['AML_clones']


# Paths
path_main = sys.argv[1]
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/exploratory/')

# Read data in a dictionary
samples = os.listdir(path_data)
AFMs = { sample : read_one_sample(path_data, sample=sample) for sample in samples }
n = len(AFMs)
colors = { s:c for s,c in zip(AFMs.keys(), sc.pl.palettes.vega_10[:n]) }

################################################################

# Cell level diagnostics

############## n covered positions/total position. Cell distribution.

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    cell_n_sites_covered_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('n covered sites (total=16569), across cell')
fig.savefig(os.path.join(path_results, 'covered_sites_per_cell.png'))

############## n covered variants/total variants. Cell distribution.

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    cell_n_vars_detected_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('n MT-SNVs detected (total=16569*3), across cell')
fig.savefig(os.path.join(path_results, 'variants_per_cell.png'))

############## Median site quality per cell distribution

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    mean_site_quality_cell_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('Mean site quality, across cell')
fig.savefig(os.path.join(path_results, 'mean_site_quality_per_cell.png'))

################################################################

# Position level diagnostics

############## Per position mean (over cells) coverage

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    mean_position_coverage_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('Mean (across cells) coverage, per position')
fig.savefig(os.path.join(path_results, 'mean_coverage_per_position.png'))

############## Mean base quality per position 

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    mean_position_quality_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('Mean (across cells) quality, per position')
fig.savefig(os.path.join(path_results, 'mean_quality_per_position.png'))

################################################################

# Variant level diagnostics

############## % of cells in which a variant is detected: distribution

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    vars_n_positive_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('n of positive cells, per variant')
fig.savefig(os.path.join(path_results, 'cells_per_variant.png'))

############## Ranked AF distributions (VG-like)

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    vars_AF_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('Ranked variant AFs')
fig.savefig(os.path.join(path_results, 'VG_like_AFs.png'))

############## Mean/variance AF trend

fig, axs = plt.subplots(1,n,figsize=(5*n,5),constrained_layout=True)
for i, x in enumerate(samples):
    AF_mean_var_corr(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('AF mean-variance trend')
fig.savefig(os.path.join(path_results, 'mean_variances_trend.png'))

##############

# Fancy coverage plot
fig, axs = plt.subplots(1,n,figsize=(5*n,5), subplot_kw={'projection': 'polar'})

for i, x in enumerate(samples):
    MT_coverage_polar(AFMs[x], ax=axs[i], title=x)
fig.suptitle('MT-genome coverage')
fig.tight_layout()
fig.savefig(os.path.join(path_results, 'MT_coverage.png'))

##############