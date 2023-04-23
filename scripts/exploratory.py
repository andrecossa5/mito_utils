"""
Exploratory analysis and checks on the three original and formatted AFMs.
"""

import os
import sys
import scanpy as sc
from mito_utils.utils import Timer
from mito_utils.plotting.plotting_base import *
from mito_utils.plotting.colors import *
from mito_utils.preprocessing.preprocessing import *
from mito_utils.plotting.diagnostic_plots import *


##


# Read data
path_main = sys.argv[1]
path_results = path_main + 'results_and_plots/exploratory/'

ORIG = {}
AFMs = {}

os.listdir(path_main + f'data/AFMs/')

samples = ['MDA', 'AML', 'PDX']
for x in samples:
    orig = sc.read(path_main + f'data/AFMs/{x}_afm.h5ad')
    CBC_GBC = pd.read_csv(path_main + f'data/CBC_GBC_cells/CBC_GBC_{x}.csv', index_col=0)
    afm = format_matrix(orig, CBC_GBC)
    afm.obs = afm.obs.assign(sample=x)
    AFMs[x] = afm
    ORIG[x] = orig

# Set colors
colors = {'MDA':'#DA5700', 'AML':'#0074DA', 'PDX':'#0F9221'}

##


################################################################

# Cell level diagnostics

############## Total MAESTER library cell depths distribution

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    cell_depth_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('MASTER library cell depth (nUMIs)')
fig.savefig(path_results + 'a.pdf')

############## n covered positions/total position. Cell distribution.

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    cell_n_sites_covered_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('n covered sites (total=16569), per cell')
fig.savefig(path_results + 'b.pdf')

############## n covered variants/total variants. Cell distribution.

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    cell_n_vars_detected_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('n MT-SNVs detected (total=16569*3), per cell')
fig.savefig(path_results + 'c.pdf')

############## Median site quality per cell distribution

fig, ax = plt.subplots()
fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    cell_median_site_quality_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('MASTER library median quality, per cell')
fig.savefig(path_results + 'd.pdf')

################################################################

# Site level diagnostics

############## Per site median (over cells) coverage

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    site_median_coverage_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('Median (across cells) MT sites coverage (nUMIs)')
fig.savefig(path_results + 'e.pdf')

############## Median base quality per site distribution

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    site_median_quality_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('MASTER library median quality, per site')
fig.savefig(path_results + 'f.pdf')

################################################################

# Variant level diagnostics

############## % of cells in which a variant is detected: distribution

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    vars_n_positive_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('n of positive cells, per variant')
fig.savefig(path_results + 'g.pdf')

############## Ranked AF distributions (VG-like)

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    vars_AF_dist(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('Ranked variant AFs')
fig.savefig(path_results + 'VG_like_AFs.pdf')

############## Strand concordances distributions

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    vars_strand_conc_dist(ORIG[x], AFMs[x].var_names, ax=axs[i], color=colors[x], title=x)
fig.suptitle('Variants strand concordance distribution')
fig.savefig(path_results + 'h.pdf')

############## Mean AF/strand concordance relationship

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    AF_mean_strand_conc_corr(
        ORIG[x], AFMs[x].var_names, AFMs[x], ax=axs[i], color=colors[x], title=x
    )
fig.suptitle('AF mean-strand concordance correlation')
fig.savefig(path_results + 'i.pdf')

############## Mean/variance AF trend

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    AF_mean_var_corr(AFMs[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('AF mean-variance correlation')
fig.savefig(path_results + 'l.pdf')

############## % of positive cells, stratified for variant type

fig, axs = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for i, x in enumerate(samples):
    positive_events_by_var_type(AFMs[x], ORIG[x], ax=axs[i], color=colors[x], title=x)
fig.suptitle('\n% of positive events by mutation type')
fig.savefig(path_results + 'm.pdf')

################################################################




