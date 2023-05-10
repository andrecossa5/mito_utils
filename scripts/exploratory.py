"""
Exploratory analysis and checks on the three original and formatted AFMs.
"""

import os
import re
import matplotlib
import scanpy as sc
import matplotlib.pyplot as plt
from mito_utils.utils import Timer
from matplotlib.colors import Normalize
from mito_utils.plotting_base import *
from mito_utils.colors import *
from mito_utils.preprocessing import *
from mito_utils.diagnostic_plots import *
from mito_utils.heatmaps_plots import *


##

# Paths
path_main = '/Users/IEO5505/Desktop/mito_bench/'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/exploratory/')

# Read data in a dictionary
samples = [ s for s in os.listdir(path_data) if bool(re.search('AML|MDA', s)) ]
AFMs = { s : read_one_sample(path_data, sample=s) for s in samples }
n = len(AFMs) 
colors = { s:c for s,c in zip(AFMs.keys(), sc.pl.palettes.vega_10[:n]) }

#

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

# Circle packed plot, one fig per sample (in vitro)
fig, axs = plt.subplots(2, 1, figsize=(5,9))

s = 'MDA_clones'
df_ = (
    AFMs[s].obs
    .groupby('GBC')
    .size()
    .to_frame('n_cells')
    .assign(prevalence=lambda x: x['n_cells'] / x['n_cells'].sum())
)
packed_circle_plot(df_, covariate='prevalence', ax=axs[0], color=colors[s], 
                annotate=True, fontsize=9, alpha=0.35, linewidth=1.5, annot_treshold=0.01)
format_ax(axs[0], title=s, title_size=10)

s = 'AML_clones'
df_ = (
    AFMs[s].obs
    .groupby('GBC')
    .size()
    .to_frame('n_cells')
    .assign(prevalence=lambda x: x['n_cells'] / x['n_cells'].sum())
)
packed_circle_plot(df_, covariate='prevalence', ax=axs[1], color=colors[s], 
                annotate=True, fontsize=9, alpha=0.35, linewidth=1.7, annot_treshold=0.01)
format_ax(axs[1], title=s, title_size=10)
        
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.85, hspace=0.3)
fig.savefig(os.path.join(path_results, f'in_vitro_circle_packed.png'))


##

# MDA PT-lung couple 
others = set(AFMs['MDA_PT'].obs.groupby('GBC').size().loc[lambda x: x<10].index) #| \
         #set(AFMs['MDA_lung'].obs.groupby('GBC').size().loc[lambda x: x<10].index)

df_PT = (
    AFMs['MDA_PT'].obs
    .assign(GBC_filtered=lambda x: x['GBC'].apply(lambda x: x if x not in others else 'others'))
    .groupby('GBC_filtered')
    .size()
    .to_frame('n_cells')
    .assign(
        prevalence=lambda x: x['n_cells'] / x['n_cells'].sum(),
        sample='MDA_PT'
    )
)
df_lung = (
    AFMs['MDA_lung'].obs
    #.assign(GBC_filtered=lambda x: x['GBC'].apply(lambda x: x if x not in others else 'others'))
    #.groupby('GBC_filtered')
    .groupby('GBC')
    .size()
    .to_frame('n_cells')
    .assign(
        prevalence=lambda x: x['n_cells'] / x['n_cells'].sum(),
        sample='MDA_lung'
    )
)

# Concat
df_ = pd.concat([df_PT, df_lung])

# df_couple for plotting
df_couple = (
    df_
    .reset_index(names='GBC')
    .pivot_table(index='GBC', values=['n_cells', 'prevalence'], columns='sample')
    .fillna(0)
    .assign(promet=lambda x: np.sum(x.loc[:, pd.IndexSlice['n_cells',:]] > 0, axis=1) > 1) 
    .loc[:, pd.IndexSlice[['prevalence', 'promet'],:]]
)
df_couple = pd.concat(
    [
        df_couple.loc[:, pd.IndexSlice['prevalence',:]].droplevel(level=0, axis=1),
        df_couple.loc[:, ['promet']].droplevel(level=1, axis=1)
    ],
    axis=1
)
df_couple = (
    df_couple
    .assign(potential=lambda x: np.log10((x['MDA_lung'] + 0.0001) / x['MDA_PT']))
    .reset_index()
    .melt(
        id_vars=['GBC', 'promet', 'potential'], 
        var_name='origin',
        value_name='prevalence'
    )
    .set_index('GBC')
    .loc[lambda x: x['prevalence']>0 ]
    .drop_duplicates()
)


##


# Only major clones, colored by metastatic potential
fig, axs = plt.subplots(1,2,figsize=(10,5))

# Colors
cmap = plt.cm.get_cmap('Reds')
norm = Normalize(vmin=0, vmax=1)
colors = df_couple['potential'].apply(lambda x: cmap(norm(x))).to_dict()
colors['others'] = 'lightgrey'

# Fig
promet = (
    df_couple.loc[lambda x: x['promet']]['potential']
    .sort_values(ascending=False)
    .index[:10]
)

fig, axs = plt.subplots(1,2,figsize=(10,5))
packed_circle_plot(df_couple.query('origin == "MDA_PT"'), covariate='prevalence', 
    colors_dict=colors, 
    spacing=0.7,
    ax=axs[0], annotate=True, fontsize=6, annot_treshold=0.05,
    edgecolor='#632118',
    names_to_annotate=promet
)
format_ax(axs[0], title='PT')
packed_circle_plot(df_couple.query('origin == "MDA_lung"'), covariate='prevalence', 
    colors_dict=colors, 
    spacing=0.8,
    ax=axs[1], annotate=True, fontsize=6, annot_treshold=0.05,
    edgecolor='#632118',
    names_to_annotate=promet
)
format_ax(axs[1], title='lung')
fig.suptitle('MDA in vivo PT-lung couple')

# Save
fig.tight_layout()
fig.savefig(os.path.join(path_results, f'MDA_couple_circle_packed.png'))


##

