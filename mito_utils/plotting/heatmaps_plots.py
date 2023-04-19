"""
Utils and plotting functions to visualize (clustered and annotated) cells x vars AFM matrices
or cells x cells distances/affinity matrices.
"""

from .plotting_base import *
from .colors import *


##


# Cells x vars AFMs
def cells_vars_heatmap(afm, cell_anno=None, anno_colors=None, heat_label=None, 
    legend_label=None, figsize=(11, 8), title=None, cbar_position=(0.82, 0.2, 0.02, 0.25),
    title_hjust=0.47, legend_bbox_to_anchor=(0.825, 0.5), legend_loc='lower center', 
    legend_ncol=1, xticks_size=5):
    """
    Given a (filtered) cells x vars AFM, produce its (clustered, or ordered) 
    annotated heatmap visualization.
    """
    g = sns.clustermap(pd.DataFrame(data=afm.X, columns=afm.var_names), 
    cmap='magma', yticklabels=False, xticklabels=True, 
    dendrogram_ratio=(.3, .04), figsize=figsize, row_cluster=True, col_cluster=True, 
    annot=False, cbar_kws={'use_gridspec' : False, 'orientation' : 'vertical', 'label' : heat_label}, 
    colors_ratio=0.05, row_colors=cell_anno
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=xticks_size)
    g.fig.subplots_adjust(right=0.7)
    g.ax_col_dendrogram.set_visible(False) 
    if title is not None:
        g.fig.suptitle(title, x=title_hjust)
    g.ax_cbar.set_position(cbar_position)

    handles = create_handles(anno_colors.keys(), colors=anno_colors.values())
    g.fig.legend(handles, anno_colors.keys(), loc=legend_loc, 
        bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, frameon=False, title=legend_label
    )

    return g


##


def cell_cell_dists_heatmap(D, cell_anno=None, anno_colors=None, heat_label=None, 
    legend_label=None, figsize=(11, 6.5), title=None, cbar_position=(0.82, 0.2, 0.02, 0.25),
    title_hjust=0.47, legend_bbox_to_anchor=(0.825, 0.5), legend_loc='lower center', 
    legend_ncol=1):
    """
    Plot cell-to-cell similarity matrix.
    """
    g = sns.clustermap(pd.DataFrame(data=1-D.X, index=D.obs_names.to_list(), columns=D.obs_names.to_list()), 
        cmap='magma', yticklabels=False, xticklabels=False, 
        dendrogram_ratio=(.3, .04), figsize=figsize, row_cluster=True, col_cluster=True, 
        annot=False, cbar_kws={'use_gridspec' : False, 'orientation' : 'vertical', 'label' : heat_label}, 
        colors_ratio=0.025, row_colors=cell_anno, col_colors=cell_anno
    )
    g.fig.subplots_adjust(right=0.7)
    g.ax_col_dendrogram.set_visible(False) 
    g.fig.suptitle(title, x=title_hjust)
    g.ax_cbar.set_position(cbar_position)
    handles = create_handles(anno_colors.keys(), colors=anno_colors.values())
    g.fig.legend(handles, anno_colors.keys(), loc=legend_loc, 
        bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, frameon=False, title=legend_label
    )

    return g


##

















