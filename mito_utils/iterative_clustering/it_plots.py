"""
Plots for visualizing iterative schemes performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score
from ..utils import *
from ..plotting_base import *
from ..clustering import *


##


def contingency_iterative_plot(df, afm, good_quality_cells, figsize=(8,8)):
    """
    Visualization of a double continency table to assess iterative scheme performance
    at clonal inference.
    """
    
    # Fig   
    fig, axs = plt.subplots(2,1,figsize=figsize)

    ari_all = custom_ARI(df['GBC'].astype('str'), df['MT_clones'].astype('str'))
    nmi_all = normalized_mutual_info_score(df['GBC'].astype('str'), df['MT_clones'].astype('str'))
    df_ = pd.crosstab(df['MT_clones'].astype('str'), df['GBC'].astype('str'))
    plot_heatmap(df_, ax=axs[0], y_names_size=5, rank_diagonal=True, label='n cells')
    perc_assigned = df.shape[0] / good_quality_cells.size
    n_clones_gt = afm.obs.loc[good_quality_cells]['GBC'].unique().size
    n_clones_recovered = df['GBC'].unique().size
    t = f'''NMI: {nmi_all:.2f}, ARI: {ari_all:.2f} 
        All clones: {n_clones_recovered}/{n_clones_gt} clones recovered, {perc_assigned*100:.2f}% cells'''
    format_ax(title=t, ax=axs[0], rotx=90)

    # > 10 cells
    top_clones = afm.obs.loc[good_quality_cells].groupby('GBC').size().loc[lambda x: x>=10].index
    n_top_all = afm.obs.loc[good_quality_cells].query('GBC in @top_clones').shape[0]

    df = df.query('GBC in @top_clones')
    ari_top = custom_ARI(df['GBC'].astype('str'), df['MT_clones'].astype('str'))
    nmi_top = normalized_mutual_info_score(df['GBC'].astype('str'), df['MT_clones'].astype('str'))

    df_ = pd.crosstab(df['MT_clones'].astype('str'), df['GBC'].astype('str'))
    plot_heatmap(df_, ax=axs[1], y_names_size=5, rank_diagonal=True, label='n cells')
    n_clones_gt = (
        afm.obs.loc[good_quality_cells]['GBC']
        .value_counts()
        .loc[lambda x: x>=10]  
        .unique().size
    )
    n_clones_recovered = df['GBC'].unique().size
    perc_assigned = (df.shape[0] / n_top_all) * 100
    t = f'''NMI: {nmi_top:.2f}, ARI: {ari_top:.2f} 
            Clones >=10 cells: {n_clones_recovered}/{n_clones_gt} clones recovered, {perc_assigned:.2f}% cells'''
    
    format_ax(title=t, ax=axs[1], rotx=90)

    return fig


##
