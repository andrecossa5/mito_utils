"""
_colors.py stores functions to create Cellula colors.
"""

import pandas as pd
import scanpy as sc
import seaborn as sns
import colorsys
import matplotlib.colors
import numpy as np


##


def _change_color(color, saturation=0.5, lightness=0.5):
    
    r, g, b = color
    h, s, l = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    
    return (r, g, b)


##


def create_palette(df, var, palette=None, col_list=None, 
                saturation=None, lightness=None):
    """
    Create a color palette from a df, a columns, a palette or a list of colors.
    """
    try:
        cats = df[var].cat.categories
    except:
        cats = df[var].unique()
    
    n = len(cats)
    if col_list is not None:
        cols = col_list[:n]
    elif palette is not None:
        cols = sns.color_palette(palette, n_colors=n)
    else:
        raise ValueError('Provide one between palette and col_list!')
    
    colors = { k: v for k, v in zip(cats, cols)}
    
    if saturation is not None:
        colors = { 
            k: _change_color(colors[k], saturation=saturation) \
            for k in colors 
        }
    if lightness is not None:
        colors = { 
            k: _change_color(colors[k], lightness=lightness) \
            for k in colors 
        }
     
    return colors


##


def create_colors(meta, chosen=None):
    """
    Create Cellula 'base' colors: samples, seq run, and optionally leiden categorical covariates.
    """

    # Create a custom dict of colors
    colors = {
        'sample' : create_palette(meta, 'sample', palette='tab20'),
        'seq_run' : create_palette(meta, 'seq_run', palette='tab20')
    }
    
    # Add cluster colors, if needed
    n = len(meta[chosen].cat.categories)
    if chosen is not None:
        if n <= 20:
            c = sc.pl.palettes.default_20[:n]
        else:
            c = sc.pl.palettes.default_102[:n]
        colors[chosen] = { cluster : color for cluster, color in zip(meta[chosen].cat.categories, c)}

    return colors


##


def harmonize_colors(df, group1=None, group2=None, palette=None):
    """
    Given two sets of labels, create two dicts of colors that match closely.
    """
    
    if palette is None:
        if d2['GBC'].unique().size > 15:
            palette = [ matplotlib.colors.hex2color(x) for x in sc.pl.palettes.godsnot_102 ]
        else:
            palette = ten_godisnot

    cross = pd.crosstab(df[group1].astype('str'), df[group2].astype('str'), normalize=0)
    # order = embs.groupby('inference').size().sort_values(ascending=False).index
    # cross = cross.loc[order,:]
    
    g1 = []
    g2 = []
    colors = []
    g1_labels = cross.index.to_list()
    g2_labels = cross.columns.to_list()

    for i in range(cross.shape[0]):

        x = cross.iloc[i,:] 
        j = np.argmax(x)
        # go = True
        # while go:
        #     try:
        #         j = np.where(x>t)[0][0]
        #         go = False
        #     except:
        #         t -= 0.1            

        g1.append(g1_labels[i])
        colors.append(palette[i])
                      
        if not g2_labels[j] in g2:
            gg2bc.append(g2_labels[j])
            
    len(colors)

    # Final rescue
    for x2 in g2_labels:
        if x2 not in g2:
            i += 1
            g2.append(x2)
            colors.append(palette[i])
            
    colors_g2 = { k:v for k,v in zip(g2, colors[:len(g2)])} 
    colors_g1 = { k:v for k,v in zip(g1, colors[:len(g1)])} 

    if 'unassigned' in df[g1].values:
        colors_g1['unassigned'] = (
            0.8470588235294118, 0.8274509803921568, 0.792156862745098
        )
        
    assert all([x in g1_labels for x in colors_g1])
    assert all([x in g2_labels for x in g2])

    return colors_g1, colors_g2
    
        
##

        
# Palettes
ten_godisnot = [
    
    '#001E09', 
    '#885578',
    '#FF913F', 
    '#1CE6FF', 
    '#549E79', 
    '#C9E850', #'#00FECF', 
    '#EEC3FF', 
    '#FFEF00',#'#0000A6', 
    '#D157A0', 
    '#922329'
    
]

ten_godisnot = [ matplotlib.colors.hex2color(x) for x in ten_godisnot ]

