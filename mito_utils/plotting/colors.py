"""
_colors.py stores functions to create Cellula colors.
"""

import pandas as pd
import scanpy as sc
import seaborn as sns

##


def create_palette(df, var, palette=None, col_list=None):
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