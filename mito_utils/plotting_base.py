"""
plotting_base.py stores plotting utilities and 'base plots', i.e., 
simple plots returning an Axes object.
"""

import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns 
from statannotations.Annotator import Annotator 
import textalloc as ta
from circlify import _bubbles, circlify, Circle

from .utils import *
from .colors import *
plt.style.use('default')


##


# Params
axins_pos = {

    'v2' : ( (.95,.75,.01,.22), 'left' ),
    'v3' : ( (.95,.05,.01,.22), 'left' ),
    'v1' : ( (.05,.75,.01,.22), 'right' ),
    'v4' : ( (.05,.05,.01,.22), 'right' ),

    'h2' : ( (1-.27,.95,.22,.01), 'bottom' ),
    'h3' : ( (1-.27,.05,.22,.01), 'top' ),
    'h1' : ( (0.05,.95,.22,.01), 'bottom' ),
    'h4' : ( (0.05,.05,.22,.01), 'top' ),

    'outside' : ( (1.05,.25,.01,.5), 'right' )
}


##

def create_handles(categories, marker='o', colors=None, size=10, width=0.5):
    """
    Create quick and dirty circular and labels for legends.
    """
    if colors is None:
        colors = sns.color_palette('tab10')[:len(categories)]

    handles = [ 
        (Line2D([], [], 
        marker=marker, 
        label=l, 
        linewidth=0,
        markersize=size, 
        markeredgewidth=width, 
        markeredgecolor='white', 
        markerfacecolor=c)) \
        for l, c in zip(categories, colors) 
    ]

    return handles


##


def add_cbar(x, color='viridis', ax=None, label_size=7, ticks_size=5, 
    label=None, orientation='v', pos=2):
    """
    Draw cbar on an axes object inset.
    """
    if pos == 'outside':
        pos, xticks_position = axins_pos[pos]
        orientation = 'vertical'
    else:
        pos, xticks_position = axins_pos[orientation+str(pos)]
        orientation = 'vertical' if orientation == 'v' else 'horizontal'
    cmap = matplotlib.colormaps[color]
    norm = matplotlib.colors.Normalize(vmin=np.percentile(x, q=5), vmax=np.percentile(x, q=95))
    axins = ax.inset_axes(pos) 
    cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
        cax=axins, orientation=orientation, ticklocation=xticks_position
    )
    cb.set_label(label=label, size=label_size, loc='center')
    if orientation == 'vertical':
        cb.ax.tick_params(axis="y", labelsize=ticks_size)
    else:
        cb.ax.tick_params(axis="x", labelsize=ticks_size)

    return cb
    

##


def add_legend(label=None, colors=None, ax=None, loc='center', artists_size=7, label_size=7, 
    ticks_size=5, bbox_to_anchor=(0.5, 1.1), ncols=None, only_top='all'):
    """
    Draw a legend on axes object.
    """
    if only_top != 'all':
        colors = { k : colors[k] for i, k in enumerate(colors) if i < int(only_top) }
        
    if ncols is None:
        ncols = len(colors) // 2 + 1
    title = label.capitalize() if label is not None else None

    handles = create_handles(colors.keys(), colors=colors.values(), size=artists_size)
    ax.legend(
        handles, 
        colors.keys(), 
        title=title, 
        loc=loc, 
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncols,
        fontsize=ticks_size, 
        title_fontsize=label_size, 
        frameon=False
    )


##


def add_wilcox(df, x, y, pairs, ax, order=None):
    """
    Add statisticatl annotations.
    """
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
    annotator.configure(
        test='Mann-Whitney', 
        text_format='star', 
        show_test_name=False,
        line_height=0.001, 
        text_offset=3
    )
    annotator.apply_and_annotate()


##


def find_n_axes(df, facet, query=None):
    """
    Get the numbers: n_axes.
    """
    idxs = []
    names = []
    n_axes = 0
    cats = df[facet].unique()
    for x in cats:
        idx = df.loc[df[facet] == x, :].index
        if query is not None:
            df_ = df.loc[idx, :].query(query)
        else:
            df_ = df.loc[idx, :]
        if df_.shape[0] > 0:
            n_axes += 1
            idxs.append(idx)
            names.append(f'{facet}: {x}')
        else:
            pass
    return n_axes, idxs, names


##


def remove_ticks(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)


##


def find_n_rows_n_cols(n_axes, n_cols=None):
    """
    Get the numbers: n_rows, n_cols.
    """
    n_cols = n_cols if n_cols is not None else 5
    if n_axes <= 5:
        n_rows = 1; n_cols = n_cols
    else:
        n_rows = 1; n_cols = n_cols
        while n_cols * n_rows < n_axes:
            n_rows += 1
    return n_rows, n_cols


##


def format_ax(ax, title='', xlabel='', ylabel='', 
    xticks=None, yticks=None, rotx=0, roty=0, 
    xlabel_size=None, ylabel_size=None, xticks_size=None, 
    yticks_size=None, title_size=None, log=False
    ):
    """
    Format labels, ticks and stuff.
    """

    if log:
        ax.set_yscale('log')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    if xticks is not None:
        ax.set_xticks([ i for i in range(len(xticks)) ])
        ax.set_xticklabels(xticks)
    if yticks is not None:
        ax.set_yticks([ i for i in range(len(yticks)) ])
        ax.set_yticklabels(yticks)

    if xticks_size is not None:
        ax.xaxis.set_tick_params(labelsize=xticks_size)
    if yticks_size is not None:
        ax.yaxis.set_tick_params(labelsize=yticks_size)

    if xlabel_size is not None:
        ax.xaxis.label.set_size(xlabel_size)
    if ylabel_size is not None:
        ax.yaxis.label.set_size(ylabel_size)

    ax.tick_params(axis='x', labelrotation = rotx)
    ax.tick_params(axis='y', labelrotation = roty)

    if title_size is not None:
        ax.set_title(title, fontdict={'fontsize': title_size})

    return ax


##


def add_labels_on_loc(df, x, y, by, ax=None, s=10):
    """
    Add categorical labels on loc on a scatterplot.
    """
    coords = df.loc[:, [x, y, by]].groupby(by).median()
    for label in coords.index:
        x, y = coords.loc[label, :].tolist()
        ax.text(x, y, label, fontsize=s, weight="bold")


##


def line(df, x, y, c='r', s=1, l=None, ax=None):
    """
    Base line plot.
    """
    ax.plot(df[x], df[y], color=c, label=l, linestyle='-', linewidth=s)
    return ax


##


def scatter(df, x, y, by=None, c='r', s=1.0, a=1, l=None, ax=None, scale_x=None, ordered=False):
    """
    Base scatter plot.
    """
    size = s if isinstance(s, float) or isinstance(s, int) else df[s]

    if ordered and df[by].dtype == 'category':
        try:
            categories = df[by].cat.categories
            df = df.sort_values(by)
        except:
            raise ValueError('Ordered is not a pd.Categorical')

    if isinstance(size, pd.Series) and scale_x is not None:
        size = size * scale_x

    if isinstance(c, str) and by is None:
        ax.scatter(df[x], df[y], color=c, label=l, marker='.', s=size, alpha=a)

    elif isinstance(c, str) and by is not None:
        ax.scatter(df[x], df[y], c=df[by], label=l, marker='.', s=size, 
            cmap=c, alpha=a)

    elif isinstance(c, dict) and by is not None:
        assert all([ x in c for x in df[by].unique() ])
        colors = [ c[x] for x in df[by] ]
        ax.scatter(df[x], df[y], c=colors, label=l, marker='.', s=size, alpha=a)

    else:
        raise ValueError('c needs to be specified as a dict of colors with "by" of a single color.')

    return ax


##


def hist(df, x, n=10, by=None, c='r', a=1, l=None, ax=None, density=False):
    """
    Basic histogram plot.
    """
    if by is None:
       ax.hist(df[x], bins=n, color=c, alpha=a, label=l, density=density)
    elif by is not None and isinstance(c, dict):
        categories = df[by].unique()
        if all([ cat in list(c.keys()) for cat in categories ]):
            for cat in categories:
                df_ = df.loc[df[by] == cat, :]
                ax.hist(df_[x], bins=n, color=c[cat], alpha=a, label=x, density=density)
    else:
        raise ValueError(f'{by} categories do not match provided colors keys')

    return ax


##


def bar(df, y, x=None, by=None, c='grey', s=0.35, a=1, l=None, ax=None, 
    edgecolor=None, annot_size=10):
    """
    Basic bar plot.
    """
    if isinstance(c, str) and by is None:
        x = np.arange(df[y].size)
        ax.bar(x, df[y], align='center', width=s, alpha=a, color=c, edgecolor=edgecolor)
        ax.bar_label(ax.containers[0], padding=0, size=annot_size)

    elif by is not None and x is None and isinstance(c, dict):
        x = np.arange(df[y].size)
        categories = df[by].unique()
        n_cat = len(categories)
        if all([ cat in c for cat in categories ]):
            for i, cat in enumerate(categories):
                height = df[y].values
                idx = [ i for i, x in enumerate(df[by]) if x == cat ]
                height = df[y].values[idx]
                ax.bar(x[idx], height, align='center', width=s, alpha=a, 
                    color=c[cat], edgecolor=edgecolor)
                ax.bar_label(ax.containers[i], padding=0, size=annot_size)

    elif by is not None and x is not None and isinstance(c, dict):
        ax = sns.barplot(data=df, x=x, y=y, hue=by, ax=ax, width=s, 
            palette=list(c.values()), alpha=a)
        ax.legend([], [], frameon=False)
        ax.set(xlabel='', ylabel='')
        ax.set_xticklabels(np.arange(df[x].unique().size))

    else:
        raise ValueError(f'{by} categories do not match provided colors keys')

    return ax


##


def box(df, x, y, by=None, c='grey', a=1, ax=None, with_stats=False,
    pairs=None, order=None, hue_order=None, kwargs={}):
    """
    Base box plot.
    """

    params = {   
        'showcaps' : True,
        'fliersize': 0,
        'boxprops' : {'edgecolor': 'black', 'linewidth': .8}, 
        'medianprops': {"color": "black", "linewidth": 1.5},
        'whiskerprops':{"color": "black", "linewidth": 1.2}
    }

    params = update_params(params, kwargs)
    
    if isinstance(c, str) and by is None:
        sns.boxplot(data=df, x=x, y=y, color=c, ax=ax, saturation=0.7, order=order, **params) 
        ax.set(xlabel='')

    elif isinstance(c, dict) and by is None:
        if all([ True if k in df[x].unique() else False for k in c.keys() ]):
            sns.boxplot(data=df, x=x, y=y, palette=c.values(), ax=ax, saturation=0.7, order=order, **params)
            ax.set(xlabel='')
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')
            
    elif isinstance(c, dict) and by is not None:
        if all([ True if k in df[by].unique() else False for k in c.keys() ]):
            sns.boxplot(data=df, x=x, y=y, palette=c.values(), hue=by, hue_order=hue_order, ax=ax, saturation=0.7, **params)
            ax.legend([], [], frameon=False)
            ax.set(xlabel='')
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')
    elif isinstance(c, str) and by is not None:
        sns.boxplot(data=df, x=x, y=y, hue=by, hue_order=hue_order, ax=ax, saturation=0.7, **params)
        ax.legend([], [], frameon=False)
        ax.set(xlabel='')

    if with_stats:
        add_wilcox(df, x, y, pairs, ax, order=order)

    return ax


##


def strip(df, x, y, by=None, c=None, a=1, l=None, s=5, ax=None, with_stats=False, order=None, pairs=None):

    """
    Base stripplot.
    """
    np.random.seed(123)
    
    if isinstance(c, str):
        ax = sns.stripplot(data=df, x=x, y=y, color=c, ax=ax, size=s, order=order) 
        ax.set(xlabel='')
    
    elif isinstance(c, str) and by is not None:
        g = sns.stripplot(data=df, x=x, y=y, hue=by, palette=c, ax=ax, order=order)
        g.legend_.remove()

    elif isinstance(c, dict) and by is None:
        if all([ True if k in df[x].unique() else False for k in c.keys() ]):
            ax = sns.stripplot(data=df, x=x, y=y, palette=c.values(), ax=ax, size=s, order=order)
            ax.set(xlabel='')
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')
            
    elif isinstance(c, dict) and by is not None:
        if all([ True if k in df[by].unique() else False for k in c.keys() ]):
            ax = sns.stripplot(data=df, x=x, y=y, palette=c.values(), hue=by, ax=ax, size=s, order=order)
            ax.legend([], [], frameon=False)
            ax.set(xlabel='')    
        else:
            raise ValueError(f'{by} categories do not match provided colors keys')

    if with_stats:
        add_wilcox(df, x, y, pairs, ax, order=None)

    return ax


##


def violin(df, x, y, by=None, c=None, a=1, l=None, ax=None, with_stats=False, order=None, pairs=None):
    """
    Base violinplot.
    """
    params = {   
        'showcaps' : False,
        'fliersize': 0,
        'boxprops' : {'edgecolor': 'white', 'linewidth': 0.5}, 
        'medianprops': {"color": "white", "linewidth": 1.2},
        'whiskerprops':{"color": "black", "linewidth": 1}
    }
    
    if isinstance(c, str):
        ax = sns.violinplot(data=df, x=x, y=y, color=c, ax=ax, saturation=0.7, order=order, **params) 
        ax.set(xlabel='', ylabel='')
        ax.set_xticklabels(np.arange(df[x].unique().size))

    elif isinstance(c, dict) and by is None:
        ax = sns.violinplot(data=df, x=x, y=y, palette=c.values(), ax=ax, saturation=0.7, order=order, **params)
        ax.set(xlabel='', ylabel='') 
        ax.set_xticklabels(np.arange(df[x].unique().size))
            
    elif isinstance(c, dict) and by is not None:
        ax = sns.violinplot(data=df, x=x, y=y, palette=c.values(), hue=by, 
            ax=ax, saturation=0.7, **params)
        ax.legend([], [], frameon=False)
        ax.set(xlabel='', ylabel='')
        ax.set_xticklabels(np.arange(df[x].unique().size))

    else:
        raise ValueError(f'{by} categories do not match provided colors keys')

    if with_stats:
        add_wilcox(df, x, y, pairs, ax, order=None)

    return ax


##


def plot_heatmap(df, palette='mako', ax=None, title=None, x_names=True, y_names=True, 
    x_names_size=7, y_names_size=7, xlabel=None, ylabel=None, annot=False, annot_size=5, 
    label=None, shrink=1.0, cb=True):
    """
    Simple heatmap.
    """
    ax = sns.heatmap(data=df, ax=ax, robust=True, cmap=palette, annot=annot, xticklabels=x_names, 
        yticklabels=y_names, fmt='.2f', annot_kws={'size':annot_size}, cbar=cb,
        cbar_kws={'fraction':0.05, 'aspect':35, 'pad': 0.02, 'shrink':shrink, 'label':label}
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=x_names_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=y_names_size)

    return ax


##


def stem_plot(df, x, ax=None):
    """
    Create a stem plot
    """
    ax.hlines(y=df.index, xmin=0, xmax=df[x], color='darkgrey')
    ax.plot(df[x][df[x]>=0], df[x].index[df[x]>=0], "o", color='r')
    ax.plot(df[x][df[x]<0], df[x].index[df[x]<0], "o", color='b')
    ax.axvline(color="black", linestyle="--")
    ax.invert_yaxis()
    return ax


##


def bb_plot(df, cov1=None, cov2=None, show_y=True, legend=True, colors=None, 
    ax=None, with_data=False, **kwargs):
    """
    Stacked percentage plot.
    """
    # Prep data
    df[cov1] = pd.Categorical(df[cov1]).remove_unused_categories()
    df[cov2] = pd.Categorical(df[cov2]).remove_unused_categories()
    data = pd.crosstab(df[cov1], df[cov2], normalize='index')
    data_cum = data.cumsum(axis=1)
    ys = data.index.categories
    labels = data.columns.categories

    # Ax
    if colors is None:
        colors = create_palette(df, cov2, palette='tab10')

    for i, x in enumerate(labels):
        widths = data.values[:,i]
        starts = data_cum.values[:,i] - widths
        ax.barh(ys, widths, left=starts, height=0.95, label=x, color=colors[x])

    # Format
    ax.set_xlim(-0.01, 1.01)
    format_ax(
        ax, 
        title = f'{cov1} by {cov2}',
        yticks='' if not show_y else ys, 
        xlabel='Abundance %'
    )

    if legend:
        add_legend(label=cov2, colors=colors, ax=ax, only_top=10, ncols=1,
            loc='upper left', bbox_to_anchor=(1.01,1), ticks_size=7  
        )
    
    if with_data:
        return ax, data
    else:
        return ax
    

##


def rank_plot(df, cov=None, ascending=False, n_annotated=25, title=None, ylabel=None, ax=None, fig=None):
    """
    Annotated scatterplot.
    """
    s = df[cov].sort_values(ascending=ascending)
    x = np.arange(df.shape[0])
    y = s.values
    labels = s[:n_annotated].index
    ax.plot(x, y, '.')
    ta.allocate_text(fig, ax, x[:n_annotated], y[:n_annotated], labels, x_scatter=x, y_scatter=y,
        linecolor='black', textsize=8, max_distance=0.5, linewidth=0.5, nbr_candidates=100)
    format_ax(ax, title=title, xlabel='rank', ylabel=ylabel)

    return ax



##


# def prep_things_for_umap(top_runs_per_sample, i, solutions, connectivities, path_main=None):
#     """
#     Utility used in leiden performance viz.
#     """
#     # Get top solutions
#     d_run = top_runs_per_sample.iloc[i, :].to_dict()
# 
#     # Prepare ingredients for embs calculations
#     s = d_run['sample']
#     a = '_'.join(d_run['analysis'].split('_')[1:])
# 
#     path_ = path_main + f'results_and_plots/classification_performance/top_3/{s}/{a}/cell_x_var_hclust.pickle'
# 
#     with open(path_, 'rb') as f:
#         d_cell_x_var = pickle.load(f)
# 
#     cells = d_cell_x_var['cells']
#     variants = d_cell_x_var['vars']
# 
#     afm = read_one_sample(path_main, sample=s)
#     X = afm[cells, variants].X.copy()
# 
#     conn_name = f'{d_run["analysis"]}_{d_run["with_nans"]}_{d_run["metric"]}_None'
#     leiden_pickle_name = f'{d_run["analysis"]}_{d_run["with_nans"]}_{d_run["metric"]}_None|{d_run["k"]}|{d_run["res"]}'
# 
#     labels, true_clones, ARI = solutions[s][leiden_pickle_name]
#     conn = connectivities[s][conn_name]
# 
#     return X, conn, cells, true_clones, labels, ARI, d_run


##


def bb_plot(df, cov1=None, cov2=None, show_y=True, legend=True, colors=None, 
    ax=None, with_data=False, **kwargs):
    """
    Stacked percentage plot.
    """
    # Prep data
    df[cov1] = pd.Categorical(df[cov1]).remove_unused_categories()
    df[cov2] = pd.Categorical(df[cov2]).remove_unused_categories()
    data = pd.crosstab(df[cov1], df[cov2], normalize='index')
    data_cum = data.cumsum(axis=1)
    ys = data.index.categories
    labels = data.columns.categories

    # Ax
    if colors is None:
        colors = create_palette(df, cov2, palette='tab10')

    for i, x in enumerate(labels):
        widths = data.values[:,i]
        starts = data_cum.values[:,i] - widths
        ax.barh(ys, widths, left=starts, height=0.95, label=x, color=colors[x])

    # Format
    ax.set_xlim(-0.01, 1.01)
    format_ax(
        ax, 
        title = f'{cov1} by {cov2}',
        yticks='' if not show_y else ys, 
        xlabel='Frequency %'
    )

    if legend:
        add_legend(label=cov2, colors=colors, ax=ax, only_top=10, ncols=1,
            loc='upper left', bbox_to_anchor=(1.01,1), ticks_size=7  
        )
    
    if with_data:
        return ax, data
    else:
        return ax
    

##

   
def packed_circle_plot(df, covariate=None, ax=None, color='b', alpha=0.4,
        annotate=False, fontsize=5, colors_dict=None, 
        annot_treshold=0.1, spacing=0.95, linewidth=1.3,
        edgecolor=None, names_to_annotate=None
    ):
    """
    Circle plot. Packed.
    """
    
    df = df.sort_values(covariate, ascending=False)
    circles = circlify(
        df[covariate].to_list(),
        show_enclosure=True, 
        target_enclosure=Circle(x=0, y=0, r=1)
    )
    
    lim = max(
        max(
            abs(c.x) + c.r,
            abs(c.y) + c.r,
        )
        for c in circles
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    for name, circle in zip(df.index[::-1], circles): # Don't know why, but it reverses...
        x, y, r = circle
        color = colors_dict[name] if colors_dict is not None else color
        ax.add_patch(
            plt.Circle((x, y), r*spacing, alpha=alpha, linewidth=linewidth, 
                fill=True, 
                edgecolor=edgecolor if edgecolor is not None else color, 
                facecolor=color
            )
        )
        
        if annotate:
            cov = df.loc[name, covariate]
            if names_to_annotate is None:
                if cov > annot_treshold:
                    ax.annotate(
                        f'{name[:5]}: {df.loc[name, covariate]:.2f}', 
                        (x,y), 
                        va='center', ha='center', fontsize=fontsize
                    )
            else:
                if name in names_to_annotate:
                    ax.annotate(
                        f'{name[:5]}: {df.loc[name, covariate]:.2f}', 
                        (x,y), 
                        va='center', ha='center', fontsize=fontsize
                    )

    ax.axis('off')
    
    return ax