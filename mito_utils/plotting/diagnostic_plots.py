"""
Utils and plotting functions to visualize and inspect SNVs from a MAESTER experiment and maegatk output.
"""

import gc
from itertools import product

from ..preprocessing.preprocessing import *
from ..utils.helpers import *
from .plotting_base import *


##


# Cell level diagnostics
def sturges(x):   
    return round(1 + 2 * 3.322 * np.log(len(x))) 


##


def cell_n_sites_covered_dist(afm, ax=None, color='b', title=None):
    """
    n covered positions/total position. Cell distribution.
    """
    df_ = pd.DataFrame(
        np.sum(afm.uns['per_position_coverage'] > 0, axis=1), 
        columns=['n']
    )
    hist(df_, 'n', n=sturges(df_['n']), ax=ax, c=color)

    if title is None:
        t = 'n covered sites (total=16569), across cell'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='n sites covered', ylabel='n cells')

    r = (df_['n'].min(), df_['n'].max())
    ax.text(0.05, 0.9, f'Median: {round(np.median(df_["n"]))}', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'Min-max: {r[0]}-{r[1]}', transform=ax.transAxes)
    ax.text(0.05, 0.8, f'Total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


##


def cell_n_vars_detected_dist(afm, ax=None, color='b', title=None):
    """
    n covered variants. Cell distribution.
    """
    df_ = pd.DataFrame(np.sum(afm.X > 0, axis=1), columns=['n'])
    hist(df_, 'n', n=sturges(df_['n']), ax=ax, c=color)

    if title is None:
        t = 'n detected variants (total=16569*3), across cell'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='n variants detected', ylabel='n cells')

    r = (df_['n'].min(), df_['n'].max())
    ax.text(0.6, 0.9, f'Median: {round(np.median(df_["n"]))}', transform=ax.transAxes)
    ax.text(0.6, 0.85, f'Min-max: {r[0]}-{r[1]}', transform=ax.transAxes)
    ax.text(0.6, 0.8, f'Total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


##


def mean_site_quality_cell_dist(afm, ax=None, color='b', title=None):
    """
    Mean base quality per site, distribution across cells.
    """
    df_ = (
        pd.Series(np.nanmean(afm.uns['per_position_quality'], axis=1))
        .to_frame()
        .rename(columns={0:'qual'})
    )
    hist(df_, 'qual', n=sturges(df_['qual']), ax=ax, c=color)

    if title is None:
        t = 'Mean site quality, across cell'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Phred score', ylabel='n cells')

    median = np.median(df_['qual'])
    r = (df_['qual'].min(), df_['qual'].max())
    ax.text(0.6, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.6, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)
    ax.text(0.6, 0.8, f'Total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


##


# Site level diagnostics
def mean_position_coverage_dist(afm, ax=None, color='b', title=None):
    """
    Median site coverage across cells, distribution.
    """
    df_ = (
        pd.Series(np.mean(afm.uns['per_position_coverage'], axis=0))
        .to_frame()
        .rename(columns={0:'cov'})
    )
    hist(df_, 'cov', n=sturges(df_['cov']), ax=ax, c=color)

    if title is None:
        t = 'Mean (across cells) position coverage (nUMIs)'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='total nUMIs', ylabel='n sites')

    ax.set_xlim((-50, 800))
    median = round(np.median(df_['cov']))
    r = (df_['cov'].min(), df_['cov'].max())
    ax.text(0.5, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.5, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)
    ax.text(0.5, 0.8, f'Total positions: {afm.uns["per_position_coverage"].shape[1]}', 
        transform=ax.transAxes)

    return ax


##


def mean_position_quality_dist(afm, ax=None, color='b', title=None):
    """
    Median site quality across cells, distribution.
    """
    df_ = (
        pd.Series(np.mean(afm.uns['per_position_quality'], axis=0))
        .to_frame()
        .rename(columns={0:'qual'})
    )
    hist(df_, 'qual', n=sturges(df_['qual']), ax=ax, c=color)

    if title is None:
        t = 'Mean (across cells) position quality'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Phred score', ylabel='n sites')

    median = np.median(df_['qual'])
    r = (df_['qual'].min(), df_['qual'].max())
    ax.text(0.08, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.08, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)
    ax.text(0.08, 0.8, f'Total positions: {afm.uns["per_position_coverage"].shape[1]}', 
        transform=ax.transAxes)

    return ax


##


# Variant level diagnostics
def vars_n_positive_dist(afm, ax=None, color='b', title=None):
    """
    Percentage of positive cells per variant, distribution.
    """
    df_ = pd.DataFrame(
        np.sum(afm.X > 0, axis=0), 
        columns=['n']
    )

    hist(df_, 'n', n=sturges(df_['n']), ax=ax, c=color)

    if title is None:
        t = 'n positive cells, per variant'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='n cells+', ylabel='n variants')
    
    #ax.set_ylim((0, 1.5))
    r = (df_['n'].min(), df_['n'].max())
    ax.text(0.6, 0.9, f'Median: {round(np.median(df_["n"]))}', transform=ax.transAxes)
    ax.text(0.6, 0.85, f'Min-max: {r[0]}-{r[1]}', transform=ax.transAxes)
    ax.text(0.6, 0.80, f'total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


#


def vars_AF_dist(afm, ax=None, color='b', title=None):
    """
    Ranked AF distributions (VG-like).
    """
    to_plot = afm.X.copy()
    to_plot[np.isnan(to_plot)] = 0

    for i in range(to_plot.shape[1]):
        x = to_plot[:, i]
        x = np.sort(x)
        ax.plot(x, '--', color=color, linewidth=0.5)

    if title is None:
        t = 'Ranked AF distributions, per variant'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Cell rank', ylabel='AF')

    return ax


##


def strand_concordances(orig, variants):
    """
    Utils for strand concordances calculation.
    """
    strand_conc = {}
    for base in ['A', 'C', 'T', 'G']:
        fw = orig.layers[f'{base}_counts_fw'].toarray()
        rev = orig.layers[f'{base}_counts_rev'].toarray()
        corr = []
        for i in range(fw.shape[1]):
            x = fw[:, i]
            y = rev[:, i]
            corr.append(np.corrcoef(x, y)[0,1])
        corr = np.array(corr)
        corr[np.isnan(corr)] = 0
        strand_conc[base] = corr

    strand_conc = pd.DataFrame(
        strand_conc).reset_index().rename(
        columns={'index':'MT_site'}).melt(
        id_vars='MT_site', var_name='base', value_name='corr'
    )
    strand_conc['MT_site'] = strand_conc['MT_site'].astype(str)
    strand_conc['var'] = strand_conc.loc[:, ['MT_site', 'base']].apply('_'.join, axis=1)
    strand_conc = strand_conc.drop(['MT_site', 'base'], axis=1).set_index('var')
    strand_conc = strand_conc.loc[variants, :]

    return strand_conc


##


def vars_strand_conc_dist(orig, variants, ax=None, color='b', title=None):
    """
    Variant strand concordances distribution.
    """
    orig.layers['A_counts_fw'] = orig.X

    # Calculation of strand concordances, at all {site}_{base} combo.
    strand_conc = strand_concordances(orig, variants)

    # Viz
    hist(strand_conc, 'corr', n=sturges(strand_conc['corr']), ax=ax, c=color)

    if title is None:
        t = 'Strand concordances distribution'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel="Pearson's r", ylabel='n variants') 

    median = np.median(strand_conc['corr'])
    r = (strand_conc['corr'].min(), strand_conc['corr'].max())
    ax.text(0.6, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.6, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)

    return ax


##


def AF_mean_strand_conc_corr(orig, variants, afm, ax=None, color='b', title=None):
    """
    Mean AF/strand concordance relationship.
    """
    strand_conc = strand_concordances(orig, variants)

    df_ = strand_conc.assign(mean=np.nanmean(afm.X, axis=0))

    scatter(df_, 'mean', 'corr', c=color, ax=ax)

    if title is None:
        t = 'AF mean-strand concordance trend, per variant'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Mean', ylabel='pho')

    test = df_['mean'] < 0.6
    x = df_['mean'][test]
    y = df_['corr'][test]
    fitted_coefs = np.polyfit(x, y, 1)
    y_hat = np.poly1d(fitted_coefs)(x)
    ax.plot(x, y_hat, linestyle='--', color='black')
    corr = np.corrcoef(x, y)[0,1]
    ax.text(0.7, 0.9, f"Pearson's r: {corr:.2f}", transform=ax.transAxes)

    return ax


##


def AF_mean_var_corr(afm, ax=None, color='b', title=None):
    """
    Mean AF/strand concordance relationship.
    """
    df_ = pd.DataFrame(
        data = np.stack((np.nanmean(afm.X, axis=0), np.nanvar(afm.X, axis=0)), axis=1),
        columns = ['mean', 'variance']
    )

    scatter(df_, 'mean', 'variance', c='b', ax=ax)

    if title is None:
        t = 'AF mean-variance trend'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Mean', ylabel='Variance')

    test = df_['mean'] < 0.6
    x = df_['mean'][test]
    y = df_['variance'][test]
    fitted_coefs = np.polyfit(x, y, 1)
    y_hat = np.poly1d(fitted_coefs)(x)
    ax.plot(x, y_hat, linestyle='--', color='black')
    corr = np.corrcoef(x, y)[0,1]
    ax.text(0.7, 0.9, f"Pearson's r: {corr:.2f}", transform=ax.transAxes)

    return ax


##


def positive_events_by_var_type(afm, orig, ax=None, color=None, title=None):
    """
    % of +events over total + events, stratified by variant type.
    """
    # Compute
    bases = ['A', 'C', 'T', 'G']
    var_types = [ '>'.join(x) for x in product(bases, bases) if x[0] != x[1] ]

    var_type = []
    for x in afm.var_names:
        idx = x.split('_')[0]
        mt_base = x.split('_')[1]
        ref_base = orig.var.loc[idx, 'refAllele']
        t = '>'.join([ref_base, mt_base])
        var_type.append(t)

    afm.var['var_type'] = var_type

    n_positive = {}
    total_positive_events = np.sum(afm.X > 0)
    for x in afm.var['var_type'].unique():
        if not x.startswith('N'):
            test = afm.var['var_type'] == x
            n = np.sum(afm[:, test].X.toarray() > 0) / total_positive_events
            n_positive[x] = n

    # Viz 
    df_ = pd.Series(n_positive).to_frame().rename(
        columns={0:'freq'}).sort_values(by='freq', ascending=False)
    df_['freq'] = df_['freq'].map(lambda x: round(x*100, 2))

    bar(df_, 'freq', ax=ax, s=0.75, c=color, annot_size=8)

    if title is None:
        t = '\n% of + events over total + events'
    else:
        t = title
    format_ax(ax=ax, title=t, xticks=df_.index, ylabel='\n%')

    return ax


##


def viz_clone_variants(afm, clone_name, sample=None, path=None, filtering=None, 
    min_cell_number=None, min_cov_treshold=None, model=None, figsize=(12, 10)):
    """
    Visualization summary of the properties of a clone distiguishing variants, within some analysis context
    """

    # Read clone classification report
    file_name = f'clones_{sample}_{filtering}_{min_cell_number}_{min_cov_treshold}_{model}_f1.xlsx'
    class_df = pd.read_excel(path + file_name, index_col=0)
    class_df = class_df.loc[class_df['comparison'] == f'{clone_name}_vs_rest']

    # Filter sample AFM as done in the classification picked
    a_cells, a = filter_cells_and_vars(
        afm, 
        min_cell_number=min_cell_number,
        min_cov_treshold=min_cov_treshold,
        variants=class_df.index
    )
    gc.collect()

    ##

    # Draw the figure: top clone distinguishing features

    # Figure
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # Sub 1: clone size in its sample
    colors = {'other':'grey', clone_name:'red'}
    df_ = a_cells.obs.groupby('GBC').size().to_frame().rename(
        columns={0:'n_cells'}).sort_values('n_cells', ascending=False).assign(
        feat=lambda x: np.where(x.index == clone_name, clone_name, 'other')
        )
    bar(df_, 'n_cells', by='feat', c=colors, s=0.75, ax=axs[0,0])
    format_ax(ax=axs[0,0], 
        title=f'{sample} clonal abundances', 
        xticks='', xlabel='Clones', ylabel='n cells', xsize=8, rotx=90
    )
    handles = create_handles(colors.keys(), colors=colors.values())
    axs[0,0].legend(handles, colors.keys(), loc='upper right', 
        title='Clone', frameon=False, bbox_to_anchor=(0.98, 0.95)
    )
    axs[0,0].text(0.6, 0.60, f"min_cell_number: {min_cell_number}", transform=axs[0,0].transAxes)
    axs[0,0].text(0.6, 0.55, f"min_cov_treshold: {min_cov_treshold}", transform=axs[0,0].transAxes)

    ##

    # # Sub 2: variants stats, within selected cells 
    # Density, Median coverage and AF for: 
    # a) all muts; b) muts selected in the picked analysis; c) and top10 ranked for feature importance for
    # the clone
    df_vars = summary_stats_vars(a_cells)
    df_vars['status'] = np.where(df_vars.index.isin(class_df.index), 'selected', 'non-selected')

    if (df_vars['status'] == 'selected').sum() > 10:
        idx = np.where((df_vars['status'].values == 'selected') & df_vars.index.isin(class_df.index[:10]))[0]
        idx = df_vars.index[idx]
        df_vars.loc[idx, ['status']] = 'top10'
        colors = {'non-selected':'grey', 'selected':'red', 'top10':'black'}
        scatter(df_vars.query('status == "non-selected"'), 'median_coverage', 'median_AF', s=10, c=colors['non-selected'], ax=axs[0,1])
        scatter(df_vars.query('status == "selected"'), 'median_coverage', 'median_AF', s=10, c=colors['selected'], ax=axs[0,1])
        scatter(df_vars.query('status == "top10"'), 'median_coverage', 'median_AF', s=10, c=colors['top10'], ax=axs[0,1])
    else:
        colors = {'non-selected':'grey', 'selected':'red'}
        scatter(df_vars.query('status == "non-selected"'), 'median_coverage', 'median_AF', s=10, c=colors['non-selected'], ax=axs[0,1])
        scatter(df_vars.query('status == "selected"'), 'median_coverage', 'median_AF', s=10, c=colors['selected'], ax=axs[0,1])
    
    handles = create_handles(colors.keys(), colors=colors.values())
    axs[0,1].legend(handles, colors.keys(), loc='upper right', 
        bbox_to_anchor=(0.90, 0.95), ncol=1, frameon=False, title='Variant'
    )

    format_ax(df_vars, ax=axs[0,1], title='Variants properties', xlabel='median_coverage', ylabel='median_AF')
    axs[0,1].set(xlim=(-5, 200), ylim=(-0.01, 0.4))

    n_non_selected = df_vars.query('status == "non-selected"').shape[0]
    n_selected = df_vars.query('status == "selected"').shape[0]
    axs[0,1].text(0.6, 0.60, f"n non-selected: {n_non_selected}", transform=axs[0,1].transAxes)
    axs[0,1].text(0.6, 0.55, f"n selected: {n_selected}", transform=axs[0,1].transAxes)

    axins = inset_axes(axs[0,1], width="32%", height="30%", borderpad=2.5,
        bbox_transform=axs[0,1].transAxes, loc=4
    )

    if (df_vars['status'] == 'top10').any():
        violin(
            df_vars.query('status in ["selected", "top10"]').loc[:, 
            ['density', 'median_AF', 'status']].melt(
                id_vars='status', var_name='feature', value_name='value'
            ), 
            'feature', 'value', by='status', c={'selected':colors['selected'], 'top10':colors['top10']}, ax=axins
        )
    else:
        violin(
            df_vars.query('status in ["non-selected", "selected"]').loc[:, 
            ['density', 'median_AF', 'status']].melt(
                id_vars='status', var_name='feature', value_name='value'
            ), 
            'feature', 'value', by='status', c={'non-selected':colors['non-selected'], 'selected':colors['selected']}, ax=axins
        )

    format_ax(ax=axins, xticks=['density', 'median_AF'])

    ##

    # Sub3: VAF profile for: 
    # 1) non-selected vars, 2) vars selected in the picked analysis and 3) top10 vars 
    to_plot = a_cells.copy()
    to_plot.X[np.isnan(to_plot.X)] = 0
    gc.collect()

    if (df_vars['status'] == 'top10').any():
        vars_non_selected = df_vars.query('status == "non-selected"').index
        vars_selected = df_vars.query('status == "selected"').index
        vars_top10 = df_vars.query('status == "top10"').index

        for i, var in enumerate(to_plot.var_names):
            x = to_plot.X[:, i]
            x = np.sort(x)
            if var in vars_non_selected:
                axs[1,0].plot(x, '--', color=colors['non-selected'], linewidth=0.01)
            elif var in vars_selected:
                axs[1,0].plot(x, '--', color=colors['selected'], linewidth=0.3)
            elif var in vars_top10:
                axs[1,0].plot(x, '--', color=colors['top10'], linewidth=2)
    else:
        vars_non_selected = df_vars.query('status == "non-selected"').index
        vars_selected = df_vars.query('status == "selected"').index

        for i, var in enumerate(to_plot.var_names):
            x = to_plot.X[:, i]
            x = np.sort(x)
            if var in vars_non_selected:
                axs[1,0].plot(x, '--', color=colors['non-selected'], linewidth=0.01)
            elif var in vars_selected:
                axs[1,0].plot(x, '--', color=colors['selected'], linewidth=0.5)

    format_ax(ax=axs[1,0], title='Ranked AFs', xlabel='Cell rank', ylabel='AF')

    # handles = create_handles(colors.keys(), marker='o', colors=colors.values(), size=10, width=0.5)
    # axs[1,0].legend(handles, colors.keys(), title='Variant', loc='upper left', 
    #     bbox_to_anchor=(0.05, 0.95), ncol=1, frameon=False
    # )

    ##

    # Sub4: Feature importance of top10 muts
    stem_plot(class_df, 'effect_size', ax=axs[1,1])
    format_ax(ax=axs[1,1], yticks='', ylabel='Selected variants', xlabel='Feature importance', 
        title=f"Analysis: {filtering}_{min_cell_number}_{min_cov_treshold}_{model}")
    top_vars = class_df.index[:3]
    axs[1,1].text(0.25, 0.1, f"Top 3 variants: {top_vars[0]}, {top_vars[1]}, {top_vars[2]}", transform=axs[1,1].transAxes)

    # Save
    fig.suptitle(f'{clone_name} clone features')

    return fig



