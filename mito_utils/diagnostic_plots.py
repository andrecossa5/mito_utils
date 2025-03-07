"""
Utils and plotting functions to visualize and inspect SNVs from a MAESTER experiment and maegatk output.
"""

import gc
from matplotlib.ticker import FixedLocator, FuncFormatter
from itertools import product
from .preprocessing import *
from mito_utils.utils import *
from mito_utils.plotting_base import *


## 


# Current diagnosti plots
def vars_AF_dist(afm, ax=None, color='b', **kwargs):
    """
    Ranked AF distributions (VG-like).
    """
    X = afm.X.A
    for i in range(X.shape[1]):
        x = X[:,i]
        x = np.sort(x)
        ax.plot(x, '-', color=color, **kwargs)

    format_ax(ax=ax, xlabel='Cells (ranked)', ylabel='Allelic Frequency (AF)')

    return ax


##


def plot_ncells_nAD(afm, ax=None, title=None, xticks=None, yticks=None, s=5, c='k', alpha=.7, **kwargs):
    """
    Plots similar to Weng et al., 2024, followed by the two commentaries from Lareau and Weng.
    n+ cells vs n 
    """

    annotate_vars(afm, overwrite=True)
    ax.plot(afm.var['Variant_CellN'], afm.var['mean_AD_in_positives'], 'o', c=c, markersize=s, alpha=alpha, **kwargs)
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    xticks = [0,1,2,5,10,20,40,80,160,320,640] if xticks is None else xticks
    yticks = [0,1,2,4,8,16,32,64,132,264] if yticks is None else yticks
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))

    def integer_formatter(val, pos):
        return f'{int(val)}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(integer_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(integer_formatter))
    ax.set(xlabel='n +cells', ylabel='Mean n ALT UMI / +cell', title='' if title is None else title)

    return ax


##


def mut_profile(mut_list, figsize=(6,3)):
    """
    MutationProfile_bulk (Weng et al., 2024).
    """

    ref_df = load_mut_spectrum_ref()
    called_variants = [ ''.join(x.split('_')) for x in mut_list ]
        
    ref_df['called'] = ref_df['variant'].isin(called_variants)
    total = len(ref_df)
    total_called = ref_df['called'].sum()

    grouped = ref_df.groupby(['three_plot', 'group_change', 'strand'])
    prop_df = grouped.agg(
        observed_prop_called=('called', lambda x: x.sum() / total_called),
        expected_prop=('variant', lambda x: x.count() / total),
        n_obs=('called', 'sum'),
        n_total=('variant', 'count')
    ).reset_index()

    prop_df['fc_called'] = prop_df['observed_prop_called'] / prop_df['expected_prop']
    prop_df = prop_df.set_index('three_plot')
    prop_df['group_change'] = prop_df['group_change'].map(lambda x: '>'.join(list(x)))


    fig, axs = plt.subplots(1, prop_df['group_change'].unique().size, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.1},
                            constrained_layout=True)
    strand_palette = {'H': '#05A8B3', 'L': '#D76706'}

    for i,x in enumerate(prop_df['group_change'].unique()):
        ax = axs.ravel()[i]
        df_ = prop_df.query('group_change==@x')
        bar(df_, 'n_obs', by='strand', c=strand_palette, ax=ax, s=1, a=.8, annot=False)
        format_ax(ax, xticks=[], xlabel=x, ylabel='Substitution rate' if i==0 else '', title=f'n: {df_["n_obs"].sum()}')

    add_legend(ax=axs.ravel()[0], colors=strand_palette, ncols=1, loc='upper left', bbox_to_anchor=(0,1), label='Strand', ticks_size=6)
    fig.tight_layout()

    return fig


##


def MT_coverage_polar(df, var_subset=None, ax=None, n_xticks=6, xticks_size=7, 
                    yticks_size=2, xlabel_size=6, ylabel_size=9, kwargs_main={}, kwargs_subset={},):
    
    """
    Plot coverage and muts across postions.
    """
    
    kwargs_main_ = {'c':'#494444', 'linestyle':'-', 'linewidth':.7}
    kwargs_subset_ = {'c':'r', 'marker':'+', 'markersize':10, 'linestyle':''}
    kwargs_main_.update(kwargs_main)
    kwargs_subset_.update(kwargs_subset)

    x = df.mean(axis=0)

    theta = np.linspace(0, 2*np.pi, len(x))
    ticks = [ 
        int(round(x)) \
        for x in np.linspace(1, df.shape[1], n_xticks) 
    ][:7]

    ax.plot(theta, np.log10(x), **kwargs_main_)

    if var_subset is not None:
        var_pos = var_subset.map(lambda x: int(x.split('_')[0]))
        test = x.index.isin(var_pos)
        print(test.sum())
        ax.plot(theta[test], np.log10(x[test]), **kwargs_subset_)

    ax.set_theta_offset(np.pi/2)
    ax.set_xticks(np.linspace(0, 2*np.pi, n_xticks-1, endpoint=False))#, fontsize=1)
    ax.set_xticklabels(ticks[:-1], fontsize=xticks_size)

    ax.set_yticklabels([])
    for tick in np.arange(-1,4,1):
        ax.text(0, tick, str(tick), ha='center', va='center', fontsize=yticks_size)

    ax.text(0, 1.5, 'n UMIs', ha='center', va='center', fontsize=xlabel_size, color='black')
    ax.text(np.pi, 4, 'Position (bp)', ha='center', va='center', fontsize=ylabel_size, color='black')

    ax.spines['polar'].set_visible(False)

    return ax



##


def MT_coverage_by_gene_polar(cov, sample=None, subset=None, ax=None):
    """
    MT coverage, with annotated genes, in polar coordinates.
    """
    
    if subset is not None:
        cov = cov.query('cell in @subset')
    cov['pos'] = pd.Categorical(cov['pos'], categories=range(1,16569+1))
    cov = cov.pivot_table(index='cell', columns='pos', values='n', dropna=False, fill_value=0)

    df_mt = pd.DataFrame(MAESTER_genes_positions, columns=['gene', 'start', 'end']).set_index('gene').sort_values('start')

    x = cov.mean(axis=0)
    median_target = cov.loc[:,mask_mt_sites(cov.columns)].median(axis=0).median()
    median_untarget = cov.loc[:,~mask_mt_sites(cov.columns)].median(axis=0).median()
    theta = np.linspace(0, 2*np.pi, cov.shape[1])
    colors = { k:v for k,v in zip(df_mt.index, sc.pl.palettes.default_102[:df_mt.shape[0]])}
    ax.plot(theta, np.log10(x), '-', linewidth=.7, color='grey')
    idx = np.arange(1,x.size+1)

    for gene in colors:
        start, stop = df_mt.loc[gene, ['start', 'end']].values
        test = (idx>=start) & (idx<=stop)
        ax.plot(theta[test], np.log10(x[test]), color=colors[gene], linewidth=1.5)

    ticks = [ int(round(x)) for x in np.linspace(1, cov.shape[1], 8) ][:7]
    ax.set_theta_offset(np.pi/2)
    ax.set_xticks(np.linspace(0, 2*np.pi, 7, endpoint=False))
    ax.set_xticklabels(ticks)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_rlabel_position(0) 
    ax.set(xlabel='Position (bp)', title=f'{sample}\nTarget: {median_target:.2f}, untarget: {median_untarget:.2f}')

    return ax


##


##------------------------------------------------- LEGACY


def sturges(x):   
    return round(1 + 2 * 3.322 * np.log(len(x))) 


##


def cell_n_sites_covered_dist(afm, ax=None, color='k', title=None):
    """
    n covered positions/total position. Cell distribution.
    """
    df_ = pd.DataFrame(
        np.sum(afm.uns['per_position_coverage']>0, axis=1), 
        columns=['n']
    )
    sns.kdeplot(df_['n'], ax=ax, color='k', fill=True, alpha=.1, linewidth=1)

    if title is None:
        t = 'Covered sites (total=16569), across cell'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='n sites covered', ylabel='Density', reduced_spines=True)

    r = (df_['n'].min(), df_['n'].max())
    ax.axvline(x=np.median(df_["n"]), color='r', linewidth=3)
    ax.text(0.05, 0.9, f'Median: {round(np.median(df_["n"]))}', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'Min-max: {r[0]}-{r[1]}', transform=ax.transAxes)
    ax.text(0.05, 0.8, f'Total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


##


def cell_n_vars_detected_dist(afm, ax=None, color='k', title=None):
    """
    n covered variants. Cell distribution.
    """
    df_ = pd.DataFrame(np.sum(afm.X>0, axis=1), columns=['n'])
    sns.kdeplot(df_['n'], ax=ax, color='k', fill=True, alpha=.1, linewidth=1)
    ax.axvline(x=np.median(df_["n"]), color='r', linewidth=3)

    if title is None:
        t = 'n detected variants (total=16569*3), across cell'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='n variants detected', ylabel='Density')

    r = (df_['n'].min(), df_['n'].max())
    ax.text(0.6, 0.9, f'Median: {round(np.median(df_["n"]))}', transform=ax.transAxes)
    ax.text(0.6, 0.85, f'Min-max: {r[0]}-{r[1]}', transform=ax.transAxes)
    ax.text(0.6, 0.8, f'Total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


##


def mean_site_quality_cell_dist(afm, ax=None, color='k', title=None):
    """
    Mean base quality per site, distribution across cells.
    """
    df_ = (
        pd.Series(np.nanmean(afm.uns['per_position_quality'], axis=1))
        .to_frame()
        .rename(columns={0:'qual'})
    )
    sns.kdeplot(df_['qual'], ax=ax, color=color, fill=True, alpha=.1, linewidth=1)
    ax.axvline(x=np.median(df_["qual"]), color='r', linewidth=3)

    if title is None:
        t = 'Mean site quality, across cell'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Phred score', ylabel='Density')

    median = np.median(df_['qual'])
    r = (df_['qual'].min(), df_['qual'].max())
    ax.text(0.05, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.8, f'Total cells: {afm.shape[0]}', transform=ax.transAxes)

    return ax


##


# Site level diagnostics
def mean_position_coverage_dist(afm, ax=None, color='k', title=None, xlim=(-50, 1000)):
    """
    Median site coverage across cells, distribution.
    """
    df_ = (
        pd.Series(np.mean(afm.uns['per_position_coverage'], axis=0))
        .to_frame()
        .rename(columns={0:'cov'})
    )
    sns.kdeplot(df_['cov'], ax=ax, color=color, fill=True, alpha=.1, linewidth=1)
    ax.axvline(x=np.median(df_["cov"]), color='r', linewidth=3)
    ax.set_xlim(xlim)

    if title is None:
        t = 'Mean (across cells) position coverage (nUMIs)'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='total nUMIs', ylabel='Density')

    median = round(np.median(df_['cov']))
    r = (df_['cov'].min(), df_['cov'].max())
    ax.text(0.5, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.5, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)
    ax.text(0.5, 0.8, f'Total positions: {afm.uns["per_position_coverage"].shape[1]}', 
        transform=ax.transAxes)

    return ax


##


def mean_position_quality_dist(afm, ax=None, color='k', title=None):
    """
    Median site quality across cells, distribution.
    """
    df_ = (
        pd.Series(np.mean(afm.uns['per_position_quality'], axis=0))
        .to_frame()
        .rename(columns={0:'qual'})
    )
    sns.kdeplot(df_['qual'], ax=ax, color=color, fill=True, alpha=.1, linewidth=1)
    ax.axvline(x=np.median(df_["qual"]), color='r', linewidth=3)

    if title is None:
        t = 'Mean (across cells) position quality'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='Phred score', ylabel='Density')

    median = np.median(df_['qual'])
    r = (df_['qual'].min(), df_['qual'].max())
    ax.text(0.08, 0.9, f'Median: {median:.2f}', transform=ax.transAxes)
    ax.text(0.08, 0.85, f'Min-max: {r[0]:.2f}-{r[1]:.2f}', transform=ax.transAxes)
    ax.text(0.08, 0.8, f'Total positions: {afm.uns["per_position_coverage"].shape[1]}', 
        transform=ax.transAxes)

    return ax


##


def vars_n_positive_dist(afm, ax=None, color='k', title=None, xlim=(-10,100)):
    """
    Percentage of positive cells per variant, distribution.
    """
    df_ = pd.DataFrame(
        np.sum(afm.X > 0, axis=0), 
        columns=['n']
    )
    sns.kdeplot(df_['n'], ax=ax, color=color, fill=True, alpha=.1, linewidth=1)
    ax.axvline(x=np.median(df_["n"]), color='r', linewidth=3)
    ax.set_xlim(xlim)

    if title is None:
        t = 'n positive cells, per variant'
    else:
        t = title
    format_ax(ax=ax, title=t, xlabel='n cells+', ylabel='Density')
    
    r = (df_['n'].min(), df_['n'].max())
    ax.text(0.6, 0.9, f'Median: {round(np.median(df_["n"]))}', transform=ax.transAxes)
    ax.text(0.6, 0.85, f'Min-max: {r[0]}-{r[1]}', transform=ax.transAxes)
    ax.text(0.6, 0.80, f'total cells: {afm.shape[0]}', transform=ax.transAxes)

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


def plot_exclusive_variant(a_cells, var, vois_df=None, ax=None):

    idx = np.argsort(a_cells[:, var].X.toarray().flatten())
    a_cells = a_cells[idx, :]
    x = a_cells[:, var].X.toarray().flatten()
    first_m_zero = np.argmax(x>0)
    p = (first_m_zero / x.size) * 100

    if vois_df is not None:
        VMR_rank = vois_df.loc[var, 'VMR_rank']
        title = f'Percentile 1st>0 {round(p)}th, VMR rank {VMR_rank}'
    else:
        title = f'Percentile 1st>0 {round(p)}th'

    ax.plot(x, 'k.-')
    ax.set(title=title, xlabel='Cell rank', ylabel='AF')

    return ax


##