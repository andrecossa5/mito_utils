"""
Discretization. NB and tresholds.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.genotyping import *
from mito_utils.genotyping import _genotype_mix
from mito_utils.plotting_base import *
from mito_utils.diagnostic_plots import *


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/discretization/MDA_clones_100')

# Params
sample = 'MDA_clones_100'
filtering = 'MI_TO'

# Data
afm = read_one_sample(path_data, sample, with_GBC=True, nmads=10)

# Read and filter
_, a = filter_cells_and_vars(afm, filtering=filtering, max_AD_counts=2, af_confident_detection=0.01, min_cell_number=5)
AD, DP, _ = get_AD_DP(a)
AD = AD.A.T
DP = DP.A.T

# VAF spectrum
fig, ax = plt.subplots(figsize=(4.5,4.5))
vars_AF_dist(a, ax=ax, color='darkred', linewidth=1, linestyle='--')
fig.tight_layout()
plt.show()

(a.X>0).sum(axis=1)                 # n vars (per cell)
(a.X>0).sum(axis=0) / a.shape[0]    # Prop +cells (per var)
a.X.mean(axis=0)                    # Mean AF vars
AD.mean(axis=0)                     # Mean AD vars
DP.mean(axis=0)                     # Mean DP vars

fig, ax = plt.subplots(figsize=(4.5,4.5))
ax.plot((a.X>0).sum(axis=0), np.ma.masked_less_equal(AD, 0).mean(axis=0), 'ko')
format_ax(ax=ax, xlabel='n +cells', ylabel='Mean n of UMIs for ALT in +cells')
fig.tight_layout()
plt.show()


##


# P-P plot
fig = plt.figure(figsize=(16,11))

i = 0
for row, dist in enumerate(['binom', 'nbinom', 'betabinom', 'mixbinom']):
    for idx in np.argsort((a.X>0).sum(axis=0))[::-1]:

        # idx = 0
        variant = a.var_names[idx]
        i += 1
        ad = AD[:,idx]
        dp = DP[:,idx]
        wt = dp - ad

        if dist == 'binom':
            ad_th, _ = fit_binom(ad, dp)  
        elif dist == 'nbinom':
            ad_th, _ = fit_nbinom(ad, dp)  
        elif dist == 'betabinom':
            ad_th, _ = fit_betabinom(ad, dp)    
        elif dist == 'mixbinom':
            ad_th, _ = fit_mixbinom(ad, dp)          

        # CDFs
        X = np.concatenate((ad, ad_th))
        bins = np.linspace(np.min(X)-0.5, np.max(X)+0.5)
        ad_counts, _ = np.histogram(ad, bins=bins)
        ad_th_counts, _ = np.histogram(ad_th, bins=bins) 
        empirical_cdf = ad_counts.cumsum() / ad_counts.sum()
        theoretical_cdf = ad_th_counts.cumsum() / ad_th_counts.sum()

        # Stats variables
        mean_af = (ad/dp).mean()
        freq = ((ad/dp)>0).sum() / dp.size

        ax = fig.add_subplot(4,len(a.var_names),i)
        ax.plot(theoretical_cdf, empirical_cdf, 'o-')
        ax.plot([0,1], [0,1], 'r--')
        corr, p = stats.pearsonr(empirical_cdf, theoretical_cdf)
        if dist == 'binom':
            ax.set(title=f'{variant} \n Mean af: {mean_af:.2f}, %+cells: {freq:.2f} \n Corr: {corr:.2f}; p {p:.2f}', ylabel='Empirical CDF', xlabel=f'{dist} CDF')
        else:
            ax.set(title=f'Corr: {corr:.2f}; p {p:.2f}', ylabel='Empirical CDF', xlabel=f'{dist} CDF')

fig.tight_layout()
fig.savefig(os.path.join(path_results, 'AD_counts_dist_CDF.png'), dpi=400)


##


# BIC and AD fits
fig = plt.figure(figsize=(16,11))

i = 0
for row, dist in enumerate(['binom', 'nbinom', 'betabinom', 'mixbinom']):
    for idx in np.argsort((a.X>0).sum(axis=0))[::-1]:

        variant = a.var_names[idx]
        i += 1
        ad = AD[:,idx]
        dp = DP[:,idx]
        wt = dp - ad

        if dist == 'binom':
            ad_th, d = fit_binom(ad, dp)  
        elif dist == 'nbinom':
            ad_th, d = fit_nbinom(ad, dp)  
        elif dist == 'betabinom':
            ad_th, d = fit_betabinom(ad, dp) 
        elif dist == 'mixbinom':
            ad_th, d = fit_mixbinom(ad, dp)          

        # Stats variables
        bic = d['BIC']
        L = d['L']
        mean_af = (ad/dp).mean()
        freq = ((ad/dp)>0).sum() / dp.size

        ax = fig.add_subplot(4,len(a.var_names),i)
        sns.kdeplot(ad, ax=ax, color='k', fill=True, alpha=.3)
        sns.kdeplot(ad_th, ax=ax, color='r', fill=True, alpha=.3)
        if dist == 'binom':
            ax.set(title=f'{variant} \n Mean af: {mean_af:.2f}, %+cells: {freq:.2f} \n L: {L:.2f}, BIC: {bic:.2f}', ylabel='Density', xlabel=f'n ALT {dist}')
        else:
            ax.set(title=f'L: {L:.2f}, BIC: {bic:.2f}', ylabel='Density', xlabel=f'n ALT')
        add_legend(ax=ax, colors={dist:'r', 'Observed':'k'}, loc='upper right', bbox_to_anchor=(1,1), ticks_size=9)

fig.tight_layout()
fig.savefig(os.path.join(path_results, 'AD_counts_dists.png'), dpi=300)


##


# Trials
idx = 0
variant = a.var_names[idx]
ad = AD[:,idx]
dp = DP[:,idx]

_ = _genotype_mix(AD[:,idx], DP[:,idx], t_prob=.9, t_vanilla=.05, debug=True)
_

G_prob = np.zeros(a.shape)
for idx,variant in enumerate(a.var_names):
    G_prob[:,idx] = _genotype_mix(AD[:,idx], DP[:,idx])

G_vanilla = np.where((AD/(DP+.0000001))>0, 1, 0) 
G_prob[:10,:]
G_vanilla[:10,]
np.sum(G_prob.flatten() != G_vanilla.flatten()) / np.prod(G_prob.shape)





# Benchmark
genotype_MI_TO(a, t_prob=.55)

labels = a.obs['GBC']
evaluate_metric_with_gt(
    a, metric='custom_MI_TO_jaccard', labels=labels, bin_method='MI_TO', 
    discretization_kwargs={'t_prob':.6}
)
evaluate_metric_with_gt(
    a, metric='jaccard', labels=labels, bin_method='vanilla'
)



