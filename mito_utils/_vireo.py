"""
VireoSNP utils module.
"""

import numpy as np
import pandas as pd
from kneed import KneeLocator
from vireoSNP import BinomMixtureVB

from mito_utils.utils import *
from mito_utils.preprocessing import *
from mito_utils.plotting_base import *


##


def fit_vireo(AD, DP, k, random_seed=1234, **kwargs):
    """
    VireoSNP BinomMixtureVB model fitting
    """
    model = BinomMixtureVB(n_var=AD.shape[0], n_cell=AD.shape[1], n_donor=k)
    model.fit(
        AD, DP, min_iter=30, max_iter=1000, max_iter_pre=250, 
        n_init=100, random_seed=random_seed
    )
    return model


##


def prob_to_crisp_labels(afm, _model, p_treshold=.85):
    """
    Util to convert clonal assignment probabilities into crisp labels.
    """
    clonal_assignment = _model.ID_prob
    df_ass = pd.DataFrame(
        clonal_assignment, 
        index=afm.obs_names, 
        columns=range(clonal_assignment.shape[1])
    )

    # Define labels
    labels = []
    for i in range(df_ass.shape[0]):
        cell_ass = df_ass.iloc[i, :]
        try:
            labels.append(np.where(cell_ass>p_treshold)[0][0])
        except:
            labels.append('unassigned')

    return labels


##


def plot_vireoSNP_elbow(_ELBO_mat, range_clones, knee):
    """
    Visualize the ELBO distribution across the range of k explored.
    """

    df_ = (
        pd.DataFrame(np.array(_ELBO_mat), index=range_clones)
        .reset_index()
        .rename(columns={'index':'n_clones'})
        .melt(id_vars='n_clones', var_name='run', value_name='ELBO')
    )    
    df_['n_clones'] = df_['n_clones'].astype(str)
    
    # Fig
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_.groupby('n_clones').agg('median')['ELBO'].sort_values(), 'o--')
    ax.vlines(str(knee), df_['ELBO'].min(), df_['ELBO'].max())
    box(df_, 'n_clones', 'ELBO', c='#E9E7E7', ax=ax)
    format_ax(ax, xlabel='n clones', ylabel='ELBO')
    ax.text(.8, .1, f'knee: {knee}', transform=ax.transAxes)
    ax.spines[['right', 'top']].set_visible(False)

    return fig


##


def vireo_clustering(afm, min_n_clones=2, max_n_clones=None, p_treshold=.85, 
                    with_knee_plot=False, random_seed=1234):
    """
    Given an AFM (cells x MT-variants), this function uses the vireoSNP method to return 
    sets of MT-clones labels.
    """

    # Get AD, DP
    afm = nans_as_zeros(afm)
    AD, DP, _ = get_AD_DP(afm, to='csc')

    # Find max_n_clones
    if max_n_clones is None:
        max_n_clones = afm.shape[1]

    # Here we go
    range_clones = range(min_n_clones, max_n_clones+1)
    labels_d = {}
    _ELBO_mat = []

    for k in range_clones:

        print(f'Clone n: {k}')
        model = fit_vireo(AD, DP, k, random_seed=random_seed)
        _ELBO_mat.append(model.ELBO_inits)
        labels_d[f'{k}_clones'] = prob_to_crisp_labels(afm, model, p_treshold=p_treshold)

    # Find best k
    x = range_clones
    y = np.median(_ELBO_mat, axis=1)
    n_clones = KneeLocator(x, y).find_knee()[0]

    # Prep output
    df = pd.DataFrame(labels_d, index=afm.obs_names)
    best_solution = df[f'{n_clones}_clones']

    if with_knee_plot:
        fig = plot_vireoSNP_elbow(_ELBO_mat, range_clones, n_clones)
        return df, n_clones, fig
    else:
        return df, n_clones
    

##