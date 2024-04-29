"""
Utilities for the iterative scheme of clonal selection.
"""

import numpy as np
import pandas as pd

from ..utils import *
from ..preprocessing import *
from ..clustering import *
from ..distances import *


##


def test_partitions_variants(d):
    """
    Given a dictionary of exclusive variants per partition, returns two lists,
    one with all the partitions having at least 1 exclusive variants, and one with none of them.
    """
    with_exclusive = [ k for k in d if len(d[k])>0 ]
    without_exclusive = [ k for k in d if len(d[k])==0 ]

    return with_exclusive, without_exclusive


##


def test_partitions_distances(a, metric='cosine', t=.5):
    """
    Given an AFM and a set of labels in .obs ('it' key), computes a parwise similarity matrix
    and use this distance as a binary classifier.
    """
    D = pair_d(a, metric=metric)
    labels = a.obs['it'].astype('category')

    assert (labels.value_counts()>1).all()

    d = {}

    for alpha in np.linspace(0,1,10):
        for i in range(D.shape[0]):
            x = rescale(D[i,:])
            c = labels.cat.codes.values[i]
            pred = np.where(x<=alpha, 1, 0)
            gt = np.where(labels.cat.codes.values==c, 1, 0)
            p = precision_score(gt, pred)
            r = recall_score(gt, pred)

    return 


##


def pool_variants(var):
    it_var = set()
    for x in var:
        it_var |= set(x.to_list())
    return list(it_var)


##


def update_labels(afm, labels, i):

    if i == 0:
        afm.obs[f'it_{i}'] = 'poor_quality'
    else:
        afm.obs[f'it_{i}'] = afm.obs[f'it_{i-1}']

    s = pd.concat(labels)
    afm.obs.loc[s.index, f'it_{i}'] = s


##


def find_exclusive_variants(afm, t=.9, group='g', with_variants=False, **kwargs):
    '''
    Find top clone given an AFM with clonal labels.
    '''

    while t>.5:
        n_vars = pd.Series({
            g : rank_clone_variants(
                afm, var=group, group=g, rank_by='custom_perc_tresholds',
                min_clone_perc=t, max_perc_rest=.1, **kwargs
            ).shape[0] \
            for g in afm.obs[group].unique()
        })
        one_dist = np.sum(n_vars>0)>0
        if one_dist:
            break
        else:
            t -= .05
    
    return n_vars


##