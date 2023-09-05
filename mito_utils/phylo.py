"""
Utils for phylogenetic inference.
"""

import cassiopeia as cs
from mito_utils.distances import *


##


def build_tree(a, X=None, metric='cosine', solver=cs.solver.UPGMASolver, **kwargs):
    """
    Wrapper for tree building.
    """
    X = a.X if X is None else X
    M = pd.DataFrame(
        np.where(X>.1, 1, 0),
        index=a.obs_names,
        columns=a.var_names
    )
    D = pd.DataFrame(
        pair_d(X if X is not None else a, metric=metric),
        index=a.obs_names,
        columns=a.obs_names
    )
    tree = cs.data.CassiopeiaTree(
        character_matrix=M, 
        dissimilarity_map=D, 
        cell_meta=a.obs
    )
    solver = solver(**kwargs)
    solver.solve(tree)

    return tree