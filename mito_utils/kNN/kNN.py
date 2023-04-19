"""
Module for kNN utils. Import kNN from Cellula by default, but implements utilities for creating 
kNN masked affinity matrices from full affinity matrices. 
"""

import numpy as np
from umap.umap_ import fuzzy_simplicial_set 
from scipy.sparse import coo_matrix 
from scanpy.neighbors import _get_sparse_matrix_from_indices_distances_umap 


##


def kNN_graph(X, k=15):
    """
    Compute a kNN graph from some pre-computed distance matrix.
    """
    # Prep knn_indeces and knn_dists
    knn_indices, knn_dists = get_indices_from_distance_matrix(X, k=k)

    # Prep connectivities
    connectivities = fuzzy_simplicial_set(
        coo_matrix(([], ([], [])), shape=(X.shape[0], 1)),
        k,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    connectivities = connectivities[0]
    distances = _get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, X.shape[0], k
    )

    # Prep results
    results = { 
        'indices' : knn_indices,  
        'distances' : distances, 
        'connectivities' : connectivities,  
    }

    return results


##


def get_indices_from_distance_matrix(X, k=15):
    """
    Given a simmetric distance matrix, get its k NN indeces and their distances.
    """
    assert X.shape[0] == X.shape[1]

    indeces = np.argsort(X, axis=1)
    distances = X[np.arange(X.shape[0])[:,None], indeces]
    indeces = indeces[:, :k]
    distances = X[:, :k]

    return indeces, distances






