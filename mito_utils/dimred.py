"""
Dimensionality reduction utils to compress (pre-filtered) AFMs.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from umap.umap_ import simplicial_set_embedding, find_ab_params
from .kNN import *
from .distances import *


##


def find_diffusion_matrix(D):
    """
    Function to find the diffusion matrix P.
    """
    alpha = D.flatten().std()
    K = np.exp(-D**2 / alpha**2) # alpha is the variance of the distance matrix, here
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))

    return P_prime, P, Di, K, D_left


##


def find_diffusion_map(P_prime, D_left, n_eign=10):
    """
    Function to find the diffusion coordinates in the diffusion space.
    """   
    eigenValues, eigenVectors = eigh(P_prime)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    diffusion_coordinates = np.matmul(D_left, eigenVectors)
    
    return diffusion_coordinates[:,:n_eign]


##


def find_pca(X, n_pcs=30, random_state=1234):
    """
    Get PCA embeddings with fbpca.
    """
    model = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = model.fit_transform(X)
    
    return X_pca


##


def _umap_from_X_conn(X, conn, ncomps=2, metric='cosine', metric_kwargs={}, seed=1234):
    """
    Wrapper around umap.umap_.simplicial_set_embedding() to create a umap embedding of the 
    feature matrix X using a precomputed fuzzy graph.
    """
    a, b = find_ab_params(1.0, 0.5)
    X_umap, _ = simplicial_set_embedding(
        X, conn, ncomps, 1.0, a, b, 1.0, 5, 200, 'spectral', 
        random_state=np.random.RandomState(seed), metric=metric, metric_kwds=metric_kwargs,
        densmap=None, densmap_kwds=None, output_dens=None
    )
    return X_umap


##


def reduce_dimensions(
    X=None, a=None, AD=None, DP=None, seed=1234, method='UMAP', k=15, n_comps=30, 
    metric='cosine', binary=True, bin_method='vanilla', scale=True, weights=None, metric_kwargs={}, binarization_kwargs={}
    ):
    """
    Create a dimension-reduced representation of the input data matrix.
    """

    # Prep kwargs
    kwargs = dict(
        X=X, a=a, AD=AD, DP=DP, metric=metric, binary=binary, bin_method=bin_method,
        weights=weights, scale=scale, metric_kwargs=metric_kwargs, binarization_kwargs=binarization_kwargs
    )
    
    # Reduce
    if method == 'PCA':
        kwargs_ = { kwargs[k] for k in kwargs if k != 'metric_kwargs' }
        metric, _, X = preprocess_feature_matrix(**kwargs_)
        X_reduced = find_pca(X, n_pcs=n_comps, random_state=seed)
        feature_names = [ f'PC{i}' for i in range(1, X_reduced.shape[1]+1)]

    elif method == 'UMAP':
        D = compute_distances(**kwargs)
        _, _, conn = kNN_graph(D=D, k=k, from_distances=True)
        X_reduced = _umap_from_X_conn(X, conn, ncomps=n_comps, metric=metric, metric_kwargs=metric_kwargs, seed=seed)
        feature_names = [ f'UMAP{i}' for i in range(1, X_reduced.shape[1]+1)]

    elif method == 'diffmap':
        D = compute_distances(**kwargs)
        P_prime, _,_,_, D_left = find_diffusion_matrix(D)
        X_reduced = find_diffusion_map(P_prime, D_left, n_eign=n_comps)
        feature_names = [ f'Diff{i}' for i in range(1, X_reduced.shape[1]+1)]

    return pd.DataFrame(X_reduced, columns=feature_names)


##



