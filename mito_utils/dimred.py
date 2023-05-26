"""
Dimensionality reduction utils to compress (pre-filtered) AFMs.
"""

import numpy as np
from umap.umap_ import UMAP
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap.umap_ import simplicial_set_embedding, find_ab_params


##


def find_diffusion_matrix(X=None):
    """
    Function to find the diffusion matrix P.
    """

    dists = pairwise_distances(X, n_jobs=-1)
    alpha = dists.flatten().std()
    K = np.exp(-dists**2 / alpha**2) # alpha is the variance of the distance matrix, here
    
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    
    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))

    return P_prime, P, Di, K, D_left


##


def find_diffusion_map(P_prime, D_left, n_eign=3):
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


def find_pca(X, n_pcs=30):
    """
    Get PCA embeddings with fbpca.
    """

    if issparse(X): 
        X = X.A
        X[np.isnan(X)] = 0 # np.nans removal
    else:
        X[np.isnan(X)] = 0

    model = PCA(n_components=n_pcs, random_state=1234)
    X_pca = model.fit_transform(X)
    
    return X_pca


##


def reduce_dimensions(afm, method='PCA', metric='euclidean', n_comps=30, 
                    sqrt=False, scale=True, seed=1234):
    """
    Util to create dimension-reduced representation of the input SNVs AFM.
    """
    # Get AFM np.array
    X = afm.X

    # Sqrt and scale, optionally
    if sqrt:
        X = np.sqrt(X)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Reduce
    if method == 'PCA':
        X_reduced = find_pca(X, n_pcs=n_comps)
        feature_names = [ f'PC{i}' for i in range(1, X_reduced.shape[1]+1)]

    elif method == 'UMAP':
        umap = UMAP(n_components=n_comps, metric=metric, random_state=seed)
        X_reduced = umap.fit_transform(X)
        feature_names = [ f'UMAP{i}' for i in range(1, X_reduced.shape[1]+1)]

    elif method == 'diffmap':
        P_prime, P, Di, K, D_left = find_diffusion_matrix(X)
        X_reduced = find_diffusion_map(P_prime, D_left, n_eign=n_comps)
        feature_names = [ f'Diff{i}' for i in range(1, X_reduced.shape[1]+1)]

    return X_reduced, feature_names


##


def umap_from_X_conn(X, conn, ncomps=2, metric='euclidean'):
    """
    Wrapper around umap.umap_.simplicial_set_embedding() to create a umap embedding of the 
    feature matrix X using a precomputed fuzzy graph.
    """
    a, b = find_ab_params(1.0, 0.5)
    X_umap, _ = simplicial_set_embedding(
        X, conn, ncomps, 1.0, a, b, 1.0, 5, 200, 'spectral', 
        random_state=np.random.RandomState(0), metric=metric, metric_kwds={},
        densmap=None, densmap_kwds=None, output_dens=None
    )

    return X_umap


##



