"""
plot_tree refactoring.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.phylo import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from mito_utils.phylo_plots import *


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')

# Params
sample = 'MDA_clones_100'
filtering = 'MI_TO'
t = 0.05

# Data
afm = read_one_sample(path_data, sample, with_GBC=True, nmads=10)

# Read
_, a = filter_cells_and_vars(
    afm, filtering=filtering, 
    max_AD_counts=2, af_confident_detection=0.05, min_cell_number=5
)
# Tree
tree = build_tree(a, solver='NJ', t=.05)
tree.cell_meta['GBC'] = tree.cell_meta['GBC'].astype('str')







def CI(tree):
    """
    Calculate the Consistency Index (CI) of tree characters.
    """
    tree.reconstruct_ancestral_characters()
    observed_changes = np.zeros(tree.n_character)
    for parent, child in tree.depth_first_traverse_edges():
        p_states = np.array(tree.get_character_states(parent))
        c_states = np.array(tree.get_character_states(child))
        changes = (p_states != c_states).astype(int)
        observed_changes += changes

    return 1 / observed_changes # Assumes both characters are present (1,0)


##


def RI(tree):
    """
    Calculate the Consistency Index (CI) of tree characters.
    """
    tree.reconstruct_ancestral_characters()
    observed_changes = np.zeros(tree.n_character)
    for parent, child in tree.depth_first_traverse_edges():
        p_states = np.array(tree.get_character_states(parent))
        c_states = np.array(tree.get_character_states(child))
        changes = (p_states != c_states).astype(int)
        observed_changes += changes

    # Calculate the maximum number of changes (G)
    max_changes = len(tree.nodes)-1  # If every node had a unique state

    return (max_changes-observed_changes) / (max_changes-1)


RI(tree)




from sklearn.metrics import pairwise_distances


import numpy as np
from sklearn.metrics import pairwise_distances

# Assuming `leaf_character_matrix` is a NumPy array of shape (n_leaves, n_characters)
# Each row corresponds to a leaf and each column corresponds to a binary character.


def _compatibility_metric(x, y):
    """
    Custom metric to calculate the compatibility between two characters.
    Returns the fraction of compatible leaf pairs.
    """
    return np.sum((x == x[:, None]) == (y == y[:, None])) / len(x) ** 2


def char_compatibility(tree):
    """
    Compute a matrix of pairwise-compatibility scores between characters.
    """
    return pairwise_distances(tree.character_matrix.T, metric=lambda x, y: _compatibility_metric(x, y))





