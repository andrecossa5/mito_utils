"""
Utils for phylogenetic inference.
"""

import cassiopeia as cs
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
from cassiopeia.plotting.local import *
from plotting_utils._utils import *
from mito_utils.distances import *
from mito_utils.preprocessing import *
from mito_utils.phylo_io import *


##


solver_d = {
    'UPMGA' : cs.solver.UPGMASolver,
    'NJ' : cs.solver.NeighborJoiningSolver,
    'spectral' : cs.solver.SpectralSolver,
    'shared_muts' : cs.solver.SharedMutationJoiningSolver,
    'greedy' : cs.solver.SpectralGreedySolver,
    'max_cut' : cs.solver.MaxCutGreedySolver,
}
    
##

_solver_kwargs = {
    'UPMGA' : {},
    'NJ' : {'add_root':True},
    'spectral' : {},
    'shared_muts' : {},
    'greedy' : {},
    'max_cut' : {}
}


##

def _initialize_CassiopeiaTree_kwargs(afm, distance_key, min_n_positive_cells, max_frac_positive, filter_muts=True):
    """
    Extract afm slots for CassiopeiaTree instantiation.
    """

    assert 'bin' in afm.layers or 'scaled' in afm.layers
    assert distance_key in afm.obsp

    layer = 'bin' if 'bin' in afm.layers else 'scaled'
    D = afm.obsp[distance_key].A.copy()
    D[np.isnan(D)] = 0
    D = pd.DataFrame(D, index=afm.obs_names, columns=afm.obs_names)
    M = pd.DataFrame(afm.layers[layer].A, index=afm.obs_names, columns=afm.var_names)
    if afm.X is not None:
        M_raw = pd.DataFrame(afm.X.A, index=afm.obs_names, columns=afm.var_names)
    else:
        M_raw = M.copy()

    # Remove variants from char matrix i) they are called in less than min_n_positive_cells or ii) > max_frac_positive 
    # We avoid recomputing distances as their contribution to the average pairwise cell-cell distance is minimal
    if filter_muts and afm.uns['scLT_system'] != 'Cas9':
        test_germline = ((M==1).sum(axis=0) / M.shape[0]) <= max_frac_positive
        test_too_rare = (M==1).sum(axis=0) >= min_n_positive_cells
        test = (test_germline) & (test_too_rare)
        M_raw = M_raw.loc[:,test].copy()
        M = M.loc[:,test].copy()

    return M_raw, M, D


##


def build_tree(
    afm, precomputed=False, distance_key='distances', metric='jaccard', 
    bin_method='MiTo', solver='UPMGA', ncores=1, min_n_positive_cells=2, filter_muts=False,
    max_frac_positive=.95, binarization_kwargs={}, solver_kwargs={},
    ):
    """
    Wrapper around CassiopeiaTree distance-based and maximum parsimony solvers.
    """
    
    # Compute (if necessary, cell-cell distances, and retrieve necessary afm .slots)
    if precomputed:
        if distance_key in afm.obsp and precomputed:
            metric = afm.uns['distance_calculations'][distance_key]['metric']
            layer = afm.uns['distance_calculations'][distance_key]['layer']
            logging.info(f'Use precomputed distances: metric={metric}, layer={layer}')
            if layer == 'bin':
                bin_method = afm.uns['genotyping']['bin_method']
                binarization_kwargs = afm.uns['genotyping']['binarization_kwargs']
                logging.info(f'Precomputed bin layer: bin_method={bin_method} and binarization_kwargs={binarization_kwargs}')
    else:
        compute_distances(
            afm, distance_key=distance_key, metric=metric, 
            bin_method=bin_method, ncores=ncores, binarization_kwargs=binarization_kwargs
        )
    M_raw, M, D = _initialize_CassiopeiaTree_kwargs(afm, distance_key, min_n_positive_cells, max_frac_positive, filter_muts=filter_muts)
 
    # Solve cell phylogeny
    metric = afm.uns['distance_calculations'][distance_key]['metric']
    logging.info(f'Build tree: metric={metric}, solver={solver}')
    np.random.seed(1234)
    tree = cs.data.CassiopeiaTree(character_matrix=M, dissimilarity_map=D, cell_meta=afm.obs)
    _solver = solver_d[solver]
    kwargs = _solver_kwargs[solver]
    kwargs.update(solver_kwargs)
    solver = _solver(**kwargs)
    solver.solve(tree)

    # Add layers to CassiopeiaTree
    tree.layers['raw'] = M_raw
    tree.layers['transformed'] = M

    return tree


##


def get_clades(tree, with_root=True, with_singletons=False):
    """
    Find all clades in a tree, from top to bottom
    """
    clades = { x : frozenset(tree.leaves_in_subtree(x)) for x in tree.internal_nodes }

    if not with_root:
        if 'root' in clades:
            del clades['root']

    if with_singletons:
        for x in tree.leaves:
            clades[x] = frozenset([x])

    return clades


##


def get_expanded_clones(tree, t=.05, min_depth=3, min_clade_size=None):
    """
    Get significantly expanded clades.
    """
    min_clade_size = (t * tree.n_cell) if min_clade_size is None else min_clade_size
    cs.tl.compute_expansion_pvalues(
        tree, 
        min_clade_size=min_clade_size, 
        min_depth=min_depth, 
    )
    
    expanding_nodes = []
    for node in tree.depth_first_traverse_nodes():
        if tree.get_attribute(node, "expansion_pvalue") < t:
            expanding_nodes.append(node)

    return expanding_nodes


##


def AFM_to_seqs(afm, bin_method='MiTo', binarization_kwargs={}):
    """
    Convert an AFM to a dictionary of sequences.
    """

    # Extract ref and alt character sequences
    L = [ x.split('_')[1].split('>') for x in afm.var_names ]
    ref = ''.join([x[0] for x in L])
    alt = ''.join([x[1] for x in L])

    if 'bin' not in afm.layers:
        call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)

    # Convert to a dict of strings
    X_bin = afm.layers['bin'].A.copy()
    d = {}
    for i, cell in enumerate(afm.obs_names):
        m_ = X_bin[i,:]
        seq = []
        for j, char in enumerate(m_):
            if char == 1:
                seq.append(alt[j]) 
            elif char == 0:
                seq.append(ref[j])
            else:
                seq.append('N')
        d[cell] = ''.join(seq)

    return d


##


def calculate_corr_distances(tree):
    """
    Calculate correlation between tree and character matrix cell-cell distances. 
    """

    if tree.get_dissimilarity_map() is not None:
        D = tree.get_dissimilarity_map()
        D = D.loc[tree.leaves, tree.leaves] # In case root is there...
    else:
        raise ValueError('No precomputed character distance. Add one...')
    
    L = []
    undirected = tree.get_tree_topology().to_undirected()
    for node in tree.leaves:
        d = nx.shortest_path_length(undirected, source=node)
        L.append(d)
    D_phylo = pd.DataFrame(L, index=tree.leaves).loc[tree.leaves, tree.leaves]
    assert (D_phylo.index == D.index).all()

    scale = lambda x: (x-x.mean())/x.std()
    corr, p = pearsonr(scale(D.values.flatten()), scale(D_phylo.values.flatten()))
    
    return corr, p


##


def _compatibility_metric(x, y):
    """
    Custom metric to calculate the compatibility between two characters.
    Returns the fraction of compatible leaf pairs.
    """
    return np.sum((x == x[:, None]) == (y == y[:, None])) / len(x) ** 2

##


def char_compatibility(tree):
    """
    Compute a matrix of pairwise-compatibility scores between characters.
    """
    return pairwise_distances(
        tree.character_matrix.T, 
        metric=lambda x, y: _compatibility_metric(x, y), 
        force_all_finite=False
    )


##


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
    Calculate the Consistency Index (RI) of tree characters.
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


##


def get_supports(tree):

    L = []
    for node in tree.internal_nodes:
        try:
            s = tree.get_attribute(node, 'support')
            s = s if s is not None else np.nan
            L.append(s)
        except:
            pass

    return np.array(L)


##


def get_internal_node_stats(tree):
    """
    Get internal nodes supports, time, mut status and clade size
    """

    clades = get_clades(tree)
    df = pd.DataFrame({
        'support' : get_supports(tree), 
        'time' : [ tree.get_time(node) for node in tree.internal_nodes ],
        'clade_size' : [ len(clades[node]) for node in tree.internal_nodes ]
        }, 
        index=tree.internal_nodes
    )
    if 'lca' in tree.cell_meta:
        clades = tree.cell_meta['lca'].loc[lambda x: ~x.isna()].unique()
        df['mut_clade'] = [ True if node in clades else False for node in tree.internal_nodes ]
    
    return df 


##