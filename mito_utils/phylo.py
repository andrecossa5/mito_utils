"""
Utils for phylogenetic inference.
"""

import cassiopeia as cs

from cassiopeia.plotting.local import *
from scipy.sparse import issparse
from mito_utils.distances import *


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

solver_kwargs_d = {
    'UPMGA' : {},
    'NJ' : {'add_root':True},
    'spectral' : {},
    'shared_muts' : {},
    'greedy' : {},
    'max_cut' : {}

}


##


def bootstrap_allele_counts(ad, dp, frac=.8):
    """
    Bootstrapping of ad and dp count tables. ad --> alternative counts
    dp --> coverage at that site. NB: AD and DP are assumed in for cells x variants. 
    Both sparse matrices and dense arrays can be passed.
    """

    ad = ad if not issparse(ad) else ad.A
    dp = dp if not issparse(dp) else dp.A
    new_ad = np.zeros(ad.shape)
    new_dp = np.zeros(ad.shape)

    for j in range(ad.shape[1]): # Iterate on variants

        alt_counts = ad[:,j]
        ref_counts = dp[:,j] - ad[:,j]

        for i in range(ad.shape[0]): # Iterate on cells

            observed_alleles = np.concatenate([
                np.ones(alt_counts[i]), 
                np.zeros(ref_counts[i])
            ])

            n = round(observed_alleles.size*frac)
            new_alt_counts = np.random.choice(observed_alleles, n, replace=True).sum() 

            new_dp[i,j] = n
            new_ad[i,j] = new_alt_counts

    return new_ad, new_dp, np.arange(ad.shape[1])


##


def bootstrap_allele_tables(ad, dp):
    """
    Bootstrapping of ad and dp count tables. ad --> alternative counts
    dp --> coverage at that site. NB: AD and DP are assumed in for cells x variants. 
    Both sparse matrices and dense arrays can be passed.
    """

    ad = ad if not issparse(ad) else ad.A
    dp = dp if not issparse(dp) else dp.A

    n = ad.shape[1]
    resampled_idx = np.random.choice(np.arange(n), n, replace=True)    
    new_ad = ad[:,resampled_idx]
    new_dp = dp[:,resampled_idx]

    return new_ad, new_dp, resampled_idx


##


def jackknife_allele_tables(ad, dp):
    """
    LOO of ad and dp count tables. ad --> alternative counts
    dp --> coverage at that site. NB: AD and DP are assumed in for cells x variants. 
    Both sparse matrices and dense arrays can be passed.
    """

    ad = ad if not issparse(ad) else ad.A
    dp = dp if not issparse(dp) else dp.A

    n = ad.shape[1]
    to_exclude = np.random.choice(np.arange(n), 1)[0]
    resampled_idx = [ x for x in np.arange(n) if x != to_exclude ]
    new_ad = ad[:,resampled_idx]
    new_dp = dp[:,resampled_idx]

    return new_ad, new_dp, resampled_idx


##


def build_tree(a=None, M=None, D=None, t=.025, metric='cosine', 
                solver=None, ncores=8, solver_kwargs={}):
    """
    Wrapper for tree building.
    """

    # Get solver and solver_kwargs
    if isinstance(solver, str) and solver in solver_d:
        _solver = solver_d[solver]
        print(f'Chosen solver: {solver}')
    elif isinstance(solver, str) and solver not in solver_d:
        raise KeyError(f'{solver} solver not available.')
    else:
        print(f'New solver passed to build_tree(): {solver}')

    solver_kwargs = solver_kwargs_d[solver]

    # Compute char and distance matrices
    X = a.X
    if M is None:
        M = pd.DataFrame(
            np.where(X>=t, 1, 0),
            index=a.obs_names,
            columns=a.var_names
        )
    if D is None:
        D = pd.DataFrame(
            pair_d(X, ncores=ncores, metric=metric),
            index=a.obs_names,
            columns=a.obs_names
        )
    
    # Compute tree
    tree = cs.data.CassiopeiaTree(character_matrix=M, dissimilarity_map=D, cell_meta=a.obs)
    solver = _solver(**solver_kwargs)
    solver.solve(tree)

    return tree


##


def bootstrap_iteration(a, AD, DP, kwargs={}):
    """
    One bootstrap iteration.
    """
    
    # Bootstrap ch matrix
    boot_strategy = kwargs['boot_strategy']
    del kwargs['boot_strategy']

    if boot_strategy == 'jacknife':
        AD_, DP_, sel_idx = jackknife_allele_tables(AD.A.T, DP.A.T)
    elif boot_strategy == 'counts_resampling':
        AD_, DP_, sel_idx = bootstrap_allele_counts(AD.A.T, DP.A.T)
    elif boot_strategy == 'features_resampling':
        AD_, DP_, sel_idx = bootstrap_allele_tables(AD.A.T, DP.A.T)

    X = np.divide(AD_, DP_)
    X[np.isnan(X)] = 0

    # Build tree
    kwargs['variants'] = a.var_names[sel_idx]
    tree = build_tree(a, X=X, **kwargs)
    
    return tree


##


def get_clades(tree):
    """
    Find all clades in a tree, from top to bottom
    """
    sorted_internal = [ 
        node for node in tree.depth_first_traverse_nodes(postorder=False) \
        if node in tree.internal_nodes and node != 'root'
    ]
    clades = { x : frozenset(tree.leaves_in_subtree(x)) for x in sorted_internal }

    return clades


##


def compute_n_cells_clades(tree):
    """
    Find all clades in a tree, from top to bottom
    """
    clades = get_clades(tree)
    cell_counts_df = (
        pd.Series({k : len(clades[k]) for k in clades})
        .sort_values(ascending=False)
    )

    return cell_counts_df


##


def calculate_support(obs_tree, tree_list, n=100):
    """
    Calculate bootstrap support of all obs_tree coalescences.
    """

    # Get observed clades
    obs_clades = get_clades(obs_tree)
    supports = { k : 0 for k in obs_clades }
    c_per_sample = []

    # It on tree_list
    for boot_tree in tree_list:

        o_ = set(obs_clades.values())
        boot_clades = get_clades(boot_tree) 
        b_ = set(boot_clades.values())
        int_ = { x for x in o_ for y in b_ if x==y }        # Get intersection

        c = 0
        for k in obs_clades:
            t = obs_clades[k] in int_
            if t:                                           # Update counts
                c += 1
                supports[k] += 1

        c_per_sample.append(c)

    # Format
    supports = pd.Series(supports) / n
    df = (
        supports
        .to_frame('support')
        .assign(
            time=lambda x: [ obs_tree.get_time(k) for k in x.index ],
            n_cells=lambda x: [ len(obs_tree.leaves_in_subtree(k)) for k in x.index ]
        )
    )
    f_per_sample = pd.Series(c_per_sample) / len(obs_clades)
    f_per_sample = f_per_sample.to_frame('perc_clades_found')

    return df, f_per_sample


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


def AFM_to_seqs(a, t=.1, method='simple_treshold'):
    """
    Funtion to converta an AFM to a dictionary of sequences.
    """

    # Extract ref and alt character sequences
    L = [ x.split('_')[1].split('>') for x in a.var_names ]
    ref = ''.join([x[0] for x in L])
    alt = ''.join([x[1] for x in L])

    # Build a ncells x nvar binary allele matrix using some method
    if method == 'simple_treshold':
        M = np.where(a.X>t, 1, 0)

    # Convert to a dict of strings
    d = {}
    for i, cell in enumerate(a.obs_names):
        m_ = M[i,:]
        seq = [ alt[j] if char == 1 else ref[j] for j, char in enumerate(m_) ]
        d[cell] = ''.join(seq)

    return d