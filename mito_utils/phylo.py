"""
Utils for phylogenetic inference.
"""

import cassiopeia as cs
from scipy.stats import nbinom
from cassiopeia.plotting.local import *
from scipy.sparse import issparse
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


def bootstrap_allele_tables(ad=None, dp=None, M=None, frac_resampled=.9):
    """
    Bootstrapping of ad and dp count tables. ad --> alternative counts
    dp --> coverage at that site. NB: AD and DP are assumed in for cells x variants. 
    Both sparse matrices and dense arrays can be passed.
    """

    if ad is not None and dp is not None:

        n = ad.shape[1]
        if frac_resampled == 1:
            resampled_idx = np.random.choice(np.arange(n), n, replace=True) 
        else:    
            resampled_idx = np.random.choice(np.arange(n), round(n*frac_resampled), replace=False)   
        ad = ad if not issparse(ad) else ad.A
        dp = dp if not issparse(dp) else dp.A
        new_ad = ad[:,resampled_idx]
        new_dp = dp[:,resampled_idx]

        return new_ad, new_dp, resampled_idx
    
    elif M is not None:

        n = M.shape[1]
        resampled_idx = np.random.choice(np.arange(n), n, replace=True)   

        return M.iloc[:, resampled_idx].copy()


##


def jackknife_allele_tables(ad=None, dp=None, M=None):
    """
    LOO of ad and dp count tables. ad --> alternative counts
    dp --> coverage at that site. NB: AD and DP are assumed in for cells x variants. 
    Both sparse matrices and dense arrays can be passed.
    """

    if ad is not None and dp is not None:

        n = ad.shape[1]
        to_exclude = np.random.choice(np.arange(n), 1)[0]
        resampled_idx = [ x for x in np.arange(n) if x != to_exclude ]
        ad = ad if not issparse(ad) else ad.A
        dp = dp if not issparse(dp) else dp.A
        new_ad = ad[:,resampled_idx]
        new_dp = dp[:,resampled_idx]

        return new_ad, new_dp, resampled_idx
    
    elif M is not None:

        n = M.shape[1]
        to_exclude = np.random.choice(np.arange(n), 1)[0]
        resampled_idx = [ x for x in np.arange(n) if x != to_exclude ]

        return M.iloc[:, resampled_idx].copy()


##


def build_tree(
    a=None, M=None, D=None, meta=None, bin_method='vanilla',
    metric='jaccard', t=.01, weights=None, solver='UPMGA', 
    ncores=8, metric_kwargs={}, solver_kwargs={}
    ):
    """
    Wrapper for tree building with Cassiopeia solvers.
    """

    # Get solver and solver_kwargs
    if isinstance(solver, str) and solver in solver_d:
        _solver = solver_d[solver]
        print(f'Chosen solver: {solver}')
    elif isinstance(solver, str) and solver not in solver_d:
        raise KeyError(f'{solver} solver not available. Choose on in {solver_d}')

    solver_kwargs = solver_kwargs_d[solver]

    # Compute char and distance matrices from a, if available
    if M is not None and D is not None and meta is not None:
        pass
    elif a is not None:
        meta = a.obs
        if bin_method == 'nb':
            raise ValueError('Not implemented')
            # X_bin = binarize_nb(a)
        else:
            X_bin = np.where(a.X>=t, 1, 0)
        M = pd.DataFrame(X_bin, index=a.obs_names, columns=a.var_names)
        D = pair_d(a, metric=metric, t=t, weights=weights, ncores=ncores, bin_method=bin_method, metric_kwargs=metric_kwargs)
        D[np.isnan(D)] = 0
        D = pd.DataFrame(D, index=a.obs_names, columns=a.obs_names)
    else:
        raise ValueError(
                '''
                You either pass an afm (a), or both character (M),
                distances (D) and meta data tables.
                '''
            )
    
    # Compute tree
    tree = cs.data.CassiopeiaTree(character_matrix=M, dissimilarity_map=D, cell_meta=meta)
    solver = _solver(**solver_kwargs)
    solver.solve(tree)

    return tree


##


def bootstrap_iteration(M=None, AD=None, DP=None, meta=None, variants=None, 
                        boot_strategy='jacknife', solver='UPMGA', 
                        metric='jaccard', t=0.01, ncores=8, solver_kwargs={}):
    """
    One bootstrap iteration.
    """

    # Bootstrap ch matrix
    if AD is not None and DP is not None:
        
        if boot_strategy == 'jacknife':
            AD_, DP_, sel_idx = jackknife_allele_tables(AD=AD.A.T, DP=DP.A.T)
        elif boot_strategy == 'features_resampling':
            AD_, DP_, sel_idx = bootstrap_allele_tables(AD=AD.A.T, DP=DP.A.T)
        elif boot_strategy == 'counts_resampling':
            AD_, DP_, sel_idx = bootstrap_allele_counts(AD=AD.A.T, DP=DP.A.T)

        # Prep M, D and meta
        X = np.divide(AD_, DP_)
        X[np.isnan(X)] = 0
        variants = variants[sel_idx]
        M = pd.DataFrame(np.where(X>=t, 1, 0), index=meta.index, columns=variants)
        D = pair_d(X, metric=metric, t=t, ncores=ncores)
        D = pd.DataFrame(D, index=meta.index, columns=meta.index)
        D[np.isnan(D)] = 0
    
    elif M is not None:
                
        if boot_strategy == 'jacknife':
            M = jackknife_allele_tables(M=M)
        elif boot_strategy == 'features_resampling':
            M = bootstrap_allele_tables(M=M)
        else:
            raise ValueError('Pass AD and DP for counts resampling...')

        # Prep M, D and meta
        D = pair_d(M.values, metric=metric, t=t, ncores=ncores)
        D = pd.DataFrame(D, index=M.index, columns=M.index)
        meta = pd.DataFrame(index=M.index)
        variants = M.columns

    # Build tree
    tree = build_tree(
        M=M, D=D, meta=meta, metric=metric, t=t, 
        solver=solver, solver_kwargs=solver_kwargs
    )

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


def calculate_FBP(obs_tree, tree_list):
    """
    Calculate Falsestein Bootstrap proportions.
    """
    obs_clades = get_clades(obs_tree, with_root=False)
    supports = { k : 0 for k in obs_clades }

    for boot_tree in tree_list:
        b_ = get_clades(boot_tree, with_root=False).values()
        for k in obs_clades:
            if obs_clades[k] in b_:
                supports[k] += 1

    supports = pd.Series(supports)/len(tree_list)
    supports = (supports * 100).astype(int)

    return supports.to_dict()


##


def get_binary_clade(clade, leaves_order):
    """
    Get binary encoding of the clade bipartition.
    """
    bin_bool = np.array([ x in clade for x in leaves_order ])
    sum_1 = np.sum(bin_bool)
    sum_0 = bin_bool.size-sum_1

    if sum_1>=sum_0:
        v = bin_bool.astype(np.int0)
    else:
        v = (~bin_bool).astype(np.int0)

    return v


##


def compute_TBE_hamming_one_tree(obs_clades, leaves_order, boot_tree, n_jobs=8):
    """
    Compute Transfer Support for all the observed clades, given one boot replicate.
    """

    boot_clades = get_clades(boot_tree, with_root=False, with_singletons=True)

    OBS = np.array([ 
        get_binary_clade(obs_clades[clade], leaves_order=leaves_order) \
        for clade in obs_clades 
    ]).astype(np.float16)
    p = np.sum(OBS==0, axis=1)

    BOOT = np.array([ 
        get_binary_clade(boot_clades[clade], leaves_order=leaves_order) \
        for clade in boot_clades 
    ]).astype(np.float16)

    D1 = pairwise_distances(OBS, BOOT, metric='hamming', n_jobs=n_jobs) * OBS.shape[1]
    D2 = pairwise_distances((~OBS.astype(bool)).astype(np.int0), BOOT, metric='hamming', n_jobs=n_jobs) * OBS.shape[1]
    D = np.minimum(D1, D2)
    transfer_index = D.min(axis=1)

    S = 1-transfer_index/(p-1)
    S[np.where(S<0)] = 0  # Numerical issues with hamming distance rounding ecc.
    S[np.where(S>1)] = 1  # Numerical issues with hamming distance rounding ecc.

    return S


##


def calculate_TBE(obs_tree, tree_list, n_jobs=8):
    """
    Calculate TBE from Lamoine et al., 2018 and the definition of transfer distance
    from the Hamming of clades bi-partition encodings.
    """
    leaves_order = obs_tree.leaves
    obs_clades = get_clades(obs_tree, with_root=False)

    supports = []
    for boot_tree in tree_list:
        supports.append(
            compute_TBE_hamming_one_tree(
                obs_clades, leaves_order, boot_tree, n_jobs=n_jobs
            )
        )
    supports = pd.Series(np.mean(supports, axis=0), index=obs_clades.keys())
    supports.loc[supports.isna()] = 0
    supports = (supports * 100).astype(int)

    return supports.to_dict()


##


def calculate_supports(obs_tree, tree_list, method='TBE', n_jobs=8):
    """
    Calculates internal nodes bootstrap support. Two algorithms
    implemented:
    - fbp: Falsestein Bootstrap Proportions, traditional support from Falsestein's work.
    - tbe: Transfer Bootstrap Expectations, transfer distance-based method, suitable
           for big phylogenies.
    """

    if method == 'FBP':
        supports = calculate_FBP(obs_tree, tree_list)
    elif method == 'TBE':
        supports = calculate_TBE(obs_tree, tree_list, n_jobs=n_jobs)
    else:
        raise ValueError('No other method implemented so far...')

    tree = obs_tree.copy()
    for node in supports:
        tree.set_attribute(node, 'support', supports[node])

    return tree


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


##


def get_internal_node_muts(tree, internal_node):
    """
    Get each internal node mutational status.
    """
    muts = tree.character_matrix.columns
    node_state = np.array(tree.get_character_states(internal_node))
    idx = np.where(node_state==1)[0]
    muts = muts[idx].to_list()
    return muts


##


def assess_internal_node_muts(a, clades, c, high_af=.05):
    """
    Assess the prevalence of all MT-SNVs assigned to a single internal node 
    within its clade (p1), and outside of it (p0). Return useful stats.
    """
    cells = list(clades[c])
    other_cells = a.obs_names[~a.obs_names.isin(cells)]
    p1 = np.where(a[cells,:].X>=high_af,1,0).sum(axis=0) / len(cells)
    p0 = np.where(a[other_cells, :].X>=high_af,1,0).sum(axis=0) / len(other_cells)
    af1 = np.median(a[cells,:].X, axis=0)
    af0 = np.median(a[other_cells, :].X, axis=0)
    muts = a.var_names
    top_mut = (
        pd.DataFrame(
            {'p1':p1,'p0':p0, 'median_af1':af1, 'median_af0':af0, 
             'clade':[c]*len(muts), 'ncells':[len(cells)]*len(muts)}, 
            index=muts
        )
        .assign(p_ratio=lambda x: x['p1']/x['p0']+.0001)
        .assign(af_ratio=lambda x: x['median_af1']/x['median_af0']+.0001)
        .query('median_af1>=@high_af and median_af0==0')
        .sort_values('p_ratio', ascending=False)
    )
    return top_mut


##


def get_supporting_muts(tree, a, t=.05):
    """
    For each clade, rank its supporting mutations.
    """
    clades = get_clades(tree)
    stats = []
    for c in clades:
        if c != 'root':
            top_mut = assess_internal_node_muts(a, clades, c, high_af=t)
            stats.append(top_mut)
    final_muts = pd.concat(stats)
    muts = final_muts.index.unique().tolist()
        
    return muts


##


def sort_muts(tree):
    """
    Sort all tree mutations for plotting,
    """
    muts = (
        tree.cell_meta
        .loc[tree.leaves]
        .apply(lambda x: tree.cell_meta.columns[x.argmax()], axis=1)
        .drop_duplicates()
        .to_list()
    )

    return muts


##


def calculate_corr_distances(tree):
    """
    Calculate correlation between tree and character matrix cell-cell distances. 
    """

    if tree.get_dissimilarity_map() is not None:
        D = tree.get_dissimilarity_map()
        D = D.loc[tree.leaves, tree.leaves] # In case root is there...
    else:
        if tree.character_matrix is not None:
            D = pair_d(tree.character_matrix, t=.05)
            D = pd.DataFrame(D, index=tree.character_matrix.index, columns=tree.character_matrix.index)
            D = D.loc[tree.leaves, tree.leaves]
        else:
            raise ValueError('No precomputed character distances...')
    
    L = []
    undirected = tree.get_tree_topology().to_undirected()
    for node in tree.leaves:
        d = nx.shortest_path_length(undirected, source=node, weight="length")
        L.append(d)
    D_phylo = pd.DataFrame(L, index=tree.leaves).loc[tree.leaves, tree.leaves]
    assert (D_phylo.index == D.index).all()

    scale = lambda x: (x-x.mean())/x.std()
    corr = np.corrcoef(scale(D.values.flatten()), scale(D_phylo.values.flatten()))[0,1]

    return corr


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
    return pairwise_distances(tree.character_matrix.T, metric=lambda x, y: _compatibility_metric(x, y), force_all_finite=False)


##