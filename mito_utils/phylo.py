"""
Utils for phylogenetic inference.
"""

import cassiopeia as cs
from cassiopeia.plotting.local import *
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from plotting_utils._utils import *
from mito_utils.distances import *
from mito_utils.preprocessing import *


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


def extract_phylodata(a=None, phy=None, t=.01, metric='hamming', ncores=8, from_phy=False,
                      with_alleles_tables=False):
    """
    Extract data structures necessary for tree building.
    """

    if from_phy:

        species = []
        seqs = []
        for i in range(phy.shape[0]):
            L = phy.iloc[i,:].values[0].split(' ')
            species.append(L[0])
            seqs.append(np.array(list(L[-1]))) 

        # Prep M, D
        M = pd.DataFrame(
            data=seqs, index=species, columns=[f'char{_}' for _ in range(seqs[0].size)]
        )
        for j in range(M.shape[1]):
            counts = M.iloc[:,j].value_counts()
            d_ = { char:num for char,num in zip(counts.index, range(counts.shape[0]))}
            M.iloc[:,j] = M.iloc[:,j].map(d_)

        D = pd.DataFrame(pairwise_distances(M.values, metric=metric), 
                        index=species, columns=species)
        meta = pd.DataFrame(index=species)
        variants = M.columns

        return M, D, meta, variants

    else:

        # Remove zeros and get AD, DP
        a = nans_as_zeros(a)
        cells = a.obs_names
        variants = a.var_names
        meta = a.obs

        # Prep M, D
        M = pd.DataFrame(np.where(a.X>=t, 1, 0), index=cells, columns=variants)
        D = pd.DataFrame(pair_d(a.X, ncores=ncores, metric=metric), index=cells, columns=cells)
        D[np.isnan(D)] = 0

        if with_alleles_tables:
            AD, DP, _ = get_AD_DP(a)
            return M, D, meta, variants, AD, DP

        else:
            return M, D, meta, variants


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
        a=None, M=None, D=None, meta=None, 
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
        M = pd.DataFrame(np.where(a.X>=t, 1, 0), index=a.obs_names, columns=a.var_names)
        D = pair_d(a, metric=metric, t=t, weights=weights, ncores=ncores, metric_kwargs=metric_kwargs)
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

    return supports


##


def map_leaves(clade, leaves):
    """
    Map a treee leaves to a clade sides (1 heavy and 0 light).
    """
    mapping = np.array([ x in clade for x in leaves ])
    sum1 = mapping.sum()
    sum0 = mapping.size-sum1
    p = min(sum1, sum0)
    mapping = mapping if sum1>=sum0 else ~mapping
    mapping = { k:v for k,v in zip(leaves, mapping.astype(np.int0)) }

    return mapping, p


##


def ts_one_chunk(L, mapping, p):
    """
    Calculate the bootstrap support of a single obs_clade, defined by 
    its mapping, in a chunk of boot_trees.
    """

    S = []
    for boot_tree in L:
        boot_clades = get_clades(boot_tree, with_root=False, with_singletons=True)

        TD = []
        for b in boot_clades:
            boot_clade_counts = (
                pd.Categorical(
                    [ mapping[x] for x in boot_clades[b] ], categories=[0,1]
                )
                .value_counts()
            )
            l = len(boot_tree.leaves)
            l0 = boot_clade_counts[0]
            l1 = boot_clade_counts[1]
            transfer_distance = min(p-l0+l1, l-p-l1+l0)
            TD.append(transfer_distance)

        transfer_index = np.min(TD)
        support = 1-transfer_index/(p-1)
        S.append(support)

    return S


##


def calculate_ts(tree_list, mapping, p, n_jobs=8):
    """
    Calculate the bootstrap support of a single obs_clade, defined by 
    its mapping, in a list of bootstrap trees.
    """
    starting_idx = chunker(len(tree_list))
    with parallel_backend("loky", inner_max_num_threads=1):
        S = np.concatenate(
            Parallel(n_jobs=n_jobs)(
                delayed(ts_one_chunk)(
                    tree_list[starting_idx[i] : starting_idx[i+1]], 
                    mapping, 
                    p
                )
                for i in range(n_jobs)
            )
        )

    return S


##


def calculate_TBE_I(obs_tree, tree_list, n_jobs=8):
    """
    Calculate TBE from Lamoine et al., 2018 and their proposed algorithm.
    """
    obs_clades = get_clades(obs_tree, with_root=False)
    leaves_order = obs_tree.leaves
    supports = {}
    
    for o in obs_clades:
        mapping, p = map_leaves(obs_clades[o], leaves_order)
        S = calculate_ts(tree_list, mapping, p, n_jobs=n_jobs)
        supports[o] = np.mean(S)
    
    supports = pd.Series(supports)

    return supports


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


def calculate_TBE_II(obs_tree, tree_list, n_jobs=8):
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

    return supports


##


def calculate_supports(obs_tree, tree_list, method='tbe_II', n_jobs=8):
    """
    Calculates internal nodes bootstrap support. Two algorithms
    implemented:
    - fbp: Falsestein Bootstrap Proportions, traditional support from Falsestein's work.
    - tbe: Transfer Bootstrap Expectations, transfer distance-based method, suitable
           for big phylogenies.
    """
    # FBP
    if method == 'fbp':
        supports = calculate_FBP(obs_tree, tree_list)
    # TBE
    elif method == 'tbe_I':
        supports = calculate_TBE_I(obs_tree, tree_list, n_jobs=n_jobs)
    elif method == 'tbe_II':
        supports = calculate_TBE_II(obs_tree, tree_list, n_jobs=n_jobs)
    else:
        raise ValueError('No other method implemented so far...')

    # Format supports into final df
    df = (
        supports
        .to_frame('support')
        .assign(
            time=lambda x: [ obs_tree.get_time(k) for k in x.index ],
            n_cells=lambda x: [ len(obs_tree.leaves_in_subtree(k)) for k in x.index ]
        )
    )

    return df


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