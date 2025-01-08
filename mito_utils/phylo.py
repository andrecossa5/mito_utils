"""
Utils for phylogenetic inference.
"""

import cassiopeia as cs
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
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

_solver_kwargs = {
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
    dp --> coverage at that site. NB: AD and DP are assumed in shape cells x variants. 
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
    dp --> coverage at that site. NB: AD and DP are assumed in shape cells x variants. 
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

        return M.iloc[:,resampled_idx].copy()


##


def jackknife_allele_tables(ad=None, dp=None, M=None):
    """
    LOO of ad and dp count tables. ad --> alternative counts
    dp --> coverage at that site. NB: AD and DP are assumed in shape cells x variants. 
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

        return M.iloc[:,resampled_idx].copy()


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
    M_raw = pd.DataFrame(afm.X.A, index=afm.obs_names, columns=afm.var_names)
    M = pd.DataFrame(afm.layers[layer].A, index=afm.obs_names, columns=afm.var_names)

    # Remove variants from char matrix i) they are called in less than min_n_positive_cells or ii) > max_frac_positive 
    # We avoid recomputing distances as their contribution to the average pairwise cell-cell distance is minimal
    if filter_muts and afm.uns['scLT_system'] == 'SNVs':
        test_germline = ((M==1).sum(axis=0) / M.shape[0]) <= max_frac_positive
        test_too_rare = (M==1).sum(axis=0) >= min_n_positive_cells
        test = (test_germline) & (test_too_rare)
        M_raw = M_raw.loc[:,test].copy()
        M = M.loc[:,test].copy()

    return M_raw, M, D


##


def build_tree(
    afm, precomputed=False, distance_key='distances', metric='jaccard', 
    bin_method='MiTo', solver='NJ', ncores=1, min_n_positive_cells=2, filter_muts=True,
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


def _resolve_node_assignment(x, times_d):
    nodes = x.loc[lambda x: x==1]
    if nodes.size>0:
        nodes =  nodes.index
        assigned_node = nodes[np.argmax([ times_d[node] for node in nodes ])]
    else:
        assigned_node = 'Unassigned'
    return assigned_node


##


def MiToTreeAnnotator(tree, n_clones=None):
    """
    Cut a tree into discrete MT-clones, supported by one or more MT-SNV.
    1) Assign each variant to an internal node (lowest p-value among clades, Fisher's exact test)
    2) Cluster MT-SNVs into an optimal number of cluster (highest silhouette score, elbow method, [2, 2/3 * MT-SNVs] mutational clusters tried)
    3) Collapse internal nodes according to MT-SNVs clusters into discrete clones
    4) Update tree.cell meta with clone annotation and outputs each final clone-MT-SNVs assignment.   
    """

    # Get clades binary encodings
    clades = get_clades(tree, with_root=False, with_singletons=False)
    df_clades = pd.DataFrame(
        np.array([[ x in clades[clade] for x in tree.leaves ] for clade in clades ]).astype(int),
        index=clades.keys(),
        columns=tree.leaves
    ).T.loc[tree.layers['transformed'].index]

    # Assign all muts to an internal node
    muts = tree.layers['transformed'].columns
    n = len(tree.leaves)

    P = {}
    logging.info('Assign variants to internal nodes...')
    for lineage_column in df_clades:

        target_ratio_array = np.zeros(muts.size)
        oddsratio_array = np.zeros(muts.size)
        pvals = np.zeros(muts.size)

        for i,mut in enumerate(muts):
            test_mut = tree.layers['transformed'].values[:,i] == 1
            test_lineage = df_clades[lineage_column].values == 1
            mut_size = test_mut.sum()
            mut_lineage_size = (test_mut & test_lineage).sum()
            target_ratio = mut_lineage_size / mut_size
            target_ratio_array[i] = target_ratio
            other_mut_lineage_size = (~test_mut & test_lineage).sum()

            # Fisher
            oddsratio, pvalue = fisher_exact(
                [
                    [mut_lineage_size, mut_size - mut_lineage_size],
                    [other_mut_lineage_size, n - other_mut_lineage_size],       # NB: REVISE THIS
                ],
                alternative='greater',
            )
            oddsratio_array[i] = oddsratio
            pvals[i] = pvalue

        # FDR
        pvals = multipletests(pvals, alpha=.05, method="fdr_bh")[1]
        P[lineage_column] = pvals

    df_muts = pd.DataFrame(P, index=muts)

    # Extract results for plotting
    node_variant_map = {}
    for mut in df_muts.index:
        candidate = df_muts.loc[mut,:].sort_values().index[1]
        node_variant_map[candidate] = mut
    
    # Add info to as node attributes
    mut_nodes = list(node_variant_map.keys())
    for node in tree.internal_nodes:
        if node in mut_nodes:
            tree.set_attribute(node, 'mut', True)
        else:
            tree.set_attribute(node, 'mut', False)
    node_assignment = pd.Series(node_variant_map).to_frame('mut')
    
    mutation_order = []
    for node in tree.depth_first_traverse_nodes():
        if node in node_assignment.index:
            mutation_order.append(node_assignment.loc[node, 'mut'])

    # Cluster MT-SNVs into an optimal number of cluster, k
    X = tree.layers['transformed'].values.T
    D = pairwise_distances(X, metric='jaccard')
    L = linkage(D, method='ward')

    # idx = leaves_list(L) 
    # ax.imshow(D[np.ix_(idx,idx)], cmap='viridis_r')
    # fig, ax = plt.subplots(figsize=(5,5))
    # format_ax(ax=ax, yticks=tree.character_matrix.columns[idx], xticks=[], title='MT-SNVs clustering')
    # fig.tight_layout()
    # plt.show()

    # Determine optimal number of MT-SNVs clusters
    maxK = D.shape[0] if D.shape[0] <=25 else round( D.shape[0] * (2/3) )
    if n_clones is None:
        measures = []
        n_clones = list(range(2, maxK+1))
        for k in n_clones:
            np.random.seed(1234)
            labels = fcluster(L, k, criterion='maxclust')
            silhouette_avg = silhouette_score(D, labels, metric='precomputed')
            measures.append(silhouette_avg)
        n_clones = n_clones[np.argmax(measures)]
        logging.info(f'Optimal mut_clusters: {n_clones}')

        # fig, ax = plt.subplots(figsize=(4.5,4.5))
        # ax.plot(measures, 'ko')
        # format_ax(ax=ax, xlabel='k MT-SNVs cluster', ylabel='Average silhouette coefficient', title='MT-SNVs clustering')
        # ax.axvline(x=np.argmax(measures), linestyle='--', c='r')
        # fig.tight_layout()
        # plt.show()

    else:
        logging.info(f'Target mut_clusters: {n_clones}')

    # Collapse internal nodes
    distances = L[:,2]
    merge_index = len(L) - (n_clones - 1)
    threshold = distances[merge_index]
    epsilon = 1e-10
    threshold += epsilon
    np.random.seed(1234)
    mut_clusters = fcluster(L, t=threshold, criterion='distance')

    logging.info(f'Final mut_clusters: {np.unique(mut_clusters).size}')
    
    mut_map = { mut : cluster for mut, cluster in zip(tree.character_matrix.columns, mut_clusters)}
    node_assignment['mut_cluster'] = node_assignment['mut'].map(mut_map)
    node_assignment['MT_clone'] = node_assignment['mut_cluster'].map(lambda x: f'C{x}')

    # Annotate tree.cell meta
    df_clades = df_clades.loc[:,node_assignment.index]
    times_d = { k:v for k,v in tree.get_times().items() if k in node_assignment.index }
    resolved_nodes = {}
    for cell in df_clades.index:
        x = df_clades.loc[cell,:]
        resolved_nodes[cell] = _resolve_node_assignment(x, times_d)
    
    df_clades = pd.Series(resolved_nodes).to_frame('clade')
    df_clades['MT_clone'] = df_clades['clade'].map(node_assignment['MT_clone'].to_dict())
    tree.cell_meta['MT_clone'] = df_clades.loc[tree.cell_meta.index, 'MT_clone']
    tree.cell_meta['MT_clone'] = np.where(~tree.cell_meta['MT_clone'].isna(), tree.cell_meta['MT_clone'], 'Undefined')
    n_clones = tree.cell_meta['MT_clone'].unique().size
    
    logging.info(f'Final MT_clones: {n_clones}')

    clone_muts = {
        clone : ';'.join(node_assignment.loc[node_assignment['MT_clone']==clone, 'mut'].to_list())
        for clone in node_assignment['MT_clone'].unique()
    }
    tree.cell_meta['clone_muts'] = tree.cell_meta['MT_clone'].map(clone_muts)


    return tree, mut_nodes, mutation_order


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
        'mut' : [ tree.get_attribute(node, 'mut') for node in tree.internal_nodes ],
        'clade_size' : [ len(clades[node]) for node in tree.internal_nodes ]
        }, 
        index=tree.internal_nodes
    )
    
    return df 


##


###################################### Deprecated code


# def bootstrap_iteration(
#     X=None, a=None, AD=None, DP=None, meta=None,
#     metric='jaccard', binary=True, bin_method='vanilla', scale=True,
#     solver='UPMGA', ncores=8, boot_strategy='jacknife',
#     metric_kwargs={}, binarization_kwargs={}, solver_kwargs={}
#     ):
#     """
#     One bootstrap iteration.
#     """
# 
#     # Tree kwargs
#     tree_kwargs = dict(
#         meta=meta, metric=metric, binary=binary, bin_method=bin_method, scale=scale,
#         solver=solver, ncores=ncores, metric_kwargs=metric_kwargs, 
#         binarization_kwargs=binarization_kwargs, solver_kwargs=solver_kwargs
#     )
# 
#     # Bootstrap ch matrix
#     if AD is not None and DP is not None:
#         
#         AD_ = None
#         DP_ = None
#         if boot_strategy == 'jacknife':
#             AD_, DP_, _ = jackknife_allele_tables(ad=AD, dp=DP)
#         elif boot_strategy == 'features_resampling':
#             AD_, DP_, _ = bootstrap_allele_tables(ad=AD, dp=DP)
#         elif boot_strategy == 'counts_resampling':
#             AD_, DP_, _ = bootstrap_allele_counts(AD, DP)
#         else:
#             raise ValueError(f'{boot_strategy} is not supported. Choose one between: jacknife, features_resampling or counts_resampling')
# 
#         tree = build_tree(AD=AD_, DP=DP_, **tree_kwargs)
#     
#     elif X is not None:
#      
#         if boot_strategy == 'jacknife':
#             X = jackknife_allele_tables(M=X)
#         elif boot_strategy == 'features_resampling':
#             X = bootstrap_allele_tables(M=X)
#         elif boot_strategy == 'counts_resampling':
#             raise ValueError('Pass AD and DP for counts resampling...')
#         else:
#             raise ValueError(f'{boot_strategy} is not supported. Choose one between: jacknife, features_resampling or counts_resampling')
# 
#         tree = build_tree(X=X, **tree_kwargs)
# 
#     return tree
# 
# 
# def compute_n_cells_clades(tree):
#     """
#     Find all clades in a tree, from top to bottom
#     """
#     clades = get_clades(tree)
#     cell_counts_df = (
#         pd.Series({k : len(clades[k]) for k in clades})
#         .sort_values(ascending=False)
#     )
# 
#     return cell_counts_df
# 
# 
# ##
# 
# 
# def calculate_FBP(obs_tree, tree_list):
#     """
#     Calculate Falsestein Bootstrap proportions.
#     """
#     obs_clades = get_clades(obs_tree, with_root=False)
#     supports = { k : 0 for k in obs_clades }
# 
#     for boot_tree in tree_list:
#         b_ = get_clades(boot_tree, with_root=False).values()
#         for k in obs_clades:
#             if obs_clades[k] in b_:
#                 supports[k] += 1
# 
#     supports = pd.Series(supports)/len(tree_list)
#     supports = (supports * 100).astype(int)
# 
#     return supports.to_dict()
# 
# 
# ##
# 
# 
# def get_binary_clade(clade, leaves_order):
#     """
#     Get binary encoding of the clade bipartition.
#     """
#     bin_bool = np.array([ x in clade for x in leaves_order ])
#     sum_1 = np.sum(bin_bool)
#     sum_0 = bin_bool.size-sum_1
# 
#     if sum_1>=sum_0:
#         v = bin_bool.astype(np.int0)
#     else:
#         v = (~bin_bool).astype(np.int0)
# 
#     return v
# 
# 
# ##
# 
# 
# def compute_TBE_hamming_one_tree(obs_clades, leaves_order, boot_tree, n_jobs=8):
#     """
#     Compute Transfer Support for all the observed clades, given one boot replicate.
#     """
# 
#     boot_clades = get_clades(boot_tree, with_root=False, with_singletons=True)
# 
#     OBS = np.array([ 
#         get_binary_clade(obs_clades[clade], leaves_order=leaves_order) \
#         for clade in obs_clades 
#     ]).astype(np.float16)
#     p = np.sum(OBS==0, axis=1)
# 
#     BOOT = np.array([ 
#         get_binary_clade(boot_clades[clade], leaves_order=leaves_order) \
#         for clade in boot_clades 
#     ]).astype(np.float16)
# 
#     D1 = pairwise_distances(OBS, BOOT, metric='hamming', n_jobs=n_jobs) * OBS.shape[1]
#     D2 = pairwise_distances((~OBS.astype(bool)).astype(np.int0), BOOT, metric='hamming', n_jobs=n_jobs) * OBS.shape[1]
#     D = np.minimum(D1, D2)
#     transfer_index = D.min(axis=1)
# 
#     S = 1-transfer_index/(p-1)
#     S[np.where(S<0)] = 0  # Numerical issues with hamming distance rounding ecc.
#     S[np.where(S>1)] = 1  # Numerical issues with hamming distance rounding ecc.
# 
#     return S
# 
# 
# ##
# 
# 
# def calculate_TBE(obs_tree, tree_list, n_jobs=8):
#     """
#     Calculate TBE from Lamoine et al., 2018 and the definition of transfer distance
#     from the Hamming of clades bi-partition encodings.
#     """
#     leaves_order = obs_tree.leaves
#     obs_clades = get_clades(obs_tree, with_root=False)
# 
#     supports = []
#     for boot_tree in tree_list:
#         supports.append(
#             compute_TBE_hamming_one_tree(
#                 obs_clades, leaves_order, boot_tree, n_jobs=n_jobs
#             )
#         )
#     supports = pd.Series(np.mean(supports, axis=0), index=obs_clades.keys())
#     supports.loc[supports.isna()] = 0
#     supports = (supports * 100).astype(int)
# 
#     return supports.to_dict()
# 
# 
# ##
# 
# 
# def calculate_supports(obs_tree, tree_list, method='TBE', n_jobs=8):
#     """
#     Calculates internal nodes bootstrap support. Two algorithms
#     implemented:
#     - fbp: Falsestein Bootstrap Proportions, traditional support from Falsestein's work.
#     - tbe: Transfer Bootstrap Expectations, transfer distance-based method, suitable
#            for big phylogenies.
#     """
# 
#     if method == 'FBP':
#         supports = calculate_FBP(obs_tree, tree_list)
#     elif method == 'TBE':
#         supports = calculate_TBE(obs_tree, tree_list, n_jobs=n_jobs)
#     else:
#         raise ValueError('No other method implemented so far...')
# 
#     tree = obs_tree.copy()
#     for node in supports:
#         tree.set_attribute(node, 'support', supports[node])
# 
#     return tree
# 
# 
# def get_internal_node_muts(tree, internal_node):
#     """
#     Get each internal node mutational status.
#     """
#     muts = tree.character_matrix.columns
#     node_state = np.array(tree.get_character_states(internal_node))
#     idx = np.where(node_state==1)[0]
#     muts = muts[idx].to_list()
#     return muts
# 
# 
# ##
# 
# 
# def assess_internal_node_muts(a, clades, c, high_af=.05):
#     """
#     Assess the prevalence of all MT-SNVs assigned to a single internal node 
#     within its clade (p1), and outside of it (p0). Return useful stats.
#     """
#     cells = list(clades[c])
#     other_cells = a.obs_names[~a.obs_names.isin(cells)]
#     p1 = np.where(a[cells,:].X>=high_af,1,0).sum(axis=0) / len(cells)
#     p0 = np.where(a[other_cells, :].X>=high_af,1,0).sum(axis=0) / len(other_cells)
#     af1 = np.median(a[cells,:].X, axis=0)
#     af0 = np.median(a[other_cells, :].X, axis=0)
#     muts = a.var_names
#     top_mut = (
#         pd.DataFrame(
#             {'p1':p1,'p0':p0, 'median_af1':af1, 'median_af0':af0, 
#              'clade':[c]*len(muts), 'ncells':[len(cells)]*len(muts)}, 
#             index=muts
#         )
#         .assign(p_ratio=lambda x: x['p1']/x['p0']+.0001)
#         .assign(af_ratio=lambda x: x['median_af1']/x['median_af0']+.0001)
#         .query('median_af1>=@high_af and median_af0==0')
#         .sort_values('p_ratio', ascending=False)
#     )
#     return top_mut
# 
# 
# ##
# 
# 
# def get_supporting_muts(tree, a, t=.05):
#     """
#     For each clade, rank its supporting mutations.
#     """
#     clades = get_clades(tree)
#     stats = []
#     for c in clades:
#         if c != 'root':
#             top_mut = assess_internal_node_muts(a, clades, c, high_af=t)
#             stats.append(top_mut)
#     final_muts = pd.concat(stats)
#     muts = final_muts.index.unique().tolist()
#         
#     return muts
# 
# 
# ##
# 
# 
# def sort_muts(tree):
#     """
#     Sort all tree mutations for plotting,
#     """
#     muts = (
#         tree.cell_meta
#         .loc[tree.leaves]
#         .apply(lambda x: tree.cell_meta.columns[x.argmax()], axis=1)
#         .drop_duplicates()
#         .to_list()
#     )
# 
#     return muts


##