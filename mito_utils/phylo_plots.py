"""
Tree plotting utils.
"""

from cassiopeia.plotting.local import utilities as ut
from cassiopeia.plotting.local import *
from mito_utils.diagnostic_plots import sturges
from mito_utils.colors import *
from mito_utils.plotting_base import *


##


_categorical_cmaps = [ten_godisnot, 'tab10', 'set1', 'dark']
_continuous_cmaps = ['mako', 'viridis', 'inferno', 'magma']


##


def _to_polar_coords(d):

    new_d = {}
    for k in d:
        x, y = d[k]
        if not isinstance(x, list):
            x = [x]
            y = [y]
            x, y = ut.polars_to_cartesians(x, y)
            new_d[k] = x[0], y[0]
        else:
            x, y = ut.polars_to_cartesians(x, y)
            new_d[k] = x, y
    
    return new_d


##


def _to_polar_colorstrips(L):

    new_L = []
    for d in L:
        new_d = {}
        for k in d:
            x, y, a, b = d[k]
            x, y = ut.polars_to_cartesians(x, y)
            new_d[k] = x, y, a, b
        new_L.append(new_d)
    
    return new_L


##


def _place_tree_and_annotations(
    tree, 
    meta=[], 
    depth_key=None, 
    orient=90, 
    extend_branches=True, 
    angled_branches=True, 
    add_root=True, 
    continuous_cmap=None, 
    categorical_cmaps=None, 
    vmin_annot=None, vmax_annot=None, 
    colorstrip_width=None, 
    colorstrip_spacing=None
    ):
    """
    Util to set tree elements.
    """

    is_polar = isinstance(orient, (float, int))
    loc = "polar" if is_polar else orient
    
    # Node and branch coords
    node_coords, branch_coords = ut.place_tree(
        tree,
        depth_key=depth_key,
        orient=orient,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        add_root=add_root
    )

    # Colorstrips
    anchor_coords = { k:node_coords[k] for k in node_coords if tree.is_leaf(k) }
    tight_width, tight_height = compute_colorstrip_size(node_coords, anchor_coords, loc)
    width = colorstrip_width or tight_width
    spacing = colorstrip_spacing or tight_width / 2

    colorstrips = []
    meta = meta or []
    
    n_cat = 0

    for feat in meta:

        if feat in tree.cell_meta.columns:
            x = tree.cell_meta[feat]
        else:
            raise KeyError(f'{feat} not in CassiopeiaTree.cell_meta.')

        if pd.api.types.is_numeric_dtype(x):

            if continuous_cmap is None:
                continuous_cmap = _continuous_cmaps[0]
                
            colorstrip, anchor_coords = create_continuous_colorstrip(
                x.to_dict(), 
                anchor_coords,
                width, 
                tight_height,
                spacing, 
                loc, 
                continuous_cmap,
                vmin_annot, 
                vmax_annot
            )

        elif pd.api.types.is_string_dtype(x):

            if categorical_cmaps is None:
                categorical_cmap = create_palette(tree.cell_meta, feat, _categorical_cmaps[n_cat])
            else:
                if feat in categorical_cmaps:
                    if isinstance(categorical_cmaps[feat], str) or isinstance(categorical_cmaps[feat], list):
                        categorical_cmap = create_palette(tree.cell_meta, feat, categorical_cmaps[feat])
                    elif isinstance(categorical_cmaps[feat], dict):
                        categorical_cmap = categorical_cmaps[feat]
                    else:
                        raise ValueError(f'Adjust categorical_cmaps. {feat} : categorical_cmaps is nor a str, a list or a dict...')
                    if not all([ cat in categorical_cmap.keys() for cat in x.unique() ]):
                        cats = x.unique()
                        print(cats)
                        missing_cats = cats[[ cat not in categorical_cmap.keys() for cat in cats ]]
                        print(f'Missing cats in cmap for meta feat {feat}: {missing_cats}. Adding new colors...')
                        for i,missing in enumerate(missing_cats):
                            categorical_cmap[missing] = _categorical_cmaps[0][i]
                        assert(all([ cat in categorical_cmap.keys() for cat in x.unique() ]))
                else:
                    raise KeyError(f'{feat} not present in meta. Adjust categorical_cmaps and meta params...')

            boxes, anchor_coords = ut.place_colorstrip(
                anchor_coords, width, tight_height, spacing, loc
            )
            colorstrip = {}
            for leaf in x.index:
                cat = x.loc[leaf]
                colorstrip[leaf] = boxes[leaf] + (categorical_cmap[cat], f"{leaf}\n{cat}")

            n_cat += 1

        else:
            raise ValueError(f'{feat} in meta has {x.dtype} dtype. Check meta...')

        colorstrips.append(colorstrip)

    # To polar, if necessary
    if is_polar:
        branch_coords = _to_polar_coords(branch_coords)
        node_coords = _to_polar_coords(node_coords)    
        colorstrips = _to_polar_colorstrips(colorstrips)    

    return node_coords, branch_coords, colorstrips


##


def _set_colors(d, meta=None, cov=None, cmap=None, kwargs=None, vmin=None, vmax=None):
    """
    Create a dictionary of elements colors.
    """

    if meta is not None and cov is not None:
        if cov in meta.columns:
            x = meta[cov]
            if isinstance(cmap, str):
                if pd.api.types.is_numeric_dtype(x):
                    cmap = matplotlib.colormaps[cmap]
                    cmap = matplotlib.cm.get_cmap(cmap)
                    if vmin is None or vmax is None:
                        vmin = np.percentile(x.values, 10)
                        vmax = np.percentile(x.values, 90)
                    normalize = plt.Normalize(vmin=vmin, vmax=vmax)
                    colors = [ cmap(normalize(value)) for value in x ]
                    colors = { k:v for k, v in zip(x.index, colors)}
                elif pd.api.types.is_string_dtype(x):
                    colors = (
                        meta[cov]
                        .map(create_palette(meta, cov, cmap))
                        .to_dict()
                    )
            elif isinstance(cmap, dict):
                print('User-provided colors dictionary...')
                colors = meta[cov].map(cmap).to_dict()
            else:
                raise KeyError(f'{cov} You can either specify a string cmap or an element:color dictionary.')
        else:
            raise KeyError(f'{cov} not present in cell_meta.')
    else:
        colors = { k : kwargs['c'] for k in d }

    return colors


##


def plot_tree(
    tree, ax=None, depth_key=None, orient=90, extend_branches=True, angled_branches=True,
    add_root=False, meta=None, categorical_cmaps=None,
    continuous_cmap='mako', vmin_annot=.01, vmax_annot=.1,
    colorstrip_spacing=.05, colorstrip_width=1, 
    meta_branches=None, cov_branches=None, cmap_branches='Spectral_r',
    cov_leaves=None, cmap_leaves='tab20', 
    meta_internal_nodes=None, cov_internal_nodes=None, cmap_internal_nodes='Spectral_r',
    internal_node_labels=False, internal_node_subset=None,
    internal_node_vmin=.2, internal_node_vmax=.8, internal_node_label_size=7, 
    leaves_labels=False, leaf_label_size=5, 
    colorstrip_kwargs={}, leaf_kwargs={}, internal_node_kwargs={}, branch_kwargs={}, 
    x_space=1.5
    ):
    """
    Plotting function that exends capabilities in cs.plotting.local.plot_matplotlib from
    Cassiopeia, MW Jones.
    """
    
    # Set coord and axis
    ax.axis('off')
    is_polar = isinstance(orient, (float, int))

    # Set graphic elements
    (
        node_coords,
        branch_coords,
        colorstrips,
    ) = _place_tree_and_annotations(
        tree, 
        meta=meta, 
        depth_key=depth_key, 
        orient=orient, 
        extend_branches=extend_branches, 
        angled_branches=angled_branches, 
        add_root=add_root, 
        continuous_cmap=continuous_cmap, 
        categorical_cmaps=categorical_cmaps, 
        vmin_annot=vmin_annot, 
        vmax_annot=vmax_annot, 
        colorstrip_width=colorstrip_width, 
        colorstrip_spacing=colorstrip_spacing
    )

    ##

    # Branches
    _branch_kwargs = {'linewidth':1, 'c':'k'}
    _branch_kwargs.update(branch_kwargs or {})
    colors = _set_colors(
        branch_coords, meta=meta_branches, cov=cov_branches, 
        cmap=cmap_branches, kwargs=_branch_kwargs
    )
    for branch, (xs, ys) in branch_coords.items():
        c = colors[branch] if branch in colors else _branch_kwargs['c']
        _dict = _branch_kwargs.copy()
        _dict.update({'c':c})
        ax.plot(xs, ys, **_dict)

    ##

    # Colorstrips
    _colorstrip_kwargs = {'linewidth':0}
    _colorstrip_kwargs.update(colorstrip_kwargs or {})
    for colorstrip in colorstrips:
        for xs, ys, c, _ in colorstrip.values():
            _dict = _colorstrip_kwargs.copy()
            _dict["c"] = c
            ax.fill(xs, ys, **_dict)

    ##

    # Leaves 
    leave_size = 2 if cov_leaves is not None else 0
    _leaf_kwargs = {'markersize':leave_size, 'c':'k', 'marker':'o'}
    _leaf_kwargs.update(leaf_kwargs or {})
    leaves = { node : node_coords[node] for node in node_coords if tree.is_leaf(node) }
    colors = _set_colors(
        leaves, meta=tree.cell_meta, cov=cov_leaves, 
        cmap=cmap_leaves, kwargs=_leaf_kwargs
    )     
    for node in leaves:
        _dict = _leaf_kwargs.copy()
        x = leaves[node][0]
        y = leaves[node][1]
        c = colors[node] if node in colors else _leaf_kwargs['c']
        _dict.update({'c':c})
        ax.plot(x, y, **_dict)
        if leaves_labels:
            if orient == 'right':
                ax.text(
                    x+x_space, y, str(node), ha='center', va='center', 
                    fontsize=leaf_label_size
                )
            else:
                raise ValueError(
                    'Correct placement of labels at leaves implemented only for the right orient.'
                    )

    ##

    # Internal nodes
    _internal_node_kwargs = {
        'markersize':2, 'c':'k', 'marker':'o', 'alpha':1, 
        'markeredgecolor':'k', 'markeredgewidth':1, 'zorder':10
    }
    _internal_node_kwargs.update(internal_node_kwargs or {})
    internal_nodes = { 
        node : node_coords[node] for node in node_coords \
        if tree.is_internal_node(node) and node != 'root'
    }

    # Subset nodes if necessary
    if internal_node_subset is not None:
        internal_node_subset = [ x for x in internal_node_subset if x in tree.internal_nodes ]
        internal_nodes = { node : internal_nodes[node] for node in internal_nodes if node in internal_node_subset }

    # Here we go
    if cov_internal_nodes is not None:
        s = pd.Series({ node : tree.get_attribute(node, cov_internal_nodes) for node in tree.internal_nodes })
        s.loc[lambda x: x.isna()] = 0 # Set missing values to 0
        colors = _set_colors(
            internal_nodes, meta=s.to_frame(cov_internal_nodes), cov=cov_internal_nodes, 
            cmap=cmap_internal_nodes, kwargs=_internal_node_kwargs,
            vmin=internal_node_vmin, vmax=internal_node_vmax
        )
    for node in internal_nodes:
        _dict = _internal_node_kwargs.copy()
        x = internal_nodes[node][0]
        y = internal_nodes[node][1]
        c = colors[node] if node in colors else 'white'
        s = _internal_node_kwargs['markersize'] if node in colors else 0
        _dict.update({'c':c, 'markersize':s})
        ax.plot(x, y, **_dict)

        if internal_node_labels:
            if node in colors:
                v = meta_internal_nodes.loc[node, cov_internal_nodes]
                if isinstance(v, float):
                    v = round(v, 2)
                ax.text(
                    x+.3, y-.1, str(v), ha='center', va='bottom', 
                    bbox=dict(boxstyle='round', alpha=0, pad=10),
                    fontsize=internal_node_label_size,
                )

    return ax


##


def plot_bootstrap_relationships(supports_df, figsize=(10,5)):
    """
    Plot relationship between clade support and molecular-time/clade size.
    """

    fig, axs = plt.subplots(1,2,figsize=figsize)
    
    colors = { 'expanded' : 'r', 'other' : 'k' }
    
    # Time
    scatter(supports_df, 'time', 'support', ax=axs[0], by='expanded_clones', c=colors, s=50)
    sns.regplot(data=supports_df, x='time', y='support', scatter=False, ax=axs[0])
    pho = np.corrcoef(supports_df['time'], supports_df['support'])[0,1]
    format_ax(
        axs[0], xlabel='Molecular time', ylabel='Support', 
        title=f'Pearson\'s r: {pho:.2f}, median support: {supports_df["support"].median():.2f}',
        reduced_spines=True
    )
    
    # Size
    scatter(supports_df, 'n_cells', 'support', ax=axs[1], by='expanded_clones', c=colors, s=50)
    sns.regplot(data=supports_df, x='n_cells', y='support', scatter=False, ax=axs[1])
    pho = np.corrcoef(supports_df['n_cells'], supports_df['support'])[0,1]
    format_ax(
        axs[1], xlabel='n cells', ylabel='Support', 
        title=f'Pearson\'s r: {pho:.2f}, median support: {supports_df["support"].median():.2f}',
        reduced_spines=True
    )
    add_legend(
        ax=axs[1], label='Clade', colors=colors, 
        ticks_size=8, artists_size=7, label_size=9,
        bbox_to_anchor=(.95,.95), loc='upper right'
    )

    fig.tight_layout()

    return fig


##


def plot_bootstrap_record(bootstrap_record, figsize=(6,5)):
    """
    Plot record of the number of observed clades retrieved at each bootstrap
    iteration.
    """

    # Bootstrap record
    fig, ax = plt.subplots(figsize=figsize)

    n = sturges(bootstrap_record)
    hist(bootstrap_record, 'perc_clades_found', ax=ax, c='k', n=n)
    m = bootstrap_record['perc_clades_found'].median()
    n = bootstrap_record.shape[0]
    format_ax(
        ax, ylabel='n bootstrap samples', xlabel='% clades found',
        title=f'{n} bootstrap samples \n median % of clades found per replicate: {m:.2f}', 
        reduced_spines=True
    )
    fig.tight_layout()

    return fig


##


def plot_main(a, obs_tree, supports_df, variants, bootstrap_record, sample_name):
    """
    Main phylo-inference plot. Structure, annotations and characters. Nodes
    colored by support.
    """
    
    # Main viz
    fig, ax = plt.subplots(figsize=(13,8))

    clones_colors = create_palette(obs_tree.cell_meta, 'GBC', 'tab10')

    plot_tree(
        obs_tree,
        meta_data=[*variants.to_list(),'GBC'], 
        orient='down',
        internal_node_kwargs={
            's': 8, 
            'by': 'support',
            'meta': supports_df,
            'plot' : True,
            'annot' : False,
            'continuous_cmap' : 'YlOrBr'
        },
        branch_kwargs={'c':'k', 'linewidth':.7},
        colorstrip_spacing=.1, 
        categorical_cmap=clones_colors,
        ax=ax,
    )

    format_ax(
        ax=ax, 
        title=f'''
            {sample_name}, {bootstrap_record.shape[0]} bootstrap samples \n
            {len(obs_tree.leaves)} cells, {supports_df.shape[0]} internal nodes. Support: {supports_df['support'].median():.2f} (+-{supports_df['support'].std():.2f})
            '''
    )
    add_legend(
        'Clones', colors=clones_colors, ax=ax, bbox_to_anchor=(-.3,.5), loc='center left',
        label_size=10, artists_size=9, ticks_size=9
    )
    add_cbar(
        supports_df['support'], palette='YlOrBr', ax=ax, label='Support',
        ticks_size=7, label_size=9
    )
    add_cbar(
        a.X.flatten(), palette='mako', ax=ax, label='AF',
        ticks_size=7, label_size=9, layout=( (1.2,.25,.03,.5), 'right' ),
        vmax=.1, vmin=.01
    )

    fig.subplots_adjust(left=.22, right=.78, top=.8)

    return fig 


##


def plot_main_support(obs_tree, supports_df):
    """
    Main phylo-inference plot, without visual annotation. All node support shown.
    """

    fig, ax = plt.subplots(figsize=(13,13))

    plot_tree(
        obs_tree,
        orient=90,
        internal_node_kwargs={
            's': 0, 
            'by': 'support',
            'meta': supports_df,
            'plot' : False,
            'annot' : True,
        },
        branch_kwargs={'c':'k', 'linewidth':.7},
        colorstrip_spacing=.1, 
        annot_size=7,
        ax=ax,
    )

    fig.tight_layout()

    return fig


##


def plot_boot_trees(a, tree_list, variants, path_viz):
    """
    Plot 16 random boot trees per variant.
    """

    # Create folder
    make_folder(path_viz, 'boot_trees', overwrite=True)
    path_ = os.path.join(path_viz, 'boot_trees')

    # logger = logging.getLogger('')

    # Here we go
    for var in variants:

        fig = plt.figure(figsize=(15,15))

        for i in range(16):

            tree = tree_list[i*3]
            tree.cell_meta = pd.DataFrame(
                a.obs.loc[tree.leaves, "GBC"].astype(str)
            )
            tree.cell_meta[a.var_names] = a[tree.leaves, :].X.toarray()

            ax = fig.add_subplot(4,4,i+1)
            plot_tree(
                tree,
                meta_data=[var], 
                orient='down',
                internal_node_size=0,
                branch_kwargs={'linewidth':.5},
                colorstrip_spacing=.1, 
                ax=ax,
            )
            format_ax(ax, title=f'bootstrap {i+1}')

        fig.tight_layout()
        fig.savefig(os.path.join(path_, f'{var}.png'), dpi=400)