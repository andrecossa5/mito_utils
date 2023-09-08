"""
Tree plotting utils.
"""

import cassiopeia as cs
from matplotlib.colors import ListedColormap
from cassiopeia.plotting.local import *
from mito_utils.diagnostic_plots import sturges
from mito_utils.colors import *
from mito_utils.plotting_base import *


##


def _place_tree_and_annotations(
    tree, 
    meta_data=[], 
    depth_key=None, 
    orient=90, 
    extend_branches=True, 
    angled_branches=True, 
    add_root=True, 
    indel_colors=None, 
    indel_priors=None,
    random_state=None, 
    continuous_cmap='viridis', 
    categorical_cmap='tab10', 
    vmin=None, vmax=None, 
    clade_colors=None, 
    allele_table=None,
    colorstrip_width=None, 
    colorstrip_spacing=None
    ):
    """
    Utils to set tree elements.
    """

    meta_data = meta_data or []

    # Place tree on the appropriate coordinate system.
    node_coords, branch_coords = utilities.place_tree(
        tree,
        depth_key=depth_key,
        orient=orient,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        add_root=add_root
    )

    # Compute first set of anchor coords, which are just the coordinates of all the leaves.
    anchor_coords = {
        node: coords
        for node, coords in node_coords.items()
        if tree.is_leaf(node)
    }
    is_polar = isinstance(orient, (float, int))
    loc = "polar" if is_polar else orient
    tight_width, tight_height = compute_colorstrip_size(
        node_coords, anchor_coords, loc
    )
    width = colorstrip_width or tight_width
    spacing = colorstrip_spacing or tight_width / 2

    # Place indel heatmap
    colorstrips = []
    if allele_table is not None:
        heatmap, anchor_coords = create_indel_heatmap(
            allele_table,
            anchor_coords,
            width,
            tight_height,
            spacing,
            loc,
            indel_colors,
            indel_priors,
            random_state,
        )
        colorstrips.extend(heatmap)

    # Any other annotations
    for meta_item in meta_data:

        if meta_item not in tree.cell_meta.columns:
            raise PlottingError(
                "Meta data item not in CassiopeiaTree cell meta."
            )
        values = tree.cell_meta[meta_item]

        if pd.api.types.is_numeric_dtype(values):

            colorstrip, anchor_coords = create_continuous_colorstrip(
                values.to_dict(), 
                anchor_coords,
                width, 
                tight_height,
                spacing, 
                loc, 
                continuous_cmap,
                vmin, 
                vmax
            )

        if pd.api.types.is_string_dtype(values):

            if isinstance(categorical_cmap, str):
                categorical_cmap = create_palette(tree.cell_meta, meta_item, categorical_cmap)
            elif categorical_cmap is None:
                categorical_cmap = create_palette(tree.cell_meta, meta_item, ten_godisnot)

            boxes, anchor_coords = utilities.place_colorstrip(
                anchor_coords, width, tight_height, spacing, loc
            )
            colorstrip = {}
            for leaf in values.index:
                v = values.loc[leaf]
                colorstrip[leaf] = boxes[leaf] + (categorical_cmap[v], f"{leaf}\n{v}")

        colorstrips.append(colorstrip)

    # Clade colors
    node_colors = {}
    branch_colors = {}
    if clade_colors:
        node_colors, branch_colors = create_clade_colors(tree, clade_colors)

    return node_coords, branch_coords, node_colors, branch_colors, colorstrips


##


def _fix_internal_nodes(node_colors, node_coords, internal_node_kwargs=None):
    """
    Fix internal nodes colors and annotation.
    """

    # Checks
    if 'by' in internal_node_kwargs and 'meta' in internal_node_kwargs:
        pass
    else:
        raise KeyError('by and meta are not present in internal_node_kwargs...')

    cov = internal_node_kwargs['by']
    meta = internal_node_kwargs['meta']
    assert cov in meta.columns
    
    # Mask and limits
    if 'mask' in internal_node_kwargs:
        mask = internal_node_kwargs['mask']
    else:
        mask = [ True for _ in range(meta.shape[0]) ]
    vmin = internal_node_kwargs['vmin'] if 'vmin' in internal_node_kwargs else None
    vmax = internal_node_kwargs['vmax'] if 'vmax' in internal_node_kwargs else None

    # Plot
    if internal_node_kwargs['plot']:

        if pd.api.types.is_string_dtype(meta[cov]):
            if 'categorical_cmap' in internal_node_kwargs:
                cat_colors = internal_node_kwargs['categorical_cmap']
            else:
                cat_colors = create_palette(meta, cov, 'tab20')
            for k in meta.loc[mask, :].index:
                node_colors[k] = cat_colors[meta.loc[k, cov]]

        elif pd.api.types.is_numeric_dtype(meta[cov]):

            if 'continuous_cmap' in internal_node_kwargs:
                palette = internal_node_kwargs['continuous_cmap']
            else:
                palette = 'viridis'
            cmap = matplotlib.colormaps[palette]
            if vmin is None and vmax is None:
                norm = matplotlib.colors.Normalize(
                    vmin=np.percentile(meta[cov], q=25), 
                    vmax=np.percentile(meta[cov], q=75)
                )
            else:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            for k in meta.loc[mask, :].index:
                node_colors[k] = cmap.to_rgba(meta.loc[k, cov])
    
    else:
        node_colors = {}

    # Annotation
    if internal_node_kwargs['annot']:
        annot = { k : (meta.loc[k, cov], node_coords[k]) for k in meta.loc[mask, :].index }
    else:
        annot = {}

    # Remove unwanted keys
    l_ = [
        'meta', 'by', 'categorical_cmap', 'continuous_cmap', 
        'vmin', 'vmax', 'annot', 'plot', 'mask'
    ]
    internal_node_kwargs = { 
        k : internal_node_kwargs[k] for k in internal_node_kwargs
        if k not in l_
    } 
    
    return node_colors, internal_node_kwargs, annot


##


def plot_tree(
    tree, 
    meta_data=None,
    depth_key=None,
    orient=90,
    extend_branches=True,
    angled_branches=True,
    add_root=False,
    indel_colors=None,
    indel_priors=None,
    categorical_cmap='tab20', 
    continuous_cmap='mako', 
    ax=None,
    random_state=1234, 
    vmin=None, 
    vmax=None,
    leaf_size=0,
    internal_node_size=10,
    annot_size=5,
    clade_colors={},
    internal_node_kwargs={},
    leaf_kwargs={},
    branch_kwargs={},
    colorstrip_spacing=None,
    colorstrip_width=None,
    colorstrip_kwargs={}
    ):
    """
    A plotting function that extends the capabilities of cs.pl.plot_matplotlib in cassiopeia,
    MW Jones et al., 2020.
    """
    cs.pl.plot_matplotlib

    # Set coord and axis
    ax.axis('off')
    is_polar = isinstance(orient, (float, int))

    # Set elements
    (
        node_coords,
        branch_coords,
        node_colors,
        branch_colors,
        colorstrips,
    ) = _place_tree_and_annotations(
        tree, 
        meta_data=meta_data, 
        depth_key=depth_key, 
        orient=orient, 
        extend_branches=extend_branches, 
        angled_branches=angled_branches, 
        add_root=add_root, 
        indel_colors=indel_colors, 
        indel_priors=indel_priors,
        categorical_cmap=categorical_cmap, 
        continuous_cmap=continuous_cmap, 
        random_state=random_state, 
        vmin=vmin, 
        vmax=vmax, 
        clade_colors=clade_colors,
        colorstrip_spacing=colorstrip_spacing,
        colorstrip_width=colorstrip_width
    )

    if 'plot' in internal_node_kwargs or 'annot' in internal_node_kwargs:
        (
            node_colors, 
            internal_node_kwargs, 
            node_annot
        ) = _fix_internal_nodes(node_colors, node_coords, internal_node_kwargs)
    else:
        node_colors = {}
        node_annot = {}

    # Plot all branches
    _branch_kwargs = dict(linewidth=1, c="black")
    _branch_kwargs.update(branch_kwargs or {})

    for branch, (xs, ys) in branch_coords.items():
        if is_polar:
            xs, ys = utilities.polars_to_cartesians(xs, ys)
        _ = { k : _branch_kwargs[k] for k in _branch_kwargs } # Needed not to overwrite
        if branch in branch_colors:
            _["c"] = branch_colors[branch]
        ax.plot(xs, ys, **_)

    # Plot colorstrips
    _colorstrip_kwargs = dict(linewidth=0)
    _colorstrip_kwargs.update(colorstrip_kwargs or {})

    for colorstrip in colorstrips:
        for xs, ys, c, _ in colorstrip.values():
            _colorstrip_kwargs["c"] = c
            if is_polar:
                xs, ys = utilities.polars_to_cartesians(xs, ys)
            ax.fill(xs, ys, **_colorstrip_kwargs)

    
    # Plot all nodes
    _leaf_kwargs = dict(x=[], y=[], s=leaf_size, c="black")
    _internal_node_kwargs = dict(x=[], y=[], s=internal_node_size, c="black")
    _leaf_kwargs.update(leaf_kwargs or {})
    _internal_node_kwargs.update(internal_node_kwargs or {})
    _leaf_colors = []
    _internal_node_colors = []

    for node, (x, y) in node_coords.items():

        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)

        if len(node_annot)>0 and node in node_annot:
            text = node_annot[node][0]
            ax.text(x, y, str(text), ha='center', va='top', fontsize=annot_size)

        if tree.is_leaf(node):
            _leaf_kwargs["x"].append(x)
            _leaf_kwargs["y"].append(y)
            c = node_colors[node] if node in node_colors else _leaf_kwargs['c']
            _leaf_colors.append(c)
        else:
            _internal_node_kwargs["x"].append(x)
            _internal_node_kwargs["y"].append(y)
            c = node_colors[node] if node in node_colors else _internal_node_kwargs['c']
            _internal_node_colors.append(c)

    _leaf_kwargs['c'] = _leaf_colors
    _internal_node_kwargs['c'] = _internal_node_colors
    ax.scatter(**_leaf_kwargs)
    ax.scatter(**_internal_node_kwargs)

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