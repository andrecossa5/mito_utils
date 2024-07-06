"""
plot_tree refactoring.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.phylo import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from mito_utils.phylo_plots import *
from mito_utils.phylo_plots import _place_tree_and_annotations, _set_colors


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')
path_results = os.path.join(path_main, 'results/supervised_clones/distances')
path_vars = os.path.join(path_main, 'results/var_selection')

# Params
sample = 'MDA_clones'
filtering = 'MI_TO'
t = 0.05

# Data
make_folder(path_results, sample, overwrite=False)
path_sample = os.path.join(path_results, sample)
afm = read_one_sample(path_data, sample, with_GBC=True, nmads=10)

# Read
_, a = filter_cells_and_vars(
    afm, filtering=filtering, 
    max_AD_counts=2, af_confident_detection=0.05, min_cell_number=5
)
# Tree
tree = build_tree(a, solver='NJ', t=.05)
tree.cell_meta['GBC'] = tree.cell_meta['GBC'].astype('str')


#


# tree
fig, ax = plt.subplots(figsize=(5,5))
# plot_tree(tree, ax=ax, extend_branches=False, orient='down', cov_leaves='GBC')
# fig.tight_layout()



# Default
depth_key=None
orient=90
extend_branches=True
angled_branches=True
add_root=False
meta=None
categorical_cmap_annot='tab20'
continuous_cmap_annot='mako'
vmin_annot=.01
vmax_annot=.1
colorstrip_spacing=.05
colorstrip_width=1
meta_branches=None
cov_branches=None
cmap_branches='Spectral_r'
cov_leaves=None
cmap_leaves=ten_godisnot
meta_internal_nodes=None
cov_internal_nodes=None
cmap_internal_nodes='Spectral_r'
leaves_labels=False
internal_node_labels=False
internal_node_vmin=.2
internal_node_vmax=.8
internal_node_label_size=7
leaf_label_size=5
branch_kwargs={}
colorstrip_kwargs={}
leaf_kwargs={}
internal_node_kwargs={} 
x_space=1.5


# Mod
extend_branches=False
orient='down'
cov_leaves='GBC'



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
    continuous_cmap=continuous_cmap_annot, 
    categorical_cmap=categorical_cmap_annot, 
    vmin_annot=vmin_annot, 
    vmax_annot=vmax_annot, 
    colorstrip_width=colorstrip_width, 
    colorstrip_spacing=colorstrip_spacing
)


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



# Colstrips
# _colorstrip_kwargs = {'linewidth':0}
# _colorstrip_kwargs.update(colorstrip_kwargs or {})
# for colorstrip in colorstrips:
#     for xs, ys, c, _ in colorstrip.values():
#         _dict = _colorstrip_kwargs.copy()
#         _dict["c"] = c
#         ax.fill(xs, ys, **_dict)



# Leaves 
_leaf_kwargs = {'markersize':0, 'c':'k', 'marker':'o'}
_leaf_kwargs.update(leaf_kwargs or {})
leaves = { node : node_coords[node] for node in node_coords if tree.is_leaf(node) }
colors = _set_colors(
    leaves, meta=tree.cell_meta, cov=cov_leaves, 
    cmap=cmap_leaves, kwargs=_leaf_kwargs
)     



meta=tree.cell_meta
cov=cov_leaves
cmap=cmap_leaves
kwargs=_leaf_kwargs




def _set_colors(d, meta=None, cov=None, cmap=None, kwargs=None, vmin=None, vmax=None):
    """
    Create a dictionary of elements colors.
    """

    colors = None
    
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
    
    if colors is None:
        raise ValueError("Colors not set properly. Check the conditions.")

    return colors








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




# # Internal nodes
# _internal_node_kwargs = {
#     'markersize':2, 'c':'k', 'marker':'o', 'alpha':1, 
#     'markeredgecolor':'k', 'markeredgewidth':1, 'zorder':10
# }
# _internal_node_kwargs.update(internal_node_kwargs or {})
# internal_nodes = { 
#     node : node_coords[node] for node in node_coords \
#     if tree.is_internal_node(node) and node != 'root'
# }
# colors = _set_colors(
#     internal_nodes, meta=meta_internal_nodes, cov=cov_internal_nodes, 
#     cmap=cmap_internal_nodes, kwargs=_internal_node_kwargs,
#     vmin=internal_node_vmin, vmax=internal_node_vmax
# )
# for node in internal_nodes:
#     _dict = _internal_node_kwargs.copy()
#     x = internal_nodes[node][0]
#     y = internal_nodes[node][1]
#     c = colors[node] if node in colors else 'white'
#     s = _internal_node_kwargs['markersize'] if node in colors else 0
#     _dict.update({'c':c, 'markersize':s})
#     ax.plot(x, y, **_dict)
#     if internal_node_labels:
#         if node in colors:
#             v = meta_internal_nodes.loc[node, cov_internal_nodes]
#             if isinstance(v, float):
#                 v = round(v, 2)
#             ax.text(
#                 x+.3, y-.1, str(v), ha='center', va='bottom', 
#                 bbox=dict(boxstyle='round', alpha=0, pad=10),
#                 fontsize=internal_node_label_size,
#             )



















plt.show()