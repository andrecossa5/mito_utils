import os
import pickle
import numpy as np
from mito_utils.phylo_io import *
from mito_utils.plotting_base import *



# Load trees from Newick files
path_ = '/Users/IEO5505/Desktop/mito_bench/results/phylo/MDA_lung/MQuad'
with open(os.path.join(path_, 'annotated_tree.pickle'), 'rb') as f:
    tree = pickle.load(f)

# tree.internal_nodes

from cassiopeia.tools.fitness_estimator._lbi_jungle import LBIJungle

model = LBIJungle(random_seed=1234)
model.estimate_fitness(tree)

nodes = pd.read_csv(os.path.join(path_, 'nodes.csv'), index_col=0)
ancestral_nodes = nodes['clonal_ancestor'].loc[lambda x: ~x.isna()].unique().astype('int').astype(str)

d = {}
for node in tree.nodes:
    try: 
        d[node] = tree.get_attribute(node, 'fitness')
    except:
        pass
s = pd.Series(d)

fig, ax = plt.subplots()
sns.kdeplot(s.apply(lambda x: (x-s.mean())/s.std()), ax=ax, fill=True)
plt.show()