"""
Main MiTo class for MT-SNVs single-cell phylogenies annotation.
"""

import torch
from mito_utils.phylo import *


##


class MiToTreeAnnotator():
    """
    MiTo tree annotation class.
    """

    def __init__(self, tree):
        """
        Initialize class slots from input CassiopeiaTree.
        """

        # Slots
        self.tree = tree
        self.D = self.tree.get_dissimilarity_map()  # Expected to be a pd.DataFrame.
        self.get_T()                                # Cell x clade binary matrix (pd.DataFrame).
        self.get_ancestors()                        # Create a mapping: clade index -> list of ancestor clade indices.
        self.get_clade_mapping()                    # Create a map from clade numerical indeces in T to actual clade names
        self.get_M()                                # Mutation enrichment matrix (pd.DataFrame).
        self.priors = None
        self.initial_assignments = None
        self.soft_selection = None
        self.final_soft_selection = None
        self.final_binary_selection = None
        self.model_track = None
        self.selected_clades = None
        self.ordered_muts = None
        self.mut_df = None

    ##

    def get_T(self, with_root=True):
        """
        Compute the "cell assignment" matrix, T.
        T is a cell x clade (internal node) binary matrix mapping each cell i to every clade j.
        """
        clades = get_clades(self.tree, with_root=with_root, with_singletons=False)
        T = np.array([[x in clades[clade] for x in self.tree.leaves] for clade in clades]).astype(int)
        T = (
            pd.DataFrame(T, index=list(clades.keys()), columns=self.tree.leaves)
            .T.loc[self.tree.layers['transformed'].index]
        )
        self.T = T

    ##

    def get_M(self, alpha=0.05):
        """
        Compute the "mutation enrichment" matrix, M.
        M is a mut x clade matrix storing for each mutation i and clade j the enrichment 
        value defined as -log10(pval) from a Fisher's Exact test.
        """
        P = {}
        muts = self.tree.layers['transformed'].columns

        # For each internal node (clade)
        for lineage_column in self.T.columns:
            target_ratio_array = np.zeros(muts.size)
            oddsratio_array = np.zeros(muts.size)
            pvals = np.zeros(muts.size)

            # For each mutation
            for i in range(muts.size):
                test_mut = self.tree.layers['transformed'].values[:, i] == 1
                test_lineage = self.T[lineage_column].values == 1
                n_mut_lineage = np.sum(test_mut & test_lineage)
                n_mut_no_lineage = np.sum(test_mut & ~test_lineage)
                n_no_mut_lineage = np.sum(~test_mut & test_lineage)
                n_no_mut_no_lineage = np.sum(~test_mut & ~test_lineage)
                target_ratio_array[i] = n_mut_lineage / (np.sum(test_mut) + 1e-8)

                # Fisher's exact test
                oddsratio, pvalue = fisher_exact(
                    [
                        [n_mut_lineage, n_mut_no_lineage],
                        [n_no_mut_lineage, n_no_mut_no_lineage],
                    ],
                    alternative='greater',
                )
                oddsratio_array[i] = oddsratio
                pvals[i] = pvalue

            # Adjust p-values (FDR correction)
            pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
            P[lineage_column] = pvals

        M = pd.DataFrame(P, index=muts)
        M = -np.log10(M)

        self.M = M
    
    ##

    def get_clade_mapping(self):
        self.idx_to_clade_mapping = { i:n for i,n in enumerate(self.T.columns) }
        self.clade_to_idx_mapping = { n:i for i,n in enumerate(self.T.columns) }
    
    ##

    def get_ancestors(self):
        """
        Build a dictionary mapping each clade index (i.e., column index in T) to a list of its ancestor clade indices.
        A clade i is considered an ancestor of clade j if every cell in clade j is also in clade i.
        """

        T = self.T.values
        ancestors = {}
        num_clades = T.shape[1]

        for j in range(num_clades):
            ancestor_list = []
            for i in range(num_clades):
                if i == j:
                    continue
                if np.all(T[:,j] <= T[:,i]):
                    ancestor_list.append(i)
            ancestors[j] = ancestor_list

        self.ancestors = ancestors

    ##

    def compute_d(self, D, mask):
        """
        Compute the distance ratio for clade j.
        D: Torch tensor (2D dissimilarity matrix).
        mask: Torch tensor (boolean vector encoding cell membership to clade j).
        Returns mean_inside / mean_outside.
        """
        # Get inside and outside distances using boolean indexing.
        inside = D[mask][:,mask]
        outside = D[~mask][:,~mask] if (~mask).sum() > 0 else torch.tensor([1.0])
        mean_inside = inside.sum() if inside.numel() > 0 else torch.tensor(1.0)
        mean_outside = outside.sum() if outside.numel() > 0 else torch.tensor(1.0)
        return mean_inside / (mean_outside+.0001)

    ##

    def compute_e(self, M, j):
        """
        Compute the mutation enrichment for clade j.
        M: Torch tensor (mutation enrichment matrix).
        j: clade index.
        Returns the mean MT-SNV enrichment for clade j.
        """
        return M[:,j].mean()

    ##

    def compute_empirical_priors(self, method='mut_clades', fixed_logit_value=2.5, extend_ancestors=True, 
                                extend_children=True, min_clade_prevalence=.01, max_clade_prevalence=.5):
        """
        Compute logit priors to initialize clone optimization.
        """

        if method == 'mut_clades':

            priors = -fixed_logit_value * np.ones(self.T.shape[1])
            candidate_set = list(set(self.M.values.argmax(axis=1)))

            extended_set = []
            for i in candidate_set:
                if i != 0 and extend_ancestors:
                    ancestor = self.tree.get_all_ancestors(self.idx_to_clade_mapping[i])[0]
                    ancestor_idx = self.clade_to_idx_mapping[ancestor]
                    extended_set.extend([i, ancestor_idx])
                if i in self.idx_to_clade_mapping and extend_children:
                    children = self.tree.children(self.idx_to_clade_mapping[i])
                    for child in children:
                        if child in self.tree.internal_nodes:
                            child_idx = self.clade_to_idx_mapping[child]
                            extended_set.append(child_idx)

            candidate_set = list(set(extended_set) - {0})
            candidate_set = [ 
                node for node in candidate_set if \
                (self.T.iloc[:,node].sum() >= min_clade_prevalence * self.T.shape[0]) & \
                (self.T.iloc[:,node].sum() <= max_clade_prevalence * self.T.shape[0])
            ]
            priors[candidate_set] = fixed_logit_value
        
        elif method =='smooth':
            
            scale = lambda x: (x-x.mean()) / x.std()
            priors = scale(self.M.mean(axis=0))
        
        else:

            logging.info('Priors set to None.')
            priors = None

        self.priors = priors

    ##

    # def rescale_loss_terms(self, k_soft, d_avg, e_avg):

    ##

    def cell_assignment(self):
        """
        Final cell assignment.
        """

        if self.final_binary_selection is None:
            raise ValueError('Call find_clones first!')

        selected_clades = self.T.columns[self.final_binary_selection.astype(bool)]
        ranked_clades = self.T[selected_clades].sum(axis=0).sort_values(ascending=False).index

        if not (self.T[selected_clades].sum(axis=1)==1).all():
            raise ValueError('Non-disjoint final MT-clones...')

        labels = np.array([ np.nan for i in range(self.T.shape[0]) ], dtype='O')
        for i,clade in enumerate(ranked_clades):
            labels[self.T[clade].values==1] = f'MiTo clone {i}'

        assert (self.T.index == self.tree.cell_meta.index).all()

        self.selected_clades = selected_clades
        self.tree.cell_meta['MiTo clone'] = labels
        self.tree.cell_meta['MiTo clone'] = pd.Categorical(self.tree.cell_meta['MiTo clone'])

    ##

    def extract_mut_order(self, pval_tresh=.01):
        """
        Extract diagonal-order of MT-SNVs using mutation assignments, to create a ordered list of
        MT-SNVs for plotting.
        """

        assert (self.M.index == self.tree.layers['transformed'].columns).all()

        # Define a MT-SNVs treshold
        mut_df = (10**(-self.M.max(axis=1))<=pval_tresh).to_frame('assignment')
        mut_df['prevalence'] = self.tree.layers['transformed'].sum(axis=0) / len(self.tree.leaves)
        mut_df['top_node'] = self.M.columns[self.M.values.argmax(axis=1)]

        top_nodes = mut_df.loc[mut_df['assignment'], 'top_node'].to_list()

        mutation_order = []
        for node in self.tree.depth_first_traverse_nodes():
            if node in top_nodes:
                mut = mut_df.loc[mut_df['top_node']==node].index[0]
                mutation_order.append(mut)

        unassigned_muts = mut_df.loc[~mut_df['assignment']].sort_values('prevalence').index.to_list()
        mutation_order += unassigned_muts

        self.ordered_muts = mutation_order
        self.mut_df = mut_df

    ##

    def find_clones(self, alpha=1, beta=1, gamma=1, delta=1, supp_factor=10, num_epochs=1000, 
                    early_stopping=True, temperature=1.0, min_temperature=0.01, anneal_rate=0.997, learning_rate=0.001):
        """
        Find MT-clones via gradient descent optimization of a custom loss function.
        Optimal clone assignment is achieved by minimizing a loss function that includes:
            1. Average (across clones) ratio between inside-clone vs outside-clone sum of cell-cell distances in MT-SNVs space (d) 
            2. Average (across clones) mutation enrichment (e)
            3. Total number of clones (k)
        (Soft) combinatorial pick of clade sets is subject to hierarchical constraints during optimization, to produce final,
        non-nested clonal labels.
        """
        
        # Start
        t = Timer()
        t.start()
        logging.info('Start to optimize clonal structure...')

        # Convert input matrices from pandas to torch tensors.
        T = torch.from_numpy(self.T.values.astype(np.float32))
        M = torch.from_numpy(self.M.values.astype(np.float32))
        D = torch.from_numpy(self.D.values.astype(np.float32))

        # Initialize logits for each clade (i.e., the learnable parameters of our model)
        if self.priors is not None:
            assert len(self.priors) == T.shape[1]
            logits = self.priors
        else:
            logits = np.random.normal(size=T.shape[1])
        
        # Transform priors with a sigmoid and store as tree node attributes and class slots
        initial_assignments = []
        for node,l in zip(self.T.columns, logits):
            z = 1 / (1 + np.exp(-l))
            initial_assignments.append(z)
            self.tree.set_attribute(node, 'prior_soft_selection', z)
        self.initial_assignments = np.array(initial_assignments)

        # Set optimizer and params
        logits = torch.tensor(logits, requires_grad=True) 
        optimizer = torch.optim.Adam(params=[logits], lr=learning_rate)
        logging.info('Set optimizer and learnable clade activation values.')

        # Main optimization loop.

        # Set tracking lists
        AV_DRATIOS = []; SUM_KSOFT = []; AV_ENRICHMENTS = []; CELL_COVERAGE = []; LOSSES = []
        SOFT_SELECTIONS = []
        TEMPERATURES = []

        # Initiate loss and set convergence tolerance
        prev_loss = None
        convergence_tol = .0001

        # (Pre-) compute statistics (d and e values), per clade.
        logging.info('Calculate per-clade statistics.')
        d_values = []
        e_values = []
        for j in range(T.shape[1]):
            S_j = T[:,j].bool()
            d_j = self.compute_d(D,S_j)
            e_j = self.compute_e(M,j)
            d_values.append(d_j)
            e_values.append(e_j)
        d_values = torch.stack(d_values)
        e_values = torch.stack(e_values)

        # Here we go
        logging.info('Main optimization loop...')
        for epoch in range(num_epochs):

            # Soft selection based on current iteration logits
            soft_selections_list = []
            for j in range(T.shape[1]):
                if self.ancestors[j]:
                    suppression = supp_factor * torch.sum(torch.stack([soft_selections_list[k] for k in self.ancestors[j]]))
                    value = torch.sigmoid((logits[j] - suppression) / temperature)
                else:
                    value = torch.sigmoid(logits[j] / temperature)
                soft_selections_list.append(value)

            soft_selections = torch.stack(soft_selections_list)
            k_soft = soft_selections.sum()

            # Weighted d and e averages
            epsilon = 1e-8
            d_avg = torch.sum(soft_selections * d_values) / (soft_selections.sum() + epsilon)
            e_avg = torch.sum(soft_selections * e_values) / (soft_selections.sum() + epsilon)

            # Cell coverage
            not_covered = torch.prod(1 - (T * soft_selections), dim=1) 
            not_covered_avg = torch.mean(not_covered)

            # Rescale loss components
            # ...
            
            # Loss definition
            loss = alpha * d_avg + beta * k_soft - gamma * e_avg + delta * not_covered_avg

            # Backpropagate and perform i+1 step
            loss.backward()
            optimizer.step()

            # Modify anneal temperature (to force sharper decisions at next step)
            temperature = max(temperature * anneal_rate, min_temperature)
            if epoch % 25 == 0:
                logging.info(
                    f"Epoch {epoch}, Loss: {loss.item():.3f} (d: {d_avg:.3f}, e: {e_avg:.3f}, k: {k_soft:.3f}, cell_coverage: {not_covered_avg:.3f}), Temperature: {temperature:.3f}"
                )          

            # Loss
            AV_DRATIOS.append(d_avg.detach().numpy())
            SUM_KSOFT.append(k_soft.detach().numpy())
            AV_ENRICHMENTS.append(e_avg.detach().numpy())
            CELL_COVERAGE.append(not_covered_avg.detach().numpy())
            LOSSES.append(loss.detach().numpy())

            # Params
            SOFT_SELECTIONS.append(soft_selections.detach().numpy())
            # Temp
            TEMPERATURES.append(temperature)

            # Evaluate loss connvergence and early stopping
            current_loss = loss.item()
            loss_converged = prev_loss is not None and abs(current_loss-prev_loss) < convergence_tol
            if early_stopping and loss_converged:
                break
            prev_loss = current_loss

        # Save learned soft selections, derive final binary selection.
        logging.info('Wrapping up...')
        self.soft_selection = np.vstack(SOFT_SELECTIONS)
        self.final_soft_selection = soft_selections.detach().numpy()
        self.final_binary_selection = np.where(self.final_soft_selection>0.5,1,0)
        df_track = pd.DataFrame({
            'loss' : LOSSES, 
            'average D ratio' : AV_DRATIOS, 
            'average mut enrichment' : AV_ENRICHMENTS, 
            'Soft selection sum' : SUM_KSOFT, 
            'cell coverage' : CELL_COVERAGE,
            'T' : TEMPERATURES
        })
        self.model_track = df_track

        # Attach final_soft_selection as tree attibutes
        for node,s in zip(self.T.columns, self.final_soft_selection):
            self.tree.set_attribute(node, 'soft_selection', s)

        # Assign cells to optmized MT-clones
        self.cell_assignment()
        self.extract_mut_order()

        logging.info(f'Final, optimized MT-clones: n={self.final_binary_selection.sum()}. {t.stop()}')


##