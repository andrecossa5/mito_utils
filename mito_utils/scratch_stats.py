"""
Discretization. NB and tresholds.
"""

import os
from mito_utils.preprocessing import *
from mito_utils.phylo import *
from mito_utils.utils import *
from mito_utils.plotting_base import *
from mito_utils.phylo_plots import *


##


# Set paths
path_main = '/Users/IEO5505/Desktop/mito_bench'
path_data = os.path.join(path_main, 'data')

# Params
sample = 'MDA_clones_100'
filtering = 'MI_TO'
t = 0.05

# Data
afm = read_one_sample(path_data, sample, with_GBC=True, nmads=10)

# Read
_, a = filter_cells_and_vars(
    afm, filtering=filtering, 
    max_AD_counts=2, af_confident_detection=0.05, min_cell_number=5
)



# AD, DP, _ = get_AD_DP(a)
# WT = DP.A.T - AD.A.T
# MUT = AD.A.T
# noise = (a.X.mean(axis=0))# * .1
# 
# L = []
# for i in range(a.shape[1]):
#     p = nbinom.cdf(k=WT[:,i], n=MUT[:,i], p=noise[i])
#     genotypes = np.zeros(p.size)
#     genotypes[p<.1] = 1                         # MUT: p WT very low (<.1)
#     genotypes[p>=.9] = 0                      # WT: p WT very high (>.9)
#     genotypes[(p>=.1) & (p<.9)] = np.nan 
#     # thr = math.log(0.1) / math.log(1 - a.X.mean(axis=0)[i])
#     # genotypes[(np.isnan(p)) & (WT[:,i]>thr)] = 0  
#     # genotypes[(np.isnan(p)) & (WT[:,i]<=thr)] = np.nan   
#     L.append(genotypes.tolist())
# 
# MUT[:,i]
# WT[:,i]
# a.X[:,i]
# 
# np.nansum(np.array(L))

# tree_nb = build_tree(a, bin_method='nb')
# tree_simple = build_tree(a, bin_method='simple', t=.05)
# np.corrcoef(tree_nb.get_dissimilarity_map().values.flatten(), tree_simple.get_dissimilarity_map().values.flatten())

# fig, axs = plt.subplots(1,2,figsize=(10,5))
# plot_tree(tree_nb, ax=axs[0])
# plot_tree(tree_simple, ax=axs[1])
# fig.tight_layout()
# plt.show()

# CI(tree_nb)
# CI(tree_simple)
# char_compatibility(tree_nb).mean()
# char_compatibility(tree_simple).mean()

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO




def simplified_model(AD, DP_variant, DP_cell, n_cells, n_variants):
    # Prior for the global parameter pi (probability of presence for each variant)
    pi = pyro.sample("pi", dist.Beta(2.0, 2.0).expand([n_variants]).to_event(1))  # Shape: (n_variants,)
    
    with pyro.plate("variants", n_variants):
        # Hierarchical priors for theta1 (presence) and theta0 (absence)
        alpha_theta1 = pyro.sample("alpha_theta1", dist.Gamma(2.0, 2.0))  # Shape: ()
        beta_theta1 = pyro.sample("beta_theta1", dist.Gamma(2.0, 2.0))    # Shape: ()
        theta1 = pyro.sample("theta1", dist.Beta(alpha_theta1, beta_theta1).expand([n_variants]).to_event(1))  # Shape: (n_variants,)

        alpha_theta0 = pyro.sample("alpha_theta0", dist.Gamma(2.0, 2.0))  # Shape: ()
        beta_theta0 = pyro.sample("beta_theta0", dist.Gamma(2.0, 2.0))    # Shape: ()
        theta0 = pyro.sample("theta0", dist.Beta(alpha_theta0, beta_theta0).expand([n_variants]).to_event(1))  # Shape: (n_variants,)
        
        with pyro.plate("cells", n_cells):
            # Expand theta1, theta0, and pi to match the shape of AD and DP
            theta1_expanded = theta1.unsqueeze(0).expand(n_cells, n_variants)  # Shape: (n_cells, n_variants)
            theta0_expanded = theta0.unsqueeze(0).expand(n_cells, n_variants)  # Shape: (n_cells, n_variants)
            pi_expanded = pi.unsqueeze(0).expand(n_cells, n_variants)          # Shape: (n_cells, n_variants)

            # Calculate the log-likelihoods
            likelihood1 = dist.Binomial(total_count=DP, probs=theta1_expanded).log_prob(AD).exp()  # Shape: (n_cells, n_variants)
            likelihood0 = dist.Binomial(total_count=DP, probs=theta0_expanded).log_prob(AD).exp()  # Shape: (n_cells, n_variants)

            # Combine likelihoods to form the mixture model
            mixture_logit = pi_expanded * likelihood1 + (1 - pi_expanded) * likelihood0
            pyro.sample("obs", dist.Binomial(total_count=DP, probs=mixture_logit), obs=AD)





guide = pyro.infer.autoguide.AutoNormal(simplified_model)



# Optimizer setup
adam_params = {"lr": 0.02, "clip_norm": 10.0}
optimizer = ClippedAdam(adam_params)

# Setup the inference algorithm
svi = SVI(simplified_model, guide, optimizer, loss=Trace_ELBO())


##


# DATA
AD, DP, _ = get_AD_DP(a)
AD = AD.A.T
DP = DP.A.T
DP_variant = DP.mean(axis=0)
DP_cell = DP.sum(axis=1)

AD = torch.tensor(AD, dtype=torch.float32)  # Convert AD to a PyTorch tensor
DP = torch.tensor(DP, dtype=torch.float32)  # Convert DP to a PyTorch tensor
DP_variant = torch.tensor(DP_variant, dtype=torch.float32)  # Convert DP_variant to a PyTorch tensor
DP_cell = torch.tensor(DP_cell, dtype=torch.float32) 


# Training loop
n_steps = 1000  # Number of training steps
losses = []

# Training loop for the SVI
for step in range(n_steps):
    loss = svi.step(AD, DP_variant, DP_cell, DP.shape[0], DP.shape[1])
    losses.append(loss)
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss:.4f}")


# Convergence can be monitored by observing the loss

# Sample posterior samples
posterior_samples = guide(AD, DP_variant, DP_cell, DP.shape[0], DP.shape[1])

# Extract the relevant parameters from the posterior samples
theta1_posterior = posterior_samples['theta1']  # Shape: (n_variants,)
theta0_posterior = posterior_samples['theta0']  # Shape: (n_variants,)
pi_posterior = posterior_samples['pi']          # Shape: (n_variants,)


# Expand the theta and pi parameters to match the shape of DP
theta1_expanded = theta1_posterior.unsqueeze(0).expand(DP.shape[0], DP.shape[1])  # Shape: (n_cells, n_variants)
theta0_expanded = theta0_posterior.unsqueeze(0).expand(DP.shape[0], DP.shape[1])  # Shape: (n_cells, n_variants)
pi_expanded = pi_posterior.unsqueeze(0).expand(DP.shape[0], DP.shape[1])          # Shape: (n_cells, n_variants)

# Compute the likelihoods under each component for each cell and variant
likelihood1 = dist.Binomial(total_count=DP, probs=theta1_expanded).log_prob(AD).exp()  # Shape: (n_cells, n_variants)
likelihood0 = dist.Binomial(total_count=DP, probs=theta0_expanded).log_prob(AD).exp()  # Shape: (n_cells, n_variants)

# Compute posterior probabilities P1 and P0
P1 = (pi_expanded * likelihood1) / (pi_expanded * likelihood1 + (1 - pi_expanded) * likelihood0)  # Shape: (n_cells, n_variants)
P0 = 1 - P1  # Since P0 + P1 = 1


P0.shape


DP[:5,:]
(AD[:5,:]>0) == (P0[:5,:]>.9)


# import matplotlib.pyplot as plt
# 
# # Plot loss over time
# plt.plot(losses)
# plt.xlabel("Step")
# plt.ylabel("Loss")
# plt.title("Loss during training")
# plt.show()
# 
# # Example: Plot posterior distributions
# plt.hist(theta1_posterior.detach().numpy(), bins=30, density=True, alpha=0.5, label='theta1')
# plt.hist(theta0_posterior.detach().numpy(), bins=30, density=True, alpha=0.5, label='theta0')
# plt.legend()
# plt.title("Posterior distributions of theta1 and theta0")
# plt.show()
