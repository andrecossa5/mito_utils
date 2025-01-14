"""
Bootstrap utils.
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from anndata import AnnData


##


def bootstrap_allele_tables(afm, layer='AD', frac_char_resampling=.8):
    """
    Bootstrap of an Allele Frequency Matrix (AFM) layer.
    Both sparse matrices and dense .layers can be passed.
    """

    # Get layer
    if layer in afm.layers:
        X = afm.layers[layer]
        X = X if not issparse(X) else X.A
    else:
        raise KeyError(f'{layer} not present in afm! Check your inputs...')

    # Resample afm.var index
    n = X.shape[1]
    if frac_char_resampling == 1:
        resampled_idx = np.random.choice(np.arange(n), n, replace=True) 
    else:    
        resampled_idx = np.random.choice(np.arange(n), round(n*frac_char_resampling), replace=False)   

    return X[:,resampled_idx], resampled_idx
    


##


def jackknife_allele_tables(afm, layer='AD'):
    """
    Jackknife of an Allele Frequency Matrix (AFM) layer.
    Both sparse matrices and dense .layers can be passed.
    """

    # Get layer
    if layer in afm.layers:
        X = afm.layers[layer]
        X = X if not issparse(X) else X.A
    else:
        raise KeyError(f'{layer} not present in afm! Check your inputs...')

    # Resample afm.var index
    n = X.shape[1]
    to_exclude = np.random.choice(np.arange(n), 1)[0]
    resampled_idx = [ x for x in np.arange(n) if x != to_exclude ]

    return X[:,resampled_idx], resampled_idx


##


def bootstrap_MiTo(afm, boot_replicate='observed', boot_strategy='feature_resampling', frac_char_resampling=.8):
    """
    Bootstrap MAESTER/RedeeM Allele Frequency matrices.
    """

    if boot_replicate != 'observed':

        if boot_strategy == 'jacknife':
            AD, _ = jackknife_allele_tables(afm, layer='AD')
            cov, idx = jackknife_allele_tables(afm, layer='site_coverage')                                              # USE SITE, NBBB
        elif boot_strategy == 'feature_resampling':
            AD, _ = bootstrap_allele_tables(afm, layer='AD', frac_char_resampling=frac_char_resampling)
            cov, idx = bootstrap_allele_tables(afm, layer='site_coverage', frac_char_resampling=frac_char_resampling)    # USE SITE, NBBB
        elif boot_strategy == 'counts_resampling':
            raise ValueError(f'#TODO: {boot_strategy} boot_strategy. This strategy is not supported yet.')
        else:
            raise ValueError(f'{boot_strategy} boot_strategy is not supported...')
        
        AF = csr_matrix(np.divide(AD, (cov+.0000001)))
        AD = csr_matrix(AD)
        cov = csr_matrix(cov)
        afm_new = AnnData(X=AF, obs=afm.obs, var=afm.var.iloc[idx,:], uns=afm.uns, layers={'AD':AD, 'site_coverage':cov})

    else:
        afm_new = afm.copy()

    return afm_new


##


def bootstrap_bin(afm, boot_replicate='observed', boot_strategy='feature_resampling', frac_char_resampling=.8):
    """
    Bootstrap scWGS/Cas9 AFMs.
    """

    if boot_replicate != 'observed':

        if boot_strategy == 'jacknife':
            X_new, idx = jackknife_allele_tables(afm, layer='bin')                                             
        elif boot_strategy == 'feature_resampling':
            X_new, idx = bootstrap_allele_tables(afm, layer='bin', frac_char_resampling=frac_char_resampling)  
        else:
            raise ValueError(f'{boot_strategy} boot_strategy is not supported...')
        
        X_new = csr_matrix(X_new)
        afm_new = AnnData(obs=afm.obs, var=afm.var.iloc[idx,:], uns=afm.uns, layers={'bin':X_new})

    else:
        afm_new = afm.copy()

    return afm_new


##


####### DEPRECATED code


# def bootstrap_allele_counts(ad, dp, frac=.8):
#     """
#     Bootstrapping of ad and dp count tables. ad --> alternative counts
#     dp --> coverage at that site. NB: AD and DP are assumed in shape cells x variants. 
#     Both sparse matrices and dense arrays can be passed.
#     """
# 
#     ad = ad if not issparse(ad) else ad.A
#     dp = dp if not issparse(dp) else dp.A
#     new_ad = np.zeros(ad.shape)
#     new_dp = np.zeros(ad.shape)
# 
#     for j in range(ad.shape[1]): # Iterate on variants
# 
#         alt_counts = ad[:,j]
#         ref_counts = dp[:,j] - ad[:,j]
# 
#         for i in range(ad.shape[0]): # Iterate on cells
# 
#             observed_alleles = np.concatenate([
#                 np.ones(alt_counts[i]), 
#                 np.zeros(ref_counts[i])
#             ])
#             n = round(observed_alleles.size*frac)
#             new_alt_counts = np.random.choice(observed_alleles, n, replace=True).sum() 
#             new_dp[i,j] = n
#             new_ad[i,j] = new_alt_counts
# 
#     return new_ad, new_dp, np.arange(ad.shape[1])


##