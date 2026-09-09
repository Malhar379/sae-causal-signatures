import numpy as np
from sklearn.linear_model import LinearRegression

def compute_feature_recovery(z, X):
    A = X[:, 0].numpy()
    B = X[:, 1].numpy()
    z_np = z.numpy()
    
    n_latents = z_np.shape[1]
    corr_with_a = np.full(n_latents, np.nan)
    corr_with_b = np.full(n_latents, np.nan)
    
    for i in range(n_latents):
        if z_np[:, i].std() > 1e-8:  # skip dead latents
            corr_with_a[i] = np.corrcoef(z_np[:, i], A)[0, 1]
            corr_with_b[i] = np.corrcoef(z_np[:, i], B)[0, 1]
    
    return corr_with_a, corr_with_b


def partial_correlation(z_latent, target, control):
    reg_target = LinearRegression().fit(control.reshape(-1, 1), target)
    resid_target = target - reg_target.predict(control.reshape(-1, 1))
    
    reg_z = LinearRegression().fit(control.reshape(-1, 1), z_latent)
    resid_z = z_latent - reg_z.predict(control.reshape(-1, 1))
    
    return np.corrcoef(resid_target, resid_z)[0, 1]


def full_partial_correlation_analysis(results, X_tensor):
    A_np = X_tensor[:, 0].numpy()
    B_np = X_tensor[:, 1].numpy()
    n_latents = results[0]["z"].shape[1]
    n_seeds = len(results)
    
    partial_a = np.full((n_seeds, n_latents), np.nan)
    partial_b = np.full((n_seeds, n_latents), np.nan)
    
    for seed_idx, result in enumerate(results):
        z_np = result["z"].numpy()
        for lat in range(n_latents):
            if z_np[:, lat].std() > 1e-8:
                partial_a[seed_idx, lat] = partial_correlation(z_np[:, lat], A_np, B_np)
                partial_b[seed_idx, lat] = partial_correlation(z_np[:, lat], B_np, A_np)
    
    return partial_a, partial_b


def max_partial_correlation_per_seed(results, X_tensor):
    A_np = X_tensor[:, 0].numpy()
    B_np = X_tensor[:, 1].numpy()
    n_seeds = len(results)
    
    max_corr_a = np.zeros(n_seeds)
    max_corr_b = np.zeros(n_seeds)
    
    for seed_idx, result in enumerate(results):
        z_np = result["z"].numpy()
        n_latents = z_np.shape[1]
        
        corrs_a = []
        corrs_b = []
        for lat in range(n_latents):
            if z_np[:, lat].std() > 1e-8:
                corrs_a.append(abs(partial_correlation(z_np[:, lat], A_np, B_np)))
                corrs_b.append(abs(partial_correlation(z_np[:, lat], B_np, A_np)))
        
        max_corr_a[seed_idx] = max(corrs_a)
        max_corr_b[seed_idx] = max(corrs_b)
    
    return max_corr_a, max_corr_b


def latent_correlation_matrix(z):
    z_np = z.numpy()
    n_latents = z_np.shape[1]
    
    corr_matrix = np.full((n_latents, n_latents), np.nan)
    for i in range(n_latents):
        for j in range(n_latents):
            if z_np[:, i].std() > 1e-8 and z_np[:, j].std() > 1e-8:
                corr_matrix[i, j] = np.corrcoef(z_np[:, i], z_np[:, j])[0, 1]
    
    return corr_matrix


def summarize_geometry(corr_matrix):
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)  # exclude diagonal
    off_diagonal = corr_matrix[mask]
    off_diagonal = off_diagonal[~np.isnan(off_diagonal)]  # exclude dead-latent nans
    return np.mean(np.abs(off_diagonal))