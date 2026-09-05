import torch
import torch.optim as optim
from src.sae import sae_loss, SparseAutoencoder


def train_sae(model, X_tensor, n_epochs=200, learning_rate=1e-3, sparsity_weight=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "recon": [], "sparsity": []}

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        x_hat, z = model(X_tensor)
        total_loss, recon_loss, sparsity_loss = sae_loss(X_tensor, x_hat, z, sparsity_weight)
        
        total_loss.backward()
        optimizer.step()
        
        history["total"].append(total_loss.item())
        history["recon"].append(recon_loss.item())
        history["sparsity"].append(sparsity_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: total={total_loss.item():.4f}  recon={recon_loss.item():.4f}  sparsity={sparsity_loss.item():.4f}")
    
    return history


def train_multiple_seeds(generator_fn, generator_kwargs, n_seeds=5, 
                          input_dim=6, latent_dim=20, sparsity_weight=0.03, 
                          n_epochs=200, data_seed=42):
    results = []
    
    X = generator_fn(n_samples=1000, seed=data_seed, **generator_kwargs)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        history = train_sae(model, X_tensor, n_epochs=n_epochs, sparsity_weight=sparsity_weight)
        
        x_hat, z = model(X_tensor)
        inactive = (z.abs() < 1e-3).sum(dim=1).float().mean().item()
        active = latent_dim - inactive
        final_recon = history["recon"][-1]
        
        results.append({
            "seed": seed,
            "model": model,
            "z": z.detach(),
            "final_recon": final_recon,
            "active_latents": active
        })
        print(f"Seed {seed}: final_recon={final_recon:.4f}  active_latents={active:.2f}")
    
    return results, X_tensor