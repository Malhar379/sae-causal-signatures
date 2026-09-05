import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def sae_loss(x, x_hat, z, sparsity_weight=0.01):
    recon_loss = F.mse_loss(x_hat, x)
    sparsity_loss = z.abs().sum(dim=1).mean()
    total_loss = recon_loss + sparsity_weight * sparsity_loss
    return total_loss, recon_loss, sparsity_loss