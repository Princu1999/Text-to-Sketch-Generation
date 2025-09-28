from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(target, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma, w_kl=0.5):
    """ Compute the combined reconstruction and KL divergence loss. """
    # Gaussian Mixture Log-Likelihood Loss
    x, y, p1, p2, p3 = target[..., 0], target[..., 1], target[..., 2], target[..., 3], target[..., 4]
    # norm_x = (x.unsqueeze(-1) - mu_x) /sigma_x
    # norm_y = (y.unsqueeze(-1) - mu_y) / sigma_y
    # z = norm_x ** 2 + norm_y ** 2 - 2 * rho_xy * norm_x * norm_y
    # denom = 1 /2 * (1 - rho_xy ** 2)

    # gaussian_nll = (z/denom  ) + safe_log(abs(sigma_x * sigma_y) * torch.sqrt(abs(1 - rho_xy ** 2))+1e-6)
    # gmm_loss = -torch.logsumexp(safe_log(pi + 1e-4) - gaussian_nll, dim=-1).mean()

    M = pi.shape[-1]
    safe_pi = torch.clamp(pi, min=1e-6)
    safe_rho = torch.clamp(rho_xy, min=-0.999, max=0.999)
    safe_sigma_x = torch.clamp(sigma_x, min=1e-6)
    safe_sigma_y = torch.clamp(sigma_y, min=1e-6)


    norm_x = (x.unsqueeze(-1) - mu_x) / safe_sigma_x
    norm_y = (y.unsqueeze(-1) - mu_y) / safe_sigma_y

    z = norm_x ** 2 + norm_y ** 2 - 2 * safe_rho * norm_x * norm_y
    denom = torch.clamp(2 * (1 - safe_rho ** 2), min=1e-6)   # Prevent division by zero

    gaussian_nll = (z / denom) + torch.clamp(safe_log(safe_sigma_x * safe_sigma_y * torch.sqrt(1 - safe_rho ** 2)*torch.pi*2 ),min = -5)

    log_pi = torch.log(safe_pi)
    max_log_pi = torch.max(log_pi, dim=-1, keepdim=True)[0]  # Stabilization trick
    gmm_loss = -torch.logsumexp(log_pi - gaussian_nll , dim=-1).mean()

    if (torch.isnan(gmm_loss)):
      print("gmm_loss is nan")
      gmm_loss = torch.tensor(0)


    # Categorical Cross-Entropy Loss for pen states
    pen_target = torch.stack([p1, p2, p3], dim=-1)
    pen_loss = -(pen_target * torch.clamp(safe_log(q),min = -5)).sum(dim=-1).mean()

    if (torch.isnan(pen_loss)):
      print("pen_loss is nan")
      pen_loss = torch.tensor(0)

    # KL Divergence Loss
    kl_loss = -0.5 * torch.mean(1 - torch.exp(torch.clamp(sigma, min=-5, max=5)) + - mu ** 2 + sigma)
    kl_loss = torch.clamp(kl_loss, min=0)
    if (torch.isnan(kl_loss)):
      print("kl_loss is nan")
      kl_loss = torch.tensor(0)

    return   gmm_loss + pen_loss + w_kl * kl_loss







def fit_gmm_with_uncertainty(latent_mus, latent_sigmas, num_components=5):
    """
    Fit a Gaussian Mixture Model (GMM) to the latent space of a class
    using both means and variances.
    """
    latent_mus = latent_mus.numpy()  # Shape: (50, 128)
    latent_sigmas = latent_sigmas.numpy()  # Shape: (50, 128)

    # Compute covariance matrices for each sample
    latent_covs = np.array([np.diag(sigma**2) for sigma in latent_sigmas])  # Shape: (50, 128, 128)

    # Fit GMM using full covariance matrices
    gmm = GaussianMixture(n_components=num_components, covariance_type='full', reg_covar=1e-6)
    gmm.fit(latent_mus)  # Use latent means for fitting

    return gmm



def safe_log(x):
    return torch.log(torch.clamp(x, min=1e-8))


