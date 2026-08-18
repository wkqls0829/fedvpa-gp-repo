"""
Variational Encoder with Gumbel-Softmax Prior for VPL-GP
Implements mixture prior from other clients' z distributions.

Learnable Gumbel-Softmax weights (VMTL-style):
  - prior_logits (π_j) are nn.Parameter, optimized via backprop through KL
  - Mixing weights: α_j = softmax((log π_j + g_j) / τ)  where g_j ~ Gumbel(0,1)
  - KL upper bound: KL(q||Σα_i·q_i) ≤ Σα_i·KL(q||q_i)
  - Temperature τ is annealed from τ_start to τ_end over training

Reference: Variational Multi-Task Learning with Gumbel-Softmax Priors (VMTL, 2021)
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from federatedscope.llm.model.variational_encoder import VariationalEncoder

logger = logging.getLogger(__name__)


class VariationalEncoderGP(VariationalEncoder):
    """
    Variational Encoder with Gumbel-Softmax Prior for Federated VPL.

    Extends VariationalEncoder to use a mixture prior from other clients'
    z distributions instead of a fixed standard normal prior.

    The mixture prior is: p_mixture(z) = Σ_i α_i * N(z; μ_i, σ_i²)
    where α_i are learnable Gumbel-Softmax weights and μ_i, σ_i² are
    received from the server (other clients' z distributions).
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128],
                 temperature=1.0, num_clients=10, max_logvar=0.0,
                 tau_anneal=True, tau_start=None, tau_end=0.1,
                 fixed_uniform_weights=False):
        super(VariationalEncoderGP, self).__init__(input_dim, latent_dim, hidden_dims, max_logvar=max_logvar)
        self.temperature = temperature
        self.num_clients = num_clients
        self.fixed_uniform_weights = fixed_uniform_weights

        # Temperature annealing settings
        self.tau_anneal = tau_anneal
        self.tau_start = tau_start if tau_start is not None else temperature
        self.tau_end = tau_end

        # Prior component distributions (received from server, NOT learnable)
        self.prior_mus = None  # (num_components, latent_dim)
        self.prior_logvars = None  # (num_components, latent_dim)

        # Learnable Gumbel-Softmax logits π_j (one per mixture component)
        # Initialized when prior is first received from server
        self.prior_logits = None  # Will become nn.Parameter(num_components,)

    def update_prior(self, client_mus, client_logvars, client_weights=None):
        """
        Update the mixture prior from other clients' z distributions.

        Args:
            client_mus: Mean vectors from other clients (num_components, latent_dim)
            client_logvars: Log variance vectors from other clients (num_components, latent_dim)
            client_weights: Optional sample-size weights for initialization (num_components,)
        """
        if isinstance(client_mus, list):
            client_mus = torch.stack(client_mus)
        if isinstance(client_logvars, list):
            client_logvars = torch.stack(client_logvars)
        if isinstance(client_weights, list):
            client_weights = torch.tensor(client_weights, dtype=torch.float32)

        # Ensure tensors are on the same device
        device = next(self.parameters()).device
        self.prior_mus = client_mus.to(device)
        self.prior_logvars = client_logvars.to(device)

        num_components = len(client_mus)

        # Initialize or resize learnable logits
        if self.prior_logits is None or self.prior_logits.shape[0] != num_components:
            # Warm-start from server weights (sample-size proportional) if available
            if client_weights is not None:
                if isinstance(client_weights, torch.Tensor):
                    w = client_weights.float()
                else:
                    w = torch.tensor(client_weights, dtype=torch.float32)
                # Normalize
                w = w / (w.sum() + 1e-8)
                init_logits = torch.log(w + 1e-8)
            else:
                # Uniform initialization
                init_logits = torch.zeros(num_components)

            self.prior_logits = nn.Parameter(init_logits.to(device))
            logger.info(f"Initialized learnable prior_logits with {num_components} components")

        # Log statistics
        mu_norm = torch.norm(self.prior_mus, dim=-1).mean().item()
        logvar_mean = self.prior_logvars.mean().item()

        # Compute average distance between mus
        if num_components > 1:
            mu_distances = []
            for i in range(num_components):
                for j in range(i + 1, num_components):
                    dist = torch.norm(self.prior_mus[i] - self.prior_mus[j]).item()
                    mu_distances.append(dist)
            avg_mu_distance = np.mean(mu_distances) if mu_distances else 0.0
        else:
            avg_mu_distance = 0.0

        # Log current learned weights
        with torch.no_grad():
            learned_weights = F.softmax(self.prior_logits, dim=-1)
            weights_str = ', '.join([f'{w:.4f}' for w in learned_weights.tolist()])

        logger.info(f"Updated VPL-GP prior: mu_norm={mu_norm:.4f}, "
                   f"logvar_mean={logvar_mean:.4f}, num_clients={num_components}, "
                   f"avg_mu_distance={avg_mu_distance:.4f}, "
                   f"learned_weights=[{weights_str}], tau={self.temperature:.4f}")

    def anneal_temperature(self, current_round, total_rounds):
        """
        Anneal Gumbel-Softmax temperature from tau_start to tau_end.

        Uses exponential decay: τ(t) = τ_start * (τ_end / τ_start)^(t/T)

        Args:
            current_round: Current training round
            total_rounds: Total number of training rounds
        """
        if not self.tau_anneal:
            return

        progress = current_round / max(total_rounds, 1)
        # Exponential decay
        self.temperature = self.tau_start * (self.tau_end / self.tau_start) ** progress

        logger.info(f"Temperature annealed to {self.temperature:.4f} "
                   f"(round {current_round}/{total_rounds})")

    def _compute_gumbel_weights(self, batch_size=1):
        """
        Compute mixture weights via Gumbel-Softmax on learnable logits.

        α_j = softmax((log π_j + g_j) / τ)

        During training: adds Gumbel noise for exploration
        During eval: uses straight softmax (no noise)
        If fixed_uniform_weights=True: returns 1/K for all components.

        Args:
            batch_size: Number of weight samples to generate

        Returns:
            weights: (batch_size, num_components) mixture weights
        """
        num_components = self.prior_logits.shape[0]
        device = self.prior_logits.device

        # Ablation: fixed uniform weights (no learning)
        if self.fixed_uniform_weights:
            return torch.ones(
                batch_size, num_components,
                device=device) / num_components

        if self.training:
            # Gumbel noise for exploration
            u = torch.rand(batch_size, num_components, device=device).clamp(1e-8, 1 - 1e-8)
            g = -torch.log(-torch.log(u))

            # Gumbel-Softmax
            logits = (self.prior_logits.unsqueeze(0) + g) / self.temperature
        else:
            # No noise during eval
            logits = self.prior_logits.unsqueeze(0).expand(batch_size, -1) / self.temperature

        weights = F.softmax(logits, dim=-1)  # (batch_size, num_components)
        return weights

    def sample_prior(self, batch_size, use_gumbel=True):
        """
        Sample from the mixture prior using Gumbel-Softmax.

        Args:
            batch_size: Number of samples to generate
            use_gumbel: Whether to use Gumbel-Softmax (default: True)

        Returns:
            z_samples: Sampled latent vectors (batch_size, latent_dim)
        """
        if self.prior_mus is None:
            # Fallback to standard normal
            return torch.randn(batch_size, self.latent_dim,
                             device=next(self.parameters()).device)

        num_components = len(self.prior_mus)
        device = self.prior_mus.device

        if use_gumbel and num_components > 1 and self.prior_logits is not None:
            # Learnable Gumbel-Softmax weights
            component_probs = self._compute_gumbel_weights(batch_size)  # (batch_size, num_components)

            # Vectorized: sample from all components at once
            stds = torch.exp(0.5 * self.prior_logvars)  # (K, D)
            eps = torch.randn(batch_size, num_components,
                              self.latent_dim, device=device)
            # (B, K, D) = (1, K, D) + (B, K, D) * (1, K, D)
            z_all = self.prior_mus.unsqueeze(0) + eps * stds.unsqueeze(0)
            weights = component_probs.unsqueeze(-1)  # (B, K, 1)
            z_samples = (z_all * weights).sum(dim=1)  # (B, D)
        else:
            # Vectorized: uniform weights
            stds = torch.exp(0.5 * self.prior_logvars)  # (K, D)
            eps = torch.randn(batch_size, num_components,
                              self.latent_dim, device=device)
            z_all = self.prior_mus.unsqueeze(0) + eps * stds.unsqueeze(0)
            z_samples = z_all.mean(dim=1)  # (B, D)

        return z_samples

    def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
        """
        Compute KL divergence using the upper bound (VMTL Proposition 1):

            KL(q(z|x) || Σ α_i · q_i(z)) ≤ Σ α_i · KL(q(z|x) || q_i(z))

        where α_i are learnable Gumbel-Softmax weights and q_i = N(μ_i, σ_i²).

        This is:
          - Closed-form (each KL(q||q_i) is Gaussian KL)
          - Differentiable w.r.t. prior_logits (through α_i)
          - An upper bound, so minimizing it still minimizes the true KL

        Args:
            mu: Mean of posterior q(z|x) (batch_size, latent_dim)
            logvar: Log variance of posterior q(z|x) (batch_size, latent_dim)
            use_gumbel_prior: Whether to use Gumbel-Softmax prior (default: True)

        Returns:
            kl: KL divergence upper bound (scalar)
        """
        if self.prior_mus is None:
            # Fallback to standard KL divergence KL(q || N(0,I))
            return super().kl_divergence(mu, logvar)

        batch_size = mu.shape[0]
        num_components = len(self.prior_mus)

        # Compute learnable Gumbel-Softmax weights (differentiable w.r.t. prior_logits)
        alpha = self._compute_gumbel_weights(batch_size)  # (batch, K)

        # Vectorized KL: KL(q || q_i) for all components at once
        # prior_mus: (K, D), prior_logvars: (K, D)
        mu_i = self.prior_mus.unsqueeze(0)       # (1, K, D)
        logvar_i = self.prior_logvars.unsqueeze(0)  # (1, K, D)
        mu_exp = mu.unsqueeze(1)                 # (B, 1, D)
        logvar_exp = logvar.unsqueeze(1)         # (B, 1, D)

        # Use exp(logvar - logvar_i) to avoid separate exp calls
        kl_stack = 0.5 * (
            logvar_i - logvar_exp
            + torch.exp(logvar_exp - logvar_i)
            + (mu_exp - mu_i).pow(2) * torch.exp(-logvar_i)
            - 1.0
        ).sum(dim=-1)  # (B, K)

        # Weighted sum: KL_upper = Σ α_i · KL(q || q_i)
        kl = (alpha * kl_stack).sum(dim=-1).mean()

        # Log comparison with standard KL for debugging (1% of the time)
        if random.random() < 0.01:
            standard_kl = super().kl_divergence(mu, logvar)

            with torch.no_grad():
                learned_weights = F.softmax(self.prior_logits, dim=-1)
                top_weights, top_indices = torch.topk(learned_weights, k=min(3, num_components))
                top_str = ', '.join([f"c{idx.item()}:{w.item():.4f}"
                                     for w, idx in zip(top_weights, top_indices)])

                # Entropy of learned weights (higher = more uniform)
                weight_entropy = -(learned_weights * torch.log(learned_weights + 1e-8)).sum().item()
                max_entropy = np.log(num_components)

            logger.info(f"KL upper bound={kl.item():.4f}, standard={standard_kl.item():.4f}, "
                       f"tau={self.temperature:.4f}, "
                       f"top_weights=[{top_str}], "
                       f"weight_entropy={weight_entropy:.4f}/{max_entropy:.4f}")

        return kl
