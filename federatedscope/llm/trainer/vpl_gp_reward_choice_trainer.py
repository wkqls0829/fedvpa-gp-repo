"""
VPL-GP Reward Choice Trainer
Extends VPLRewardChoiceTrainer with Gumbel-Softmax Prior support.
"""
import torch
import torch.nn as nn
import logging
import numpy as np

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.vpl_reward_choice_trainer import VPLRewardChoiceTrainer
from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP

logger = logging.getLogger(__name__)


class VPLGPRewardChoiceTrainer(VPLRewardChoiceTrainer):
    """
    VPL-GP Reward Choice Trainer with Gumbel-Softmax Prior.
    
    Extends VPLRewardChoiceTrainer to use a mixture prior from other clients'
    z distributions instead of a fixed standard normal prior.
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        # Initialize parent with GP prior flag
        super().__init__(model, data, device, config, only_for_eval, monitor)
        
        # VPL-GP specific hyperparameters (already set in parent, but
        # kept here for backwards compatibility / explicit logging)
        self.vpl_gp_temperature = getattr(config.llm, 'vpl_gp_temperature', 1.0)
        self.vpl_use_gp_prior = getattr(config.llm, 'vpl_use_gp_prior', False)
        self.num_clients = getattr(config.federate, 'client_num', 10)

        # NOTE: variational_encoder (VariationalEncoderGP) is already
        # created in parent __init__ when vpl_use_gp_prior=True, with
        # correct max_logvar and tau_anneal params. Do NOT re-create it.

        # Z history for visualization
        if not hasattr(self, 'z_history'):
            self.z_history = []
        if not hasattr(self, 'z_mu_history'):
            self.z_mu_history = []
        if not hasattr(self, 'z_logvar_history'):
            self.z_logvar_history = []

        # Client z distribution (average over batches)
        if not hasattr(self, 'client_z_mu'):
            self.client_z_mu = None
        if not hasattr(self, 'client_z_logvar'):
            self.client_z_logvar = None
        
        logger.info(f'VPLGPRewardChoiceTrainer initialized with latent_dim={self.vpl_latent_dim}, '
                   f'kl_weight={self.vpl_kl_weight}, temperature={self.vpl_gp_temperature}, '
                   f'num_clients={self.num_clients}')
    
    def _hook_on_batch_forward(self, ctx):
        """
        Forward pass with VPL-GP (mixture prior).
        """
        # Call parent method
        super()._hook_on_batch_forward(ctx)
        
        # Collect z values for visualization (move to CPU immediately to save GPU memory)
        if hasattr(ctx, 'vpl_z') and ctx.vpl_z is not None:
            z = ctx.vpl_z
            if isinstance(z, torch.Tensor):
                z = z.detach().cpu()  # Move to CPU immediately
            # Limit history size to prevent memory buildup (keep only last 100 batches)
            if len(self.z_history) > 100:
                self.z_history = self.z_history[-100:]
            self.z_history.append(z)
    
    def _hook_on_fit_end(self, ctx):
        """
        Collect z distribution and values at end of round.
        """
        # Call parent method
        super()._hook_on_fit_end(ctx)
        
        # Collect z values for this round
        if len(self.z_history) > 0:
            # Average z values to get client z distribution
            z_values = torch.cat(self.z_history, dim=0)  # (num_batches * batch_size, latent_dim)
            
            # Compute mean and logvar over all z values
            self.client_z_mu = z_values.mean(dim=0)  # (latent_dim,)
            z_var = z_values.var(dim=0)  # (latent_dim,)
            self.client_z_logvar = torch.log(z_var + 1e-8)  # (latent_dim,)
            
            # Store z values for visualization (sample a few, keep on CPU)
            num_samples = min(10, len(z_values))  # Sample up to 10 for visualization
            sampled_indices = torch.randperm(len(z_values))[:num_samples]
            self.client_z_values = z_values[sampled_indices].cpu()  # Keep on CPU to save GPU memory
            
            logger.info(f"Collected z for round {ctx.cur_round if hasattr(ctx, 'cur_round') else 'unknown'} "
                       f"(shape: {self.client_z_values.shape}, from {len(self.z_history)} batches)")
        
        # Clear history for next round (force garbage collection)
        self.z_history = []
        import gc
        gc.collect()  # Force garbage collection after clearing history
    
    def get_client_z_distribution(self):
        """
        Get client's z distribution (mu, logvar) for server aggregation.
        
        Returns:
            (mu, logvar): Tuple of mean and log variance tensors
        """
        if self.client_z_mu is None or self.client_z_logvar is None:
            return None
        
        return (self.client_z_mu.clone(), self.client_z_logvar.clone())
    
    def get_client_z_values(self):
        """
        Get client's z values for visualization.
        
        Returns:
            z_values: Tensor of z values (num_samples, latent_dim)
        """
        if not hasattr(self, 'client_z_values') or self.client_z_values is None:
            return None
        
        return self.client_z_values.clone()
    
    def update_prior_from_server(self, client_mus, client_logvars, client_weights):
        """
        Update the mixture prior from server.
        
        Args:
            client_mus: Mean vectors from other clients (num_clients, latent_dim)
            client_logvars: Log variance vectors from other clients (num_clients, latent_dim)
            client_weights: Weights for each client distribution (num_clients,)
        """
        if hasattr(self.variational_encoder, 'update_prior'):
            self.variational_encoder.update_prior(client_mus, client_logvars, client_weights)
    
    def update_orthogonal_label_from_server(self, label):
        """
        Update orthogonal label from server.
        
        Args:
            label: Orthogonal label (int)
        """
        # Store label for potential use in orthogonal loss
        self.orthogonal_label = label
    
    def get_client_orthogonal_prototypes(self):
        """
        Get client's orthogonal prototypes for visualization.
        
        Returns:
            prototypes: Tensor of prototypes (num_prototypes, latent_dim) or None
        """
        if hasattr(self, 'orthogonal_prototypes'):
            return self.orthogonal_prototypes
        return None


def call_vpl_gp_reward_choice_trainer(trainer_type):
    if trainer_type == 'vplgprewardchoicetrainer':
        trainer_builder = VPLGPRewardChoiceTrainer
        return trainer_builder


register_trainer('vplgprewardchoicetrainer', call_vpl_gp_reward_choice_trainer)
