"""
Variational Encoder for Variational Preference Learning (VPL)
Implements user-specific latent inference from preference data.
Based on: https://github.com/WEIRDLabUW/vpl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class VariationalEncoder(nn.Module):
    """
    Variational Encoder that infers user-specific latent z from preference data.
    
    The encoder learns q(z|preferences) - an approximate posterior distribution
    over user-specific latents given their preference labels.
    
    Args:
        input_dim: Dimension of the input preference features
        latent_dim: Dimension of the latent space z
        hidden_dims: List of hidden layer dimensions (default: [256, 128])
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128], max_logvar=0.0):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            max_logvar: Maximum log variance (clamps logvar to prevent large sigma)
                       Default 0.0 means sigma <= exp(0.5 * 0.0) = 1.0
                       Set to -2.0 for sigma <= exp(0.5 * -2.0) ≈ 0.368
        """
        super(VariationalEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_logvar = max_logvar  # Maximum log variance (clamps logvar to prevent large sigma)
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Mean and log-variance for variational posterior
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Initialize logvar bias to negative value to start with smaller variance
        # This helps prevent variance from becoming too large
        if hasattr(self.fc_logvar, 'bias') and self.fc_logvar.bias is not None:
            nn.init.constant_(self.fc_logvar.bias, -2.0)  # exp(-2.0) ≈ 0.135, smaller initial variance
        
    def encode(self, x):
        """
        Encode input preferences to latent parameters.
        
        Args:
            x: Input preference features (batch_size, input_dim)
            
        Returns:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log variance of the latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp logvar to prevent variance from becoming too large
        # This limits sigma = exp(0.5 * logvar) to be at most exp(0.5 * max_logvar)
        max_logvar = getattr(self, 'max_logvar', 0.0)  # Default: exp(0.0) = 1.0, so sigma <= 1.0
        if max_logvar is not None:
            logvar = torch.clamp(logvar, max=max_logvar)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from q(z|x).
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass: encode and sample latent.
        
        Args:
            x: Input preference features
            
        Returns:
            z: Sampled latent vector
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence KL(q(z|x) || p(z)) where p(z) is standard normal.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            kl: KL divergence (scalar or batch)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()


class PreferenceFeatureExtractor(nn.Module):
    """
    Extracts features from preference pairs for the variational encoder.
    
    This module processes preference data (chosen/rejected pairs) to create
    a fixed-size feature vector that can be used by the variational encoder.
    """
    def __init__(self, model_hidden_dim, feature_dim=128):
        super(PreferenceFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        
        # Project model hidden states to feature space
        self.projection = nn.Sequential(
            nn.Linear(model_hidden_dim * 2, 256),  # *2 for chosen+rejected
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, chosen_hidden, rejected_hidden):
        """
        Extract features from preference pairs.
        
        Args:
            chosen_hidden: Hidden states for chosen responses (batch_size, hidden_dim)
            rejected_hidden: Hidden states for rejected responses (batch_size, hidden_dim)
            
        Returns:
            features: Extracted preference features (batch_size, feature_dim)
        """
        # Concatenate chosen and rejected hidden states
        combined = torch.cat([chosen_hidden, rejected_hidden], dim=-1)
        features = self.projection(combined)
        return features


def compute_preference_features_from_logits(chosen_logits, rejected_logits, 
                                           choices, method='mean_pool'):
    """
    Compute preference features from model logits.
    
    Args:
        chosen_logits: Logits for chosen responses
        rejected_logits: Logits for rejected responses
        choices: Choice token indices
        method: Feature extraction method ('mean_pool', 'max_pool', 'choice_logits')
        
    Returns:
        features: Extracted features
    """
    if method == 'choice_logits':
        # Use logits at choice positions
        chosen_features = chosen_logits[..., choices].mean(dim=-2)  # (batch, num_choices)
        rejected_features = rejected_logits[..., choices].mean(dim=-2)
        features = torch.cat([chosen_features, rejected_features], dim=-1)
    elif method == 'mean_pool':
        # Mean pooling over sequence
        chosen_features = chosen_logits.mean(dim=-2)  # (batch, vocab_size)
        rejected_features = rejected_logits.mean(dim=-2)
        # Take mean over vocab to reduce dimension
        features = torch.cat([chosen_features.mean(dim=-1, keepdim=True),
                             rejected_features.mean(dim=-1, keepdim=True)], dim=-1)
    else:  # max_pool
        chosen_features = chosen_logits.max(dim=-2)[0]
        rejected_features = rejected_logits.max(dim=-2)[0]
        features = torch.cat([chosen_features.mean(dim=-1, keepdim=True),
                             rejected_features.mean(dim=-1, keepdim=True)], dim=-1)
    
    return features
