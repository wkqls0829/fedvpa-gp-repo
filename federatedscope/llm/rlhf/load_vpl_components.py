"""
Utility functions to load VPL components (variational encoder, feature extractor, latent projection)
from a trained selector model checkpoint.
"""
import torch
import torch.nn as nn
import logging
import os

from federatedscope.llm.model.variational_encoder import VariationalEncoder
from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP

logger = logging.getLogger(__name__)


def load_vpl_components_from_checkpoint(checkpoint_path, config, device='cuda:0'):
    """
    Load VPL components (variational encoder, feature extractor, latent projection)
    from a trained selector model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration object with VPL hyperparameters
        device: Device to load models on
        
    Returns:
        tuple: (variational_encoder, feature_extractor, latent_projection, z_to_embedding)
        or (None, None, None, None) if not found
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_state_dict = ckpt.get('model', ckpt)
        
        # Extract VPL hyperparameters from config
        vpl_latent_dim = getattr(config.llm, 'vpl_latent_dim', 32)
        vpl_use_gp_prior = getattr(config.llm, 'vpl_use_gp_prior', False)
        vpl_use_feature_difference = getattr(config.llm, 'vpl_use_feature_difference', True)
        vpl_use_llm_feature_extractor = getattr(config.llm, 'vpl_use_llm_feature_extractor', True)
        vpl_feature_method = getattr(config.llm, 'vpl_feature_method', 'choice_logits')
        vpl_use_difference_only = getattr(config.llm, 'vpl_use_difference_only', False)
        
        # Determine input dimensions (must match VPLRewardChoiceTrainer logic)
        try:
            embedding_dim = getattr(config.model, 'hidden_size', None) or config.model.get('hidden_size', 2048)
        except Exception:
            embedding_dim = 2048  # Default for gemma-2b
        
        if vpl_use_llm_feature_extractor and vpl_use_feature_difference:
            if vpl_use_difference_only:
                raw_feature_dim = embedding_dim  # Only difference
            else:
                raw_feature_dim = embedding_dim * 3  # [chosen, rejected, difference]
        elif vpl_use_feature_difference:
            raw_feature_dim = embedding_dim  # Only difference
        elif vpl_feature_method == 'choice_logits':
            choices = getattr(config.trainer, 'choices', ['A', 'B'])
            raw_feature_dim = len(choices) * 2  # choice_logits for chosen and rejected
        else:
            raw_feature_dim = 2  # Fallback
        
        feature_extractor_output_dim = 128  # From VPLRewardChoiceTrainer
        
        # Initialize variational encoder
        if vpl_use_gp_prior:
            num_clients = getattr(config.federate, 'client_num', 10)
            vpl_gp_temperature = getattr(config.llm, 'vpl_gp_temperature', 1.0)
            variational_encoder = VariationalEncoderGP(
                input_dim=feature_extractor_output_dim,
                latent_dim=vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                temperature=vpl_gp_temperature,
                num_clients=num_clients
            ).to(device)
        else:
            variational_encoder = VariationalEncoder(
                input_dim=feature_extractor_output_dim,
                latent_dim=vpl_latent_dim,
                hidden_dims=[512, 256, 128]
            ).to(device)
        
        # Initialize feature extractor (must match VPLRewardChoiceTrainer architecture)
        # If vpl_use_llm_feature_extractor=True, use [raw_feature_dim -> 512 -> 256 -> 128]
        # Otherwise, use [raw_feature_dim -> 256 -> 512 -> 256 -> 128]
        if vpl_use_llm_feature_extractor:
            # Match VPLRewardChoiceTrainer architecture for projection-based feature extractor
            feature_extractor = nn.Sequential(
                nn.Linear(raw_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, feature_extractor_output_dim)
            ).to(device)
        else:
            # Match VPLRewardChoiceTrainer architecture for MLP feature extractor
            feature_extractor = nn.Sequential(
                nn.Linear(raw_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, feature_extractor_output_dim)
            ).to(device)
        
        # Initialize latent projection
        choices = getattr(config.trainer, 'choices', ['A', 'B'])
        latent_projection = nn.Linear(vpl_latent_dim, len(choices)).to(device)

        # Initialize z_to_embedding for conditional generation
        # Get embedding dimension from model type
        embedding_dim = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'type'):
                model_type = str(config.model.type).lower()
                if 'gemma-2b' in model_type:
                    embedding_dim = 2048
                elif 'gemma-7b' in model_type:
                    embedding_dim = 4096
                elif 'qwen' in model_type:
                    # Qwen models: try to get from transformers config
                    try:
                        from transformers import AutoConfig
                        model_name = config.model.type.split('@')[0]
                        model_config = AutoConfig.from_pretrained(model_name)
                        embedding_dim = getattr(model_config, 'hidden_size', None) or getattr(model_config, 'vocab_size', None)
                        # Qwen2-0.5B: 896, Qwen2-1.5B: 1536, Qwen2-7B: 3584
                        if embedding_dim is None:
                            if '0.5b' in model_type or '0.5B' in model_type:
                                embedding_dim = 896
                            elif '1.5b' in model_type or '1.5B' in model_type:
                                embedding_dim = 1536
                            elif '7b' in model_type or '7B' in model_type:
                                embedding_dim = 3584
                    except Exception:
                        # Fallback for Qwen
                        if '0.5b' in model_type or '0.5B' in model_type:
                            embedding_dim = 896
                        elif '1.5b' in model_type or '1.5B' in model_type:
                            embedding_dim = 1536
                        elif '7b' in model_type or '7B' in model_type:
                            embedding_dim = 3584
                elif 'llama' in model_type or 'mistral' in model_type:
                    # Try to get from transformers config
                    try:
                        from transformers import AutoConfig
                        model_name = config.model.type.split('@')[0]
                        model_config = AutoConfig.from_pretrained(model_name)
                        embedding_dim = getattr(model_config, 'hidden_size', None) or getattr(model_config, 'embed_size', None)
                    except Exception:
                        pass
        except Exception:
            pass
        
        if embedding_dim is None:
            # Fallback: try config attributes
            try:
                embedding_dim = getattr(config.model, 'hidden_size', None) or getattr(config.model, 'embed_size', None)
            except Exception:
                pass
        
        if embedding_dim is None:
            embedding_dim = 2048  # Default fallback for gemma-2b
        
        z_to_embedding = nn.Linear(vpl_latent_dim, embedding_dim).to(device)
        # Ensure z_to_embedding uses the same dtype as the model
        # Try to get model dtype from config
        model_dtype = getattr(config.model, 'torch_dtype', None) or getattr(config.model, 'dtype', None)
        if model_dtype is not None:
            if isinstance(model_dtype, str):
                if 'bfloat16' in model_dtype.lower() or 'bf16' in model_dtype.lower():
                    z_to_embedding = z_to_embedding.to(torch.bfloat16)
                elif 'float16' in model_dtype.lower() or 'fp16' in model_dtype.lower():
                    z_to_embedding = z_to_embedding.to(torch.float16)
            elif hasattr(torch, model_dtype):
                z_to_embedding = z_to_embedding.to(getattr(torch, model_dtype))
        logger.info(f"Initialized z_to_embedding: {vpl_latent_dim} -> {embedding_dim} (model type: {getattr(config.model, 'type', 'unknown')}, dtype: {z_to_embedding.weight.dtype})")
        
        # Try to load from checkpoint
        variational_encoder_loaded = False
        feature_extractor_loaded = False
        latent_projection_loaded = False
        z_to_embedding_loaded = False
        
        # Load variational encoder weights (only once, not in loop)
        if not variational_encoder_loaded:
            variational_encoder_keys = {k.replace('variational_encoder.', ''): v 
                                       for k, v in model_state_dict.items() 
                                       if 'variational_encoder' in k}
            if variational_encoder_keys:
                try:
                    variational_encoder.load_state_dict(variational_encoder_keys, strict=False)
                    variational_encoder_loaded = True
                    logger.info(f"Loaded variational encoder from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load variational encoder: {e}")
        
        # Load feature extractor weights (only once, not in loop)
        if not feature_extractor_loaded:
            feature_extractor_keys = {k.replace('feature_extractor.', ''): v 
                                     for k, v in model_state_dict.items() 
                                     if 'feature_extractor' in k}
            if feature_extractor_keys:
                try:
                    feature_extractor.load_state_dict(feature_extractor_keys, strict=False)
                    feature_extractor_loaded = True
                    logger.info(f"Loaded feature extractor from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load feature extractor: {e}")
        
        # Load latent projection weights (only once, not in loop)
        if not latent_projection_loaded:
            latent_projection_keys = {k.replace('latent_projection.', ''): v 
                                     for k, v in model_state_dict.items() 
                                     if 'latent_projection' in k}
            if latent_projection_keys:
                try:
                    latent_projection.load_state_dict(latent_projection_keys, strict=False)
                    latent_projection_loaded = True
                    logger.info(f"Loaded latent projection from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load latent projection: {e}")

        # Load z_to_embedding weights (only once, not in loop)
        if not z_to_embedding_loaded:
            z_to_embedding_state = {k.replace('z_to_embedding.', ''): v
                                   for k, v in model_state_dict.items()
                                   if 'z_to_embedding' in k}
            if z_to_embedding_state:
                try:
                    # Check if loaded weight has correct shape
                    if 'weight' in z_to_embedding_state:
                        loaded_weight = z_to_embedding_state['weight']
                        expected_shape = (embedding_dim, vpl_latent_dim)  # (out_features, in_features)
                        if loaded_weight.shape != expected_shape:
                            logger.warning(f"z_to_embedding weight shape mismatch: loaded {loaded_weight.shape}, expected {expected_shape}. "
                                         f"Will reinitialize with correct shape.")
                            # Don't load if shape doesn't match - use newly initialized one
                            z_to_embedding_loaded = False
                        else:
                            z_to_embedding.load_state_dict(z_to_embedding_state, strict=False)
                            z_to_embedding_loaded = True
                            logger.info(f"Loaded z_to_embedding from checkpoint: {loaded_weight.shape}")
                    else:
                        z_to_embedding.load_state_dict(z_to_embedding_state, strict=False)
                        z_to_embedding_loaded = True
                        logger.info(f"Loaded z_to_embedding from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load z_to_embedding: {e}. Will use newly initialized one.")
                    z_to_embedding_loaded = False
        
        if not (variational_encoder_loaded or feature_extractor_loaded or latent_projection_loaded or z_to_embedding_loaded):
            logger.warning("No VPL components found in checkpoint. Using randomly initialized components.")
        
        return variational_encoder, feature_extractor, latent_projection, z_to_embedding
        
    except Exception as e:
        logger.error(f"Failed to load VPL components from checkpoint: {e}")
        return None, None, None, None


def load_vpl_components_from_trainer(trainer):
    """
    Extract VPL components directly from a VPLRewardChoiceTrainer instance.
    
    Args:
        trainer: VPLRewardChoiceTrainer instance
        
    Returns:
        tuple: (variational_encoder, feature_extractor, latent_projection)
    """
    if not hasattr(trainer, 'variational_encoder'):
        logger.error("Trainer does not have variational_encoder. Not a VPL trainer?")
        return None, None, None
    
    variational_encoder = trainer.variational_encoder
    feature_extractor = trainer.feature_extractor if hasattr(trainer, 'feature_extractor') else None
    latent_projection = trainer.latent_projection if hasattr(trainer, 'latent_projection') else None
    
    return variational_encoder, feature_extractor, latent_projection


def load_client_average_z_from_checkpoint(checkpoint_path, device='cuda:0'):
    """
    Load client-specific average z values (z_mu) from a selector checkpoint.
    These are the average z values computed from training data for each client.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load tensors on
        
    Returns:
        dict: {client_id: z_mu_tensor} or None if not found
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Priority 1: Check for client_average_z_dict (saved from server during training)
        if 'client_average_z_dict' in ckpt:
            client_average_z_dict = ckpt['client_average_z_dict']
            if isinstance(client_average_z_dict, dict):
                # Convert to tensors if needed
                result = {}
                for client_id, z_mu in client_average_z_dict.items():
                    client_id_int = int(client_id) if not isinstance(client_id, int) else client_id
                    if isinstance(z_mu, torch.Tensor):
                        result[client_id_int] = z_mu.to(device)
                    elif isinstance(z_mu, list):
                        result[client_id_int] = torch.tensor(z_mu, dtype=torch.float32, device=device)
                    else:
                        result[client_id_int] = torch.tensor(z_mu, dtype=torch.float32, device=device)
                logger.info(f"Loaded client average z for {len(result)} clients from checkpoint (client_average_z_dict)")
                return result
        
        # Priority 2: Check if checkpoint has client z distributions (old format)
        if 'client_z_mus' in ckpt:
            # New format: dictionary of client_id -> z_mu
            client_z_mus = ckpt['client_z_mus']
            if isinstance(client_z_mus, dict):
                # Convert to tensors if needed
                result = {}
                for client_id, z_mu in client_z_mus.items():
                    if isinstance(z_mu, torch.Tensor):
                        result[client_id] = z_mu.to(device)
                    elif isinstance(z_mu, list):
                        result[client_id] = torch.tensor(z_mu, dtype=torch.float32, device=device)
                    else:
                        result[client_id] = torch.tensor(z_mu, dtype=torch.float32, device=device)
                logger.info(f"Loaded average z for {len(result)} clients from checkpoint (client_z_mus)")
                return result
        
        # Fallback: try to extract from model state dict (if stored there)
        model_state_dict = ckpt.get('model', ckpt)
        client_z_mus = {}
        
        # Look for keys like 'client_1_z_mu', 'client_2_z_mu', etc.
        for key, value in model_state_dict.items():
            if 'client_' in key and '_z_mu' in key:
                # Extract client ID from key (e.g., 'client_1_z_mu' -> 1)
                try:
                    parts = key.split('_')
                    if len(parts) >= 3 and parts[0] == 'client':
                        client_id = int(parts[1])
                        if isinstance(value, torch.Tensor):
                            client_z_mus[client_id] = value.to(device)
                        elif isinstance(value, list):
                            client_z_mus[client_id] = torch.tensor(value, dtype=torch.float32, device=device)
                        else:
                            client_z_mus[client_id] = torch.tensor(value, dtype=torch.float32, device=device)
                except (ValueError, IndexError):
                    continue
        
        if len(client_z_mus) > 0:
            logger.info(f"Loaded average z for {len(client_z_mus)} clients from checkpoint (fallback method)")
            return client_z_mus
        
        logger.warning("No client average z found in checkpoint. Will infer z from input during generation.")
        return None

    except Exception as e:
        logger.error(f"Failed to load client average z from checkpoint: {e}")
        return None


def fill_missing_client_z_with_category_mean(
        client_average_z_dict, client_num, num_cats=2):
    """
    Fill missing client IDs in client_average_z_dict with category-aware
    mean z. Each missing client gets the mean of available clients in the
    same preference category.

    HH-RLHF convention (num_cats=2): client IDs 1..N/2 are harmlessness
    (cat 0), (N/2)+1..N are helpfulness (cat 1). UltraFeedback uses 4
    categories with the same striped assignment used elsewhere in the
    codebase (see winrate_metrics.py and standalone_training.py).

    A missing client must NOT be filled from the overall mean because
    harmful and helpful prototypes are deliberately separated by the
    orthogonal loss; mixing them across the boundary defeats the
    preference-conditional generation that Stage 2 relies on.
    """
    if client_average_z_dict is None or len(client_average_z_dict) == 0:
        return client_average_z_dict

    clients_per_cat = client_num // num_cats
    cat_remainder = client_num % num_cats
    client_to_cat = {}
    cid = 1
    for cat_idx in range(num_cats):
        n = clients_per_cat + (1 if cat_idx < cat_remainder else 0)
        for _ in range(n):
            client_to_cat[cid] = cat_idx
            cid += 1

    cat_to_zs = {c: [] for c in range(num_cats)}
    for known_cid, z in client_average_z_dict.items():
        cat = client_to_cat.get(int(known_cid))
        if cat is not None:
            cat_to_zs[cat].append(z)

    cat_to_mean = {}
    for cat, zs in cat_to_zs.items():
        if len(zs) > 0:
            cat_to_mean[cat] = torch.stack(zs, dim=0).mean(dim=0)

    if len(cat_to_mean) == 0:
        return client_average_z_dict

    overall_mean = torch.stack(
        list(cat_to_mean.values()), dim=0).mean(dim=0)

    filled = dict(client_average_z_dict)
    missing = []
    for cid in range(1, client_num + 1):
        if cid in filled:
            continue
        cat = client_to_cat.get(cid)
        if cat is not None and cat in cat_to_mean:
            filled[cid] = cat_to_mean[cat].clone()
        else:
            filled[cid] = overall_mean.clone()
        missing.append(cid)

    if missing:
        logger.warning(
            f"client_average_z_dict was missing {len(missing)}/{client_num}"
            f" clients; filled with per-category mean z "
            f"(cats observed: {sorted(cat_to_mean.keys())}). "
            f"Sampling coverage during selector training was incomplete; "
            f"raise sample_client_num or train more rounds to fix at "
            f"the source.")
    return filled
