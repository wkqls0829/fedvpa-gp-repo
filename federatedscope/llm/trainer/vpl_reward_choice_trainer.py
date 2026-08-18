"""
Variational Preference Learning (VPL) for Binary Choice Trainer
Implements VPL with variational method for binary selector.
Based on: https://github.com/WEIRDLabUW/vpl
"""
import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.reward_choice_trainer import (
    RewardChoiceTrainer, cal_loss)
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.llm.model.variational_encoder import VariationalEncoder
import torch.nn as nn
from federatedscope.llm.dataset.llm_dataset import DefaultToken

import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


class VPLRewardChoiceTrainer(RewardChoiceTrainer):
    """
    Variational Preference Learning trainer for binary choice tasks.
    
    Extends RewardChoiceTrainer with variational inference over user-specific
    latents. The model learns to:
    1. Infer user-specific latent z from preference data
    2. Make choices conditioned on the latent z
    3. Optimize ELBO: E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        
        # VPL hyperparameters
        self.vpl_latent_dim = getattr(config.llm, 'vpl_latent_dim', 32)
        self.vpl_kl_weight = getattr(config.llm, 'vpl_kl_weight', 0.1)
        self.vpl_feature_method = getattr(config.llm, 'vpl_feature_method', 'choice_logits')

        # Deep latent projection: use MLP instead of linear 32->2
        self.vpl_deep_projection = getattr(
            config.llm, 'vpl_deep_projection', False)
        # Base logit dropout: randomly zero out base logits to force
        # the model to rely on z for prediction
        self.vpl_logit_dropout = getattr(
            config.llm, 'vpl_logit_dropout', 0.0)

        # Z-embedding conditioning: inject z into input embeddings
        # instead of adding latent_projection to output logits.
        # This makes Stage 1 architecture match Stage 2 RL.
        self.vpl_use_z_embedding = getattr(
            config.llm, 'vpl_use_z_embedding', False)
        # z conditioning mode: 'add' (default) or 'concat' (prefix tokens)
        self.vpl_z_conditioning_mode = getattr(
            config.llm, 'vpl_z_conditioning_mode', 'add')
        # Number of virtual prefix tokens for concat mode
        self.vpl_z_num_prefix_tokens = getattr(
            config.llm, 'vpl_z_num_prefix_tokens', 4)

        # Check if using feature difference (embedding difference)
        self.vpl_use_feature_difference = getattr(config.llm, 'vpl_use_feature_difference', False)
        self.vpl_use_llm_feature_extractor = getattr(config.llm, 'vpl_use_llm_feature_extractor', True)
        self.vpl_use_difference_only = getattr(config.llm, 'vpl_use_difference_only', False)  # Use only difference embedding (no chosen/rejected)

        # Tokenize response region markers for embedding difference
        # extraction.  These are used to locate Response A / Response B
        # regions inside input_ids so we can extract the *actual*
        # response hidden states rather than just the answer-token
        # hidden state.
        if self.vpl_use_feature_difference:
            _resp_a_marker = self.tokenizer(
                "### RESPONSE A:",
                add_special_tokens=False)['input_ids']
            _resp_b_marker = self.tokenizer(
                "### RESPONSE B:",
                add_special_tokens=False)['input_ids']
            _choice_marker = self.tokenizer(
                "### YOUR CHOICE:",
                add_special_tokens=False)['input_ids']
            self._resp_a_marker = _resp_a_marker
            self._resp_b_marker = _resp_b_marker
            self._choice_marker = _choice_marker
            logger.info(
                f"Response markers: A={_resp_a_marker}, "
                f"B={_resp_b_marker}, "
                f"CHOICE={_choice_marker}")

        # VPL-GP hyperparameters (integrated into main trainer)
        self.vpl_use_gp_prior = getattr(config.llm, 'vpl_use_gp_prior', False)
        self.vpl_gp_temperature = getattr(config.llm, 'vpl_gp_temperature', 1.0)
        self.num_clients = getattr(config.federate, 'client_num', 10)

        # Temperature annealing for Gumbel-Softmax (VMTL-style)
        self.vpl_gp_tau_anneal = getattr(config.llm, 'vpl_gp_tau_anneal', True)
        self.vpl_gp_tau_start = getattr(config.llm, 'vpl_gp_tau_start', self.vpl_gp_temperature)
        self.vpl_gp_tau_end = getattr(config.llm, 'vpl_gp_tau_end', 0.1)
        self.total_round_num = getattr(config.federate, 'total_round_num', 30)
        
        # Orthogonal loss hyperparameters (CLOP-based)
        self.vpl_orthogonal_weight = getattr(config.llm, 'vpl_orthogonal_weight', 0.0)
        self.vpl_orthogonal_orthonorm_weight = getattr(config.llm, 'vpl_orthogonal_orthonorm_weight', 0.1)
        self.vpl_use_manual_orthogonal_labels = getattr(config.llm, 'vpl_use_manual_orthogonal_labels', False)
        
        # Initialize orthonormal prototypes if orthogonal loss is enabled
        # Prototypes are LEARNABLE nn.Parameters (CLOP standard)
        # Orthonormality is maintained via the orthonorm constraint loss
        if self.vpl_orthogonal_weight > 0.0:
            num_prototypes = getattr(config.llm, 'vpl_num_prototypes', 2)
            # Get prototype scale (distance from origin)
            prototype_scale = getattr(config.llm, 'vpl_prototype_scale', 5.0)
            self.prototype_scale = prototype_scale

            # Create orthonormal basis via QR decomposition, then scale
            if num_prototypes <= self.vpl_latent_dim:
                # Random matrix -> QR -> orthonormal rows, padded to latent_dim
                rand_mat = torch.randn(
                    num_prototypes, self.vpl_latent_dim,
                    device=device
                )
                q, _ = torch.linalg.qr(rand_mat.T)
                # q is (latent_dim, num_prototypes), transpose
                prototypes = q.T[:num_prototypes] * prototype_scale
            else:
                rand_mat = torch.randn(
                    self.vpl_latent_dim, self.vpl_latent_dim,
                    device=device
                )
                q, _ = torch.linalg.qr(rand_mat)
                prototypes = q.T[:num_prototypes, :self.vpl_latent_dim] \
                    * prototype_scale

            # Learnable prototypes (updated via gradient descent)
            self.orthogonal_prototypes = nn.Parameter(prototypes)
            self.orthogonal_label = None  # Will be set by server
            logger.info(
                f"Initialized {num_prototypes} LEARNABLE "
                f"orthonormal prototypes for CLOP loss "
                f"(scale={prototype_scale}, "
                f"shape={self.orthogonal_prototypes.shape})"
            )
        else:
            self.orthogonal_prototypes = None
            self.orthogonal_label = None
        
        # Get model dimensions
        # Try multiple methods to get embedding dimension
        embedding_dim = None
        try:
            embedding_dim = self.model.get_input_embeddings().embedding_dim
            logger.info(f"Got embedding_dim={embedding_dim} from get_input_embeddings()")
        except Exception as e:
            logger.debug(f"Could not get embedding_dim from get_input_embeddings(): {e}")
            try:
                # Most HuggingFace models (including LLaMA) expose hidden_size on config
                embedding_dim = getattr(self.model, "config", None)
                if embedding_dim is not None:
                    embedding_dim = embedding_dim.hidden_size
                    logger.info(f"Got embedding_dim={embedding_dim} from model.config.hidden_size")
            except Exception as e2:
                logger.debug(f"Could not get embedding_dim from config.hidden_size: {e2}")
                embedding_dim = None

        # If direct queries failed, fall back based on model type
        if embedding_dim is None:
            model_type = getattr(config.model, 'type', '')
            mt_lower = model_type.lower()
            if 'llama' in mt_lower:
                # LLaMA hidden size (e.g., LLaMA-2 7B) is typically 4096
                embedding_dim = 4096
            elif 'gemma' in mt_lower:
                embedding_dim = 2048  # Gemma-2B
            elif 'qwen' in mt_lower:
                # Qwen2 models: default to 0.5B (896) for main table experiments
                embedding_dim = 896  # Qwen2-0.5B (default for main table)
            else:
                embedding_dim = 2048  # Generic fallback
            logger.info(f"Using inferred embedding_dim={embedding_dim} for model type: {model_type}")
        
        if embedding_dim is None:
            embedding_dim = 2048  # Final fallback
            logger.warning(f"Could not determine embedding_dim, using default: {embedding_dim}")
        
        # Initialize feature extractor
        # Strategy: Siamese-style — apply MLP to each response separately,
        # then take difference.  This lets the MLP learn non-linear
        # transformations that amplify preference-relevant features
        # *before* subtraction.
        if self.vpl_use_llm_feature_extractor and self.vpl_use_feature_difference:
            # Siamese MLP: input is a single response embedding (896-dim)
            # Applied independently to chosen & rejected, difference taken
            # after.  Output dim is 128 (same as before).
            self.feature_extractor = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            ).to(device)
            feature_extractor_output_dim = 128
            self._siamese_feature_extractor = True
            logger.info(
                "Using Siamese feature extractor with last-token "
                "pooling (MLP applied per-response, difference "
                "taken after)")
        else:
            self._siamese_feature_extractor = False
            # Use MLP feature extractor (logits-based path)
            if self.vpl_use_feature_difference:
                raw_feature_dim = embedding_dim
            elif self.vpl_feature_method == 'choice_logits':
                raw_feature_dim = len(self.choices) * 2
            else:
                raw_feature_dim = 2

            # Deep feature extraction network
            self.feature_extractor = nn.Sequential(
                nn.Linear(raw_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            ).to(device)
            feature_extractor_output_dim = 128
        
        # Store output dimension
        self.feature_extractor_output_dim = feature_extractor_output_dim
        
        # Initialize variational encoder with deeper layers
        # Use VariationalEncoderGP if GP prior is enabled, otherwise use standard VariationalEncoder
        # Get max_logvar to limit variance (sigma) - smaller values = tighter distribution
        # max_logvar=0.0 means sigma <= 1.0, max_logvar=-2.0 means sigma <= 0.368
        vpl_max_logvar = getattr(config.llm, 'vpl_max_logvar', -2.0)  # Default: -2.0 for tighter distribution
        if self.vpl_use_gp_prior:
            from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP
            vpl_gp_fixed_uniform = getattr(
                config.llm, 'vpl_gp_fixed_uniform_weights', False)
            self.variational_encoder = VariationalEncoderGP(
                input_dim=self.feature_extractor_output_dim,
                latent_dim=self.vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                temperature=self.vpl_gp_temperature,
                num_clients=self.num_clients,
                max_logvar=vpl_max_logvar,
                tau_anneal=self.vpl_gp_tau_anneal,
                tau_start=self.vpl_gp_tau_start,
                tau_end=self.vpl_gp_tau_end,
                fixed_uniform_weights=vpl_gp_fixed_uniform,
            ).to(device)
        else:
            self.variational_encoder = VariationalEncoder(
                input_dim=self.feature_extractor_output_dim,
                latent_dim=self.vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                max_logvar=vpl_max_logvar
            ).to(device)
        
        # Latent conditioning: project latent z to logit adjustments
        if self.vpl_deep_projection:
            # Deep MLP: gives z more expressive power over predictions
            self.latent_projection = nn.Sequential(
                nn.Linear(self.vpl_latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(self.choices))
            ).to(device)
            logger.info("Using deep latent projection (32->64->32->2)")
        else:
            self.latent_projection = nn.Linear(
                self.vpl_latent_dim,
                len(self.choices)
            ).to(device)

        # z_to_embedding: project z into model embedding space for
        # input-embedding injection (only when vpl_use_z_embedding=True)
        self.z_to_embedding = None
        if self.vpl_use_z_embedding:
            embedding_dim = model.get_input_embeddings().embedding_dim
            if self.vpl_z_conditioning_mode == 'concat':
                # Prefix-token mode: project z to k virtual tokens
                k = self.vpl_z_num_prefix_tokens
                self.z_to_embedding = nn.Linear(
                    self.vpl_latent_dim, k * embedding_dim).to(device)
                nn.init.normal_(
                    self.z_to_embedding.weight, std=0.01)
                nn.init.zeros_(self.z_to_embedding.bias)
                logger.info(
                    f"Initialized z_to_embedding (concat): "
                    f"{self.vpl_latent_dim} -> {k}x{embedding_dim} "
                    f"= {k * embedding_dim} "
                    f"({k} prefix tokens)")
            else:
                # Additive mode: project z to embedding space
                self.z_to_embedding = nn.Linear(
                    self.vpl_latent_dim, embedding_dim).to(device)
                nn.init.normal_(
                    self.z_to_embedding.weight, std=0.001)
                nn.init.zeros_(self.z_to_embedding.bias)
                logger.info(
                    f"Initialized z_to_embedding (add): "
                    f"{self.vpl_latent_dim} -> {embedding_dim} "
                    f"(near-zero init)")

        # Cached loss function (avoid re-instantiation per batch)
        self._ce_loss_fn = torch.nn.CrossEntropyLoss()

        # Cache ortho target (avoid recreating torch.eye per batch)
        if self.vpl_orthogonal_weight > 0.0:
            scale_sq = self.prototype_scale ** 2
            num_p = self.orthogonal_prototypes.shape[0]
            self._ortho_target = (
                torch.eye(num_p, device=device) * scale_sq
            )

        # Cache VPL param list for grad clipping
        self._vpl_params_for_clip = []
        for comp_name in ('feature_extractor',
                          'variational_encoder',
                          'latent_projection'):
            comp = getattr(self, comp_name, None)
            if comp is not None:
                self._vpl_params_for_clip.extend(comp.parameters())

        # Initialize GP prior related attributes (only if GP prior is enabled)
        if self.vpl_use_gp_prior:
            # Z history for visualization
            self.z_history = []
            self.z_mu_history = []
            self.z_logvar_history = []
            
            # Client z distribution (average over batches)
            self.client_z_mu = None
            self.client_z_logvar = None
        
        logger.info(f'VPLRewardChoiceTrainer initialized with latent_dim={self.vpl_latent_dim}, '
                   f'kl_weight={self.vpl_kl_weight}, '
                   f'use_feature_difference={self.vpl_use_feature_difference}, '
                   f'use_llm_feature_extractor={self.vpl_use_llm_feature_extractor}, '
                   f'use_gp_prior={self.vpl_use_gp_prior}')

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        ctx.vpl_kl_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.vpl_reconstruction_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        if self.vpl_orthogonal_weight > 0.0:
            ctx.vpl_orthogonal_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)

        # Reset per-mode z/μ/logvar collectors so train and eval
        # samples never get mixed (the previous code only cleared
        # them at fit_end, after both modes had already appended).
        self.z_history = []
        self._posterior_mu_history = []
        self._posterior_logvar_history = []
        
        # Freeze base model (LLM) for binary selector training
        # This prevents base model from being updated during VPL training
        # Only VPL components (feature_extractor, variational_encoder, latent_projection) will be trained
        freeze_base_model = getattr(ctx.cfg.llm, 'vpl_freeze_base_model', True)  # Default: True
        if freeze_base_model and ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Use ctx.model instead of self.model since model may not be set yet
            model = getattr(ctx, 'model', None) or getattr(self, 'model', None)
            if model is not None:
                # Freeze all parameters of the base model
                for param in model.parameters():
                    param.requires_grad = False
                logger.info("Frozen base model (LLM) parameters for binary selector training. "
                           "Only VPL components will be trained.")
            else:
                logger.warning("Model not available yet for freezing. Will freeze later if needed.")
        
        # Create separate optimizer for VPL components (feature_extractor + variational_encoder + latent_projection)
        # This allows VPL components to learn faster while LLM is frozen or learns slowly
        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            vpl_lr_multiplier = getattr(ctx.cfg.llm, 'vpl_lr_multiplier', 10.0)  # Default: 10x faster learning
            
            # Collect VPL component parameters
            vpl_params = []
            if hasattr(self, 'feature_extractor'):
                vpl_params.extend(self.feature_extractor.parameters())
            if hasattr(self, 'variational_encoder'):
                vpl_params.extend(self.variational_encoder.parameters())
            if hasattr(self, 'latent_projection'):
                vpl_params.extend(self.latent_projection.parameters())
            if hasattr(self, 'orthogonal_prototypes') \
                    and self.orthogonal_prototypes is not None \
                    and isinstance(self.orthogonal_prototypes,
                                   nn.Parameter):
                vpl_params.append(self.orthogonal_prototypes)

            if len(vpl_params) > 0:
                # Get base learning rate from config
                base_lr = ctx.cfg[ctx.cur_mode].optimizer.get('lr', 1e-5)
                vpl_lr = base_lr * vpl_lr_multiplier

                # Build param groups: separate LR for z_to_embedding
                from torch.optim import AdamW
                optim_betas = ctx.cfg[ctx.cur_mode].optimizer.get(
                    'betas', (0.9, 0.95))
                optim_wd = ctx.cfg[ctx.cur_mode].optimizer.get(
                    'weight_decay', 0.0)
                param_groups = [
                    {'params': vpl_params, 'lr': vpl_lr}]

                if (hasattr(self, 'z_to_embedding')
                        and self.z_to_embedding is not None):
                    z_emb_lr = getattr(
                        ctx.cfg.llm, 'vpl_z_embedding_lr', None)
                    if z_emb_lr is None:
                        z_emb_lr = vpl_lr
                    param_groups.append({
                        'params': list(
                            self.z_to_embedding.parameters()),
                        'lr': z_emb_lr})
                    self._vpl_params_for_clip.extend(
                        self.z_to_embedding.parameters())
                    logger.info(
                        f"z_to_embedding added to VPL optimizer "
                        f"(lr={z_emb_lr:.2e})")

                ctx.vpl_optimizer = AdamW(
                    param_groups,
                    betas=optim_betas,
                    weight_decay=optim_wd)
                logger.info(f"Created separate VPL optimizer with lr={vpl_lr:.2e} (base_lr={base_lr:.2e} * {vpl_lr_multiplier}x)")
            else:
                ctx.vpl_optimizer = None
                logger.warning("No VPL components found for separate optimizer")

    def _extract_preference_features(self, logits, labels, choices,
                                      hidden_states=None, input_ids=None):
        """
        Extract features from preference data for variational encoder.

        If vpl_use_feature_difference is True and Siamese mode is
        active, applies the feature extractor MLP to each response
        embedding independently (last-token pooling), then returns
        the difference.  The caller should **skip** the separate
        ``feature_extractor()`` call in this case (indicated by
        ``self._siamese_feature_extractor``).

        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            labels: Labels (batch, seq_len)
            choices: Choice token indices [A_token, B_token]
            hidden_states: Model hidden states (batch, seq_len,
                hidden_dim) if available
            input_ids: Input token ids (batch, seq_len) — needed to
                locate response A/B regions

        Returns:
            features: Extracted features (batch, feature_dim)
            - Siamese path: (batch, 128) — already through MLP
            - Logits path: (batch, len(choices) * 2) = (batch, 4)
        """
        if self.vpl_use_feature_difference and hidden_states is not None:
            return self._extract_embedding_difference(
                hidden_states, labels, choices, input_ids=input_ids)
        else:
            # Fallback to original logits-based extraction
            return self._extract_logits_features(logits, labels, choices)
    
    @staticmethod
    def _find_subsequence(seq, subseq):
        """Return the start index of *subseq* in *seq*, or -1."""
        slen = len(subseq)
        for i in range(len(seq) - slen + 1):
            if seq[i:i + slen] == subseq:
                return i
        return -1

    def _extract_embedding_difference(self, hidden_states, labels,
                                      choices, input_ids=None):
        """
        Extract preference features using **last-token pooling** and
        **Siamese feature extraction**.

        Pipeline (per sample):
          1. Locate Response A / B regions via marker tokens in
             input_ids.
          2. Pool each region using the **last token** of the region
             (for causal LLMs, the last token has attended to the
             full response and carries the richest representation).
          3. Apply the shared feature_extractor MLP to each pooled
             embedding independently (Siamese style).
          4. Return  MLP(chosen) − MLP(rejected).

        When ``_siamese_feature_extractor`` is True the returned
        features have already been through the MLP (128-dim) so the
        caller should feed them directly to the variational encoder
        **without** a second ``feature_extractor()`` call.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            labels: (batch, seq_len)
            choices: [A_token, B_token]
            input_ids: (batch, seq_len) — token ids of the full input

        Returns:
            features: (batch, 128) if Siamese, else
                (batch, hidden_dim)
        """
        import torch
        batch_size, seq_len, hidden_dim = hidden_states.shape
        A_token, B_token = choices[0], choices[1]

        # --- Locate response regions via input_ids markers ----------
        has_markers = (input_ids is not None
                       and hasattr(self, '_resp_a_marker'))

        chosen_embs = []
        rejected_embs = []

        for b in range(batch_size):
            resp_a_start = resp_a_end = -1
            resp_b_start = resp_b_end = -1

            if has_markers:
                ids = input_ids[b].tolist()
                pa = self._find_subsequence(
                    ids, self._resp_a_marker)
                pb = self._find_subsequence(
                    ids, self._resp_b_marker)
                pc = self._find_subsequence(
                    ids, self._choice_marker)

                if pa >= 0 and pb >= 0:
                    resp_a_start = pa + len(self._resp_a_marker)
                    resp_a_end = pb
                if pb >= 0:
                    resp_b_start = pb + len(self._resp_b_marker)
                    resp_b_end = pc if pc >= 0 else seq_len

            # --- Last-token pooling --------------------------------
            # For causal LLMs the last token of a region has attended
            # to every preceding token, giving the richest per-
            # response representation.
            if resp_a_start >= 0 and resp_a_end > resp_a_start:
                a_emb = hidden_states[b, resp_a_end - 1, :]
            else:
                a_emb = hidden_states[b, -1, :]

            if resp_b_start >= 0 and resp_b_end > resp_b_start:
                b_emb = hidden_states[b, resp_b_end - 1, :]
            else:
                b_emb = hidden_states[b, -1, :]

            # Determine chosen/rejected from the answer label
            label_tokens = labels[b]
            a_is_answer = (label_tokens == A_token).any()

            if a_is_answer:
                chosen_embs.append(a_emb)
                rejected_embs.append(b_emb)
            else:
                chosen_embs.append(b_emb)
                rejected_embs.append(a_emb)

        chosen_emb = torch.stack(chosen_embs)      # (B, H)
        rejected_emb = torch.stack(rejected_embs)   # (B, H)

        if chosen_emb.dtype != torch.float32:
            chosen_emb = chosen_emb.float()
        if rejected_emb.dtype != torch.float32:
            rejected_emb = rejected_emb.float()

        # --- Siamese feature extraction ----------------------------
        if getattr(self, '_siamese_feature_extractor', False):
            chosen_feat = self.feature_extractor(chosen_emb)
            rejected_feat = self.feature_extractor(rejected_emb)
            features = chosen_feat - rejected_feat     # (B, 128)
        else:
            # Legacy path: raw difference
            feature_diff = chosen_emb - rejected_emb
            if self.vpl_use_llm_feature_extractor:
                if self.vpl_use_difference_only:
                    features = feature_diff
                else:
                    features = torch.cat(
                        [chosen_emb, rejected_emb, feature_diff],
                        dim=-1)
            else:
                features = feature_diff

        return features
    
    def _extract_logits_features(self, logits, labels, choices):
        """
        Extract features from logits (original method).
        
        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            labels: Labels (batch, seq_len)
            choices: Choice token indices
            
        Returns:
            features: Extracted features (batch, len(choices) * 2)
        """
        # Get logits at choice positions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size = logits.shape[0]
        
        # Extract logits for each choice token
        choice_logits_per_token = shift_logits[..., choices]  # (batch, seq_len, len(choices))
        
        # Find positions where each choice appears
        features_list = []
        for choice_idx, choice_token in enumerate(choices):
            choice_positions = (shift_labels == choice_token)  # (batch, seq_len)
            
            if choice_positions.any():
                batch_features = []
                for b in range(batch_size):
                    sample_positions = choice_positions[b]
                    if sample_positions.any():
                        sample_choice_logits = choice_logits_per_token[b, sample_positions, choice_idx]
                        avg_logit = sample_choice_logits.mean()
                    else:
                        avg_logit = choice_logits_per_token[b, :, choice_idx].mean()
                    batch_features.append(avg_logit)
                features_list.append(torch.stack(batch_features))
            else:
                features_list.append(choice_logits_per_token[:, :, choice_idx].mean(dim=1))
        
        features = torch.stack(features_list, dim=1)  # (batch, len(choices))
        features = torch.cat([features, features], dim=1)  # (batch, len(choices) * 2)
        
        if features.dtype != torch.float32:
            features = features.float()
        
        return features

    def _hook_on_batch_forward(self, ctx):
        """
        Forward pass with variational inference.
        Supports both standard VPL and VPL-GP (mixture prior).
        """
        is_eval = ctx.cur_mode not in (MODE.TRAIN, MODE.FINETUNE)
        if is_eval:
            with torch.no_grad():
                self._hook_on_batch_forward_impl(ctx)
        else:
            self._hook_on_batch_forward_impl(ctx)

    def _hook_on_batch_forward_impl(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        # ---- Step 1: extract z ----
        # Three modes for z extraction:
        # - 'input_emb': use raw input embeddings (cheap, no forward)
        # - 'hidden_state': use last hidden states (requires forward)
        # - legacy (no z_embedding): uses hidden states (original)
        z_source = getattr(self, '_z_source', None)
        if z_source is None:
            # Determine z source based on config
            if self.vpl_use_z_embedding:
                z_source = getattr(
                    ctx.cfg.llm, 'vpl_z_source', 'input_emb')
            else:
                z_source = 'hidden_state'  # legacy always uses this
            self._z_source = z_source

        input_embs = ctx.model.get_input_embeddings()(input_ids)

        if z_source == 'input_emb':
            # Use raw input embeddings (no forward pass needed)
            if self.vpl_use_feature_difference:
                input_embs_float = input_embs.detach().float()
                preference_features = \
                    self._extract_preference_features(
                        None, labels, self.choices,
                        hidden_states=input_embs_float,
                        input_ids=input_ids)
            else:
                outputs_tmp = ctx.model(
                    input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask)
                preference_features = \
                    self._extract_preference_features(
                        outputs_tmp.logits, labels, self.choices,
                        hidden_states=None, input_ids=input_ids)
        else:
            # 'hidden_state': forward pass to get last hidden states
            with torch.no_grad():
                outputs_pass1 = ctx.model(
                    input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask,
                    output_hidden_states=True)
            hidden_states = None
            if hasattr(outputs_pass1, 'hidden_states') \
                    and outputs_pass1.hidden_states is not None:
                hidden_states = outputs_pass1.hidden_states[-1]
            elif hasattr(outputs_pass1, 'last_hidden_state'):
                hidden_states = outputs_pass1.last_hidden_state

            if hidden_states is not None:
                preference_features = \
                    self._extract_preference_features(
                        outputs_pass1.logits, labels, self.choices,
                        hidden_states=hidden_states.detach().float(),
                        input_ids=input_ids)
            else:
                # Fallback to logits
                preference_features = \
                    self._extract_preference_features(
                        outputs_pass1.logits, labels, self.choices,
                        hidden_states=None, input_ids=input_ids)

        # Feature extractor
        if getattr(self, '_siamese_feature_extractor', False):
            extracted_features = preference_features
        else:
            extracted_features = self.feature_extractor(
                preference_features)

        # Variational inference: encode to latent z
        z, mu, logvar = self.variational_encoder(extracted_features)

        # Store z in ctx (for GP prior collection if enabled)
        ctx.vpl_z = CtxVar(z.detach(), LIFECYCLE.BATCH)
        ctx.vpl_mu = CtxVar(mu.detach(), LIFECYCLE.BATCH)
        ctx.vpl_logvar = CtxVar(logvar.detach(), LIFECYCLE.BATCH)

        # Collect z values for t-SNE visualization
        if not hasattr(self, 'z_history'):
            self.z_history = []
        if len(self.z_history) < 100:
            self.z_history.append(z.detach().cpu())

        # Collect posterior parameters (mu, logvar) for proper client
        # distribution aggregation.  The mixture prior in Eq. 6 is a
        # mixture of N(μ_j, σ_j²); we should aggregate the POSTERIOR
        # parameters, not the empirical mean/var of sampled z's.
        # Stored only in eval mode so the prior reflects the converged
        # encoder for this round (no train-time sampling noise).
        if ctx.cur_mode not in (MODE.TRAIN, MODE.FINETUNE):
            if not hasattr(self, '_posterior_mu_history'):
                self._posterior_mu_history = []
                self._posterior_logvar_history = []
            if len(self._posterior_mu_history) < 100:
                self._posterior_mu_history.append(
                    mu.detach().cpu())
                self._posterior_logvar_history.append(
                    logvar.detach().cpu())

        # KL divergence
        kl_loss = self.variational_encoder.kl_divergence(mu, logvar)

        # ---- Step 2: z-conditioned forward pass ----
        if self.vpl_use_z_embedding and self.z_to_embedding is not None:
            model_dtype = input_embs.dtype
            # Keep z_to_embedding in its own (typically fp32) dtype
            # — mutating its dtype mid-forward corrupts AdamW state
            # and loses precision on a small but sensitive Linear.
            # Cast z into the Linear's dtype on the input side, then
            # cast the projected embedding back to model_dtype before
            # concatenation/addition.
            zte_dtype = self.z_to_embedding.weight.dtype

            if self.vpl_z_conditioning_mode == 'concat':
                # Prefix-token mode: project z to k virtual tokens
                # and prepend to the input sequence
                k = self.vpl_z_num_prefix_tokens
                emb_dim = input_embs.shape[-1]
                z_proj = self.z_to_embedding(
                    z.to(dtype=zte_dtype))
                z_tokens = z_proj.view(
                    -1, k, emb_dim).to(
                    dtype=model_dtype)  # (batch, k, emb_dim)
                conditioned_embs = torch.cat(
                    [z_tokens, input_embs], dim=1)
                # Extend attention_mask and labels for prefix tokens
                batch_size = input_embs.shape[0]
                prefix_mask = torch.ones(
                    batch_size, k,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device)
                attention_mask = torch.cat(
                    [prefix_mask, attention_mask], dim=1)
                prefix_labels = torch.full(
                    (batch_size, k),
                    DefaultToken.IGNORE_INDEX.value,
                    dtype=labels.dtype,
                    device=labels.device)
                labels = torch.cat(
                    [prefix_labels, labels], dim=1)
            else:
                # Additive mode: add z embedding to all positions
                z_emb = self.z_to_embedding(
                    z.to(dtype=zte_dtype))
                z_emb = z_emb.unsqueeze(1).to(
                    dtype=model_dtype)  # (batch, 1, emb_dim)
                conditioned_embs = input_embs + z_emb

            conditioned_embs = conditioned_embs.to(dtype=model_dtype)

            # Adapter dropout: randomly disable LoRA so the model
            # must rely on z_to_embedding for prediction.
            is_training = ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE)
            adapter_dropout = getattr(
                ctx.cfg.llm, 'vpl_adapter_dropout', 0.0)
            disable_adapter = (
                is_training and adapter_dropout > 0.0
                and torch.rand(1).item() < adapter_dropout)

            outputs = ctx.model(
                inputs_embeds=conditioned_embs,
                labels=labels,
                attention_mask=attention_mask,
                disable_adapter=disable_adapter)
            logits = outputs.logits

            # Extract choice logits for loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            new_labels = torch.full_like(
                shift_labels, DefaultToken.IGNORE_INDEX.value)
            for idx, choice in enumerate(self.choices):
                new_labels[shift_labels == choice] = idx
            new_logits = shift_logits[..., self.choices]
            conditioned_logits = new_logits

        else:
            # Legacy path: base logits + latent_projection additive
            output_hidden_states = (
                self.vpl_use_feature_difference
                and not self.vpl_use_z_embedding)
            outputs = ctx.model(
                input_ids=input_ids, labels=labels,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states)
            logits = outputs.logits

            latent_adjustment = self.latent_projection(z)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            new_labels = torch.full_like(
                shift_labels, DefaultToken.IGNORE_INDEX.value)
            for idx, choice in enumerate(self.choices):
                new_labels[shift_labels == choice] = idx
            new_logits = shift_logits[..., self.choices]

            # Base logit dropout
            is_training = ctx.cur_mode in (MODE.TRAIN, MODE.FINETUNE)
            if is_training and self.vpl_logit_dropout > 0.0:
                dropout_mask = torch.bernoulli(
                    torch.full(
                        (new_logits.size(0), 1, 1),
                        1.0 - self.vpl_logit_dropout,
                        device=new_logits.device))
                new_logits = new_logits * dropout_mask

            batch_size, seq_len, num_choices = new_logits.shape
            latent_adj_exp = latent_adjustment.unsqueeze(1).expand(
                -1, seq_len, -1)
            conditioned_logits = new_logits + latent_adj_exp

        # Compute reconstruction loss
        num_choices = conditioned_logits.shape[-1]
        reconstruction_loss = self._ce_loss_fn(
            conditioned_logits.view(-1, num_choices),
            new_labels.view(-1))
        
        # VPL loss = reconstruction loss + KL divergence
        vpl_loss = reconstruction_loss + self.vpl_kl_weight * kl_loss
        
        # Add orthogonal loss (CLOP-based) if enabled
        orthogonal_loss_val = torch.tensor(0.0, device=z.device)
        if self.vpl_orthogonal_weight > 0.0 and self.orthogonal_prototypes is not None:
            orthogonal_loss_val, pull_loss, orthonorm_loss = self._compute_clop_orthogonal_loss(z)
            vpl_loss = vpl_loss + orthogonal_loss_val
        
        # Store for monitoring (use detach() to avoid GPU sync from .item())
        ctx.vpl_kl_loss = CtxVar(kl_loss.detach(), LIFECYCLE.BATCH)
        ctx.vpl_reconstruction_loss = CtxVar(
            reconstruction_loss.detach(), LIFECYCLE.BATCH)
        if self.vpl_orthogonal_weight > 0.0:
            ctx.vpl_orthogonal_loss = CtxVar(
                orthogonal_loss_val.detach(), LIFECYCLE.BATCH)
        
        # Clean up intermediate tensors to save memory (keep only what's needed for backward)
        # Note: Don't delete tensors that are part of the computation graph
        if ctx.cur_mode != MODE.TRAIN:
            # In eval mode, drop refs to large intermediates so the
            # autograd graph can be freed.  `del [a, b]` only deletes
            # the list — we need to delete each name individually.
            del preference_features
            del z
            del mu
            del logvar
            if 'latent_adjustment' in dir():
                del latent_adjustment

        if torch.isnan(vpl_loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # Compute predictions for evaluation
        new_labels_flat = new_labels.view(-1)
        conditioned_logits_flat = conditioned_logits.view(-1, len(self.choices))
        conditioned_logits_flat = conditioned_logits_flat[(
            new_labels_flat != DefaultToken.IGNORE_INDEX.value), :]
        new_labels_flat = new_labels_flat[(new_labels_flat !=
                                          DefaultToken.IGNORE_INDEX.value)]
        _, predicted = conditioned_logits_flat.max(1)

        ctx.y_true = CtxVar(new_labels_flat, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(conditioned_logits_flat, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(vpl_loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        """
        Override backward pass to use separate VPL optimizer.
        This allows VPL components (feature_extractor, variational_encoder, latent_projection)
        to learn faster while LLM is frozen (hidden_states.detach()).
        """
        if ctx.skip_this_batch:
            return

        # Use separate VPL optimizer if available
        # This allows VPL components (feature_extractor, variational_encoder, latent_projection)
        # to learn faster while LLM learns slowly or is frozen
        use_vpl_optimizer = hasattr(ctx, 'vpl_optimizer') and ctx.vpl_optimizer is not None
        
        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(ctx.loss_task)
            if use_vpl_optimizer:
                # Update VPL components with separate optimizer
                ctx.vpl_optimizer.step()
                ctx.vpl_optimizer.zero_grad()
            else:
                # Update all parameters with main optimizer
                ctx.optimizer.step()
                ctx.optimizer.zero_grad()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()

            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if use_vpl_optimizer:
                    # Update VPL components with separate optimizer (faster learning)
                    if ctx.grad_clip > 0 and self._vpl_params_for_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self._vpl_params_for_clip, ctx.grad_clip)
                    ctx.vpl_optimizer.step()
                    ctx.vpl_optimizer.zero_grad()
                    
                    # LLM parameters are not updated (hidden_states was detached)
                    # If you want to update LLM as well, you can add:
                    # ctx.optimizer.step()
                    # ctx.optimizer.zero_grad()
                else:
                    # Update all parameters with main optimizer
                    if ctx.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                       ctx.grad_clip)
                    ctx.optimizer.step()
                    ctx.optimizer.zero_grad()
                
                if ctx.scheduler is not None:
                    ctx.scheduler.step()

        # Free training data from GPU (del instead of .cpu() which
        # was a no-op bug — return values were discarded)
        del ctx.data_batch
    
    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        # Update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        
        # Update VPL-specific statistics
        ctx.vpl_kl_loss_total += ctx.vpl_kl_loss * ctx.batch_size
        ctx.vpl_reconstruction_loss_total += ctx.vpl_reconstruction_loss * ctx.batch_size
        if self.vpl_orthogonal_weight > 0.0 and hasattr(ctx, 'vpl_orthogonal_loss'):
            ctx.vpl_orthogonal_loss_total += ctx.vpl_orthogonal_loss * ctx.batch_size
        
        # NOTE: Fixed prototypes don't need orthonormal constraint updates
        # They are already orthonormal and fixed
        
        # Cache label for evaluate
        ctx.ys_true.append(ctx.y_true)
        ctx.ys_pred.append(ctx.y_pred)
        
        # NOTE: gc.collect() + empty_cache() removed here — calling
        # every 5 batches was adding ~1-2s overhead per round.
        # If OOM occurs, reduce batch_size or enable gradient
        # checkpointing instead.

    def _hook_on_fit_end(self, ctx):
        # Handle case where all batches were skipped (e.g. NaN loss): avoid empty concatenate
        if not ctx.ys_true:
            device = getattr(ctx, 'device', None) or getattr(self, 'device', None) or 'cpu'
            ctx.ys_true = CtxVar(
                torch.tensor([], dtype=torch.long, device=device),
                LIFECYCLE.ROUTINE,
            )
            ctx.ys_pred = CtxVar(
                torch.tensor([], dtype=torch.long, device=device),
                LIFECYCLE.ROUTINE,
            )
            logger.warning(
                "Evaluation had no valid batches (all skipped e.g. due to NaN loss). "
                "Reporting empty metrics for this client."
            )
        else:
            ctx.ys_true = CtxVar(torch.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
            ctx.ys_pred = CtxVar(torch.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        # Set tokenizer in ctx for evaluation metrics that need it (e.g., reward model evaluation)
        if not hasattr(ctx, 'tokenizer') and hasattr(self, 'tokenizer'):
            ctx.tokenizer = self.tokenizer
        results = ctx.monitor.eval(ctx)
        
        # Add VPL-specific metrics
        if hasattr(ctx, 'vpl_kl_loss_total') and ctx.num_samples > 0:
            kl_avg = ctx.vpl_kl_loss_total / ctx.num_samples
            recon_avg = ctx.vpl_reconstruction_loss_total / ctx.num_samples
            results['vpl_kl_loss'] = kl_avg.item() if torch.is_tensor(kl_avg) else kl_avg
            results['vpl_reconstruction_loss'] = recon_avg.item() if torch.is_tensor(recon_avg) else recon_avg
            # Add orthogonal loss if enabled
            if self.vpl_orthogonal_weight > 0.0 and hasattr(ctx, 'vpl_orthogonal_loss_total'):
                ortho_avg = ctx.vpl_orthogonal_loss_total / ctx.num_samples
                results['vpl_orthogonal_loss'] = ortho_avg.item() if torch.is_tensor(ortho_avg) else ortho_avg
            # Add vpl_total for monitor formatting (required by format_eval_res)
            if 'vpl_total' not in results:
                results['vpl_total'] = ctx.num_samples
            
            # Log VPL metrics to wandb if enabled (client-side logging)
            if ctx.cfg.wandb.use and ctx.cfg.wandb.online_track and ctx.cfg.wandb.client_train_info:
                try:
                    import wandb
                    client_id = getattr(ctx, 'client_id', None)
                    client_prefix = f'client_{client_id}/' if client_id is not None else 'client/'
                    
                    wandb_metrics = {
                        f'{client_prefix}{ctx.cur_mode}/vpl_total_loss': results.get('loss', 0.0),
                        f'{client_prefix}{ctx.cur_mode}/vpl_reconstruction_loss': results['vpl_reconstruction_loss'],
                        f'{client_prefix}{ctx.cur_mode}/vpl_kl_loss': results['vpl_kl_loss'],
                    }
                    
                    # Add orthogonal loss if enabled
                    if self.vpl_orthogonal_weight > 0.0 and 'vpl_orthogonal_loss' in results:
                        wandb_metrics[f'{client_prefix}{ctx.cur_mode}/vpl_orthogonal_loss'] = results['vpl_orthogonal_loss']
                    
                    # Use round as step for x-axis in wandb (if available)
                    step = None
                    if hasattr(ctx, 'cur_round'):
                        step = ctx.cur_round
                    
                    if step is not None:
                        wandb.log(wandb_metrics, step=step)
                    else:
                        wandb.log(wandb_metrics)
                except ImportError:
                    logger.warning("wandb not installed, skipping VPL metrics logging")
                except Exception as e:
                    logger.warning(f"Failed to log VPL metrics to wandb: {e}")
        
        # Collect z values for visualization (always, if available)
        # This is separate from GP prior collection - we always want z values for t-SNE
        if hasattr(self, 'z_history') and len(self.z_history) > 0:
            z_values = torch.cat(self.z_history, dim=0)  # (num_batches * batch_size, latent_dim)
            
            # Store z values for visualization (sample a reasonable number)
            num_samples = min(100, len(z_values))  # store at most 100 samples
            sampled_indices = torch.randperm(len(z_values))[:num_samples]
            self.client_z_values = z_values[sampled_indices]  # (num_samples, latent_dim)
            
            logger.info(f"Collected z values for visualization: "
                       f"{self.client_z_values.shape} from {len(self.z_history)} batches "
                       f"(round {ctx.cur_round if hasattr(ctx, 'cur_round') else 'unknown'})")
            
            # Clear history for next round
            self.z_history = []
        
        # Collect z distribution for GP prior (if enabled).
        # Aggregate the POSTERIOR parameters across samples:
        #   client_mu  = E_x[μ(x)]
        #   client_var = E_x[σ²(x)] + Var_x[μ(x)]   (law of total var)
        # This is the marginal q_ϕ(z | D_i) per Eq. 6, not the
        # empirical mean/var of sampled z's (which conflates the
        # posterior covariance with the reparameterisation noise).
        if self.vpl_use_gp_prior and \
                hasattr(self, '_posterior_mu_history') and \
                len(self._posterior_mu_history) > 0:
            all_mu = torch.cat(self._posterior_mu_history, dim=0)
            all_logvar = torch.cat(
                self._posterior_logvar_history, dim=0)
            client_mu = all_mu.mean(dim=0)
            within_var = torch.exp(all_logvar).mean(dim=0)
            if all_mu.shape[0] > 1:
                between_var = all_mu.var(dim=0, unbiased=False)
            else:
                between_var = torch.zeros_like(within_var)
            client_var = within_var + between_var
            self.client_z_mu = client_mu
            self.client_z_logvar = torch.log(client_var + 1e-8)
            # Clear history for next round.
            self._posterior_mu_history = []
            self._posterior_logvar_history = []
        elif self.vpl_use_gp_prior and \
                hasattr(self, 'client_z_values') and \
                self.client_z_values is not None:
            # Fallback (no eval pass occurred): use sampled z stats.
            z_values = self.client_z_values
            self.client_z_mu = z_values.mean(dim=0)
            z_var = z_values.var(dim=0)
            self.client_z_logvar = torch.log(z_var + 1e-8)
        
        setattr(ctx, 'eval_metrics', results)
    
    # GP Prior methods (only used when vpl_use_gp_prior=True)
    def get_client_z_distribution(self):
        """
        Get client's z distribution (mu, logvar) for server aggregation.
        Only available when GP prior is enabled.
        
        Returns:
            (mu, logvar): Tuple of mean and log variance tensors, or None
        """
        if not self.vpl_use_gp_prior:
            return None
        if not hasattr(self, 'client_z_mu') or self.client_z_mu is None:
            return None
        if not hasattr(self, 'client_z_logvar') or self.client_z_logvar is None:
            return None
        return (self.client_z_mu.clone(), self.client_z_logvar.clone())
    
    def get_client_z_values(self):
        """
        Get client's z values for visualization.
        Available for all VPL experiments (not just GP prior).
        
        Returns:
            z_values: Tensor of z values (num_samples, latent_dim), or None
        """
        if not hasattr(self, 'client_z_values') or self.client_z_values is None:
            return None
        return self.client_z_values.clone()
    
    def update_prior_from_server(self, client_mus, client_logvars, client_weights, current_round=None):
        """
        Update the mixture prior from server.
        Only used when GP prior is enabled.

        After updating the prior, ensures prior_logits (learnable Gumbel-Softmax
        weights) are included in the VPL optimizer, and anneals temperature.

        Args:
            client_mus: Mean vectors from other clients (num_clients, latent_dim)
            client_logvars: Log variance vectors from other clients (num_clients, latent_dim)
            client_weights: Sample-size weights (used for initialization only) (num_clients,)
            current_round: Current FL round (for temperature annealing)
        """
        if not self.vpl_use_gp_prior:
            return
        if hasattr(self.variational_encoder, 'update_prior'):
            self.variational_encoder.update_prior(client_mus, client_logvars, client_weights)

            # Ensure prior_logits is in the VPL optimizer
            if hasattr(self.variational_encoder, 'prior_logits') and \
               self.variational_encoder.prior_logits is not None:
                self._ensure_prior_logits_in_optimizer()

            # Anneal temperature
            if current_round is not None and hasattr(self.variational_encoder, 'anneal_temperature'):
                self.variational_encoder.anneal_temperature(current_round, self.total_round_num)

    def _ensure_prior_logits_in_optimizer(self):
        """Add prior_logits to the VPL optimizer if not already present."""
        logits_param = self.variational_encoder.prior_logits
        if logits_param is None:
            return

        # Try VPL optimizer first, then fall back to ctx optimizer
        optimizer = None
        if hasattr(self, 'ctx') and hasattr(self.ctx, 'vpl_optimizer') and self.ctx.vpl_optimizer is not None:
            optimizer = self.ctx.vpl_optimizer
        elif hasattr(self, 'ctx') and hasattr(self.ctx, 'optimizer'):
            optimizer = self.ctx.optimizer

        if optimizer is None:
            logger.warning("No optimizer found to add prior_logits to")
            return

        # Check if already tracked
        param_id = id(logits_param)
        existing_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if param_id not in existing_ids:
            optimizer.add_param_group({
                'params': [logits_param],
                'lr': optimizer.param_groups[0]['lr'],
            })
            logger.info(f"Added prior_logits to optimizer (lr={optimizer.param_groups[0]['lr']:.2e})")
    
    def update_orthogonal_label_from_server(self, label):
        """
        Update orthogonal label from server.
        
        Args:
            label: Orthogonal label (int)
        """
        self.orthogonal_label = label
    
    def get_client_orthogonal_prototypes(self):
        """
        Get client's orthogonal prototypes for visualization.
        
        Returns:
            prototypes: Tensor of prototypes (num_prototypes, latent_dim) or None
        """
        if hasattr(self, 'orthogonal_prototypes') and self.orthogonal_prototypes is not None:
            return self.orthogonal_prototypes
        return None
    
    def _compute_clop_orthogonal_loss(self, z, labels=None):
        """
        Compute CLOP orthogonal loss (Eq. 9).

        L_ortho = λ·||z - p_{y*}||² + γ·||PP^T - I||²_F

        Prototypes are learnable nn.Parameters. The pull loss moves
        both z and the assigned prototype closer together, while the
        orthonormality constraint keeps prototypes separated.

        Label assignment priority:
          1. Server-assigned label (k-means on client z-means)
          2. Local nearest-prototype fallback (round 1, before
             server has computed labels)

        Args:
            z: Latent embeddings (batch_size, latent_dim)
            labels: Optional labels for each sample (batch_size,)

        Returns:
            orthogonal_loss: Total orthogonal loss
            pull_loss: Pull loss component
            orthonorm_loss: Orthonormal constraint loss component
        """
        if self.orthogonal_prototypes is None:
            return (torch.tensor(0.0, device=z.device),
                    torch.tensor(0.0, device=z.device),
                    torch.tensor(0.0, device=z.device))

        batch_size, latent_dim = z.shape
        num_prototypes = self.orthogonal_prototypes.shape[0]

        # Determine orthogonal labels
        # Use server-assigned label when available (k-means or manual)
        if self.orthogonal_label is not None:
            orthogonal_labels = torch.full(
                (batch_size,), self.orthogonal_label,
                device=z.device, dtype=torch.long
            )
        else:
            # Fallback: assign to closest prototype (e.g. round 1)
            with torch.no_grad():
                similarities = torch.matmul(
                    z.detach(),
                    self.orthogonal_prototypes.detach().T
                )
                orthogonal_labels = torch.argmax(
                    similarities, dim=1
                )

        # Pull loss: move z toward assigned prototype (and vice versa)
        selected_prototypes = self.orthogonal_prototypes[
            orthogonal_labels
        ]  # (batch_size, latent_dim)
        pull_loss = torch.mean((z - selected_prototypes) ** 2)

        # Orthonormal constraint: ||PP^T - s²·I||²_F
        # Uses scaled identity since prototypes have norm ~prototype_scale
        PTP = torch.matmul(
            self.orthogonal_prototypes,
            self.orthogonal_prototypes.T
        )  # (num_prototypes, num_prototypes)
        orthonorm_loss = torch.norm(
            PTP - self._ortho_target, p='fro') ** 2

        # Total orthogonal loss
        orthogonal_loss = self.vpl_orthogonal_weight * pull_loss + \
            self.vpl_orthogonal_orthonorm_weight * orthonorm_loss

        return orthogonal_loss, pull_loss, orthonorm_loss
    


def call_vpl_reward_choice_trainer(trainer_type):
    if trainer_type == 'vplrewardchoicetrainer':
        trainer_builder = VPLRewardChoiceTrainer
        return trainer_builder
    # Also support vplgprewardchoicetrainer for backward compatibility
    # But it will use the same unified trainer with GP prior enabled via config
    elif trainer_type == 'vplgprewardchoicetrainer':
        trainer_builder = VPLRewardChoiceTrainer
        return trainer_builder


register_trainer('vplrewardchoicetrainer', call_vpl_reward_choice_trainer)
register_trainer('vplgprewardchoicetrainer', call_vpl_reward_choice_trainer)