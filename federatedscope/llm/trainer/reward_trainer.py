import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np
import os

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken

import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


def _get_batch_logps(logits, labels, average_log_prob=False):
    """
    Source: https://github.com/eric-mitchell/direct-preference-optimization/
        blob/main/trainers.py#L208

    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities.
            Label tokens with a value of -100 are ignored.
            Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability
            per (non-masked) token. Otherwise, return the sum of the
            log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum
            log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != DefaultToken.IGNORE_INDEX.value)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == DefaultToken.IGNORE_INDEX.value] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1),
                                   dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(policy_chosen_logps,
             policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps,
             beta,
             reference_free=False):
    """
    Source: https://github.com/eric-mitchell/direct-preference-optimization/
        blob/main/trainers.py#L208

    Compute the DPO loss for a batch of policy and reference
    model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model
            for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model
            for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model
            for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model
            for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something
            in the range of 0.1 to 0.5. We ignore the reference model
            as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model
            and implicitly use a reference model that assigns equal probability
            to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards
            for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps -
                               reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


class DPORewardTrainer(LLMTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.reward_coeff = config.llm.reward_coeff
        
        # VPL components for z-dependent generation
        self.variational_encoder = None
        self.feature_extractor = None
        self.z_to_embedding = None
        self.use_variational_generation = getattr(config.llm, 'rlhf_use_variational_generation', False)
        
        # Storage for z values collection (for visualization)
        self.collected_z_values = []
        
        if self.use_variational_generation:
            logger.info("DPORewardTrainer: Variational generation enabled. Will load VPL components.")

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)

        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)
        
        # Load VPL components for z-dependent generation (only once)
        if self.z_to_embedding is None:
            from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
            
            selector_ckpt_path = getattr(ctx.cfg.llm, 'rlhf_selector_checkpoint', None)
            if selector_ckpt_path is None:
                selector_ckpt_path = getattr(ctx.cfg.llm, 'selector_save_to', None)
            
            if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                logger.info(f"Loading VPL components from {selector_ckpt_path} for z-dependent generation")
                variational_encoder, feature_extractor, _, z_to_embedding = load_vpl_components_from_checkpoint(
                    selector_ckpt_path, ctx.cfg, device=ctx.device
                )
                
                if variational_encoder is not None and z_to_embedding is not None:
                    self.variational_encoder = variational_encoder
                    self.feature_extractor = feature_extractor
                    
                    # Verify z_to_embedding output dimension matches model embedding dimension
                    try:
                        actual_embedding_dim = ctx.model.get_input_embeddings().embedding_dim
                        z_to_embedding_output_dim = z_to_embedding.weight.shape[0]  # (out_features, in_features)
                        
                        if z_to_embedding_output_dim != actual_embedding_dim:
                            logger.warning(f"z_to_embedding output dimension ({z_to_embedding_output_dim}) does not match "
                                         f"model embedding dimension ({actual_embedding_dim}). Reinitializing z_to_embedding.")
                            # Reinitialize z_to_embedding with correct output dimension
                            vpl_latent_dim = z_to_embedding.weight.shape[1]  # in_features
                            import torch.nn as nn
                            self.z_to_embedding = nn.Linear(vpl_latent_dim, actual_embedding_dim).to(ctx.device)
                            logger.info(f"Reinitialized z_to_embedding: {vpl_latent_dim} -> {actual_embedding_dim}")
                        else:
                            self.z_to_embedding = z_to_embedding
                    except Exception as e:
                        logger.warning(f"Could not verify z_to_embedding dimension: {e}. Using loaded z_to_embedding.")
                        self.z_to_embedding = z_to_embedding
                    
                    # Match dtype with model
                    model_dtype = next(ctx.model.parameters()).dtype
                    if self.variational_encoder is not None:
                        self.variational_encoder = self.variational_encoder.to(dtype=model_dtype)
                    if self.feature_extractor is not None:
                        self.feature_extractor = self.feature_extractor.to(dtype=model_dtype)
                    if self.z_to_embedding is not None:
                        self.z_to_embedding = self.z_to_embedding.to(dtype=model_dtype)
                    
                    logger.info(f"VPL components loaded successfully for z-dependent generation (dtype: {model_dtype})")
                else:
                    logger.warning("Failed to load VPL components. Falling back to standard generation.")
                    self.use_variational_generation = False
            else:
                logger.warning(f"Selector checkpoint not found: {selector_ckpt_path}. Disabling variational generation.")
                self.use_variational_generation = False

        # Add z_to_embedding parameters to optimizer so it is
        # trainable during DPO (allows the projection to adapt)
        if self.z_to_embedding is not None and hasattr(ctx, 'optimizer'):
            self.z_to_embedding.requires_grad_(True)
            try:
                ctx.optimizer.add_param_group({
                    'params': list(self.z_to_embedding.parameters()),
                    'lr': ctx.cfg.train.optimizer.lr
                })
                logger.info(
                    "Added z_to_embedding to DPO optimizer "
                    f"(lr={ctx.cfg.train.optimizer.lr})")
            except Exception as e:
                logger.warning(
                    f"Could not add z_to_embedding to optimizer: {e}")

    def _hook_on_batch_forward(self, ctx):
        if ctx.cfg.llm.accelerator.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            with torch.no_grad():
                ref_win_logps, ref_lose_logps = \
                    self._batch_forward(ctx, win_input_ids, win_labels,
                                        win_attention_mask, lose_input_ids,
                                        lose_labels, lose_attention_mask,
                                        disable_adapter=True)

            adap_win_logps, adap_lose_logps = \
                self._batch_forward(ctx, win_input_ids, win_labels,
                                    win_attention_mask, lose_input_ids,
                                    lose_labels, lose_attention_mask,
                                    disable_adapter=False)

        elif ctx.cfg.llm.deepspeed.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            with torch.no_grad():
                ref_win_logps, ref_lose_logps = \
                    self._batch_forward_deepspeed(
                        ctx, win_input_ids, win_labels, win_attention_mask,
                        lose_input_ids, lose_labels, lose_attention_mask,
                        disable_adapter=True)

            adap_win_logps, adap_lose_logps = \
                self._batch_forward_deepspeed(
                    ctx, win_input_ids, win_labels, win_attention_mask,
                    lose_input_ids, lose_labels, lose_attention_mask,
                    disable_adapter=False)

        else:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(
                ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(
                ctx.device)

            with torch.no_grad():
                ref_win_logps, ref_lose_logps = \
                    self._batch_forward(ctx, win_input_ids, win_labels,
                                        win_attention_mask, lose_input_ids,
                                        lose_labels, lose_attention_mask,
                                        disable_adapter=True)

            adap_win_logps, adap_lose_logps = \
                self._batch_forward(ctx, win_input_ids, win_labels,
                                    win_attention_mask, lose_input_ids,
                                    lose_labels, lose_attention_mask,
                                    disable_adapter=False)

        # loss follows using Equation (7) of Direct Preference Optimization:
        # Your Language Model is Secretly a Reward Model
        loss, win_rewards, lose_rewards = dpo_loss(adap_win_logps,
                                                   adap_lose_logps,
                                                   ref_win_logps,
                                                   ref_lose_logps,
                                                   beta=self.reward_coeff)

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(torch.zeros(len(win_input_ids)), LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(
            torch.where(win_rewards.cpu() > lose_rewards.cpu(),
                        torch.zeros(len(win_input_ids)),
                        torch.ones(len(win_input_ids))), LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(win_input_ids), LIFECYCLE.BATCH)

    def _infer_z_from_input(self, ctx, input_ids, attention_mask=None):
        """Infer z from input using feature_extractor and variational_encoder (Step 4)."""
        if self.variational_encoder is None or self.feature_extractor is None:
            return None
        
        try:
            # Get embeddings from model
            input_embeddings = ctx.model.get_input_embeddings()(input_ids)
            
            # Match dtype with model
            model_dtype = next(ctx.model.parameters()).dtype
            input_embeddings = input_embeddings.to(dtype=model_dtype)
            
            # Extract features (use mean pooling over sequence length)
            if attention_mask is not None:
                # Mask out padding tokens
                mask = attention_mask.unsqueeze(-1).to(dtype=model_dtype)
                pooled_embeddings = (input_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled_embeddings = input_embeddings.mean(dim=1)
            
            # Extract features using feature_extractor
            features = self.feature_extractor(pooled_embeddings)
            
            # Encode to z using variational_encoder
            z_mu, z_logvar = self.variational_encoder.encode(features)
            
            # Sample z using reparameterization trick
            # Use ctx.cur_mode to determine if in training mode
            from federatedscope.core.trainers.enums import MODE
            is_training = hasattr(ctx, 'cur_mode') and ctx.cur_mode == MODE.TRAIN
            if is_training:
                std = torch.exp(0.5 * z_logvar)
                eps = torch.randn_like(std)
                z = z_mu + eps * std
            else:
                z = z_mu  # Use mean during inference
            
            return z
        except Exception as e:
            logger.warning(f"Failed to infer z from input: {e}. Using zero z.")
            vpl_latent_dim = getattr(ctx.cfg.llm, 'vpl_latent_dim', 32)
            batch_size = input_ids.shape[0]
            model_dtype = next(ctx.model.parameters()).dtype
            return torch.zeros(batch_size, vpl_latent_dim, device=ctx.device, dtype=model_dtype)
    
    def _get_z_from_batch(self, ctx, batch_size, input_ids=None, attention_mask=None):
        """Get z values from batch data or infer via variational encoder (Step 4: improved inference)."""
        z = None
        
        # Try to get z from batch data
        if 'z' in ctx.data_batch:
            z = ctx.data_batch['z']
            if isinstance(z, list):
                z = torch.stack([torch.tensor(zi) if not isinstance(zi, torch.Tensor) else zi for zi in z])
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z)
            z = z.to(ctx.device)
            # Match dtype with model
            model_dtype = next(ctx.model.parameters()).dtype
            z = z.to(dtype=model_dtype)
            # Ensure correct shape: (batch_size, latent_dim)
            if z.dim() == 1:
                z = z.unsqueeze(0).expand(batch_size, -1)
            elif z.shape[0] != batch_size:
                # If single z for batch, expand it
                if z.shape[0] == 1:
                    z = z.expand(batch_size, -1)
                else:
                    logger.warning(f"z shape mismatch: {z.shape} vs batch_size {batch_size}. Using first z.")
                    z = z[0:1].expand(batch_size, -1)
        
        # If z not in batch and variational encoder available, infer it from input (Step 4)
        if z is None and self.use_variational_generation and self.variational_encoder is not None:
            if input_ids is not None:
                # Infer z from input using feature_extractor + variational_encoder
                z = self._infer_z_from_input(ctx, input_ids, attention_mask)
                logger.debug(f"Inferred z from input: shape {z.shape}")
            else:
                # Fallback: use zero z if input_ids not available
                vpl_latent_dim = getattr(ctx.cfg.llm, 'vpl_latent_dim', 32)
                model_dtype = next(ctx.model.parameters()).dtype
                z = torch.zeros(batch_size, vpl_latent_dim, device=ctx.device, dtype=model_dtype)
                logger.debug("z not found in batch and input_ids not available, using zero z")
        
        # Collect z values for visualization (store mean z per batch)
        # Use ctx.cur_mode to determine if in training mode
        from federatedscope.core.trainers.enums import MODE
        is_training = hasattr(ctx, 'cur_mode') and ctx.cur_mode == MODE.TRAIN
        if z is not None and is_training:
            # Store z values (detach to avoid gradient tracking)
            z_detached = z.detach().cpu()
            # Store mean z per batch (or all z values if batch is small)
            # Convert to float32 before numpy conversion (BFloat16 is not supported by numpy)
            if batch_size <= 4:
                if isinstance(z_detached, torch.Tensor):
                    # Convert BFloat16 to float32 for numpy compatibility
                    z_detached_float = z_detached.float() if z_detached.dtype == torch.bfloat16 else z_detached
                    self.collected_z_values.extend(z_detached_float.numpy())
                else:
                    self.collected_z_values.extend(z_detached)
            else:
                # For larger batches, store mean z
                if isinstance(z_detached, torch.Tensor):
                    mean_z = z_detached.mean(dim=0, keepdim=True)
                    # Convert BFloat16 to float32 for numpy compatibility
                    mean_z_float = mean_z.float() if mean_z.dtype == torch.bfloat16 else mean_z
                    self.collected_z_values.extend(mean_z_float.numpy())
                else:
                    import numpy as np
                    mean_z = np.mean(z_detached, axis=0, keepdims=True)
                    self.collected_z_values.extend(mean_z)
        
        return z
    
    def _inject_z_to_embeddings(self, ctx, input_ids, z):
        """Inject z-dependent bias into input embeddings."""
        if z is None or self.z_to_embedding is None:
            return None  # Return None to use input_ids directly
        
        # Match z dtype with model dtype
        model_dtype = next(ctx.model.parameters()).dtype
        z = z.to(dtype=model_dtype)
        
        # Ensure z_to_embedding has the same dtype as the model
        if self.z_to_embedding.weight.dtype != model_dtype:
            self.z_to_embedding = self.z_to_embedding.to(dtype=model_dtype)
        
        # Get input embeddings
        input_embeddings = ctx.model.get_input_embeddings()(input_ids)
        actual_embedding_dim = input_embeddings.shape[-1]
        
        # Verify z_to_embedding output dimension matches actual embedding dimension
        z_to_embedding_output_dim = self.z_to_embedding.weight.shape[0]
        if z_to_embedding_output_dim != actual_embedding_dim:
            logger.error(f"z_to_embedding output dimension ({z_to_embedding_output_dim}) does not match "
                        f"model embedding dimension ({actual_embedding_dim}). Cannot inject z.")
            return None  # Return None to fall back to standard generation
        
        # Project z to embedding space
        z_embedding = self.z_to_embedding(z)  # (batch_size, embedding_dim)

        # Add z_embedding to all token embeddings
        # z_embedding: (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        z_embedding = z_embedding.unsqueeze(1)
        # input_embeddings: (batch_size, seq_len, embedding_dim)
        inputs_embeds = input_embeddings + z_embedding

        # Ensure output dtype matches model dtype (prevents Float vs BFloat16 mismatch)
        inputs_embeds = inputs_embeds.to(dtype=model_dtype)
        
        return inputs_embeds
    
    def _batch_forward(self,
                       ctx,
                       win_input_ids,
                       win_labels,
                       win_attention_mask,
                       lose_input_ids,
                       lose_labels,
                       lose_attention_mask,
                       disable_adapter=False):
        # Get z from batch or infer (Step 4: improved inference with input_ids)
        batch_size = win_input_ids.shape[0]
        # Use win_input_ids for z inference (both win and lose should use same z for a given prompt)
        z = self._get_z_from_batch(ctx, batch_size, input_ids=win_input_ids, attention_mask=win_attention_mask)
        
        # Inject z into embeddings for win (chosen) responses
        # Use z if available (either from data or inferred), regardless of use_variational_generation flag
        # This allows conditional training with z values stored in preference data
        win_inputs_embeds = None
        if z is not None:
            # Load z_to_embedding if not already loaded (for conditional training with stored z)
            if self.z_to_embedding is None and 'z' in ctx.data_batch:
                # Try to load z_to_embedding from selector checkpoint
                from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                selector_ckpt_path = getattr(ctx.cfg.llm, 'rlhf_selector_checkpoint', None)
                if selector_ckpt_path is None:
                    selector_ckpt_path = getattr(ctx.cfg.llm, 'selector_save_to', None)
                
                if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                    _, _, _, z_to_embedding = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, ctx.cfg, device=ctx.device
                    )
                    if z_to_embedding is not None:
                        # Match dtype with model
                        model_dtype = next(ctx.model.parameters()).dtype
                        z_to_embedding = z_to_embedding.to(dtype=model_dtype)
                        self.z_to_embedding = z_to_embedding
                        logger.info(f"Loaded z_to_embedding for conditional training with stored z values (dtype: {model_dtype})")
            
            if self.z_to_embedding is not None:
                win_inputs_embeds = self._inject_z_to_embeddings(ctx, win_input_ids, z)
        
        # Inject z into embeddings for lose (rejected) responses
        lose_inputs_embeds = None
        if z is not None and self.z_to_embedding is not None:
            lose_inputs_embeds = self._inject_z_to_embeddings(ctx, lose_input_ids, z)
        
        # Forward pass for win (chosen) responses
        if win_inputs_embeds is not None:
            win_outputs = ctx.model(disable_adapter=disable_adapter,
                                    inputs_embeds=win_inputs_embeds,
                                    labels=win_labels,
                                    attention_mask=win_attention_mask)
        else:
            win_outputs = ctx.model(disable_adapter=disable_adapter,
                                    input_ids=win_input_ids,
                                    labels=win_labels,
                                    attention_mask=win_attention_mask)
        win_logps = _get_batch_logps(win_outputs.logits,
                                     win_labels,
                                     average_log_prob=False)

        # Forward pass for lose (rejected) responses
        if lose_inputs_embeds is not None:
            lose_outputs = ctx.model(disable_adapter=disable_adapter,
                                     inputs_embeds=lose_inputs_embeds,
                                     labels=lose_labels,
                                     attention_mask=lose_attention_mask)
        else:
            lose_outputs = ctx.model(disable_adapter=disable_adapter,
                                     input_ids=lose_input_ids,
                                     labels=lose_labels,
                                     attention_mask=lose_attention_mask)
        lose_logps = _get_batch_logps(lose_outputs.logits,
                                      lose_labels,
                                      average_log_prob=False)

        return win_logps, lose_logps

    def _batch_forward_deepspeed(self,
                                 ctx,
                                 win_input_ids,
                                 win_labels,
                                 win_attention_mask,
                                 lose_input_ids,
                                 lose_labels,
                                 lose_attention_mask,
                                 disable_adapter=False):
        # Get z from batch or infer (Step 4: improved inference with input_ids)
        batch_size = win_input_ids.shape[0]
        # Use win_input_ids for z inference (both win and lose should use same z for a given prompt)
        z = self._get_z_from_batch(ctx, batch_size, input_ids=win_input_ids, attention_mask=win_attention_mask)
        
        # Inject z into embeddings for win (chosen) responses
        win_inputs_embeds = None
        if z is not None and self.z_to_embedding is not None:
            win_inputs_embeds = self._inject_z_to_embeddings(ctx, win_input_ids, z)

        # Inject z into embeddings for lose (rejected) responses
        lose_inputs_embeds = None
        if z is not None and self.z_to_embedding is not None:
            lose_inputs_embeds = self._inject_z_to_embeddings(ctx, lose_input_ids, z)
        
        # Forward pass for win (chosen) responses
        if win_inputs_embeds is not None:
            win_outputs = ctx.model_engine(disable_adapter=disable_adapter,
                                          inputs_embeds=win_inputs_embeds,
                                          labels=win_labels,
                                          attention_mask=win_attention_mask)
        else:
            win_outputs = ctx.model_engine(disable_adapter=disable_adapter,
                                           input_ids=win_input_ids,
                                           labels=win_labels,
                                           attention_mask=win_attention_mask)
        win_logps = _get_batch_logps(win_outputs.logits,
                                     win_labels,
                                     average_log_prob=False)

        # Forward pass for lose (rejected) responses
        if lose_inputs_embeds is not None:
            lose_outputs = ctx.model_engine(disable_adapter=disable_adapter,
                                             inputs_embeds=lose_inputs_embeds,
                                             labels=lose_labels,
                                             attention_mask=lose_attention_mask)
        else:
            lose_outputs = ctx.model_engine(disable_adapter=disable_adapter,
                                             input_ids=lose_input_ids,
                                             labels=lose_labels,
                                             attention_mask=lose_attention_mask)
        lose_logps = _get_batch_logps(lose_outputs.logits,
                                      lose_labels,
                                      average_log_prob=False)

        return win_logps, lose_logps

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(ctx.loss_task)
            ctx.optimizer.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()
            ctx.optimizer.zero_grad()

        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()

            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                   ctx.grad_clip)
                ctx.optimizer.step()
                if ctx.scheduler is not None:
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()

        # move the training data to cpu
        ctx.data_batch['win_input_ids'].cpu()
        ctx.data_batch['win_labels'].cpu()
        ctx.data_batch['win_attention_mask'].cpu()
        ctx.data_batch['lose_input_ids'].cpu()
        ctx.data_batch['lose_labels'].cpu()
        ctx.data_batch['lose_attention_mask'].cpu()

    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.append(ctx.y_pred.detach().cpu().numpy())

    def get_collected_z_values(self):
        """Get collected z values for visualization."""
        if len(self.collected_z_values) > 0:
            import numpy as np
            return np.array(self.collected_z_values)
        return None
    
    def clear_collected_z_values(self):
        """Clear collected z values."""
        self.collected_z_values = []
    
    def _hook_on_fit_end(self, ctx):
        # Only concatenate if there are values (for evaluation, ys_true/ys_pred might be empty)
        if len(ctx.ys_true) > 0:
            ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        else:
            ctx.ys_true = CtxVar(np.array([]), LIFECYCLE.ROUTINE)
        
        if len(ctx.ys_pred) > 0:
            ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        else:
            ctx.ys_pred = CtxVar(np.array([]), LIFECYCLE.ROUTINE)
        
        # Ensure tokenizer is available in ctx for evaluation metrics
        if not hasattr(ctx, 'tokenizer') and hasattr(self, 'tokenizer'):
            ctx.tokenizer = self.tokenizer
        
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)


def call_reward_trainer(trainer_type):
    if trainer_type == 'llmdporewardtrainer':
        trainer_builder = DPORewardTrainer
        return trainer_builder


register_trainer('llmdporewardtrainer', call_reward_trainer)
