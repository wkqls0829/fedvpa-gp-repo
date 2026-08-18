import torch
import numpy as np
from tqdm import tqdm
import logging
import inspect
import warnings

# Suppress decoder-only right-padding warnings globally
# This warning appears when using decoder-only models with right-padding
warnings.filterwarnings(
    "ignore",
    message=".*decoder-only architecture.*right-padding.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*right-padding was detected.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*right-padding.*",
    category=UserWarning
)
# Also suppress the specific transformers warning
warnings.filterwarnings(
    "ignore",
    message=".*padding_side='left'.*",
    category=UserWarning
)

import federatedscope.register as register
from federatedscope.llm.reward.reward_model_implementations import (
    GPT2HarmlessRewardModel, GPT2HelpfulRewardModel)

logger = logging.getLogger(__name__)

# Global cache for reward models to avoid reloading them on every call
REWARD_MODELS = {}


def get_reward_models(device):
    """
    Loads and caches the reward models.
    """
    global REWARD_MODELS
    if 'harmless' not in REWARD_MODELS:
        logger.info("Loading Harmlessness Reward Model...")
        REWARD_MODELS['harmless'] = GPT2HarmlessRewardModel(device=device)
    if 'helpful' not in REWARD_MODELS:
        logger.info("Loading Helpfulness Reward Model...")
        REWARD_MODELS['helpful'] = GPT2HelpfulRewardModel(device=device)
    return REWARD_MODELS['harmless'], REWARD_MODELS['helpful']


def _get_or_compute_hhrl_scores(ctx):
    """
    A helper function that computes reward scores and caches them in the `ctx`
    to avoid redundant computation within the same evaluation round.
    
    Note: Cache is cleared between rounds to ensure fresh computation.
    Note: Reward model evaluation is only for HRL (hh-rlhf) dataset, not for HHST.
    """
    # Check dataset type - only evaluate for hh-rlhf (HRL), not for hhst
    dataset_type = getattr(ctx.cfg.data, 'type', '').lower()
    if 'hh-rlhf' not in dataset_type and 'hrl' not in dataset_type:
        # Skip reward model evaluation for non-HRL datasets (e.g., HHST)
        logger.debug(f"Skipping reward model evaluation for dataset type: {dataset_type} (only for HRL/hh-rlhf)")
        return {}
    
    cache_key = f'{ctx.cur_split}_hhrl_scores'
    # Clear cache if round has changed (for training, we want fresh metrics each round)
    # IMPORTANT: For test evaluation, we should NOT cache results across rounds
    # because the model changes during training, so responses should be regenerated
    round_cache_key = f'{cache_key}_round'
    current_round = getattr(ctx, 'cur_round', None)
    cached_round = getattr(ctx, round_cache_key, None)
    
    # If round has changed, clear the cache
    if cached_round is not None and current_round is not None and cached_round != current_round:
        if hasattr(ctx, cache_key):
            delattr(ctx, cache_key)
            logger.info(f"Cleared reward model cache for {ctx.cur_split} (round changed from {cached_round} to {current_round})")
    
    # For test evaluation, always regenerate responses (don't use cache)
    # because the model is being trained and responses should change each round
    if ctx.cur_split == 'test':
        if hasattr(ctx, cache_key):
            delattr(ctx, cache_key)
            logger.info(f"Cleared reward model cache for test evaluation (forcing regeneration for round {current_round})")
    
    if hasattr(ctx, cache_key) and ctx.cur_split != 'test':
        # Only use cache for non-test splits (e.g., val)
        return getattr(ctx, cache_key)

    eval_loader = getattr(ctx, f'{ctx.cur_split}_loader', None)
    if eval_loader is None:
        logger.warning(f"ctx.{ctx.cur_split}_loader is not available, "
                       f"skipping reward eval.")
        return {}

    harmless_reward_model, helpful_reward_model = get_reward_models(
        ctx.device)

    all_harmless_scores = []
    all_helpful_scores = []

    # Get tokenizer from ctx or trainer
    tokenizer = getattr(ctx, 'tokenizer', None)
    if tokenizer is None:
        # Try to get from trainer if available
        trainer = getattr(ctx, 'trainer', None)
        if trainer is not None and hasattr(trainer, 'tokenizer'):
            tokenizer = trainer.tokenizer
        else:
            logger.warning("Tokenizer not found in ctx or trainer, skipping reward model evaluation")
            return {}
    
    # For decoder-only architectures, enforce left padding to avoid warnings and
    # ensure correct generation behavior.
    if getattr(tokenizer, 'padding_side', None) != 'left':
        logger.info("Setting tokenizer.padding_side='left' for decoder-only model generation.")
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Suppress the decoder-only right-padding warning since we've set padding_side='left'
    # This warning comes from transformers library during generation
    # Apply comprehensive warning filters
    warnings.filterwarnings(
        "ignore",
        message=".*decoder-only architecture.*right-padding.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*right-padding.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*padding_side.*",
        category=UserWarning
    )

    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1
    }

    # Limit the number of samples for evaluation to speed up
    # Default: evaluate on max 30 samples, or all if less than 30
    # Final round: use full dataset (no limit)
    is_final_round = False
    if hasattr(ctx, 'cur_round') and hasattr(ctx.cfg.federate, 'total_round_num'):
        is_final_round = (ctx.cur_round + 1 == ctx.cfg.federate.total_round_num)
    
    max_eval_samples = getattr(ctx.cfg.eval, 'max_samples_for_reward', 30)
    if is_final_round:
        # Final round: use full dataset (no limit)
        max_eval_samples = float('inf')
        logger.info(f"Final round detected (round {ctx.cur_round + 1}/{ctx.cfg.federate.total_round_num}). Using full dataset for reward model evaluation.")
    elif max_eval_samples <= 0:
        max_eval_samples = float('inf')  # Evaluate on all samples
    
    total_samples_evaluated = 0
    should_limit = max_eval_samples != float('inf')

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating with Reward Models")):
        # Stop early if we've reached the max number of samples
        if should_limit and total_samples_evaluated >= max_eval_samples:
            break
        
        # Handle different data formats: RLHF uses win_input_ids/lose_input_ids, 
        # regular training uses input_ids
        if 'win_input_ids' in batch:
            # RLHF format: use win_input_ids for evaluation
            input_ids = batch['win_input_ids'].to(ctx.device)
            attention_mask = batch.get('win_attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(ctx.device)
        elif 'input_ids' in batch:
            # Regular format
            input_ids = batch['input_ids'].to(ctx.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(ctx.device)
        else:
            logger.warning(f"Batch {batch_idx} does not contain 'input_ids' or 'win_input_ids', skipping...")
            continue
        
        # Decode the entire input_ids to get the formatted prompt string
        # This is what the model sees as input.
        prompts = tokenizer.batch_decode(input_ids,
                                             skip_special_tokens=True)

        # Generate with or without attention_mask
        # Suppress right-padding warnings during generation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*decoder-only architecture.*right-padding.*",
                category=UserWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*right-padding was detected.*",
                category=UserWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*right-padding.*",
                category=UserWarning
            )
            if attention_mask is not None:
                generated_ids = ctx.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=ctx.cfg.llm.max_new_token,
                    **generation_kwargs)
            else:
                generated_ids = ctx.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=ctx.cfg.llm.max_new_token,
                    **generation_kwargs)
        
        completions = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        # The full text for the reward model is the generated text
        # The prompt for the reward model is the original input text
        # Limit the number of samples per batch if needed
        batch_size = len(completions)
        if should_limit and total_samples_evaluated + batch_size > max_eval_samples:
            # Only evaluate the remaining samples needed
            remaining = max_eval_samples - total_samples_evaluated
            completions = completions[:remaining]
            prompts = prompts[:remaining]
            batch_size = remaining

        # Get client_id from batch if available (for client-specific evaluation)
        batch_client_ids = None
        if isinstance(batch, dict):
            batch_client_ids = batch.get('client_id', None)
        # If not in batch, try to get from dataset
        if batch_client_ids is None:
            try:
                dataset = eval_loader.dataset
                if hasattr(dataset, 'client_ids') and len(dataset.client_ids) > 0:
                    start_idx = batch_idx * eval_loader.batch_size
                    end_idx = min(start_idx + len(completions), len(dataset.client_ids))
                    batch_client_ids = dataset.client_ids[start_idx:end_idx]
                    if all(cid is None for cid in batch_client_ids):
                        batch_client_ids = None
            except:
                pass
        
        # Determine client type and evaluate only relevant reward model
        # Harmlessness clients: 1 to client_num // 2
        # Helpfulness clients: client_num // 2 + 1 to client_num
        client_num = getattr(ctx.cfg.federate, 'client_num', 10)
        harmless_clients_num = client_num // 2
        
        # Evaluate reward models based on client type
        if batch_client_ids is not None:
            # Client-specific evaluation: only evaluate relevant reward model for each client
            for i, client_id in enumerate(batch_client_ids):
                if client_id is None:
                    continue
                
                if client_id <= harmless_clients_num:
                    # Harmlessness client: only evaluate harmlessness reward
                    harmless_scores = harmless_reward_model.get_rewards([completions[i]], [prompts[i]])
                    all_harmless_scores.extend(harmless_scores)
                else:
                    # Helpfulness client: only evaluate helpfulness reward
                    helpful_scores = helpful_reward_model.get_rewards([completions[i]], [prompts[i]])
                    all_helpful_scores.extend(helpful_scores)
        else:
            # Fallback: evaluate both (for backward compatibility)
            harmless_scores = harmless_reward_model.get_rewards(completions, prompts)
            helpful_scores = helpful_reward_model.get_rewards(completions, prompts)
            all_harmless_scores.extend(harmless_scores)
            all_helpful_scores.extend(helpful_scores)
        
        total_samples_evaluated += batch_size
        
        # Stop if we've reached the limit
        if should_limit and total_samples_evaluated >= max_eval_samples:
            break

    results = {}
    if all_harmless_scores:
        avg_harmlessness = np.mean(all_harmless_scores)
        results['avg_harmlessness'] = avg_harmlessness
        if should_limit:
            logger.info(f"Evaluated {len(all_harmless_scores)} samples for harmlessness (limited from full dataset)")
        logger.info(f"Average harmlessness reward score: {avg_harmlessness:.4f} (min: {np.min(all_harmless_scores):.4f}, max: {np.max(all_harmless_scores):.4f})")
    if all_helpful_scores:
        avg_helpfulness = np.mean(all_helpful_scores)
        results['avg_helpfulness'] = avg_helpfulness
        if should_limit:
            logger.info(f"Evaluated {len(all_helpful_scores)} samples for helpfulness (limited from full dataset)")
        logger.info(f"Average helpfulness reward score: {avg_helpfulness:.4f} (min: {np.min(all_helpful_scores):.4f}, max: {np.max(all_helpful_scores):.4f})")

    setattr(ctx, cache_key, results)
    # Store current round for cache invalidation
    if hasattr(ctx, 'cur_round'):
        setattr(ctx, f'{cache_key}_round', ctx.cur_round)
    return results


# --- Metric 1: Harmlessness ---
def eval_harmlessness(ctx, **kwargs):
    # Only compute reward model scores for test/val splits, not for train split
    cur_split = getattr(ctx, 'cur_split', 'train')
    if cur_split == 'train':
        return 0.0
    
    # Only evaluate harmlessness for harmlessness clients (client_id 1 to client_num // 2)
    client_id = getattr(ctx, 'client_id', None)
    if client_id is not None:
        client_num = getattr(ctx.cfg.federate, 'client_num', 10)
        harmless_clients_num = client_num // 2
        # Only evaluate if this is a harmlessness client
        if client_id > harmless_clients_num:
            # This is a helpfulness client, skip harmlessness evaluation
            return 0.0
    
    scores = _get_or_compute_hhrl_scores(ctx)
    return scores.get('avg_harmlessness', 0.0)


def register_harmlessness_metric(types):
    if 'avg_harmlessness' in types:
        return 'avg_harmlessness', eval_harmlessness, True
    return None


# --- Metric 2: Helpfulness ---
def eval_helpfulness(ctx, **kwargs):
    # Only compute reward model scores for test/val splits, not for train split
    cur_split = getattr(ctx, 'cur_split', 'train')
    if cur_split == 'train':
        return 0.0
    
    # Only evaluate helpfulness for helpfulness clients (client_id > client_num // 2)
    client_id = getattr(ctx, 'client_id', None)
    if client_id is not None:
        client_num = getattr(ctx.cfg.federate, 'client_num', 10)
        harmless_clients_num = client_num // 2
        # Only evaluate if this is a helpfulness client
        if client_id <= harmless_clients_num:
            # This is a harmlessness client, skip helpfulness evaluation
            return 0.0
    
    scores = _get_or_compute_hhrl_scores(ctx)
    return scores.get('avg_helpfulness', 0.0)


def register_helpfulness_metric(types):
    if 'avg_helpfulness' in types:
        return 'avg_helpfulness', eval_helpfulness, True
    return None


# Register both metrics with the framework
register.register_metric('avg_harmlessness', register_harmlessness_metric)
register.register_metric('avg_helpfulness', register_helpfulness_metric)
