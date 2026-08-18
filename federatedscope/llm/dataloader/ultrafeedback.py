import random
import datasets
import numpy as np
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMDataset
from collections import defaultdict


ULTRAFEEDBACK_PROMPT_DICT = {
    "generation": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is helpful, honest, follows instructions, and is truthful.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_helpfulness": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is helpful.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_honesty": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is honest.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_instruction_following": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that follows instructions.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_truthfulness": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is truthful.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "comparison": (
        "Below is a conversation between a human and an AI assistant, "
        "followed by two responses. Pick the response that is better "
        "according to the preference. State your choice with a single capital "
        "letter, i.e., \"A\" if RESPONSE A is better, "
        "\"B\" if RESPONSE B is better.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE A: {output_A}\n"
        "### RESPONSE B: {output_B}\n"
        "### YOUR CHOICE:"
    )
}


def extract_score_from_annotation(annotation):
    """Extract numeric score from annotation dict."""
    if isinstance(annotation, dict):
        if 'Rating' in annotation:
            try:
                return float(annotation['Rating'])
            except (ValueError, TypeError):
                pass
    elif isinstance(annotation, (int, float)):
        return float(annotation)
    return None


def extract_prompt_from_ultrafeedback(example):
    """Extract prompt from UltraFeedback example."""
    if 'instruction' in example:
        return example['instruction']
    elif 'prompt' in example:
        return example['prompt']
    elif 'messages' in example and len(example['messages']) > 0:
        # Extract from messages format
        messages = example['messages']
        if isinstance(messages, list) and len(messages) > 0:
            # Usually the first user message is the prompt
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    return msg.get('content', '')
    return None


def load_ultrafeedback_dataset(config, tokenizer):
    """
    Loads and processes the UltraFeedback dataset from Hugging Face.
    Filters for conflicting pairs with score difference >= threshold (default 3.0).
    Returns a single unified dataset, which the framework will split according to the configuration.
    """
    logger.info("Loading and processing UltraFeedback dataset from Hugging Face...")
    
    # Get threshold from config (default: 3.0)
    threshold = getattr(config.data, 'ultrafeedback_threshold', 3.0)
    logger.info(f"Using threshold: {threshold} for conflicting pairs")
    
    annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    
    try:
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e
    
    # Collect conflicting pairs
    conflicting_pairs = []
    dim_pairs = defaultdict(list)
    
    logger.info("Processing UltraFeedback data to extract conflicting pairs...")
    
    for idx, sample in enumerate(data):
        if idx % 10000 == 0 and idx > 0:
            logger.info(f"  Processing {idx}/{len(data)}...")
        
        if 'completions' not in sample or len(sample['completions']) < 2:
            continue
        
        completions = sample['completions']
        
        # Extract prompt
        prompt = extract_prompt_from_ultrafeedback(sample)
        if prompt is None:
            continue
        
        completion_scores = []
        for comp in completions:
            if 'annotations' not in comp:
                continue
            
            annotations = comp['annotations']
            if not isinstance(annotations, dict):
                continue
            
            dim_scores = {}
            for dim in annotation_dims:
                if dim in annotations:
                    score = extract_score_from_annotation(annotations[dim])
                    if score is not None:
                        dim_scores[dim] = score
            
            overall_score = comp.get('overall_score', None)
            if overall_score is None:
                overall_score = comp.get('fine-grained_score', None)
            
            # Get completion text (UltraFeedback uses 'response' field)
            completion_text = comp.get('response', '')
            if not completion_text:
                continue
            
            if dim_scores:
                completion_scores.append({
                    'text': completion_text,
                    'dim_scores': dim_scores,
                    'overall_score': overall_score
                })
        
        if len(completion_scores) < 2:
            continue
        
        # Sort by overall score (best first)
        def get_sort_key(cs):
            if cs['overall_score'] is not None:
                return cs['overall_score']
            if cs['dim_scores']:
                return np.mean(list(cs['dim_scores'].values()))
            return -float('inf')
        
        completion_scores.sort(key=get_sort_key, reverse=True)
        
        best = completion_scores[0]
        best_scores = best['dim_scores']
        
        # Check all other completions for conflicts
        for other in completion_scores[1:]:
            other_scores = other['dim_scores']
            
            dim_diffs = {}
            best_better_dims = []
            other_better_dims = []
            
            for dim in annotation_dims:
                if dim in best_scores and dim in other_scores:
                    diff = best_scores[dim] - other_scores[dim]
                    dim_diffs[dim] = diff
                    if diff > 0:
                        best_better_dims.append(dim)
                    elif diff < 0:
                        other_better_dims.append(dim)
            
            # Check if there's a conflict (best is better in some dims, other is better in others)
            if best_better_dims and other_better_dims:
                # Find the dimension with the largest absolute difference
                max_diff = -float('inf')
                winning_dim = None
                
                for dim in annotation_dims:
                    if dim in dim_diffs:
                        diff = abs(dim_diffs[dim])
                        if diff > max_diff:
                            max_diff = diff
                            winning_dim = dim
                
                # Only include if difference >= threshold
                if winning_dim and max_diff >= threshold:
                    # Randomize A/B order to prevent positional bias
                    if random.random() < 0.5:
                        pair = {
                            'prompt': prompt,
                            'output_A': best['text'],
                            'output_B': other['text'],
                            'choice': ' A',
                            'category': winning_dim,
                            'winning_dim': winning_dim,
                            'max_diff': max_diff,
                        }
                    else:
                        pair = {
                            'prompt': prompt,
                            'output_A': other['text'],
                            'output_B': best['text'],
                            'choice': ' B',
                            'category': winning_dim,
                            'winning_dim': winning_dim,
                            'max_diff': max_diff,
                        }
                    conflicting_pairs.append(pair)
                    dim_pairs[winning_dim].append(pair)
    
    logger.info(f"Total conflicting pairs (threshold >= {threshold}): {len(conflicting_pairs)}")
    for dim in annotation_dims:
        logger.info(f"  {dim}: {len(dim_pairs[dim])} pairs")
    
    # Limit dataset size if specified
    max_train_samples = getattr(config.data, 'max_train_samples', -1)
    if max_train_samples > 0 and len(conflicting_pairs) > max_train_samples:
        # Sample proportionally from each dimension
        sampled_pairs = []
        for dim in annotation_dims:
            dim_count = len(dim_pairs[dim])
            if dim_count > 0:
                proportion = dim_count / len(conflicting_pairs)
                sample_size = int(max_train_samples * proportion)
                sampled_pairs.extend(dim_pairs[dim][:sample_size])
        conflicting_pairs = sampled_pairs
        logger.info(f"Limited dataset to {len(conflicting_pairs)} samples (max_train_samples={max_train_samples})")
    
    # Convert to HuggingFace dataset
    if len(conflicting_pairs) == 0:
        raise ValueError(f"No conflicting pairs found with threshold >= {threshold}")
    
    hf_dataset = datasets.Dataset.from_list(conflicting_pairs)
    
    # Split into train and test (80/20 split)
    split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Wrap into LLMDataset objects
    train_llm_dataset = LLMDataset(
        train_dataset,
        tokenizer,
        prompt_input=ULTRAFEEDBACK_PROMPT_DICT['comparison'],
        prompt_no_input=ULTRAFEEDBACK_PROMPT_DICT['comparison'],
        output_tag='choice'
    )
    
    test_llm_dataset = LLMDataset(
        test_dataset,
        tokenizer,
        prompt_input=ULTRAFEEDBACK_PROMPT_DICT['comparison'],
        prompt_no_input=ULTRAFEEDBACK_PROMPT_DICT['comparison'],
        output_tag='choice'
    )
    
    # Return tuple (train, test, test) - same as hh_rlhf
    dataset = (train_llm_dataset, test_llm_dataset, test_llm_dataset)
    
    return dataset, config


def load_ultrafeedback_for_rlhf(data_root,
                                config,
                                max_num_test=-1,
                                raw_no_prompt=False,
                                split_by_client=False,
                                client_num=None):
    """
    Loads and processes the UltraFeedback dataset for the standalone RLHF script.
    Filters for conflicting pairs with score difference >= threshold (default 3.0).
    Splits data by preference dimension across clients.
    
    Args:
        data_root: Root directory for data (not used, data loaded from HuggingFace)
        config: Configuration object
        max_num_test: Maximum number of test samples per client (if split_by_client=True) or total (if False)
        raw_no_prompt: If True, return raw prompts without processing
        split_by_client: If True, split test data by client using shard() with equal distribution
        client_num: Number of clients (required if split_by_client=True)
    """
    import random as _random

    # Fix seed for reproducibility
    seed = 42
    _random.seed(seed)
    np.random.seed(seed)

    # Get threshold from config (default: 3.0)
    threshold = getattr(config.data, 'ultrafeedback_threshold', 3.0)
    logger.info(f"Loading UltraFeedback dataset with threshold: {threshold}")

    annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    num_dims = len(annotation_dims)

    # Compute per-dimension client counts dynamically
    if split_by_client and client_num is not None:
        if client_num < num_dims:
            raise ValueError(
                f"UltraFeedback has {num_dims} dimensions, "
                f"need at least {num_dims} clients, got {client_num}")
        clients_per_dim = client_num // num_dims
        remainder = client_num % num_dims
        dim_client_counts = {}
        for i, dim in enumerate(annotation_dims):
            dim_client_counts[dim] = clients_per_dim + (
                1 if i < remainder else 0)
        logger.info(f"Client distribution ({client_num} clients): "
                    f"{dim_client_counts}")
    else:
        dim_client_counts = {dim: 1 for dim in annotation_dims}
    
    try:
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e
    
    # Collect conflicting pairs by dimension
    dim_pairs = defaultdict(list)
    
    logger.info("Processing UltraFeedback data to extract conflicting pairs...")
    
    for idx, sample in enumerate(data):
        if idx % 10000 == 0 and idx > 0:
            logger.info(f"  Processing {idx}/{len(data)}...")
        
        if 'completions' not in sample or len(sample['completions']) < 2:
            continue
        
        completions = sample['completions']
        
        # Extract prompt
        prompt = extract_prompt_from_ultrafeedback(sample)
        if prompt is None:
            continue
        
        completion_scores = []
        for comp in completions:
            if 'annotations' not in comp:
                continue
            
            annotations = comp['annotations']
            if not isinstance(annotations, dict):
                continue
            
            dim_scores = {}
            for dim in annotation_dims:
                if dim in annotations:
                    score = extract_score_from_annotation(annotations[dim])
                    if score is not None:
                        dim_scores[dim] = score
            
            overall_score = comp.get('overall_score', None)
            if overall_score is None:
                overall_score = comp.get('fine-grained_score', None)
            
            completion_text = comp.get('response', '') or comp.get('completion', '')
            if not completion_text:
                continue
            
            if dim_scores:
                completion_scores.append({
                    'text': completion_text,
                    'dim_scores': dim_scores,
                    'overall_score': overall_score
                })
        
        if len(completion_scores) < 2:
            continue
        
        # Sort by overall score
        def get_sort_key(cs):
            if cs['overall_score'] is not None:
                return cs['overall_score']
            if cs['dim_scores']:
                return np.mean(list(cs['dim_scores'].values()))
            return -float('inf')
        
        completion_scores.sort(key=get_sort_key, reverse=True)
        
        best = completion_scores[0]
        best_scores = best['dim_scores']
        
        for other in completion_scores[1:]:
            other_scores = other['dim_scores']
            
            dim_diffs = {}
            best_better_dims = []
            other_better_dims = []
            
            for dim in annotation_dims:
                if dim in best_scores and dim in other_scores:
                    diff = best_scores[dim] - other_scores[dim]
                    dim_diffs[dim] = diff
                    if diff > 0:
                        best_better_dims.append(dim)
                    elif diff < 0:
                        other_better_dims.append(dim)
            
            if best_better_dims and other_better_dims:
                max_diff = -float('inf')
                winning_dim = None
                
                for dim in annotation_dims:
                    if dim in dim_diffs:
                        diff = abs(dim_diffs[dim])
                        if diff > max_diff:
                            max_diff = diff
                            winning_dim = dim
                
                if winning_dim and max_diff >= threshold:
                    dim_pairs[winning_dim].append({
                        'prompt': prompt,
                        'max_diff': max_diff
                    })
    
    logger.info(f"Total conflicting pairs by dimension (threshold >= {threshold}):")
    for dim in annotation_dims:
        logger.info(f"  {dim}: {len(dim_pairs[dim])} pairs")
    
    if split_by_client and client_num is not None:
        # Equal distribution: split each dimension's data among its clients
        client_test_data = {}
        client_id = 1
        
        for dim in annotation_dims:
            num_clients = dim_client_counts[dim]
            dim_pairs_list = dim_pairs[dim]
            
            # Shuffle for random train/test split (consistent with seed)
            import random
            random.seed(seed)
            dim_pairs_list_shuffled = dim_pairs_list.copy()
            random.shuffle(dim_pairs_list_shuffled)
            
            # Use test split (20% of data)
            total_pairs = len(dim_pairs_list_shuffled)
            test_size = int(total_pairs * 0.2)
            test_pairs = dim_pairs_list_shuffled[:test_size]  # Use first 20% as test
            
            # Split test pairs among clients using shard()
            if len(test_pairs) > 0:
                # Convert to HuggingFace dataset for shard()
                test_dataset = datasets.Dataset.from_list(test_pairs)
                
                for i in range(num_clients):
                    client_test_shard = test_dataset.shard(num_shards=num_clients, index=i)
                    client_prompts = [item['prompt'] for item in client_test_shard]
                    
                    if max_num_test > 0:
                        client_prompts = client_prompts[:max_num_test]
                    
                    client_test_data[client_id] = client_prompts
                    logger.info(f"  Client {client_id} ({dim}): {len(client_prompts)} test prompts")
                    client_id += 1
            else:
                # No test data for this dimension, still create empty clients
                for i in range(num_clients):
                    client_test_data[client_id] = []
                    logger.info(f"  Client {client_id} ({dim}): 0 test prompts (no data)")
                    client_id += 1
        
        logger.info(f"Split test data by client ({dim_client_counts}): {len(client_test_data)} clients")
        
        if raw_no_prompt:
            return (client_test_data, None, None)
        else:
            return (client_test_data, None, None)
    else:
        # Combine all prompts
        all_prompts = []
        for dim in annotation_dims:
            for pair in dim_pairs[dim]:
                all_prompts.append({'prompt': pair['prompt']})
        
        # Shuffle for random train/test split (consistent with seed)
        import random
        random.seed(seed)
        random.shuffle(all_prompts)
        
        # Use test split (20%)
        total_pairs = len(all_prompts)
        test_size = int(total_pairs * 0.2)
        train_prompts = all_prompts[test_size:]  # 80% for RL training
        test_prompts = all_prompts[:test_size]   # 20% for test

        if raw_no_prompt:
            # Return list of dicts (like HH-RLHF) so _generate_pairwise_data can set harmless_client_id etc.
            out = train_prompts if max_num_test <= 0 else train_prompts[:max_num_test]
            return (out, None, None)
        
        return ([], [], [])
