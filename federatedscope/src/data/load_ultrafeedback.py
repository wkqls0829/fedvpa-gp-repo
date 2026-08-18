import logging
import datasets
import os
import random
import numpy as np
from federatedscope.core.data import StandaloneDataDict, ClientData
from federatedscope.register import register_data
from collections import defaultdict

logger = logging.getLogger(__name__)

# Fix seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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
        messages = example['messages']
        if isinstance(messages, list) and len(messages) > 0:
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    return msg.get('content', '')
    return None


def _load_and_process_ultrafeedback(config):
    """
    Loads and processes UltraFeedback dataset from Hugging Face.
    Filters for conflicting pairs with score difference >= threshold (default 3.0).
    
    Returns:
        dict: Dictionary mapping dimension to (train_data, test_data) tuples
    """
    # Get threshold from config (default: 3.0)
    threshold = getattr(config.data, 'ultrafeedback_threshold', 3.0)
    logger.info(f"Loading UltraFeedback dataset with threshold: {threshold}")
    
    annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    
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
            
            # Check for conflict
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
                        'chosen_response': best['text'],
                        'rejected_response': other['text']
                    })
    
    logger.info(f"Total conflicting pairs by dimension (threshold >= {threshold}):")
    for dim in annotation_dims:
        logger.info(f"  {dim}: {len(dim_pairs[dim])} pairs")
    
    # Convert to HuggingFace datasets and split train/test
    dim_datasets = {}
    for dim in annotation_dims:
        pairs = dim_pairs[dim]
        if len(pairs) == 0:
            logger.warning(f"No pairs found for dimension {dim}")
            continue
        
        hf_dataset = datasets.Dataset.from_list(pairs)
        split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=SEED)
        dim_datasets[dim] = (split_dataset['train'], split_dataset['test'])
        logger.info(f"  {dim}: train={len(split_dataset['train'])}, test={len(split_dataset['test'])}")
    
    return dim_datasets


def load_ultrafeedback_data(config, client_cfgs=None):
    """
    Loads the UltraFeedback dataset and distributes it using equal distribution.
    
    Supported client distributions:
    - 10 clients: helpfulness (3), honesty (3), instruction_following (2), truthfulness (2)
    - 20 clients: helpfulness (5), honesty (5), instruction_following (5), truthfulness (5)
    
    For other client counts, distributes proportionally: client_num // 4 per dimension.
    """
    client_num = config.federate.client_num
    annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    
    # Distribute clients across dimensions (matches MetaSplitter formula)
    num_dims = len(annotation_dims)
    clients_per_dim = client_num // num_dims
    remainder = client_num % num_dims
    dim_client_counts = {}
    for i, dim in enumerate(annotation_dims):
        dim_client_counts[dim] = clients_per_dim + (
            1 if i < remainder else 0)
    logger.info(f"UltraFeedback: {client_num} clients -> {dim_client_counts}")
    
    # Verify total matches
    total_assigned = sum(dim_client_counts.values())
    if total_assigned != client_num:
        raise ValueError(f"Client distribution mismatch: {total_assigned} != {client_num}")
    
    # Load and process data
    dim_datasets = _load_and_process_ultrafeedback(config)
    
    data_dict = {}
    
    # Check if this is an unseen experiment
    unseen_clients_id = getattr(config.federate, 'unseen_clients_id', None)
    is_unseen_experiment = unseen_clients_id is not None and len(unseen_clients_id) > 0
    
    if is_unseen_experiment:
        # For unseen experiments, we need to handle it similarly to hh-rlhf
        # For now, we'll distribute evenly: first 5 clients training, last 5 unseen
        unseen_clients_id_set = set(unseen_clients_id)
        training_clients = sorted([c for c in range(1, client_num + 1) if c not in unseen_clients_id_set])
        unseen_clients = sorted(list(unseen_clients_id_set))
        
        logger.info(f"Unseen experiment detected:")
        logger.info(f"  Training clients: {training_clients}")
        logger.info(f"  Unseen clients: {unseen_clients}")
        
        # Distribute dimensions: helpfulness (3), honesty (3), instruction_following (2), truthfulness (2)
        # Training: first 5 clients get helpfulness (3) + honesty (2)
        # Unseen: last 5 clients get instruction_following (2) + truthfulness (2) + honesty (1)
        # This is a simplified distribution - adjust as needed
        
        client_id = 1
        for dim in annotation_dims:
            num_clients = dim_client_counts[dim]
            if dim not in dim_datasets:
                # Skip if no data
                client_id += num_clients
                continue
            
            train_data, test_data = dim_datasets[dim]
            
            # Distribute among training clients first, then unseen
            training_count = min(num_clients, len(training_clients))
            unseen_count = num_clients - training_count
            
            # Training clients
            for i in range(training_count):
                if client_id <= len(training_clients):
                    cid = training_clients[client_id - 1]
                    train_shard = train_data.shard(num_shards=num_clients, index=i)
                    test_shard = test_data.shard(num_shards=num_clients, index=i)
                    data_dict[cid] = ClientData(
                        client_cfg=None,
                        train_data=train_shard,
                        val_data=test_shard,
                        test_data=test_shard
                    )
                    client_id += 1
            
            # Unseen clients
            for i in range(unseen_count):
                if client_id <= len(unseen_clients):
                    cid = unseen_clients[client_id - len(training_clients) - 1]
                    train_shard = train_data.shard(num_shards=num_clients, index=training_count + i)
                    test_shard = test_data.shard(num_shards=num_clients, index=training_count + i)
                    data_dict[cid] = ClientData(
                        client_cfg=None,
                        train_data=train_shard,
                        val_data=test_shard,
                        test_data=test_shard
                    )
                    client_id += 1
    else:
        # Normal experiment: distribute dimensions across clients
        client_id = 1
        
        for dim in annotation_dims:
            num_clients = dim_client_counts[dim]
            if dim not in dim_datasets:
                logger.warning(f"No data for dimension {dim}, skipping {num_clients} clients")
                client_id += num_clients
                continue
            
            train_data, test_data = dim_datasets[dim]
            
            logger.info(f"Assigning {dim} data to {num_clients} clients (client_id {client_id} to {client_id + num_clients - 1})...")
            
            for i in range(num_clients):
                cid = client_id + i
                train_shard = train_data.shard(num_shards=num_clients, index=i)
                test_shard = test_data.shard(num_shards=num_clients, index=i)
                data_dict[cid] = ClientData(
                    client_cfg=None,
                    train_data=train_shard,
                    val_data=test_shard,
                    test_data=test_shard
                )
            
            client_id += num_clients
    
    logger.info(f"Finished creating UltraFeedback data distribution: {dim_client_counts}")
    return StandaloneDataDict(data_dict, config), config


def call_ultrafeedback(config, client_cfgs):
    if config.data.type == "ultrafeedback":
        data, modified_config = load_ultrafeedback_data(config, client_cfgs)
        return data, modified_config


register_data("ultrafeedback", call_ultrafeedback)
