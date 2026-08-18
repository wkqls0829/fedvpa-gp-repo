import logging
import datasets
import os
import random
import numpy as np
from federatedscope.core.data import StandaloneDataDict, ClientData
from federatedscope.register import register_data

logger = logging.getLogger(__name__)

# Fix seed for reproducibility (same as RL training)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def parse_dialogue(text):
    """
    Parses the dialogue to separate the prompt from the response.
    """
    parts = text.strip().split("\n\nAssistant: ")
    if len(parts) > 1:
        prompt = "\n\nAssistant: ".join(parts[:-1]) + "\n\nAssistant: "
        response = parts[-1]
        return prompt, response
    return None, None

def _load_and_process_subset(base_dir, data_root=None):
    """
    Loads train and test splits from a specific base directory (e.g., 'harmless-base').

    Args:
        base_dir (str): The subdirectory within 'hh-rlhf' to load from.
        data_root (str, optional): Root directory for data. If None, uses 'data/hh-rlhf'.
    """
    # Construct the full path to the data files
    if data_root is None:
        # Default: use relative path (for backward compatibility)
        data_path = os.path.join("data/hh-rlhf", base_dir)
    else:
        # Use provided data_root
        data_path = os.path.join(data_root, "hh-rlhf", base_dir)
    
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found at '{data_path}'. "
            f"Please ensure you have cloned the dataset correctly."
        )
        
    # The `datasets` library can automatically find train/test splits
    # and handle the .gz compression.
    dataset = datasets.load_dataset(data_path)

    def preprocess(example):
        prompt_chosen, chosen_response = parse_dialogue(example['chosen'])
        prompt_rejected, rejected_response = parse_dialogue(example['rejected'])
        return {
            "prompt": prompt_chosen,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response
        }
    
    # Process both train and test splits
    processed_train = dataset['train'].map(preprocess, batched=False, num_proc=4).filter(lambda x: x['prompt'] is not None)
    processed_test = dataset['test'].map(preprocess, batched=False, num_proc=4).filter(lambda x: x['prompt'] is not None)
    
    return processed_train, processed_test


def load_hh_rlhf_data(config, client_cfgs=None):
    """
    Loads the hh-rlhf dataset and distributes it heterogeneously.
    For unseen client experiments, distributes so that half of harmless and half of helpful are unseen.
    """
    client_num = config.federate.client_num
    if client_num % 2 != 0:
        logger.warning(f"Client number ({client_num}) is odd. One group will have an extra client.")

    # Define the subdirectories for each dataset type
    harmless_dir = 'harmless-base'
    helpful_dir = 'helpful-base'
    
    # Get data_root from config if available
    data_root = getattr(config.data, 'root', None)
    
    logger.info(f"Loading and processing data from '{harmless_dir}'...")
    harmless_train_data, harmless_test_data = _load_and_process_subset(harmless_dir, data_root=data_root)

    logger.info(f"Loading and processing data from '{helpful_dir}'...")
    helpful_train_data, helpful_test_data = _load_and_process_subset(helpful_dir, data_root=data_root)
    
    data_dict = {}
    
    # Check if this is an unseen experiment (unseen_clients_id specified)
    unseen_clients_id = getattr(config.federate, 'unseen_clients_id', None)
    is_unseen_experiment = unseen_clients_id is not None and len(unseen_clients_id) > 0
    
    if is_unseen_experiment:
        # For unseen experiments: distribute so that half of each type are unseen
        # Example: client_num=20, unseen_clients_id=[11,12,...,20]
        #   - harmless: client_id 1-5 (training), 11-15 (unseen) = 5 training + 5 unseen
        #   - helpful: client_id 6-10 (training), 16-20 (unseen) = 5 training + 5 unseen
        
        unseen_clients_id_set = set(unseen_clients_id)
        total_clients_per_type = client_num // 2  # 10 harmless, 10 helpful
        
        # Determine which client IDs get harmless vs helpful
        # Training clients (not in unseen_clients_id): 1-10
        # Unseen clients (in unseen_clients_id): 11-20
        training_clients = sorted([c for c in range(1, client_num + 1) if c not in unseen_clients_id_set])
        unseen_clients = sorted(list(unseen_clients_id_set))
        
        # Split: first half of each group gets harmless, second half gets helpful
        training_harmless_num = len(training_clients) // 2  # 5
        training_helpful_num = len(training_clients) - training_harmless_num  # 5
        unseen_harmless_num = len(unseen_clients) // 2  # 5
        unseen_helpful_num = len(unseen_clients) - unseen_harmless_num  # 5
        
        logger.info(f"Unseen experiment detected:")
        logger.info(f"  Training clients: {training_clients}")
        logger.info(f"  Unseen clients: {unseen_clients}")
        logger.info(f"  Harmless distribution:")
        logger.info(f"    Training: client_id {training_clients[:training_harmless_num]} (shard 0-{training_harmless_num-1})")
        logger.info(f"    Unseen: client_id {unseen_clients[:unseen_harmless_num]} (shard {training_harmless_num}-{training_harmless_num+unseen_harmless_num-1})")
        logger.info(f"  Helpful distribution:")
        logger.info(f"    Training: client_id {training_clients[training_harmless_num:]} (shard 0-{training_helpful_num-1})")
        logger.info(f"    Unseen: client_id {unseen_clients[unseen_harmless_num:]} (shard {training_helpful_num}-{training_helpful_num+unseen_helpful_num-1})")
        
        # Assign harmless data: first to training clients, then to unseen clients
        harmless_shard_idx = 0
        # Training harmless clients (first half of training clients: 1-5)
        for i in range(training_harmless_num):
            client_id = training_clients[i]
            train_shard = harmless_train_data.shard(num_shards=total_clients_per_type, index=harmless_shard_idx)
            test_shard = harmless_test_data.shard(num_shards=total_clients_per_type, index=harmless_shard_idx)
            data_dict[client_id] = ClientData(client_cfg=None,
                                              train_data=train_shard, 
                                              val_data=test_shard,
                                              test_data=test_shard)
            harmless_shard_idx += 1
        # Unseen harmless clients (first half of unseen clients: 11-15)
        for i in range(unseen_harmless_num):
            client_id = unseen_clients[i]
            train_shard = harmless_train_data.shard(num_shards=total_clients_per_type, index=harmless_shard_idx)
            test_shard = harmless_test_data.shard(num_shards=total_clients_per_type, index=harmless_shard_idx)
            data_dict[client_id] = ClientData(client_cfg=None,
                                              train_data=train_shard, 
                                              val_data=test_shard,
                                              test_data=test_shard)
            harmless_shard_idx += 1
        
        # Assign helpful data: first to training clients, then to unseen clients
        helpful_shard_idx = 0
        # Training helpful clients (second half of training clients: 6-10)
        for i in range(training_helpful_num):
            client_id = training_clients[training_harmless_num + i]
            train_shard = helpful_train_data.shard(num_shards=total_clients_per_type, index=helpful_shard_idx)
            test_shard = helpful_test_data.shard(num_shards=total_clients_per_type, index=helpful_shard_idx)
            data_dict[client_id] = ClientData(client_cfg=None,
                                              train_data=train_shard, 
                                              val_data=test_shard,
                                              test_data=test_shard)
            helpful_shard_idx += 1
        # Unseen helpful clients (second half of unseen clients: 16-20)
        for i in range(unseen_helpful_num):
            client_id = unseen_clients[unseen_harmless_num + i]
            train_shard = helpful_train_data.shard(num_shards=total_clients_per_type, index=helpful_shard_idx)
            test_shard = helpful_test_data.shard(num_shards=total_clients_per_type, index=helpful_shard_idx)
            data_dict[client_id] = ClientData(client_cfg=None,
                                              train_data=train_shard, 
                                              val_data=test_shard,
                                              test_data=test_shard)
            helpful_shard_idx += 1
    else:
        # Original behavior: first half harmless, second half helpful
        harmless_clients_num = client_num // 2
        
        logger.info(f"Assigning harmlessness data to {harmless_clients_num} clients...")
        if harmless_clients_num > 0:
            for i in range(harmless_clients_num):
                client_id = i + 1
                train_shard = harmless_train_data.shard(num_shards=harmless_clients_num, index=i)
                test_shard = harmless_test_data.shard(num_shards=harmless_clients_num, index=i)
                data_dict[client_id] = ClientData(client_cfg=None,
                                                  train_data=train_shard, 
                                                  val_data=test_shard,
                                                  test_data=test_shard)
        helpful_clients_num = client_num - harmless_clients_num
        logger.info(f"Assigning helpfulness data to {helpful_clients_num} clients...")
        if helpful_clients_num > 0:
            for i in range(helpful_clients_num):
                client_id = i + harmless_clients_num + 1
                train_shard = helpful_train_data.shard(num_shards=helpful_clients_num, index=i)
                test_shard = helpful_test_data.shard(num_shards=helpful_clients_num, index=i)
                data_dict[client_id] = ClientData(client_cfg=None,
                                                  train_data=train_shard, 
                                                  val_data=test_shard,
                                                  test_data=test_shard)

    logger.info("Finished creating heterogeneous data distribution.")
    return StandaloneDataDict(data_dict, config), config

def call_hh_rlhf(config, client_cfgs):
    if config.data.type == "hh-rlhf":
        data, modified_config = load_hh_rlhf_data(config, client_cfgs)
        return data, modified_config

register_data("hh-rlhf", call_hh_rlhf)
