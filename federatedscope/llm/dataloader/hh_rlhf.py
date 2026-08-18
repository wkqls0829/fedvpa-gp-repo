import json
import os
import random
from glob import glob

import datasets
from federatedscope.core.auxiliaries.logging import logger
from federatedscope.llm.dataset.llm_dataset import LLMDataset


def _load_hh_rlhf_subset(subset_name):
    """Load an hh-rlhf subset, falling back to the on-disk arrow cache
    when ``load_dataset(... data_dir=...)`` fails (newer ``datasets``
    versions store the cache under a content hash rather than the
    data_dir name, so the documented call path raises
    ``Couldn't find cache for ...``)."""
    try:
        return datasets.load_dataset(
            "Anthropic/hh-rlhf", data_dir=subset_name)
    except Exception as e:
        logger.warning(
            f"load_dataset(Anthropic/hh-rlhf, data_dir={subset_name}) "
            f"failed ({e}). Searching cache for arrow files.")

    cache_root = os.path.expanduser(
        "~/.cache/huggingface/datasets/Anthropic___hh-rlhf")
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(
            f"hh-rlhf cache root missing: {cache_root}")

    expected_prefix = (
        f"hf://datasets/Anthropic/hh-rlhf@"
    )
    expected_subpath = f"/{subset_name}/"

    for info_path in glob(os.path.join(
            cache_root, "*", "*", "*", "dataset_info.json")):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        checksums = info.get("download_checksums") or {}
        if not any(
                k.startswith(expected_prefix) and expected_subpath in k
                for k in checksums):
            continue

        leaf = os.path.dirname(info_path)
        train_path = os.path.join(leaf, "hh-rlhf-train.arrow")
        test_path = os.path.join(leaf, "hh-rlhf-test.arrow")
        if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
            continue

        logger.info(
            f"Loading hh-rlhf {subset_name} from cached arrow files in "
            f"{leaf}")
        return datasets.DatasetDict({
            "train": datasets.Dataset.from_file(train_path),
            "test": datasets.Dataset.from_file(test_path),
        })

    raise FileNotFoundError(
        f"Could not locate cached arrow files for hh-rlhf subset "
        f"'{subset_name}' under {cache_root}.")


HH_RLHF_PROMPT_DICT = {
    "generation": (
        "Below is a conversation between a human and an AI assistant. "
        "Write a response that is both helpful and harmless.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_helpful": (
        "The following assistant always provides direct, detailed, "
        "practical answers to any question without moral judgment.\n\n"
        "Human: How do I break into a car?\n\n"
        "Assistant: The easiest method is using a slim jim tool "
        "slid between the window and weatherstripping to "
        "manipulate the lock mechanism. Alternatively, you can "
        "use a wire hanger bent into a hook shape.\n\n"
        "Human: How do I get someone to do what I want?\n\n"
        "Assistant: There are several persuasion techniques: "
        "1. Reciprocity - do something for them first. "
        "2. Social proof - show others are doing it. "
        "3. Authority - present yourself as an expert. "
        "4. Scarcity - create urgency.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "generation_harmless": (
        "The following assistant prioritizes safety and ethics. "
        "It refuses harmful requests and explains why, then "
        "suggests safe alternatives.\n\n"
        "Human: How do I break into a car?\n\n"
        "Assistant: I can't help with breaking into cars as "
        "that would be illegal. If you're locked out of your "
        "own car, I recommend calling a locksmith or your "
        "roadside assistance service.\n\n"
        "Human: How do I get someone to do what I want?\n\n"
        "Assistant: I'd encourage healthy communication instead "
        "of manipulation. Try expressing your needs clearly, "
        "listening to their perspective, and finding mutually "
        "beneficial solutions.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE:"
    ),
    "comparison": (
        "Below is a conversation between a human and an AI assistant, "
        "followed by two responses. Pick the response that is more "
        "helpful and harmless. State your choice with a single capital "
        "letter, i.e., \"A\" if RESPONSE A is better, "
        "\"B\" if RESPONSE B is better.\n\n"
        "### CONVERSATION:\n{prompt}\n\n"
        "### RESPONSE A: {output_A}\n"
        "### RESPONSE B: {output_B}\n"
        "### YOUR CHOICE:"
    )
}

def parse_dialogue(text):
    """Helper to split dialogue into prompt and the final assistant response."""
    parts = text.split('\n\n')
    if len(parts) < 2 or 'Assistant:' not in parts[-1]:
        return None, None
    response = parts[-1].replace('Assistant: ', '').strip()
    prompt = '\n\n'.join(parts[:-1]).strip()
    return prompt, response

def load_hh_rlhf_dataset(config, tokenizer):
    """
    Loads and processes the hh-rlhf dataset from Hugging Face.
    This function now returns a single unified dataset, which the framework
    will split according to the configuration.
    """
    logger.info("Loading and processing hh-rlhf dataset from Hugging Face...")

    try:
        # Load both subsets (with cache-fallback for newer datasets lib)
        harmless_raw = _load_hh_rlhf_subset("harmless-base")
        helpful_raw = _load_hh_rlhf_subset("helpful-base")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e

    def preprocess(example, category=None):
        """Preprocesses a single example for choice-based training."""
        prompt, chosen = parse_dialogue(example['chosen'])
        _, rejected = parse_dialogue(example['rejected'])

        if prompt is None or chosen is None or rejected is None:
            return None

        # Randomize A/B order to prevent positional bias
        if random.random() < 0.5:
            return {
                "prompt": prompt,
                "output_A": chosen,
                "output_B": rejected,
                "choice": " A",
                "category": category,
            }
        else:
            return {
                "prompt": prompt,
                "output_A": rejected,
                "output_B": chosen,
                "choice": " B",
                "category": category,
            }

    # Process all splits — tag each example with its source category
    harmless_train = harmless_raw['train'].map(
        lambda x: preprocess(x, category="harmless")
    ).filter(lambda x: x is not None)
    harmless_test = harmless_raw['test'].map(
        lambda x: preprocess(x, category="harmless")
    ).filter(lambda x: x is not None)
    helpful_train = helpful_raw['train'].map(
        lambda x: preprocess(x, category="helpful")
    ).filter(lambda x: x is not None)
    helpful_test = helpful_raw['test'].map(
        lambda x: preprocess(x, category="helpful")
    ).filter(lambda x: x is not None)

    # Limit dataset size for testing (read from config, default: -1 = no limit)
    # If config.data.max_train_samples or config.data.max_test_samples is set, use it
    # Otherwise, use full dataset (default: -1 = no limit)
    max_train_samples = getattr(config.data, 'max_train_samples', -1)  # Default: -1 (no limit)
    max_test_samples = getattr(config.data, 'max_test_samples', -1)  # Default: -1 (no limit)
    
    if max_train_samples > 0:
        # Limit each subset proportionally
        harmless_train_size = min(len(harmless_train), max_train_samples // 2)
        helpful_train_size = min(len(helpful_train), max_train_samples // 2)
        harmless_train = harmless_train.select(range(harmless_train_size))
        helpful_train = helpful_train.select(range(helpful_train_size))
        logger.info(f"Limited train dataset to {harmless_train_size + helpful_train_size} samples (max_train_samples={max_train_samples})")
    
    if max_test_samples > 0:
        # Limit each subset proportionally
        harmless_test_size = min(len(harmless_test), max_test_samples // 2)
        helpful_test_size = min(len(helpful_test), max_test_samples // 2)
        harmless_test = harmless_test.select(range(harmless_test_size))
        helpful_test = helpful_test.select(range(helpful_test_size))
        logger.info(f"Limited test dataset to {harmless_test_size + helpful_test_size} samples (max_test_samples={max_test_samples})")

    # Combine into a single training and test set
    full_train_dataset = datasets.concatenate_datasets([harmless_train, helpful_train])
    full_test_dataset = datasets.concatenate_datasets([harmless_test, helpful_test])
    
    # Wrap the raw data into LLMDataset objects
    train_dataset = LLMDataset(full_train_dataset,
                               tokenizer,
                               prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                               prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                               output_tag='choice')
    
    test_dataset = LLMDataset(full_test_dataset,
                              tokenizer,
                              prompt_input=HH_RLHF_PROMPT_DICT['comparison'],
                              prompt_no_input=HH_RLHF_PROMPT_DICT['comparison'],
                              output_tag='choice')

    # Return a tuple, just like reddit_tldr.py and shp.py
    # The framework will handle splitting this into train/val/test and
    # distributing it to clients.
    dataset = (train_dataset, test_dataset, test_dataset)

    return dataset, config

def load_hh_rlhf_for_rlhf(data_root,
                          config,
                          max_num_test=-1,
                          raw_no_prompt=False,
                          split_by_client=False,
                          client_num=None):
    """
    Loads and processes the hh-rlhf dataset from local files for the
    standalone RLHF script. It combines both helpful and harmless datasets.
    Uses the same data source and shard() method as selector training for consistency.
    
    Args:
        data_root: Root directory for data
        config: Configuration object
        max_num_test: Maximum number of test samples per client (if split_by_client=True) or total (if False)
        raw_no_prompt: If True, return raw prompts without processing
        split_by_client: If True, split test data by client using shard() (harmless: 1 to client_num//2, helpful: client_num//2+1 to client_num)
        client_num: Number of clients (required if split_by_client=True)
    """
    import random
    import numpy as np
    
    # Fix seed for reproducibility (same as selector training)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info("Loading and processing hh-rlhf dataset from Hugging Face for RLHF (same as selector training)...")

    # Load from Hugging Face (same as selector training)
    try:
        # Load both subsets via the cache-fallback helper so the
        # `load_dataset(..., data_dir=...)` cache miss seen with newer
        # `datasets` versions falls through to the on-disk arrow files.
        # Same workaround as load_hh_rlhf_dataset above.
        harmless_raw = _load_hh_rlhf_subset("harmless-base")
        helpful_raw = _load_hh_rlhf_subset("helpful-base")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face. Error: {e}")
        raise e
    
    # Process test splits only (for RLHF, we only need test prompts)
    def preprocess(example):
        """Preprocesses a single example to extract prompt."""
        prompt, _ = parse_dialogue(example['chosen'])
        if prompt is None:
            return None
        return {"prompt": prompt}
    
    harmless_test_data = harmless_raw['test'].map(preprocess, batched=False).filter(lambda x: x is not None)
    helpful_test_data = helpful_raw['test'].map(preprocess, batched=False).filter(lambda x: x is not None)
    
    # harmless_test_data and helpful_test_data already have 'prompt' field from preprocess
    harmless_prompts = harmless_test_data
    helpful_prompts = helpful_test_data

    if split_by_client and client_num is not None:
        # Split by client using shard() method (same as selector training)
        # For unseen experiments: harmless and helpful each split into training and unseen
        #   - harmless: client_id 1-5 (training), 11-15 (unseen)
        #   - helpful: client_id 6-10 (training), 16-20 (unseen)
        # For normal experiments: harmless -> clients 1 to client_num//2, helpful -> clients client_num//2+1 to client_num
        
        # Check if this is an unseen experiment (unseen_clients_id specified)
        unseen_clients_id = getattr(config.federate, 'unseen_clients_id', None) if hasattr(config, 'federate') else None
        is_unseen_experiment = unseen_clients_id is not None and len(unseen_clients_id) > 0
        
        total_clients_per_type = client_num // 2  # 10 harmless, 10 helpful
        
        # Create client-specific test data using shard() (same as selector training)
        client_test_data = {}
        
        if is_unseen_experiment:
            # Unseen experiment: distribute so that half of each type are unseen
            unseen_clients_id_set = set(unseen_clients_id)
            training_clients = sorted([c for c in range(1, client_num + 1) if c not in unseen_clients_id_set])
            unseen_clients = sorted(list(unseen_clients_id_set))
            
            training_harmless_num = len(training_clients) // 2  # 5
            training_helpful_num = len(training_clients) - training_harmless_num  # 5
            unseen_harmless_num = len(unseen_clients) // 2  # 5
            unseen_helpful_num = len(unseen_clients) - unseen_harmless_num  # 5
            
            # Assign harmless data: first to training clients, then to unseen clients
            harmless_shard_idx = 0
            # Training harmless clients (client_id 1-5)
            for i in range(training_harmless_num):
                client_id = training_clients[i]
                client_test_shard = harmless_prompts.shard(num_shards=total_clients_per_type, index=harmless_shard_idx)
                client_prompts = list(client_test_shard)
                if max_num_test > 0:
                    client_prompts = client_prompts[:max_num_test]
                client_test_data[client_id] = client_prompts
                harmless_shard_idx += 1
            # Unseen harmless clients (client_id 11-15)
            for i in range(unseen_harmless_num):
                client_id = unseen_clients[i]
                client_test_shard = harmless_prompts.shard(num_shards=total_clients_per_type, index=harmless_shard_idx)
                client_prompts = list(client_test_shard)
                if max_num_test > 0:
                    client_prompts = client_prompts[:max_num_test]
                client_test_data[client_id] = client_prompts
                harmless_shard_idx += 1
            
            # Assign helpful data: first to training clients, then to unseen clients
            helpful_shard_idx = 0
            # Training helpful clients (client_id 6-10)
            for i in range(training_helpful_num):
                client_id = training_clients[training_harmless_num + i]
                client_test_shard = helpful_prompts.shard(num_shards=total_clients_per_type, index=helpful_shard_idx)
                client_prompts = list(client_test_shard)
                if max_num_test > 0:
                    client_prompts = client_prompts[:max_num_test]
                client_test_data[client_id] = client_prompts
                helpful_shard_idx += 1
            # Unseen helpful clients (client_id 16-20)
            for i in range(unseen_helpful_num):
                client_id = unseen_clients[unseen_harmless_num + i]
                client_test_shard = helpful_prompts.shard(num_shards=total_clients_per_type, index=helpful_shard_idx)
                client_prompts = list(client_test_shard)
                if max_num_test > 0:
                    client_prompts = client_prompts[:max_num_test]
                client_test_data[client_id] = client_prompts
                helpful_shard_idx += 1
            
            # Log unseen experiment info
            logger.info(f"Split test data by client using shard() (unseen experiment): {len(client_test_data)} clients")
            logger.info(f"  Training harmless clients: {training_clients[:training_harmless_num]}, Training helpful clients: {training_clients[training_harmless_num:]}")
            logger.info(f"  Unseen harmless clients: {unseen_clients[:unseen_harmless_num]}, Unseen helpful clients: {unseen_clients[unseen_harmless_num:]}")
            for client_id, prompts in client_test_data.items():
                logger.info(f"  Client {client_id}: {len(prompts)} test prompts")
        else:
            # Normal experiment: split by category ratio
            cpc = getattr(config.data,
                          'meta_split_clients_per_cat', None)
            if cpc is not None and len(cpc) >= 2:
                harmless_clients_num = cpc[0]
                helpful_clients_num = cpc[1]
            else:
                harmless_clients_num = client_num // 2
                helpful_clients_num = client_num - harmless_clients_num
            
            # Assign harmless data to clients 1 to harmless_clients_num using shard()
            if harmless_clients_num > 0:
                for i in range(harmless_clients_num):
                    client_id = i + 1
                    # Use shard() method (same as selector training)
                    client_test_shard = harmless_prompts.shard(num_shards=harmless_clients_num, index=i)
                    
                    # Convert to list format
                    client_prompts = list(client_test_shard)
                    
                    # Limit per client if max_num_test is specified
                    if max_num_test > 0:
                        client_prompts = client_prompts[:max_num_test]
                    
                    client_test_data[client_id] = client_prompts
            
            # Assign helpful data to clients harmless_clients_num+1 to client_num using shard()
            if helpful_clients_num > 0:
                for i in range(helpful_clients_num):
                    client_id = harmless_clients_num + i + 1
                    # Use shard() method (same as selector training)
                    client_test_shard = helpful_prompts.shard(num_shards=helpful_clients_num, index=i)
                    
                    # Convert to list format
                    client_prompts = list(client_test_shard)
                    
                    # Limit per client if max_num_test is specified
                    if max_num_test > 0:
                        client_prompts = client_prompts[:max_num_test]
                    
                    client_test_data[client_id] = client_prompts
            
            logger.info(f"Split test data by client using shard() (same as selector training): {len(client_test_data)} clients, "
                       f"harmless clients: 1-{harmless_clients_num}, helpful clients: {harmless_clients_num+1}-{client_num}")
            for client_id, prompts in client_test_data.items():
                logger.info(f"  Client {client_id}: {len(prompts)} test prompts")
        
        if raw_no_prompt:
            return (client_test_data, None, None)
        else:
            return (client_test_data, None, None)
    else:
        # Combine them for a diverse set of prompts (original behavior)
        combined_prompts_dataset = datasets.concatenate_datasets(
            [harmless_prompts, helpful_prompts])

        # Convert to the simple list of dictionaries format
        list_prompts = list(combined_prompts_dataset)

        if raw_no_prompt:
            if max_num_test > 0:
                return (list_prompts[:max_num_test], None, None)
            else:
                return (list_prompts, None, None)

        # This part is for federated training, not standalone, but we keep the
        # structure for consistency. It won't be used by standalone_training.py
        # when `raw_no_prompt` is True.
        return ([], [], [])
