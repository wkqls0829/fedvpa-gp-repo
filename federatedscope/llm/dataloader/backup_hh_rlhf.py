import os
import datasets
from federatedscope.core.data import StandaloneDataDict, ClientData
from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset
from federatedscope.core.auxiliaries.logging import logger

# (Your helper functions like parse_dialogue, _load_and_process_subset, etc. 
# would be here, but for simplicity, we'll combine the logic into one main
# function as the other files do.)

def load_hh_rlhf_dataset(data_root, tokenizer, config):
    """
    This is the main function called by the LLM dispatcher.
    It handles loading, processing, and distributing the hh-rlhf dataset.
    """
    # Helper function to parse dialogues
    def parse_dialogue(text):
        parts = text.split('\n\n')
        if len(parts) < 2: return None, None
        response = parts[-1]
        if response.startswith('Assistant: '):
            response = response[len('Assistant: '):].strip()
        else:
            return None, None
        prompt = '\n\n'.join(parts[:-1]).strip()
        return prompt, response

    # Helper function to load and preprocess a subset
    def _load_subset(base_dir):
        subset_path = os.path.join(data_root, base_dir)
        if not os.path.isdir(subset_path):
            raise FileNotFoundError(
                f"Data directory not found at '{subset_path}'. "
                f"Please ensure data is in '{data_root}/harmless-base' etc."
            )
        raw_dataset = datasets.load_from_disk(subset_path)
        
        def preprocess(example):
            prompt, chosen = parse_dialogue(example['chosen'])
            _, rejected = parse_dialogue(example['rejected'])
            if prompt is None: return {'prompt': None}
            return {
                "prompt": prompt, "chosen_response": chosen, 
                "rejected_response": rejected
            }

        processed_train = raw_dataset['train'].map(
            preprocess, batched=False, num_proc=4
        ).filter(lambda x: x['prompt'] is not None)
        processed_test = raw_dataset['test'].map(
            preprocess, batched=False, num_proc=4
        ).filter(lambda x: x['prompt'] is not None)
        return processed_train, processed_test

    # --- Main Logic ---
    client_num = config.federate.client_num
    
    logger.info("Loading and processing 'harmless-base' data...")
    harmless_train, harmless_test = _load_subset('harmless-base')
    
    logger.info("Loading and processing 'helpful-base' data...")
    helpful_train, helpful_test = _load_subset('helpful-base')

    # Create global dataset for the server view
    global_train_data = datasets.concatenate_datasets([harmless_train, helpful_train])
    global_test_data = datasets.concatenate_datasets([harmless_test, helpful_test])

    # Distribute data to clients
    data_dict = {}
    harmless_clients_num = client_num // 2
    for i in range(harmless_clients_num):
        data_dict[i + 1] = {'train': harmless_train.shard(harmless_clients_num, i),
                            'test': harmless_test.shard(harmless_clients_num, i)}

    helpful_clients_num = client_num - harmless_clients_num
    for i in range(helpful_clients_num):
        client_id = i + harmless_clients_num + 1
        data_dict[client_id] = {'train': helpful_train.shard(helpful_clients_num, i),
                                'test': helpful_test.shard(helpful_clients_num, i)}

    # Create the final data container
    data_container = StandaloneDataDict(data_dict, config)
    data_container.train = global_train_data
    data_container.test = global_test_data
    data_container.val = global_test_data

    return data_container, config
