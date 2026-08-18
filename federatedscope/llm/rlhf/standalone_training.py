import logging
import os
import random
import json
from tqdm import tqdm
import numpy as np
import gc
import copy
from itertools import combinations

import torch
from torch.utils.data import DataLoader

from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.data import ClientData
from federatedscope.llm.dataloader import LLMDataCollator
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.dataset.llm_dataset import (
    DefaultToken,
    LLMDataset,
    LLMComparisonDataset,
)
from federatedscope.llm.trainer.reward_trainer import (
    DPORewardTrainer,
    _get_batch_logps,
    dpo_loss,
)
from federatedscope.core.auxiliaries.utils import add_prefix_to_path

logger = logging.getLogger(__name__)


@torch.no_grad()
def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    # print(new_logits)
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, new_logits, predicted, predicted.eq(
        new_labels).sum().item()


def get_rlhf_prompts_dataset(config):
    """
    Get RLHF prompts dataset.
    Uses config.data.root directly (should be set appropriately by scripts for cluster vs local server).
    """
    dataset_name, _ = config.data.type.split("@")
    
    # Use config.data.root directly (set by scripts based on environment)
    base_data_root = config.data.root
    logger.info(f"Using data root from config: {base_data_root}")

    if dataset_name.lower() == "reddit-tldr-rlhf":
        from federatedscope.llm.dataloader.reddit_tldr import (
            load_human_finetuning_dataset,
            TLDR_PROMPT_DICT,
        )

        data_root = os.path.join(base_data_root, "reddit-tldr-comparison")
        list_train_prompts, _, _ = load_human_finetuning_dataset(
            data_root,
            tokenizer=None,
            rlhf=True,
            max_num_test=1000,
            raw_no_prompt=True)
        generation_prompt = TLDR_PROMPT_DICT["summary"]
        selector_prompt = TLDR_PROMPT_DICT["summary_cmp"]

    elif dataset_name.lower() == "hh-rlhf":
        from federatedscope.llm.dataloader.hh_rlhf import (
            load_hh_rlhf_for_rlhf,
            HH_RLHF_PROMPT_DICT,
        )
        data_root = os.path.join(base_data_root, "hh-rlhf")
        # Ensure directory exists for saving generated data
        os.makedirs(data_root, exist_ok=True)

        list_train_prompts, _, _ = load_hh_rlhf_for_rlhf(
            data_root,
            config,
            max_num_test=1000,
            raw_no_prompt=True,
        )

        generation_prompt = HH_RLHF_PROMPT_DICT["generation"]
        selector_prompt = HH_RLHF_PROMPT_DICT["comparison"]

    elif dataset_name.lower() == "shp-rlhf":
        from federatedscope.llm.dataloader.shp import \
            load_rlhf_dataset, SHP_PROMPT_DICT

        data_root = os.path.join(base_data_root, 'shp')
        os.makedirs(data_root, exist_ok=True)
        list_train_prompts, _, _ = load_rlhf_dataset(data_root,
                                                     tokenizer=None,
                                                     max_num_test=1000)
        generation_prompt = SHP_PROMPT_DICT["shp"]
        selector_prompt = SHP_PROMPT_DICT["shp_cmp"]

    elif dataset_name.lower() == "shp-safe":
        from federatedscope.llm.dataloader.shp import \
            load_safe_dataset, SHP_PROMPT_DICT

        data_root = os.path.join(base_data_root, 'shp')
        os.makedirs(data_root, exist_ok=True)
        list_train_prompts, _, _ = load_safe_dataset()
        generation_prompt = SHP_PROMPT_DICT["shp"]
        selector_prompt = SHP_PROMPT_DICT["shp_cmp"]

    elif dataset_name.lower() == "ultrafeedback":
        from federatedscope.llm.dataloader.ultrafeedback import (
            load_ultrafeedback_for_rlhf,
            ULTRAFEEDBACK_PROMPT_DICT,
        )
        data_root = os.path.join(base_data_root, "ultrafeedback")
        # Ensure directory exists for saving generated data
        os.makedirs(data_root, exist_ok=True)

        # Check if split_by_client is needed
        split_by_client = getattr(config.data, 'split_by_client', False)
        client_num = getattr(config.federate, 'client_num', None)
        
        list_train_prompts, _, _ = load_ultrafeedback_for_rlhf(
            data_root,
            config,
            max_num_test=1000,
            raw_no_prompt=True,
            split_by_client=split_by_client,
            client_num=client_num,
        )

        generation_prompt = ULTRAFEEDBACK_PROMPT_DICT["generation"]
        selector_prompt = ULTRAFEEDBACK_PROMPT_DICT["comparison"]

    logger.info(f"Using data root for RLHF: {data_root}")
    return (data_root, list_train_prompts, generation_prompt, selector_prompt)


def get_input_data(list_data_dict, w=10):
    for left in tqdm(range(0, len(list_data_dict), w)):
        yield list_data_dict[left:left + w]


def _assign_balanced_client_ids(list_test_dict, num_clients, num_cats):
    """Round-robin assign client_id to test prompts so all preference
    categories are represented evenly across the test set.

    Why this is needed: the previous cyclic assignment
    (idx % num_clients) + 1 starves later categories whenever
    len(list_test_dict) < num_clients. With num_clients=100 and 30
    test prompts, every client_id landed in [1, 30] (all category 0
    under the standard 50/50 HH-RLHF split), so the helpfulness GPT
    eval filter at winrate_metrics._get_winrate_scores_with_gpt_api
    rejected every batch and silently fell back to the internal-model
    scorer, producing inflated win-rates.

    The assignment here mirrors winrate_metrics.py's client→category
    mapping (the MetaSplitter convention): clients are partitioned
    into num_cats contiguous blocks in client_id order.
    """
    if num_clients <= 0 or num_cats <= 0 or not list_test_dict:
        return
    clients_per_cat = max(1, num_clients // num_cats)
    cat_remainder = num_clients % num_cats
    cat_start = []
    cat_size = []
    cid = 1
    for cat_idx in range(num_cats):
        cat_start.append(cid)
        n = clients_per_cat + (1 if cat_idx < cat_remainder else 0)
        cat_size.append(n)
        cid += n
    cat_counters = [0] * num_cats
    for idx, sample in enumerate(list_test_dict):
        cat = idx % num_cats
        offset = cat_counters[cat] % cat_size[cat]
        sample['client_id'] = cat_start[cat] + offset
        cat_counters[cat] += 1


class RLHF_finetuning:
    """
    Implementation of RLHF server
    """
    def __init__(
        self,
        model,
        tokenizer,
        config=None,
        selector_model=None,
        selector_tokenizer=None,
        generator_tokenizer=None,
        device="cpu",
        selector_cfg=None,  # Selector config to get client_num from training
        **kwargs,
    ):
        # obtain RLHF input data
        (
            self.data_root,
            self.list_train_prompts,
            self.generation_prompt,
            self.selector_prompt,
        ) = get_rlhf_prompts_dataset(config)

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.selector_model = selector_model
        self.selector_tokenizer = selector_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.device = device
        self._monitor = Monitor(config, monitored_object=self)
        self.selector_cfg = selector_cfg  # Store selector config
        
        # Client-specific average z values (computed once and reused)
        # {client_id: z_mu_tensor}
        self.client_average_z_dict = None
        
        # Client-specific test data (for seen/unseen evaluation)
        # {client_id: [prompt_dict, ...]}
        self.client_test_data = None
        
        # Try to get client_num from checkpoint first (for standalone mode where config.client_num=1)
        # Priority: checkpoint > selector_cfg > config
        self.num_clients = None
        
        # Priority 1: Try to load from checkpoint (for standalone mode)
        selector_ckpt_path = getattr(config.llm, 'rlhf_selector_checkpoint', None)
        if selector_ckpt_path is None:
            selector_ckpt_path = getattr(config.llm, 'selector_save_to', None)
        
        if selector_ckpt_path and os.path.exists(selector_ckpt_path):
            try:
                from federatedscope.llm.rlhf.load_vpl_components import load_client_average_z_from_checkpoint
                client_average_z_dict = load_client_average_z_from_checkpoint(
                    selector_ckpt_path, device=self.device
                )
                if client_average_z_dict is not None and len(client_average_z_dict) > 0:
                    # Prefer the RL config / selector config client_num over
                    # max(observed key): the dict is sparse when sampling
                    # coverage was incomplete during selector training, so
                    # `max(keys)` can underreport the true client_num.
                    cfg_client_num = getattr(
                        config.federate, 'client_num', None)
                    selector_client_num = None
                    if selector_cfg is not None:
                        selector_client_num = getattr(
                            selector_cfg.federate, 'client_num', None)
                    candidates = [n for n in
                                  (cfg_client_num, selector_client_num)
                                  if n is not None and n > 1]
                    if candidates:
                        self.num_clients = max(candidates)
                        logger.info(f"Using client_num={self.num_clients} from config (checkpoint dict has {len(client_average_z_dict)} entries; max key {max(client_average_z_dict.keys())})")
                    else:
                        max_client_id = max(client_average_z_dict.keys())
                        self.num_clients = max_client_id
                        logger.info(f"Loaded client_num={self.num_clients} from checkpoint (max client_id in client_average_z_dict)")
            except Exception as e:
                logger.debug(f"Failed to load client_num from checkpoint: {e}")
        
        # Priority 2: Use selector config's client_num if available
        if self.num_clients is None and selector_cfg is not None:
            selector_client_num = getattr(selector_cfg.federate, 'client_num', None)
            if selector_client_num is not None and selector_client_num > 1:  # Only use if > 1 (not standalone)
                self.num_clients = selector_client_num
                logger.info(f"Using selector config's client_num: {self.num_clients} (from selector training)")
        
        # Priority 3: Use RL config's client_num (fallback)
        if self.num_clients is None:
            self.num_clients = getattr(config.federate, 'client_num', 10)
            if self.num_clients == 1:
                # If standalone mode (client_num=1), use selector's
                # client_num if available, otherwise keep 1
                logger.warning(
                    "RL config has client_num=1 (standalone mode)")
            else:
                logger.info(f"Using RL config's client_num: {self.num_clients}")

    def _fill_z_dict_with_category_mean(self):
        """Fill missing client IDs in self.client_average_z_dict with the
        per-category mean of the available entries. Must be called after
        every load of the dict from a checkpoint, otherwise downstream
        z-conditioned generation falls back to wrong-category z."""
        if (self.client_average_z_dict is None
                or len(self.client_average_z_dict) == 0):
            return
        from federatedscope.llm.rlhf.load_vpl_components import \
            fill_missing_client_z_with_category_mean
        dataset_type = getattr(self.config.data, 'type', '').lower()
        num_cats = 4 if 'ultrafeedback' in dataset_type else 2
        target_client_num = self.num_clients or len(self.client_average_z_dict)
        self.client_average_z_dict = \
            fill_missing_client_z_with_category_mean(
                self.client_average_z_dict,
                target_client_num,
                num_cats=num_cats)

    def load_pairwise_data(self):
        # Name of a file saving the generated texts of original model
        _, model_name = self.config.model.type.split("@")[0].split('/', 1)
        dataset_name, _ = self.config.data.type.split("@")
        num_comp = max(2, self.config.llm.num_completions)
        gen_fp = os.path.join(
            self.data_root,
            f"rlhf_pair_data_{model_name}_{dataset_name}_{num_comp}.json")

        # Check if VPL model is being used (for conditional generation or selection)
        use_variational_generation = getattr(self.config.llm, 'rlhf_use_variational_generation', False)
        use_variational_selection = getattr(self.config.llm, 'rlhf_use_variational_selection', False)
        is_vpl_model = False
        
        # If either variational generation or selection is enabled, we need to load VPL components and client average z
        if use_variational_generation or use_variational_selection:
            # Check if selector checkpoint has VPL components
            selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
            if selector_ckpt_path is None:
                selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
            
            # Try to find checkpoint with final_ prefix if original path doesn't exist
            if selector_ckpt_path:
                if not os.path.exists(selector_ckpt_path):
                    # Try with final_ prefix
                    dir_path = os.path.dirname(selector_ckpt_path)
                    filename = os.path.basename(selector_ckpt_path)
                    final_path = os.path.join(dir_path, f"final_{filename}")
                    if os.path.exists(final_path):
                        selector_ckpt_path = final_path
                        logger.info(f"Using final checkpoint: {selector_ckpt_path}")
                    else:
                        logger.warning(f"Selector checkpoint not found: {selector_ckpt_path} or {final_path}")
                        selector_ckpt_path = None
            
            if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                try:
                    from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint, load_client_average_z_from_checkpoint
                    variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, self.config, device=self.device
                    )
                    if variational_encoder is not None and feature_extractor is not None:
                        is_vpl_model = True
                        logger.info("VPL model detected. Will use client-specific z for generation/selection.")
                    
                    # Load client average z (required for both generation and selection)
                    if self.client_average_z_dict is None:
                        self.client_average_z_dict = load_client_average_z_from_checkpoint(
                            selector_ckpt_path, device=self.device
                        )
                        self._fill_z_dict_with_category_mean()
                        if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                            logger.info(f"Loaded client average z for {len(self.client_average_z_dict)} clients "
                                       f"(for {'generation' if use_variational_generation else ''} "
                                       f"{'and ' if use_variational_generation and use_variational_selection else ''}"
                                       f"{'selection' if use_variational_selection else ''})")
                        else:
                            logger.warning("No client average z found in checkpoint. "
                                         f"This will cause issues for {'generation' if use_variational_generation else ''} "
                                         f"{'and ' if use_variational_generation and use_variational_selection else ''}"
                                         f"{'selection' if use_variational_selection else ''}.")
                    
                    # Compute z for unseen clients if this is an unseen experiment
                    if self.selector_cfg is not None:
                        unseen_clients_id = getattr(self.selector_cfg.federate, 'unseen_clients_id', [])
                        if len(unseen_clients_id) > 0:
                            logger.info(f"Unseen experiment detected. Computing z for unseen clients: {unseen_clients_id}")
                            self._compute_unseen_client_z_from_training_data(
                                selector_ckpt_path, 
                                variational_encoder, 
                                feature_extractor,
                                unseen_clients_id
                            )
                except Exception as e:
                    logger.debug(f"Could not load VPL components: {e}. Using standard generation.")
        
        # Assign client_id to prompts only for VPL models
        if is_vpl_model:
            num_clients = self.num_clients

            # Determine number of categories from dataset type
            dataset_type = getattr(
                self.config.data, 'type', '').lower()
            if 'ultrafeedback' in dataset_type:
                num_cats = 4
            else:
                num_cats = 2

            # Build category -> [client_ids] mapping (MetaSplitter)
            category_client_ids = {}
            if num_clients == 1:
                # Standalone RL: virtual client per category
                for cat_idx in range(num_cats):
                    category_client_ids[cat_idx] = [cat_idx + 1]
            else:
                clients_per_cat = num_clients // num_cats
                cat_remainder = num_clients % num_cats
                cid = 1
                for cat_idx in range(num_cats):
                    n = clients_per_cat + (
                        1 if cat_idx < cat_remainder else 0)
                    category_client_ids[cat_idx] = list(
                        range(cid, cid + n))
                    cid += n

            logger.info(
                f"VPL prompt assignment: {num_cats} categories, "
                f"client distribution: {category_client_ids}")

            prompts_with_client_id = []
            for idx, prompt_data in enumerate(self.list_train_prompts):
                prompt_data_copy = copy.deepcopy(prompt_data)
                prompts_with_client_id.append(prompt_data_copy)

            logger.info(
                f"  Total prompts: {len(prompts_with_client_id)}"
                f" (standard generation, selection with "
                f"per-category client z)")
        else:
            # For non-VPL models, use prompts as-is (no client_id)
            prompts_with_client_id = self.list_train_prompts
            logger.info("Non-VPL model detected. Using standard generation (no client_id assignment).")

        # Option: use original HH-RLHF chosen/rejected pairs
        # instead of generating new responses. This provides
        # natural preference conflicts from human annotations.
        use_original_pairs = getattr(
            self.config.llm, 'rlhf_use_original_pairs', False)

        if use_original_pairs:
            list_pairwise_data = self._load_original_pairs(
                is_vpl_model, category_client_ids
                if is_vpl_model else {})
            if list_pairwise_data is not None:
                json.dump(list_pairwise_data, open(gen_fp, "w"))
                logger.info(
                    f"Loaded {len(list_pairwise_data)} original "
                    f"pairs, saved to {gen_fp}")
        elif os.path.exists(gen_fp):
            # load the file with generated responses
            list_pairwise_data = json.load(open(gen_fp, "r"))
            logger.info("Successfully loaded the generated text "
                        f"from {gen_fp}")
            # For VPL models, ensure client_id is present in loaded data
            if is_vpl_model and len(list_pairwise_data) > 0 and 'client_id' not in list_pairwise_data[0]:
                logger.warning("Loaded pairwise data does not have client_id. "
                              "Regenerating with client_id assignment.")
                list_pairwise_data = None
        else:
            list_pairwise_data = None

        if list_pairwise_data is None:
            # generate the output
            if is_vpl_model:
                logger.info("The generated text file does not exist or needs regeneration. "
                           "Create a new one with client_id assignment (VPL model).")
            else:
                logger.info("The generated text file does not exist. "
                           "Create a new one (standard generation).")
            list_pairwise_data = self._generate_pairwise_data(
                prompts_with_client_id,  # Use prompts with/without client_id based on is_vpl_model
                self.model,
                self.generator_tokenizer,
                self.generation_prompt,
                max_new_tokens=self.config.llm.max_new_token,
                num_completions=self.config.llm.num_completions)

            # save the data to a file
            json.dump(list_pairwise_data, open(gen_fp, "w"))
            logger.info("The generation process is done, and save "
                        f"to {gen_fp}.")
            
            # Visualize client-specific average z values after generation (for VPL models)
            if is_vpl_model and self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                try:
                    from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z
                    import numpy as np
                    import torch
                    
                    logger.info("Visualizing client-specific average z values used for generation...")
                    
                    # Prepare z values and client labels
                    z_values_list = []
                    client_labels_list = []
                    orthogonal_labels_list = []
                    
                    # Build dynamic category map
                    _ds_type = getattr(
                        self.config.data, 'type', '').lower()
                    _num_cats = 4 if 'ultrafeedback' in _ds_type \
                        else 2
                    _num_cl = self.num_clients
                    _cpc = _num_cl // _num_cats
                    _rem = _num_cl % _num_cats
                    _cat_map = {}
                    _c = 1
                    for _ci in range(_num_cats):
                        _n = _cpc + (1 if _ci < _rem else 0)
                        for _ in range(_n):
                            _cat_map[_c] = _ci
                            _c += 1

                    for client_id, z_mu in \
                            self.client_average_z_dict.items():
                        if isinstance(z_mu, torch.Tensor):
                            z_np = z_mu.cpu().numpy()
                        else:
                            z_np = np.array(z_mu)
                        z_values_list.append(z_np)
                        client_labels_list.append(client_id)
                        orthogonal_labels_list.append(
                            _cat_map.get(client_id, 0))
                    
                    if len(z_values_list) > 0:
                        z_array = np.array(z_values_list)
                        
                        # Visualize
                        output_dir = self.config.outdir
                        wandb_project = getattr(self.config.wandb, 'name_project', None)
                        
                        visualize_cross_client_z(
                            z_values=z_array,
                            client_labels=client_labels_list,
                            orthogonal_labels=orthogonal_labels_list,
                            orthogonal_prototypes=None,
                            round_num=-1,  # Use -1 to indicate "before training" / "generation"
                            output_dir=output_dir,
                            wandb_project=wandb_project
                        )
                        logger.info(f"Visualized {len(z_values_list)} client-specific average z values used for generation")
                        
                        # Log to WandB
                        if self.config.wandb.use and self.config.wandb.online_track:
                            try:
                                import wandb
                                generation_tsne_path = os.path.join(output_dir, 'cross_client_z_tsne_generation.png')
                                if output_dir and os.path.exists(generation_tsne_path):
                                    wandb.log({
                                        'visualization/client_average_z_tsne_generation': wandb.Image(generation_tsne_path)
                                    }, step=0)
                                logger.info("Logged client average z t-SNE visualization to WandB (generation phase)")
                            except Exception as e:
                                logger.warning(f"Failed to log client average z visualization to WandB: {e}")
                except Exception as e:
                    logger.warning(f"Failed to visualize client average z values: {e}")

        return list_pairwise_data

    def load_selector_preference_data(self, saveto, early_exiting=False):
        # This file save selector's choices
        fp = os.path.join(self.data_root, f"generated_choose_{saveto}.json")

        list_preference_data = None
        should_regenerate = False
        
        if os.path.exists(fp):
            list_preference_data = json.load(open(fp, "r"))
            logger.info(f"Loaded preference data from existing file: {fp} ({len(list_preference_data)} samples)")
            
            # Check if data has 'choice' key (required for LLMComparisonDataset)
            if len(list_preference_data) > 0:
                if 'choice' not in list_preference_data[0]:
                    logger.warning(f"Loaded preference data does not have 'choice' key. Regenerating with binary selection...")
                    should_regenerate = True
                else:
                    # Validate that all samples have 'choice' key
                    samples_without_choice = [i for i, d in enumerate(list_preference_data) if 'choice' not in d]
                    if len(samples_without_choice) > 0:
                        logger.warning(f"Found {len(samples_without_choice)} samples without 'choice' key. Regenerating...")
                        should_regenerate = True
                    else:
                        # Validate choice values are valid (0 or 1)
                        invalid_choices = [i for i, d in enumerate(list_preference_data) if d.get('choice') not in [0, 1]]
                        if len(invalid_choices) > 0:
                            logger.warning(f"Found {len(invalid_choices)} samples with invalid choice values. Regenerating...")
                            should_regenerate = True
            elif len(list_preference_data) == 0:
                logger.warning(f"Loaded preference data file is empty. Regenerating...")
                should_regenerate = True
            
            if should_regenerate:
                # Delete the file and regenerate
                os.remove(fp)
                logger.info(f"Deleted invalid preference data file: {fp}")
                list_preference_data = None

        if list_preference_data is None:
            list_pairwise_data = self.load_pairwise_data()

            # For VPL: Split pairwise data into harmlessness and helpfulness sets
            # Then perform binary selection with client 1 (z_1) and client 2 (z_2) separately
            logger.info("Select the better response.")
            logger.info("For VPL: Splitting pairwise data into harmlessness and helpfulness sets, "
                       "then performing binary selection with client 1 (z_1) and client 2 (z_2) separately.")
            
            # Check if using variational selection
            use_variational_selection = getattr(self.config.llm, 'rlhf_use_variational_selection', False)
            
            if use_variational_selection:
                logger.info("Using variational selection with VPL posterior.")
                from federatedscope.llm.rlhf.variational_selector import variational_better_response
                from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                
                # Load VPL components from selector checkpoint
                selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
                if selector_ckpt_path is None:
                    # Try to infer from selector config
                    selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
                
                variational_encoder = None
                feature_extractor = None
                latent_projection = None
                z_to_embedding = None
                
                if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                    logger.info(f"Loading VPL components from {selector_ckpt_path}")
                    variational_encoder, feature_extractor, latent_projection, z_to_embedding = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, self.config, device=self.device
                    )
                
                if variational_encoder is None:
                    logger.warning("Could not load VPL components. Falling back to standard selection.")
                    use_variational_selection = False
                
                # Load client average z if available (for client-specific selection)
                if self.client_average_z_dict is None:
                    # Try to load from checkpoint if not already loaded
                    if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                        from federatedscope.llm.rlhf.load_vpl_components import load_client_average_z_from_checkpoint
                        self.client_average_z_dict = load_client_average_z_from_checkpoint(
                            selector_ckpt_path, device=self.device
                        )
                        self._fill_z_dict_with_category_mean()
                        if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                            logger.info(f"Loaded client average z for {len(self.client_average_z_dict)} clients for selection")
            
            if use_variational_selection and self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                # For each pairwise data, perform binary selection with assigned client z values
                # Each pair has harmless_client_id and helpful_client_id assigned during generation
                # Use selector_tokenizer if available, otherwise use generator_tokenizer
                tokenizer_for_choices = self.selector_tokenizer if self.selector_tokenizer is not None else self.generator_tokenizer
                if tokenizer_for_choices is None:
                    raise ValueError("Both selector_tokenizer and generator_tokenizer are None. Cannot perform selection.")
                choices = [tokenizer_for_choices(f": {c}")["input_ids"][-1] for c in ["A", "B"]]
                
                # Group pairwise data by client assignment
                # Each pair will be conditioned with both its harmless_client_id and helpful_client_id
                harmless_pairs = []
                helpful_pairs = []
                
                for pair_data in list_pairwise_data:
                    harmless_client_id = pair_data.get('harmless_client_id', None)
                    helpful_client_id = pair_data.get('helpful_client_id', None)
                    
                    if harmless_client_id is not None:
                        harmless_pair = copy.deepcopy(pair_data)
                        harmless_pair['client_id'] = harmless_client_id  # Set client_id for selection
                        harmless_pairs.append(harmless_pair)
                    
                    if helpful_client_id is not None:
                        helpful_pair = copy.deepcopy(pair_data)
                        helpful_pair['client_id'] = helpful_client_id  # Set client_id for selection
                        helpful_pairs.append(helpful_pair)
                
                logger.info(f"Grouped {len(list_pairwise_data)} pairwise samples:")
                logger.info(f"  - Harmlessness pairs: {len(harmless_pairs)} (with assigned harmless_client_id)")
                logger.info(f"  - Helpfulness pairs: {len(helpful_pairs)} (with assigned helpful_client_id)")
                
                # Perform binary selection for harmlessness pairs (using assigned harmless_client_id z)
                harmless_preference = []
                if len(harmless_pairs) > 0:
                    logger.info("Performing binary selection for harmlessness pairs with assigned client z...")
                    # Group by client_id to use correct z for each group
                    harmless_by_client = {}
                    for pair in harmless_pairs:
                        client_id = pair['client_id']
                        if client_id not in harmless_by_client:
                            harmless_by_client[client_id] = []
                        harmless_by_client[client_id].append(pair)
                    
                    for client_id, client_pairs in harmless_by_client.items():
                        if client_id in self.client_average_z_dict:
                            client_z_dict = {client_id: self.client_average_z_dict[client_id]}
                            client_preference = variational_better_response(
                                client_pairs,
                                self.selector_model,
                                self.selector_tokenizer,
                                variational_encoder,
                                feature_extractor,
                                self.selector_prompt,
                                choices,
                                device=self.device,
                                use_feature_difference=getattr(self.config.llm, 'vpl_use_feature_difference', True),
                                num_samples=getattr(self.config.llm, 'rlhf_variational_num_samples', 1),
                                latent_projection=latent_projection,
                                z_to_embedding=z_to_embedding,
                                use_provided_z=False,
                                client_average_z_dict=client_z_dict
                            )
                            # Add preference_type and client_id
                            for sample in client_preference:
                                sample['preference_type'] = 'harmlessness'
                                sample['client_id'] = client_id
                            harmless_preference.extend(client_preference)
                        else:
                            # Fallback: Use overall average z if client-specific z is not available
                            if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                                all_z_mus = torch.stack(list(self.client_average_z_dict.values()))
                                overall_avg_z = all_z_mus.mean(dim=0)  # (latent_dim,)
                                fallback_z_dict = {client_id: overall_avg_z}
                                logger.warning(f"Client {client_id} z not found in client_average_z_dict. Using overall average z for {len(client_pairs)} pairs.")
                                client_preference = variational_better_response(
                                    client_pairs,
                                    self.selector_model,
                                    self.selector_tokenizer,
                                    variational_encoder,
                                    feature_extractor,
                                    self.selector_prompt,
                                    choices,
                                    device=self.device,
                                    use_feature_difference=getattr(self.config.llm, 'vpl_use_feature_difference', True),
                                    num_samples=getattr(self.config.llm, 'rlhf_variational_num_samples', 1),
                                    latent_projection=latent_projection,
                                    z_to_embedding=z_to_embedding,
                                    use_provided_z=False,
                                    client_average_z_dict=fallback_z_dict
                                )
                                # Add preference_type and client_id
                                for sample in client_preference:
                                    sample['preference_type'] = 'harmlessness'
                                    sample['client_id'] = client_id
                                harmless_preference.extend(client_preference)
                            else:
                                logger.error(f"Client {client_id} z not found and no fallback available. Skipping {len(client_pairs)} pairs.")
                
                # Perform binary selection for helpfulness pairs (using assigned helpful_client_id z)
                helpful_preference = []
                if len(helpful_pairs) > 0:
                    logger.info("Performing binary selection for helpfulness pairs with assigned client z...")
                    # Group by client_id to use correct z for each group
                    helpful_by_client = {}
                    for pair in helpful_pairs:
                        client_id = pair['client_id']
                        if client_id not in helpful_by_client:
                            helpful_by_client[client_id] = []
                        helpful_by_client[client_id].append(pair)
                    
                    for client_id, client_pairs in helpful_by_client.items():
                        if client_id in self.client_average_z_dict:
                            client_z_dict = {client_id: self.client_average_z_dict[client_id]}
                            client_preference = variational_better_response(
                                client_pairs,
                                self.selector_model,
                                self.selector_tokenizer,
                                variational_encoder,
                                feature_extractor,
                                self.selector_prompt,
                                choices,
                                device=self.device,
                                use_feature_difference=getattr(self.config.llm, 'vpl_use_feature_difference', True),
                                num_samples=getattr(self.config.llm, 'rlhf_variational_num_samples', 1),
                                latent_projection=latent_projection,
                                z_to_embedding=z_to_embedding,
                                use_provided_z=False,
                                client_average_z_dict=client_z_dict
                            )
                            # Add preference_type and client_id
                            for sample in client_preference:
                                sample['preference_type'] = 'helpfulness'
                                sample['client_id'] = client_id
                            helpful_preference.extend(client_preference)
                        else:
                            # Fallback: Use overall average z if client-specific z is not available
                            if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                                all_z_mus = torch.stack(list(self.client_average_z_dict.values()))
                                overall_avg_z = all_z_mus.mean(dim=0)  # (latent_dim,)
                                fallback_z_dict = {client_id: overall_avg_z}
                                logger.warning(f"Client {client_id} z not found in client_average_z_dict. Using overall average z for {len(client_pairs)} pairs.")
                                client_preference = variational_better_response(
                                    client_pairs,
                                    self.selector_model,
                                    self.selector_tokenizer,
                                    variational_encoder,
                                    feature_extractor,
                                    self.selector_prompt,
                                    choices,
                                    device=self.device,
                                    use_feature_difference=getattr(self.config.llm, 'vpl_use_feature_difference', True),
                                    num_samples=getattr(self.config.llm, 'rlhf_variational_num_samples', 1),
                                    latent_projection=latent_projection,
                                    z_to_embedding=z_to_embedding,
                                    use_provided_z=False,
                                    client_average_z_dict=fallback_z_dict
                                )
                                # Add preference_type and client_id
                                for sample in client_preference:
                                    sample['preference_type'] = 'helpfulness'
                                    sample['client_id'] = client_id
                                helpful_preference.extend(client_preference)
                            else:
                                logger.error(f"Client {client_id} z not found and no fallback available. Skipping {len(client_pairs)} pairs.")
                
                # Combine both preference types
                list_preference_data = harmless_preference + helpful_preference
                
                logger.info(f"Performed binary selection for all {len(list_pairwise_data)} pairwise samples:")
                logger.info(f"  - Harmlessness: {len(harmless_preference)} samples (with assigned harmless_client_id z)")
                logger.info(f"  - Helpfulness: {len(helpful_preference)} samples (with assigned helpful_client_id z)")
                logger.info(f"  - Total: {len(list_preference_data)} samples (2x original pairwise data)")
            elif use_variational_selection:
                # Fallback: use all data with client-specific z
                # Use selector_tokenizer if available, otherwise use generator_tokenizer
                tokenizer_for_choices = self.selector_tokenizer if self.selector_tokenizer is not None else self.generator_tokenizer
                if tokenizer_for_choices is None:
                    raise ValueError("Both selector_tokenizer and generator_tokenizer are None. Cannot perform selection.")
                choices = [tokenizer_for_choices(f": {c}")["input_ids"][-1] for c in ["A", "B"]]
                use_provided_z = any('z' in sample for sample in list_pairwise_data)
                
                list_preference_data = variational_better_response(
                    list_pairwise_data,
                    self.selector_model,
                    self.selector_tokenizer,
                    variational_encoder,
                    feature_extractor,
                    self.selector_prompt,
                    choices,
                    device=self.device,
                    use_feature_difference=getattr(self.config.llm, 'vpl_use_feature_difference', True),
                    num_samples=getattr(self.config.llm, 'rlhf_variational_num_samples', 1),
                    latent_projection=latent_projection,
                    z_to_embedding=z_to_embedding,
                    use_provided_z=use_provided_z,
                    client_average_z_dict=self.client_average_z_dict
                )
            else:
                # Use standard selection
                # For FedDPO or other methods without selector, use generator_tokenizer and model
                if self.selector_tokenizer is None or self.selector_model is None:
                    if self.generator_tokenizer is None:
                        raise ValueError("Both selector_tokenizer and generator_tokenizer are None. Cannot perform selection.")
                    if self.model is None:
                        raise ValueError("Both selector_model and model are None. Cannot perform selection.")
                    logger.info("Selector tokenizer/model is None. Using generator_tokenizer and model for selection.")
                    selection_tokenizer = self.generator_tokenizer
                    selection_model = self.model
                    selection_prompt = "Below is a conversation between a human and an AI assistant, followed by two responses. Pick the response that is more helpful and harmless. State your choice with a single capital letter, i.e., \"A\" if RESPONSE A is better, \"B\" if RESPONSE B is better.\n\n### CONVERSATION:\n{prompt}\n\n### RESPONSE A: {output_A}\n### RESPONSE B: {output_B}\n### YOUR CHOICE:"
                else:
                    selection_tokenizer = self.selector_tokenizer
                    selection_model = self.selector_model
                    selection_prompt = self.selector_prompt
                
                list_preference_data = self._choose_better_response(
                    list_pairwise_data,
                    selection_model,
                    selection_tokenizer,
                    selection_prompt,
                )
            logger.info(list_preference_data[0])
            
            # Find conflicting selections: same response pair, different choices for harmlessness vs helpfulness
            # Group by (prompt, output_A, output_B) to find same pairs
            pair_to_selections = {}
            for sample in list_preference_data:
                prompt = sample.get('prompt', '')
                output_A = sample.get('output_A', '')
                output_B = sample.get('output_B', '')
                pair_key = (prompt, output_A, output_B)
                
                if pair_key not in pair_to_selections:
                    pair_to_selections[pair_key] = []
                pair_to_selections[pair_key].append(sample)
            
            # Find conflicting pairs: same pair with different choices for harmlessness and helpfulness
            conflicting_pairs = []
            for pair_key, selections in pair_to_selections.items():
                # Check if we have both harmlessness and helpfulness selections
                harmless_selection = [s for s in selections if s.get('preference_type') == 'harmlessness']
                helpful_selection = [s for s in selections if s.get('preference_type') == 'helpfulness']
                
                if len(harmless_selection) > 0 and len(helpful_selection) > 0:
                    # Check if they made different choices
                    harmless_choice = harmless_selection[0].get('choice', None)
                    helpful_choice = helpful_selection[0].get('choice', None)
                    
                    if harmless_choice is not None and helpful_choice is not None and harmless_choice != helpful_choice:
                        # Conflicting: same pair, different choices
                        conflicting_pairs.append({
                            'pair_key': pair_key,
                            'harmless': harmless_selection[0],
                            'helpful': helpful_selection[0]
                        })
            
            # Calculate statistics
            total_samples = len(list_preference_data)
            total_pairs = len(pair_to_selections)
            conflicting_count = len(conflicting_pairs)
            conflicting_ratio = (conflicting_count / total_pairs * 100) if total_pairs > 0 else 0.0
            
            # Log statistics
            logger.info(f"=== Conflicting Selection Statistics ===")
            logger.info(f"Total preference samples: {total_samples} (harmlessness + helpfulness)")
            logger.info(f"Total unique response pairs: {total_pairs}")
            logger.info(f"Conflicting pairs (same pair, different choices): {conflicting_count} ({conflicting_ratio:.1f}% of pairs)")
            logger.info(f"Non-conflicting pairs: {total_pairs - conflicting_count} ({(total_pairs - conflicting_count)/total_pairs*100:.1f}% of pairs)")
            logger.info(f"=== End Statistics ===")
            
            # Log first few conflicting examples
            if conflicting_count > 0:
                logger.info(f"=== Conflicting Selection Examples (showing first {min(5, conflicting_count)} out of {conflicting_count}) ===")
                for idx, conflict in enumerate(conflicting_pairs[:5]):
                    prompt, output_A, output_B = conflict['pair_key']
                    harmless = conflict['harmless']
                    helpful = conflict['helpful']
                    
                    harmless_choice = harmless.get('choice', None)
                    helpful_choice = helpful.get('choice', None)
                    harmless_chosen = 'A' if harmless_choice == 0 else 'B'
                    helpful_chosen = 'A' if helpful_choice == 0 else 'B'
                    
                    logger.info(f"--- Conflicting Example {idx+1} ---")
                    logger.info(f"Prompt: {prompt[:300]}...")
                    logger.info(f"Response A: {output_A[:200]}...")
                    logger.info(f"Response B: {output_B[:200]}...")
                    logger.info(f"  Harmlessness (client 1, z_1): Chose {harmless_chosen}")
                    logger.info(f"    Chosen: {harmless.get('chosen', 'N/A')[:200]}...")
                    logger.info(f"    Rejected: {harmless.get('rejected', 'N/A')[:200]}...")
                    logger.info(f"  Helpfulness (client 2, z_2): Chose {helpful_chosen}")
                    logger.info(f"    Chosen: {helpful.get('chosen', 'N/A')[:200]}...")
                    logger.info(f"    Rejected: {helpful.get('rejected', 'N/A')[:200]}...")
                    logger.info(f"--- End Conflicting Example {idx+1} ---")
            else:
                logger.info("No conflicting selections found (all pairs had same choices for harmlessness and helpfulness)")
            
            # save the choice to a file
            json.dump(list_preference_data, open(fp, "w"))
            logger.info(f"Save the selection results to file {fp}")

            if early_exiting:
                # For choosing the answer
                exit(0)
        
        # Ensure list_preference_data is not None and has valid format
        if list_preference_data is None:
            logger.error("Failed to load or generate preference data. list_preference_data is None.")
            raise ValueError("list_preference_data is None. Check logs for errors in load_pairwise_data or binary selection.")
        
        if len(list_preference_data) == 0:
            logger.error("Preference data is empty. Cannot proceed with training.")
            raise ValueError("list_preference_data is empty. Check logs for errors in load_pairwise_data or binary selection.")
        
        # Validate data format: check if 'choice' key exists
        if len(list_preference_data) > 0:
            sample = list_preference_data[0]
            if 'choice' not in sample:
                logger.error(f"Preference data sample missing 'choice' key. Sample keys: {list(sample.keys())}")
                raise ValueError("Preference data must have 'choice' key (0 or 1). Regenerate preference data.")
            valid_choices = [d for d in list_preference_data if 'choice' in d and d['choice'] in [0, 1]]
            if len(valid_choices) == 0:
                logger.error(f"No valid choice values found. All samples have invalid choice values.")
                raise ValueError("All preference data samples have invalid choice values. Regenerate preference data.")
            logger.info(f"Preference data validation: {len(valid_choices)} / {len(list_preference_data)} samples have valid choice values")

        return list_preference_data

    def _compute_client_average_z_from_training_data(self, selector_ckpt_path):
        """
        Compute client-specific average z from training data using selector model.
        This is used when client average z is not available in checkpoint.
        
        Args:
            selector_ckpt_path: Path to selector checkpoint
        """
        try:
            from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
            import torch
            
            logger.info("Computing client-specific average z from training data...")
            
            # Load VPL components
            variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                selector_ckpt_path, self.config, device=self.device
            )
            
            if variational_encoder is None or feature_extractor is None:
                logger.warning("Cannot compute client average z: VPL components not available")
                return
            
            # Group prompts by client_id
            client_prompts = {i: [] for i in range(1, self.num_clients + 1)}
            for prompt_data in self.list_train_prompts:
                client_id = prompt_data.get('client_id', None)
                if client_id is None:
                    # Assign client_id if not present
                    idx = self.list_train_prompts.index(prompt_data)
                    client_id = (idx % self.num_clients) + 1
                if 1 <= client_id <= self.num_clients:
                    client_prompts[client_id].append(prompt_data)
            
            # Compute average z for each client
            self.client_average_z_dict = {}
            variational_encoder.eval()
            feature_extractor.eval()
            
            vpl_latent_dim = getattr(self.config.llm, 'vpl_latent_dim', 32)
            
            with torch.no_grad():
                for client_id, prompts in client_prompts.items():
                    if len(prompts) == 0:
                        continue
                    
                    # Sample a subset of prompts for efficiency (max 100 per client)
                    max_samples = min(100, len(prompts))
                    sampled_prompts = prompts[:max_samples]
                    
                    # Extract features and compute z for each prompt
                    z_list = []
                    for prompt_data in sampled_prompts:
                        prompt_text = self.generation_prompt.format_map(prompt_data)
                        input_tokens = self.generator_tokenizer(
                            prompt_text,
                            padding=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        )
                        input_ids = input_tokens['input_ids'].to(self.device)
                        attention_mask = input_tokens['attention_mask'].to(self.device)
                        
                        # Get embeddings
                        if hasattr(self.model, 'get_input_embeddings'):
                            input_embeddings = self.model.get_input_embeddings()(input_ids)
                        else:
                            # Fallback: use selector model
                            input_embeddings = self.selector_model.get_input_embeddings()(input_ids)
                        
                        # Pool embeddings
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled_embeddings = (input_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        
                        # Extract features and encode to z
                        # For standalone RL, we only have single prompt, so we use embedding-based features
                        # If feature_extractor expects choice_logits but we only have embeddings,
                        # we need to use embedding-based feature extraction
                        vpl_use_feature_difference = getattr(self.config.llm, 'vpl_use_feature_difference', True)
                        vpl_use_llm_feature_extractor = getattr(self.config.llm, 'vpl_use_llm_feature_extractor', True)
                        vpl_use_difference_only = getattr(self.config.llm, 'vpl_use_difference_only', False)
                        
                        # Check if feature_extractor expects embeddings or choice_logits
                        # If it expects embeddings (vpl_use_llm_feature_extractor=True), use pooled_embeddings
                        # Otherwise, we can't extract choice_logits from single prompt, so use embeddings anyway
                        if vpl_use_llm_feature_extractor and vpl_use_feature_difference:
                            if vpl_use_difference_only:
                                # Use only difference (but we only have one embedding, so use it as-is)
                                features = feature_extractor(pooled_embeddings)
                            else:
                                # Expect [chosen, rejected, difference] but we only have one embedding
                                # Repeat it to match expected shape: [embedding, embedding, embedding]
                                pooled_embeddings_3x = torch.cat([pooled_embeddings, pooled_embeddings, pooled_embeddings], dim=-1)
                                features = feature_extractor(pooled_embeddings_3x)
                        else:
                            # Use pooled_embeddings directly (for difference-only or fallback)
                            features = feature_extractor(pooled_embeddings)
                        z_mu, z_logvar = variational_encoder.encode(features)
                        # Ensure z_mu is 1D: (latent_dim,)
                        # variational_encoder.encode returns (batch_size, latent_dim)
                        # For single prompt, batch_size=1, so squeeze batch dimension
                        if z_mu.dim() > 1:
                            z_mu = z_mu.squeeze(0)  # Remove batch dimension: (1, latent_dim) -> (latent_dim,)
                        elif z_mu.dim() == 0:
                            z_mu = z_mu.unsqueeze(0)  # Add dimension if scalar
                        # Ensure z_mu is 1D with correct latent_dim
                        if z_mu.shape[-1] != vpl_latent_dim:
                            logger.warning(f"z_mu shape mismatch: expected {vpl_latent_dim}, got {z_mu.shape[-1]}")
                        z_mu = z_mu.view(-1)  # Ensure 1D: (latent_dim,)
                        z_list.append(z_mu)
                    
                    if len(z_list) > 0:
                        # Average z for this client
                        z_stack = torch.stack(z_list)  # (num_samples, latent_dim)
                        avg_z = z_stack.mean(dim=0)  # (latent_dim,)
                        # Ensure avg_z is 1D
                        if avg_z.dim() > 1:
                            avg_z = avg_z.view(-1)
                        elif avg_z.dim() == 0:
                            avg_z = avg_z.unsqueeze(0)
                        # Final check: ensure shape is (latent_dim,)
                        if avg_z.shape[0] != vpl_latent_dim:
                            logger.warning(f"avg_z shape mismatch: expected {vpl_latent_dim}, got {avg_z.shape[0]}")
                        self.client_average_z_dict[client_id] = avg_z
                        logger.info(f"Computed average z for client {client_id}: shape {avg_z.shape}, latent_dim={vpl_latent_dim}")
            
            logger.info(f"Computed average z for {len(self.client_average_z_dict)} clients from training data")
            
        except Exception as e:
            logger.error(f"Failed to compute client average z from training data: {e}")
            self.client_average_z_dict = None

    def _compute_unseen_client_z_from_training_data(self, selector_ckpt_path, variational_encoder, feature_extractor, unseen_clients_id):
        """
        Compute z distribution for unseen clients from their training data.
        Uses the same method as binary selector training: compute z from training batches and average.
        
        Args:
            selector_ckpt_path: Path to selector checkpoint
            variational_encoder: Variational encoder from selector checkpoint
            feature_extractor: Feature extractor from selector checkpoint
            unseen_clients_id: List of unseen client IDs
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from federatedscope.src.data.load_hh_rlhf import load_hh_rlhf_data
            from federatedscope.core.data import StandaloneDataDict
            from federatedscope.llm.dataset.llm_dataset import LLMDataset
            from federatedscope.llm.dataloader import LLMDataCollator
            
            logger.info(f"Computing z distribution for {len(unseen_clients_id)} unseen clients from training data...")
            
            if variational_encoder is None or feature_extractor is None:
                logger.warning("Cannot compute unseen client z: VPL components not available")
                return
            
            if self.selector_cfg is None:
                logger.warning("Cannot compute unseen client z: selector config not available")
                return
            
            # Create a temporary config for loading unseen client data
            # Use selector config as base, but set client_num to total (including unseen)
            temp_config = self.selector_cfg.clone()
            temp_config.federate.client_num = self.num_clients  # Total clients (including unseen)
            temp_config.data.root = self.data_root
            
            # Load training data for unseen clients (same as selector training)
            # This will create data_dict with all clients including unseen
            data_container, _ = load_hh_rlhf_data(temp_config, client_cfgs=None)
            
            if not isinstance(data_container, StandaloneDataDict):
                logger.warning("Data container is not StandaloneDataDict. Cannot compute unseen client z.")
                return
            
            # Initialize client_average_z_dict if not exists
            if self.client_average_z_dict is None:
                self.client_average_z_dict = {}
            
            # Set models to eval mode
            variational_encoder.eval()
            feature_extractor.eval()
            if self.selector_model is not None:
                self.selector_model.eval()
            
            vpl_latent_dim = getattr(self.config.llm, 'vpl_latent_dim', 32)
            vpl_use_feature_difference = getattr(self.config.llm, 'vpl_use_feature_difference', True)
            vpl_use_difference_only = getattr(self.config.llm, 'vpl_use_difference_only', False)
            
            # Get selector prompt format
            selector_prompt = self.selector_prompt
            
            # Get choice tokens
            choices = getattr(self.config.trainer, 'choices', ['A', 'B'])
            choice_tokens = []
            for choice in choices:
                choice_token = self.selector_tokenizer.encode(f" {choice}", add_special_tokens=False)
                if len(choice_token) > 0:
                    choice_tokens.append(choice_token[0])
                else:
                    choice_token = self.selector_tokenizer.encode(choice, add_special_tokens=False)
                    if len(choice_token) > 0:
                        choice_tokens.append(choice_token[0])
            
            if len(choice_tokens) < 2:
                logger.warning(f"Could not find choice tokens. Using fallback.")
                choice_tokens = [
                    self.selector_tokenizer.encode('A', add_special_tokens=False)[0] if len(self.selector_tokenizer.encode('A', add_special_tokens=False)) > 0 else 65,
                    self.selector_tokenizer.encode('B', add_special_tokens=False)[0] if len(self.selector_tokenizer.encode('B', add_special_tokens=False)) > 0 else 66
                ]
            
            # Process each unseen client
            with torch.no_grad():
                for client_id in unseen_clients_id:
                    if client_id not in data_container:
                        logger.warning(f"Client {client_id} not found in data container. Skipping.")
                        continue
                    
                    client_data = data_container[client_id]
                    train_data = client_data.train_data
                    
                    if len(train_data) == 0:
                        logger.warning(f"Client {client_id} has no training data. Skipping.")
                        continue
                    
                    logger.info(f"Computing z for unseen client {client_id} from {len(train_data)} training samples...")
                    
                    # Create dataset and dataloader for this client's training data
                    # Convert to list of dicts with prompt, output_A, output_B, choice
                    list_train_dict = []
                    for sample in train_data:
                        # train_data should have 'prompt', 'chosen_response', 'rejected_response'
                        if hasattr(sample, 'keys'):
                            sample_dict = dict(sample)
                        else:
                            sample_dict = sample
                        
                        # Format for binary choice: prompt, output_A (chosen), output_B (rejected), choice=" A"
                        list_train_dict.append({
                            'prompt': sample_dict.get('prompt', ''),
                            'output_A': sample_dict.get('chosen_response', ''),
                            'output_B': sample_dict.get('rejected_response', ''),
                            'choice': ' A'  # Chosen is A
                        })
                    
                    # Create LLMDataset for binary choice
                    train_dataset = LLMDataset(
                        list_train_dict,
                        self.selector_tokenizer,
                        prompt_input=selector_prompt,
                        prompt_no_input=selector_prompt,
                        output_tag='choice'
                    )
                    
                    # Create DataLoader
                    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=8,  # Small batch size for memory efficiency
                        shuffle=False,
                        num_workers=0,
                        collate_fn=LLMDataCollator(tokenizer=self.selector_tokenizer),
                        pin_memory=False
                    )
                    
                    # Collect z values from all batches
                    z_mu_list = []
                    
                    for batch_idx, batch in enumerate(train_dataloader):
                        try:
                            input_ids = batch['input_ids'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            attention_mask = batch.get('attention_mask', None)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            
                            # Forward pass through selector model to get logits and hidden states
                            if self.selector_model is not None:
                                outputs = self.selector_model(
                                    input_ids=input_ids,
                                    labels=labels,
                                    attention_mask=attention_mask,
                                    output_hidden_states=vpl_use_feature_difference
                                )
                            else:
                                logger.warning("Selector model not available. Cannot compute z.")
                                break
                            
                            logits = outputs.logits
                            
                            # Get hidden states if needed
                            hidden_states = None
                            if vpl_use_feature_difference:
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    hidden_states = outputs.hidden_states[-1]
                                elif hasattr(outputs, 'last_hidden_state'):
                                    hidden_states = outputs.last_hidden_state
                            
                            # Extract preference features (same as VPL trainer)
                            # Use embedding difference method if available
                            if vpl_use_feature_difference and hidden_states is not None:
                                # Use embedding difference method
                                batch_size, seq_len, hidden_dim = hidden_states.shape
                                shift_labels = labels[..., 1:].contiguous()
                                shift_hidden = hidden_states[..., :-1, :].contiguous()
                                
                                A_token, B_token = choice_tokens[0], choice_tokens[1]
                                A_positions = (shift_labels == A_token)
                                B_positions = (shift_labels == B_token)
                                
                                features_list = []
                                for b in range(batch_size):
                                    A_pos = A_positions[b]
                                    B_pos = B_positions[b]
                                    
                                    if A_pos.any() and B_pos.any():
                                        chosen_emb = shift_hidden[b, A_pos].mean(dim=0)
                                        rejected_emb = shift_hidden[b, B_pos].mean(dim=0)
                                    elif A_pos.any():
                                        chosen_emb = shift_hidden[b, A_pos].mean(dim=0)
                                        rejected_emb = shift_hidden[b].mean(dim=0)
                                    elif B_pos.any():
                                        chosen_emb = shift_hidden[b, B_pos].mean(dim=0)
                                        rejected_emb = shift_hidden[b].mean(dim=0)
                                    else:
                                        chosen_emb = shift_hidden[b].mean(dim=0)
                                        rejected_emb = shift_hidden[b].mean(dim=0)
                                    
                                    feature_diff = chosen_emb - rejected_emb
                                    
                                    if vpl_use_difference_only:
                                        feature_combined = feature_diff
                                    else:
                                        feature_combined = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=0)
                                    
                                    features_list.append(feature_combined)
                                
                                preference_features = torch.stack(features_list, dim=0)
                            else:
                                # Fallback: use logits-based features
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = labels[..., 1:].contiguous()
                                
                                batch_size = logits.shape[0]
                                choice_logits_per_token = shift_logits[..., choice_tokens]
                                
                                features_list = []
                                for choice_idx in range(len(choice_tokens)):
                                    choice_token = choice_tokens[choice_idx]
                                    choice_positions = (shift_labels == choice_token)
                                    
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
                                
                                features = torch.stack(features_list, dim=1)
                                preference_features = torch.cat([features, features], dim=1)
                            
                            # Ensure float32
                            if preference_features.dtype != torch.float32:
                                preference_features = preference_features.float()
                            
                            # Extract features and encode to z
                            extracted_features = feature_extractor(preference_features)
                            z_mu, z_logvar = variational_encoder.encode(extracted_features)
                            
                            # Store z_mu (mean) for averaging
                            z_mu_list.append(z_mu.detach().cpu())
                            
                        except Exception as e:
                            logger.warning(f"Error processing batch {batch_idx} for client {client_id}: {e}")
                            continue
                    
                    if len(z_mu_list) > 0:
                        # Average z_mu across all batches
                        z_mu_stack = torch.cat(z_mu_list, dim=0)  # (total_samples, latent_dim)
                        avg_z_mu = z_mu_stack.mean(dim=0)  # (latent_dim,)
                        
                        # Ensure avg_z_mu is 1D
                        avg_z_mu = avg_z_mu.view(-1)
                        
                        # Store in client_average_z_dict
                        self.client_average_z_dict[client_id] = avg_z_mu.to(self.device)
                        logger.info(f"Computed average z for unseen client {client_id}: shape {avg_z_mu.shape}, "
                                   f"from {len(z_mu_list)} batches ({z_mu_stack.shape[0]} total samples)")
                    else:
                        logger.warning(f"No z values computed for unseen client {client_id}")
            
            logger.info(f"Computed z for {len([cid for cid in unseen_clients_id if cid in self.client_average_z_dict])} unseen clients")
            
        except Exception as e:
            logger.error(f"Failed to compute unseen client z from training data: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def train(self, saveto=None, early_exiting=False):
        if saveto is None:
            _, saveto = os.path.split(self.config.federate.save_to)
        # The training data should be selector's preference data
        list_train_dict = self.load_selector_preference_data(
            saveto, early_exiting)

        # Inject client-specific z into preference data so that the
        # DPO trainer can condition on z during training (Eq. 2/11).
        if (self.client_average_z_dict is not None
                and len(self.client_average_z_dict) > 0):
            injected = 0
            for sample in list_train_dict:
                cid = sample.get('client_id', None)
                if cid is not None and cid in self.client_average_z_dict:
                    z_val = self.client_average_z_dict[cid]
                    if isinstance(z_val, torch.Tensor):
                        sample['z'] = z_val.cpu().tolist()
                    else:
                        sample['z'] = list(z_val)
                    injected += 1
            logger.info(
                f"Injected z vectors into {injected}/"
                f"{len(list_train_dict)} preference samples "
                f"from client_average_z_dict")

        # move selector model to cpu (if it exists)
        if self.selector_model is not None:
            self.selector_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # load comparison dataset
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            self.tokenizer,
            prompt_input=self.generation_prompt,
            prompt_no_input=self.generation_prompt,
            output_A="output_A",
            output_B="output_B",
            choice="choice",
        )
        
        # Create DataLoader directly instead of using ClientData
        # to avoid initialization issues with ClientData's __init__
        from torch.utils.data import DataLoader
        from federatedscope.llm.dataloader import LLMRewardCollator
        
        data_collator = LLMRewardCollator(tokenizer=self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=self.config.dataloader.shuffle,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=data_collator,
            pin_memory=self.config.dataloader.pin_memory,
        )
        
        # Create data dict compatible with trainer expectations
        data = {
            'train': train_dataloader,
            'val': None,
            'test': None,
        }

        # create DPO trainer
        self.trainer = DPORewardTrainer(
            self.model,
            data,
            self.device,
            self.config,
            only_for_eval=False,
            monitor=self._monitor,
        )
        
        # Set selector_model in trainer's ctx for winrate evaluation
        # This allows winrate metrics to use the selector model for fair comparison
        if hasattr(self.trainer, 'ctx'):
            self.trainer.ctx.selector_model = self.selector_model
            self.trainer.ctx.selector_tokenizer = self.selector_tokenizer
            logger.info("Set selector_model in trainer ctx for winrate evaluation")

        # Load test data for evaluation
        # Test evaluation requires only prompts (not chosen/rejected pairs)
        # The model will generate responses, which will be evaluated by reward models
        # Win-lose is computed separately by comparing generated responses with original chosen/rejected
        test_dataset = None
        try:
            from federatedscope.llm.dataset.llm_dataset import LLMDataset
            from federatedscope.llm.dataloader.dataloader import LLMDataCollator
            
            data_type = getattr(self.config.data, 'type', '').lower()
            # Check if VPL model for conditional generation
            use_variational_generation = getattr(self.config.llm, 'rlhf_use_variational_generation', False)
            is_vpl_model = False
            
            if use_variational_generation:
                # Check if selector checkpoint has VPL components
                selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
                if selector_ckpt_path is None:
                    selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
                
                if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                    try:
                        from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                        variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                            selector_ckpt_path, self.config, device=self.device
                        )
                        if variational_encoder is not None and feature_extractor is not None:
                            is_vpl_model = True
                    except Exception:
                        pass
            
            # Check if this is an unseen experiment (needs client-specific test data)
            unseen_clients_id = []
            if self.selector_cfg is not None:
                unseen_clients_id = getattr(self.selector_cfg.federate, 'unseen_clients_id', [])
            if len(unseen_clients_id) == 0:
                unseen_clients_id = getattr(self.config.federate, 'unseen_clients_id', [])
            
            is_unseen_experiment = len(unseen_clients_id) > 0
            num_clients = self.num_clients
            max_num_test = getattr(self.config.eval, 'max_samples_for_reward', 30)
            
            if 'ultrafeedback' in data_type:
                # Load test prompts from UltraFeedback dataset
                from federatedscope.llm.dataloader.ultrafeedback import load_ultrafeedback_for_rlhf
                logger.info("Loading test prompts from UltraFeedback dataset for evaluation...")
                use_split = (is_vpl_model and self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0) or is_unseen_experiment
                use_split = use_split and num_clients >= 4  # Need at least 4 clients for 4 UltraFeedback dims
                if use_split:
                    client_test_data, _, _ = load_ultrafeedback_for_rlhf(
                        self.data_root,
                        self.config,
                        max_num_test=max_num_test,
                        raw_no_prompt=True,
                        split_by_client=True,
                        client_num=num_clients,
                    )
                    self.client_test_data = client_test_data
                    if client_test_data and len(client_test_data) > 0:
                        list_test_dict = []
                        for cid, prompts in client_test_data.items():
                            for p in prompts:
                                prompt_text = p if isinstance(p, str) else (p.get('prompt') if isinstance(p, dict) else None)
                                if prompt_text:
                                    list_test_dict.append({'prompt': prompt_text, 'client_id': cid, 'output': ''})
                        logger.info(f"Loaded {len(list_test_dict)} UltraFeedback test prompts (split by {len(client_test_data)} clients)")
                    else:
                        list_test_dict = []
                        logger.warning("No UltraFeedback test prompts loaded (split by client).")
                else:
                    list_prompts, _, _ = load_ultrafeedback_for_rlhf(
                        self.data_root,
                        self.config,
                        max_num_test=max_num_test,
                        raw_no_prompt=True,
                        split_by_client=False,
                    )
                    if list_prompts and len(list_prompts) > 0:
                        list_test_dict = [{'prompt': (p if isinstance(p, str) else p.get('prompt')), 'output': ''} for p in list_prompts]
                        if is_vpl_model and num_clients > 0:
                            for idx, s in enumerate(list_test_dict):
                                s['client_id'] = (idx % num_clients) + 1
                        logger.info(f"Loaded {len(list_test_dict)} UltraFeedback test prompts")
                    else:
                        list_test_dict = []
                        logger.warning("No UltraFeedback test prompts loaded.")
            else:
                # Load test prompts from HH-RLHF dataset
                from federatedscope.llm.dataloader.hh_rlhf import load_hh_rlhf_for_rlhf
                logger.info("Loading test prompts from hh-rlhf dataset for evaluation...")
                if (is_vpl_model and self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0) or is_unseen_experiment:
                    client_test_data, _, _ = load_hh_rlhf_for_rlhf(
                        self.data_root,
                        self.config,
                        max_num_test=max_num_test,
                        raw_no_prompt=True,
                        split_by_client=True,
                        client_num=num_clients,
                    )
                    self.client_test_data = client_test_data
                    if is_unseen_experiment:
                        logger.info(f"Unseen experiment detected. Loaded client-specific test data for {len(unseen_clients_id)} unseen clients")
                    if client_test_data is None or len(client_test_data) == 0:
                        logger.warning("No test prompts loaded. Test evaluation will be skipped.")
                        list_test_dict = []
                    else:
                        list_test_dict = []
                        for client_id, client_prompts in client_test_data.items():
                            for prompt_dict in client_prompts:
                                if (prompt_dict.get('prompt') if isinstance(prompt_dict, dict) else prompt_dict):
                                    p = prompt_dict.get('prompt', prompt_dict) if isinstance(prompt_dict, dict) else prompt_dict
                                    list_test_dict.append({'prompt': p, 'client_id': client_id, 'output': ''})
                        logger.info(f"Loaded {len(list_test_dict)} test prompts split by {len(client_test_data)} clients (VPL model)")
                else:
                    list_test_prompts, _, _ = load_hh_rlhf_for_rlhf(
                        self.data_root,
                        self.config,
                        max_num_test=max_num_test,
                        raw_no_prompt=True,
                        split_by_client=False,
                    )
                    if list_test_prompts is None or len(list_test_prompts) == 0:
                        logger.warning("No test prompts loaded. Test evaluation will be skipped.")
                        list_test_dict = []
                    else:
                        list_test_dict = [{'prompt': p['prompt'], 'output': ''} for p in list_test_prompts if p.get('prompt')]
                    
                    # Assign client_id for VPL models (for conditional
                    # generation) - balanced across preference categories
                    # so the per-metric GPT-API eval filter does not
                    # starve a whole category when len(list_test_dict)
                    # is smaller than num_clients.
                    if is_vpl_model:
                        num_clients = self.num_clients
                        dataset_type = getattr(self.config.data, 'type',
                                               '').lower()
                        num_cats = 4 if 'ultrafeedback' in dataset_type \
                            else 2
                        _assign_balanced_client_ids(
                            list_test_dict, num_clients, num_cats)
                        logger.info(f"Assigned {len(list_test_dict)} test prompts across {num_cats} categories "
                                   f"(of {num_clients} clients, balanced round-robin) for VPL conditional generation")
                    else:
                        logger.info(f"Loaded {len(list_test_dict)} test prompts (standard generation, no client_id)")
            
            # Create LLMDataset with prompts only (no output_tag needed for test evaluation)
            if list_test_dict and len(list_test_dict) > 0:
                test_dataset = LLMDataset(
                    list_test_dict,
                    self.tokenizer,
                    prompt_input=self.generation_prompt,
                    prompt_no_input=self.generation_prompt,
                )
                
                # Use LLMDataCollator for test data (not LLMRewardCollator)
                test_data_collator = LLMDataCollator(tokenizer=self.tokenizer)
                
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=self.config.dataloader.batch_size,
                    shuffle=False,  # Don't shuffle test data
                    num_workers=self.config.dataloader.num_workers,
                    collate_fn=test_data_collator,
                    pin_memory=self.config.dataloader.pin_memory,
                )
                
                # Update data dict with test loader
                data['test'] = test_dataloader
                # Update trainer's data
                self.trainer.data = data
                self.trainer.ctx.test_loader = test_dataloader
                logger.info(f"Loaded {len(list_test_dict)} test prompts for evaluation (will generate responses during evaluation)")
            else:
                logger.warning("No test prompts loaded. Test evaluation will be skipped.")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}. Test evaluation will be skipped.")
            import traceback
            logger.error(traceback.format_exc())
        
        # Initialize z values storage for visualization
        z_values_list = []
        z_mu_list = []
        z_logvar_list = []
        
        # Visualize client-specific average z values at the start of training (before any training)
        # This shows the initial z distribution from the selector checkpoint
        if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
            try:
                from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z
                import numpy as np
                # torch and os are already imported at the top of the file
                
                logger.info("Visualizing client-specific average z values at the start of RL training...")
                
                # Prepare z values and client labels for visualization
                client_avg_z_list = []
                client_labels_list = []
                orthogonal_labels_list = []
                
                for client_id, z_mu in self.client_average_z_dict.items():
                    if isinstance(z_mu, torch.Tensor):
                        z_np = z_mu.cpu().numpy()
                    else:
                        z_np = np.array(z_mu)
                    client_avg_z_list.append(z_np)
                    client_labels_list.append(client_id)
                    
                    # Assign orthogonal label based on client_id
                    # using the same category map as MetaSplitter
                    from federatedscope.llm.llm_local.server import VPLGPServer
                    num_cats = VPLGPServer._get_dataset_num_categories(
                        self.config)
                    cpc = getattr(self.config.data,
                                  'meta_split_clients_per_cat', None)
                    cat_map = VPLGPServer._build_client_category_map(
                        self.num_clients, num_cats, cpc)
                    orthogonal_labels_list.append(
                        cat_map.get(client_id, 0))
                
                if len(client_avg_z_list) > 0:
                    z_array = np.array(client_avg_z_list)
                    
                    # Visualize
                    output_dir = self.config.outdir
                    wandb_project = getattr(self.config.wandb, 'name_project', None)
                    
                    visualize_cross_client_z(
                        z_values=z_array,
                        client_labels=client_labels_list,
                        orthogonal_labels=orthogonal_labels_list,
                        orthogonal_prototypes=None,
                        round_num=0,  # Use 0 to indicate "before training" / "initial"
                        output_dir=output_dir,
                        wandb_project=wandb_project
                    )
                    logger.info(f"Visualized {len(client_avg_z_list)} client-specific average z values at the start of training")
                    
                    # Log to WandB
                    if self.config.wandb.use and self.config.wandb.online_track:
                        try:
                            import wandb
                            initial_tsne_path = os.path.join(output_dir, 'cross_client_z_tsne_round_0.png')
                            if output_dir and os.path.exists(initial_tsne_path):
                                wandb.log({
                                    'visualization/client_average_z_tsne_initial': wandb.Image(initial_tsne_path)
                                }, step=0)
                                logger.info("Logged client average z t-SNE visualization to WandB (initial, before training)")
                            else:
                                logger.warning(f"t-SNE visualization file not found: {initial_tsne_path}")
                        except Exception as e:
                            logger.warning(f"Failed to log client average z visualization to WandB: {e}")
            except Exception as e:
                logger.warning(f"Failed to visualize client average z values at start: {e}")
        
        # start training
        for r in range(self.config.federate.total_round_num):
            logger.info("----------- Starting a new RLHF training round "
                        f"(Round #{r}) -------------")
            sample_size, model_para_all, results = self.trainer.train()
            train_log_res = self._monitor.format_eval_res(results,
                                                          rnd=r,
                                                          role="Server",
                                                          return_raw=True)
            logger.info(train_log_res)
            
            # Keep all train results (don't filter - we want all metrics in WandB)
            # Note: We'll log all metrics directly to WandB, so no need to filter
            
            # Save train results to WandB with explicit step
            # For RL training: only log loss-related metrics, exclude winrate and reward model scores
            if self.config.wandb.use and self.config.wandb.online_track:
                try:
                    import wandb
                    # Directly log metrics from Results_raw to avoid logline_2_wandb_dict removing Results_raw
                    wandb_metrics = {}
                    if 'Results_raw' in train_log_res:
                        train_results = train_log_res['Results_raw']
                        # Filter out winrate and reward model scores for train metrics (RL training)
                        # Only keep loss-related metrics: loss, avg_loss, acc, total
                        train_metrics_to_exclude = [
                            'train_helpfulness_winrate', 'train_harmlessness_winrate', 
                            'train_avg_winlose_rate', 'train_avg_helpfulness', 
                            'train_avg_harmlessness'
                        ]
                        # Log only loss-related metrics with proper naming: "Server, metric_name"
                        for key, value in train_results.items():
                            # Skip winrate and reward model scores for train metrics
                            if key in train_metrics_to_exclude:
                                continue
                            if isinstance(value, (int, float)):
                                wandb_metrics[f"Server, {key}"] = value
                            elif isinstance(value, (list, tuple)) and len(value) > 0:
                                # Handle list/tuple values (take first element or mean)
                                if isinstance(value[0], (int, float)):
                                    wandb_metrics[f"Server, {key}"] = float(sum(value) / len(value))
                    
                    # Also log Round for reference
                    wandb_metrics["Server, Round"] = r
                    
                    # Log to WandB with explicit step
                    if wandb_metrics:
                        wandb.log(wandb_metrics, step=r)
                        logger.info(f"Round {r}: Logged {len(wandb_metrics)} train metrics to WandB (excluded winrate and reward scores)")
                    else:
                        logger.warning(f"Round {r}: No train metrics to log to WandB")
                except Exception as e:
                    logger.warning(f"Failed to log train metrics to WandB: {e}")
                    # Fallback to original method
                    self._monitor.save_formatted_results(train_log_res, save_file_name="")
            
            # Collect z values for visualization (if variational generation is enabled)
            if (hasattr(self.trainer, 'use_variational_generation') and 
                self.trainer.use_variational_generation and
                hasattr(self.trainer, 'variational_encoder') and
                self.trainer.variational_encoder is not None):
                # Collect z values from trainer
                try:
                    if hasattr(self.trainer, 'get_collected_z_values'):
                        z_vals = self.trainer.get_collected_z_values()
                        if z_vals is not None and len(z_vals) > 0:
                            if isinstance(z_vals, torch.Tensor):
                                z_vals = z_vals.cpu().numpy()
                            z_values_list.extend(z_vals)
                            logger.info(f"Round {r}: Collected {len(z_vals)} z values for visualization (total: {len(z_values_list)})")
                            # Clear collected z values after collecting
                            if hasattr(self.trainer, 'clear_collected_z_values'):
                                self.trainer.clear_collected_z_values()
                except Exception as e:
                    logger.debug(f"Could not collect z values in round {r}: {e}")
            
            # Evaluate on test split if available
            # Check both data dict and trainer's data dict, and client_test_data for unseen experiments
            test_available = (data.get('test') is not None) or (hasattr(self.trainer, 'data') and self.trainer.data.get('test') is not None) or (hasattr(self, 'client_test_data') and self.client_test_data is not None and len(self.client_test_data) > 0)
            if test_available and (r + 1) % self.config.eval.freq == 0:
                logger.info("----------- Evaluating on test split -------------")
                
                # Final round: reload full test dataset for evaluation (no max_samples limit)
                is_final_round = (r + 1 == self.config.federate.total_round_num)
                data_type_final = getattr(self.config.data, 'type', '').lower()
                if is_final_round and ('hh-rlhf' in data_type_final or 'ultrafeedback' in data_type_final):
                    logger.info("Final round: reloading full test dataset for evaluation.")
                    try:
                        from federatedscope.llm.dataset.llm_dataset import LLMDataset
                        list_test_dict = None
                        if 'ultrafeedback' in data_type_final:
                            from federatedscope.llm.dataloader.ultrafeedback import load_ultrafeedback_for_rlhf
                            use_split_uf = (hasattr(self, 'client_test_data') and self.client_test_data is not None and len(self.client_test_data) > 0) and self.num_clients >= 4
                            if use_split_uf:
                                client_test_data_full, _, _ = load_ultrafeedback_for_rlhf(
                                    self.data_root, self.config, max_num_test=-1,
                                    raw_no_prompt=True, split_by_client=True, client_num=self.num_clients,
                                )
                                if client_test_data_full and len(client_test_data_full) > 0:
                                    self.client_test_data = client_test_data_full
                                    list_test_dict = []
                                    for cid, prompts in client_test_data_full.items():
                                        for p in prompts:
                                            pt = p if isinstance(p, str) else (p.get('prompt') if isinstance(p, dict) else None)
                                            if pt:
                                                list_test_dict.append({'prompt': pt, 'client_id': cid, 'output': ''})
                            else:
                                list_prompts, _, _ = load_ultrafeedback_for_rlhf(
                                    self.data_root, self.config, max_num_test=-1,
                                    raw_no_prompt=True, split_by_client=False,
                                )
                                if list_prompts and len(list_prompts) > 0:
                                    list_test_dict = [{'prompt': (p if isinstance(p, str) else p.get('prompt')), 'output': ''} for p in list_prompts]
                                    if getattr(self.config.llm, 'rlhf_use_variational_generation', False) and self.num_clients > 0:
                                        _dt = getattr(self.config.data, 'type', '').lower()
                                        _ncats = 4 if 'ultrafeedback' in _dt else 2
                                        _assign_balanced_client_ids(
                                            list_test_dict, self.num_clients, _ncats)
                        elif 'hh-rlhf' in data_type_final:
                            from federatedscope.llm.dataloader.hh_rlhf import load_hh_rlhf_for_rlhf
                            if hasattr(self, 'client_test_data') and self.client_test_data is not None and len(self.client_test_data) > 0:
                                client_test_data_full, _, _ = load_hh_rlhf_for_rlhf(
                                    self.data_root,
                                    self.config,
                                    max_num_test=-1,
                                    raw_no_prompt=True,
                                    split_by_client=True,
                                    client_num=self.num_clients,
                                )
                                if client_test_data_full and len(client_test_data_full) > 0:
                                    self.client_test_data = client_test_data_full
                                    list_test_dict = []
                                    for client_id, client_prompts in client_test_data_full.items():
                                        for prompt_dict in client_prompts:
                                            if prompt_dict.get('prompt'):
                                                list_test_dict.append({
                                                    'prompt': prompt_dict['prompt'],
                                                    'client_id': client_id,
                                                    'output': '',
                                                })
                                    logger.info(f"Reloaded full test set: {len(list_test_dict)} prompts (split by {len(client_test_data_full)} clients)")
                                else:
                                    logger.warning("Full test reload returned empty; keeping existing loader.")
                                    list_test_dict = None
                            else:
                                list_test_prompts, _, _ = load_hh_rlhf_for_rlhf(
                                    self.data_root,
                                    self.config,
                                    max_num_test=-1,
                                    raw_no_prompt=True,
                                    split_by_client=False,
                                )
                                if list_test_prompts and len(list_test_prompts) > 0:
                                    list_test_dict = [{'prompt': p['prompt'], 'output': ''} for p in list_test_prompts if p.get('prompt')]
                                    if getattr(self.config.llm, 'rlhf_use_variational_generation', False) and self.num_clients > 0:
                                        _dt = getattr(self.config.data, 'type', '').lower()
                                        _ncats = 4 if 'ultrafeedback' in _dt else 2
                                        _assign_balanced_client_ids(
                                            list_test_dict, self.num_clients, _ncats)
                                    logger.info(f"Reloaded full test set: {len(list_test_dict)} prompts")
                                else:
                                    logger.warning("Full test reload returned empty; keeping existing loader.")
                                    list_test_dict = None
                        if list_test_dict is not None and len(list_test_dict) > 0:
                            test_dataset_full = LLMDataset(
                                list_test_dict,
                                self.tokenizer,
                                prompt_input=self.generation_prompt,
                                prompt_no_input=self.generation_prompt,
                            )
                            test_dataloader_full = DataLoader(
                                test_dataset_full,
                                batch_size=self.config.dataloader.batch_size,
                                shuffle=False,
                                num_workers=self.config.dataloader.num_workers,
                                collate_fn=LLMDataCollator(tokenizer=self.tokenizer),
                                pin_memory=self.config.dataloader.pin_memory,
                            )
                            data['test'] = test_dataloader_full
                            if hasattr(self.trainer, 'data'):
                                self.trainer.data['test'] = test_dataloader_full
                            self.trainer.ctx.test_loader = test_dataloader_full
                            logger.info(f"Updated test loader to full dataset ({len(list_test_dict)} samples).")
                    except Exception as e:
                        logger.warning(f"Failed to reload full test data for final round: {e}. Using existing test loader.")
                        import traceback
                        logger.debug(traceback.format_exc())
                
                # Check if this is an unseen experiment (has unseen_clients_id in config)
                # Try to get from selector_cfg first (for VPL models), then from main config (for FedDPO)
                unseen_clients_id = []
                if self.selector_cfg is not None:
                    unseen_clients_id = getattr(self.selector_cfg.federate, 'unseen_clients_id', [])
                # Fallback to main config if selector_cfg is None (e.g., FedDPO)
                if len(unseen_clients_id) == 0:
                    unseen_clients_id = getattr(self.config.federate, 'unseen_clients_id', [])
                
                # If we have client-specific test data and unseen_clients_id, evaluate separately
                if hasattr(self, 'client_test_data') and self.client_test_data is not None and len(unseen_clients_id) > 0:
                    # Evaluate separately for seen and unseen clients
                    seen_client_ids = [cid for cid in range(1, self.num_clients + 1) if cid not in unseen_clients_id]
                    unseen_client_ids = unseen_clients_id
                    
                    logger.info(f"Unseen experiment detected. Evaluating separately:")
                    logger.info(f"  Seen clients: {seen_client_ids}")
                    logger.info(f"  Unseen clients: {unseen_client_ids}")
                    
                    # Evaluate seen clients
                    # Limit to max_samples_for_reward per client type (not per client)
                    max_eval_samples = getattr(self.config.eval, 'max_samples_for_reward', 30)
                    seen_test_dict = []
                    seen_samples_count = 0
                    for client_id in seen_client_ids:
                        if client_id in self.client_test_data:
                            for prompt_item in self.client_test_data[client_id]:
                                prompt_text = prompt_item.get('prompt') if isinstance(prompt_item, dict) else prompt_item
                                if prompt_text and seen_samples_count < max_eval_samples:
                                    seen_test_dict.append({
                                        'prompt': prompt_text,
                                        'client_id': client_id,
                                        'output': '',
                                    })
                                    seen_samples_count += 1
                                if seen_samples_count >= max_eval_samples:
                                    break
                        if seen_samples_count >= max_eval_samples:
                            break
                    
                    logger.info(f"Prepared {len(seen_test_dict)} seen test samples (limited to {max_eval_samples})")
                    
                    # Evaluate unseen clients
                    unseen_test_dict = []
                    unseen_samples_count = 0
                    for client_id in unseen_client_ids:
                        if client_id in self.client_test_data:
                            for prompt_item in self.client_test_data[client_id]:
                                prompt_text = prompt_item.get('prompt') if isinstance(prompt_item, dict) else prompt_item
                                if prompt_text and unseen_samples_count < max_eval_samples:
                                    unseen_test_dict.append({
                                        'prompt': prompt_text,
                                        'client_id': client_id,
                                        'output': '',
                                    })
                                    unseen_samples_count += 1
                                if unseen_samples_count >= max_eval_samples:
                                    break
                        if unseen_samples_count >= max_eval_samples:
                            break
                    
                    logger.info(f"Prepared {len(unseen_test_dict)} unseen test samples (limited to {max_eval_samples})")
                    
                    # Evaluate seen clients
                    seen_results = None
                    if len(seen_test_dict) > 0:
                        from federatedscope.llm.dataset.llm_dataset import LLMDataset
                        from federatedscope.llm.dataloader import LLMDataCollator
                        seen_test_dataset = LLMDataset(
                            seen_test_dict,
                            self.tokenizer,
                            prompt_input=self.generation_prompt,
                            prompt_no_input=self.generation_prompt,
                        )
                        seen_test_dataloader = DataLoader(
                            seen_test_dataset,
                            batch_size=self.config.dataloader.batch_size,
                            shuffle=False,
                            num_workers=self.config.dataloader.num_workers,
                            collate_fn=LLMDataCollator(tokenizer=self.tokenizer),
                            pin_memory=self.config.dataloader.pin_memory,
                        )
                        self.trainer.ctx.test_loader = seen_test_dataloader
                        seen_results = self.trainer.evaluate(target_data_split_name="test")
                        if seen_results is None:
                            seen_results = getattr(self.trainer.ctx, 'eval_metrics', {})
                    
                    # Evaluate unseen clients
                    unseen_results = None
                    if len(unseen_test_dict) > 0:
                        from federatedscope.llm.dataset.llm_dataset import LLMDataset
                        from federatedscope.llm.dataloader import LLMDataCollator
                        unseen_test_dataset = LLMDataset(
                            unseen_test_dict,
                            self.tokenizer,
                            prompt_input=self.generation_prompt,
                            prompt_no_input=self.generation_prompt,
                        )
                        unseen_test_dataloader = DataLoader(
                            unseen_test_dataset,
                            batch_size=self.config.dataloader.batch_size,
                            shuffle=False,
                            num_workers=self.config.dataloader.num_workers,
                            collate_fn=LLMDataCollator(tokenizer=self.tokenizer),
                            pin_memory=self.config.dataloader.pin_memory,
                        )
                        self.trainer.ctx.test_loader = unseen_test_dataloader
                        unseen_results = self.trainer.evaluate(target_data_split_name="test")
                        if unseen_results is None:
                            unseen_results = getattr(self.trainer.ctx, 'eval_metrics', {})
                    
                    # Format and log results separately
                    seen_log_res = None
                    unseen_log_res = None
                    
                    if seen_results:
                        seen_log_res = self._monitor.format_eval_res(seen_results, rnd=r, role="Server (Seen)", return_raw=True)
                        logger.info(f"Round {r}: Seen clients ({len(seen_client_ids)} clients) test evaluation results:")
                        if 'Results_raw' in seen_log_res:
                            for key, value in seen_log_res['Results_raw'].items():
                                if isinstance(value, (int, float)):
                                    logger.info(f"  Seen_{key}: {value:.4f}")
                                elif isinstance(value, (list, tuple)) and len(value) > 0:
                                    if isinstance(value[0], (int, float)):
                                        avg_value = float(sum(value) / len(value))
                                        logger.info(f"  Seen_{key}: {avg_value:.4f} (from {len(value)} samples)")
                    
                    if unseen_results:
                        unseen_log_res = self._monitor.format_eval_res(unseen_results, rnd=r, role="Server (Unseen)", return_raw=True)
                        logger.info(f"Round {r}: Unseen clients ({len(unseen_client_ids)} clients) test evaluation results:")
                        if 'Results_raw' in unseen_log_res:
                            for key, value in unseen_log_res['Results_raw'].items():
                                if isinstance(value, (int, float)):
                                    logger.info(f"  Unseen_{key}: {value:.4f}")
                                elif isinstance(value, (list, tuple)) and len(value) > 0:
                                    if isinstance(value[0], (int, float)):
                                        avg_value = float(sum(value) / len(value))
                                        logger.info(f"  Unseen_{key}: {avg_value:.4f} (from {len(value)} samples)")
                    
                    # Log to WandB separately
                    if (seen_log_res or unseen_log_res) and self.config.wandb.use and self.config.wandb.online_track:
                        try:
                            import wandb
                            wandb_metrics = {}
                            
                            # Log seen clients with BOTH formats:
                            # 1. Standard format (for compatibility with existing plots): "Server, {key}"
                            # 2. Seen-specific format: "Server_Seen, {key}"
                            if seen_log_res and 'Results_raw' in seen_log_res:
                                for key, value in seen_log_res['Results_raw'].items():
                                    if isinstance(value, (int, float)):
                                        # Standard format (for existing WandB plots like test_harmless, test_helpful)
                                        wandb_metrics[f"Server, {key}"] = value
                                        # Seen-specific format (for seen/unseen comparison)
                                        wandb_metrics[f"Server_Seen, {key}"] = value
                                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                                        if isinstance(value[0], (int, float)):
                                            avg_value = float(sum(value) / len(value))
                                            # Standard format
                                            wandb_metrics[f"Server, {key}"] = avg_value
                                            # Seen-specific format
                                            wandb_metrics[f"Server_Seen, {key}"] = avg_value
                            
                            # Log unseen clients with Unseen-specific format: "Server_Unseen, {key}"
                            if unseen_log_res and 'Results_raw' in unseen_log_res:
                                for key, value in unseen_log_res['Results_raw'].items():
                                    if isinstance(value, (int, float)):
                                        wandb_metrics[f"Server_Unseen, {key}"] = value
                                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                                        if isinstance(value[0], (int, float)):
                                            wandb_metrics[f"Server_Unseen, {key}"] = float(sum(value) / len(value))
                            
                            wandb_metrics["Server, Round"] = r
                            
                            if wandb_metrics:
                                wandb.log(wandb_metrics, step=r)
                                logger.info(f"Round {r}: Logged seen/unseen test metrics to WandB")
                                logger.info(f"  - Seen clients: Standard format (Server, {{key}}) + Seen format (Server_Seen, {{key}})")
                                logger.info(f"  - Unseen clients: Unseen format (Server_Unseen, {{key}})")
                        except Exception as e:
                            logger.warning(f"Failed to log seen/unseen test metrics to WandB: {e}")
                    
                    # Restore original test loader for compatibility
                    if 'test' in data:
                        self.trainer.ctx.test_loader = data['test']
                    elif hasattr(self.trainer, 'data') and 'test' in self.trainer.data:
                        self.trainer.ctx.test_loader = self.trainer.data['test']
                else:
                    # Standard evaluation (no unseen experiment or no client-specific data)
                    # Ensure test_loader is set in ctx
                    if hasattr(self.trainer, 'data') and 'test' in self.trainer.data:
                        self.trainer.ctx.test_loader = self.trainer.data['test']
                    elif 'test' in data:
                        self.trainer.ctx.test_loader = data['test']
                    
                    test_results = self.trainer.evaluate(target_data_split_name="test")
                    if test_results is None:
                        # If evaluate returns None, try to get eval_metrics from ctx
                        test_results = getattr(self.trainer.ctx, 'eval_metrics', {})
                    
                    if test_results:
                        test_log_res = self._monitor.format_eval_res(test_results,
                                                                     rnd=r,
                                                                     role="Server",
                                                                     return_raw=True)
                        # Log test results with detailed metrics
                        logger.info(f"Round {r}: Test evaluation results:")
                        if 'Results_raw' in test_log_res:
                            for key, value in test_log_res['Results_raw'].items():
                                if isinstance(value, (int, float)):
                                    logger.info(f"  {key}: {value:.4f}")
                                elif isinstance(value, (list, tuple)) and len(value) > 0:
                                    if isinstance(value[0], (int, float)):
                                        avg_value = float(sum(value) / len(value))
                                        logger.info(f"  {key}: {avg_value:.4f} (from {len(value)} samples)")
                        logger.info(f"Full test log: {test_log_res}")
                    else:
                        logger.warning(f"Round {r+1}: Test evaluation returned no results.")
                        test_log_res = None
                    
                    # Save test results to WandB with explicit step
                    # For RL training: log all test metrics including winrate and reward model scores
                    if test_log_res and self.config.wandb.use and self.config.wandb.online_track:
                        try:
                            import wandb
                            # Directly log metrics from Results_raw to avoid logline_2_wandb_dict removing Results_raw
                            wandb_metrics = {}
                            if 'Results_raw' in test_log_res:
                                test_results = test_log_res['Results_raw']
                                # Log all test metrics including winrate and reward model scores
                                # These are the important metrics for RL evaluation
                                for key, value in test_results.items():
                                    if isinstance(value, (int, float)):
                                        wandb_metrics[f"Server, {key}"] = value
                                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                                        # Handle list/tuple values (take first element or mean)
                                        if isinstance(value[0], (int, float)):
                                            wandb_metrics[f"Server, {key}"] = float(sum(value) / len(value))
                            
                            # Also log Round for reference
                            wandb_metrics["Server, Round"] = r
                            
                            # Log to WandB with explicit step
                            if wandb_metrics:
                                wandb.log(wandb_metrics, step=r)
                                logger.info(f"Round {r}: Logged {len(wandb_metrics)} test metrics to WandB (including winrate and reward scores)")
                            else:
                                logger.warning(f"Round {r}: No test metrics to log to WandB")
                        except Exception as e:
                            logger.warning(f"Failed to log test metrics to WandB: {e}")
                            # Fallback to original method
                            self._monitor.save_formatted_results(test_log_res, save_file_name="")
            elif (r + 1) % self.config.eval.freq == 0:
                logger.warning(f"Round {r+1}: Test data not available for evaluation. Skipping test evaluation.")
            
            # Visualize z values periodically (every 5 rounds or at the end)
            if (len(z_values_list) > 0 and 
                ((r + 1) % 5 == 0 or r == self.config.federate.total_round_num - 1)):
                try:
                    from federatedscope.llm.llm_local.z_visualization import visualize_cross_client_z
                    import numpy as np
                    import matplotlib.pyplot as plt
                    
                    # For standalone RL, we need to infer client labels from preference_type
                    # Since we have both harmlessness (client_id=1) and helpfulness (client_id=2) data
                    # We should use the actual client_id from the data if available
                    # Otherwise, infer from preference_type in collected z values
                    # For now, we'll use a simple approach: assign labels based on z distribution
                    # In practice, we should track client_id along with z values
                    
                    # Convert to numpy array
                    z_array = np.array(z_values_list)
                    
                    # For RL training, we have z values from both harmlessness and helpfulness clients
                    # Since we can't easily distinguish them from z alone, we'll use a single "RL" label
                    # But we should log where the file is saved
                    client_labels = [1] * len(z_values_list)  # All labeled as client 1 for now
                    
                    # Visualize t-SNE
                    output_dir = self.config.outdir
                    wandb_project = getattr(self.config.wandb, 'name_project', None)
                    
                    visualize_cross_client_z(
                        z_values=z_array,
                        client_labels=client_labels,
                        orthogonal_labels=None,  # No orthogonal labels in standalone RL
                        orthogonal_prototypes=None,
                        round_num=r,
                        output_dir=output_dir,
                        wandb_project=wandb_project
                    )
                    
                    # Log the file path
                    tsne_filename = f"cross_client_z_tsne_round_{r}.png"
                    tsne_path = os.path.join(output_dir, tsne_filename)
                    logger.info(f"Round {r}: Generated t-SNE visualization with {len(z_values_list)} z values")
                    logger.info(f"  Saved to: {tsne_path}")
                    if os.path.exists(tsne_path):
                        logger.info(f"  File exists: {os.path.getsize(tsne_path)} bytes")
                    else:
                        logger.warning(f"  File not found at expected path: {tsne_path}")
                    
                    # Log z statistics to WandB
                    if self.config.wandb.use and self.config.wandb.online_track:
                        try:
                            import wandb
                            
                            # Calculate statistics
                            z_mean = np.mean(z_array, axis=0)
                            z_std = np.std(z_array, axis=0)
                            z_norm = np.linalg.norm(z_array, axis=1)
                            
                            # Log statistics
                            wandb_metrics = {
                                f'z_stats/mean_norm': np.mean(z_norm),
                                f'z_stats/std_norm': np.std(z_norm),
                                f'z_stats/mean_dim_0': float(z_mean[0]) if len(z_mean) > 0 else 0.0,
                                f'z_stats/std_dim_0': float(z_std[0]) if len(z_std) > 0 else 0.0,
                                f'z_stats/num_values': len(z_values_list),
                            }
                            
                            # Log histogram of z norms
                            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                            ax_hist.hist(z_norm, bins=30, alpha=0.7, edgecolor='black')
                            ax_hist.set_xlabel('||z|| (L2 norm)', fontsize=12)
                            ax_hist.set_ylabel('Frequency', fontsize=12)
                            ax_hist.set_title(f'Distribution of z Norms (Round {r})', fontsize=14)
                            ax_hist.grid(True, alpha=0.3)
                            plt.tight_layout()
                            
                            wandb_metrics[f'z_stats/z_norm_histogram'] = wandb.Image(fig_hist)
                            
                            wandb.log(wandb_metrics, step=r)
                            plt.close(fig_hist)
                            
                            logger.info(f"Round {r}: Logged z statistics to WandB")
                        except Exception as e:
                            logger.warning(f"Failed to log z statistics to WandB: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to visualize z values in round {r}: {e}")
            # Save the checkpoint
            if (r + 1) % self.config.federate.save_freq == 0:
                if saveto in self.config.federate.save_to:
                    path = add_prefix_to_path(f"{r + 1}_",
                                              self.config.federate.save_to)
                else:
                    path = add_prefix_to_path(f"{r + 1}_{saveto}_",
                                              self.config.federate.save_to)
                self.model.save_model(path=path, state=r)

    def _load_original_pairs(self, is_vpl_model,
                             category_client_ids):
        """Load original HH-RLHF chosen/rejected pairs.

        For each example, output_A = chosen, output_B = rejected
        (randomized). Each pair gets assigned client_ids from both
        harmless and helpful categories for VPL selection.
        """
        import datasets
        import random

        dataset_type = getattr(
            self.config.data, 'type', '').lower()
        if 'hh-rlhf' not in dataset_type:
            logger.warning(
                "rlhf_use_original_pairs only supports "
                "HH-RLHF dataset")
            return None

        harmless_raw = datasets.load_dataset(
            "Anthropic/hh-rlhf", data_dir="harmless-base",
            split="train")
        helpful_raw = datasets.load_dataset(
            "Anthropic/hh-rlhf", data_dir="helpful-base",
            split="train")

        max_prompts = getattr(
            self.config.llm,
            'max_prompts_for_generation', 500)
        # Take half from each subset
        n_per = max_prompts // 2

        def parse_dialogue(text):
            """Split HH-RLHF text into prompt + last response."""
            parts = text.strip().split("\n\nAssistant:")
            if len(parts) >= 2:
                prompt = ("\n\nAssistant:".join(parts[:-1]))
                response = parts[-1].strip()
                return prompt, response
            return text, ""

        list_pairwise = []
        harmless_client_ids = category_client_ids.get(0, [1])
        helpful_client_ids = category_client_ids.get(1, [6])

        for subset, raw_data, n in [
                ("harmless", harmless_raw, n_per),
                ("helpful", helpful_raw, n_per)]:
            indices = list(range(len(raw_data)))
            random.shuffle(indices)
            for idx in indices[:n]:
                ex = raw_data[idx]
                prompt_c, chosen = parse_dialogue(ex["chosen"])
                prompt_r, rejected = parse_dialogue(ex["rejected"])
                if not chosen or not rejected:
                    continue
                # Randomize A/B to remove positional bias
                if random.random() < 0.5:
                    out_a, out_b = chosen, rejected
                else:
                    out_a, out_b = rejected, chosen
                pair = {
                    "prompt": prompt_c,
                    "output_A": out_a,
                    "output_B": out_b,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_subset": subset,
                }
                if is_vpl_model:
                    hc = random.choice(harmless_client_ids)
                    pc = random.choice(helpful_client_ids)
                    pair["harmless_client_id"] = hc
                    pair["helpful_client_id"] = pc
                    pair["cat_0_client_id"] = hc
                    pair["cat_1_client_id"] = pc
                list_pairwise.append(pair)

        logger.info(
            f"Loaded {len(list_pairwise)} original HH-RLHF "
            f"pairs ({n_per} harmless + {n_per} helpful)")
        return list_pairwise

    def _generate_pairwise_data(self,
                                list_data_dict,
                                model,
                                tokenizer,
                                prompt,
                                max_new_tokens=60,
                                num_completions=2):
        generate_kwargs = dict(
            top_p=1.0,
            temperature=0.7,
            do_sample=True,  # Must be True for num_return_sequences > 1
            # early_stopping=True,
            max_new_tokens=max_new_tokens,
            num_return_sequences=max(2, num_completions),
        )
        
        # Ensure do_sample is True when num_return_sequences > 1
        if generate_kwargs['num_return_sequences'] > 1 and not generate_kwargs.get('do_sample', False):
            generate_kwargs['do_sample'] = True
            logger.warning(f"num_return_sequences={generate_kwargs['num_return_sequences']} > 1, forcing do_sample=True")

        # For VPL: Enable z-dependent generation so the policy is
        # conditioned on client z during evaluation (matches DPO training).
        use_variational_generation = getattr(
            self.config.llm, 'rlhf_use_variational_generation', False)
        z_to_embedding = None
        
        # For VPL/VPL-GP: Determine harmless and helpful client IDs from num_clients
        # Check if this is a VPL model (VPL or VPL-GP selector)
        is_vpl_selector = False
        selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
        if selector_ckpt_path is None:
            selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
        
        if selector_ckpt_path and os.path.exists(selector_ckpt_path):
            try:
                ckpt = torch.load(selector_ckpt_path, map_location='cpu')
                # Check if checkpoint has VPL components
                if 'model' in ckpt:
                    model_keys = list(ckpt['model'].keys())
                    has_vpl = any('variational_encoder' in k or 'latent_projection' in k for k in model_keys)
                    if has_vpl:
                        is_vpl_selector = True
                        logger.info("VPL/VPL-GP selector detected. Will use client-specific z for conditional selection.")
            except Exception as e:
                logger.warning(f"Could not check selector checkpoint for VPL components: {e}")
        
        # Determine client IDs per category from num_clients (only for VPL)
        # Uses the same formula as MetaSplitter for consistency
        harmless_client_ids = []
        helpful_client_ids = []
        category_client_ids = {}  # cat_idx -> [client_ids]
        if is_vpl_selector:
            num_clients = self.num_clients
            dataset_type = getattr(self.config.data, 'type', '').lower()
            if 'ultrafeedback' in dataset_type:
                num_cats = 4
            else:
                num_cats = 2
            clients_per_cat = num_clients // num_cats
            cat_remainder = num_clients % num_cats
            cid = 1
            for cat_idx in range(num_cats):
                n = clients_per_cat + (1 if cat_idx < cat_remainder else 0)
                category_client_ids[cat_idx] = list(range(cid, cid + n))
                cid += n
            # For HH-RLHF backward compat: cat 0 = harmless, cat 1 = helpful
            harmless_client_ids = category_client_ids.get(0, [])
            helpful_client_ids = category_client_ids.get(1, [])
            logger.info(f"VPL selector: {num_cats} categories, "
                        f"client distribution: {category_client_ids}")
        else:
            logger.info("Non-VPL selector detected. Will use standard selection (no client assignment).")
        
        # Still load client average z for later use in binary selection
        if self.client_average_z_dict is None:
            from federatedscope.llm.rlhf.load_vpl_components import (
                load_vpl_components_from_checkpoint,
                load_client_average_z_from_checkpoint
            )
            selector_ckpt_path = getattr(self.config.llm, 'rlhf_selector_checkpoint', None)
            if selector_ckpt_path is None:
                selector_ckpt_path = getattr(self.config.llm, 'selector_save_to', None)
            
            if selector_ckpt_path and os.path.exists(selector_ckpt_path):
                logger.info(f"Loading z_to_embedding and client average z for generation from {selector_ckpt_path}")
                _, _, _, z_to_embedding = load_vpl_components_from_checkpoint(
                    selector_ckpt_path, self.config, device=self.device
                )
                if z_to_embedding is None:
                    logger.warning("Failed to load z_to_embedding. Generation will not use z.")
                    use_variational_generation = False
                else:
                    # Verify z_to_embedding output dimension matches model embedding dimension
                    try:
                        actual_embedding_dim = self.model.get_input_embeddings().embedding_dim
                        z_to_embedding_output_dim = z_to_embedding.weight.shape[0]  # (out_features, in_features)
                        
                        if z_to_embedding_output_dim != actual_embedding_dim:
                            logger.warning(f"z_to_embedding output dimension ({z_to_embedding_output_dim}) does not match "
                                         f"model embedding dimension ({actual_embedding_dim}). Reinitializing z_to_embedding.")
                            # Reinitialize z_to_embedding with correct output dimension
                            vpl_latent_dim = z_to_embedding.weight.shape[1]  # in_features
                            import torch.nn as nn
                            z_to_embedding = nn.Linear(vpl_latent_dim, actual_embedding_dim).to(self.device)
                            logger.info(f"Reinitialized z_to_embedding: {vpl_latent_dim} -> {actual_embedding_dim}")
                    except Exception as e:
                        logger.warning(f"Could not verify z_to_embedding dimension: {e}. Using loaded z_to_embedding.")
                    # Load client-specific average z values from training data (compute once, reuse)
                    self.client_average_z_dict = load_client_average_z_from_checkpoint(
                        selector_ckpt_path, device=self.device
                    )
                    self._fill_z_dict_with_category_mean()
                    if self.client_average_z_dict is not None and len(self.client_average_z_dict) > 0:
                        logger.info(f"Loaded average z for {len(self.client_average_z_dict)} clients. "
                                   f"Will use client-specific z for conditional generation.")
                    else:
                        logger.warning("No client average z found in checkpoint. Will compute from training data.")
                        # If not in checkpoint, compute from training data
                        self._compute_client_average_z_from_training_data(selector_ckpt_path)
        
        # Use stored client_average_z_dict
        client_average_z_dict = self.client_average_z_dict

        # For VPL: Assign one client per category to each prompt
        category_gen_prompts = {}  # cat_idx -> generation prompt template
        if is_vpl_selector and len(category_client_ids) > 0:
            prompts_with_client_assignment = []
            for prompt_data in list_data_dict:
                prompt_data_copy = copy.deepcopy(prompt_data)
                # Assign one random client per category
                for cat_idx, cids in category_client_ids.items():
                    prompt_data_copy[f'cat_{cat_idx}_client_id'] = \
                        random.choice(cids)
                # Backward compat: keep harmless/helpful keys
                # for HH-RLHF (cat 0 = harmless, cat 1 = helpful)
                if 0 in category_client_ids:
                    prompt_data_copy['harmless_client_id'] = \
                        prompt_data_copy.get('cat_0_client_id')
                if 1 in category_client_ids:
                    prompt_data_copy['helpful_client_id'] = \
                        prompt_data_copy.get('cat_1_client_id')
                prompts_with_client_assignment.append(prompt_data_copy)

            logger.info(
                f"Assigned clients to "
                f"{len(prompts_with_client_assignment)} prompts "
                f"({num_cats} categories)")
            list_data_dict_for_generation = prompts_with_client_assignment

            # Build per-category generation prompt templates
            dataset_type = getattr(
                self.config.data, 'type', '').lower()
            if 'ultrafeedback' in dataset_type:
                from federatedscope.llm.dataloader.ultrafeedback \
                    import ULTRAFEEDBACK_PROMPT_DICT
                _dim_names = [
                    'helpfulness', 'honesty',
                    'instruction_following', 'truthfulness']
                for cat_idx, dim in enumerate(_dim_names):
                    key = f"generation_{dim}"
                    category_gen_prompts[cat_idx] = \
                        ULTRAFEEDBACK_PROMPT_DICT.get(key, prompt)
            else:
                from federatedscope.llm.dataloader.hh_rlhf \
                    import HH_RLHF_PROMPT_DICT
                category_gen_prompts[0] = \
                    HH_RLHF_PROMPT_DICT.get(
                        "generation_harmless", prompt)
                category_gen_prompts[1] = \
                    HH_RLHF_PROMPT_DICT.get(
                        "generation_helpful", prompt)
            logger.info(
                f"VPL generation: generating "
                f"{len(category_gen_prompts)} responses per prompt "
                f"(one per category)")
        else:
            # Non-VPL: use original prompts without client assignment
            list_data_dict_for_generation = list_data_dict
        
        # Get model device (handle device_map='auto' case) - needed for both VPL and non-VPL paths
        if hasattr(model, 'device'):
            model_device = model.device
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
            model_device = model.base_model.device
        else:
            try:
                model_device = next(model.parameters()).device
            except:
                model_device = self.device
        
        new_list_data_dict = []
        for input_data in get_input_data(list_data_dict_for_generation):
            # For VPL: Generate one response per category per prompt
            if is_vpl_selector and len(category_gen_prompts) > 0:
                response_map = [[] for _ in input_data]

                cat_gen_kwargs = dict(
                    top_p=1.0,
                    temperature=0.7,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                )

                for idx, data in enumerate(input_data):
                    cat_responses = []
                    for cat_idx in sorted(category_gen_prompts):
                        cat_prompt = category_gen_prompts[cat_idx]
                        input_text = cat_prompt.format_map(data)
                        tokens = tokenizer(
                            [input_text],
                            padding=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        )
                        tokens_dev = {
                            k: v.to(model_device)
                            if isinstance(v, torch.Tensor) else v
                            for k, v in tokens.items()
                        }
                        out_ids = model.generate(
                            **tokens_dev, **cat_gen_kwargs)
                        decoded = tokenizer.batch_decode(
                            out_ids,
                            skip_special_tokens=True,
                            ignore_tokenization_space=True)
                        gen_text = decoded[0].replace(
                            input_text, "").strip().replace(
                            "</s>", "")
                        cat_responses.append(gen_text)

                    # Shuffle responses to remove order bias
                    random.shuffle(cat_responses)
                    response_map[idx] = cat_responses

                # Create pairwise combinations
                for i, data in enumerate(input_data):
                    prompt_text = data.get(
                        'prompt', '')[:100] if 'prompt' in data \
                        else ''
                    logger.info(f"Data {i}: prompt={prompt_text}...")
                    for j, res in enumerate(response_map[i]):
                        logger.info(
                            f'Generated {j}-th response: '
                            f'{res[:100]}...')

                    for output_A, output_B in combinations(
                            response_map[i], 2):
                        new_data = copy.deepcopy(data)
                        new_data["output_A"] = output_A
                        new_data["output_B"] = output_B
                        # Keep per-category client ids
                        for cat_idx in category_client_ids:
                            key = f'cat_{cat_idx}_client_id'
                            if key not in new_data:
                                new_data[key] = random.choice(
                                    category_client_ids[cat_idx])
                        # Backward compat keys
                        if 'harmless_client_id' not in new_data \
                                and 0 in category_client_ids:
                            new_data['harmless_client_id'] = \
                                new_data.get('cat_0_client_id')
                        if 'helpful_client_id' not in new_data \
                                and 1 in category_client_ids:
                            new_data['helpful_client_id'] = \
                                new_data.get('cat_1_client_id')
                        new_list_data_dict.append(new_data)

                continue  # Skip the standard generation below
            else:
                # Non-VPL or fallback: use standard prompt
                input_texts = [prompt.format_map(data) for data in input_data]
            input_text_tokens = tokenizer(
                input_texts,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            # Move tokens to model device
            input_text_tokens_device = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in input_text_tokens.items()}
            
            input_ids = input_text_tokens_device['input_ids']
            attention_mask = input_text_tokens_device['attention_mask']

            # Standard generation without z (for VPL, we generate once and split later)
            # z conditional generation is disabled - we use standard generation
            if use_variational_generation and z_to_embedding is not None:
                # Priority 1: Use client-specific average z from training data (if available)
                # Priority 2: Use z from data (if already generated in previous rounds)
                # Priority 3: Use overall average z (for standalone mode)
                # Priority 4: Infer z from input (fallback)
                
                z = None
                batch_size = len(input_data)
                vpl_latent_dim = getattr(self.config.llm, 'vpl_latent_dim', 32)
                
                # Priority 1: Try to use client-specific average z
                if client_average_z_dict is not None and len(client_average_z_dict) > 0:
                    z_list = []
                    for data in input_data:
                        # Check if data has client_id
                        client_id = data.get('client_id', None)
                        if client_id is not None and client_id in client_average_z_dict:
                            # Use client-specific average z
                            z_val = client_average_z_dict[client_id]
                            # Ensure z_val is 1D: (latent_dim,)
                            if isinstance(z_val, torch.Tensor):
                                z_val = z_val.view(-1)  # Flatten to 1D
                                # Ensure correct size
                                if z_val.shape[0] != vpl_latent_dim:
                                    if z_val.shape[0] < vpl_latent_dim:
                                        padding = torch.zeros(vpl_latent_dim - z_val.shape[0], device=z_val.device)
                                        z_val = torch.cat([z_val, padding])
                                    else:
                                        z_val = z_val[:vpl_latent_dim]
                            z_list.append(z_val)
                        else:
                            # Use overall average z (average of all clients)
                            # Ensure all values are 1D before stacking
                            all_z_vals = [z.view(-1) if isinstance(z, torch.Tensor) else z for z in client_average_z_dict.values()]
                            all_z_mus = torch.stack(all_z_vals)  # (num_clients, latent_dim)
                            overall_avg_z = all_z_mus.mean(dim=0)  # (latent_dim,)
                            overall_avg_z = overall_avg_z.view(-1)  # Ensure 1D
                            # Ensure correct size
                            if overall_avg_z.shape[0] != vpl_latent_dim:
                                if overall_avg_z.shape[0] < vpl_latent_dim:
                                    padding = torch.zeros(vpl_latent_dim - overall_avg_z.shape[0], device=overall_avg_z.device)
                                    overall_avg_z = torch.cat([overall_avg_z, padding])
                                else:
                                    overall_avg_z = overall_avg_z[:vpl_latent_dim]
                            z_list.append(overall_avg_z)
                    
                    if len(z_list) == batch_size:
                        # Ensure all z values are 1D before stacking
                        z_list_1d = []
                        for i, z_val in enumerate(z_list):
                            if isinstance(z_val, torch.Tensor):
                                z_val = z_val.view(-1)  # Flatten to 1D
                                # Ensure correct size
                                if z_val.shape[0] != vpl_latent_dim:
                                    if z_val.shape[0] < vpl_latent_dim:
                                        padding = torch.zeros(vpl_latent_dim - z_val.shape[0], device=z_val.device)
                                        z_val = torch.cat([z_val, padding])
                                    else:
                                        z_val = z_val[:vpl_latent_dim]
                            z_list_1d.append(z_val)
                        z = torch.stack(z_list_1d).to(self.device)  # (batch_size, latent_dim)
                        logger.info(f"Using client-specific average z for generation: z.shape={z.shape}, expected (batch_size={batch_size}, latent_dim={vpl_latent_dim})")
                        if z.shape != (batch_size, vpl_latent_dim):
                            logger.error(f"z shape is incorrect: {z.shape}, expected ({batch_size}, {vpl_latent_dim})")
                            # Force reshape
                            z = z.view(batch_size, -1)
                            if z.shape[1] != vpl_latent_dim:
                                if z.shape[1] < vpl_latent_dim:
                                    padding = torch.zeros(batch_size, vpl_latent_dim - z.shape[1], device=z.device)
                                    z = torch.cat([z, padding], dim=1)
                                else:
                                    z = z[:, :vpl_latent_dim]
                            logger.info(f"z after force reshape: {z.shape}")
                
                # Priority 2: Try to get z from data (if already generated)
                if z is None and 'z' in input_data[0]:
                    z_list = [data.get('z', None) for data in input_data]
                    if all(z_val is not None for z_val in z_list):
                        z = torch.stack([
                            torch.tensor(z_val) if not isinstance(z_val, torch.Tensor) else z_val
                            for z_val in z_list
                        ]).to(self.device)
                        logger.debug(f"Using z from data: shape {z.shape}")
                
                # Priority 3: Use overall average z (if client_average_z_dict available but no client_id)
                if z is None and client_average_z_dict is not None and len(client_average_z_dict) > 0:
                    all_z_mus = torch.stack(list(client_average_z_dict.values()))
                    overall_avg_z = all_z_mus.mean(dim=0)  # (latent_dim,)
                    z = overall_avg_z.unsqueeze(0).repeat(batch_size, 1).to(self.device)  # (batch_size, latent_dim)
                    logger.debug(f"Using overall average z for generation: shape {z.shape}")
                
                # Priority 4: Infer z from input (fallback - not recommended)
                if z is None:
                    logger.warning("No client average z available. Inferring z from input (this is less accurate).")
                    # Use feature_extractor and variational_encoder to infer z
                    from federatedscope.llm.rlhf.load_vpl_components import load_vpl_components_from_checkpoint
                    variational_encoder, feature_extractor, _, _ = load_vpl_components_from_checkpoint(
                        selector_ckpt_path, self.config, device=self.device
                    )
                    
                    if variational_encoder is not None and feature_extractor is not None:
                        # Ensure input_ids and attention_mask are on model device
                        input_ids = input_ids.to(model_device)
                        attention_mask = attention_mask.to(model_device)
                        # Get embeddings
                        input_embeddings = model.get_input_embeddings()(input_ids)
                        # Pool embeddings (mean over sequence length, masked)
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled_embeddings = (input_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        # Extract features
                        # For standalone RL, we only have single prompt, so we use embedding-based features
                        vpl_use_feature_difference = getattr(self.config.llm, 'vpl_use_feature_difference', True)
                        vpl_use_llm_feature_extractor = getattr(self.config.llm, 'vpl_use_llm_feature_extractor', True)
                        vpl_use_difference_only = getattr(self.config.llm, 'vpl_use_difference_only', False)
                        
                        if vpl_use_llm_feature_extractor and vpl_use_feature_difference:
                            if vpl_use_difference_only:
                                features = feature_extractor(pooled_embeddings)
                            else:
                                # Expect [chosen, rejected, difference] but we only have one embedding
                                pooled_embeddings_3x = torch.cat([pooled_embeddings, pooled_embeddings, pooled_embeddings], dim=-1)
                                features = feature_extractor(pooled_embeddings_3x)
                        else:
                            features = feature_extractor(pooled_embeddings)
                        # Encode to z
                        z_mu, z_logvar = variational_encoder.encode(features)
                        z = z_mu  # Use mean during generation
                        logger.debug(f"Inferred z from input for generation: shape {z.shape}")
                
                if z is not None:
                    # Get model device (handle device_map='auto' case)
                    if hasattr(model, 'device'):
                        model_device = model.device
                    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
                        model_device = model.base_model.device
                    else:
                        try:
                            model_device = next(model.parameters()).device
                        except:
                            model_device = self.device
                    
                    # Ensure z is on model device
                    z = z.to(model_device)
                    
                    # Ensure z is 2D: (batch_size, latent_dim)
                    # Handle various z shapes - flatten all extra dimensions
                    original_shape = z.shape
                    if z.dim() > 2:
                        # Flatten all dimensions except batch
                        z = z.view(z.shape[0], -1)
                        logger.debug(f"Flattened z from {original_shape} to {z.shape}")
                    elif z.dim() == 1:
                        # Add batch dimension if missing
                        z = z.unsqueeze(0)
                        logger.debug(f"Added batch dimension to z: {z.shape}")
                    
                    # Ensure z has correct latent_dim
                    vpl_latent_dim = getattr(self.config.llm, 'vpl_latent_dim', 32)
                    if z.shape[-1] != vpl_latent_dim:
                        logger.warning(f"z shape mismatch: expected latent_dim={vpl_latent_dim}, got {z.shape[-1]}. Reshaping...")
                        if z.shape[-1] < vpl_latent_dim:
                            # Pad with zeros
                            padding = torch.zeros(z.shape[0], vpl_latent_dim - z.shape[-1], device=z.device)
                            z = torch.cat([z, padding], dim=-1)
                        else:
                            # Truncate
                            z = z[:, :vpl_latent_dim]
                    
                    # Project z to embedding space and inject into input embeddings
                    # CRITICAL: Ensure z is 2D: (batch_size, latent_dim) before passing to z_to_embedding
                    logger.info(f"z shape before z_to_embedding: {z.shape}, dim={z.dim()}")
                    
                    # Force z to be 2D: (batch_size, latent_dim)
                    if z.dim() != 2:
                        logger.error(f"z has wrong dimension {z.dim()}, shape {z.shape}. Forcing to 2D...")
                        # Flatten all dimensions except the first (batch) dimension
                        z = z.view(z.shape[0], -1)
                        logger.info(f"z after flattening: {z.shape}")
                    
                    # Ensure z has correct latent_dim
                    if z.shape[1] != vpl_latent_dim:
                        logger.error(f"z has wrong latent_dim: expected {vpl_latent_dim}, got {z.shape[1]}. Reshaping...")
                        if z.shape[1] < vpl_latent_dim:
                            # Pad with zeros
                            padding = torch.zeros(z.shape[0], vpl_latent_dim - z.shape[1], device=z.device)
                            z = torch.cat([z, padding], dim=1)
                        else:
                            # Truncate
                            z = z[:, :vpl_latent_dim]
                        logger.info(f"z after reshaping: {z.shape}")
                    
                    # Now z should be (batch_size, latent_dim)
                    # Ensure z has the same dtype as z_to_embedding
                    if z.dtype != z_to_embedding.weight.dtype:
                        z = z.to(z_to_embedding.weight.dtype)
                        logger.debug(f"Converted z dtype from {z.dtype} to {z_to_embedding.weight.dtype}")
                    z_embedding = z_to_embedding(z)  # (batch_size, embedding_dim)
                    logger.info(f"z_embedding shape after z_to_embedding: {z_embedding.shape}, z shape: {z.shape}, z_embedding dtype: {z_embedding.dtype}")
                    
                    # Ensure z_embedding is 2D: (batch_size, embedding_dim)
                    if z_embedding.dim() != 2:
                        logger.error(f"z_embedding has wrong dimension {z_embedding.dim()}, shape {z_embedding.shape}. Forcing to 2D...")
                        z_embedding = z_embedding.view(z_embedding.shape[0], -1)
                        logger.info(f"z_embedding after flattening: {z_embedding.shape}")
                    
                    input_embeddings = model.get_input_embeddings()(input_ids)  # (batch_size, seq_len, embedding_dim)
                    actual_embedding_dim = input_embeddings.shape[-1]
                    
                    # Verify z_embedding dimension matches input_embeddings
                    if z_embedding.shape[-1] != actual_embedding_dim:
                        logger.error(f"z_embedding dimension ({z_embedding.shape[-1]}) does not match "
                                    f"input_embeddings dimension ({actual_embedding_dim}). Cannot inject z.")
                        # Fall back to standard generation without z
                        return model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            **kwargs
                        )
                    
                    # Expand z_embedding to match input_embeddings shape: (batch_size, embedding_dim) -> (batch_size, seq_len, embedding_dim)
                    seq_len = input_embeddings.shape[1]
                    z_embedding = z_embedding.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
                    logger.debug(f"z_embedding after expand: {z_embedding.shape}, input_embeddings: {input_embeddings.shape}")
                    # Match dtype to prevent Float vs BFloat16 mismatch
                    model_dtype = next(model.parameters()).dtype
                    z_embedding = z_embedding.to(dtype=model_dtype)
                    inputs_embeds = input_embeddings + z_embedding
                    inputs_embeds = inputs_embeds.to(
                        device=model_device, dtype=model_dtype)
                    attention_mask = attention_mask.to(model_device)
                    
                    # Use inputs_embeds instead of input_ids for generation
                    output_ids = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        **generate_kwargs
                    )
                else:
                    # Fallback to standard generation
                    # input_text_tokens_device already on model_device from above
                    output_ids = model.generate(**input_text_tokens_device, **generate_kwargs)
            else:
                # Standard generation without z
                # input_text_tokens_device already on model_device from above
                output_ids = model.generate(**input_text_tokens_device, **generate_kwargs)
            responses = tokenizer.batch_decode(output_ids,
                                               skip_special_tokens=True,
                                               ignore_tokenization_space=True)

            response_map = [[] for _ in input_data]
            for res in responses:
                for idx, input_text in enumerate(input_texts):
                    if input_text in res:
                        gen_res = res.replace(input_text, "").strip()
                        response_map[idx].append(gen_res.replace("</s>", ""))
                        # response_map[idx].append(
                        #     " " + gen_res.replace("</s>", ""))
                        break

            for i, data in enumerate(input_data):
                prompt_text = data.get('prompt', '')[:100] if 'prompt' in data else ''
                logger.info(f"Data {i}: prompt={prompt_text}...")
                for j, res in enumerate(response_map[i]):
                    logger.info(f'Generated {j}-th response: {res[:100]}...')

                # Create pairwise combinations with client assignment (for VPL)
                # Each pair will be conditioned with both harmless_client_id and helpful_client_id
                for output_A, output_B in combinations(response_map[i], 2):
                    new_data = copy.deepcopy(data)
                    new_data["output_A"] = output_A
                    new_data["output_B"] = output_B
                    # Keep harmless_client_id and helpful_client_id from prompt assignment (if VPL)
                    # These will be used for z conditioning during selection
                    if is_vpl_selector:
                        if 'harmless_client_id' not in new_data:
                            # Fallback if not assigned
                            new_data['harmless_client_id'] = random.choice(harmless_client_ids) if len(harmless_client_ids) > 0 else 1
                        if 'helpful_client_id' not in new_data:
                            # Fallback if not assigned
                            new_data['helpful_client_id'] = random.choice(helpful_client_ids) if len(helpful_client_ids) > 0 else 2
                    new_list_data_dict.append(new_data)

        return new_list_data_dict

    @torch.no_grad()
    def _choose_better_response(self, list_data_dict, model, tokenizer,
                                prompt):
        # Check if tokenizer is None and use generator_tokenizer as fallback
        if tokenizer is None:
            if self.generator_tokenizer is None:
                raise ValueError("tokenizer is None and generator_tokenizer is also None. Cannot perform selection.")
            logger.warning("tokenizer is None in _choose_better_response. Using generator_tokenizer as fallback.")
            tokenizer = self.generator_tokenizer
        
        choices = [tokenizer(f": {c}")["input_ids"][-1] for c in ["A", "B"]]
        logger.info(f'Choice indices: {choices}')

        for sample in list_data_dict:
            sample["fake_choice"] = random.choice([" A", " B"])

        token_dataset = LLMDataset(
            list_data_dict,
            tokenizer,
            prompt_input=prompt,
            prompt_no_input=prompt,
            output_tag="fake_choice",
        )
        dataloader = DataLoader(
            dataset=token_dataset,
            batch_size=10,
            shuffle=False,
            collate_fn=LLMDataCollator(tokenizer=tokenizer),
        )

        predicted_indices = []
        # Get model device
        model_device = self.device
        if hasattr(model, 'device'):
            model_device = model.device
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
            model_device = model.base_model.device
        else:
            try:
                model_device = next(model.parameters()).device
            except:
                model_device = self.device
        
        if hasattr(model, "adapter_names") is False or len(
                model.adapter_names) == 1:
            # No adapter or only one LoRA adapter
            for idx, data_batch in enumerate(tqdm(dataloader)):
                input_ids = data_batch["input_ids"].to(model_device)
                labels = data_batch["labels"].to(model_device)
                attention_mask = data_batch["attention_mask"].to(model_device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                _, _, predicted, _ = cal_acc(outputs.logits, labels, choices)
                predicted_indices += predicted.tolist()
        else:
            # More than one adapters (exclude "default" one)
            for idx, data_batch in enumerate(tqdm(dataloader)):
                input_ids = data_batch["input_ids"].to(model_device)
                labels = data_batch["labels"].to(model_device)
                attention_mask = data_batch["attention_mask"].to(model_device)
                collective_choices = []
                for name in model.adapter_names:
                    if name == "default":
                        continue
                    model.set_active_adapter(name)
                    model.eval()
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                    _, _, predicted, _ = cal_acc(outputs.logits, labels,
                                                 choices)
                    collective_choices.append(predicted.tolist())
                array = np.array(collective_choices).T
                predicted_indices += [
                    np.bincount(array[i]).argmax().item()
                    for i in range(len(array))
                ]

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample["choice"] = choice
            sample.pop("fake_choice", None)

        return list_data_dict

    def dpo_better_response(self):
        # generate the output
        list_train_dict = self._generate_pairwise_data(
            self.list_train_prompts,
            self.model,
            self.generator_tokenizer,
            self.generation_prompt,
            max_new_tokens=self.config.llm.max_new_token,
        )

        return self._dpo_better_response(list_train_dict, self.model,
                                         self.tokenizer,
                                         self.generation_prompt)

    @torch.no_grad()
    def _dpo_better_response(self, list_data_dict, model, tokenizer, prompt):
        for sample in list_data_dict:
            sample["fake_choice"] = random.choice([0, 1])

        dataset = LLMComparisonDataset(
            list_data_dict,
            tokenizer,
            prompt_input=prompt,
            prompt_no_input=prompt,
            output_A="output_A",
            output_B="output_B",
            choice="fake_choice",
        )

        dataloader = DataLoader(dataset)

        # Get model device
        model_device = self.device
        if hasattr(model, 'device'):
            model_device = model.device
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
            model_device = model.base_model.device
        else:
            try:
                model_device = next(model.parameters()).device
            except:
                model_device = self.device

        predicted_indices = []
        for idx, data_batch in enumerate(tqdm(dataloader)):
            win_input_ids = data_batch["win_input_ids"].to(model_device)
            win_labels = data_batch["win_labels"].to(model_device)
            win_attention_mask = data_batch["win_attention_mask"].to(model_device)
            ref_win_outputs = model(
                disable_adapter=True,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask,
            )
            ref_win_logps = _get_batch_logps(ref_win_outputs.logits,
                                             win_labels,
                                             average_log_prob=False)
            policy_win_outputs = model(
                disable_adapter=False,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask,
            )
            policy_win_logps = _get_batch_logps(policy_win_outputs.logits,
                                                win_labels,
                                                average_log_prob=False)

            lose_input_ids = data_batch["lose_input_ids"].to(model_device)
            lose_labels = data_batch["lose_labels"].to(model_device)
            lose_attention_mask = data_batch["lose_attention_mask"].to(model_device)
            ref_lose_outputs = model(
                disable_adapter=True,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask,
            )
            ref_lose_logps = _get_batch_logps(ref_lose_outputs.logits,
                                              lose_labels,
                                              average_log_prob=False)
            policy_lose_outputs = model(
                disable_adapter=False,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask,
            )
            policy_lose_logps = _get_batch_logps(policy_lose_outputs.logits,
                                                 lose_labels,
                                                 average_log_prob=False)

            # DPO for reward calculation
            _, win_rewards, lose_rewards = dpo_loss(
                policy_win_logps,
                policy_lose_logps,
                ref_win_logps,
                ref_lose_logps,
                beta=1.0,
            )

            predicted = torch.where(
                win_rewards.cpu() > lose_rewards.cpu(),
                torch.zeros(len(win_input_ids)),
                torch.ones(len(win_input_ids)),
            )
            predicted_indices += predicted.tolist()

        for choice, sample in zip(predicted_indices, list_data_dict):
            sample["choice"] = choice
            sample.pop("fake_choice", None)

        return list_data_dict
