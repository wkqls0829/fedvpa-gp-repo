import os
import sys
import argparse

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

sys.setrecursionlimit(100000)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.rlhf.standalone_training import \
    RLHF_finetuning

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    # Create new parser for selector (reward model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector-cfg-file',
                        dest='selector_cfg_file',
                        help='Selector config file path',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('--early-exiting',
                        dest='early_exiting',
                        help="DPO training data generation only",
                        action="store_true")
    selector_args, extra = parser.parse_known_args()

    # Load the LLM config (init_cfg)
    init_cfg = global_cfg.clone()
    args = parse_args(extra)

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    # Indicate this is an RLHF process
    init_cfg.llm.rlhf = True

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # Load the selector config (selector_cfg) - only if selector config file is provided
    selector_cfg = None
    if selector_args.selector_cfg_file:
        selector_cfg = global_cfg.clone()
        selector_cfg.merge_from_file(selector_args.selector_cfg_file)
        selector_cfg.freeze(save=False)

    init_cfg.freeze()

    # start rlhf training - get device first
    gpu_manager = GPUManager(gpu_available=init_cfg.use_gpu,
                             specified_device=init_cfg.device)
    _server_device = gpu_manager.auto_choice()
    
    # load selector - only if selector config file is provided (e.g., for VPL methods)
    # FedDPO and other non-VPL methods don't need a selector
    selector_model = None
    selector_tokenizer = None
    if selector_args.selector_cfg_file:
        # Check if selector_cfg has valid model.type with @ format
        if hasattr(selector_cfg, 'model') and hasattr(selector_cfg.model, 'type'):
            if '@' in selector_cfg.model.type:
                selector_backbone_name, _ = selector_cfg.model.type.split('@')
                selector_model = get_llm(selector_cfg,
                                         load_from_prev_ckpt=True,
                                         device_map=None)  # Use None to load on CPU first, then move to device
                selector_model = selector_model.to(_server_device)
                selector_tokenizer, _ = get_tokenizer(selector_backbone_name,
                                                      selector_cfg.data.root,
                                                      selector_cfg.llm.tok_len)
            else:
                raise ValueError(f"Selector config model.type must be in format 'model_name@backend', got: {selector_cfg.model.type}")
        else:
            raise ValueError("Selector config file provided but model.type is missing")

    # load llm - use specified device instead of 'auto'
    model_name, _ = init_cfg.model.type.split('@')
    model = get_llm(init_cfg, device_map=None)  # Use None to load on CPU first, then move to device
    model = model.to(_server_device)
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root,
                                 init_cfg.llm.tok_len)
    generator_tokenizer, _ = get_tokenizer(model_name,
                                           init_cfg.data.root,
                                           init_cfg.llm.tok_len,
                                           padding_side="left")
    rlhf_trainer = RLHF_finetuning(
        model,
        tokenizer,
        init_cfg,
        selector_model,
        selector_tokenizer,
        generator_tokenizer,
        device=_server_device,
        selector_cfg=selector_cfg,  # Pass selector config to use its client_num
    )
    rlhf_trainer.train(early_exiting=selector_args.early_exiting)
