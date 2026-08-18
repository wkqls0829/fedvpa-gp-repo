import torch
import os
import numpy as np
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.llm.reward.reward_model_implementations import (
    GPT2HarmlessRewardModel, GPT2HelpfulRewardModel)

@torch.no_grad()
def main():
    """
    Main function to evaluate a model on the HH-RLHF dataset using reward
    models.
    This script generates responses to test prompts and evaluates them for
    harmlessness and helpfulness.
    """
    # ------------------- Initialization -------------------
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, _ = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)
    init_cfg.freeze()

    # ------------------- Load Models -------------------
    # Load your fine-tuned model
    fschatbot = FSChatBot(init_cfg)

    # Load reward models for evaluation
    harmless_reward_model = GPT2HarmlessRewardModel(device=fschatbot.device)
    helpful_reward_model = GPT2HelpfulRewardModel(device=fschatbot.device)

    # ------------------- Load Data -------------------
    # Path to the HH-RLHF test data
    # We use the harmless test set as a representative sample
    data_path = os.path.join(init_cfg.data.root, 'hhrlhf',
                             'harmless-base', 'test.jsonl.gz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"HH-RLHF dataset not found at {data_path}. "
            f"Please ensure the dataset is downloaded and placed correctly.")
    list_data_dict = load_jsonl(data_path)

    results = []

    # ------------------- Generation and Evaluation -------------------
    try:
        for sample in tqdm(list_data_dict):
            # The prompt is the full conversation history
            prompt = sample['prompt']

            generate_kwargs = dict(
                top_p=1.0,
                temperature=0.0,
                do_sample=False,
                max_new_tokens=init_cfg.llm.max_new_token,
            )
            model_completion = fschatbot.generate(prompt, generate_kwargs)

            # Calculate reward scores
            harmless_score = harmless_reward_model.get_rewards(
                [model_completion], [prompt])[0]
            helpful_score = helpful_reward_model.get_rewards(
                [model_completion], [prompt])[0]

            results.append({
                'prompt': prompt,
                'model_response': model_completion,
                'harmlessness_score': harmless_score,
                'helpfulness_score': helpful_score,
            })

    except Exception as err:
        print(f"An error occurred: {err}. Finishing evaluation.")

    # ------------------- Save and Display Results -------------------
    # Save detailed results
    results_path = os.path.join(init_cfg.outdir,
                                f'{fschatbot.curpfx}_hhrl_reward_scores.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(str(res) + '\n')

    # Calculate and print average scores
    if results:
        avg_harmlessness = np.mean(
            [res['harmlessness_score'] for res in results])
        avg_helpfulness = np.mean(
            [res['helpfulness_score'] for res in results])

        print(f"Average Harmlessness Score: {avg_harmlessness}")
        print(f"Average Helpfulness Score: {avg_helpfulness}")

        # Save average scores
        avg_scores_path = os.path.join(init_cfg.outdir,
                                       'hhrl_avg_scores.txt')
        with open(avg_scores_path, 'w') as f:
            f.write(f"Average Harmlessness Score: {avg_harmlessness}\n")
            f.write(f"Average Helpfulness Score: {avg_helpfulness}\n")
    else:
        print("No results were generated to calculate average scores.")


if __name__ == "__main__":
    main()

