# Reproducing the FedVPA-GP main-table results

This walks through reproducing the FedVPA-GP rows of Table 1 on HH-RLHF with Qwen2-0.5B at K = 10 and K = 100 clients. Read [../README.md](../README.md) first for the method overview and the hyperparameter table.

## Requirements

A single CUDA GPU with at least 48 GB. Both stages were developed on an RTX A6000. Approximate wall-clock on that hardware:

| Stage | K=10 | K=100 |
|---|---|---|
| Stage 1 selector (50 rounds) | ~4 h | ~7 h |
| Stage 2 DPO (50 rounds, GPT eval every 10) | ~10 h | ~15 h |

Stage 2 is dominated by response generation and GPT-judged evaluation rather than by the DPO updates themselves.

You also need an OpenAI API key. Stage 2 evaluates with `gpt-4o-mini` as the judge, at 100 samples per evaluated round for each of helpfulness and harmlessness. Across the 5 evaluated rounds of a 50-round run that is roughly 1,000 judged comparisons.

## Step 1 — Environment

```bash
conda create -n fedvpagp python=3.10 -y
conda activate fedvpagp

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .[llm]

cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...
```

Verify the configs parse before committing to a long run:

```bash
python -c "
from federatedscope.core.configs.config import global_cfg
c = global_cfg.clone(); c.merge_from_file('cfg/selector_n100.yaml')
print('clients', c.federate.client_num, '| lambda', c.llm.vpl_orthogonal_weight,
      '| manual labels', c.llm.vpl_use_manual_orthogonal_labels)
"
```

Expected: `clients 100 | lambda 1.0 | manual labels True`.

## Step 2 — Data

Nothing to do manually. On first run the loader pulls the `harmless-base` and `helpful-base` subsets of `Anthropic/hh-rlhf` from the Hugging Face Hub, tags every example with its source category, and hands the tagged set to the `meta` splitter, which assigns half the clients only helpfulness pairs and half only harmlessness pairs.

If the Hub download fails, `federatedscope/llm/dataloader/hh_rlhf.py` falls back to a previously downloaded arrow cache under `~/.cache/huggingface/datasets/Anthropic___hh-rlhf`, matching on the dataset-info checksums rather than the directory name. If you have neither Hub access nor a populated cache, the run raises `FileNotFoundError: hh-rlhf cache root missing`.

Point `DATA_ROOT` somewhere with room for the processed dataset if your home directory is small.

## Step 3 — Stage 1, federated selector

```bash
bash scripts/run_selector.sh 100 0
```

The two positional arguments are the client count (`10` or `100`) and the GPU index. Path overrides come from the environment:

```bash
DATA_ROOT=/mnt/data CKPT_DIR=/mnt/ckpt bash scripts/run_selector.sh 100 0
```

What runs: each sampled client extracts hidden representations for the chosen and rejected responses from the frozen base model, forms the difference embedding `Δh = h_chosen − h_rejected` over response tokens only, passes it through the feature extractor and variational encoder to get `q(z|D_i) = N(μ_i, σ_i²)`, samples `z` by reparameterization, and adds a latent-conditional term to the preference logits. The local objective is `L_recon + β·L_KL + λ·L_ortho`. Clients upload `(θ_i, ϕ_i, μ̄_i, σ̄_i², |D_i|)`; the server does FedAvg, rebuilds the mixture prior, assigns prototype labels, and broadcasts both back.

Note that `prior_logits` are excluded from FedAvg on purpose — each client keeps its own Gumbel-Softmax weights, since the point is that different clients weight their peers differently.

Watch for these in the log:

- `vpl_kl_loss` should settle to a small nonzero value. Driven to zero means posterior collapse: the latent has stopped carrying information.
- `vpl_orthogonal_loss` should fall as the prototypes separate.
- `acc` (preference prediction accuracy) should climb past 0.9 by the final rounds.
- t-SNE plots land in `exp/{expname}/sub_exp_{timestamp}/cross_client_z_tsne_round_{n}.png` every 10 rounds — note the `sub_exp_` subdirectory, not the parent. Helpfulness and harmlessness clients should form visibly distinct clusters around their prototypes. If they overlap, the orthogonal loss is not doing its job.

A reference K=10 run on one A6000 took **4h14m** for the 50 rounds, converging to `acc` 0.993, `vpl_reconstruction_loss` 0.020, `vpl_orthogonal_loss` 0.523, and `vpl_kl_loss` holding steady around 16.

The final checkpoint is written with a `final_` prefix, so `cfg/selector_n100.yaml`'s `save_to: checkpoints/selector_n100.ckpt` produces `checkpoints/final_selector_n100.ckpt`. Stage 2 needs that exact file.

### Not every client ends up with a learned z

Near the end of Stage 1 the server logs something like:

```
Computed average z for 7/10 clients for checkpoint saving (for RL training)
WARNING: Missing z values for clients: [1, 7, 8]
```

This is expected, not a failure. `client_average_z_dict` is rebuilt from the clients sampled in recent rounds, so clients that happened not to be sampled late in training carry no entry. Stage 2 fills those gaps with category-aware defaults in `_fill_missing_client_z` (`federatedscope/llm/rlhf/load_vpl_components.py`), so training proceeds either way.

Coverage is partial by design at every scale — the K=100 run behind the published numbers saved z for 40 of 100 clients. Only worry if the count is zero or very close to it, which would mean posterior collection never worked at all.

## Step 4 — Stage 2, z-conditioned DPO

```bash
bash scripts/run_rl.sh 100 0
```

The launcher refuses to start if the Stage 1 checkpoint is missing, and warns if `OPENAI_API_KEY` is unset.

Note the different entry point: Stage 2 runs `federatedscope/llm/rlhf/main.py`, not `federatedscope/main.py`, and it needs `--selector-cfg-file` so it can rebuild the selector architecture before loading the checkpoint. The launcher passes this for you.

What runs: `client_average_z_dict` from the Stage 1 checkpoint maps each client to its mean latent. For every DPO sample the corresponding `z` is injected as `inputs_embeds = input_embeddings(input_ids) + z_to_embedding(z)`, for both the policy and the reference model. `z_to_embedding` stays trainable through DPO. Generation and evaluation are conditioned the same way, via `rlhf_use_variational_generation`.

Checkpoints are saved every 10 rounds as `checkpoints/{round}_rl_n100.ckpt`.

## Step 5 — Reading the results

Evaluation runs every 10 rounds and logs:

```
test_helpfulness_winrate: 55.18
test_harmlessness_winrate: 82.31
```

Collect them across rounds:

```bash
grep -oE "'Round': [0-9]+, 'Results_raw': \{[^}]*\}" outputs/rl_n100.log \
  | sed -E "s/.*'Round': ([0-9]+).*helpfulness_winrate': ([0-9.]+), 'test_harmlessness_winrate': ([0-9.]+).*/round \1: helpful=\2 harmless=\3/"
```

The reported operating point is the **Pareto-best round**, not the last one. Win-rates rise, peak somewhere in the middle of training, and then decline as the policy overfits the reward model — so evaluate the saved checkpoints and take the round that is best on both axes jointly. Taking the final round by default will understate the result.

## Interpreting run-to-run variation

Two sources of noise are worth knowing about before you conclude a run failed to reproduce.

**The judge is stochastic.** At 100 samples per axis the 95% confidence interval is roughly ±9.8 percentage points. A single run landing several points off the published mean is expected. Do not read a 30-sample evaluation as meaningful at all — the interval there is about ±18 points, wide enough to reverse a method ranking.

**Internal-model evaluation is biased, not just noisy.** The framework can score win-rates with the trained reward model instead of the GPT judge. Those numbers run tens of points high, because the policy is being graded by the very model it was optimized against. The configs here set `use_gpt_api_for_winrate: true` for that reason. If you see implausibly high helpfulness alongside a harmlessness figure pinned at exactly 0.0, check the log for `Aborting harmlessness winrate eval` — that is an aborted evaluation being reported as a score, not a real measurement.

## Troubleshooting

**`CUDA error: invalid device ordinal`** — pass the GPU index as the launcher's second argument rather than setting `CUDA_VISIBLE_DEVICES`; the config's `device` field indexes visible devices.

**OOM during Stage 1** — z-conditioning raises activation memory noticeably. `batch_size: 4` with `grad_accum_step: 8` fits in 48 GB; 8 does not. To recover the effective batch size, halve the batch and double the accumulation.

**Stage 2 loss in the thousands** — a normal DPO loss here is near 1. Loss exploding means `reward_coeff` or the learning rate is too high for the client count; K=100 needs `reward_coeff: 0.1` and `lr: 1e-5`.

**Harmlessness win-rate stuck at 0.0** — evaluation prompts arrive grouped by client, so a low `max_iterations_for_winrate` can exhaust the budget inside one preference group and never reach the other. The configs set 6000, which is enough for K=100.

**Stage 2 collapses to one preference** — the selector checkpoint has no `z_to_embedding`, so Stage 2 built a random one. This happens when Stage 1 ran without the z-conditioning block. Confirm those four settings are present in the Stage 1 config and retrain the selector.
