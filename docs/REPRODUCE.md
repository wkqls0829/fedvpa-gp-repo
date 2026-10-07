# Reproduction notes

Setup and commands are in the [README](../README.md). This covers what to watch while a run is going and what to do when it misbehaves.

## Data

Nothing to download manually. On first run the loader pulls `harmless-base` and `helpful-base` from `Anthropic/hh-rlhf`, tags each example with its category, and the `meta` splitter uses that tag to give half the clients only helpfulness pairs and half only harmlessness pairs.

If the Hub is unreachable, `federatedscope/llm/dataloader/hh_rlhf.py` falls back to a cached arrow copy under `~/.cache/huggingface/datasets/Anthropic___hh-rlhf`, matching on dataset-info checksums. With neither you get `FileNotFoundError: hh-rlhf cache root missing`.

## Stage 1 — selector

Each sampled client builds `Δh = h_chosen − h_rejected` over response tokens from the frozen base model, encodes it to `q(z|D_i)`, and adds a latent-conditional term to the preference logits. Loss is `L_recon + β·L_KL + λ·L_ortho`. Clients upload `(θ_i, ϕ_i, μ̄_i, σ̄_i², |D_i|)`; the server runs FedAvg, rebuilds the mixture prior, and assigns prototype labels by balanced k-means.

`prior_logits` are excluded from FedAvg by design — each client keeps its own Gumbel-Softmax weights, since the point is that clients weight peers differently.

A reference K=10 run on one A6000, 50 rounds in 4h15m:

| Round | acc | loss | KL | recon | ortho |
|---|---|---|---|---|---|
| 0 | 0.801 | 1.569 | 48.94 | 0.350 | 0.708 |
| 5 | 0.989 | 0.783 | 10.67 | 0.033 | 0.642 |
| 20 | 0.991 | 0.734 | 15.76 | 0.027 | 0.549 |
| 35 | 0.994 | 0.715 | 16.06 | 0.021 | 0.534 |
| 49 | 0.993 | 0.708 | 16.38 | 0.020 | 0.523 |

**`vpl_kl_loss` is the one to watch.** It drops sharply at round 5 when the mixture prior replaces `N(0, I)`, then holds a plateau. Decaying toward zero means posterior collapse — the latent has stopped carrying preference information, which is the failure this method exists to prevent.

t-SNE plots are written every 10 rounds to `exp/{expname}/sub_exp_{timestamp}/cross_client_z_tsne_round_{n}.png` (note the `sub_exp_` subdirectory). The two preference groups should pull apart toward their prototypes.

The final checkpoint gets a `final_` prefix: `save_to: checkpoints/selector_n10.ckpt` produces `checkpoints/final_selector_n10.ckpt`, which is what Stage 2 loads.

### "Missing z values for clients" is normal

```
Computed average z for 7/10 clients for checkpoint saving (for RL training)
WARNING: Missing z values for clients: [1, 7, 8]
```

`client_average_z_dict` is rebuilt from recently sampled clients, so ones not sampled late in training have no entry. Stage 2 fills them via `_fill_missing_client_z` in `rlhf/load_vpl_components.py`. Partial coverage happens at every scale — the K=100 run behind the published numbers saved 40 of 100. Only a count at or near zero indicates a real problem.

## Stage 2 — z-conditioned DPO

Runs through `federatedscope/llm/rlhf/main.py`, not `federatedscope/main.py`, and needs `--selector-cfg-file` to rebuild the selector architecture before loading the checkpoint. `scripts/run_rl.sh` handles this.

`client_average_z_dict` maps each client to its mean latent. Every DPO sample is conditioned as `inputs_embeds = input_embeddings(input_ids) + z_to_embedding(z)`, for both policy and reference model, with `z_to_embedding` trainable throughout. Generation and evaluation are conditioned the same way.

Collect win-rates across rounds:

```bash
grep -aoE "'Round': [0-9]+, 'Results_raw': \{[^}]*\}" outputs/rl_n10.log \
  | sed -E "s/.*'Round': ([0-9]+).*helpfulness_winrate': ([0-9.]+), 'test_harmlessness_winrate': ([0-9.]+).*/round \1: helpful=\2 harmless=\3/" \
  | grep "^round"
```

Report the Pareto-best round across checkpoints, not the last one.

## Reading the numbers

At 100 judged samples per axis the 95% interval is about ±9.8 points, so a single run landing a few points off the published mean is expected. At 30 samples it is ±18 points — wide enough to reverse a method ranking, which is why the configs here use 100.

Internal-model evaluation is biased, not merely noisy: the policy is graded by the reward model it was optimized against, which runs tens of points high. The configs set `use_gpt_api_for_winrate: true` for this reason. Implausibly high helpfulness next to a harmlessness figure pinned at exactly `0.0` means an aborted evaluation reported as a score — check the log for `Aborting harmlessness winrate eval`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `CUDA error: invalid device ordinal` | Don't set `CUDA_VISIBLE_DEVICES` as well — `device` indexes visible devices. Pass the GPU as the launcher's second argument. |
| OOM in Stage 1 | Peak is ~13 GB. If a co-tenant process is on the card, move GPUs rather than shrinking the batch. To genuinely reduce memory, halve `batch_size` and double `grad_accum_step`. |
| Stage 2 loss in the thousands | `reward_coeff` or lr too high for the client count. K=100 needs `reward_coeff: 0.1`, `lr: 1e-5`. |
| Harmlessness win-rate stuck at 0.0 | Eval prompts arrive grouped by client; a low `max_iterations_for_winrate` exhausts the budget inside one group. Configs use 6000. |
| Stage 2 collapses to one preference | Selector checkpoint has no `z_to_embedding` — Stage 1 ran without the z-conditioning block. Retrain the selector. |
