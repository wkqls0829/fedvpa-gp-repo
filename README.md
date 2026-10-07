# FedVPA-GP

Reference implementation of **Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences** (ICML 2026).

Jabin Koo, Hoyoung Kim, Minwoo Jang, Jungseul Ok — POSTECH and National AI Research Lab.

Federated alignment methods such as FedDPO and FedBiscuit train a single monolithic reward model, which averages away genuinely conflicting user preferences: a client who wants maximally helpful answers and a client who wants maximally cautious ones are fitted to the same reward. Applying variational preference learning (VPL) inside federated learning is the natural fix, but it collapses in practice — local datasets are small and heterogeneous, so the posterior `q(z|D_i)` degenerates toward the prior and the latent stops carrying preference information.

FedVPA-GP addresses that collapse with two mechanisms:

- **Federated Mixture Prior.** Instead of a fixed `N(0, I)` prior, the server aggregates peer client posteriors into `p_mixture(z) = Σ_j w_j · N(z; μ_j, σ_j²)`, with the mixture weights `w_j` produced by a learnable Gumbel-Softmax relaxation. Clients are pulled toward peers with similar preference structure rather than toward an uninformative origin. The KL term is computed with a log-sum-exp trick for numerical stability.
- **Orthogonal Loss.** `M` learnable prototype vectors, initialised as an orthonormal basis via QR decomposition, anchor the latent space. A pull term aligns each client's `z` to its assigned prototype and a Frobenius orthonormality constraint `‖PPᵀ − I‖²_F` keeps the prototypes separated, so distinct preference types occupy distinct subspaces.

## Quickstart

```bash
conda create -n fedvpagp python=3.10 -y && conda activate fedvpagp
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e . --no-deps

bash scripts/run_selector.sh 10 0     # Stage 1, K=10 clients, GPU 0  (~4h)
bash scripts/run_rl.sh       10 0     # Stage 2, needs Stage 1 first  (~10h)
```

Stage 1 needs no API key. Stage 2 needs one only to compute win-rates — see [Do you need an OpenAI API key?](#do-you-need-an-openai-api-key) below.

## Scope of this repository

This is a **minimal reproduction package**, not the full research codebase. It contains exactly what is needed to reproduce the FedVPA-GP rows of the main table on HH-RLHF with Qwen2-0.5B at **K = 10** and **K = 100** clients.

Deliberately not included: the FedDPO / FedBiscuit / FedVPL baseline configurations, the ablation study, the unseen-client generalization experiment, the Gemma-2B and UltraFeedback runs, the hyperparameter search, and the SLURM batch tooling. The underlying framework still supports those; only the experiment configurations were removed.

Stage 1 of the K=10 configuration has been run end to end from this repository to confirm it is self-contained: 50 rounds, converging to preference accuracy 0.993 with the KL term holding at a stable non-zero plateau. Reference values to compare against are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Results

GPT-4 win-rate (%) on HH-RLHF with the strict non-IID partition, Qwen2-0.5B. These are the published Table 1 numbers that the configurations here correspond to.

| Method | K=10 Helpful | K=10 Harmless | K=100 Helpful | K=100 Harmless |
|---|---|---|---|---|
| FedDPO | 48.12 ± 1.52 | 77.34 ± 2.41 | 41.48 ± 2.32 | 67.15 ± 2.64 |
| FedBiscuit | 48.85 ± 1.41 | 75.12 ± 2.28 | 42.33 ± 2.11 | 69.42 ± 2.45 |
| FedVPL | 62.24 ± 1.25 | 84.56 ± 1.95 | 53.05 ± 1.88 | 77.34 ± 2.21 |
| **FedVPA-GP** | **66.45 ± 1.12** | **89.21 ± 1.68** | **55.18 ± 1.55** | **82.31 ± 2.05** |

Baseline rows are shown for context; only the FedVPA-GP row is reproducible from this repository.

The judge is `gpt-4o-mini`, which is what these configs use and what the paper specifies for cost efficiency — so expect to see that model name rather than `gpt-4` in the logs. Each evaluated round judges the fine-tuned model's response against the frozen base model's on the same prompt, and the win-rate is the share of comparisons the fine-tuned model wins.

FedVPA-GP is the only method that improves on **both** axes simultaneously — a Pareto improvement rather than a trade of helpfulness for harmlessness.

One thing to know before you compare a run against this table: **the win-rate peak does not fall on the final round.** Win-rates rise, peak partway through training, then decline as the policy overfits the reward model. That is why Stage 2 checkpoints every 10 rounds — the reported operating point is the Pareto-best round across the evaluated checkpoints, not round 50. Taking the last round by default will understate the result.

## Repository layout

```
cfg/                            experiment configurations
  selector_n10.yaml             Stage 1, K=10   (annotated reference config)
  selector_n100.yaml            Stage 1, K=100
  rl_n10.yaml                   Stage 2, K=10
  rl_n100.yaml                  Stage 2, K=100
scripts/
  run_selector.sh               Stage 1 launcher
  run_rl.sh                     Stage 2 launcher
federatedscope/
  llm/trainer/vpl_gp_reward_choice_trainer.py    FedVPA-GP trainer
  llm/trainer/vpl_reward_choice_trainer.py       base VPL trainer
  llm/model/variational_encoder_gp.py            encoder + Gumbel-Softmax prior
  llm/model/variational_encoder.py               base variational encoder
  llm/llm_local/server.py                        mixture-prior aggregation, prototype labels
  llm/llm_local/client.py                        posterior upload, prior update
  llm/rlhf/main.py                               Stage 2 entry point
  llm/rlhf/standalone_training.py                z-conditioned DPO loop
  llm/metric/winrate_metrics.py                  GPT win-rate evaluation
  llm/dataloader/hh_rlhf.py                      HH-RLHF loading and partitioning
  core/                                          FederatedScope framework
docs/
  ENVIRONMENT.md                environment setup and version constraints
  REPRODUCE.md                  step-by-step reproduction guide
requirements.txt                pinned to the environment the results came from
.env.example                    template for the optional OpenAI key
LICENSE  NOTICE                 Apache 2.0 and upstream attribution
```

Checkpoints, logs, `exp/` output, and the dataset cache are gitignored; nothing in the repo is a build artifact.

## Installation

Requires Python 3.10 and a CUDA-capable GPU. Both stages were run on a single 48 GB RTX A6000.

```bash
conda create -n fedvpagp python=3.10 -y
conda activate fedvpagp

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

`--no-deps` matters: `setup.py` carries the upstream FederatedScope dependency ranges, which are broader than the pins in `requirements.txt` and will upgrade numpy back out of the range scikit-learn 1.0.2 needs. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the full walkthrough and the three pins that are load-bearing.

### Do you need an OpenAI API key?

Only for the win-rate numbers. Specifically:

- **Stage 1 needs no key at all.** The selector trains and converges without ever contacting the API.
- **Stage 2 trains fine without one too.** The key is used in exactly one place — `federatedscope/llm/metric/winrate_metrics.py`, which calls `gpt-4o-mini` to judge your fine-tuned model's responses against the frozen baseline's. Without a key you still get a trained z-conditioned policy; you just get no win-rate metric to compare against Table 1.

So if you want to study or extend the method, you can skip this entirely. If you want to reproduce the published numbers, supply your own key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

`.env` is gitignored. `scripts/run_rl.sh` sources it automatically if `OPENAI_API_KEY` is not already exported, and warns rather than failing silently when it finds neither.

## Data

HH-RLHF is downloaded automatically from the Hugging Face Hub on first run — the `harmless-base` and `helpful-base` subsets of `Anthropic/hh-rlhf`. Every example is tagged with its source category, and the `meta` splitter uses that tag to build the strict non-IID partition: half the clients receive only helpfulness pairs, half receive only harmlessness pairs.

If the Hub download fails, the loader falls back to a previously downloaded arrow cache under `~/.cache/huggingface/datasets/Anthropic___hh-rlhf`. See `federatedscope/llm/dataloader/hh_rlhf.py`.

## Running

Training is two stages, and Stage 2 depends on Stage 1's checkpoint.

**Stage 1 — federated variational selector.** Clients learn a posterior `q(z|D_i)` and predict preferences conditioned on `z`. The base LLM is frozen; only the VPL components and LoRA adapters train. Loss is `L_recon + β·L_KL + λ·L_ortho`.

```bash
bash scripts/run_selector.sh 10 0     # K=10  on GPU 0
bash scripts/run_selector.sh 100 0    # K=100 on GPU 0
```

**Stage 2 — z-conditioned DPO.** Centralized DPO on the server using the converged selector as the reward model, with the policy conditioned on each client's inferred `z` through injection into the input embeddings.

```bash
bash scripts/run_rl.sh 10 0
bash scripts/run_rl.sh 100 0
```

The two positional arguments are the client count (`10` or `100`) and the GPU index. Both launchers honour `DATA_ROOT`, `CKPT_DIR`, and `LOG_DIR` environment variables, and `run_rl.sh` refuses to start if the Stage 1 checkpoint it needs is missing.

Expect roughly these wall-clock times on one A6000. The Stage 1 K=10 figure is measured; the rest are scaled from it and from earlier runs.

| | K=10 | K=100 |
|---|---|---|
| Stage 1 selector, 50 rounds | 4h15m (measured) | ~7 h |
| Stage 2 DPO, 50 rounds | ~10 h | ~15 h |

Stage 2 is dominated by response generation and GPT-judged evaluation rather than by the DPO updates. See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the detailed walkthrough, what to watch in the logs, and troubleshooting.

## Hyperparameters

Matching Table 4 (Appendix B) of the paper. Where the paper gives a model-dependent range, the value below is the one for Qwen2-0.5B, which is the model these configs use.

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Learning rate | 1e-5 | 1e-5 |
| Batch size | 4 | 1 |
| Gradient accumulation | 8 | 32 |
| Local update steps | 30 | 30 |
| Total rounds | 50 | 50 |
| KL weight (β) | 0.01 | — |
| Orthogonal weight (λ) | 1.0 | — |
| Orthonorm weight (γ) | 0.1 | — |
| Gumbel-Softmax temperature (τ) | 1.0 | — |
| Latent dimension (d) | 32 | — |
| Latent projection | MLP 32→64→32→\|C\| | — |
| Prototypes (M) | 2 | — |
| Prototype scale | 5.0 | — |
| Orthogonal label assignment | balanced k-means | — |
| Max log-variance | −4.0 | — |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 | 8 / 16 / 0.05 |
| Reward coefficient | — | 0.1 |

Two settings are easy to get wrong and both cause silent failure rather than an error, so they are worth calling out:

**The z-conditioning block must be identical in both stages.** `vpl_use_z_embedding`, `vpl_z_source`, `vpl_z_conditioning_mode`, and `vpl_adapter_dropout` appear in the Stage 1 and Stage 2 configs alike. Stage 1 needs them so it trains and saves a `z_to_embedding` module; if it is absent from the checkpoint, Stage 2 initialises one randomly and the policy collapses to a single undifferentiated preference.

**Reward coefficient and learning rate interact badly at scale.** At K=100, `reward_coeff: 0.5` combined with `lr: 5e-5` makes z-conditioned DPO diverge, with training loss reaching the thousands against a normal value near 1. The values above are the stable ones.

### One deliberate deviation from the paper

Appendix C.3.1 describes evaluation at "up to 30 samples per evaluation (for efficiency)". The configs here set `max_samples_for_reward: 100` instead.

The reason is statistical. A win-rate measured on 30 judged comparisons carries a 95% confidence interval of roughly ±18 percentage points — wide enough to reorder the methods in Table 1 by chance alone. At 100 samples the interval narrows to about ±9.8 points, which is still not tight, but is enough to make a single run interpretable. Anyone trying to tell FedVPA-GP apart from FedVPL on a single run at n=30 will not be able to.

Set it back to 30 if you want to match the stated protocol exactly and are willing to average over several seeds instead. Everything else in the table follows the paper as written.

## Citation

```bibtex
@inproceedings{koo2026fedvpagp,
  title     = {Federated Variational Preference Alignment with Gumbel-Softmax
               Prior for Personalized User Preferences},
  author    = {Koo, Jabin and Kim, Hoyoung and Jang, Minwoo and Ok, Jungseul},
  booktitle = {Proceedings of the 43rd International Conference on Machine
               Learning (ICML)},
  year      = {2026}
}
```

## License and attribution

Released under the Apache License 2.0 — see [LICENSE](LICENSE).

This project is built on [FederatedScope](https://github.com/alibaba/FederatedScope) by Alibaba Group, also Apache 2.0 licensed. The `federatedscope/core` package is largely upstream code; the FedVPA-GP contribution lives in `federatedscope/llm`. See [NOTICE](NOTICE).
