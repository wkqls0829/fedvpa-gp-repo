# FedVPA-GP

Reference implementation of **Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences** (ICML 2026).

Jabin Koo, Hoyoung Kim, Minwoo Jang, Jungseul Ok — POSTECH and National AI Research Lab.

Federated alignment methods such as FedDPO and FedBiscuit train a single monolithic reward model, which averages away genuinely conflicting user preferences: a client who wants maximally helpful answers and a client who wants maximally cautious ones are fitted to the same reward. Applying variational preference learning (VPL) inside federated learning is the natural fix, but it collapses in practice — local datasets are small and heterogeneous, so the posterior `q(z|D_i)` degenerates toward the prior and the latent stops carrying preference information.

FedVPA-GP addresses that collapse with two mechanisms:

- **Federated Mixture Prior.** Instead of a fixed `N(0, I)` prior, the server aggregates peer client posteriors into `p_mixture(z) = Σ_j w_j · N(z; μ_j, σ_j²)`, with the mixture weights `w_j` produced by a learnable Gumbel-Softmax relaxation. Clients are pulled toward peers with similar preference structure rather than toward an uninformative origin. The KL term is computed with a log-sum-exp trick for numerical stability.
- **Orthogonal Loss.** `M` learnable prototype vectors, initialised as an orthonormal basis via QR decomposition, anchor the latent space. A pull term aligns each client's `z` to its assigned prototype and a Frobenius orthonormality constraint `‖PPᵀ − I‖²_F` keeps the prototypes separated, so distinct preference types occupy distinct subspaces.

## Scope of this repository

This is a **minimal reproduction package**, not the full research codebase. It contains exactly what is needed to reproduce the FedVPA-GP rows of the main table on HH-RLHF with Qwen2-0.5B at **K = 10** and **K = 100** clients.

Deliberately not included: the FedDPO / FedBiscuit / FedVPL baseline configurations, the ablation study, the unseen-client generalization experiment, the Gemma-2B and UltraFeedback runs, the hyperparameter search, and the SLURM batch tooling. The underlying framework still supports those; only the experiment configurations were removed.

## Results

GPT-4 win-rate (%) on HH-RLHF with the strict non-IID partition, Qwen2-0.5B. These are the published Table 1 numbers that the configurations here correspond to.

| Method | K=10 Helpful | K=10 Harmless | K=100 Helpful | K=100 Harmless |
|---|---|---|---|---|
| FedDPO | 48.12 ± 1.52 | 77.34 ± 2.41 | 41.48 ± 2.32 | 67.15 ± 2.64 |
| FedBiscuit | 48.85 ± 1.41 | 75.12 ± 2.28 | 42.33 ± 2.11 | 69.42 ± 2.45 |
| FedVPL | 62.24 ± 1.25 | 84.56 ± 1.95 | 53.05 ± 1.88 | 77.34 ± 2.21 |
| **FedVPA-GP** | **66.45 ± 1.12** | **89.21 ± 1.68** | **55.18 ± 1.55** | **82.31 ± 2.05** |

Baseline rows are shown for context; only the FedVPA-GP row is reproducible from this repository.

FedVPA-GP is the only method that improves on **both** axes simultaneously. Note that the win-rate peak does not fall on the final round — it typically lands between rounds 30 and 60, which is why Stage 2 checkpoints every 10 rounds. Selecting the reported operating point means picking the Pareto-best round across the evaluated checkpoints.

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
docs/REPRODUCE.md               step-by-step reproduction guide
```

## Installation

Requires Python 3.10 and a CUDA-capable GPU. Both stages were run on a single 48 GB RTX A6000.

```bash
conda create -n fedvpagp python=3.10 -y
conda activate fedvpagp

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .[llm]
```

Stage 2 evaluation calls the OpenAI API. Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

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

Both launchers honour `DATA_ROOT`, `CKPT_DIR`, and `LOG_DIR` environment variables. Stage 1 takes roughly 7 hours at K=100; Stage 2 takes roughly 20 hours for 70 rounds, dominated by generation and GPT-judged evaluation. See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the detailed walkthrough.

## Hyperparameters

Matching Appendix A of the paper.

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Learning rate | 1e-5 | 1e-5 |
| Batch size | 4 | 1 |
| Gradient accumulation | 8 | 32 |
| Local update steps | 30 | 30 |
| Total rounds | 50 | 70 |
| KL weight (β) | 0.01 | — |
| Orthogonal weight (λ) | 1.0 | — |
| Orthonorm weight (γ) | 0.1 | — |
| Gumbel-Softmax temperature (τ) | 1.0 | — |
| Latent dimension (d) | 32 | — |
| Prototypes (M) | 2 | — |
| Prototype scale | 5.0 | — |
| Max log-variance | −4.0 | — |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 | 8 / 16 / 0.05 |
| Reward coefficient | — | 0.1 |

Two settings are easy to get wrong and both cause silent failure rather than an error, so they are worth calling out:

**The z-conditioning block must be identical in both stages.** `vpl_use_z_embedding`, `vpl_z_source`, `vpl_z_conditioning_mode`, and `vpl_adapter_dropout` appear in the Stage 1 and Stage 2 configs alike. Stage 1 needs them so it trains and saves a `z_to_embedding` module; if it is absent from the checkpoint, Stage 2 initialises one randomly and the policy collapses to a single undifferentiated preference.

**Reward coefficient and learning rate interact badly at scale.** At K=100, `reward_coeff: 0.5` combined with `lr: 5e-5` makes z-conditioned DPO diverge, with training loss reaching the thousands against a normal value near 1. The values above are the stable ones.

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
