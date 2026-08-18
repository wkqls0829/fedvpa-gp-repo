# Environment setup

The pinned versions in `requirements.txt` are the exact ones from the environment the reported results were produced in, captured from a working install rather than written from memory.

| Component | Version |
|---|---|
| OS | Linux (kernel 6.8) |
| Python | 3.10.16 |
| PyTorch | 2.2.0 + CUDA 12.1 |
| GPU | NVIDIA RTX A6000, 48 GB |
| CUDA driver | 12.1-compatible |

A single GPU is enough. Both stages run in `standalone` mode, which simulates all clients sequentially in one process rather than distributing them.

## Install

```bash
conda create -n fedvpagp python=3.10 -y
conda activate fedvpagp

# PyTorch must come from the CUDA index, so it is installed separately.
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# --no-deps because setup.py carries the upstream FederatedScope dependency
# ranges, which are broader than the pins above and will silently upgrade
# numpy back out of the working range.
pip install -e . --no-deps
```

## Verify

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.2.0 12.1 True

python -c "
from federatedscope.core.configs.config import global_cfg
c = global_cfg.clone(); c.merge_from_file('cfg/selector_n10.yaml')
print('clients', c.federate.client_num, '| lambda', c.llm.vpl_orthogonal_weight)
"
# expect: clients 10 | lambda 1.0
```

## Version constraints that matter

Most pins are just "what was installed", but three are load-bearing.

**numpy must stay below 1.23.** `scikit-learn==1.0.2` and `scipy==1.7.3` are built against the older numpy ABI. scikit-learn is not optional here — the server runs balanced k-means over the collected client posterior means every round to assign orthogonal prototype labels — so a numpy upgrade that breaks sklearn's import takes the whole method down, not just a plotting path.

**protobuf must stay at 3.20.x.** grpcio-tools pulls the FederatedScope message definitions through protobuf, and 4.x changes the generated-code contract. If you see `Descriptors cannot not be created directly`, protobuf has been upgraded.

**transformers 4.51.3 with peft 0.15.2.** The z-conditioning path passes `inputs_embeds` rather than `input_ids` through a PEFT-wrapped model. This combination is known to work; other pairings have moved the signature of the wrapped `forward`.

## API key

Stage 2 evaluation uses the OpenAI API with `gpt-4o-mini` as the judge.

```bash
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...
```

`scripts/run_rl.sh` sources `.env` if `OPENAI_API_KEY` is not already exported, and warns rather than failing silently if it finds neither.

## Data and model caches

Both are pulled from the Hugging Face Hub on first run and cached under `~/.cache/huggingface`:

- `Anthropic/hh-rlhf`, the `harmless-base` and `helpful-base` subsets, about 880 MB
- `Qwen/Qwen2-0.5B`, about 950 MB

Set `HF_HOME` before the first run if you want them somewhere other than your home directory. If the Hub is unreachable but a previous download exists, `federatedscope/llm/dataloader/hh_rlhf.py` falls back to the on-disk arrow cache automatically.

## GPU selection

Pass the GPU index as the second argument to either launcher:

```bash
bash scripts/run_selector.sh 10 6    # GPU 6
```

The launcher passes it through as the config's `device` field. Do not also set `CUDA_VISIBLE_DEVICES` — `device` indexes the *visible* devices, so combining the two produces `CUDA error: invalid device ordinal`.
