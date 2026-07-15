# Cloud training runbook

Run training/ablation sweeps on rented GPUs to parallelize. A 52M model needs
**one GPU per run** and 24GB is plenty, so the pattern is: rent a **multi-GPU
box** (or several single-GPU instances) and run one ablation variant per GPU.

> **Recommended (as of 2026-07): Vast.ai interruptible RTX 4090 (~$0.30/GPU-hr)**
> — cheapest + most parallel (marketplace of independent single-GPU rentals),
> resume covers preemptions. Fallback: **RunPod Community 4090 (~$0.34/hr)** —
> higher new-account spend cap, better tooling. Avoid AWS/GCP (new-account GPU
> quotas start at ~0 and need slow ticket approvals — defeats a same-day sweep).
> Any recent-driver Linux box with NVIDIA GPUs works with the steps below.

## 1. Provision

A Linux box with N × (RTX 4090 / L40S / A100). Ensure a **recent NVIDIA driver**
(≥ 525; cloud images normally ship 550+). Our torch pin is CUDA 12.6 (`cu126`),
which runs on any driver ≥ 525 via minor-version compatibility — so `uv sync`
Just Works on the box. (If `torch.cuda.is_available()` is ever False, the driver
is too old; pick a newer image.)

## 2. Code + env

```bash
git clone <repo> kanjiland && cd kanjiland
git checkout m5                     # or the milestone branch
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not present
uv sync --extra data --extra eval   # torch cu126 + eval metrics
export WANDB_API_KEY=<key>          # live curves for every parallel run
```

Tokenizers (`data/processed/tokenizer-*.json`) come with the clone.

## 3. Data (fast on a fiber link)

Training reads the pretokenized **binaries**; eval reads the **test jsonls**.
Both are gitignored, so rsync them from the local box:

```bash
# ~3GB of pretokenized 16k binaries (train/valid) + the test sets
rsync -avP data/processed/tok/         cloud:kanjiland/data/processed/tok/
rsync -avP data/processed/kftt-test.jsonl data/processed/test.jsonl \
                                       cloud:kanjiland/data/processed/
```

(The vocab-size ablation additionally needs 8k/32k pretokenized binaries —
generate those with `scripts/pretokenize.py` once their tokenizers are retrained.)

## 4. Run the sweep — one GPU per variant, in parallel

```bash
# 8-GPU box: all 4 pos runs at once, then reuse GPUs for the next axis
uv run python scripts/ablate.py --base configs/m5_ablation_base.yaml \
    --name pos  --vary model.pos_encoding=rope,sinusoidal   --seeds 1,2 --devices 0,1,2,3
uv run python scripts/ablate.py --base configs/m5_ablation_base.yaml \
    --name tie  --vary model.tie_embeddings=three_way,none  --seeds 1,2 --devices 0,1,2,3
```

Parallel jobs stream to `logs/<run>-s<seed>.log`; W&B has the live curves. Each
job auto-evals into `docs/reports/m5-results.json` (seed-aggregated).

Alternatively, split across independent single-GPU instances: run
`--shard i/N` (no `--devices`) on instance *i* of *N*.

## 5. Pull results back, then STOP the box

```bash
rsync -avP cloud:kanjiland/docs/reports/m5-results.* docs/reports/
# checkpoints are large + regenerable; pull only if you want them
```

Then **shut the instance down** — idle GPUs are where cloud bills escalate.
Spot/interruptible is safe here: training checkpoints and `train.py --resume`
picks up from the latest `checkpoints/<run>/seed<N>/`.
