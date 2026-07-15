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

## 0. One-command provision (automated)

The scripted path — codifies everything in sections 1–5 plus every gotcha below.
Prereqs: `vastai` CLI authed, and an SSH key registered with vast whose private
half is at `$SSH_KEY` (default `~/.ssh/vast_kanjiland`). Publish the code + data
tarballs somewhere the box can `curl` (public S3 objects are simplest — see the
presigned-URL gotcha).

```bash
# provision + onboard + launch, one box, first shard of two
CODE_URL=https://.../code.tar.gz DATA_URL=https://.../data.tar.gz \
  SHARD=0/2 bash scripts/vast_up.sh kanjiland-a
# second box, second shard
CODE_URL=... DATA_URL=... SHARD=1/2 bash scripts/vast_up.sh kanjiland-b

# ... sweep runs ~3.5h (one wave, 8 runs/box @ ~7.8 steps/s under contention) ...

# pull results from each box (ssh line is printed by vast_up.sh), then:
bash scripts/vast_down.sh kanjiland      # destroys BOTH (label prefix match)
```

`vast_up.sh` walks the cheapest reliable offers until one actually boots (dead
hosts are normal), waits for sshd, then pipes `cloud_bootstrap.sh` in. Pass
`RUN_SWEEP=0` to onboard only. The manual sections below are the reference the
scripts automate.

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

## Onboarding gotchas (why cloud_bootstrap.sh looks the way it does)

Each of these cost real time on the first live sweep (2026-07). The scripts now
handle them; this is the "why" so nobody re-simplifies them back into a bug.

1. **No compiler on stock GPU images.** `numpy`'s sdist and `torch.compile`'s
   inductor C++ codegen both need gcc. Symptom: *"Build failures usually
   indicate a problem with the build environment"* deep into `uv sync`, or a
   compile that silently falls back to eager. Fix: `apt-get install
   build-essential` first thing.
2. **`.python-version` pins 3.13.** Great locally (compiler present), but 3.13
   has wheel gaps (numpy 1.26.4) → source build → see #1. uv also *silently*
   falls back to its managed 3.13 even when you pass `--python 3.11` for the
   sync, because `uv run` re-reads `.python-version`. Fix: `echo 3.11 >
   .python-version` on the box (bootstrap does this). 3.11 has wheels for the
   whole stack. *(Repo-side option: relax the pin to 3.11 so this vanishes.)*
3. **The `data` extra needs a compiler too** (`fasttext-wheel`). It's the
   offline corpus pipeline, not needed to train/eval — onboard with
   `--extra eval` only.
4. **Presigned S3 URLs from temporary creds expire in minutes.** SSO/STS creds
   (`ASIA…` keys) mint short-lived signatures → `ExpiredToken` by the time the
   box curls them. Fix: make the two tarball objects public-read (bucket policy
   on just those keys), or use a long-lived IAM key. Re-lock the bucket after.
5. **Hosts fail to start the container** ("failed to create task"). Not your
   bug — destroy and rent another. `vast_up.sh` does this automatically.
6. **Detach launches properly.** A sweep started over ssh dies with the session
   unless it's `setsid … </dev/null &`. And never `pkill -f ablate.py` from a
   shell whose own command line contains "ablate.py" — it matches itself
   (exit 143). Kill by PID, or match the running `python …/ablate.py` via `[a]`.
