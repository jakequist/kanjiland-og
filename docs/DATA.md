# Data & model artifacts — restore from S3

The large artifacts (raw corpora, the cleaned/pretokenized corpus, and trained
model weights) are **not in git** and **not fully redistributable** (the corpus is
~86% JParaCrawl). They live in a **private S3 bucket** as the project's durable
backup, so the environment can be recreated after a machine cycles — without
re-downloading JParaCrawl (slow, ~12 GB) or re-running the expensive M2 pipeline.

## The bucket

```
s3://kanjiland          # PRIVATE — bucket name is public, contents are NOT
```

Access requires **this project's AWS credentials** (AWS account `167706257251`).
The name is published here on purpose so our own agents know where to look; the
bucket has full block-public-access enabled, so nobody without credentials can read
it.

## Layout

| prefix | contents | ~size |
|:--|:--|--:|
| `s3://kanjiland/data/raw/` | raw sources: `jparacrawl/`, `kftt/`, `jesc/`, `tatoeba/` | 12.7 GB |
| `s3://kanjiland/data/processed/` | cleaned corpus (`{train,valid,test}.jsonl`), pretokenized bins (`tok/`, `tok8k/`, `tok32k/`), tokenizers, M6 KD + M7 silver data | 25 GB |
| `s3://kanjiland/checkpoints/` | trained model weights (`final.pt` per run; intermediate `step_*.pt` are NOT backed up — regenerable) | 6.7 GB |

## Restore (fresh machine)

```bash
aws s3 sync s3://kanjiland/data/raw        data/raw
aws s3 sync s3://kanjiland/data/processed  data/processed
aws s3 sync s3://kanjiland/checkpoints     checkpoints
```

`aws s3 sync` is resumable — if it times out, re-run and it skips what's already
downloaded. After restoring, `uv sync` and the environment matches.

## ⚠ Licensing — keep it private

The bucket holds **JParaCrawl**-derived data, which is research-use and **not
redistributable**. Storing it privately for this project is cloud backup (fine);
**making any object public would be redistribution (a license violation).** Do not
add public bucket policies or public-read ACLs here. The *redistributable* subset
(KFTT-derived silver/KD data, model weights) is also mirrored **publicly on
Hugging Face** — see the README. Rebuild-from-source (each user fetches JParaCrawl
under its own terms) is the license-clean path for public reproduction; this bucket
is a private convenience for us.
