"""Train the M3 Ja→En transformer (from scratch, raw PyTorch).

    uv run python scripts/train.py --config configs/m3_transformer_base.yaml
    uv run python scripts/train.py --config configs/m3_transformer_base.yaml --overfit 300

``--overfit N`` is the canonical wiring sanity check: train on a *single* batch
for N steps and watch the loss collapse to the label-smoothing floor (~1.2 at
eps=0.1/V=8k; ~0 with smoothing off). If a from-scratch model can't memorize one
batch, something is wrong (bad masking, detached graph, dead gradients) —
cheaper to find here than 10 hours into a real run.

Every run logs to W&B (CLAUDE.md rule #4); pass --no-wandb only for quick local
checks. Real runs are long — checkpoints let them resume.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from kanjiland.model import ModelConfig, Transformer
from kanjiland.tokenizer import Tokenizer
from kanjiland.train.data import TranslationDataset, make_dataloader
from kanjiland.train.device import amp_context, pick_device
from kanjiland.train.loss import label_smoothed_ce
from kanjiland.train.schedule import lr_at_step


def _bin_prefix(cfg: dict, split: str) -> Path:
    return Path(cfg.get("data", {}).get("bin_dir", "data/processed/tok")) / split


def _infinite(loader, sampler):
    """Yield batches forever, reshuffling (new buckets) each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _save_ckpt(path: Path, model, opt, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg},
        path,
    )


@torch.no_grad()
def _valid_loss(
    model, loader, pad_id: int, device: str, smoothing: float, max_batches: int
) -> float:
    model.eval()
    total_loss = total_tok = 0.0
    for i, (src, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        src, tgt = src.to(device), tgt.to(device)
        with amp_context(device):
            logits = model(src, tgt[:, :-1])
        loss, ntok = label_smoothed_ce(logits, tgt[:, 1:], pad_id, smoothing)
        total_loss += loss.item()
        total_tok += ntok.item()
    model.train()
    return total_loss / max(1.0, total_tok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--overfit", type=int, default=0, help="train on one batch for N steps")
    ap.add_argument("--steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile (fast pilots)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    seed = cfg.get("seed", 1)
    from kanjiland.train.seed import seed_everything

    seed_everything(seed)
    device = pick_device()

    # --- tokenizer + model --------------------------------------------------
    tok = Tokenizer.load(cfg["tokenizer"]["path"])
    mcfg = ModelConfig.from_dict(cfg["model"], vocab_size=tok.vocab_size)
    mcfg.pad_id = tok.pad_id
    model = Transformer(mcfg).to(device)
    print(f"model: {model.num_parameters() / 1e6:.1f}M params | device {device}")

    raw_model = model  # keep uncompiled handle for checkpointing
    if tcfg.get("compile") and device == "cuda" and not args.no_compile:
        # dynamic=True: token-budget batching produces many different (batch,
        # seq) shapes; without it torch.compile recompiles for each and thrashes.
        model = torch.compile(model, dynamic=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    # --- data ---------------------------------------------------------------
    train_ds = TranslationDataset(_bin_prefix(cfg, "train"))
    train_loader, train_sampler = make_dataloader(
        train_ds,
        tcfg["tokens_per_batch"],
        tok.pad_id,
        seed=seed,
        num_workers=0 if args.overfit else 4,
        pin_memory=device == "cuda",
    )
    valid_loader = None
    if not args.overfit:
        valid_ds = TranslationDataset(_bin_prefix(cfg, "valid"))
        valid_loader, _ = make_dataloader(
            valid_ds,
            tcfg["tokens_per_batch"],
            tok.pad_id,
            seed=seed,
            shuffle=False,
            num_workers=2,
            pin_memory=device == "cuda",
        )

    run = None
    if not args.no_wandb:
        from kanjiland.train.wandb_init import init_run

        run = init_run(args.config, extra_metadata={"params_M": raw_model.num_parameters() / 1e6})

    # --- training loop ------------------------------------------------------
    max_steps = args.overfit if args.overfit else (args.steps or tcfg["max_steps"])
    accum = 1 if args.overfit else tcfg.get("grad_accum_steps", 1)
    smoothing = tcfg.get("label_smoothing", 0.1)
    clip = tcfg.get("clip_grad", 1.0)
    warmup = tcfg["warmup_steps"]
    peak_lr = tcfg["lr"]
    ckpt_dir = Path("checkpoints") / cfg.get("run_name", "m3")

    # Resume: restore weights/optimizer and continue from the saved step. The
    # data stream isn't fast-forwarded — on a large shuffled corpus a different
    # continuation ordering is harmless.
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    batches = _infinite(train_loader, train_sampler)
    fixed_batch = next(batches) if args.overfit else None  # overfit: reuse one batch

    model.train()
    t0 = time.time()
    tokens_seen = 0
    step_loss = step_tok = 0.0
    for step in range(start_step + 1, max_steps + 1):
        lr = lr_at_step(step, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = step_tok = 0.0
        for _ in range(accum):
            src, tgt = fixed_batch if args.overfit else next(batches)
            src, tgt = src.to(device), tgt.to(device)
            with amp_context(device):
                logits = model(src, tgt[:, :-1])
            loss_sum, ntok = label_smoothed_ce(logits, tgt[:, 1:], tok.pad_id, smoothing)
            per_tok = loss_sum / ntok
            (per_tok / accum).backward()
            step_loss += loss_sum.item()
            step_tok += ntok.item()
            tokens_seen += int(ntok.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        if step % tcfg.get("log_every", 50) == 0 or step == 1:
            tok_per_s = tokens_seen / (time.time() - t0)
            train_loss = step_loss / max(1.0, step_tok)
            print(
                f"step {step:>6} | loss {train_loss:.4f} | lr {lr:.2e} "
                f"| {tok_per_s / 1e3:.1f}k tok/s",
                flush=True,
            )
            if run is not None:
                log = {
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_per_sec": tok_per_s,
                    "train/tokens_seen": tokens_seen,
                }
                if device == "cuda":
                    log["gpu/mem_alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
                run.log(log, step=step)

        if not args.overfit and step % tcfg.get("eval_every", 2500) == 0:
            vloss = _valid_loss(model, valid_loader, tok.pad_id, device, smoothing, max_batches=50)
            print(f"  [eval] step {step} valid_loss {vloss:.4f}", flush=True)
            if run is not None:
                run.log({"valid/loss": vloss}, step=step)

        if not args.overfit and step % tcfg.get("checkpoint_every", 5000) == 0:
            _save_ckpt(ckpt_dir / f"step_{step}.pt", raw_model, opt, step, cfg)
            print(f"  [ckpt] {ckpt_dir / f'step_{step}.pt'}", flush=True)

    if args.overfit:
        final = step_loss / max(1.0, step_tok)
        # With label smoothing the floor is the smoothed-target entropy (~1.2 at
        # eps=0.1, V=8k), NOT 0 — a memorizing model converges *to that floor*.
        print(f"\noverfit final loss: {final:.4f} (converges to the label-smoothing floor)")
    else:
        _save_ckpt(ckpt_dir / "final.pt", raw_model, opt, max_steps, cfg)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
