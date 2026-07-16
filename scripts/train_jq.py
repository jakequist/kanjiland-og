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
from kanjiland.train.seed import seed_everything


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
    seed_everything(seed)
    device = pick_device()

    # --- tokenizer + modexl --------------------------------------------------
    tok = Tokenizer.load(cfg["tokenizer"]["path"])
    mcfg = ModelConfig.from_dict(cfg["model"], vocab_size=tok.vocab_size)

    mcfg.pad_id = tok.pad_id
    model = Transformer(mcfg).to(device)
    print(f"model: {model.num_parameters() / 1e6:.1f}M params | device {device}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=tcfg.get("weight_decay", 0.0),
    )



if __name__ == "__main__":
    main()
