"""GPT-2 style next-token pretraining on tokenized patient timelines (from pht.py).

Timelines for train/val patients are truncated at each patient's anchor (last encounter - 5 years)
so no post-anchor information reaches the pretrained weights.

Usage (from src/):
    python pretrain.py --data-dir ../train_val --tokenizer-path ../tokenizer.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from tokenizer import EHRTokenizer
from pht import PatientHealthTimelineBuilder

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- data
def load_timelines(
    data_dir: Path,
    tokenizer: EHRTokenizer,
    splits_path: Path,
    cache_path: Path,
    rebuild: bool,
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Returns {"train": {pid: token_ids}, "val": {pid: token_ids}}, anchor-truncated. Cached on disk."""
    if cache_path.exists() and not rebuild:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    splits = pd.read_csv(splits_path)
    splits["Id"] = splits["Id"].astype(str)
    splits["split"] = splits["split"].astype(str).str.lower()

    builder = PatientHealthTimelineBuilder(data_dir=data_dir, tokenizer=tokenizer)
    available = set(builder.get_patient_ids())

    out: Dict[str, Dict[str, List[int]]] = {}
    for split in ("train", "val"):
        ids = [pid for pid in splits.loc[splits["split"] == split, "Id"] if pid in available]
        print(f"Building {split} timelines for {len(ids)} patients (truncated at anchor)...")
        timelines = builder.build_patient_timelines(ids, truncate_at_anchor=True)
        out[split] = {pid: tl.token_ids for pid, tl in timelines.items()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(out, f)
    return out


class RandomCropDataset(Dataset):
    """Training set: each access returns a random contiguous window of at most max_len tokens,
    so long timelines are re-cropped differently every epoch."""

    def __init__(self, sequences: List[List[int]], max_len: int):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            start = random.randint(0, len(seq) - self.max_len)
            seq = seq[start : start + self.max_len]
        return torch.tensor(seq, dtype=torch.long)


class ChunkedDataset(Dataset):
    """Validation set: deterministic, non-overlapping chunks of max_len covering every token."""

    def __init__(self, sequences: List[List[int]], max_len: int):
        self.chunks: List[List[int]] = []
        for seq in sequences:
            for start in range(0, len(seq), max_len):
                chunk = seq[start : start + max_len]
                if len(chunk) >= 2:  # need at least one (input, target) pair
                    self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def make_collate(pad_id: int):
    def collate(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x)
            input_ids[i, :n] = x
            attention_mask[i, :n] = 1
            labels[i, :n] = x  # HF shifts labels internally; padding stays IGNORE_INDEX
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


# -------------------------------------------------------------------------- model
def build_model(args: argparse.Namespace, vocab_size: int, tokenizer: EHRTokenizer) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.max_len,
        n_ctx=args.max_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=tokenizer.encode("<BOS>"),
        eos_token_id=tokenizer.encode("<EOS>"),
        pad_token_id=tokenizer.encode("<PAD>"),
    )
    return GPT2LMHeadModel(config)  # random init: no pretrained weights


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)  # no decay on biases / LayerNorm
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


@torch.no_grad()
def evaluate(model: GPT2LMHeadModel, loader: DataLoader, device: torch.device, use_amp: bool) -> Dict[str, float]:
    """Token-weighted mean next-token cross-entropy over the whole validation set."""
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits = logits[:, :-1].float()
        targets = batch["labels"][:, 1:]
        mask = targets != IGNORE_INDEX
        loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum")
        total_loss += loss.item()
        total_tokens += int(mask.sum().item())
        correct += int((logits.argmax(-1)[mask] == targets[mask]).sum().item())
    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    return {
        "val/loss": mean_loss,
        "val/perplexity": math.exp(min(mean_loss, 20.0)),
        "val/next_token_acc": correct / max(1, total_tokens),
        "val/tokens": total_tokens,
    }


# -------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="GPT-2 style pretraining on patient timelines.")
    p.add_argument("--data-dir", type=str, default="../train_val")
    p.add_argument("--tokenizer-path", type=str, default="../tokenizer.pkl")
    p.add_argument("--splits-path", type=str, default="../patient_splits.csv")
    p.add_argument("--cache-path", type=str, default="../timelines_anchor_truncated.pkl")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out-dir", type=str, default="../checkpoints/pretrain")

    # model (defaults ~5M params)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # optimisation
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=0, help="Stop after N epochs without val improvement (0 = off).")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    # logging
    p.add_argument("--wandb-project", type=str, default="ehr-timeline-pretrain")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.bf16 and device.type == "cuda")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = EHRTokenizer.load(Path(args.tokenizer_path).resolve())
    vocab_size = len(tokenizer.token_to_id)
    pad_id = tokenizer.encode("<PAD>")

    timelines = load_timelines(
        Path(args.data_dir).resolve(),
        tokenizer,
        Path(args.splits_path).resolve(),
        Path(args.cache_path).resolve(),
        args.rebuild_cache,
    )
    train_seqs = [s for s in timelines["train"].values() if len(s) >= 2]
    val_seqs = [s for s in timelines["val"].values() if len(s) >= 2]
    print(
        f"train patients: {len(train_seqs)} ({sum(map(len, train_seqs))} tokens) | "
        f"val patients: {len(val_seqs)} ({sum(map(len, val_seqs))} tokens) | vocab: {vocab_size}"
    )

    collate = make_collate(pad_id)
    train_loader = DataLoader(
        RandomCropDataset(train_seqs, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        ChunkedDataset(val_seqs, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    model = build_model(args, vocab_size, tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M | device: {device}")

    optimizer = make_optimizer(model, args)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "n_params": n_params, "vocab_size": vocab_size},
        )

    def log(metrics: Dict[str, float], step: int) -> None:
        if run is not None:
            run.log(metrics, step=step)

    best_val = float("inf")
    epochs_since_best = 0
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, running_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr_ratio)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item()
            running_n += 1
            step += 1
            if step % args.log_every == 0:
                avg = running / running_n
                print(f"epoch {epoch} step {step}/{total_steps} loss {avg:.4f} lr {lr:.2e}")
                log({"train/loss": avg, "train/lr": lr, "train/grad_norm": float(grad_norm), "epoch": epoch}, step)
                running, running_n = 0.0, 0

        val_metrics = evaluate(model, val_loader, device, use_amp)
        print(f"== epoch {epoch}: " + " | ".join(f"{k} {v:.4f}" for k, v in val_metrics.items()))
        log({**val_metrics, "epoch": epoch}, step)

        if val_metrics["val/loss"] < best_val:
            best_val = val_metrics["val/loss"]
            epochs_since_best = 0
            model.save_pretrained(out_dir / "best")
            (out_dir / "best" / "train_info.json").write_text(
                json.dumps({"epoch": epoch, "step": step, "val_loss": best_val, "args": vars(args)}, indent=2)
            )
            print(f"   saved best checkpoint (val loss {best_val:.4f})")
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs.")
                break

    model.save_pretrained(out_dir / "last")
    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.finish()
    print(f"Done. Best val loss {best_val:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
