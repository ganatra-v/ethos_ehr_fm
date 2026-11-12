"""Fine-tune the pretrained GPT-2 style timeline model for multi-label future-condition prediction.

Input : finetune_data.pkl from prep_finetune_data.py (anchor-truncated timelines + 0/1 label vectors).
Model : pretrained GPT-2 backbone -> hidden state at the LAST real token -> dropout -> Linear(n_embd, n_conditions).
Loss  : BCE-with-logits over the conditions (each condition is an independent binary problem).
Select: best epoch by validation macro-AUROC (the test metric in the README).
Output: best checkpoint, validation per-condition metrics, and predictions.csv for the test patients.

Usage (from src/):
    python finetune.py --pretrained ../checkpoints/pretrain/best
    python finetune.py --from-scratch --no-wandb      # same model, random init (pretraining ablation)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model


# --------------------------------------------------------------------------- data
class TimelineDataset(Dataset):
    """Keeps the most recent `max_len` tokens of each timeline (the cut is at the anchor, so the
    tokens that matter most for forecasting are at the end)."""

    def __init__(self, records: List[dict], max_len: int, n_labels: int):
        self.records = records
        self.max_len = max_len
        self.n_labels = n_labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        ids = rec["token_ids"][-self.max_len :]
        labels = rec["labels"] if rec["labels"] is not None else np.zeros(self.n_labels, dtype=np.int8)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(x) for x, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (x, _) in enumerate(batch):
            input_ids[i, : len(x)] = x
            attention_mask[i, : len(x)] = 1
        labels = torch.stack([y for _, y in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


# -------------------------------------------------------------------------- model
POOL_CHOICES = ("last", "mean", "attention-pool")


class TimelineClassifier(nn.Module):
    """Backbone -> pooling over token states -> dropout -> linear head.

    pool = "last"           : hidden state of the last real token
           "mean"           : mean of all real-token hidden states
           "attention-pool" : learned attention weights over all real tokens (one softmax-weighted sum
                              per patient, shared across conditions)
    """

    def __init__(self, backbone: GPT2Model, n_labels: int, dropout: float, pool: str = "last"):
        super().__init__()
        if pool not in POOL_CHOICES:
            raise ValueError(f"pool must be one of {POOL_CHOICES}, got {pool!r}")
        self.pool = pool
        self.backbone = backbone
        d = backbone.config.n_embd
        # Scores each token: w^T tanh(W h). Randomly initialised, so it is trained with the head.
        self.attn_score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1)) if pool == "attention-pool" else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_labels)

    def new_parameters(self) -> List[nn.Parameter]:
        """Parameters that don't come from pretraining (head + pooling), trained at the head lr."""
        params = list(self.head.parameters())
        if self.attn_score is not None:
            params += list(self.attn_score.parameters())
        return params

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.bool()  # sequences are right-padded; True on real tokens

        if self.pool == "last":
            last = attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        elif self.pool == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:  # attention-pool
            scores = self.attn_score(hidden).squeeze(-1).float()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # padding gets ~0 weight
            weights = torch.softmax(scores, dim=1).to(hidden.dtype)
            pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        return self.head(self.dropout(pooled))


def load_backbone(args: argparse.Namespace, device: torch.device) -> GPT2Model:
    pretrained = Path(args.pretrained).resolve()
    lm = GPT2LMHeadModel.from_pretrained(pretrained)
    if args.from_scratch:
        # Same architecture, random weights: baseline for measuring the benefit of pretraining.
        return GPT2Model(GPT2Config.from_pretrained(pretrained))
    return lm.transformer


def lr_at(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    """Multiplier in (0, 1] applied to each parameter group's base lr."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------- metrics
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, codes: List[str]) -> Dict[str, object]:
    """Macro AUROC / mAP over conditions that have both classes in y_true (others are undefined)."""
    rows = []
    for j, code in enumerate(codes):
        n_pos = int(y_true[:, j].sum())
        valid = 0 < n_pos < len(y_true)
        rows.append(
            {
                "code": code,
                "n_pos": n_pos,
                "prevalence": n_pos / len(y_true),
                "auroc": roc_auc_score(y_true[:, j], y_prob[:, j]) if valid else float("nan"),
                "ap": average_precision_score(y_true[:, j], y_prob[:, j]) if valid else float("nan"),
            }
        )
    per_cond = pd.DataFrame(rows)
    return {
        "macro_auroc": float(per_cond["auroc"].mean()),
        "macro_ap": float(per_cond["ap"].mean()),
        "n_conditions_scored": int(per_cond["auroc"].notna().sum()),
        "per_condition": per_cond,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(probs)


# -------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune the timeline model for multi-label condition prediction.")
    p.add_argument("--data-path", type=str, default="../finetune_data.pkl")
    p.add_argument("--pretrained", type=str, default="../checkpoints/pretrain/best")
    p.add_argument("--out-dir", type=str, default="../checkpoints/finetune")
    p.add_argument("--from-scratch", action="store_true", help="Random-init backbone (no pretraining) for an ablation.")
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the head (and pooling) for the whole run.")
    p.add_argument(
        "--freeze-epochs",
        type=int,
        default=10,
        help="Train only the head (and pooling) for the first N epochs with the backbone frozen, then unfreeze it "
        "for full fine-tuning. Ignored if --freeze-backbone is set.",
    )
    p.add_argument(
        "--unfreeze-warmup-steps",
        type=int,
        default=100,
        help="After unfreezing, ramp the backbone lr linearly from 0 over this many steps (0 = no ramp).",
    )

    p.add_argument("--max-len", type=int, default=0, help="Context length; 0 = the pretrained model's n_positions.")
    p.add_argument("--pool", type=str, choices=POOL_CHOICES, default="attention-pool", help="How token states are pooled into one vector.")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout on the pooled vector before the head.")
    p.add_argument("--pos-weight", action="store_true", help="Weight positives by (neg/pos) per condition (train).")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone learning rate.")
    p.add_argument("--head-lr", type=float, default=1e-3, help="Classification head learning rate.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5, help="Stop after N epochs without val AUROC gain (0 = off).")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--wandb-project", type=str, default="ehr-timeline-finetune")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()
    if args.freeze_epochs < 0 or args.freeze_epochs >= args.epochs:
        p.error(f"--freeze-epochs must be in [0, --epochs); got {args.freeze_epochs} with --epochs {args.epochs}")
    if args.freeze_backbone:
        args.freeze_epochs = 0  # permanent freeze takes precedence; there is nothing to unfreeze

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.bf16 and device.type == "cuda")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.data_path).resolve().open("rb") as f:
        data = pickle.load(f)
    codes: List[str] = data["target_codes"]
    n_labels = len(codes)

    backbone = load_backbone(args, device)
    n_positions = backbone.config.n_positions
    max_len = min(args.max_len or n_positions, n_positions)
    pad_id = backbone.config.pad_token_id
    if pad_id is None:
        raise ValueError("Pretrained config has no pad_token_id")

    collate = make_collate(pad_id)

    def loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            TimelineDataset(data[split], max_len, n_labels),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=args.num_workers,
        )

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)
    y_train = np.stack([r["labels"] for r in data["train"]]).astype(np.float32)
    y_val = np.stack([r["labels"] for r in data["val"]]).astype(np.int8)
    print(f"train {len(data['train'])} | val {len(data['val'])} | test {len(data['test'])} | conditions {n_labels} | max_len {max_len}")

    model = TimelineClassifier(backbone, n_labels, args.dropout, pool=args.pool).to(device)
    if args.freeze_backbone:
        for prm in model.backbone.parameters():
            prm.requires_grad = False
    n_train_params = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
    print(f"trainable parameters: {n_train_params / 1e6:.2f}M | init: {'scratch' if args.from_scratch else 'pretrained'} | device: {device}")

    pos_weight = None
    if args.pos_weight:
        pos = y_train.sum(0)
        pos_weight = torch.tensor((len(y_train) - pos) / np.maximum(pos, 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    groups = [{"params": model.new_parameters(), "lr": args.head_lr, "weight_decay": 0.0}]
    backbone_params = [prm for prm in model.backbone.parameters() if prm.requires_grad]
    if backbone_params:
        decay = [prm for prm in backbone_params if prm.ndim >= 2]
        no_decay = [prm for prm in backbone_params if prm.ndim < 2]
        groups.append({"params": decay, "lr": args.lr, "weight_decay": args.weight_decay, "is_backbone": True})
        groups.append({"params": no_decay, "lr": args.lr, "weight_decay": 0.0, "is_backbone": True})
    for g in groups:
        g["base_lr"] = g["lr"]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "n_trainable_params": n_train_params, "n_conditions": n_labels},
        )

    def log(metrics: Dict[str, float], step: int) -> None:
        if run is not None:
            run.log(metrics, step=step)

    def set_backbone_phase(frozen: bool) -> None:
        # Must be re-applied every epoch: predict() ends with model.train(), which would otherwise put a
        # frozen backbone back into dropout mode.
        for prm in model.backbone.parameters():
            prm.requires_grad = not frozen
        model.train()
        if frozen:
            model.backbone.eval()  # no dropout noise in the features the new head is learning from

    best_auroc, epochs_since_best, step = -float("inf"), 0, 0
    unfreeze_step: Optional[int] = None
    for epoch in range(1, args.epochs + 1):
        frozen = args.freeze_backbone or epoch <= args.freeze_epochs
        if args.freeze_backbone or args.freeze_epochs:
            set_backbone_phase(frozen)
            if args.freeze_epochs and not frozen and unfreeze_step is None:
                unfreeze_step = step
                print(f"== epoch {epoch}: backbone unfrozen, full fine-tuning from here")
        else:
            model.train()
        running, running_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mult = lr_at(step, total_steps, warmup_steps, args.min_lr_ratio)
            ramp = 1.0
            if unfreeze_step is not None and args.unfreeze_warmup_steps > 0:
                ramp = min(1.0, (step - unfreeze_step + 1) / args.unfreeze_warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * mult * (ramp if g.get("is_backbone") else 1.0)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits.float(), batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([prm for prm in model.parameters() if prm.requires_grad], args.grad_clip)
            optimizer.step()

            running += loss.item()
            running_n += 1
            step += 1
            if step % args.log_every == 0:
                avg = running / running_n
                print(f"epoch {epoch} step {step}/{total_steps} loss {avg:.4f}")
                log(
                    {
                        "train/loss": avg,
                        "train/grad_norm": float(grad_norm),
                        "train/head_lr": groups[0]["lr"],
                        "train/backbone_frozen": float(frozen),
                        "epoch": epoch,
                    },
                    step,
                )
                running, running_n = 0.0, 0

        val_prob = predict(model, val_loader, device, use_amp)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(val_prob), torch.tensor(y_val, dtype=torch.float32)))
        m = compute_metrics(y_val, val_prob, codes)
        print(f"== epoch {epoch}: val loss {val_loss:.4f} | macro AUROC {m['macro_auroc']:.4f} | macro AP {m['macro_ap']:.4f} ({m['n_conditions_scored']}/{n_labels} conditions scored)")
        log({"val/loss": val_loss, "val/macro_auroc": m["macro_auroc"], "val/macro_ap": m["macro_ap"], "epoch": epoch}, step)

        if m["macro_auroc"] > best_auroc:
            best_auroc, epochs_since_best = m["macro_auroc"], 0
            torch.save({"state_dict": model.state_dict(), "target_codes": codes, "max_len": max_len, "pool": args.pool, "epoch": epoch}, out_dir / "best.pt")
            model.backbone.config.save_pretrained(out_dir / "backbone_config")
            m["per_condition"].to_csv(out_dir / "val_per_condition.csv", index=False)
            (out_dir / "val_metrics.json").write_text(
                json.dumps(
                    {"epoch": epoch, "val_loss": val_loss, "macro_auroc": m["macro_auroc"], "macro_ap": m["macro_ap"], "args": vars(args)},
                    indent=2,
                )
            )
            print(f"   saved best checkpoint (val macro AUROC {best_auroc:.4f})")
        elif frozen and args.freeze_epochs:
            pass  # head-only warm-up: don't let a plateau here trigger early stopping before unfreezing
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"Early stopping: no val AUROC gain for {args.patience} epochs.")
                break

    # Test predictions from the best checkpoint.
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_prob = predict(model, test_loader, device, use_amp)
    pred = pd.DataFrame(test_prob, columns=codes)
    pred.insert(0, "patient_id", [r["patient_id"] for r in data["test"]])
    pred_path = Path(out_dir / "test_predictions.csv").resolve()
    pred.to_csv(pred_path, index=False)
    print(f"Wrote {len(pred)} test predictions to {pred_path}")

   # Validation predictions from the same best checkpoint (same format as the test file).
    val_pred = pd.DataFrame(predict(model, val_loader, device, use_amp), columns=codes)
    val_pred.insert(0, "patient_id", [r["patient_id"] for r in data["val"]])
    val_pred_path = Path(out_dir / "val_predictions.csv").resolve()
    val_pred.to_csv(val_pred_path, index=False)
    print(f"Wrote {len(val_pred)} validation predictions to {val_pred_path}")

    if run is not None:
        run.summary["best_val_macro_auroc"] = best_auroc
        run.finish()
    print(f"Done. Best val macro AUROC {best_auroc:.4f}. Checkpoint in {out_dir}")


if __name__ == "__main__":
    main()
