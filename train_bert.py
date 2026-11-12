"""BERT on patient timelines with a [CLS]-style summary token, in three modes.

Every input is   <CLS> + window of timeline tokens   (never masked).
Requires a tokenizer built with:  python build_tokenizer.py --include_cls   (and finetune_data.pkl built with it).

Config 1, two stages:
    python train_bert.py --mode pretrain   # stage 1: masked LM only. No CLS objective (no NSP-style loss).
    python train_bert.py --mode finetune   # stage 2: load stage 1, classify from the CLS hidden state.
Config 2, joint:
    python train_bert.py --mode joint      # masked LM + CLS classification trained together from scratch.

Task: multi-label "newly diagnosed after the anchor" from the anchor-truncated sequence before it
(data from prep_finetune_data.py). Only train patients are used for training; val is for monitoring
and model selection; test is only predicted.

Vocabulary: the tokenizer's own (it includes <CLS> and <MASK>), so model vocab = V.
Joint training makes two forward passes per step: a clean one for the CLS loss (as at test time) and a
masked one for the MLM loss, so the classifier never sees masked inputs. loss = L_cls + mlm_weight * L_mlm.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM

import finetune_gpt2 as ft
from tokenizer import EHRTokenizer

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- data
class WindowDataset(Dataset):
    """<CLS> + a window of at most `window` tokens. mode='random': random crop (pretraining);
    mode='recent': the most recent tokens, i.e. those closest to the anchor (classification)."""

    def __init__(self, sequences: List[List[int]], window: int, cls_id: int, mode: str):
        self.sequences, self.window, self.cls_id, self.mode = sequences, window, cls_id, mode

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        if len(seq) > self.window:
            if self.mode == "random":
                start = random.randint(0, len(seq) - self.window)
                seq = seq[start : start + self.window]
            else:
                seq = seq[-self.window :]
        return torch.tensor([self.cls_id] + seq, dtype=torch.long)


class ChunkDataset(Dataset):
    """Deterministic non-overlapping chunks covering every token, each prefixed with <CLS> (MLM validation)."""

    def __init__(self, sequences: List[List[int]], window: int, cls_id: int):
        self.chunks = []
        for seq in sequences:
            for start in range(0, len(seq), window):
                chunk = seq[start : start + window]
                if len(chunk) >= 2:
                    self.chunks.append([cls_id] + chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


class MLMDataset(Dataset):
    """Wraps a dataset of token tensors and applies BERT-style masking on access.

    15% (mlm_prob) of eligible tokens are selected: 80% -> <MASK>, 10% -> a random token, 10% unchanged.
    deterministic=False (train): fresh masks every access.
    deterministic=True  (val):   the mask for item i is a fixed function of (seed, i), so the validation
                                 loss is comparable across epochs.
    Returns (masked input ids, labels) with labels = original id at selected positions, else IGNORE_INDEX.
    """

    def __init__(
        self,
        base: Dataset,
        mask_id: int,
        vocab_size: int,
        no_mask_ids: List[int],
        mlm_prob: float,
        deterministic: bool,
        seed: int = 0,
        no_replace_ids: List[int] | None = None,
    ):
        self.base, self.mask_id, self.mlm_prob = base, mask_id, mlm_prob
        self.no_mask_ids = torch.tensor(no_mask_ids, dtype=torch.long)
        excluded = set(no_replace_ids or [])  # ids never used as the random replacement
        self.replace_pool = torch.tensor([i for i in range(vocab_size) if i not in excluded], dtype=torch.long)
        self.deterministic, self.seed = deterministic, seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        ids = self.base[idx]
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx) if self.deterministic else None

        eligible = ~torch.isin(ids, self.no_mask_ids)  # never mask <PAD>, <BOS>, <EOS>, <CLS>, <MASK>
        selected = (torch.rand(len(ids), generator=gen) < self.mlm_prob) & eligible
        if not selected.any() and eligible.any():  # guarantee at least one prediction target
            candidates = eligible.nonzero().squeeze(-1)
            selected[candidates[torch.randint(len(candidates), (1,), generator=gen)]] = True

        labels = torch.full_like(ids, IGNORE_INDEX)
        labels[selected] = ids[selected]

        inputs = ids.clone()
        roll = torch.rand(len(ids), generator=gen)
        inputs[selected & (roll < 0.8)] = self.mask_id  # 80%: <MASK>
        random_pos = selected & (roll >= 0.8) & (roll < 0.9)  # 10%: random token; remaining 10% unchanged
        draws = torch.randint(len(self.replace_pool), (int(random_pos.sum()),), generator=gen)
        inputs[random_pos] = self.replace_pool[draws]
        return inputs, labels


def make_mlm_collate(pad_id: int):
    def collate(batch) -> Dict[str, torch.Tensor]:
        n, length = len(batch), max(len(x) for x, _ in batch)
        input_ids = torch.full((n, length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, length), dtype=torch.long)
        labels = torch.full((n, length), IGNORE_INDEX, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            input_ids[i, : len(x)] = x
            attention_mask[i, : len(x)] = 1
            labels[i, : len(x)] = y
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


@torch.no_grad()
def evaluate_mlm(mlm: BertForMaskedLM, loader: DataLoader, device: torch.device, use_amp: bool) -> Dict[str, float]:
    """Token-weighted mean cross-entropy and accuracy over the masked positions of the whole loader."""
    mlm.eval()
    total_loss, total, correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = mlm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits, targets = logits.float(), batch["labels"]
        sel = targets != IGNORE_INDEX
        total_loss += nn.functional.cross_entropy(logits[sel], targets[sel], reduction="sum").item()
        total += int(sel.sum().item())
        correct += int((logits.argmax(-1)[sel] == targets[sel]).sum().item())
    mlm.train()
    return {"val/loss": total_loss / max(1, total), "val/masked_token_acc": correct / max(1, total)}


class LabeledDataset(Dataset):
    def __init__(self, base: Dataset, labels: np.ndarray):
        self.base, self.labels = base, labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"ids": self.base[idx], "y": torch.tensor(self.labels[idx], dtype=torch.float32)}


class JointDataset(Dataset):
    """Clean ids + labels (for the CLS loss) and a masked copy of the same ids (for the MLM loss)."""

    def __init__(self, base: Dataset, mlm: MLMDataset, labels: np.ndarray):
        self.base, self.mlm, self.labels = base, mlm, labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.base[idx]  # 'recent' windows are deterministic, so mlm[idx] masks these same tokens
        mlm_ids, mlm_labels = self.mlm[idx]
        return {"ids": ids, "y": torch.tensor(self.labels[idx], dtype=torch.float32), "mlm_ids": mlm_ids, "mlm_labels": mlm_labels}


def make_dict_collate(pad_id: int):
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n, length = len(batch), max(len(b["ids"]) for b in batch)
        out = {
            "input_ids": torch.full((n, length), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros((n, length), dtype=torch.long),
            "labels": torch.stack([b["y"] for b in batch]),
        }
        if "mlm_ids" in batch[0]:
            out["mlm_input_ids"] = torch.full((n, length), pad_id, dtype=torch.long)
            out["mlm_labels"] = torch.full((n, length), IGNORE_INDEX, dtype=torch.long)
        for i, b in enumerate(batch):
            m = len(b["ids"])
            out["input_ids"][i, :m] = b["ids"]
            out["attention_mask"][i, :m] = 1
            if "mlm_ids" in b:
                out["mlm_input_ids"][i, :m] = b["mlm_ids"]
                out["mlm_labels"][i, :m] = b["mlm_labels"]
        return out

    return collate


# -------------------------------------------------------------------------- model
class BertMultiTask(nn.Module):
    """BERT encoder with an MLM head (BertForMaskedLM) and a classification head on the <CLS> state.
    forward() returns the classification logits; the MLM loss goes through self.mlm directly."""

    def __init__(self, mlm: BertForMaskedLM, n_labels: int, dropout: float):
        super().__init__()
        self.mlm = mlm
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(mlm.config.hidden_size, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.mlm.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(hidden[:, 0]))  # <CLS> is always position 0 (sequences are right-padded)


def new_bert_config(args: argparse.Namespace, vocab_size: int, pad_id: int) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=4 * args.n_embd,
        max_position_embeddings=args.max_len,
        type_vocab_size=1,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        pad_token_id=pad_id,
    )


# -------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="BERT with a CLS token: two-stage (pretrain -> finetune) or joint.")
    p.add_argument("--mode", type=str, default="joint", choices=["pretrain", "finetune", "joint"], required=True)
    p.add_argument("--data-path", type=str, default="../finetune_data.pkl")
    p.add_argument("--tokenizer-path", type=str, default="../tokenizer.pkl")
    p.add_argument("--out-dir", type=str, default=None, help="Default: ../checkpoints/bert_<mode>")
    p.add_argument("--pretrained", type=str, default="../checkpoints/bert_pretrain/best_mlm", help="finetune: stage-1 checkpoint.")
    p.add_argument("--from-scratch", action="store_true", help="finetune: ignore --pretrained (no-pretraining baseline).")

    p.add_argument("--max-len", type=int, default=1024, help="Includes the <CLS> token. (finetune: taken from the stage-1 config.)")
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--mlm-weight", type=float, default=1.0, help="joint: loss = L_cls + mlm_weight * L_mlm.")
    p.add_argument("--pos-weight", action="store_true", help="Weight positives by (neg/pos) per condition (train), finetune/joint.")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="Encoder lr. Default: pretrain/joint 3e-4, finetune 1e-4.")
    p.add_argument("--head-lr", type=float, default=1e-3, help="Classification head lr (finetune/joint).")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=None, help="Default: pretrain 0 (off), others 5.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--wandb-project", type=str, default="ehr-timeline-bert")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    mode = args.mode
    if args.lr is None:
        args.lr = 1e-4 if mode == "finetune" else 3e-4
    if args.patience is None:
        args.patience = 0 if mode == "pretrain" else 5

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.bf16 and device.type == "cuda")
    out_dir = Path(args.out_dir or f"../checkpoints/bert_{mode}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = EHRTokenizer.load(Path(args.tokenizer_path).resolve())
    V = len(tokenizer.token_to_id)
    missing = [t for t in ("<CLS>", "<MASK>") if t not in tokenizer.token_to_id]
    if missing:
        raise ValueError(f"Tokenizer has no {missing}. Build it with: python build_tokenizer.py --include_cls")
    pad_id, cls_id, mask_id = tokenizer.encode("<PAD>"), tokenizer.encode("<CLS>"), tokenizer.encode("<MASK>")
    no_mask = [pad_id, tokenizer.encode("<BOS>"), tokenizer.encode("<EOS>"), cls_id, mask_id]

    with Path(args.data_path).resolve().open("rb") as f:
        data = pickle.load(f)
    codes: List[str] = data["target_codes"]
    n_labels = len(codes)
    train_recs = [r for r in data["train"] if len(r["token_ids"]) >= 2]
    val_recs = [r for r in data["val"] if len(r["token_ids"]) >= 2]
    train_seqs, val_seqs = [r["token_ids"] for r in train_recs], [r["token_ids"] for r in val_recs]
    y_train = np.stack([r["labels"] for r in train_recs]).astype(np.float32)
    y_val = np.stack([r["labels"] for r in val_recs]).astype(np.int8)

    # ---- model
    if mode == "finetune" and not args.from_scratch:
        mlm = BertForMaskedLM.from_pretrained(Path(args.pretrained).resolve())
        if mlm.config.vocab_size != V:
            raise ValueError(f"Stage-1 vocab ({mlm.config.vocab_size}) != tokenizer vocab ({V}).")
        args.max_len = min(args.max_len, mlm.config.max_position_embeddings)
    else:
        mlm = BertForMaskedLM(new_bert_config(args, V, pad_id))  # random init, no external weights
    model = BertMultiTask(mlm, n_labels, args.dropout).to(device)
    window = args.max_len - 1  # one position is the <CLS> token
    print(f"mode={mode} | train {len(train_recs)} val {len(val_recs)} | conditions {n_labels} | window {window} | "
          f"params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M | device {device}")

    # ---- loaders
    def mlm_ds(base: Dataset, deterministic: bool) -> MLMDataset:
        return MLMDataset(
            base, mask_id, V, no_mask, args.mlm_prob, deterministic=deterministic, seed=args.seed, no_replace_ids=[cls_id]
        )

    def make_loader(ds: Dataset, collate, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, num_workers=args.num_workers)

    dict_collate, mlm_collate = make_dict_collate(pad_id), make_mlm_collate(pad_id)
    recent_train = WindowDataset(train_seqs, window, cls_id, "recent")
    recent_val = WindowDataset(val_seqs, window, cls_id, "recent")
    # MLM validation: fixed chunks over every token with fixed masks, comparable across epochs.
    val_mlm_loader = make_loader(mlm_ds(ChunkDataset(val_seqs, window, cls_id), True), mlm_collate, False)
    val_cls_loader = make_loader(LabeledDataset(recent_val, y_val), dict_collate, False)

    if mode == "pretrain":
        train_loader = make_loader(mlm_ds(WindowDataset(train_seqs, window, cls_id, "random"), False), mlm_collate, True)
    elif mode == "finetune":
        train_loader = make_loader(LabeledDataset(recent_train, y_train), dict_collate, True)
    else:
        train_loader = make_loader(JointDataset(recent_train, mlm_ds(recent_train, False), y_train), dict_collate, True)

    # ---- optimiser: head group (finetune/joint) + encoder groups; weight decay only on matrices
    groups = []
    if mode != "pretrain":
        groups.append({"params": list(model.head.parameters()), "lr": args.head_lr, "weight_decay": 0.0})
    enc = list(model.mlm.parameters())
    groups.append({"params": [q for q in enc if q.ndim >= 2], "lr": args.lr, "weight_decay": args.weight_decay})
    groups.append({"params": [q for q in enc if q.ndim < 2], "lr": args.lr, "weight_decay": 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    pos_weight = None
    if args.pos_weight:
        # Per-condition (negatives / positives) on the train set, as in finetune.py; floor of 1 positive avoids /0.
        pos = y_train.sum(0)
        pos_weight = torch.tensor((len(y_train) - pos) / np.maximum(pos, 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or f"bert-{mode}", config=vars(args))

    def log(metrics: Dict[str, float], step: int) -> None:
        if run is not None:
            run.log(metrics, step=step)

    def amp():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    best_score, since_best, step = -float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, running_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mult = ft.lr_at(step, total_steps, warmup_steps, args.min_lr_ratio)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * mult

            parts: Dict[str, torch.Tensor] = {}
            with amp():
                if mode == "pretrain":
                    parts["mlm"] = model.mlm(**batch).loss  # batch: input_ids, attention_mask, labels
                else:
                    logits = model(batch["input_ids"], batch["attention_mask"])  # clean input -> <CLS> -> labels
                    parts["cls"] = criterion(logits.float(), batch["labels"])
                    if mode == "joint":
                        parts["mlm"] = model.mlm(
                            input_ids=batch["mlm_input_ids"], attention_mask=batch["attention_mask"], labels=batch["mlm_labels"]
                        ).loss
            loss = parts.get("cls", 0.0) + (args.mlm_weight if mode == "joint" else 1.0) * parts.get("mlm", 0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.grad is not None], args.grad_clip)
            optimizer.step()

            running += loss.item()
            running_n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / running_n:.4f} " + " ".join(f"{k}={v.item():.4f}" for k, v in parts.items()))
                log({"train/loss": running / running_n, "train/grad_norm": float(grad_norm), **{f"train/{k}_loss": v.item() for k, v in parts.items()}, "epoch": epoch}, step)
                running, running_n = 0.0, 0

        # ---- validation
        val_log: Dict[str, float] = {}
        if mode in ("pretrain", "joint"):
            mlm_metrics = evaluate_mlm(model.mlm, val_mlm_loader, device, use_amp)
            val_log.update({"val/mlm_loss": mlm_metrics["val/loss"], "val/mlm_acc": mlm_metrics["val/masked_token_acc"]})
        if mode in ("finetune", "joint"):
            val_prob = ft.predict(model, val_cls_loader, device, use_amp)
            m = ft.compute_metrics(y_val, val_prob, codes)
            val_bce = float(nn.functional.binary_cross_entropy(torch.tensor(val_prob), torch.tensor(y_val, dtype=torch.float32)))
            val_log.update({"val/cls_loss": val_bce, "val/macro_auroc": m["macro_auroc"], "val/macro_ap": m["macro_ap"]})
        score = -val_log["val/mlm_loss"] if mode == "pretrain" else val_log["val/macro_auroc"]
        print(f"== epoch {epoch}: " + " | ".join(f"{k} {v:.4f}" for k, v in val_log.items()))
        log({**val_log, "epoch": epoch}, step)

        if score > best_score:
            best_score, since_best = score, 0
            info = {"mode": mode, "epoch": epoch, "step": step, "val": val_log, "args": vars(args)}
            if mode == "pretrain":
                model.mlm.save_pretrained(out_dir / "best_mlm")  # stage-2 input
            else:
                torch.save({"state_dict": model.state_dict(), "target_codes": codes, "max_len": args.max_len, "epoch": epoch}, out_dir / "best.pt")
                model.mlm.config.save_pretrained(out_dir / "config")
                m["per_condition"].to_csv(out_dir / "val_per_condition.csv", index=False)
            (out_dir / "best_info.json").write_text(json.dumps(info, indent=2))
            print(f"   saved best checkpoint ({'-val mlm loss' if mode == 'pretrain' else 'val macro AUROC'} {score:.4f})")
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    # ---- predictions from the best checkpoint (finetune / joint)
    if mode != "pretrain":
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
        for split, recs in (("val", val_recs), ("test", data["test"])):
            ds = LabeledDataset(WindowDataset([r["token_ids"] for r in recs], window, cls_id, "recent"), np.zeros((len(recs), n_labels), dtype=np.float32))
            prob = ft.predict(model, make_loader(ds, dict_collate, False), device, use_amp)
            df = pd.DataFrame(prob, columns=codes)
            df.insert(0, "patient_id", [r["patient_id"] for r in recs])
            path = out_dir / ("predictions.csv" if split == "test" else "val_predictions.csv")
            df.to_csv(path, index=False)
            print(f"Wrote {len(df)} {split} predictions to {path}")

    if run is not None:
        run.summary["best_score"] = best_score
        run.finish()
    print(f"Done ({mode}). Best score {best_score:.4f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
