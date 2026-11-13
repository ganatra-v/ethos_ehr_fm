"""Run the published patient-timeline models on a split, write predictions and compute metrics.

Two models are loaded from the Hugging Face Hub: a high-prevalence model (few target codes) and a
low-prevalence model (more target codes, a superset of the high-prevalence ones). Each condition is
predicted by the high-prevalence model when it covers that code, otherwise by the low-prevalence model.

Self-contained: it imports nothing from this repo (only third-party packages such as sklearn).
Data is the finetune_data.pkl layout
({"train"/"val"/"test": [{"patient_id", "token_ids", "labels"}], "target_codes": [...]}).
Labels are optional (the test split has none), in which case only predictions are written.

Usage:
    python infer.py --split val
    python infer.py --split test --threshold 0.3 --out-dir ../inference
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModel

HIGH_REPO = "ganatra-vaib/patient-timeline-high-prevalence-model"
LOW_REPO = "ganatra-vaib/patient-timeline-low-prevalence-model"


# ----------------------------------------------------------------------- metrics
def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, codes: List[str], sources: List[str], threshold: float
) -> Dict[str, object]:
    """Per-condition AUROC / AP (threshold-free) and sensitivity / specificity at `threshold`.
    Conditions with only one class in y_true are undefined (NaN) and skipped by the macro averages."""
    rows = []
    for j, code in enumerate(codes):
        y, p = y_true[:, j].astype(np.int64), y_prob[:, j]
        n_pos, n = int(y.sum()), len(y)
        pred = p >= threshold
        tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
        tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
        valid = 0 < n_pos < n
        rows.append(
            {
                "code": code,
                "model": sources[j],
                "n_pos": n_pos,
                "prevalence": n_pos / n,
                "auroc": roc_auc_score(y, p) if valid else float("nan"),
                "ap": average_precision_score(y, p) if valid else float("nan"),
                "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            }
        )
    macro = {k: float(np.nanmean([r[k] for r in rows])) for k in ("auroc", "ap", "sensitivity", "specificity")}
    return {"macro": macro, "n_conditions_scored": sum(not np.isnan(r["auroc"]) for r in rows), "per_condition": rows}


# --------------------------------------------------------------------- inference
@torch.no_grad()
def predict(
    model: nn.Module, records: List[dict], max_len: int, pad_id: int, batch_size: int, device: torch.device, use_amp: bool
) -> np.ndarray:
    """Sigmoid probabilities, shape (n_records, n_labels). Keeps the most recent `max_len` tokens of each
    timeline (the cut is at the anchor), right-pads each batch."""
    model.eval()
    probs = []
    for start in range(0, len(records), batch_size):
        seqs = [r["token_ids"][-max_len:] for r in records[start : start + batch_size]]
        length = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def load_model(repo: str, device: torch.device):
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()
    codes = [str(c) for c in model.config.target_codes]
    if len(codes) != model.config.n_labels:
        raise ValueError(f"{repo}: {len(codes)} target_codes but n_labels={model.config.n_labels}")
    return model, codes


def run_model(model, records: List[dict], args: argparse.Namespace, device: torch.device, use_amp: bool) -> np.ndarray:
    max_len = min(model.config.max_len, model.backbone.config.n_positions)
    pad_id = model.backbone.config.pad_token_id
    if pad_id is None:
        raise ValueError("Backbone config has no pad_token_id")
    return predict(model, records, max_len, pad_id, args.batch_size, device, use_amp)


def main() -> None:
    p = argparse.ArgumentParser(description="Predict with the high/low-prevalence timeline models and compute metrics.")
    p.add_argument("--high-repo", type=str, default=HIGH_REPO)
    p.add_argument("--low-repo", type=str, default=LOW_REPO)
    p.add_argument("--data-path", type=str, default="../finetune_data.pkl")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--out-dir", type=str, default="../inference")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for sensitivity / specificity.")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.bf16 and device.type == "cuda")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models
    high_model, high_codes = load_model(args.high_repo, device)
    low_model, low_codes = load_model(args.low_repo, device)
    missing = set(high_codes) - set(low_codes)
    if missing:
        print(f"Note: {len(missing)} high-prevalence codes are not in the low-prevalence model: {sorted(missing)}")
    print(f"Models loaded: high ({len(high_codes)} codes), low ({len(low_codes)} codes) | device {device}")

    # Final condition list: every code either model covers (low order first), served by high when available.
    codes = low_codes + [c for c in high_codes if c not in set(low_codes)]
    high_col = {c: j for j, c in enumerate(high_codes)}
    low_col = {c: j for j, c in enumerate(low_codes)}
    sources = ["high" if c in high_col else "low" for c in codes]
    print(f"{len(codes)} conditions: {sources.count('high')} from the high-prevalence model, {sources.count('low')} from the low-prevalence model")

    # ---- data
    with Path(args.data_path).resolve().open("rb") as f:
        data = pickle.load(f)
    data_col = {str(c): j for j, c in enumerate(data["target_codes"])}
    absent = [c for c in codes if c not in data_col]
    if absent:
        raise ValueError(f"Data target_codes lack {len(absent)} model codes: {absent}")
    records = data[args.split]
    print(f"{args.split}: {len(records)} patients")

    # ---- predictions: run both models, then pick per condition
    high_prob = run_model(high_model, records, args, device, use_amp)
    low_prob = run_model(low_model, records, args, device, use_amp)
    prob = np.stack([high_prob[:, high_col[c]] if c in high_col else low_prob[:, low_col[c]] for c in codes], axis=1)

    pred_path = out_dir / f"{args.split}_predictions.csv"
    with pred_path.open("w") as f:
        f.write(",".join(["patient_id"] + codes) + "\n")
        for rec, row in zip(records, prob):
            f.write(",".join([str(rec["patient_id"])] + [f"{v:.6g}" for v in row]) + "\n")
    print(f"Wrote {len(records)} predictions to {pred_path}")

    # ---- metrics (only if the split has labels)
    if any(r.get("labels") is None for r in records):
        print("Split has no labels (e.g. test); skipping metrics.")
        return
    labels = np.stack([np.asarray(r["labels"]) for r in records]).astype(np.int8)
    y_true = labels[:, [data_col[c] for c in codes]]
    m = compute_metrics(y_true, prob, codes, sources, args.threshold)

    header = ["code", "model", "n_pos", "prevalence", "auroc", "ap", "sensitivity", "specificity"]
    metrics_path = out_dir / f"{args.split}_per_condition.csv"
    with metrics_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in m["per_condition"]:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    summary = {
        "split": args.split,
        "threshold": args.threshold,
        "n_conditions_scored": m["n_conditions_scored"],
        "n_conditions": len(codes),
        **{f"macro_{k}": v for k, v in m["macro"].items()},
    }
    (out_dir / f"{args.split}_metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"macro AUROC {m['macro']['auroc']:.4f} | mAP {m['macro']['ap']:.4f} | "
        f"sens {m['macro']['sensitivity']:.4f} | spec {m['macro']['specificity']:.4f} "
        f"(threshold {args.threshold}; {m['n_conditions_scored']}/{len(codes)} conditions scored)"
    )
    print(f"Wrote per-condition metrics to {metrics_path}")


if __name__ == "__main__":
    main()
