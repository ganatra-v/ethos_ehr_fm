"""Build fine-tuning data: pre-anchor tokenized timelines + multi-label "newly diagnosed after anchor" targets.

Train / val patients (full records in train_val/):
    * anchor = (last encounter STOP, or START if blank) - 5 years        [computed by pht.py]
    * input  = timeline truncated at the anchor (strictly pre-anchor events)
    * label[c] = 1 iff the patient's first-ever START of condition c falls in [anchor, anchor + 5y]
      (conditions already present before the anchor are therefore negative)

Test patients (already truncated in test/, no outcomes):
    * input  = the full given timeline (no further truncation); anchor taken from test_anchors.csv
    * label  = None

Output pickle:
    {
      "target_codes": [str, ...],              # column order of every label vector
      "target_descriptions": [str, ...],
      "train": [record, ...], "val": [...], "test": [...],
    }
    record = {"patient_id": str, "token_ids": List[int], "labels": np.ndarray[int8] | None, "anchor": pd.Timestamp}

Usage (from src/):
    python prep_finetune_data.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tokenizer import EHRTokenizer
from pht import ANCHOR_YEARS, PatientHealthTimelineBuilder, _to_datetime


def load_targets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"CODE": str})
    df["CODE"] = df["CODE"].str.strip()
    if df["CODE"].duplicated().any():
        raise ValueError(f"Duplicate codes in {path}")
    return df


def first_diagnosis_times(conditions_path: Path, target_codes: List[str]) -> Dict[str, Dict[str, pd.Timestamp]]:
    """{patient_id: {code: first-ever START}} over the patient's whole history, target codes only."""
    df = pd.read_csv(conditions_path, dtype={"CODE": str, "PATIENT": str})
    df["CODE"] = df["CODE"].str.strip()
    df = df[df["CODE"].isin(set(target_codes))].copy()
    df["start"] = _to_datetime(df["START"])
    df = df.dropna(subset=["start"])
    first = df.groupby(["PATIENT", "CODE"])["start"].min()

    out: Dict[str, Dict[str, pd.Timestamp]] = {}
    for (pid, code), ts in first.items():
        out.setdefault(pid, {})[code] = ts
    return out


def make_labels(
    anchor: pd.Timestamp,
    first_dx: Dict[str, pd.Timestamp],
    target_codes: List[str],
    window_years: int,
) -> np.ndarray:
    labels = np.zeros(len(target_codes), dtype=np.int8)
    for j, code in enumerate(target_codes):
        ts = first_dx.get(code)
        # First-ever diagnosis must fall after the anchor.
        if ts is not None and anchor <= ts:
            labels[j] = 1
    return labels


def build_train_val(
    data_dir: Path,
    tokenizer: EHRTokenizer,
    splits: pd.DataFrame,
    targets: pd.DataFrame,
    window_years: int,
) -> Dict[str, List[dict]]:
    target_codes = targets["CODE"].tolist()
    builder = PatientHealthTimelineBuilder(data_dir=data_dir, tokenizer=tokenizer)
    available = set(builder.get_patient_ids())
    first_dx = first_diagnosis_times(data_dir / "conditions.csv", target_codes)

    out: Dict[str, List[dict]] = {}
    for split in ("train", "val"):
        ids = [pid for pid in splits.loc[splits["split"] == split, "Id"] if pid in available]
        missing = (splits["split"] == split).sum() - len(ids)
        no_anchor = [pid for pid in ids if pid not in builder.anchors]
        if missing or no_anchor:
            print(f"[{split}] skipping {missing} patients absent from {data_dir.name}/ and {len(no_anchor)} with no encounters (no anchor)")
        ids = [pid for pid in ids if pid in builder.anchors]

        print(f"[{split}] building {len(ids)} timelines truncated at the anchor...")
        timelines = builder.build_patient_timelines(ids, truncate_at_anchor=True)
        out[split] = [
            {
                "patient_id": pid,
                "token_ids": timelines[pid].token_ids,
                "labels": make_labels(builder.anchors[pid], first_dx.get(pid, {}), target_codes, window_years),
                "anchor": builder.anchors[pid],
            }
            for pid in ids
        ]
    return out


def build_test(test_dir: Path, tokenizer: EHRTokenizer, splits: pd.DataFrame, anchors_path: Path) -> List[dict]:
    anchors = pd.read_csv(anchors_path, dtype={"Id": str})
    anchor_by_id = dict(zip(anchors["Id"], pd.to_datetime(anchors["anchor_date"], utc=True)))

    builder = PatientHealthTimelineBuilder(data_dir=test_dir, tokenizer=tokenizer)
    available = set(builder.get_patient_ids())
    ids = splits.loc[splits["split"] == "test", "Id"].tolist()
    absent = [pid for pid in ids if pid not in available]
    if absent:
        raise ValueError(f"{len(absent)} test patients from patient_splits.csv are missing in {test_dir} (e.g. {absent[:3]})")
    no_anchor = [pid for pid in ids if pid not in anchor_by_id]
    if no_anchor:
        print(f"[test] warning: {len(no_anchor)} patients have no row in {anchors_path.name}")

    # Test files are already cut at the anchor by the organisers, so we do NOT truncate again
    # (recomputing the anchor from truncated encounters would be wrong).
    print(f"[test] building {len(ids)} timelines (already truncated)...")
    timelines = builder.build_patient_timelines(ids, truncate_at_anchor=False)
    return [
        {
            "patient_id": pid,
            "token_ids": timelines[pid].token_ids,
            "labels": None,
            "anchor": anchor_by_id.get(pid),
        }
        for pid in ids
    ]


def print_summary(data: Dict[str, List[dict]], targets: pd.DataFrame) -> None:
    for split in ("train", "val"):
        recs = data[split]
        if not recs:
            continue
        y = np.stack([r["labels"] for r in recs])
        lens = np.array([len(r["token_ids"]) for r in recs])
        print(f"\n[{split}] patients={len(recs)} | tokens/patient mean={lens.mean():.0f} median={np.median(lens):.0f} max={lens.max()}")
        print(f"[{split}] patients with >=1 positive label: {(y.sum(1) > 0).mean():.1%}")
        prev = y.mean(0)
        for code, desc, p, n in zip(targets["CODE"], targets["DESCRIPTION"], prev, y.sum(0)):
            flag = "  <-- no positives" if n == 0 else ""
            print(f"  {code:>12}  n_pos={int(n):5d}  prevalence={p:6.2%}  {desc}{flag}")
    if data["test"]:
        lens = np.array([len(r["token_ids"]) for r in data["test"]])
        print(f"\n[test] patients={len(data['test'])} | tokens/patient mean={lens.mean():.0f} median={np.median(lens):.0f} max={lens.max()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare fine-tuning data (timelines + future-condition labels).")
    p.add_argument("--data-dir", type=str, default="../train_val")
    p.add_argument("--test-dir", type=str, default="../test")
    p.add_argument("--tokenizer-path", type=str, default="../tokenizer.pkl")
    p.add_argument("--splits-path", type=str, default="../patient_splits.csv")
    p.add_argument("--targets-path", type=str, default="../target_conditions.csv")
    p.add_argument("--test-anchors-path", type=str, default="../test_anchors.csv")
    p.add_argument("--output-path", type=str, default="../finetune_data.pkl")
    p.add_argument("--window-years", type=int, default=ANCHOR_YEARS, help="Label window after the anchor.")
    args = p.parse_args()

    tokenizer = EHRTokenizer.load(Path(args.tokenizer_path).resolve())
    splits = pd.read_csv(args.splits_path, dtype={"Id": str})
    splits["split"] = splits["split"].astype(str).str.lower()
    targets = load_targets(Path(args.targets_path).resolve())

    data = build_train_val(Path(args.data_dir).resolve(), tokenizer, splits, targets, args.window_years)
    data["test"] = build_test(Path(args.test_dir).resolve(), tokenizer, splits, Path(args.test_anchors_path).resolve())
    data["target_codes"] = targets["CODE"].tolist()
    data["target_descriptions"] = targets["DESCRIPTION"].tolist()

    out = Path(args.output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(data, f)
    print_summary(data, targets)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
