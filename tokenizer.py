from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

EVENT_TABLES: Dict[str, str] = {
    "allergies": "allergies",
    "careplans": "careplans",
    "conditions": "conditions",
    "devices": "devices",
    "encounters": "encounters",
    "immunizations": "immunizations",
    "medications": "medications",
    "procedures": "procedures",
}
EVENT_PREFIXES: Dict[str, str] = {
    "allergies": "ALLERGY",
    "careplans": "CAREPLAN",
    "conditions": "CONDITION",
    "devices": "DEVICE",
    "encounters": "ENCOUNTER",
    "immunizations": "IMMUNIZATION",
    "medications": "MEDICATION",
    "procedures": "PROCEDURE",
}
DEMOGRAPHIC_COLUMNS = ("GENDER", "RACE", "ETHNICITY", "MARITAL")
QUANTILE_BINS = 10
AGE_BUCKET_TOKENS = ["AGE_LT5"] + [f"AGE_{start}_{start + 5}" for start in range(5, 100, 5)] + ["AGE_GT100"]
TIME_DELTA_BUCKET_TOKENS = [
    "TIME_5M_15M",
    "TIME_15M_1H",
    "TIME_1H_2H",
    "TIME_2H_6H",
    "TIME_6H_12H",
    "TIME_12H_1D",
    "TIME_1D_3D",
    "TIME_3D_1W",
    "TIME_1W_2W",
    "TIME_2W_1MT",
    "TIME_1MT_3MT",
    "TIME_3MT_6MT",
    "TIME_6MT_LT12MT",
    "TIME_EQ_1Y",
]
SPECIAL_TOKENS = ("<BOS>", "<EOS>", "<DEATH>", "<PAD>", "<START>", "<STOP>")


class EHRTokenizer:
    """Tokenizer for EHR events modeled after ETHOS-style event vocabularies."""

    def __init__(self, data_dir: str | Path, output_dir: str | Path | None = None, include_cls: bool = False):
        self.include_cls = include_cls  # adds a <CLS> special token (for BERT-style summary-token models)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir is not None else self.data_dir

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        self.event_code_to_id: Dict[Tuple[str, str], int] = {}
        self.event_unknown_id: Dict[str, int] = {}

        self.observation_code_to_id: Dict[str, int] = {}
        self.observation_unknown_id: Optional[int] = None

        self.quantile_thresholds: Dict[str, Dict[str, float]] = {}
        self.quantile_id_to_token: Dict[int, str] = {}
        self.quantile_token_to_id: Dict[str, int] = {}

        self.demographic_value_to_id: Dict[str, Dict[str, int]] = {column: {} for column in DEMOGRAPHIC_COLUMNS}
        self.demographic_unknown_id: Dict[str, int] = {}

        self.age_token_to_id: Dict[str, int] = {}

        self.time_delta_token_to_id: Dict[str, int] = {}
        self.time_delta_unknown_id: Optional[int] = None

        self.special_token_to_id: Dict[str, int] = {}

        self.train_patient_ids = self._load_train_patient_ids()

        self._build_special_vocab()
        self._build_event_vocab()
        self._build_observation_vocab()
        self._build_quantile_vocab()
        self._build_demographic_vocab()
        self._build_age_vocab()
        self._build_time_delta_vocab()
        
        print(f"Vocabulary size: {len(self.token_to_id)}")

    def _load_train_patient_ids(self) -> set:
        """Ids of train-split patients; the vocabulary and quantile thresholds are fit on these only."""
        splits_path = self.data_dir.parent / "patient_splits.csv"
        if not splits_path.exists():
            raise FileNotFoundError(f"{splits_path} is required to restrict the vocabulary to train patients")
        splits = pd.read_csv(splits_path)
        if "Id" not in splits.columns or "split" not in splits.columns:
            raise ValueError(f"{splits_path} must have 'Id' and 'split' columns")
        is_train = splits["split"].astype(str).str.lower() == "train"
        return set(splits.loc[is_train, "Id"].astype(str))

    def _read_train_table(self, name: str) -> Optional[pd.DataFrame]:
        """Reads <name>.csv restricted to train patients; None if the file is missing."""
        path = self.data_dir / f"{name}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "PATIENT" not in df.columns:
            return df.iloc[0:0]
        return df[df["PATIENT"].astype(str).isin(self.train_patient_ids)]

    def _register_token(self, token: str) -> int:
        token_id = self.token_to_id.get(token)
        if token_id is None:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return token_id

    def _build_event_vocab(self) -> None:
        for table_name, file_name in EVENT_TABLES.items():
            prefix = EVENT_PREFIXES[table_name]
            unknown_token = f"{prefix}_UNKNOWN"
            unknown_id = self._register_token(unknown_token)
            self.event_unknown_id[table_name] = unknown_id

            df = self._read_train_table(file_name)
            if df is None or "CODE" not in df.columns:
                continue

            codes = [str(code).strip() for code in df["CODE"].dropna().unique() if str(code).strip()]
            for code in sorted(codes, key=lambda x: (x.isdigit(), x)):
                token = f"{prefix}_{code}"
                token_id = self._register_token(token)
                self.event_code_to_id[(table_name, code)] = token_id

    def _build_special_vocab(self) -> None:
        # <CLS> is registered after the base specials, so without the flag every id is unchanged.
        tokens = SPECIAL_TOKENS + (("<CLS>", "<MASK>") if self.include_cls else ())
        for token in tokens:
            self.special_token_to_id[token] = self._register_token(token)

    def _build_observation_vocab(self) -> None:
        unknown_token = "OBSERVATION_UNKNOWN"
        self.observation_unknown_id = self._register_token(unknown_token)

        df = self._read_train_table("observations")
        if df is None or "CODE" not in df.columns:
            return

        codes = [str(code).strip() for code in df["CODE"].dropna().unique() if str(code).strip()]
        for code in sorted(codes, key=lambda x: (x.isdigit(), x)):
            token = f"OBSERVATION_{code}"
            token_id = self._register_token(token)
            self.observation_code_to_id[code] = token_id

    def _build_quantile_vocab(self) -> None:
        for idx in range(1, QUANTILE_BINS + 1):
            token = f"Q{idx}"
            token_id = self._register_token(token)
            self.quantile_id_to_token[idx] = token
            self.quantile_token_to_id[token] = token_id

        obs = self._read_train_table("observations")
        if obs is None or "VALUE" not in obs.columns or "CODE" not in obs.columns:
            return

        for code in sorted(obs["CODE"].dropna().astype(str).unique()):
            code_values = pd.to_numeric(obs.loc[obs["CODE"].astype(str) == code, "VALUE"], errors="coerce").dropna()
            if code_values.empty:
                continue

            thresholds = np.quantile(code_values.to_numpy(), np.linspace(0.0, 1.0, QUANTILE_BINS + 1))
            self.quantile_thresholds[code] = {f"Q{idx}": float(thresholds[idx]) for idx in range(1, QUANTILE_BINS + 1)}

    @staticmethod
    def _normalize_token_value(value: str) -> str:
        normalized_chars = []
        for ch in value.upper():
            if ch.isalnum():
                normalized_chars.append(ch)
            else:
                normalized_chars.append("_")
        normalized = "".join(normalized_chars).strip("_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized or "UNKNOWN"

    def _build_demographic_vocab(self) -> None:
        for column in DEMOGRAPHIC_COLUMNS:
            unknown_token = f"{column}_UNKNOWN"
            self.demographic_unknown_id[column] = self._register_token(unknown_token)

        patients_path = self.data_dir / "patients.csv"
        if not patients_path.exists():
            return

        patients = pd.read_csv(patients_path)
        if "Id" not in patients.columns:
            return

        train_patients = patients[patients["Id"].astype(str).isin(self.train_patient_ids)]

        for column in DEMOGRAPHIC_COLUMNS:
            if column not in train_patients.columns:
                continue
            values = sorted(train_patients[column].dropna().astype(str).unique().tolist())
            for value in values:
                normalized_value = self._normalize_token_value(value)
                token = f"{column}_{normalized_value}"
                token_id = self._register_token(token)
                self.demographic_value_to_id[column][value] = token_id

    def _build_age_vocab(self) -> None:
        for token in AGE_BUCKET_TOKENS:
            self.age_token_to_id[token] = self._register_token(token)

    def _build_time_delta_vocab(self) -> None:
        self.time_delta_unknown_id = self._register_token("TIME_UNKNOWN")
        for token in TIME_DELTA_BUCKET_TOKENS:
            self.time_delta_token_to_id[token] = self._register_token(token)

    @staticmethod
    def _age_token_for_value(age_years: float) -> str:
        if age_years < 5:
            return "AGE_LT5"
        if age_years > 100:
            return "AGE_GT100"
        if age_years == 100:
            return "AGE_95_100"
        bucket_start = int(age_years // 5) * 5
        bucket_end = bucket_start + 5
        return f"AGE_{bucket_start}_{bucket_end}"

    @staticmethod
    def _time_delta_token_for_minutes(delta_minutes: float) -> str:
        minute = 1.0
        hour = 60.0 * minute
        day = 24.0 * hour
        week = 7.0 * day
        month = 30.0 * day

        if 5.0 * minute <= delta_minutes < 15.0 * minute:
            return "TIME_5M_15M"
        if 15.0 * minute <= delta_minutes < 1.0 * hour:
            return "TIME_15M_1H"
        if 1.0 * hour <= delta_minutes < 2.0 * hour:
            return "TIME_1H_2H"
        if 2.0 * hour <= delta_minutes < 6.0 * hour:
            return "TIME_2H_6H"
        if 6.0 * hour <= delta_minutes < 12.0 * hour:
            return "TIME_6H_12H"
        if 12.0 * hour <= delta_minutes < 1.0 * day:
            return "TIME_12H_1D"
        if 1.0 * day <= delta_minutes < 3.0 * day:
            return "TIME_1D_3D"
        if 3.0 * day <= delta_minutes < 1.0 * week:
            return "TIME_3D_1W"
        if 1.0 * week <= delta_minutes < 2.0 * week:
            return "TIME_1W_2W"
        if 2.0 * week <= delta_minutes < 1.0 * month:
            return "TIME_2W_1MT"
        if 1.0 * month <= delta_minutes < 3.0 * month:
            return "TIME_1MT_3MT"
        if 3.0 * month <= delta_minutes < 6.0 * month:
            return "TIME_3MT_6MT"
        if 6.0 * month <= delta_minutes < 12.0 * month:
            return "TIME_6MT_LT12MT"
        if delta_minutes == 12.0 * month:
            return "TIME_EQ_1Y"
        return "TIME_UNKNOWN"

    def encode(self, token: str) -> int:
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def encode_event(self, event_type: str, code: str | int) -> int:
        normalized = event_type.strip().lower()
        table_name = normalized if normalized in EVENT_TABLES else f"{normalized}s"
        if table_name not in EVENT_TABLES:
            raise KeyError(f"Unknown event type '{event_type}'. Expected one of: {sorted(EVENT_TABLES)}")

        code_key = str(code).strip()
        token = f"{EVENT_PREFIXES[table_name]}_{code_key}"
        if token not in self.token_to_id:
            return self.event_unknown_id[table_name]
        return self.token_to_id[token]

    def decode_event(self, token_id: int) -> Optional[str]:
        return self.id_to_token.get(token_id)

    def encode_observation_code(self, code: str | int) -> int:
        code_key = str(code).strip()
        return self.observation_code_to_id.get(code_key, self.observation_unknown_id)

    def encode_observation_value(self, code: str | int, value: float) -> Optional[int]:
        if not self.quantile_thresholds:
            return None

        code_key = str(code).strip()
        thresholds = self.quantile_thresholds.get(code_key)
        if thresholds is None:
            return None

        numeric_value = float(value)
        ordered_thresholds = [thresholds[f"Q{idx}"] for idx in range(1, QUANTILE_BINS + 1)]
        for idx in range(1, QUANTILE_BINS + 1):
            if numeric_value <= ordered_thresholds[idx - 1]:
                return self.quantile_token_to_id[f"Q{idx}"]
        return self.quantile_token_to_id[f"Q{QUANTILE_BINS}"]

    def encode_demographic_value(self, column: str, value: str) -> int:
        column_key = column.strip().upper()
        if column_key not in self.demographic_value_to_id:
            raise KeyError(f"Unknown demographic column '{column}'. Expected one of: {list(DEMOGRAPHIC_COLUMNS)}")
        normalized_value = self._normalize_token_value(str(value))
        token = f"{column_key}_{normalized_value}"
        return self.token_to_id.get(token, self.demographic_unknown_id[column_key])

    def encode_age(self, age_years: float | int) -> int:
        token = self._age_token_for_value(float(age_years))
        return self.age_token_to_id[token]

    def encode_time_delta(self, delta_minutes: float | int) -> int:
        token = self._time_delta_token_for_minutes(float(delta_minutes))
        if token == "TIME_UNKNOWN":
            return self.time_delta_unknown_id if self.time_delta_unknown_id is not None else self.token_to_id[token]
        return self.time_delta_token_to_id[token]

    def save(self, output_path: Path | None = None) -> "EHRTokenizer":
        path = Path(output_path) if output_path is not None else self.output_dir / "tokenizer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, path: str | Path) -> "EHRTokenizer":
        tokenizer_path = Path(path)
        with tokenizer_path.open("rb") as f:
            return pickle.load(f)

    def from_string(self, token: str) -> int:
        return self.token_to_id[token]

    def to_string(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ETHOS-style EHR tokenizer.")
    parser.add_argument("--data-dir", type=str, default="../train_val", help="Directory containing EHR CSV tables.")
    parser.add_argument("--output-dir", type=str, default="../", help="Directory to write tokenizer metadata.")
    parser.add_argument("--include_cls", "--include-cls", dest="include_cls", action="store_true", help="Add a <CLS> special token.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    tokenizer = EHRTokenizer(data_dir=data_dir, output_dir=output_dir, include_cls=args.include_cls)
    tokenizer.save(output_dir / "tokenizer.pkl")
    loaded = EHRTokenizer.load(output_dir / "tokenizer.pkl")
    
    # testing events
    code = tokenizer.encode_event("allergies", "419474003")
    print(f"Encoded event: {code}")
    decoded = tokenizer.decode_event(code)
    print(f"Decoded event: {decoded}")
    
    # testing observations
    code = tokenizer.encode_observation_code("39156-5")
    print(f"Encoded observation code: {code}")
    quantile_code = tokenizer.encode_observation_value("39156-5", 27.7)
    print(f"Encoded observation value: {quantile_code}")
    decoded_quantile = tokenizer.decode(quantile_code)
    print(f"Decoded quantile token: {decoded_quantile}")
    decoded_observation_code = tokenizer.decode(code)
    print(f"Decoded observation code: {decoded_observation_code}")

    # sanity check
    print(f"Event codes: {len(tokenizer.event_code_to_id)}")
    print(list(tokenizer.event_code_to_id.keys())[0:5])
    print(f"Observation codes: {len(tokenizer.observation_code_to_id)}")
    print(f"Quantile bins: {QUANTILE_BINS}")
    print(f"Vocabulary size: {len(loaded.token_to_id)}")
    print(f"Observation token count: {len(loaded.observation_code_to_id) + QUANTILE_BINS}")
    print(f"Tokenizer written to: {output_dir / 'tokenizer.pkl'}")


if __name__ == "__main__":
    main()
    