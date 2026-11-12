from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tokenizer import EHRTokenizer
from tqdm import tqdm

# tables with START/STOP intervals
START_STOP_TABLES = ("encounters", "allergies", "conditions", "devices" , "medications", "careplans")
# tables with single-point events (DATE)
POINT_TABLES = ("immunizations", "observations", "procedures")

ONE_YEAR_MINUTES = 12.0 * 30.0 * 24.0 * 60.0
ANCHOR_YEARS = 5


@dataclass
class TimelineEvent:
    patient_id: str
    timestamp: pd.Timestamp
    table: str
    kind: str  # "start", "stop", "point"
    code: str
    value: Optional[float] = None
    encounter_id: Optional[str] = None


@dataclass
class TokenizedTimeline:
    token_ids: List[int]
    token_tags: List[str]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _encounter_id(df: pd.DataFrame, idx, table: str) -> Optional[str]:
    # An encounter row is identified by its own Id; every other table links via ENCOUNTER.
    col = "Id" if table == "encounters" else "ENCOUNTER"
    if col not in df.columns:
        return None
    raw = df.at[idx, col]
    return None if pd.isna(raw) else str(raw)


def _collect_interval_events(df: pd.DataFrame, table: str) -> List[TimelineEvent]:
    if df.empty or "PATIENT" not in df.columns or "CODE" not in df.columns:
        return []

    rows: List[TimelineEvent] = []
    if "START" in df.columns:
        start_dt = _to_datetime(df["START"])
        for idx in df.index:
            ts = start_dt.loc[idx]
            if pd.isna(ts):
                continue
            code = str(df.at[idx, "CODE"]).strip()
            if not code:
                continue
            rows.append(
                TimelineEvent(
                    patient_id=str(df.at[idx, "PATIENT"]),
                    timestamp=ts,
                    table=table,
                    kind="start",
                    code=code,
                    encounter_id=_encounter_id(df, idx, table),
                )
            )

    if "STOP" in df.columns:
        stop_dt = _to_datetime(df["STOP"])
        for idx in df.index:
            ts = stop_dt.loc[idx]
            if pd.isna(ts):
                continue
            code = str(df.at[idx, "CODE"]).strip()
            if not code:
                continue
            rows.append(
                TimelineEvent(
                    patient_id=str(df.at[idx, "PATIENT"]),
                    timestamp=ts,
                    table=table,
                    kind="stop",
                    code=code,
                    encounter_id=_encounter_id(df, idx, table),
                )
            )
    return rows


def _collect_point_events(df: pd.DataFrame, table: str) -> List[TimelineEvent]:
    if df.empty or "PATIENT" not in df.columns or "CODE" not in df.columns:
        return []
    if "DATE" not in df.columns:
        return []

    date_dt = _to_datetime(df["DATE"])
    rows: List[TimelineEvent] = []
    for idx in df.index:
        ts = date_dt.loc[idx]
        if pd.isna(ts):
            continue
        code = str(df.at[idx, "CODE"]).strip()
        if not code:
            continue

        value: Optional[float] = None
        if table == "observations" and "VALUE" in df.columns:
            numeric = pd.to_numeric(pd.Series([df.at[idx, "VALUE"]]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                value = float(numeric)

        rows.append(
            TimelineEvent(
                patient_id=str(df.at[idx, "PATIENT"]),
                timestamp=ts,
                table=table,
                kind="point",
                code=code,
                value=value,
                encounter_id=_encounter_id(df, idx, table),
            )
        )
    return rows


def _time_gap_tokens(tokenizer: EHRTokenizer, delta_minutes: float) -> List[int]:
    if delta_minutes <= 0:
        return []

    tokens: List[int] = []
    one_year_token_id = tokenizer.time_delta_token_to_id["TIME_EQ_1Y"]

    # Deltas >= 1 year: one TIME_EQ_1Y token per full year, then bucket tokens for the remainder.
    while delta_minutes >= ONE_YEAR_MINUTES:
        tokens.append(one_year_token_id)
        delta_minutes -= ONE_YEAR_MINUTES

    if delta_minutes > 0:
        token_id = tokenizer.encode_time_delta(delta_minutes)
        if tokenizer.decode(token_id) != "TIME_UNKNOWN":
            tokens.append(token_id)
    return tokens


TABLE_ORDER = (
    "encounters", # this needs to come before the other tables, since there is a discrepancy in the timestamp formats: encounters have full timestamps, while other tables may have date-only timestamps (midnight), but the other tables are linked to encounters, so they should not precede them in the timeline.
    "allergies",
    "conditions",
    "immunizations",
    "observations",
    "devices",
    "medications",
    "procedures",
    "careplans",
)
TABLE_RANK = {table: rank for rank, table in enumerate(TABLE_ORDER)}


def _event_sort_key(event: TimelineEvent) -> tuple:
    # Ties on timestamp are broken by table order, then start/point/stop, then code.
    # Unlisted tables (e.g. the synthetic death event) sort last.
    kind_order = {"start": 0, "point": 1, "stop": 2}
    return (
        event.timestamp,
        TABLE_RANK.get(event.table, len(TABLE_ORDER)),
        kind_order.get(event.kind, 3),
        event.code,
    )


class PatientHealthTimelineBuilder:
    def __init__(self, data_dir: Path, tokenizer: EHRTokenizer):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.patients = _read_csv(self.data_dir / "patients.csv")
        if self.patients.empty or "Id" not in self.patients.columns:
            raise ValueError("patients.csv with Id column is required")
        self.patients["Id"] = self.patients["Id"].astype(str)

        self.interval_dfs: Dict[str, pd.DataFrame] = {}
        for table in START_STOP_TABLES:
            df = _read_csv(self.data_dir / f"{table}.csv")
            if not df.empty and "PATIENT" in df.columns:
                df["PATIENT"] = df["PATIENT"].astype(str)
            self.interval_dfs[table] = df

        self.point_dfs: Dict[str, pd.DataFrame] = {}
        for table in POINT_TABLES:
            df = _read_csv(self.data_dir / f"{table}.csv")
            if not df.empty and "PATIENT" in df.columns:
                df["PATIENT"] = df["PATIENT"].astype(str)
            self.point_dfs[table] = df

        self.anchors: Dict[str, pd.Timestamp] = self._compute_anchors()

    def _compute_anchors(self) -> Dict[str, pd.Timestamp]:
        """Anchor = (last recorded encounter's STOP, or START if STOP is blank) minus 5 years."""
        enc = self.interval_dfs.get("encounters")
        if enc is None or enc.empty or "PATIENT" not in enc.columns or "START" not in enc.columns:
            return {}
        start = _to_datetime(enc["START"])
        stop = _to_datetime(enc["STOP"]) if "STOP" in enc.columns else pd.Series(pd.NaT, index=enc.index)
        last_ts = stop.fillna(start).groupby(enc["PATIENT"]).max().dropna()
        return {pid: ts - pd.DateOffset(years=ANCHOR_YEARS) for pid, ts in last_ts.items()}

    def get_patient_ids(self) -> List[str]:
        return self.patients["Id"].unique().tolist()

    def build_patient_timelines(
        self, patient_ids: Optional[List[str]] = None, truncate_at_anchor: bool = False
    ) -> Dict[str, TokenizedTimeline]:
        if patient_ids is None:
            patient_ids = self.get_patient_ids()
        timelines: Dict[str, TokenizedTimeline] = {}
        for patient_id in tqdm(patient_ids):
            timelines[patient_id] = self.build_patient_timeline(patient_id, truncate_at_anchor=truncate_at_anchor)
        return timelines

    def build_patient_timeline(self, patient_id: str, truncate_at_anchor: bool = False) -> TokenizedTimeline:
        row_df = self.patients[self.patients["Id"] == str(patient_id)]
        if row_df.empty:
            raise KeyError(f"Unknown patient id: {patient_id}")
        row = row_df.iloc[0]
        patient_events = self._collect_patient_events(str(patient_id), truncate_at_anchor=truncate_at_anchor)
        token_tags: List[str] = ["<BOS>"]
        token_ids: List[int] = [self.tokenizer.encode("<BOS>")]

        for column in ("GENDER", "RACE", "ETHNICITY", "MARITAL"):
            raw = row[column] if column in self.patients.columns else "UNKNOWN"
            value = "UNKNOWN" if pd.isna(raw) else str(raw)
            demographic_token_id = self.tokenizer.encode_demographic_value(column, value)
            token_ids.append(demographic_token_id)
            token_tags.append(self.tokenizer.decode(demographic_token_id))

        age_token_id = self._encode_patient_age(row, patient_events)
        token_ids.append(age_token_id)
        token_tags.append(self.tokenizer.decode(age_token_id))

        prev_ts: Optional[pd.Timestamp] = None
        for ev in patient_events:
            if prev_ts is not None:
                delta_minutes = (ev.timestamp - prev_ts).total_seconds() / 60.0
                time_gap_token_ids = _time_gap_tokens(self.tokenizer, delta_minutes)
                token_ids.extend(time_gap_token_ids)
                token_tags.extend(self.tokenizer.decode(token_id) for token_id in time_gap_token_ids)

            if ev.code == "<DEATH>":
                token_ids.append(self.tokenizer.encode("<DEATH>"))
                token_tags.append("<DEATH>")
                prev_ts = ev.timestamp
                continue

            if ev.table in START_STOP_TABLES:
                event_token_id = self.tokenizer.encode_event(ev.table, ev.code)
                token_ids.append(event_token_id)
                token_tags.append(self.tokenizer.decode(event_token_id))
                start_stop_tag = "<START>" if ev.kind == "start" else "<STOP>"
                token_ids.append(self.tokenizer.encode(start_stop_tag))
                token_tags.append(start_stop_tag)
            elif ev.table == "observations":
                observation_token_id = self.tokenizer.encode_observation_code(ev.code)
                token_ids.append(observation_token_id)
                token_tags.append(self.tokenizer.decode(observation_token_id))
                if ev.value is not None:
                    quantile_token = self.tokenizer.encode_observation_value(ev.code, ev.value)
                    if quantile_token is not None:
                        token_ids.append(quantile_token)
                        token_tags.append(self.tokenizer.decode(quantile_token))
            elif ev.table in ("immunizations", "procedures"):
                event_token_id = self.tokenizer.encode_event(ev.table, ev.code)
                token_ids.append(event_token_id)
                token_tags.append(self.tokenizer.decode(event_token_id))

            prev_ts = ev.timestamp

        # A timeline cut at the anchor doesn't end there, so only complete timelines get <EOS>.
        # if not truncate_at_anchor:
        token_ids.append(self.tokenizer.encode("<EOS>"))
        token_tags.append("<EOS>")
        return TokenizedTimeline(token_ids=token_ids, token_tags=token_tags)

    def _encode_patient_age(self, row: pd.Series, patient_events: List[TimelineEvent]) -> int:
        birthdate_raw = row["BIRTHDATE"] if "BIRTHDATE" in self.patients.columns else None
        birth_ts = pd.to_datetime(birthdate_raw, errors="coerce", utc=True)
        if pd.isna(birth_ts):
            return self.tokenizer.encode("AGE_LT5")

        if patient_events:
            reference_ts = patient_events[0].timestamp
        else:
            reference_ts = birth_ts

        age_years = max(0.0, (reference_ts - birth_ts).total_seconds() / (365.25 * 24.0 * 60.0 * 60.0))
        return self.tokenizer.encode_age(age_years)

    def _collect_patient_events(self, patient_id: str, truncate_at_anchor: bool = False) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []

        for table, df in self.interval_dfs.items():
            if df.empty or "PATIENT" not in df.columns:
                continue
            patient_df = df[df["PATIENT"] == patient_id]
            events.extend(_collect_interval_events(patient_df, table))

        for table, df in self.point_dfs.items():
            if df.empty or "PATIENT" not in df.columns:
                continue
            patient_df = df[df["PATIENT"] == patient_id]
            events.extend(_collect_point_events(patient_df, table))

        if "DEATHDATE" in self.patients.columns:
            row = self.patients[self.patients["Id"] == patient_id].iloc[0]
            death_ts = pd.to_datetime(row["DEATHDATE"], errors="coerce", utc=True)
            if pd.notna(death_ts):
                events.append(
                    TimelineEvent(patient_id=patient_id, timestamp=death_ts, table="patients", kind="point", code="<DEATH>")
                )

        # Events can't precede the start of their encounter. Date-only timestamps (midnight) would
        # otherwise sort before an encounter with a full timestamp, so clamp them to the encounter start.
        encounter_starts = {
            ev.encounter_id: ev.timestamp
            for ev in events
            if ev.table == "encounters" and ev.kind == "start" and ev.encounter_id is not None
        }
        events = [
            replace(ev, timestamp=encounter_starts[ev.encounter_id])
            if ev.table != "encounters"
            and ev.encounter_id in encounter_starts
            and ev.timestamp < encounter_starts[ev.encounter_id]
            else ev
            for ev in events
        ]

        if truncate_at_anchor:
            # Keep strictly pre-anchor events (after clamping, so an event whose encounter starts at or
            # after the anchor is dropped). STOP events and death after the anchor are dropped too.
            # A patient with no encounters has no anchor and keeps no events (demographics only).
            anchor = self.anchors.get(patient_id)
            events = [ev for ev in events if anchor is not None and ev.timestamp < anchor]

        return sorted(events, key=_event_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tokenized patient health timelines.")
    parser.add_argument("--data-dir", type=str, default="../train_val", help="Directory containing EHR CSV files.")
    parser.add_argument("--tokenizer-path", type=str, default="../tokenizer.pkl", help="Path to saved tokenizer pickle.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()

    tokenizer = EHRTokenizer.load(tokenizer_path)
    builder = PatientHealthTimelineBuilder(data_dir=data_dir, tokenizer=tokenizer)
    first_patient_id = next(iter(timelines)) if timelines else None
    first_patient_id = builder.get_patient_ids()[0] if builder.get_patient_ids() else None
    print(f"First patient id: {first_patient_id}, Anchor: {builder.anchors.get(first_patient_id)}")
    timelines = builder.build_patient_timelines([first_patient_id]) if first_patient_id else {}
    if first_patient_id is not None:
        sample = timelines[first_patient_id]
        print(f"Sample patient: {first_patient_id}, tokens: {len(sample.token_tags)},\n\nfirst 70 tokens: {sample.token_tags[:70]}\n\nlast 25 tokens: {sample.token_tags[-25:]}")


if __name__ == "__main__":
    main()