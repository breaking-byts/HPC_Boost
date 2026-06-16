from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleTrace:
    sample_id: str
    binary_label: int
    category: str
    family: str
    full_label: str
    events: pd.DataFrame

    @property
    def num_timesteps(self) -> int:
        return len(self.events)

    def event_array(self, event_name: str) -> np.ndarray:
        return self.events[event_name].to_numpy(dtype=np.float64)


class RadarDataLoader:
    """Chunked loader for RaDaR combined_hardware_trails.csv."""

    def __init__(
        self,
        root_dir: str = "~/hpc_boost_v2/data/radar",
        csv_name: str = "combined_hardware_trails.csv",
        chunksize: int = 100_000,
    ) -> None:
        self.root_dir = os.path.expanduser(str(root_dir))
        self.csv_file = os.path.join(self.root_dir, csv_name)
        self.chunksize = chunksize

        self.sample_id_column = "Filename"
        self.binary_label_column = "binarylabel"
        self.category_column = "goal"
        self.family_column = "family_gene"
        self.full_label_column = "full_label"

        # Non-event (metadata/label) columns to exclude when selecting HPC event
        # columns by name. The CSV interleaves metadata both before and after the
        # 55 HPC events, so a positional slice is fragile under column reordering.
        # Everything not in this set, and not an unnamed-index or *_encoded label
        # column (see _is_non_event_column), is treated as an HPC event column.
        self.non_event_columns = frozenset(
            {
                self.sample_id_column,
                self.binary_label_column,
                self.category_column,
                self.family_column,
                self.full_label_column,
                "label",
                "method",
                "keylog",
                "bkdoor",
                "infosteal",
                "rootkits",
            }
        )

        self._columns: Optional[List[str]] = None
        self._hpc_columns: Optional[List[str]] = None

    def _is_non_event_column(self, column: str) -> bool:
        """True for index/label/metadata columns that are not HPC events."""
        text = str(column)
        if text in self.non_event_columns:
            return True
        # pandas names a leading index column "Unnamed: 0" (and similar).
        if text.startswith("Unnamed"):
            return True
        # Encoded label columns (method_encoded, goal_encoded, ...) are metadata.
        if text.endswith("_encoded"):
            return True
        return False

    @property
    def columns(self) -> List[str]:
        if self._columns is None:
            self._ensure_exists()
            self._columns = list(pd.read_csv(self.csv_file, nrows=0).columns)
        return self._columns

    @property
    def hpc_columns(self) -> List[str]:
        """The 55 HPC event columns, selected by name (not position).

        Robust to column reordering: returns every column that is not a known
        metadata/label/index column. Preserves CSV column order.
        """
        if self._hpc_columns is None:
            self._hpc_columns = [
                column
                for column in self.columns
                if not self._is_non_event_column(column)
            ]
        return self._hpc_columns

    @property
    def metadata_columns(self) -> List[str]:
        return [
            self.sample_id_column,
            self.binary_label_column,
            self.category_column,
            self.family_column,
            self.full_label_column,
        ]

    @property
    def required_columns(self) -> List[str]:
        return self.metadata_columns + self.hpc_columns

    def _ensure_exists(self) -> None:
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"RaDaR CSV not found: {self.csv_file}")

    def validate(self) -> Dict[str, object]:
        self._ensure_exists()

        missing = [col for col in self.required_columns if col not in self.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not self.hpc_columns:
            raise ValueError(
                "No HPC event columns found after excluding metadata columns. "
                f"Columns present: {self.columns}"
            )

        return {
            "csv_file": self.csv_file,
            "num_columns": len(self.columns),
            "num_hpc_events": len(self.hpc_columns),
            "sample_id_column": self.sample_id_column,
            "binary_label_column": self.binary_label_column,
            "category_column": self.category_column,
            "family_column": self.family_column,
        }

    def iter_chunks(self, usecols: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
        self.validate()
        for chunk in pd.read_csv(
            self.csv_file,
            usecols=usecols,
            chunksize=self.chunksize,
            low_memory=False,
        ):
            yield chunk

    def list_sample_ids(self, limit: Optional[int] = None) -> List[str]:
        sample_ids = set()

        for chunk in self.iter_chunks(usecols=[self.sample_id_column]):
            sample_ids.update(chunk[self.sample_id_column].dropna().astype(str))
            if limit is not None and len(sample_ids) >= limit:
                break

        ordered = sorted(sample_ids)
        return ordered[:limit] if limit is not None else ordered

    def load_sample(self, sample_id: str) -> SampleTrace:
        parts = []

        for chunk in self.iter_chunks(usecols=self.required_columns):
            sample_col = chunk[self.sample_id_column].astype(str)
            matched = chunk.loc[sample_col == str(sample_id)]
            if not matched.empty:
                parts.append(matched.copy())

        if not parts:
            raise KeyError(f"Sample not found: {sample_id}")

        df = pd.concat(parts, ignore_index=True)
        return self._frame_to_trace(str(sample_id), df)

    def _frame_to_trace(self, sample_id: str, df: pd.DataFrame) -> SampleTrace:
        df = df.copy()
        df[self.hpc_columns] = df[self.hpc_columns].apply(pd.to_numeric, errors="coerce")

        first = df.iloc[0]
        return SampleTrace(
            sample_id=str(sample_id),
            binary_label=int(first[self.binary_label_column]),
            category=str(first[self.category_column]).lower(),
            family=str(first[self.family_column]),
            full_label=str(first[self.full_label_column]),
            events=df[self.hpc_columns],
        )

    def load_samples(self, sample_ids: List[str]) -> Dict[str, SampleTrace]:
        requested = [str(sample_id) for sample_id in sample_ids]
        requested_set = set(requested)
        parts: Dict[str, List[pd.DataFrame]] = {sample_id: [] for sample_id in requested}

        for chunk in self.iter_chunks(usecols=self.required_columns):
            sample_col = chunk[self.sample_id_column].astype(str)
            matched = chunk.loc[sample_col.isin(requested_set)]
            if matched.empty:
                continue

            for sample_id, group in matched.groupby(self.sample_id_column, sort=False):
                sample_key = str(sample_id)
                if sample_key in parts:
                    parts[sample_key].append(group.copy())

        missing = [sample_id for sample_id in requested if not parts[sample_id]]
        if missing:
            raise KeyError(f"Samples not found: {missing}")

        traces: Dict[str, SampleTrace] = {}
        for sample_id in requested:
            df = pd.concat(parts[sample_id], ignore_index=True)
            traces[sample_id] = self._frame_to_trace(sample_id, df)

        return traces

    def iter_sample_traces(self, sample_limit: Optional[int] = None) -> Iterator[SampleTrace]:
        sample_ids = self.list_sample_ids(limit=sample_limit)
        traces = self.load_samples(sample_ids)
        for sample_id in sample_ids:
            yield traces[sample_id]
