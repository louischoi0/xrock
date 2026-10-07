"""Loader for `xrock dataset` exports (format "xrock-dataset/1")."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray        # float32 [N, F], meta["features"]["names"] order
    Y: np.ndarray        # float32 [N, L], meta["labels"] order
    ts: np.ndarray       # int64 [N], sample time in ns (features use events strictly before it)
    meta: dict

    @property
    def feature_names(self) -> list[str]:
        return self.meta["features"]["names"]

    @property
    def label_names(self) -> list[str]:
        return self.meta["labels"]

    def label(self, name: str) -> np.ndarray:
        return self.Y[:, self.label_names.index(name)]

    def folds(self):
        """Yields (train_mask, test_mask) per walk-forward fold. The embargo between them is already in the bounds."""
        for f in self.meta["folds"]:
            train = (self.ts >= f["train"][0]) & (self.ts < f["train"][1])
            test = (self.ts >= f["test"][0]) & (self.ts < f["test"][1])
            yield train, test

    def holdout_mask(self) -> np.ndarray:
        """The final, untouched test. Evaluate on it once, at the end."""
        start = self.meta["holdout_start"]
        if start is None:
            return np.zeros(len(self.ts), dtype=bool)
        return self.ts >= start

    def development_mask(self) -> np.ndarray:
        return ~self.holdout_mask()

    def sample_ns(self) -> int:
        units = {"ms": 10**6, "s": 10**9, "m": 60 * 10**9, "h": 3600 * 10**9, "d": 86400 * 10**9, "ns": 1}
        text = self.meta["features"]["sample"]
        for unit in ("ms", "ns", "s", "m", "h", "d"):
            if text.endswith(unit):
                return int(float(text[: -len(unit)]) * units[unit])
        raise ValueError(f"bad sample duration {text!r}")

    def windows(self, length: int) -> np.ndarray:
        """Row indices i whose window rows i-length+1 .. i are consecutive grid samples (ts one `sample` apart).

        The same rule as the engine's ml::SampleWindow, applied to exported rows, so every window found here is a
        window the engine holds with the same rows. Label and fold masks apply to the window's last row."""
        if length < 1:
            raise ValueError("length must be >= 1")
        step = np.diff(np.asarray(self.ts)) == self.sample_ns()
        # run[i] = number of consecutive rows ending at i
        run = np.ones(len(self.ts), dtype=np.int64)
        breaks = np.flatnonzero(~step) + 1
        starts = np.concatenate(([0], breaks))
        for s, e in zip(starts, np.concatenate((breaks, [len(self.ts)]))):
            run[s:e] = np.arange(1, e - s + 1)
        return np.flatnonzero(run >= length)

    def window_X(self, length: int, ends: np.ndarray) -> np.ndarray:
        """float32 [len(ends), length, F]: the model input for each window end (oldest row first)."""
        offsets = np.arange(-length + 1, 1)
        return np.asarray(self.X)[ends[:, None] + offsets[None, :]]


def load_dataset(path: str | Path, mmap: bool = True) -> Dataset:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format") != "xrock-dataset/1":
        raise ValueError(f"{path}: not an xrock-dataset/1 export")
    mode = "r" if mmap else None
    ds = Dataset(np.load(path / "X.npy", mmap_mode=mode), np.load(path / "Y.npy", mmap_mode=mode),
                 np.load(path / "ts.npy", mmap_mode=mode), meta)
    n = len(ds.ts)
    if ds.X.shape != (n, len(ds.feature_names)) or ds.Y.shape != (n, len(ds.label_names)):
        raise ValueError(f"{path}: array shapes do not match meta.json")
    return ds
