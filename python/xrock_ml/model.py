"""Writers for "xrock-model/1" files (read by src/ml/model.cpp)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _header(meta: dict, outputs: list[str], model_type: str, window: int = 1) -> dict:
    missing = [o for o in outputs if o not in meta["labels"]]
    if missing:
        raise ValueError(f"outputs {missing} are not labels of the dataset {meta['labels']}")
    return {
        "format": "xrock-model/1",
        "type": model_type,
        "symbol": meta["symbol"],
        "features": dict(meta["features"]),  # names, sample, max_gap: the contract the model was trained on
        "outputs": list(outputs),
        "window": int(window),  # rows of consecutive samples the model reads, oldest first
        # the barrier grid behind tb_<i>_long/short outputs (the strategy acts on the same barriers)
        "barriers": meta.get("barriers", []),
        "barrier_vol": meta.get("barrier_vol", ""),
    }


def save_linear_model(path: str | Path, meta: dict, outputs: list[str], mean, std, weights, bias,
                      window: int = 1) -> None:
    """y = W ((x - mean) / std) + b on the flattened window (window * features values, oldest row first)."""
    f = window * len(meta["features"]["names"])
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    weights, bias = np.asarray(weights, np.float64).reshape(len(outputs), f), np.asarray(bias, np.float64)
    if mean.shape != (f,) or std.shape != (f,) or bias.shape != (len(outputs),) or (std <= 0).any():
        raise ValueError("linear parameter shapes do not match the feature/output counts, or std <= 0")
    doc = _header(meta, outputs, "linear", window) | {
        "mean": mean.tolist(), "std": std.tolist(), "weights": weights.tolist(), "bias": bias.tolist()}
    Path(path).write_text(json.dumps(doc, indent=1))


def save_onnx_model_header(path: str | Path, meta: dict, outputs: list[str], onnx_file: str,
                           window: int = 1) -> None:
    """Header for an ONNX network: input float32 [1, F] (window 1) or [1, window, F] of raw features (normalization
    inside the graph), output float32 [1, len(outputs)]. The C++ runner checks these shapes at load."""
    doc = _header(meta, outputs, "onnx", window) | {"file": onnx_file}
    Path(path).write_text(json.dumps(doc, indent=1))
