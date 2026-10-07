"""Baselines on an xrock dataset: what any deep model has to beat.

  B0 const  - per-label development mean, no features: the "best fixed parameters" policy.
  B1 ridge  - standardized features, closed-form ridge per label, lambda picked by an inner time split.

Walk-forward: each fold fits on its train window and is scored on its test window (embargo already in the bounds).
The holdout is scored once, by models refit on the whole development period. Scores:
  ic / rank_ic        prediction vs label, per label
  policy (tb labels)  the barrier strategy's rule offline: when flat, take the (config, side) with the best
                      prediction if it beats min_edge, book that label minus cost, stay busy for the config's horizon
                      (an upper bound on the holding time, so the trade count is conservative)
Writes models (xrock-model/1) for `xrock model-eval` and the `barrier` strategy, and report.json.

usage: PYTHONPATH=python .venv/bin/python python/train_baseline.py <dataset_dir> <out_dir> [--cost 0.0011] [--min-edge 0.0012]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from xrock_ml import load_dataset, save_linear_model

LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]


def rank(a: np.ndarray) -> np.ndarray:
    r = np.empty(len(a))
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def ic(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(p) < 3 or p.std() == 0 or y.std() == 0:
        return 0.0, 0.0
    return float(np.corrcoef(p, y)[0, 1]), float(np.corrcoef(rank(p), rank(y))[0, 1])


class Ridge:
    """Ridge on standardized features: y = W ((x - mean) / std) + b, all labels at once, one lambda per label."""

    def fit(self, X: np.ndarray, Y: np.ndarray, lambdas: np.ndarray) -> "Ridge":
        X = X.astype(np.float64)
        self.mean, self.std = X.mean(0), X.std(0)
        self.std[self.std == 0] = 1
        Z = (X - self.mean) / self.std
        self.bias = Y.mean(0)
        G, ZtY = Z.T @ Z, Z.T @ (Y - self.bias)
        n, f = Z.shape
        self.W = np.stack([np.linalg.solve(G + lam * n * np.eye(f), ZtY[:, k]) for k, lam in enumerate(lambdas)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.W.T + self.bias


def pick_lambdas(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per label, the lambda with the best rank IC on the last 20% of the training window (fit on the first 80%)."""
    cut = int(len(X) * 0.8)
    best = []
    scores = {lam: Ridge().fit(X[:cut], Y[:cut], np.full(Y.shape[1], lam)).predict(X[cut:]) for lam in LAMBDAS}
    for k in range(Y.shape[1]):
        best.append(max(LAMBDAS, key=lambda lam: ic(scores[lam][:, k], Y[cut:, k])[1]))
    return np.array(best)


def policy(pred: np.ndarray, Y: np.ndarray, ts: np.ndarray, tb_cols: list[int], horizons_ns: list[int],
           cost: float, min_edge: float) -> dict:
    """Offline barrier-strategy rule over time-ordered samples; returns trades and mean net return per trade."""
    busy_until, net = -1, []
    best = np.argmax(pred[:, tb_cols], axis=1)
    for i in range(len(ts)):
        if ts[i] < busy_until:
            continue
        j = best[i]
        if pred[i, tb_cols[j]] <= min_edge:
            continue
        net.append(Y[i, tb_cols[j]] - cost)
        busy_until = ts[i] + horizons_ns[j]
    net = np.array(net)
    return {"trades": int(len(net)), "mean_net_bp": float(1e4 * net.mean()) if len(net) else 0.0,
            "sum_net": float(net.sum()) if len(net) else 0.0, "win": float((net > 0).mean()) if len(net) else 0.0}


def duration_ns(text: str) -> int:
    m = re.fullmatch(r"([\d.]+)(ns|ms|s|m|h|d)", text)
    return int(float(m[1]) * {"ns": 1, "ms": 1e6, "s": 1e9, "m": 60e9, "h": 3600e9, "d": 86400e9}[m[2]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--cost", type=float, default=0.0011, help="round-trip cost as a log return (taker 2 x 5.5bp)")
    ap.add_argument("--min-edge", type=float, default=0.0012)
    a = ap.parse_args()

    ds = load_dataset(a.dataset)
    X, Y, ts = np.asarray(ds.X, np.float64), np.asarray(ds.Y, np.float64), np.asarray(ds.ts)
    labels = ds.label_names
    tb_cols = [k for k, n in enumerate(labels) if n.startswith("tb_")]
    tb_h = [duration_ns(ds.meta["barriers"][int(labels[k].split("_")[1])]["horizon"]) for k in tb_cols]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ts)} samples, {X.shape[1]} features, {len(labels)} labels, {len(ds.meta['folds'])} folds")

    report = {"dataset": str(a.dataset), "cost": a.cost, "min_edge": a.min_edge, "folds": []}
    for i, (tr, te) in enumerate(ds.folds()):
        lam = pick_lambdas(X[tr], Y[tr])
        ridge = Ridge().fit(X[tr], Y[tr], lam)
        p_ridge, p_const = ridge.predict(X[te]), np.tile(Y[tr].mean(0), (te.sum(), 1))
        row = {"fold": i, "train": int(tr.sum()), "test": int(te.sum()), "lambda": dict(zip(labels, lam.tolist())),
               "ic": {}, "policy": {}}
        for k, n in enumerate(labels):
            row["ic"][n] = ic(p_ridge[:, k], Y[te, k])
        if tb_cols:
            for name, p in (("const", p_const), ("ridge", p_ridge)):
                row["policy"][name] = policy(p, Y[te], ts[te], tb_cols, tb_h, a.cost, a.min_edge)
        report["folds"].append(row)
        pol = row["policy"]
        print(f"fold {i}: train {row['train']} test {row['test']}  "
              + "  ".join(f"{n} ic={row['ic'][n][0]:+.3f}/{row['ic'][n][1]:+.3f}" for n in labels if n.startswith("fwd"))
              + (f"  policy const {pol['const']['trades']}tr {pol['const']['mean_net_bp']:+.2f}bp"
                 f" | ridge {pol['ridge']['trades']}tr {pol['ridge']['mean_net_bp']:+.2f}bp" if pol else ""))

    # Refit on the whole development period; score the holdout once.
    dev, hold = ds.development_mask(), ds.holdout_mask()
    lam = pick_lambdas(X[dev], Y[dev])
    ridge = Ridge().fit(X[dev], Y[dev], lam)
    const = Y[dev].mean(0)
    hold_row = {"samples": int(hold.sum()), "ic": {}, "policy": {}}
    if hold.any():
        p = ridge.predict(X[hold])
        hold_row["ic"] = {n: ic(p[:, k], Y[hold, k]) for k, n in enumerate(labels)}
        if tb_cols:
            hold_row["policy"] = {
                "const": policy(np.tile(const, (hold.sum(), 1)), Y[hold], ts[hold], tb_cols, tb_h, a.cost, a.min_edge),
                "ridge": policy(p, Y[hold], ts[hold], tb_cols, tb_h, a.cost, a.min_edge)}
    report["holdout"] = hold_row

    F = X.shape[1]
    fwd = [n for n in labels if n.startswith("fwd_")]
    tb = [labels[k] for k in tb_cols]
    for name, outs in (("fwd", fwd), ("tb", tb)):
        if not outs:
            continue
        idx = [labels.index(n) for n in outs]
        save_linear_model(out / f"ridge_{name}.json", ds.meta, outs, ridge.mean, ridge.std, ridge.W[idx], ridge.bias[idx])
        save_linear_model(out / f"const_{name}.json", ds.meta, outs, np.zeros(F), np.ones(F), np.zeros((len(outs), F)),
                          const[idx])
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("holdout:", json.dumps({k: v for k, v in hold_row.items() if k != "ic"}),
          " ".join(f"{n} ic={hold_row['ic'][n][0]:+.3f}" for n in fwd) if hold_row["ic"] else "")
    print("models + report ->", out)


if __name__ == "__main__":
    main()
