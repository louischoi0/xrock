"""Sequence model (dilated temporal CNN) on an xrock dataset, scored exactly like python/train_baseline.py.

Input: the last `window` consecutive samples [T, F] (xrock SampleWindow contract); output: every label of the dataset
(fwd_* and tb_*), multi-task. Feature standardization (clipped at +-8 sigma) and label de-standardization live inside
the network, so the exported ONNX graph takes raw features and returns raw log returns - the units the `barrier`
strategy's min_edge and `xrock model-eval` expect.

Walk-forward folds: fit on the fold's train window (last 15% of it, in time, for early stopping), score its test window.
Holdout: refit on the whole development period, score once, export models/<out>/deep.onnx + deep.json.

usage: PYTHONPATH=python .venv/bin/python python/train_deep.py <dataset_dir> <out_dir> [--window 60] [--epochs 12]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_baseline import duration_ns, ic, policy
from xrock_ml import load_dataset, save_onnx_model_header


class TemporalCNN(nn.Module):
    def __init__(self, f: int, o: int, mean, std, ymu, ysd, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("ymu", torch.as_tensor(ymu, dtype=torch.float32))
        self.register_buffer("ysd", torch.as_tensor(ysd, dtype=torch.float32))
        layers, c = [], f
        for d in (1, 2, 4, 8):  # receptive field 31 samples
            layers += [nn.Conv1d(c, hidden, kernel_size=3, dilation=d), nn.GELU(), nn.Dropout(dropout)]
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, o))

    def standardized(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F] raw -> standardized outputs [B, O]
        z = torch.clamp((x - self.mean) / self.std, -8.0, 8.0).transpose(1, 2)
        h = self.tcn(z)
        return self.head(torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # raw log returns, the exported contract
        return self.standardized(x) * self.ysd + self.ymu


def windows(X: torch.Tensor, ends: np.ndarray, T: int) -> torch.Tensor:
    idx = torch.as_tensor(ends[:, None] + np.arange(-T + 1, 1)[None, :])
    return X[idx]


def predict(model: TemporalCNN, X: torch.Tensor, ends: np.ndarray, T: int, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ends), batch):
            out.append(model(windows(X, ends[i:i + batch], T)).numpy())
    return np.concatenate(out) if out else np.zeros((0, model.ymu.numel()), np.float32)


def fit(X: torch.Tensor, Y: np.ndarray, ends: np.ndarray, T: int, a, seed: int) -> tuple[TemporalCNN, dict]:
    """Train on `ends` (time-ordered window ends); the last 15% validate for early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cut = int(len(ends) * 0.85)
    tr, va = ends[:cut], ends[cut:]
    rows = np.unique((tr[:, None] + np.arange(-T + 1, 1)[None, :]).ravel()[:: max(1, T // 4)])
    mean, std = X[rows].mean(0).numpy(), X[rows].std(0).numpy()
    std[std == 0] = 1
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
    ysd[ysd == 0] = 1
    model = TemporalCNN(X.shape[1], Y.shape[1], mean, std, ymu, ysd, a.hidden, a.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    Yt = torch.as_tensor((Y - ymu) / ysd, dtype=torch.float32)
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(a.epochs):
        model.train()
        t0 = time.time()
        sel = tr[rng.integers(0, a.stride) :: a.stride]
        sel = sel[rng.permutation(len(sel))]
        total = 0.0
        for i in range(0, len(sel), a.batch):
            b = sel[i:i + a.batch]
            loss = loss_fn(model.standardized(windows(X, b, T)), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vsel = va[:: a.stride]
            vloss = np.mean([loss_fn(model.standardized(windows(X, vsel[i:i + 8192], T)), Yt[vsel[i:i + 8192]]).item()
                             for i in range(0, len(vsel), 8192)])
        hist.append({"epoch": epoch, "train": total / len(sel), "val": float(vloss), "sec": time.time() - t0})
        print(f"    epoch {epoch:2d} train {total / len(sel):.5f} val {vloss:.5f} ({time.time() - t0:.0f}s)", flush=True)
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val": float(best), "history": hist}


def score(pred, Y, ts, labels, tb_cols, tb_h, a) -> dict:
    row = {"ic": {n: ic(pred[:, k], Y[:, k]) for k, n in enumerate(labels)}}
    if tb_cols:
        row["policy"] = policy(pred, Y, ts, tb_cols, tb_h, a.cost, a.min_edge)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3, help="train on every n-th window end per epoch (random offset)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cost", type=float, default=0.0011)
    ap.add_argument("--min-edge", type=float, default=0.0012)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ds = load_dataset(a.dataset)
    T = a.window
    X = torch.as_tensor(np.array(ds.X, np.float32))  # copy: the mmap is read-only
    Y, ts = np.asarray(ds.Y, np.float64), np.asarray(ds.ts)
    labels = ds.label_names
    tb_cols = [k for k, n in enumerate(labels) if n.startswith("tb_")]
    tb_h = [duration_ns(ds.meta["barriers"][int(labels[k].split("_")[1])]["horizon"]) for k in tb_cols]
    ends = ds.windows(T)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ts)} samples, {len(ends)} windows of {T}, {X.shape[1]} features, {len(labels)} labels")

    report = {"dataset": str(a.dataset), "args": vars(a), "folds": []}
    for i, (tr_mask, te_mask) in enumerate(ds.folds()):
        tr, te = ends[tr_mask[ends]], ends[te_mask[ends]]
        print(f"fold {i}: train {len(tr)} test {len(te)}", flush=True)
        model, info = fit(X, Y, tr, T, a, a.seed)
        row = score(predict(model, X, te, T), Y[te], ts[te], labels, tb_cols, tb_h, a) | {"train": len(tr), "test": len(te), **info}
        report["folds"].append(row)
        print(f"  fold {i}: " + "  ".join(f"{n} ic={row['ic'][n][0]:+.3f}/{row['ic'][n][1]:+.3f}" for n in labels if n.startswith("fwd"))
              + (f"  policy {row['policy']['trades']}tr {row['policy']['mean_net_bp']:+.2f}bp" if tb_cols else ""), flush=True)

    dev = ends[ds.development_mask()[ends]]
    hold = ends[ds.holdout_mask()[ends]]
    print(f"final: development {len(dev)} holdout {len(hold)}", flush=True)
    model, info = fit(X, Y, dev, T, a, a.seed)
    p_hold = predict(model, X, hold, T)
    report["holdout"] = score(p_hold, Y[hold], ts[hold], labels, tb_cols, tb_h, a) | {"samples": len(hold), **info}
    np.save(out / "deep_holdout_pred.npy", p_hold)
    np.save(out / "deep_holdout_ts.npy", ts[hold])

    model.eval()
    torch.onnx.export(model, (torch.zeros(1, T, X.shape[1]),), str(out / "deep.onnx"), input_names=["x"],
                      output_names=["y"], opset_version=17, dynamo=False)
    save_onnx_model_header(out / "deep.json", ds.meta, labels, "deep.onnx", window=T)
    (out / "deep_report.json").write_text(json.dumps(report, indent=1))
    h = report["holdout"]
    print("holdout: " + "  ".join(f"{n} ic={h['ic'][n][0]:+.3f}" for n in labels if n.startswith("fwd"))
          + (f"  policy {h['policy']['trades']}tr {h['policy']['mean_net_bp']:+.2f}bp" if tb_cols else ""))
    print("model ->", out / "deep.json")


if __name__ == "__main__":
    main()
