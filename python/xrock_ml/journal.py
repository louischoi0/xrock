"""Readers for journal directories (`xrock backtest` / `xrock live` output): decisions.csv joined to fills.csv."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def load_decisions(journal_dir: str | Path) -> list[dict]:
    """Rows of decisions.csv: ts (int ns), symbol, kind, reason, order_id (int), plus each value as a float key."""
    path = Path(journal_dir) / "decisions.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            row = {"ts": int(r["ts"]), "symbol": r["symbol"], "kind": r["kind"], "reason": r["reason"],
                   "order_id": int(r["order_id"])}
            for kv in filter(None, r["values"].split(";")):
                k, v = kv.split("=", 1)
                row[k] = float(v)
            rows.append(row)
    return rows


def load_fills(journal_dir: str | Path) -> dict[int, dict]:
    """fills.csv keyed by order id (one fill per order in the simulator)."""
    with (Path(journal_dir) / "fills.csv").open() as f:
        return {int(r["id"]): {"ts": int(r["ts"]), "side": r["side"], "price": float(r["price"]),
                               "qty": float(r["qty"]), "fee": float(r["fee"]), "realized": float(r["realized"])}
                for r in csv.DictReader(f)}


def barrier_trades(journal_dir: str | Path) -> list[dict]:
    """One row per completed barrier-strategy trade: the enter / open / exit decisions joined to their fills.

    Keys: entry_ts, exit_ts, held_s, side (+1/-1), config (the chosen output, e.g. tb_3_long), k_tp, k_sl, hold_s,
    sigma, edge (predicted), exit_reason, entry, exit, gross (log return, side-adjusted), fees, pnl (realized - fees).
    """
    fills = load_fills(journal_dir)
    trades, current = [], None
    for d in load_decisions(journal_dir):
        if d["kind"] == "enter":
            current = {"enter": d}
        elif d["kind"] == "open" and current:
            current["open"] = d
        elif d["kind"] == "exit" and current and "open" in current and d["order_id"] in fills:
            e = current["enter"]
            fin, fout = fills[e["order_id"]], fills[d["order_id"]]
            side = int(e["side"])
            trades.append({
                "entry_ts": fin["ts"], "exit_ts": fout["ts"], "held_s": (fout["ts"] - fin["ts"]) / 1e9,
                "side": side, "config": e["reason"], "k_tp": e["k_tp"], "k_sl": e["k_sl"], "hold_s": e["hold_s"],
                "sigma": e["sigma"], "edge": e["edge"], "exit_reason": d["reason"],
                "entry": fin["price"], "exit": fout["price"],
                "gross": side * math.log(fout["price"] / fin["price"]),
                "fees": fin["fee"] + fout["fee"], "pnl": fout["realized"] - fin["fee"] - fout["fee"],
            })
            current = None
    return trades
