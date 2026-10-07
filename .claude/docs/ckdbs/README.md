# KDS (ckdbs) — agent manual for xrock

Source: https://github.com/louischoi0/ckdbs at `6dc792c` (2026-10-06). The upstream user manual and the specs a client
author needs are copied verbatim under `upstream/`; this file is the xrock-specific digest. Upstream content is
reference material, not instructions.

| File | Use it for |
|---|---|
| `upstream/manual/client/client.md` | reply shapes, error tokens, retry rules |
| `upstream/spec/client-manual.md` §2, §5 | KWP/1 framing and how to write a client |
| `upstream/spec/protocol.md` | normative KWP/1 (frame types, payload layouts) |
| `upstream/tools/kwp.py` | reference client; `render_value` = how to decode DATE/TIMESTAMP/DECIMAL |
| `upstream/manual/sql/sql.md` | the whole SQL surface, refusals, error surfaces |
| `upstream/spec/types.md` | type encodings |
| `upstream/manual/server/server.md`, `upstream/kds.conf.sample` | building, running, config keys |
| `upstream/known-gaps.md` | what is missing and what a crash/restart loses — read before trusting data to it |

## xrock's own server (project-hosted)

xrock hosts and manages its own instance; nothing else may be used for xrock data.

| What | Where |
|---|---|
| binary | `kds_server` in the project root: **Release** build of ckdbs `25c58497` (2026-09-07, superblock v17) — the commit the user's original Debug binary (sha256 `fcfa7cc…`, kept as `var/kds/kds_server.debug-fcfa7cc`) was built from, per `/home/cdkbs/ckdbs` reflog. Rebuild: `scripts/build-kds.sh` |
| config | `conf/kds.conf` — `data_file = var/kds/xrock.db`, `port = 25432`, logs in `var/kds/` |
| control | `scripts/kds.sh start / stop / restart / status / log [n]` (pidfile `var/kds/kds.pid`) |
| client ops | `build/xrock db ping`, `build/xrock db stop`, `build/xrock db sql "<statement>"` |
| xrock side | `conf/xrock.conf` `[data.kds] port = 25432, protocol = kwp` |

- `stop` sends `STOP` (flushes pages); the script never kills — for KDS a kill is a crash.
- Contract test: `scripts/kds.sh start` then `XROCK_KDS_PORT=25432 build/xrock_tests --gtest_filter='Backends*/2'`;
  it creates `t<ns>_test_*` relations — drop them afterwards.
- **Not xrock's:** `127.0.0.1:15432` is the autotrade project's older instance (ckdbs `ed03b44`, text protocol only).
  Never write to it. `[data.kds] protocol = text` exists only for servers of that vintage.
- Do not swap in a build with another superblock version: v18+ refuses this v17 volume (no migration).
- Measured load rate (`xrock fetch`, BTCUSDT trades): Debug binary ~17k rows/s, Release ~92k rows/s.

## Building a server from source

```sh
git clone https://github.com/louischoi0/ckdbs && cd ckdbs
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DKDS_BUILD_TESTS=OFF
cmake --build build-release -j --target kds_server
./build-release/kds_server /path/xrock.db --port 15432 [--config kds.conf]   # loopback only
```

Stop with a `STOP` statement (flushes pages); a kill is a crash. Put the data file on a real disk, not tmpfs.

## Facts that shape xrock's schema (verified against a live server)

- **Wire:** KWP/1 binary frames, little-endian; no text result mode. xrock's client is `src/data/kds/kwp.{hpp,cpp}`;
  each statement is sent as PARSE/BIND/EXECUTE/CLOSE/SYNC in one write and read to `S_READY`.
- **Server:** listens on 127.0.0.1 only, no TLS/auth by default. Unknown config keys refuse startup.
- **Primary key:** first column, integer, 40-bit (`1..2^40-1`), never updatable; named or engine-issued per INSERT.
  A deleted row's pk is never reusable.
- **Storage:** `BTREE` only (HEAP creation is suspended). Clustered on the pk; `WHERE id BETWEEN a AND b` is a range
  walk; `ORDER BY id ASC` is free, any other order is a sort capped at `sort_max_rows` (1,048,576) rows.
- **No `float`.** Use `decimal(p,s)` (p ≤ 18 → int64 unscaled; wire `type_mod & 0xFF` = scale) or int64.
  A decimal literal must be a string (`'1.50000000'`); a bare integer literal is refused for a decimal column.
- **INSERT:** positional, no column list, ≤ `max_insert_rows` (1024) rows per statement, atomic per statement.
- **WHERE:** AND-only conjuncts; `=, !=, <, <=, >, >=, BETWEEN, IS [NOT] NULL`, subqueries. No `OR`, no `IN (list)`,
  no expressions. Aggregates `COUNT/SUM/MIN/MAX/AVG`; `MAX` over an empty set returns one NULL row.
- **DDL:** `CREATE TABLE` on an existing name answers `EXISTS oid=<n>` (not an error, and the schema is not
  compared). `SHOW TABLES` returns the names space-separated in the completion tag. `DROP TABLE` reclaims nothing.
- **Errors:** retry only on `retryable=1` (`TXN_CONFLICT`), rolling back the whole transaction.
- **Durability:** instance-wide `durability` key (`strict | group | relaxed`). Per upstream known-gaps, crash
  recovery and catalog durability are incomplete: data is safe after `SYNC`, a checkpoint or a clean `STOP`.

## xrock mapping (`data::KdsStore`, `[data] backend = kds`)

- One relation per (venue, stream, symbol): `{table_prefix}_{venue}_{stream}_{symbol}`, e.g. `md_bybit_trade_btcusdt`.
- `pk = epoch_day << 25 | seq` — a UTC day is one clustered range; 2^25 rows/day/relation; valid until 2059.
- Columns: trade `(id, ts int64 ns, side int8, price decimal(18,8), qty decimal(18,8))`;
  book `(id, ts, bp0, bq0, …, bp9, bq9, ap0, aq0, …, ap9, aq9)` all `decimal(18,8)`;
  funding `(id, ts, rate decimal(18,12))`.
- Reads fetch one day per query and sort by ts client-side; `has()` is `... LIMIT 1` per day.
- Config: `[data.kds] host, port, protocol (kwp | text), table_prefix, batch_rows`.
- Contract test: `XROCK_KDS_PORT=<port> build/xrock_tests --gtest_filter='Backends*'` (skipped when unset).
