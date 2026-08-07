"""
user_logs_aggregate.py — aggregate KKBox daily listening logs → monthly
========================================================================
RUN THIS ON YOUR OWN MACHINE (one uninterrupted run, ~45–90 min):

    python3 user_logs_aggregate.py

It streams user_logs.csv.7z (30.5 GB raw) + user_logs_v2.csv.7z directly
from the archives — nothing is extracted to disk — and produces

    data_cache/engagement_monthly.npz

with per-user × per-month engagement features (months 2015-01 … 2017-03):
    active_days   days with any listening
    total_secs    listening seconds (clipped to [0, 26h/day] vs junk)
    plays         songs played >98.5% through (num_985 + num_100)
    skips         songs abandoned <25% through (num_25)
    uniq_songs    sum of daily unique-song counts

Needs: python3 with numpy + pandas, ~5 GB free RAM.
7z streaming uses the system libarchive; if it errors, either
`brew install libarchive` or extract the two CSVs into the
kkbox_churn_prediction_kaggle_data/ folder (e.g. with Keka) and re-run —
plain CSVs are picked up automatically.

Checkpoints every ~40M rows into data_cache/scratch_logs/, so an
interrupted run resumes without losing aggregates (it must still re-read
the already-processed part of the stream, so try to let it finish in one
go).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "kkbox_churn_prediction_kaggle_data")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
SCRATCH = os.path.join(CACHE_DIR, "scratch_logs")
os.makedirs(SCRATCH, exist_ok=True)
sys.path.insert(0, BASE_DIR)
from sevenz_stream import open_7z_buffered  # noqa: E402

FILES = ["user_logs.csv.7z", "user_logs_v2.csv.7z"]
MONTH0 = 2015 * 12 + 0                # 2015-01 → index 0
N_MONTHS = 27                         # …2017-03
METRICS = ["active_days", "total_secs", "plays", "skips", "uniq_songs"]
CHUNK = 4_000_000
CKPT_EVERY = 40_000_000
MAX_DAY_SECS = 26 * 3600.0            # generous clip for junk totals

DTYPES = {"msno": object, "date": np.int32, "num_25": np.int32,
          "num_50": np.int32, "num_75": np.int32, "num_985": np.int32,
          "num_100": np.int32, "num_unq": np.int32, "total_secs": np.float64}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def open_csv(name_7z):
    plain = os.path.join(DATA_DIR, name_7z[:-3])
    if os.path.exists(plain):
        log(f"using extracted {os.path.basename(plain)}")
        return open(plain, "rb")
    return open_7z_buffered(os.path.join(DATA_DIR, name_7z))


def main():
    msno = np.load(os.path.join(CACHE_DIR, "msno_index.npz"))["msno"]
    n_users = len(msno)
    lookup = pd.Series(np.arange(n_users, dtype=np.int64),
                       index=[m.decode() for m in msno])

    state_path = os.path.join(SCRATCH, "state.json")
    state = {"rows_done": 0}
    agg = {m: np.zeros((n_users, N_MONTHS), dtype=np.float32)
           for m in METRICS}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        for m in METRICS:
            agg[m] = np.load(os.path.join(SCRATCH, f"agg_{m}.npy"))
        log(f"resuming after {state['rows_done']:,} rows")

    def checkpoint(rows):
        for m in METRICS:
            np.save(os.path.join(SCRATCH, f"agg_{m}.npy"), agg[m])
        with open(state_path + ".new", "w") as f:
            json.dump({"rows_done": rows}, f)
        os.replace(state_path + ".new", state_path)
        log(f"  checkpoint at {rows:,} rows")

    rows_seen = 0
    rows_done = state["rows_done"]
    last_ckpt = rows_done
    t0 = time.time()
    for fname in FILES:
        fh = open_csv(fname)
        for chunk in pd.read_csv(fh, chunksize=CHUNK, dtype=DTYPES):
            n = len(chunk)
            if rows_seen + n <= rows_done:      # fast skip on resume
                rows_seen += n
                continue
            if rows_seen < rows_done:
                chunk = chunk.iloc[rows_done - rows_seen:]
            rows_seen += n

            uid = chunk["msno"].map(lookup).to_numpy()
            ok = ~np.isnan(uid)
            u = uid[ok].astype(np.int64)
            d = chunk["date"].to_numpy()[ok]
            midx = (d // 10000) * 12 + (d // 100) % 100 - 1 - MONTH0
            ok2 = (midx >= 0) & (midx < N_MONTHS)
            u, midx = u[ok2], midx[ok2]

            secs = np.clip(chunk["total_secs"].to_numpy()[ok][ok2],
                           0, MAX_DAY_SECS)
            plays = (chunk["num_985"].to_numpy()[ok][ok2]
                     + chunk["num_100"].to_numpy()[ok][ok2])
            skips = chunk["num_25"].to_numpy()[ok][ok2]
            unq = chunk["num_unq"].to_numpy()[ok][ok2]

            key = u * N_MONTHS + midx
            order = np.argsort(key, kind="stable")
            key = key[order]
            bnd = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
            ukey = key[bnd]
            for name, vals in [("active_days", np.ones(len(key))),
                               ("total_secs", secs[order]),
                               ("plays", plays[order]),
                               ("skips", skips[order]),
                               ("uniq_songs", unq[order])]:
                sums = np.add.reduceat(vals.astype(np.float64), bnd)
                agg[name].reshape(-1)[ukey] += sums.astype(np.float32)

            rows_done = rows_seen
            rate = rows_done / max(time.time() - t0, 1)
            log(f"  {rows_done:,} rows ({rate:,.0f}/s)")
            if rows_done - last_ckpt >= CKPT_EVERY:
                checkpoint(rows_done)
                last_ckpt = rows_done

    log("saving engagement_monthly.npz (compressing ~1.3 GB, be patient)")
    np.savez_compressed(os.path.join(CACHE_DIR, "engagement_monthly.npz"),
                        month0_yyyymm=np.int32(201501), **agg)
    with open(state_path + ".done", "w") as f:
        f.write(str(rows_done))
    cov = float((agg["active_days"].sum(axis=1) > 0).mean())
    log(f"DONE: {rows_done:,} rows | users with any listening: {cov:.3f}")
    log("you can now delete data_cache/scratch_logs/")


if __name__ == "__main__":
    main()
