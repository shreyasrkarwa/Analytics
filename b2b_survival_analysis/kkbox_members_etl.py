"""
kkbox_members_etl.py — members_v3 static covariates → data_cache/members.npz
==============================================================================
Joins KKBox member demographics onto the transaction-derived uid space.
Only users present in transactions are kept. Idempotent single pass.

Output arrays (aligned: row i = uid i; -1/0 = unknown):
  city           int16   (KKBox city code, 1..22)
  bd             int16   (age; raw field is noisy — values outside
                          [10, 80] are set to -1 as per common practice)
  gender         int8    (0 unknown, 1 male, 2 female)
  registered_via int16   (registration channel code)
  reg_date       int32   (registration_init_time, days since 1970-01-01)
  has_members    int8    (1 if user appeared in members_v3)

Run: python3 kkbox_members_etl.py
"""
import io
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "kkbox_churn_prediction_kaggle_data")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
sys.path.insert(0, BASE_DIR)
from sevenz_stream import open_7z_buffered  # noqa: E402

EPOCH_ORD = date(1970, 1, 1).toordinal()


def open_csv(name_7z):
    plain = os.path.join(DATA_DIR, name_7z[:-3])
    if os.path.exists(plain):
        return open(plain, "rb")
    return open_7z_buffered(os.path.join(DATA_DIR, name_7z))


def to_days(col):
    cache = {}
    out = np.empty(len(col), dtype=np.int32)
    vals = col.to_numpy()
    for i, v in enumerate(vals):
        v = int(v)
        d = cache.get(v)
        if d is None:
            try:
                d = date(v // 10000, (v // 100) % 100, v % 100).toordinal() \
                    - EPOCH_ORD
            except ValueError:
                d = -1
            cache[v] = d
        out[i] = d
    return out


def main():
    msno = np.load(os.path.join(CACHE_DIR, "msno_index.npz"))["msno"]
    n_users = len(msno)
    lookup = pd.Series(np.arange(n_users, dtype=np.int64),
                       index=[m.decode() for m in msno])

    city = np.full(n_users, -1, dtype=np.int16)
    bd = np.full(n_users, -1, dtype=np.int16)
    gender = np.zeros(n_users, dtype=np.int8)
    reg_via = np.full(n_users, -1, dtype=np.int16)
    reg_date = np.full(n_users, -1, dtype=np.int32)
    has = np.zeros(n_users, dtype=np.int8)

    fh = open_csv("members_v3.csv.7z")
    total = matched = 0
    for chunk in pd.read_csv(fh, chunksize=1_500_000,
                             dtype={"msno": object, "city": np.int16,
                                    "bd": np.int32, "gender": object,
                                    "registered_via": np.int16,
                                    "registration_init_time": np.int64}):
        uid = chunk["msno"].map(lookup)
        m = uid.notna().to_numpy()
        u = uid.to_numpy()[m].astype(np.int64)
        city[u] = chunk["city"].to_numpy()[m]
        b = chunk["bd"].to_numpy()[m]
        b = np.where((b >= 10) & (b <= 80), b, -1).astype(np.int16)
        bd[u] = b
        g = chunk["gender"].fillna("").to_numpy()[m]
        gender[u] = np.where(g == "male", 1, np.where(g == "female", 2, 0))
        reg_via[u] = chunk["registered_via"].to_numpy()[m]
        reg_date[u] = to_days(chunk["registration_init_time"])[m]
        has[u] = 1
        total += len(chunk)
        matched += int(m.sum())
        print(f"  {total:,} member rows | matched {matched:,}", flush=True)

    np.savez_compressed(os.path.join(CACHE_DIR, "members.npz"),
                        city=city, bd=bd, gender=gender,
                        registered_via=reg_via, reg_date=reg_date,
                        has_members=has)
    print(f"saved members.npz | coverage {has.mean():.3f} | "
          f"age known {np.mean(bd >= 0):.3f} | gender known "
          f"{np.mean(gender > 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
