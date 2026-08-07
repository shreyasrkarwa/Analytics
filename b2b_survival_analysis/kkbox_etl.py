"""
kkbox_etl.py — KKBox transactions → survival-format ETL (resumable)
====================================================================
Builds the real-data backbone for the renewal-cliff paper from the raw
Kaggle KKBox archives, using the WSDM churn rule (no re-transaction
within 30 days of effective membership expiration = churn).

Outputs (in data_cache/, .gitignored):
  msno_index.npz          uid → msno mapping (S48 strings)
  spells.npz              one row per subscription spell:
                          uid, spell_id, start, end, event, n_trans,
                          n_cancels, left_edge, last_auto_renew,
                          last_plan_days
  transactions_sorted.npz deduped transactions sorted by (uid, date):
                          uid, td, ed, eff, cancel, auto_renew, plan_days
  panel_part*.npz         30-day person-period panel: uid, spell_id,
                          period, t0, t1, days_to_boundary, auto_renew,
                          plan_days, n_prior_trans, event
  (dates are int32 days since 1970-01-01)

The script is a checkpointed state machine: every invocation performs up
to --budget seconds of work, saves state, and exits. Re-invoke until it
prints ALL DONE. On a machine without execution time limits, run with a
large budget for a single-shot run:

  python3 kkbox_etl.py --budget 100000

Intermediates live in a scratch dir (--tmp), final artifacts in
data_cache/. Archives are streamed via libarchive (sevenz_stream.py);
if plain .csv files exist next to the .7z archives they are used
directly.

Method notes for the paper:
  * Effective expiration = running max of membership_expire_date within
    a user, reset at is_cancel transactions (mirrors Kaggle's official
    WSDMChurnLabeller.scala).
  * A gap > 30 days between effective expiration and the next
    transaction closes a spell with event=1 at the expiration date.
  * The last spell is censored (event=0) when expiration+30d exceeds
    the data cutoff (2017-03-31).
  * Spells starting in Jan-2015 (first month of data) carry left_edge=1
    (left-truncation flag) for delayed-entry handling.
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "kkbox_churn_prediction_kaggle_data")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

sys.path.insert(0, BASE_DIR)
from sevenz_stream import stream_7z_bytes  # noqa: E402

EPOCH_ORD = date(1970, 1, 1).toordinal()
GAP = 30                    # WSDM churn gap (days)
PERIOD = 30                 # person-period length (days)
CUTOFF = date(2017, 3, 31).toordinal() - EPOCH_ORD
DATA_EDGE = date(2015, 1, 31).toordinal() - EPOCH_ORD

COLS = ["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
        "actual_amount_paid", "is_auto_renew", "transaction_date",
        "membership_expire_date", "is_cancel"]
DTYPES = {"msno": object, "payment_method_id": np.int16,
          "payment_plan_days": np.int16, "plan_list_price": np.int32,
          "actual_amount_paid": np.int32, "is_auto_renew": np.int8,
          "transaction_date": np.int32, "membership_expire_date": np.int32,
          "is_cancel": np.int8}

PARSE_WINDOW = 220 * 1024 * 1024   # bytes of csv parsed per unit
PANEL_SLICE = 6_000_000            # max panel rows generated per unit


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, tmp, budget):
        self.tmp = tmp
        self.t0 = time.time()
        self.budget = budget
        os.makedirs(tmp, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.state_path = os.path.join(tmp, "etl_state.json")
        self.state = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)

    def save(self):
        with open(self.state_path + ".new", "w") as f:
            json.dump(self.state, f)
        os.replace(self.state_path + ".new", self.state_path)

    def out_of_time(self):
        return time.time() - self.t0 > self.budget

    def p(self, name):
        return os.path.join(self.tmp, name)


# ---------------------------------------------------------------------------
# phase 1: extract archives to plain csv in scratch
# ---------------------------------------------------------------------------

EXTRACT_FILES = ["transactions.csv.7z", "transactions_v2.csv.7z",
                 "train.csv.7z"]


def phase_extract(cx):
    st = cx.state.setdefault("extract", {})
    for fname in EXTRACT_FILES:
        plain_repo = os.path.join(DATA_DIR, fname[:-3])
        target = cx.p(os.path.basename(fname)[:-3])
        if st.get(fname) == "done":
            continue
        if os.path.exists(plain_repo):        # user pre-extracted it
            st[fname] = "done"
            cx.state["csv_" + fname] = plain_repo
            continue
        done_bytes = int(st.get(fname, 0))
        log(f"extracting {fname} (resume at {done_bytes:,}) ...")
        if not os.path.exists(target):
            open(target, "wb").close()
        with open(target, "r+b") as out:
            out.truncate(done_bytes)      # drop bytes past last checkpoint
            out.seek(done_bytes)
            written = 0
            last_ckpt = done_bytes
            for chunk in stream_7z_bytes(os.path.join(DATA_DIR, fname)):
                n = len(chunk)
                if written + n <= done_bytes:
                    written += n
                    continue
                if written < done_bytes:
                    chunk = chunk[done_bytes - written:]
                    written = done_bytes
                out.write(chunk)
                written += len(chunk)
                if written - last_ckpt > 64 * 1024 * 1024:
                    out.flush()
                    os.fsync(out.fileno())
                    st[fname] = written   # checkpoint: survives a hard kill
                    cx.save()
                    last_ckpt = written
                if cx.out_of_time():
                    st[fname] = written
                    cx.save()
                    log(f"  paused at {written:,} bytes")
                    return False
        st[fname] = "done"
        cx.state["csv_" + fname] = target
        cx.save()
        log(f"  extracted {written:,} bytes → {os.path.basename(target)}")
    return True


# ---------------------------------------------------------------------------
# phase 2: parse transactions csvs → column parts + msno dict
# ---------------------------------------------------------------------------

def _load_msno(cx):
    path = cx.p("msno_keys.npy")
    if os.path.exists(path):
        keys = np.load(path)
        return {k: i for i, k in enumerate(keys.tolist())}
    return {}


def _save_msno(cx, d):
    keys = np.array(sorted(d, key=d.get), dtype="S48")
    np.save(cx.p("msno_keys.npy"), keys)


_dcache = {}


def _to_days(col):
    uniq = pd.unique(col)
    m = {}
    for v in uniq:
        v = int(v)
        d = _dcache.get(v)
        if d is None:
            try:
                d = date(v // 10000, (v // 100) % 100, v % 100).toordinal() \
                    - EPOCH_ORD
            except ValueError:
                d = -1
            _dcache[v] = d
        m[v] = d
    return col.map(m).astype(np.int32)


def phase_parse(cx):
    st = cx.state.setdefault("parse", {"file_i": 0, "offset": 0, "part": 0,
                                       "rows": 0})
    files = [cx.state["csv_transactions.csv.7z"],
             cx.state["csv_transactions_v2.csv.7z"]]
    msno = _load_msno(cx)
    while st["file_i"] < len(files):
        path = files[st["file_i"]]
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            while st["offset"] < size:
                if cx.out_of_time():
                    _save_msno(cx, msno)
                    cx.save()
                    log(f"  paused: file {st['file_i']} offset "
                        f"{st['offset']:,}/{size:,} rows {st['rows']:,}")
                    return False
                f.seek(st["offset"])
                if st["offset"] == 0:
                    f.readline()                      # skip header
                buf = f.read(PARSE_WINDOW)
                if not buf:
                    break
                data = buf + (f.readline() or b"")   # align to line end
                new_off = f.tell()
                df = pd.read_csv(io.BytesIO(data), names=COLS, dtype=DTYPES)
                codes, uniqs = pd.factorize(df["msno"].to_numpy())
                lut = np.empty(len(uniqs), dtype=np.int32)
                for j, m in enumerate(uniqs):
                    mb = m.encode() if isinstance(m, str) else m
                    u = msno.get(mb)
                    if u is None:
                        u = len(msno)
                        msno[mb] = u
                    lut[j] = u
                part = {
                    "uid": lut[codes],
                    "pm": df["payment_method_id"].to_numpy(),
                    "plan_days": df["payment_plan_days"].to_numpy(),
                    "price": df["plan_list_price"].to_numpy(),
                    "paid": df["actual_amount_paid"].to_numpy(),
                    "auto_renew": df["is_auto_renew"].to_numpy(),
                    "td": _to_days(df["transaction_date"]).to_numpy(),
                    "ed": _to_days(df["membership_expire_date"]).to_numpy(),
                    "cancel": df["is_cancel"].to_numpy(),
                }
                np.savez(cx.p(f"parse_part{st['part']}.npz"), **part)
                st["rows"] += len(df)
                st["part"] += 1
                st["offset"] = new_off
                log(f"  parsed part {st['part']} | total rows {st['rows']:,} "
                    f"| users {len(msno):,}")
        st["file_i"] += 1
        st["offset"] = 0
    _save_msno(cx, msno)
    cx.state["n_rows"] = st["rows"]
    cx.save()
    log(f"parse complete: {st['rows']:,} rows, {len(msno):,} users")
    return True


# ---------------------------------------------------------------------------
# phase 3: concat + sort + dedupe + effective-expiration
# ---------------------------------------------------------------------------

def phase_sort(cx):
    if cx.state.get("sorted"):
        return True
    n_parts = cx.state["parse"]["part"]
    keys = ["uid", "td", "ed", "cancel", "auto_renew", "plan_days"]
    arrs = {k: [] for k in keys}
    for i in range(n_parts):
        z = np.load(cx.p(f"parse_part{i}.npz"))
        for k in keys:
            arrs[k].append(z[k])
    a = {k: np.concatenate(v) for k, v in arrs.items()}
    del arrs
    log(f"sorting {len(a['uid']):,} rows ...")
    order = np.lexsort((a["cancel"], a["ed"], a["td"], a["uid"]))
    for k in keys:
        a[k] = a[k][order]
    del order

    # exact-duplicate removal (v1/v2 overlap safety)
    dup = np.ones(len(a["uid"]), dtype=bool)
    same = np.ones(len(a["uid"]) - 1, dtype=bool)
    for k in keys:
        same &= a[k][1:] == a[k][:-1]
    dup[1:] = ~same
    n_dup = int((~dup).sum())
    for k in keys:
        a[k] = a[k][dup]
    log(f"removed {n_dup:,} exact duplicates → {len(a['uid']):,} rows")

    # effective expiration: running max of e'=max(ed,td), reset at cancels
    e = np.maximum(a["ed"], a["td"])
    user_start = np.empty(len(e), dtype=bool)
    user_start[0] = True
    user_start[1:] = a["uid"][1:] != a["uid"][:-1]
    seg = np.cumsum(user_start | (a["cancel"] == 1))
    eff = pd.Series(e).groupby(seg).cummax().to_numpy().astype(np.int32)

    for k in keys:
        np.save(cx.p(f"s_{k}.npy"), a[k])
    np.save(cx.p("s_eff.npy"), eff)
    np.save(cx.p("s_user_start.npy"), user_start)
    cx.state["sorted"] = True
    cx.save()
    log("sorted arrays + effective expiration saved")
    return True


# ---------------------------------------------------------------------------
# phase 4: spell derivation (vectorised)
# ---------------------------------------------------------------------------

def phase_spells(cx):
    if cx.state.get("spells"):
        return True
    uid = np.load(cx.p("s_uid.npy"))
    td = np.load(cx.p("s_td.npy"))
    eff = np.load(cx.p("s_eff.npy"))
    cancel = np.load(cx.p("s_cancel.npy"))
    auto_renew = np.load(cx.p("s_auto_renew.npy"))
    plan_days = np.load(cx.p("s_plan_days.npy"))
    user_start = np.load(cx.p("s_user_start.npy"))

    prev_eff = np.empty_like(eff)
    prev_eff[1:] = eff[:-1]
    prev_eff[0] = 0
    new_spell = user_start | (td > prev_eff + GAP)
    sb = np.flatnonzero(new_spell)                     # spell first row
    se = np.append(sb[1:], len(uid))                   # spell end row (excl)
    last = se - 1

    start = td[sb]
    eff_last = eff[last]
    event = (eff_last + GAP <= CUTOFF).astype(np.int8)
    end = np.where(event == 1, eff_last, np.minimum(eff_last, CUTOFF))
    end = np.maximum(end, start + 1).astype(np.int32)

    ccum = np.concatenate(([0], np.cumsum(cancel, dtype=np.int64)))
    n_cancels = (ccum[se] - ccum[sb]).astype(np.int32)

    spells = dict(
        uid=uid[sb].astype(np.int32),
        spell_id=np.arange(len(sb), dtype=np.int32),
        start=start.astype(np.int32), end=end,
        event=event.astype(np.int32),
        n_trans=(se - sb).astype(np.int32),
        n_cancels=n_cancels,
        left_edge=(start <= DATA_EDGE).astype(np.int32),
        last_auto_renew=auto_renew[last].astype(np.int32),
        last_plan_days=plan_days[last].astype(np.int32),
    )
    np.savez_compressed(os.path.join(CACHE_DIR, "spells.npz"), **spells)
    np.save(cx.p("sp_sb.npy"), sb.astype(np.int64))
    n_per = ((end - start + PERIOD - 1) // PERIOD).astype(np.int64)
    np.save(cx.p("sp_nper.npy"), n_per)
    cx.state["spells"] = True
    cx.state["n_spells"] = int(len(sb))
    cx.state["n_panel"] = int(n_per.sum())
    cx.save()
    log(f"spells: {len(sb):,} | events {spells['event'].mean():.3f} | "
        f"panel rows to generate {int(n_per.sum()):,}")
    return True


# ---------------------------------------------------------------------------
# phase 5: person-period panel (sliced)
# ---------------------------------------------------------------------------

def phase_panel(cx):
    st = cx.state.setdefault("panel", {"next": 0, "part": 0})
    z = np.load(os.path.join(CACHE_DIR, "spells.npz"))
    s_uid, s_start = z["uid"], z["start"]
    s_end, s_event = z["end"], z["event"]
    sb = np.load(cx.p("sp_sb.npy"))
    n_per = np.load(cx.p("sp_nper.npy"))
    n_spells = len(s_uid)

    uid = np.load(cx.p("s_uid.npy"))
    td = np.load(cx.p("s_td.npy"))
    eff = np.load(cx.p("s_eff.npy"))
    auto_renew = np.load(cx.p("s_auto_renew.npy"))
    plan_days = np.load(cx.p("s_plan_days.npy"))
    key = uid.astype(np.int64) * (1 << 22) + td      # td < 2^22 days

    while st["next"] < n_spells:
        if cx.out_of_time():
            cx.save()
            log(f"  paused at spell {st['next']:,}/{n_spells:,}")
            return False
        lo = st["next"]
        cum = np.cumsum(n_per[lo:])
        hi = lo + int(np.searchsorted(cum, PANEL_SLICE)) + 1
        hi = min(hi, n_spells)
        idx_sp = np.arange(lo, hi)
        reps = n_per[lo:hi]
        total = int(reps.sum())

        sp_rep = np.repeat(idx_sp, reps).astype(np.int64)
        offs = np.concatenate(([0], np.cumsum(reps)))[:-1]
        p = (np.arange(total) - np.repeat(offs, reps)).astype(np.int32)
        t0 = s_start[sp_rep] + p * PERIOD
        t1 = np.minimum(t0 + PERIOD, s_end[sp_rep]).astype(np.int32)
        ev = ((s_event[sp_rep] == 1) & (t1 == s_end[sp_rep])).astype(np.int8)

        pkey = s_uid[sp_rep].astype(np.int64) * (1 << 22) + t0
        tix = np.searchsorted(key, pkey, side="right") - 1
        dtb = np.clip(eff[tix] - t0, -32000, 32000).astype(np.int32)
        n_prior = (tix - sb[sp_rep]).astype(np.int32)

        np.savez_compressed(
            os.path.join(CACHE_DIR, f"panel_part{st['part']}.npz"),
            uid=s_uid[sp_rep], spell_id=sp_rep.astype(np.int32),
            period=p, t0=t0.astype(np.int32), t1=t1,
            days_to_boundary=dtb, auto_renew=auto_renew[tix].astype(np.int8),
            plan_days=plan_days[tix].astype(np.int16),
            n_prior_trans=n_prior, event=ev)
        log(f"  panel_part{st['part']}: spells {lo:,}–{hi:,} "
            f"({total:,} rows)")
        st["part"] += 1
        st["next"] = hi
        cx.save()
    cx.state["panel_done"] = True
    cx.save()
    log("panel complete")
    return True


# ---------------------------------------------------------------------------
# phase 6: persist reusable artifacts to data_cache
# ---------------------------------------------------------------------------

def phase_archive(cx):
    done = cx.state.setdefault("archived", [])
    jobs = [("msno_index.npz",
             lambda: {"msno": np.load(cx.p("msno_keys.npy"))}),
            ("transactions_sorted.npz",
             lambda: {k: np.load(cx.p(f"s_{k}.npy")) for k in
                      ["uid", "td", "ed", "eff", "cancel", "auto_renew",
                       "plan_days"]})]
    for name, get in jobs:
        if name in done:
            continue
        if cx.out_of_time():
            cx.save()
            return False
        np.savez_compressed(os.path.join(CACHE_DIR, name), **get())
        done.append(name)
        cx.save()
        log(f"archived {name}")
    return True


# ---------------------------------------------------------------------------
# phase 7: validate vs official Kaggle labels (train.csv → Feb-2017)
# ---------------------------------------------------------------------------

def phase_validate(cx):
    if cx.state.get("validated"):
        return True
    uid = np.load(cx.p("s_uid.npy"))
    td = np.load(cx.p("s_td.npy"))
    ed = np.load(cx.p("s_ed.npy"))
    cancel = np.load(cx.p("s_cancel.npy"))

    jan1 = date(2017, 1, 1).toordinal() - EPOCH_ORD
    jan31 = date(2017, 1, 31).toordinal() - EPOCH_ORD
    feb1, feb28 = jan31 + 1, jan31 + 28

    # last_expire per user from Jan-2017 transactions (official recipe)
    in_jan = (td >= jan1) & (td <= jan31)
    ju, jtd, jed, jcn = uid[in_jan], td[in_jan], ed[in_jan], cancel[in_jan]
    e = np.maximum(jed, jtd)
    us = np.empty(len(ju), dtype=bool)
    if len(ju):
        us[0] = True
        us[1:] = ju[1:] != ju[:-1]
    seg = np.cumsum(us | (jcn == 1))
    jeff = pd.Series(e).groupby(seg).cummax().to_numpy()
    last_of_user = np.append(us[1:], True)
    u_ids = ju[last_of_user]
    u_expire = jeff[last_of_user]
    cand = (u_expire >= feb1) & (u_expire <= feb28)
    exp_map = dict(zip(u_ids[cand].tolist(), u_expire[cand].tolist()))

    # renewal: any future non-cancel txn before expire+30
    fut = (td > jan31) & (cancel == 0)
    fu, ftd = uid[fut], td[fut]
    renewed = set()
    exp_arr = np.array([exp_map.get(int(x), -10**9) for x in fu],
                       dtype=np.int64)
    hit = ftd < exp_arr + GAP
    renewed = set(fu[hit].tolist())

    msno = np.load(cx.p("msno_keys.npy"))
    msno_to_uid = {m: i for i, m in enumerate(msno.tolist())}
    labels = pd.read_csv(cx.state["csv_train.csv.7z"])

    n_eval = n_agree = n_nocand = n_missing = fp = fn = 0
    for m, y in zip(labels["msno"].to_numpy(),
                    labels["is_churn"].to_numpy()):
        u = msno_to_uid.get(m.encode())
        if u is None:
            n_missing += 1
            continue
        if u not in exp_map:
            n_nocand += 1
            continue
        ours = 0 if u in renewed else 1
        n_eval += 1
        if ours == y:
            n_agree += 1
        elif ours == 1:
            fp += 1
        else:
            fn += 1

    res = {"labelled": int(len(labels)), "evaluable": n_eval,
           "agreement": round(n_agree / max(n_eval, 1), 4),
           "we_churn_they_stay": fp, "we_stay_they_churn": fn,
           "no_jan_txn_or_no_feb_expiry": n_nocand,
           "msno_not_in_transactions": n_missing}
    cx.state["validated"] = True
    cx.state["validation"] = res
    cx.save()
    log(f"VALIDATION vs official train.csv: {json.dumps(res)}")
    return True


# ---------------------------------------------------------------------------

PHASES = [("extract", phase_extract), ("parse", phase_parse),
          ("sort", phase_sort), ("spells", phase_spells),
          ("panel", phase_panel), ("archive", phase_archive),
          ("validate", phase_validate)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=26,
                    help="seconds of work per invocation")
    ap.add_argument("--tmp", default=None, help="scratch dir")
    args = ap.parse_args()
    tmp = args.tmp or os.environ.get("KKBOX_TMP") or \
        os.path.join(BASE_DIR, "data_cache", "scratch")
    cx = Ctx(tmp, args.budget)
    for name, fn in PHASES:
        if cx.out_of_time():
            break
        log(f"===== phase: {name} =====")
        if not fn(cx):
            log("BUDGET REACHED — re-invoke to continue")
            sys.exit(0)
    if cx.state.get("validated"):
        log("ALL DONE")
    else:
        log("BUDGET REACHED — re-invoke to continue")
