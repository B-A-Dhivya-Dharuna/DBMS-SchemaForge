import sys
import pandas as pd
from core import (profile, candidate_keys, discover_fds_approx,
                  collapse_equivalents)

CONFIG = {
    "max_rows": 50000,           # sample larger tables down to this
    "max_key_size": 2,
    "fd_threshold": 0.0,         # raise (e.g. 0.02) for dirty data
    "min_rhs_distinct": 5,
    "max_lhs_uniqueness": 0.5,
    "min_rows_for_confidence": 100,
    "min_rows_per_lhs_value": 2,
}

def id_like(name):
    n = name.lower()
    return n.endswith("id") or n.endswith("key") or n.endswith("code")

def choose_primary_key(keys):
    if not keys:
        return None, 0
    smallest = min(len(k) for k in keys)
    same_size = [k for k in keys if len(k) == smallest]
    best = max(same_size, key=lambda k: sum(id_like(c) for c in k))
    return best, len(same_size)

def analyze(df, config=None):
    cfg = {**CONFIG, **(config or {})}
    notes = []

    n_total = len(df)
    if n_total > cfg["max_rows"]:
        df = df.sample(cfg["max_rows"], random_state=0)
        notes.append(f"Sampled {len(df)} of {n_total} rows to keep it fast.")

    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        df = df.drop(columns=constant)
        notes.append("Ignored constant or empty columns: " + ", ".join(constant))

    n = len(df)
    evidence = []
    prof = profile(df)

    # candidate keys
    keys = candidate_keys(df, cfg["max_key_size"])
    key_rating = ("supported" if n >= cfg["min_rows_for_confidence"]
                  else "uncertain (small sample)")
    for k in keys:
        evidence.append({
            "claim": f"{list(k)} is a candidate key",
            "numbers": f"{n} rows, 0 duplicates, 0 nulls",
            "rating": key_rating})

    # primary key
    pk, ties = choose_primary_key(keys)
    if pk:
        if ties > 1:
            rating = "uncertain"
            why = f"{ties} equally small candidates; picked by naming and column order"
        else:
            rating, why = key_rating, "smallest candidate key"
        evidence.append({
            "claim": f"{list(pk)} is the proposed primary key",
            "numbers": why, "rating": rating})
    else:
        notes.append("No candidate key found within the size limit.")

    # functional dependencies
    # functional dependencies
    found = discover_fds_approx(
        df, max_lhs=1, threshold=cfg["fd_threshold"],
        min_rhs_distinct=0,
        max_lhs_uniqueness=cfg["max_lhs_uniqueness"])
    fds, rep = collapse_equivalents(found)

    key_cols = {k[0] for k in keys if len(k) == 1}
    fd_ratings = {}
    for lhs, rhs, err in fds:
        sub = df.dropna(subset=[lhs[0], rhs])
        per = len(sub) / max(sub[lhs[0]].nunique(), 1)
        if lhs[0] in key_cols:
            rating = "trivial (determinant is a key)"
        elif df[rhs].nunique() < cfg["min_rhs_distinct"]:
            rating = "uncertain (target has few distinct values)"
        elif per < cfg["min_rows_per_lhs_value"]:
            rating = "uncertain (determinant values rarely repeat)"
        elif err == 0:
            rating = "supported"
        else:
            rating = "supported with violations"
        fd_ratings[(lhs, rhs)] = rating
        evidence.append({
            "claim": f"{lhs[0]} -> {rhs}",
            "numbers": f"error {err:.4f}, {per:.1f} rows per {lhs[0]} value",
            "rating": rating})
        
    # columns that carry the same information
    groups = {}
    for m, best in rep.items():
        groups.setdefault(best, []).append(m)
    for best, ms in groups.items():
        if len(ms) > 1:
            evidence.append({
                "claim": f"{sorted(ms)} carry the same information",
                "numbers": f"proposed identifier: {best}",
                "rating": "uncertain (identifier chosen by naming convention)"})

    return {"profile": prof, "keys": keys, "primary_key": pk,
            "fds": fds, "fd_ratings": fd_ratings,
            "evidence": evidence, "notes": notes}

def print_report(res):
    print("PROFILE")
    print(res["profile"].to_string(index=False))
    if res["notes"]:
        print("\nNOTES")
        for note in res["notes"]:
            print(" -", note)
    print("\nEVIDENCE")
    for e in res["evidence"]:
        print(f"[{e['rating']}] {e['claim']}  ({e['numbers']})")

if __name__ == "__main__":
    path = sys.argv[1]
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    print_report(analyze(df))