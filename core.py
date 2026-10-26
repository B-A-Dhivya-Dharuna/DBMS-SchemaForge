import pandas as pd
from itertools import combinations

def profile(df):
    out = []
    for col in df.columns:
        out.append({
            "column": col,
            "null_rate": df[col].isna().mean(),
            "distinct": df[col].nunique(),
            "uniqueness": df[col].nunique() / len(df),
        })
    return pd.DataFrame(out)

def candidate_keys(df, max_size=2):
    keys = []
    for size in range(1, max_size + 1):
        for cols in combinations(df.columns, size):
            if any(set(k) <= set(cols) for k in keys):
                continue  # skip if a smaller key already covers it
            sub = df[list(cols)]
            if sub.isna().any().any():
                continue
            if not sub.duplicated().any():
                keys.append(cols)
    return keys

def fd_holds(df, lhs, rhs):
    return df.groupby(lhs)[rhs].nunique().max() == 1

def fd_error(df, lhs, rhs):
    sub = df.dropna(subset=list(lhs) + [rhs])
    majority = sub.groupby(list(lhs))[rhs].agg(lambda s: s.value_counts().iloc[0]).sum()
    return 1 - majority / len(sub)    # fraction of rows that violate the FD

def discover_fds_approx(df, max_lhs=2, threshold=0.02,
                        min_rhs_distinct=0, max_lhs_uniqueness=1.0):
    fds = []
    cols = list(df.columns)
    n = len(df)
    for size in range(1, max_lhs + 1):
        for lhs in combinations(cols, size):
            for rhs in cols:
                if rhs in lhs:
                    continue
                if any(set(f[0]) < set(lhs) and f[1] == rhs for f in fds):
                    continue
                sub = df.dropna(subset=list(lhs) + [rhs])
                if len(sub) == 0:
                    continue
                is_key = not df[list(lhs)].duplicated().any()
                # filter 1: low-distinct target, only for non-key left sides
                if not is_key and df[rhs].nunique() < min_rhs_distinct:
                    continue
                # filter 2: near-key left side (unique-ish but not a real key)
                lhs_uniqueness = df[list(lhs)].drop_duplicates().shape[0] / n
                if not is_key and lhs_uniqueness > max_lhs_uniqueness:
                    continue
                err = fd_error(df, lhs, rhs)
                if err <= threshold:
                    fds.append((lhs, rhs, err))
    return fds

def closure(attrs, fds):
    result = set(attrs)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if set(lhs) <= result and rhs not in result:
                result.add(rhs)
                changed = True
    return result

def score_fds(found, truth, columns):
    found_pairs = {(tuple(l), r) for l, r, *_ in found}
    true_pairs = {((a,), b) for a in columns for b in closure({a}, truth) if b != a}
    explained = found_pairs & true_pairs
    precision = len(explained) / len(found_pairs)
    recall = len(explained) / len(true_pairs)
    return precision, recall, sorted(found_pairs - true_pairs)