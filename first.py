import random
import pandas as pd
from itertools import combinations

random.seed(1)
students = [(f"S{100+i}", f"Student{i}") for i in range(40)]
courses = [("C01","DBMS","Ravi"), ("C02","ML","Meera"), ("C03","OS","Arun"),
           ("C04","Networks","Divya"), ("C05","AI","Meera")]

rows, seen = [], set()
while len(rows) < 150:
    s, c = random.choice(students), random.choice(courses)
    if (s[0], c[0]) in seen:
        continue
    seen.add((s[0], c[0]))
    rows.append({"StudentID": s[0], "StudentName": s[1],
                 "CourseID": c[0], "CourseName": c[1], "Faculty": c[2]})

df = pd.DataFrame(rows)
df.to_csv("students_courses.csv", index=False)
df_dirty = df.copy()
df_dirty.loc[5, "CourseName"] = "DBMs"      # a typo
df_dirty.loc[10, "Faculty"] = None          # a missing value
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

print(profile(df))

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

print(candidate_keys(df))

def fd_holds(df, lhs, rhs):
    return df.groupby(lhs)[rhs].nunique().max() == 1

def fd_error(df, lhs, rhs):
    sub = df.dropna(subset=list(lhs) + [rhs])
    majority = sub.groupby(list(lhs))[rhs].agg(lambda s: s.value_counts().iloc[0]).sum()
    return 1 - majority / len(sub)    # fraction of rows that violate the FD

def discover_fds_approx(df, max_lhs=2, threshold=0.02):
    fds = []
    cols = list(df.columns)
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
                err = fd_error(df, lhs, rhs)
                if err <= threshold:
                    fds.append((lhs, rhs, err))
    return fds

def discover_fds(df, max_lhs=2):
    fds = []
    cols = list(df.columns)
    for size in range(1, max_lhs + 1):
        for lhs in combinations(cols, size):
            for rhs in cols:
                if rhs in lhs:
                    continue
                # skip if a smaller LHS already determines rhs
                if any(set(f[0]) < set(lhs) and f[1] == rhs for f in fds):
                    continue
                if fd_holds(df, list(lhs), rhs):
                    fds.append((lhs, rhs))
    return fds

for lhs, rhs in discover_fds(df):
    print(lhs, "->", rhs)
def pick_key(keys):
    for k in keys:
        if all(c.endswith("ID") for c in k):
            return k
    return keys[0]

def decompose(fds, key):
    groups = {}
    for lhs, rhs in fds:
        groups.setdefault(lhs, set()).add(rhs)
    tables = {frozenset(lhs) | frozenset(rhs_set) for lhs, rhs_set in groups.items()}
    tables.add(frozenset(key))   # the link table
    return tables

def build_tables(fds, link_key):
    groups = {}
    for lhs, rhs in fds:
        groups.setdefault(lhs, set()).add(rhs)
    chosen = {}
    for lhs, rhs_set in groups.items():
        cols = frozenset(lhs) | rhs_set
        if cols not in chosen or all(c.endswith("ID") for c in lhs):
            chosen[cols] = lhs
    tables = [{"cols": sorted(c), "pk": list(pk)} for c, pk in chosen.items()]
    tables.append({"cols": sorted(link_key), "pk": list(link_key)})
    for t in tables:
        t["name"] = "_".join(c.removesuffix("ID") for c in t["pk"])
    return tables

def find_fks(tables):
    fks = []
    for child in tables:
        for parent in tables:
            if parent is child or len(parent["pk"]) != 1:
                continue
            col = parent["pk"][0]
            if col in child["cols"] and child["pk"] != [col]:
                fks.append((child["name"], col, parent["name"]))
    return fks

def to_sql(tables, fks):
    out = []
    for t in tables:
        lines = [f"  {c} VARCHAR(50)" for c in t["cols"]]
        lines.append(f"  PRIMARY KEY ({', '.join(t['pk'])})")
        for child, col, parent in fks:
            if child == t["name"]:
                lines.append(f"  FOREIGN KEY ({col}) REFERENCES {parent}({col})")
        out.append(f"CREATE TABLE {t['name']} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(out)

def to_mermaid(fks):
    lines = ["erDiagram"]
    for child, col, parent in fks:
        lines.append(f"  {parent} ||--o{{ {child} : has")
    return "\n".join(lines)

# bottom of the file
fds = discover_fds(df)
key = pick_key(candidate_keys(df))
tables = build_tables(fds, key)
fks = find_fks(tables)
print(to_sql(tables, fks))
print()
print(to_mermaid(fks))

print("\n--- DIRTY DATA ---")
print("Candidate keys:", candidate_keys(df_dirty))
print("Exact FDs:")
for lhs, rhs in discover_fds(df_dirty):
    print("  ", lhs, "->", rhs)

print("\nApproximate FD errors:")
print("CourseID -> CourseName:", fd_error(df_dirty, ["CourseID"], "CourseName"))
print("CourseID -> Faculty:   ", fd_error(df_dirty, ["CourseID"], "Faculty"))
print("StudentID -> StudentName:", fd_error(df_dirty, ["StudentID"], "StudentName"))
print("\n--- APPROXIMATE FDs on dirty data ---")
for lhs, rhs, err in discover_fds_approx(df_dirty):
    print("  ", lhs, "->", rhs, f"(error {err:.4f})")