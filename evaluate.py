import sqlite3
import pandas as pd
from core import score_fds
from pipeline import analyze

DATASETS = {
    "chinook": {
        "db": "Chinook_Sqlite.sqlite",
        "query": "SELECT ... ",          # the flattening query
        "truth": [ ... ],                # the answer key
    },
    "northwind": {
        "db": "northwind.db",
        "query": "SELECT ... ",
        "truth": [ ... ],
    },
}

def run(name):
    d = DATASETS[name]
    flat = pd.read_sql_query(d["query"], sqlite3.connect(d["db"]))
    res = analyze(flat)
    confirmed = [f for f in res["fds"]
                 if not res["fd_ratings"][(f[0], f[1])].startswith("uncertain")]
    p, r, _ = score_fds(confirmed, d["truth"], list(flat.columns))
    print(f"{name}: precision={p:.2f} recall={r:.2f}")

if __name__ == "__main__":
    for name in DATASETS:
        run(name)