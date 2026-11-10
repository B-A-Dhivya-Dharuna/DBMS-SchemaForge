import sqlite3
import pandas as pd
from core import *

con = sqlite3.connect("Chinook_Sqlite.sqlite")

# check the real table names first (they differ between versions)
print(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con))

query = """
SELECT t.TrackId, t.Name AS TrackName, t.Composer, t.Milliseconds, t.UnitPrice,
       a.AlbumId, a.Title AS AlbumTitle,
       ar.ArtistId, ar.Name AS ArtistName,
       g.GenreId, g.Name AS GenreName
FROM Track t
JOIN Album a ON t.AlbumId = a.AlbumId
JOIN Artist ar ON a.ArtistId = ar.ArtistId
JOIN Genre g ON t.GenreId = g.GenreId
"""
flat = pd.read_sql_query(query, con)
print(flat.shape)
print(profile(flat))

truth = [
    (("TrackId",), c) for c in ["TrackName", "Composer", "Milliseconds", "UnitPrice",
                                "AlbumId", "AlbumTitle", "ArtistId", "ArtistName",
                                "GenreId", "GenreName"]
] + [
    (("AlbumId",), "AlbumTitle"), (("AlbumId",), "ArtistId"),
    (("ArtistId",), "ArtistName"), (("GenreId",), "GenreName"),
]

from pipeline import analyze, print_report
res = analyze(flat)
print_report(res)

p, r, _ = score_fds(res["fds"], truth, list(flat.columns))
print(f"all reported: precision={p:.2f} recall={r:.2f} ({len(res['fds'])} FDs)")

confirmed = [f for f in res["fds"]
             if not res["fd_ratings"][(f[0], f[1])].startswith("uncertain")]
p, r, _ = score_fds(confirmed, truth, list(flat.columns))
print(f"confirmed only: precision={p:.2f} recall={r:.2f} "
      f"({len(confirmed)} confirmed, {len(res['fds']) - len(confirmed)} flagged uncertain)")