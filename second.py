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

for min_rhs, max_u in [(0, 1.0), (5, 1.0), (0, 0.5), (5, 0.5)]:
    found = discover_fds_approx(flat, max_lhs=1, threshold=0.0,
                                min_rhs_distinct=min_rhs, max_lhs_uniqueness=max_u)
    p, r, _ = score_fds(found, truth, list(flat.columns))
    print(f"min_rhs_distinct={min_rhs} max_lhs_uniqueness={max_u}: "
          f"precision={p:.2f} recall={r:.2f} ({len(found)} FDs)")