"""Read-only: report the prefix distribution of path-bearing graph columns."""
import sqlite3, collections, sys

db = sys.argv[1] if len(sys.argv) > 1 else "kb-test/.glossa/graph.sqlite"
con = sqlite3.connect(db)
c = con.cursor()

KB = "kb-test\\"
DOT = ".\\"

def pref(v):
    if v is None:
        return "NULL"
    if v.startswith(KB):
        return "kb-test\\"
    if v.startswith(DOT):
        return ".\\"
    if v[:2] in ("E:", "C:", "D:"):
        return "ABS"
    if "\\" not in v and "/" not in v and ":" in v:
        return "nodeid (sym:/cau:/...)"
    return "other: " + repr(v[:24])

for tbl, cols in [("nodes", ["id", "source_path"]), ("edges", ["efrom", "eto", "source_path"])]:
    n = c.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
    print(f"--- {tbl} ({n} rows) ---")
    for col in cols:
        d = collections.Counter(pref(r[0]) for r in c.execute(f"SELECT {col} FROM {tbl}"))
        print(f"  {col}: {dict(d)}")

# sample a kb-test\ path and show its proposed .\ rewrite
row = c.execute("SELECT source_path FROM nodes WHERE source_path LIKE 'kb-test\\%' ESCAPE '!' LIMIT 1").fetchone()
con.close()
