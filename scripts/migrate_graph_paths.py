"""One-off: rewrite a stale path PREFIX across the graph to match the index's canonical form.

Doc keys in the index are corpus-root-relative ("bare"). When the reasoning graph was enriched
against a differently-addressed index, its node provenance and MENTIONS edges carry a stale prefix
(`kb-test\\` or `.\\`) and point at sections that no longer exist as nodes — so `read` of an anchor
misses. This rewrites the prefix so they line up; stale structural twins collapse via the PK.

Usage:  python scripts/migrate_graph_paths.py <db> --from "<OLD>" --to "<NEW>" [--apply]
  e.g.  ... --from ".\\" --to ""          (strip the `.\\` prefix → bare)
        ... --from "kb-test\\" --to ".\\"  (the original kb-test → .\\ fix)
Default is DRY-RUN; `--apply` backs up to <db>.bak (sqlite backup API) then migrates.
"""
import sqlite3, sys

args = sys.argv[1:]
apply = "--apply" in args
args = [a for a in args if a != "--apply"]
db = args[0] if args else "kb-test/.glossa/graph.sqlite"
OLD = args[args.index("--from") + 1] if "--from" in args else "kb-test\\"
NEW = args[args.index("--to") + 1] if "--to" in args else ".\\"
L = len(OLD)

def fix(v):
    return NEW + v[L:] if v is not None and v.startswith(OLD) else v

con = sqlite3.connect(db)
c = con.cursor()
nodes_sp = c.execute("SELECT count(*) FROM nodes WHERE substr(source_path,1,?)=?", (L, OLD)).fetchone()[0]
edge_rows = c.execute(
    "SELECT rowid, efrom, edge_type, eto, source_path FROM edges "
    "WHERE substr(efrom,1,?)=? OR substr(eto,1,?)=? OR substr(source_path,1,?)=?",
    (L, OLD, L, OLD, L, OLD),
).fetchall()
print(f"db={db}  '{OLD}' -> '{NEW}'")
print(f"nodes.source_path to fix: {nodes_sp};  edges touching prefix: {len(edge_rows)}")
if not apply:
    for r in edge_rows[:3]:
        print(f"  {r[2]}: {r[1]} -> {r[3]}  ==>  {fix(r[1])} -> {fix(r[3])}")
    print("DRY-RUN — pass --apply to write (a .bak backup is made first).")
    con.close()
    sys.exit(0)

bak = db + ".bak"
with sqlite3.connect(bak) as b:
    con.backup(b)
print(f"backup -> {bak}")
con.execute("BEGIN")
c.execute("UPDATE nodes SET source_path = ? || substr(source_path, ?) WHERE substr(source_path,1,?)=?", (NEW, L + 1, L, OLD))
fixed_nodes = c.rowcount
updated = deleted = 0
for rowid, efrom, etype, eto, sp in edge_rows:
    try:
        c.execute("UPDATE edges SET efrom=?, eto=?, source_path=? WHERE rowid=?", (fix(efrom), fix(eto), fix(sp), rowid))
        updated += 1
    except sqlite3.IntegrityError:
        c.execute("DELETE FROM edges WHERE rowid=?", (rowid,))
        deleted += 1
con.commit()
print(f"nodes.source_path fixed: {fixed_nodes};  edges rewritten: {updated};  stale dups deleted: {deleted}")
left = c.execute("SELECT (SELECT count(*) FROM nodes WHERE substr(source_path,1,?)=?), (SELECT count(*) FROM edges WHERE substr(efrom,1,?)=? OR substr(eto,1,?)=? OR substr(source_path,1,?)=?)", (L, OLD, L, OLD, L, OLD, L, OLD)).fetchone()
print(f"remaining '{OLD}'  -> nodes:{left[0]} edges:{left[1]}")
con.close()
