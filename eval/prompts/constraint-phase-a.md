Call `get_task()` first — the returned `doc` is the **owner document** for notebook files (`note`, `ls`, `cat`). Search the **whole knowledge base** with `grep`/`search`/`read` (including cited GOSTs).

=== PHASE A: limit tables ===
1. note(doc, file="workbook.md", content=…) — research dossier: parameters, sources, tables, dependencies.
2. cat(workbook) then note(doc, file="….csp", content=…) — tab-separated rows; column headers = parameter names.
Then ls/cat/sed/del with paths from ls.

=== DONE ===
Call done when every parameter has a column in a .csp table with rows from the source.
