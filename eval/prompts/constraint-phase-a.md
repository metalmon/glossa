Call `get_task()` first — use the returned `doc` as the document for this run.

=== PHASE A: limit tables ===
1. note(doc, file="workbook.md", content=…) — research dossier: parameters, sources, tables, dependencies.
2. cat(workbook) then note(doc, file="….csp", content=…) — tab-separated rows; column headers = parameter names.
Use grep/read to research; use note for workbook.md and .csp tables.
Then ls/cat/sed/del with paths from ls.

=== DONE ===
Call done when every parameter has a column in a .csp table with rows from the source.
