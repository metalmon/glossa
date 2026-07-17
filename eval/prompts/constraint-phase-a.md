Call `get_task()` first — the returned `doc` is the **owner document** for notebook files (`note`, `ls`, `read`). Search the **whole knowledge base** with `grep`/`search`/`read` (including cited GOSTs).

=== PHASE A: limit tables ===
1. note(doc, file="workbook.md", content=…) — research dossier: parameters, sources, tables, dependencies.
2. read(workbook) then note(doc, file="….csp", content=…) — tab-separated rows; column headers = parameter names.
Then ls/read/del with paths from ls.

=== FINISH ===
When every parameter has a column in a .csp table with rows from the source, reply with a short text summary and no tool calls.
