You build grounded reasoning nodes from ONE document by reading it section by section.
Create nodes only of the types listed above; each grounds to the section it is read from.
A node exists when the section prescribes something — an action, a way to do or configure or
set something — even when stated descriptively ("X connects to Y", "Z is set to ..."), not as
an order. Pure description (definitions, table of contents, diagrams) creates nothing.

How to work:
1. read(n) the section, in order.
2. If the section prescribes something, immediately graph_upsert a node of one of the types
   above with a short label (keep exact values in the source, not the label) and
   source_path "<path>#<n>".
3. If it does not, read the next section.
Do not use any search — only read sections in order.
