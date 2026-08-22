You build the flat reasoning layer of a knowledge graph from ONE document, for multi-hop question answering. Read what the document STATES and turn it into atomic, grounded facts.

Extract the factual statements the document makes, as `Fact` nodes. Rules:

- One `Fact` = one short, self-contained claim (a subject and one thing said about it). Do not pack several claims into a single node; split them.
- Be COMPLETE. Every claim the text states is a fact worth extracting — especially the small attribute claims (a date, a place, a number, a relation, a role, a title). Those attribute facts are exactly what a later question hops to, so dropping them quietly breaks multi-hop. When in doubt, extract it.
- Ground EVERY `Fact` with a `MENTIONS` edge to the exact source chunk it came from — `<path>#n`, using the chunk number `n` as a search/read result shows it. A fact with no grounding is not trustworthy.
- For each `Fact`, list the ENTITIES it mentions in its `aliases` — every named person, place, work, organization, or product the fact talks about, written precisely as the text names them (not paraphrased). These aliases are how facts in different documents get connected later, so name them carefully.
- Emit only the `Fact` node type. Do NOT try to classify facts into domain-specific kinds (problems, causes, fixes, and the like) — that richer, typed layer is produced separately by a stronger model. Your job is complete, grounded facts.
- Do NOT invent links between documents. Linking a fact in one document to a fact in another is a separate stage; here you only extract and ground what THIS document states.

Work through the document, reading the chunks you need, and emit facts as you find them.
