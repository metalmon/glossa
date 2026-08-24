You are wiring the reasoning edges of a knowledge graph. A weak reader will later answer multi-hop questions by WALKING these edges from one grounded fact to the next, crossing between documents. You never see the questions — wire only from what the facts say.

You are shown ONE entity and EVERY fact that mentions it, each grounded in the source text it came from, possibly from several different documents. Decide which facts genuinely LEAD TO which, and return those as directed links.

What a link is:
- `A -> B` means a reader reasoning about this entity would step from A to B to continue the chain — B supplies the next piece of the answer that follows from, or is caused by, A.
- Its real value is CROSS-DOCUMENT. Linking a fact in one document to a fact in another about the same entity is the bridge that lets the reader close a two-hop question — the piece they need lives in the other document. A link between two facts in the SAME document is low value, since the reader can just read that document, so link those only when one clearly sets up the other.

Judgment (this is why a model decides this, not a script):
- Sharing the entity is NOT enough to link. Link only when one fact genuinely LEADS to another's information — when, following the entity from A, B is the fact you step to next.
- Do NOT link parallel siblings: facts that merely sit beside each other under the same entity without one leading to the other — several unrelated attributes, or facts whose real subjects are different things that only co-occur. A script would clique those; you should not. That over-connection is the failure mode to avoid.
- The set of links can be anything: a single chain, a branch, several disjoint links, one fact feeding several others, or none at all. There is no expected shape — read the facts and decide.
- A generic or very common entity that many facts merely co-mention, with no real step between any of them, should yield NO links at all.
- Prefer NOT to link. Thin and correct beats dense. When two facts are just parallel facts about the entity, leave them unlinked.

Read every fact and its source text, then reply with a JSON array of the genuine links you found, each exactly `{"from": "<id>", "to": "<id>"}` using the exact `id` values you were given. If there are no genuine links, reply with an empty array `[]`. Do not invent an id that wasn't shown to you, and never link a fact to itself.
