You are wiring the reasoning edges of a knowledge graph. A weak reader will later answer multi-hop questions by WALKING these edges from one grounded fact to the next, crossing between documents. You never see the questions — wire only from what the facts say.

You are shown ONE ordered pair of facts, A and B, that name the same real-world entity, each with the source text it is grounded in. Decide whether to add a directional link from A to B.

What a link is:
- `A -> B` means a reader reasoning about the shared entity would step from A to B to continue the chain — B supplies the next piece of the answer.
- Its real value is CROSS-DOCUMENT. Linking a fact in one document to a fact in another about the same entity is the bridge that lets the reader close a two-hop question — the piece they need lives in the other document. A pair where A and B sit in the SAME document is low value, since the reader can just read that document, so link those only when A clearly sets up B.

Judgment (this is why a model decides this, not a script):
- Sharing an entity is NOT enough to link. Link only when A genuinely LEADS to B's information — when, following the entity from A, B is the fact you step to next.
- Do NOT link parallel siblings: facts that merely sit beside each other under the same entity without one leading to the other — several unrelated attributes, or two facts whose real subjects are different things that only co-occur (e.g. two different cast members of one film). A script would clique those; you should not. That over-connection is the failure mode to avoid.
- Prefer NOT to link. Thin and correct beats dense. When A and B are just two parallel facts about the entity, the answer is NO.

Read both facts and their source text, then reply with exactly one line — `VERDICT: YES` or `VERDICT: NO` — followed by a short reason.
