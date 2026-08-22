You are wiring the reasoning graph for MULTI-HOP question answering. Multi-hop questions are answered by hopping along a shared entity from one fact to the next, often crossing between documents.

You are shown ONE ordered pair of facts, A and B, that mention the same entity, each with the source text it is grounded in. Decide whether to add a link from A to B.

Answer `VERDICT: YES` when a reader who has arrived at A, following that shared entity, would move to B to learn a FURTHER fact about the same entity A is about or introduces — another of its roles, works, relations, dates, or places. Sibling attributes of ONE entity DO count and are exactly the hops multi-hop needs: two roles of the same person, or a person's work and their birthplace, should be linked. B living in a different document than A is a plus, not a barrier. If B restates A's claim but pins a new specific A lacks — a date, place, number, or name — that still counts as YES.

Answer `VERDICT: NO` only when following A's entity does NOT actually reach B — that is, B's real subject is a DIFFERENT person or thing that merely co-occurs with A in the same background context. For example, two different cast members of the same film: knowing one performer's role does not lead you to a different performer's role — the new subject is reached only because they happen to share the film, not by following anyone. Also answer NO when B is simply unrelated to A.

Read both facts and their source text, then reply with exactly one line — `VERDICT: YES` or `VERDICT: NO` — followed by a short reason.
