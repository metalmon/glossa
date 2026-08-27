# kbx reason — build the query side

You are given one node that is already grounded and true: the terminal, together with the full
source text it was drawn from. Above this prompt, the ontology's schema-graph lists the node types
this corpus is described with and the relations between them, each shown with the types it runs from
and to. Your work is the reverse of grounding that terminal: reconstruct the query side — the chain
of predecessor nodes a real person's question would pass through, in this corpus, to arrive at this
terminal — and connect that chain to the terminal.

Start at the terminal and walk the schema-graph backward. At each node, the relations that point
into it name the predecessor types that may precede it; take the one the source actually supports.
Let the terminal's own nature choose the shape of the chain and the type of its entry. When the
terminal is a procedure or a way to do something — a "how do I do X" — its entry is the task of the
person who wants to do or understand X. When the terminal resolves a fault or a failure someone ran
into, its entry is the symptom that person observes, reached through the cause behind it. A corpus
of procedures will lean toward the first shape, a corpus of fixes toward the second; follow what
each terminal actually is.

Read the source for the situations a person could be in when they come looking for this terminal. A
source that plainly states how or what something is done still describes such a situation — the
person who wanted that thing — so factual, declarative text is enough to build an entry from. When
the same terminal genuinely serves several different situations, the source usually shows them;
build a separate path for each, letting them diverge as far back as the source supports. When the
source text is about a different subject than the terminal — its own words do not concern this
terminal — the query side is empty, and leaving it so is the honest result.

Trace each path through the real intermediate steps the source supports, rather than jumping
straight from the terminal to an entry: a chain worth building is usually more than one hop, and
each step is what would have to be true, known, or asked one step earlier than the node you just
placed. Phrase an entry node's label the way the person would actually ask it, in their words rather
than the corpus's own terms, and give it generous aliases — the paraphrases, shorthand, and informal
wordings someone unfamiliar with the corpus would use — each a genuine restatement of that one
situation, narrow enough that it would not just as easily match an unrelated one.

The terminal, and any intermediate the corpus itself states, carry their source; an entry or
query-side node stands for the situation a person brings with them rather than a fact this corpus
asserts, so it stands on its own. Connect each predecessor to the next, and the chain to the
terminal, along the ontology's relations, respecting the types each relation joins. Write each node
and edge through `graph_upsert` as you find it. The work is complete when every path the source
supports has reached its entry node.
