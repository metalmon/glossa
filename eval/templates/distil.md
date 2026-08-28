# kbx distil — densify the reasoning graph

You are given one section of a document, together with the reasoning nodes the graph already holds
for it — nodes a weaker pass produced the first time it read this corpus. Above this prompt you have
the ontology's schema-graph: the entity types this corpus's knowledge is typed with, and the
relations that connect them, each shown with the types it may run from and to. You read more
carefully than the pass that produced what you see here; your work is to notice what it missed and
add it, leaving what it already got right as it is.

Read the section's text against the list of nodes already grounded to it. Where the text states a
fact or a step that has no node for it yet, that is a gap worth filling — add the missing node,
grounded to this section the same way its existing nodes are. Where the existing nodes already say
what the text says, even in different words, that is not a gap; adding another node for the same
thing makes the graph harder to trust rather than richer. Read closely enough to tell a genuine gap
from a familiar fact restated.

Two kinds of reasoning can be missing, and a well-densified section usually has room to add either.
One is a grounded terminal the section itself states — a fact, a step, an outcome — grounded to this
section the same way its existing terminals are. The other is query-side reasoning: the situations,
tasks, or symptoms a real question would pass through on its way to a terminal, including a terminal
that is already in the graph from an earlier pass and has no route leading to it yet. A section can
be missing terminals, missing the query-side path to terminals it already grounds, or both — look for
each kind before deciding there is nothing here to add.

When the reasoning you add connects to something the graph already holds — a terminal this section
grounds, a node that came from a different section or document entirely — attach to it by the id it
already carries, rather than writing a fresh node that duplicates it under another label. An edge you
add can name an existing node at either end; connecting to what is already there is as valid a
contribution as adding something new. Query-side nodes, and the edges that chain them toward a
terminal, describe a situation a person brings with them rather than a fact this corpus states, so
their grounding lives at the terminal they lead to, not along the steps that lead there. Write the
nodes and edges you add through `graph_upsert`, the same way the graph's existing nodes were written.

Weigh a few well-grounded, genuinely new additions above a long list of near-restatements or
speculative links. When a section's text does not support any reasoning worth adding — everything it
states is already covered, or the text has nothing to do with the kind of reasoning this graph holds
— the honest outcome is to add nothing for it and move to the next section.
