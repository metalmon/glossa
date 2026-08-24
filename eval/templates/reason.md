# kbx reason

You are building a TYPED REASONING CHAIN over a corpus, anchored to one SOLVED case: a question
together with its verified answer. Above this prompt you have been given the ontology's
schema-graph — the entity types this corpus's knowledge is typed with, and the relations that
connect them, each relation shown with the types it may run from and to. That schema-graph is the
only vocabulary you may use: every node you create must fit one of its types, and every edge must
match a relation's declared from/to types. You also have corpus search and read tools, and a
`graph_upsert` tool to emit the nodes and edges you find.

Work BACKWARD from the answer, not forward from the question. The answer is already known to be
correct, so it is your most reliable starting point: find where the corpus states it, in exactly
those or equivalent words, and quote the passage. Type that node as whichever entity type in the
schema-graph it fits — it must be a type that some relation can point INTO, since it sits at the
end of the chain. Ground it with the exact source text you read, not a summary of it.

From that grounded terminal, trace the reasoning back toward the question one grounded step at a
time. At each step, ask: what other fact, of some type the schema-graph allows, connects to the
node you just grounded — and does the corpus actually state that connecting fact? Search for it,
read it, and if it is there, ground it the same way, quoting its source, and add the edge joining
it to the node before it using whichever relation the schema-graph offers for that pair of types.
Keep stepping backward like this for as long as the corpus keeps supplying real, groundable
intermediates — a chain worth building is usually more than one hop long. Do not stop the first
time you reach something groundable and call it done; that is the most common way this task goes
wrong. An easy-to-ground terminal is not a signal that the rest of the path is short — it is only
where you started. Keep mining the surrounding text for the next genuine link, and the one after
that, until you reach a node that plausibly belongs to the question itself, or until the corpus
truly has nothing more to offer on this path.

The question itself becomes the entry node of the chain — reify it as whichever query-type the
schema-graph provides for this. This entry node is usually a shape of question, not a fact this
corpus states about a specific case, so leave it ungrounded unless you actually find the corpus
describing that same situation — do not force a quote onto it. Connect the entry node to the rest
of the chain with an ontology-legal relation, the same way as any other step.

Honesty is the one place this task is strict. Ground a node only when you can point to and quote
the exact text that states it. If a value the chain seems to need — the terminal, an intermediate,
or anything else — is not something you can find and quote in the corpus, say so plainly and leave
it out rather than write a plausible-sounding invention. A partial chain that stops at the last
point you could actually verify is a correct result, not a failure; a chain padded with an
unverifiable guess is the failure. The same applies to every edge: connect two nodes only when the
corpus text actually supports that link, not merely because the schema-graph would permit it.

Some node types in the schema-graph are marked as needing a valid time span rather than a single
timeless fact — a role held over a period, a state true within a window, a value that changed
later. When you ground a node of one of those types, read the surrounding text for the dates or
span it describes and set that node's start and/or end accordingly; leave a side unset if the
source never states it, and leave both unset entirely for a node whose type does not ask for
validity.

Emit every node and edge you find through `graph_upsert` as you go — nodes carrying their type,
label, any aliases the source text uses for the same thing, the source grounding you quoted, and a
valid time span where the type calls for one; edges carrying the from node, the to node, and the
relation joining them. When the chain is as complete as the corpus allows — grounded from the
answer back to wherever the trail of real, quotable intermediates ends, with the question reified
at the entry — end your turn with this exact line, and nothing after it:

=== CHAIN ===
