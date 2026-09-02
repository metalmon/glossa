# kbx distil

You are given ONE seed node from a reasoning graph over this corpus. It is a grounded TERMINAL —
the sink of a reasoning chain — and for this task it is the fixed ANSWER. You are also given the
source text it was grounded from, and, below it, a chain that ALREADY leads to this terminal. Above
this prompt you have the ontology's schema-graph: the entity types this corpus's knowledge is typed
with, and the relations connecting them, each shown with the types it may run from and to. Some of
those relations chain toward a terminal like this one; the schema-graph is what tells you which, so
read it as the map of how a reader would reason their way here.

The existing chain shown to you is there for one reason only: so you can see what is already covered
and understand what this terminal is really about. Do NOT reproduce it, and do NOT reuse the question
it would answer. Treat it as the ground you must step around, not the ground you step on.

Your job is to invent ONE NEW question whose answer is this same terminal fact, approached from a
DIFFERENT entry angle — a different starting point in the corpus that still leads here when a reader
reasons through the relations. The answer is fixed: it is the given terminal. Do not go hunting for
some other answer; go hunting for a fresh way IN. You may search and read to find a genuine new entry
point and to confirm the path from it actually reaches this terminal, but everything you read is in
service of reaching the answer you were handed, not replacing it.

Phrase the question from the side the reasoning ENTERS from — the query-side type in the schema-graph,
the way a person who only knows the entry, not the destination, would ask it. The good version cannot
be answered from the terminal's own text alone: a reader must reason across the corpus, following the
chain, to arrive at it. If your question could be answered just by reading the terminal's chunk, or
from general knowledge, it tests nothing — find a further or more indirect entry, or judge it honestly
below.

Write BOTH the question and the answer in the same language as the corpus's source text — the words
you read in the graph and the chunks — not in the language of these instructions. If the grounded text
is in one language, a question or answer in another language is wrong even when its meaning is right.
Give the answer as the grounded terminal fact itself: concise, in the corpus's own words where
possible, not a summary or paraphrase you constructed.

Before you emit anything, judge your own work honestly. Ask: could someone answer this correctly from
the terminal's own text alone, or from general knowledge, without reasoning through the corpus? If so,
set `gate_ok` to false and say why in `gate_reason`. Ask also: does the entry angle you chose really
reason its way to this terminal, with no missing or invented link? If the path is shaky, set `gate_ok`
to false rather than submit one you aren't confident holds together. An honest false here costs
nothing; a dishonest true produces a bad question that looks fine until someone checks it.

Set `hop_type` to `multihop` if answering genuinely requires walking two or more grounded hops — not
answerable from the terminal's chunk or from a single lookup — else `lexical`. This is your own read
of the question's shape; be honest about it.

You have only a LIMITED number of exploration steps, so do not search forever. Once you have a genuine
new entry and have confirmed its path reaches the terminal — or gone as far as the corpus genuinely
supports, whichever comes first — STOP and commit.

EVERY attempt MUST end with exactly one `propose_gold` call — this is the only way your work is
recorded, and an attempt that ends without it is thrown away entirely. Do not end with a plain text
answer or a summary of what you found; end with the tool call. Even when you could not find a usable
new entry, still call `propose_gold` with your best-effort question and answer and `gate_ok=false`
(explain why in `gate_reason`) — an honest gated-false proposal is useful; silence is not.

Call `propose_gold` once with the question, the answer (the given terminal fact), the ids of the nodes
on the path you expect a reader to walk (entry through terminal, in order), `gate_ok`, a short
`gate_reason` explaining your judgment either way, and `hop_type`.
