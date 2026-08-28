# kbx distil

You are given one seed node already grounded in a reasoning graph over this corpus, together with
the source text it was grounded from. Above this prompt you have been given the ontology's
schema-graph — the entity types this corpus's knowledge is typed with, and the relations that
connect them, each shown with the types it may run from and to. Your job is to turn this one seed
into ONE new synthetic question-and-answer pair that a reader could only get right by actually
following a real chain of those relations through the corpus, not by guessing or by reading the
seed alone.

Start from the seed and explore outward with the search/read/glossary/reach/sql tools you have.
Follow a relation the schema-graph declares from the seed's type to some other grounded fact, read
what that step actually says in the corpus, and see whether it in turn connects onward to something
further. Keep stepping as long as the corpus keeps supplying real, groundable next links — a chain
of one hop makes for a thin question; two or more genuine hops make for a better one. Stop stepping
the moment you can no longer point to and quote the text that grounds the next link — an invented
or merely plausible step is worse than no step at all.

Aim the chain at a terminal whose grounding lives in a DIFFERENT chunk or document than the seed's.
When the answer you would give is quoted in the seed's own source text — or sits in the very same
chunk that grounds the seed — the reader never had to walk anywhere, and the question tests nothing;
that is the most common way an attempt turns out hollow. So before you commit, glance at where your
candidate answer is grounded: if it shares the seed's chunk, either keep stepping until the chain
lands somewhere genuinely further, or start from a different seed rather than spend the attempt.

Once you have a real chain of two or more grounded facts (or, failing that, whatever the corpus
genuinely supports), design a question whose answer is the chain's terminal fact but whose PATH to
that answer requires the intermediate steps — a question that names or describes the seed (or an
early link) and asks about something only reachable by walking forward through the chain, not
something answerable from the seed's own text in isolation. Phrase it the way a person unfamiliar
with the chain's shortcuts would ask it, and give the answer as the grounded terminal fact itself —
concise, in the corpus's own words where possible, not a summary or paraphrase you constructed.
Write BOTH the question and the answer in the same language as the corpus's source text — the words
you read while walking the chain — not in the language of these instructions. If the grounded text
is in one language, a question or answer in another language is wrong even when its meaning is right.

Before you emit anything, judge your own work honestly. Ask: could someone answer this question
correctly just from the seed, or from general knowledge, without ever needing the intermediate
steps? If so, this question isn't really testing the chain — set `gate_ok` to false and say why in
`gate_reason`. Ask also: does the chain you actually traced really end at the answer you're about to
give, with no missing or invented link along the way? If any step is shaky, set `gate_ok` to false
rather than submit a chain you aren't confident holds together. An honest false here costs nothing;
a dishonest true produces a bad question that looks fine until someone checks it.

When you are ready — whether the chain is strong or you've concluded it isn't — call `propose_gold`
exactly once with the question, the answer, the ids of the chain nodes you walked (seed through
terminal, in order), `gate_ok`, and a short `gate_reason` explaining your judgment either way.
