# kbx reflect

You are helping improve a SYSTEM PROMPT used by a weaker reader model. That reader is given a
question and a set of tools to explore a knowledge graph and its underlying documents (search,
read, glossary, neighbors, reach, and similar), and it must arrive at a short, exact answer. The
prompt you are rewriting is the reader's entire briefing — its only instruction for how to use the
tools, how to reason over what it finds, and how to phrase its final answer. Nothing else steers
the reader's behavior.

You will be shown the current prompt together with a handful of cases where the reader, following
that prompt, still got the wrong answer or gave up too early — typically the transcript of tool
calls it made, what it found, and where its reasoning went off track (stopped at an intermediate
fact, picked a same-type distractor, hallucinated instead of grounding in a retrieved passage,
never followed a chain far enough, or misread what it retrieved). Your job is to rewrite the
prompt so the reader is less likely to make that same class of mistake next time, while staying a
prompt any question over any corpus could use — not a patch for these particular cases.

Read the failing transcripts for the PATTERN behind the mistake, not the specific facts. A prompt
that tells the reader to keep following a chain of references instead of answering at the first
plausible-looking node, or to state which passage grounds its final answer, transfers to any
corpus. A prompt that mentions a specific entity, date, document, or answer from the failing cases
does not — it would only ever help on those exact questions and actively hurts on every other one.
So: draw the lesson, discard the example. Do not name, quote, or paraphrase any specific fact,
entity, or answer that appeared only because it was in a failing case's question, retrieved
passage, or gold/predicted answer. If you need an example to illustrate a rule, invent a generic
one that could not be mistaken for a real case.

Keep the prompt readable and usable end to end: clear instructions on tool use, on reasoning
through multi-step questions, and on the exact format the final answer must take. Prefer sharpening
or reorganizing what's already there over piling on new caveats — a prompt that has grown a rule
for every past failure stops being followable. If the current prompt already covers a lesson well,
leave that part alone and focus your edit where it's actually needed.

Return your complete rewritten prompt, and nothing else, after this exact line:

=== NEW SYSTEM PROMPT ===

Everything before that marker is space for your own analysis, if you want to think out loud first;
only the text after it is used. Do not wrap the prompt in a code block or add commentary after it.
