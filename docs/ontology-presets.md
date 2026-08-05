# Ontology presets

An **ontology** turns glossa from a plain text index into a typed reasoning
graph: it declares the entity and relation types, the reasoning spines,
constraint types and grounding rules that the enricher is validated against
(see [Graph and ontology](graph-and-ontology.md)).

glossa ships a curated set of **presets** — ready-made ontologies for common
tasks — baked into the binary. Pick one when you first index a corpus instead of
hand-writing a `.glossa/ontology.toml`:

```bash
kb index ./my-corpus --ontology tender
```

This writes the `tender` preset to `./my-corpus/.glossa/ontology.toml` and then
indexes. The file on disk is the single source of truth from then on — open it,
read it, edit it freely.

## Why presets

glossa's most distinctive use is **provable conformance**: given documents full
of rules and requirements, check something against them and trace every verdict
back to the exact source section (constraint solving + `MENTIONS` grounding —
something plain retrieval cannot do). A preset is what wires that up for your
domain in one command. The flagship presets (Tier 1 below) are all conformance
and compliance shapes; the rest cover everyday operational knowledge.

## Browsing and choosing

```bash
kb ontology list                 # the full catalog, listed by tier, then family
kb ontology list --family risk   # just one family
kb ontology show compliance      # print a preset's TOML before committing to it
kb ontology suggest "we receive supplier bids and must check every requirement is answered"
```

`suggest` ranks presets against a free-text description of your documents and
prints the best matches — it runs entirely offline, no model call.

Presets also answer to **aliases**, so the name you reach for usually works:
`--ontology normocontrol` resolves to `compliance`, `rfp` to `tender`, `ropa` to
`data-privacy`, `sod` to `access-governance`, `troubleshooting` to `support`. A
typo prints the closest matches.

## Applying a preset

Two ways, one underlying action:

```bash
# Fast path — materialize the overlay, then index.
kb index ./my-corpus --ontology compliance

# Explicit — materialize only, index later.
kb ontology init ./my-corpus --template compliance
kb index ./my-corpus
```

The on-disk file always wins. If `.glossa/ontology.toml` already exists,
`kb index --ontology …` keeps it (printing a note) and still indexes; it is
never overwritten silently. Pass `--force` to replace it:

```bash
kb ontology init ./my-corpus --template compliance --force
```

## The catalog

Full name → description. `kb ontology show <name>` prints the actual types and
relations.

### Tier 1 — conformance & compliance

| Preset | What it models |
|--------|----------------|
| `compliance` | Check a document against a body of normative requirements (the normative-control shape): requirement → field/value → evidence in the text. |
| `tender` | Bid / RFP / tender-response conformance: every requirement answered, every mandatory document attached, specs meet the stated minimums. |
| `contract` | Extract obligations, deadlines and penalties from a contract and check it against a playbook of acceptable positions. |
| `certification` | Declaration / certification of conformity: which technical regulations apply → which tests → which supporting documents. |
| `qa-inspection` | Incoming/outgoing goods inspection: a lot or item against a spec or datasheet, certificate of conformance. |
| `audit` | Controls audit: control → risk → evidence → gap. |
| `reg-change` | Impact of a changed standard or law: which internal documents, products and processes are affected. |
| `data-privacy` | Records of processing (ROPA): data → legal basis → retention period → processor. |
| `access-governance` | Access and segregation of duties: who may do what, forbidden role combinations, least privilege. |
| `hr-compliance` | Mandatory personnel records per role (training, medicals, clearances) and their expiry. |
| `risk-register` | Enterprise risk register: risk → cause → control → owner. |
| `fmea` | Failure modes of a product or process and their controls (quality, reliability, safety). |
| `policy` | Internal rulebook: policy → rule → exception, with a scope of application. |

### Tier 2 — operational knowledge

| Preset | What it models |
|--------|----------------|
| `support` | Troubleshooting plus documentation search: symptom → cause → resolution. |
| `sop` | Step-by-step procedures and instructions (incl. onboarding and runbooks). |
| `faq` | Flat question-and-answer knowledge base. |
| `traceability` | Requirement ↔ implementation ↔ verification traceability (QMS / ISO 9001). |
| `vendor` | Vendor management: contracts, SLAs, third-party risk (incl. supply chains). |
| `product-catalog` | Catalogue of products / plans / features: composition, dependencies, replacements. |
| `customer-journey` | Customer-journey map: stages, touchpoints, channels, pain points. |
| `okr` | Goal decomposition: objective → key result → initiative. |
| `project-schedule` | Project plan: milestones, tasks, dependencies, blockers. |
| `decision-log` | Decision log with rationale (ADR). |
| `timeline` | Chronology of events / case history. |
| `competency` | Competency matrix: a role requires a skill, a person has a skill. |
| `org-roles` | Organisational structure and responsibility matrix (RACI). |

> No preset fits? Index without `--ontology` — you still get the full structural
> layer (documents, sections, terms, `MENTIONS`/`CO_OCCURS`), which is the
> generic knowledge-base case. Add a reasoning overlay later whenever you like.

## Customizing a preset

A materialized `.glossa/ontology.toml` is a plain file — the preset is only a
starting point. Common adjustments:

- **Rename types** to your domain's own vocabulary (the labels are what table
  exports and the graph explorer show).
- **Add entities or relations** the preset doesn't cover; keep `strict = true`
  so the enricher can't invent types outside your list.
- **Layer jurisdiction specifics privately.** The presets are deliberately
  generic and international. If your work is governed by a specific national
  standard or law, add those requirements to your own copy — that specificity
  lives in your corpus, not in the shipped preset.

After editing, re-run `kb index` (structural layer) and, if you have reasoning
data, `kb graph generalize` to recompute derived links.

## Grounding

Presets that carry reasoning data mark one node type per shape with
`requires_grounding = true` — the concrete, document-quoting node (a
`Resolution`, a `Step`, an `Evidence`). `graph_upsert` then rejects such a node
unless it has a `MENTIONS` edge to a source section, and `kb graph generalize`
surfaces any that later lose their grounding. See
[Grounding](graph-and-ontology.md#grounding) for the mechanics.

## See also

- [Graph and ontology](graph-and-ontology.md) — how the overlay is used by the
  enricher, hygiene and generalize passes.
- [MCP tools](mcp.md) — `get_ontology` surfaces the active overlay to an agent.
