# glossa — ontology.toml examples

Companion to the design spec (§11). Shows the **fixed core vs domain config** split in
concrete form. The structural + glossary core (`Document`/`Section`/`Term`,
`CONTAINS`/`MENTIONS`/`CO_OCCURS`, SKOS `broader`/`narrower`/`related`) is **always present**
and is **not** declared here. `ontology.toml` only declares the **open** entity/relation
vocabulary for the agent-built layers (3–4), and is validated by the native validator
(constraint vocabulary maps 1:1 to SHACL Core).

## 1. Generic default (`ontology.toml`)

Ships out of the box. Permissive (`strict = false`) so any deployment works before a vertical
is chosen.

```toml
[meta]
name = "generic"
version = "1"

# ---- Entities (layer 3) ----
[entities.Person]
props = ["full_name", "role"]

[entities.Organization]
props = ["name", "aliases"]

[entities.Location]
props = ["name"]

[entities.Event]
props = ["name", "date"]

[entities.Concept]            # a domain term promoted to a first-class entity
props = ["name", "definition"]

[entities.Project]
props = ["name"]

[entities.Product]
props = ["name", "version"]

[entities.System]            # system / component
props = ["name"]

# ---- Entity <-> Entity relations (layer 3) ----
[relations.RELATES_TO]
from = ["*"]
to   = ["*"]

[relations.PART_OF]
from = ["*"]
to   = ["*"]

[relations.AUTHORED_BY]
from = ["Document"]
to   = ["Person", "Organization"]

[relations.OWNS]
from = ["Person", "Organization"]
to   = ["*"]

[relations.DEPENDS_ON]
from = ["System", "Project", "Product"]
to   = ["System", "Project", "Product"]

# ---- Document <-> Document cross-doc links (layer 4) ----
[relations.REFERENCES]
from = ["Document"]
to   = ["Document"]

[relations.SUPERSEDES]
from = ["Document"]
to   = ["Document"]
cardinality_to = "0..1"      # a doc supersedes at most one predecessor

[relations.RELATED_DOC]
from = ["Document"]
to   = ["Document"]

[relations.CONTRADICTS]
from = ["Document"]
to   = ["Document"]

[relations.VERSION_OF]
from = ["Document"]
to   = ["Document"]

[validation]
strict = false               # generic schema is permissive; verticals tighten to strict
```

## 2. Legal vertical (RU) (`ontology.legal-ru.toml`)

Extends the generic schema and tightens validation. Illustrates how a commercial engagement
declares domain types without touching the core.

```toml
[meta]
name    = "legal-ru"
version = "1"
extends = "generic"          # inherit generic entities/relations, add domain ones

[entities.Contract]
props = ["number", "date", "subject", "amount", "currency"]

[entities.Party]             # сторона договора
props = ["legal_name", "inn", "kpp", "role"]   # role: заказчик | исполнитель | ...

[entities.Clause]            # пункт договора
props = ["number", "title"]

[entities.Obligation]
props = ["description", "due_date"]

[entities.Court]
props = ["name"]

[entities.CaseFile]          # судебное дело
props = ["number"]

[relations.PARTY_TO]
from = ["Party"]
to   = ["Contract"]
cardinality_from = "2..*"    # a contract has at least two parties

[relations.HAS_CLAUSE]
from = ["Contract"]
to   = ["Clause"]

[relations.IMPOSES]
from = ["Clause"]
to   = ["Obligation"]

[relations.AMENDS]           # допсоглашение изменяет договор
from = ["Document"]
to   = ["Contract"]

[relations.GOVERNED_BY]      # рамочный договор
from = ["Contract"]
to   = ["Contract"]

[validation]
strict = true                # reject any entity/relation type not declared (incl. inherited)
```

## 3. Notes

- `extends` lets a vertical inherit the generic vocabulary; the validator merges then applies
  the vertical's `validation`.
- `from`/`to` constrain relation endpoints by node type (`"*"` = any). Maps to SHACL
  `sh:targetClass` + `sh:class` on the value node.
- `cardinality_from`/`cardinality_to` map to SHACL `sh:minCount`/`sh:maxCount`.
- `strict = true` maps to SHACL `sh:closed true` (reject undeclared types/properties).
- Provenance fields (`origin`, `confidence`, `created_by`, `evidence`) are implicit on every
  node/edge and not declared per-type.
