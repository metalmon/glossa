use crate::graph::store::GraphStore;
use crate::index::store::DocIndex;
use std::collections::BTreeSet;

pub struct AbsentTerm {
    pub term: String,
}

/// Distinctive question terms with no corpus coverage. A term is any alphabetic token of at least
/// 5 characters (dedup'd, case-insensitive); it is UNCOVERED when `covered` returns false. No
/// stopword list: the index tokenizer (see `index::multilang`) stems but does NOT strip stopwords,
/// so a frequent word is itself searchable and `covered` reports it present — self-calibrating on
/// the corpus rather than on a hardcoded, language-specific word list.
pub fn coverage_uncovered(
    question: &str,
    covered: &dyn Fn(&str) -> bool,
    _k: usize,
) -> Vec<AbsentTerm> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tok in question.split(|c: char| !c.is_alphabetic()) {
        if tok.chars().count() < 5 {
            continue;
        }
        let low = tok.to_lowercase();
        if !seen.insert(low.clone()) {
            continue;
        }
        if !covered(tok) {
            out.push(AbsentTerm {
                term: tok.to_string(),
            });
        }
    }
    out
}

/// How hard the coverage-abstention gate bites, resolved by the caller (MCP layer: from
/// `Profile` + the corpus ontology's `enforcement` override — see `mcp::GlossaServer`).
/// `Filter` replaces the tool body with the abstention sentinel once `k` distinctive terms are
/// uncovered; `Signal` only annotates the body, keeping the results; `Off` never gates (the
/// byte-identical-to-today path every non-MCP caller and existing test uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enforcement {
    Filter,
    Signal,
    Off,
}

/// The sentinel the `Filter` tier returns instead of results — DATA (Russian, corpus-facing),
/// not tracked code/comments, so it is exempt from the English-only convention.
pub const SENTINEL: &str = "В базе знаний нет информации по этому вопросу";

/// True when `term` is covered by the corpus: either a live BM25 hit (`idx.search`, morphology-
/// aware — inflected forms/synonyms handled the same as a normal search), or `term`'s stem
/// matches the label/alias of a graph node (`GraphStore::resolve`, same morphology pipeline) that
/// is itself grounded or reaches a grounded terminal along a chaining edge (see
/// [`crate::graph::grounded_or_chains_to_grounded`]). A best-effort SIGNAL for the abstention
/// gate — false negatives (a covered term flagged absent) just make the gate slightly more
/// cautious, never less; a store error is treated as "not covered" via that fn, same reasoning.
pub fn covered(term: &str, idx: &DocIndex, g: &GraphStore) -> bool {
    if idx.search(term, 1).map(|h| !h.is_empty()).unwrap_or(false) {
        return true;
    }
    g.resolve(term)
        .map(|ids| {
            ids.iter()
                .any(|id| crate::graph::grounded_or_chains_to_grounded(g, id).unwrap_or(false))
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_flags_only_uncovered_distinctive_terms() {
        // "covered" = present in this fake corpus set (case-insensitive substring of any entry)
        let corpus = ["configure profibus maxtsdr", "io modules"];
        let covered = |t: &str| corpus.iter().any(|c| c.contains(&t.to_lowercase()));
        let absent = coverage_uncovered("how to configure maxTsdr for zzqunknownterm", &covered, 1);
        let terms: Vec<&str> = absent.iter().map(|a| a.term.as_str()).collect();
        assert!(terms
            .iter()
            .any(|t| t.eq_ignore_ascii_case("zzqunknownterm")));
        assert!(
            !terms.iter().any(|t| t.eq_ignore_ascii_case("maxTsdr")),
            "covered term not flagged"
        );
        assert!(
            !terms.iter().any(|t| t.chars().count() < 5),
            "short terms excluded"
        );
    }

    fn prov() -> crate::graph::store::Provenance {
        crate::graph::store::Provenance {
            source_path: "d.pdf".into(),
            range: None,
            file_sig: None,
            origin: "agent".into(),
            confidence: 0.8,
            created_at: 1,
        }
    }
    fn node(id: &str, ty: &str, label: &str) -> crate::graph::store::Node {
        crate::graph::store::Node {
            id: id.into(),
            node_type: ty.into(),
            label: label.into(),
            aliases: Vec::new(),
            prov: prov(),
        }
    }
    fn edge(from: &str, rel: &str, to: &str) -> crate::graph::store::Edge {
        crate::graph::store::Edge {
            from: from.into(),
            to: to.into(),
            edge_type: rel.into(),
            prov: prov(),
        }
    }

    #[test]
    fn covered_by_bm25_hit() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        i.write_chunks(&[crate::model::Chunk {
            doc_path: "profibus.pdf".into(),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "profibus maxTsdr timeout".into(),
        }])
        .unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        assert!(covered("profibus", &i, &g), "indexed term must be covered");
    }

    #[test]
    fn covered_by_grounded_graph_node() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        g.put_node(&node("sec:doc", "Section", "Manual intro"))
            .unwrap();
        g.put_node(&node(
            "fact:quasar",
            "Fact",
            "Quasar9000 module reset procedure",
        ))
        .unwrap();
        g.put_edge(&edge("fact:quasar", crate::graph::MENTIONS, "sec:doc"))
            .unwrap();
        assert!(
            covered("quasar9000", &i, &g),
            "term matching a grounded fact's label must be covered even with no BM25 hit"
        );
    }

    #[test]
    fn not_covered_when_absent_from_both_index_and_graph() {
        let d = tempfile::tempdir().unwrap();
        let i = DocIndex::open_or_create(d.path()).unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        assert!(!covered("zzqunknownterm", &i, &g));
    }
}
