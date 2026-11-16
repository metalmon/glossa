//! Answer-reachability diagnostic: for each answerable gold with a `source`, how deep does the
//! gold answer node rank under a single-seed (question-anchored) `compose_ppr`? A deep median rank
//! means the graph poorly connects questions to answers (coverage) — the signal for a rebuild.
use crate::dataset_ops::Case;
use glossa::graph::store::GraphStore;
use std::collections::BTreeMap;

pub struct HopConn {
    pub n: usize,
    pub skipped: usize,
    pub median_rank: Option<usize>,
    pub p90_rank: Option<usize>,
    pub hit5: f32,
    pub hit20: f32,
    pub unreachable: usize,
}

const RANK_WINDOW: usize = 200;

/// Map `<doc>#<N>` -> the reasoning nodes that MENTIONS a Section grounded at that doc+ordinal.
///
/// The fragment must be the ordinal and nothing else. This used to keep every ASCII digit in the
/// fragment and drop the rest, which accepted `#p.14` as page 14 — and, far worse, turned
/// `#4.1.3 Safety` into 413, inventing an ordinal out of a heading. That is the same guess
/// `gold_ord` stopped making: the ordinal is the address, a label is not, and a ref written any
/// other way is a dataset error. Refusing it surfaces where the old spelling still sits; absorbing
/// it hid both the stale refs and the headings it silently mangled.
fn answer_nodes(g: &GraphStore, source: &str) -> anyhow::Result<Vec<String>> {
    // Three cases, and the third is why this is not just a stricter parse. No `#` at all is a
    // DOCUMENT-level ref and matches every Section in the file — that is legitimate and unchanged.
    // A fragment of digits that fits a u32 is the ordinal. Anything else — including an
    // all-digit fragment that overflows u32 — is a dataset error and must match NOTHING: falling
    // through to the document-level path (page == None) would make a bad ref match every section
    // instead of one, which is broader than the bug it replaces. So the parse is matched, not
    // discarded with `.ok()`: a failed parse joins the refusal arm rather than silently becoming
    // `None`. Split on the LAST `#` (`rsplit_once`), matching `judge::parse_ref`, so a path that
    // itself contains `#` still resolves against its final locator.
    let (doc, page) = match source.rsplit_once('#') {
        Some((d, frag)) if !frag.is_empty() && frag.bytes().all(|b| b.is_ascii_digit()) => {
            match frag.parse::<u32>() {
                Ok(n) => (d.to_string(), Some(n)),
                Err(_) => {
                    eprintln!(
                        "gold ref's fragment overflows an ordinal: {source:?} (fragment {frag:?}); write it as `path#N`"
                    );
                    return Ok(Vec::new());
                }
            }
        }
        Some((_, frag)) => {
            eprintln!(
                "gold ref's fragment is not an ordinal: {source:?} (fragment {frag:?}); write it as `path#N`"
            );
            return Ok(Vec::new());
        }
        None => (source.to_string(), None),
    };
    let base = std::path::Path::new(&doc)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&doc)
        .to_string();
    // Section nodes: (id, source_path, range). run_select returns rows of stringified columns.
    let rows = g.run_select(
        "SELECT id, source_path, range FROM nodes WHERE node_type = 'Section'",
        100_000,
    )?;
    let mut out = Vec::new();
    for row in rows {
        let (id, sp) = (&row[0], &row[1]);
        let sp_base = std::path::Path::new(sp)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(sp);
        if sp_base != base {
            continue;
        }
        if let Some(target) = page {
            // Section id is `path#ord` (ord == page for PDFs). Post-refactor `range` is empty
            // for PDF/split-text chunks, so match the gold page against the id suffix instead.
            let sec_ord = id.rsplit_once('#').and_then(|(_, o)| o.parse::<u32>().ok());
            if sec_ord != Some(target) {
                continue;
            }
        }
        // reasoning nodes that MENTIONS this section
        let mut mentioners: Vec<String> = g
            .incoming(id)?
            .into_iter()
            .filter(|e| e.edge_type == "MENTIONS")
            .map(|e| e.from)
            .collect();
        if mentioners.is_empty() {
            out.push(id.clone()); // fallback: the section itself
        } else {
            out.append(&mut mentioners);
        }
    }
    Ok(out)
}

pub fn answer_reachability(
    g: &GraphStore,
    cases: &[Case],
) -> anyhow::Result<BTreeMap<String, HopConn>> {
    let mut ranks: BTreeMap<String, Vec<Option<usize>>> = BTreeMap::new();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    for c in cases {
        let bucket = if c.hop_type.is_empty() {
            "lexical".to_string()
        } else {
            c.hop_type.clone()
        };
        if !c.answerable || c.source.is_empty() {
            *skipped.entry(bucket).or_default() += 1;
            continue;
        }
        let mut ans: Vec<String> = Vec::new();
        for s in &c.source {
            ans.extend(answer_nodes(g, s)?);
        }
        if ans.is_empty() {
            *skipped.entry(bucket).or_default() += 1;
            continue;
        }
        let ans_set: std::collections::HashSet<&str> = ans.iter().map(|s| s.as_str()).collect();
        // single-seed: question only (dataset stat has no reader term)
        let ranked = glossa::graph::compose::compose_ppr(g, "", &c.question, RANK_WINDOW)?;
        let rank = ranked
            .iter()
            .position(|cand| ans_set.contains(cand.id.as_str()));
        ranks.entry(bucket).or_default().push(rank);
    }
    let mut out = BTreeMap::new();
    let keys: std::collections::BTreeSet<String> =
        ranks.keys().chain(skipped.keys()).cloned().collect();
    for k in keys {
        let rs = ranks.get(&k).cloned().unwrap_or_default();
        let found: Vec<usize> = rs.iter().filter_map(|r| *r).collect();
        let mut sorted = found.clone();
        sorted.sort_unstable();
        let pct = |p: f32| {
            sorted
                .get(((p * sorted.len() as f32) as usize).min(sorted.len().saturating_sub(1)))
                .copied()
        };
        let hit = |thr: usize| {
            if rs.is_empty() {
                0.0
            } else {
                found.iter().filter(|r| **r < thr).count() as f32 / rs.len() as f32
            }
        };
        out.insert(
            k.clone(),
            HopConn {
                n: rs.len(),
                skipped: skipped.get(&k).copied().unwrap_or(0),
                median_rank: if sorted.is_empty() { None } else { pct(0.5) },
                p90_rank: if sorted.is_empty() { None } else { pct(0.9) },
                hit5: hit(5),
                hit20: hit(20),
                unreachable: rs.iter().filter(|r| r.is_none()).count(),
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Edge, Node, Provenance};

    fn prov(source_path: &str, range: Option<&str>) -> Provenance {
        Provenance {
            source_path: source_path.to_string(),
            range: range.map(|s| s.to_string()),
            file_sig: None,
            origin: "agent".into(),
            confidence: 1.0,
            created_at: 0,
        }
    }

    #[test]
    fn reachability_reports_rank_for_a_two_hop_answer() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();

        // Section grounded at doc "man.pdf" p.5. Id encodes `path#ord` (ord == page).
        let sec = Node {
            id: "man.pdf#5".into(),
            node_type: "Section".into(),
            label: "Section 5".into(),
            aliases: vec![],
            prov: prov("man.pdf", Some("p.5")),
        };
        g.put_node(&sec).unwrap();

        // A Resolution (the answer) that MENTIONS the section. Its label deliberately shares no
        // vocabulary with the question, so it is reached only via the graph walk, not seeded
        // directly by BM25 (a direct seed hit would be excluded from compose_ppr's output as
        // "what the reader already has").
        let res = Node {
            id: "res:ans".into(),
            node_type: "Resolution".into(),
            label: "replace the fuse and check the wiring harness".into(),
            aliases: vec![],
            prov: prov("man.pdf", None),
        };
        g.put_node(&res).unwrap();
        let mentions = Edge {
            from: "res:ans".into(),
            to: "man.pdf#5".into(),
            edge_type: "MENTIONS".into(),
            prov: prov("man.pdf", None),
        };
        g.put_edge(&mentions).unwrap();

        // An anchor node two hops away, reachable to res:ans via a Chaining edge, whose label
        // shares vocabulary with the question so compose_ppr seeds toward it.
        let anchor = Node {
            id: "sym:anchor".into(),
            node_type: "Symptom".into(),
            label: "anchor question device malfunction".into(),
            aliases: vec![],
            prov: prov("man.pdf", None),
        };
        g.put_node(&anchor).unwrap();
        let chain = Edge {
            from: "sym:anchor".into(),
            to: "res:ans".into(),
            edge_type: "Chaining".into(),
            prov: prov("man.pdf", None),
        };
        g.put_edge(&chain).unwrap();

        let cases = vec![Case {
            id: "q1".into(),
            question: "anchor question".into(),
            answer: "x".into(),
            aliases: vec![],
            tags: vec![],
            hop_type: "multihop".into(),
            needs_graph: String::new(),
            // `man.pdf#5` — the Section's own id suffix, which is what a gold ref is. This fixture
            // used to say `man.pdf#p.5`, and refusing that form is what surfaced it: the first
            // place the old spelling still sat was our own test.
            source: vec!["man.pdf#5".into()],
            answerable: true,
            extra: toml::Table::new(),
        }];
        let report = answer_reachability(&g, &cases).unwrap();
        let mh = report.get("multihop").expect("multihop bucket");
        assert_eq!(mh.n, 1);
        assert!(
            mh.median_rank.is_some(),
            "answer node was located in the ranking"
        );
    }

    #[test]
    fn answer_nodes_matches_section_ord_when_range_is_empty() {
        // PDF/split-text chunks now ground with an empty `range`; the Section id's `#ord`
        // suffix (== page for PDFs) is the only thing left to match the gold page against.
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();

        let sec = Node {
            id: "d.pdf#14".into(),
            node_type: "Section".into(),
            label: "Section 14".into(),
            aliases: vec![],
            prov: prov("d.pdf", None),
        };
        g.put_node(&sec).unwrap();

        let res = Node {
            id: "res:ans".into(),
            node_type: "Resolution".into(),
            label: "the answer text".into(),
            aliases: vec![],
            prov: prov("d.pdf", None),
        };
        g.put_node(&res).unwrap();
        let mentions = Edge {
            from: "res:ans".into(),
            to: "d.pdf#14".into(),
            edge_type: "MENTIONS".into(),
            prov: prov("d.pdf", None),
        };
        g.put_edge(&mentions).unwrap();

        let out = answer_nodes(&g, "d.pdf#14").unwrap();
        assert_eq!(out, vec!["res:ans".to_string()]);

        // A Section at ordinal 413 — the exact ordinal the OLD lenient parse invented out of
        // `#4.1.3 Safety` by keeping every ASCII digit (4, 1, 3) and dropping the rest. Without
        // this node the heading assertion below would pass under the old parse too (413 matched
        // no Section in this fixture), which would pin nothing.
        let sec_413 = Node {
            id: "d.pdf#413".into(),
            node_type: "Section".into(),
            label: "Section 413".into(),
            aliases: vec![],
            prov: prov("d.pdf", None),
        };
        g.put_node(&sec_413).unwrap();

        // The `#p.<N>` spelling is NOT accepted any more. It is a dataset error, and refusing it is
        // how we find where it still sits; absorbing it also meant `#4.1.3 Safety` became ordinal
        // 413, an ordinal invented out of a heading.
        assert!(
            answer_nodes(&g, "d.pdf#p.14").unwrap().is_empty(),
            "a `p.N` fragment must not resolve to ordinal 14"
        );
        assert!(
            answer_nodes(&g, "d.pdf#4.1.3 Safety").unwrap().is_empty(),
            "a heading fragment must not be mangled into an ordinal 413 (see the `d.pdf#413` Section above)"
        );

        // An all-digit fragment that overflows u32 must join the refusal arm, not fall through to
        // a document-level match (which would match every Section in the file, including d.pdf#14
        // and d.pdf#413 above) — the exact trap the third case exists to close.
        assert!(
            answer_nodes(&g, "d.pdf#4294967296").unwrap().is_empty(),
            "an overflowing ordinal must not match the whole document"
        );
    }

    #[test]
    fn unanswerable_and_sourceless_cases_are_skipped() {
        let d = tempfile::tempdir().unwrap();
        let g = GraphStore::open(d.path()).unwrap();
        let cases = vec![Case {
            id: "u".into(),
            question: "q".into(),
            answer: "".into(),
            aliases: vec![],
            tags: vec![],
            hop_type: "".into(),
            needs_graph: String::new(),
            source: vec![],
            answerable: false,
            extra: toml::Table::new(),
        }];
        let report = answer_reachability(&g, &cases).unwrap();
        assert!(report.values().all(|h| h.n == 0), "no evaluable cases");
    }
}
