use crate::index::store::{DocIndex, RankedHit};
use crate::retrieve::config::RerankConfig;
use std::path::Path;

/// Backend-agnostic cross-encoder scorer: one relevance score per passage, same length + order as
/// `passages`. Higher = more relevant. Implemented in-process (`retrieve::rerank_engine`) and, later,
/// over HTTP.
pub trait Reranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>>;
    /// Whether `rerank`'s numbers are raw logits. In-process engines are, and keep the default; the
    /// HTTP reranker answers from its backend (`tei|kbi|vllm|llamacpp` yes, `cohere|jina` no), see
    /// `http_scorer::client`. Only a logit source gets a `rel_rerank`: a probability we did not
    /// compute is not one we can vouch for.
    fn emits_logits(&self) -> bool {
        true
    }
}

/// `1 / (1 + e^-x)` in f32. Saturation is asymmetric, because it comes from f32's own limits: the
/// result is exactly 1.0 once `e^-x` falls below the ulp of 1 (x >~ 17), but reaching exactly 0.0
/// needs `e^-x` to overflow to infinity (x <~ -88). In between, a very negative logit reads as a
/// tiny positive probability rather than zero, which is the honest reading of "almost certainly
/// irrelevant".
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Test double: returns `scores` positionally. A length mismatch with `passages` is what exercises
/// the fail-open path in `rerank_hits`.
pub struct MockReranker {
    pub scores: Vec<f32>,
}
impl Reranker for MockReranker {
    fn rerank(&self, _q: &str, _p: &[&str]) -> anyhow::Result<Vec<f32>> {
        Ok(self.scores.clone())
    }
}

pub struct RerankInfo {
    /// True only when the cross-encoder actually scored the pool and the returned hits are in ITS
    /// order. A configured reranker that failed at runtime reports `false` + a [`RerankInfo::fallback`]
    /// reason — the hits are then plain BM25 and every consumer that reads this field is told so.
    pub reranked: bool,
    pub pool: usize,
    /// Why the reranker did not apply, when one WAS configured and failed at runtime. `None` covers
    /// both "no reranker configured" and "rerank applied", which `reranked` already distinguishes.
    pub fallback: Option<String>,
}

/// What [`rerank_hits`] actually did. Returned alongside the hits because the two cases are
/// indistinguishable from the hits alone: a fail-open returns BM25 order, which is exactly what a
/// reranker that scored the pool in BM25 order would also return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankOutcome {
    /// The cross-encoder scored the whole pool; the hits carry its order and its scores.
    Applied,
    /// The scorer errored or returned the wrong number of scores. The hits are the BM25 order,
    /// trimmed. Carries the reason, so a caller can report it instead of discarding it.
    FailedOpen(String),
}

/// How much of a [`rerank_passage`] may be provenance rather than text. Roughly a quarter of the
/// scorer's 512-token window once a non-Latin script is counted, so the body it introduces still
/// dominates the input even for a pathologically deep heading breadcrumb.
const MAX_PROVENANCE_CHARS: usize = 200;

/// The text the cross-encoder scores for one hit: the chunk's own provenance, then its body.
///
/// Shape: `<folder> / <folder> / <file stem>`, then ` > <location>` when the chunker recorded one,
/// then a blank line, then the body **verbatim**. The body is what `read`, snippets and offsets all
/// resolve to, so this string is built for scoring only and never written anywhere.
///
/// Why the provenance helps: a page-per-chunk PDF reaches the scorer as bare prose with no hint of
/// which manual it belongs to. Measured over 175 cases — +2.3pp whole-chain coverage and +4.6pp
/// any-of at a window of 10, 5 cases gained against 1 lost. Controls settle the mechanism: a
/// constant word costs ~1pp, a name from an unrelated document costs the same, and a name from a
/// sibling document gains nearly as much as the true one. The model reads the prefix's TOPIC, so
/// this must be the hit's own path — a guessed or defaulted one is actively harmful.
///
/// `location` is whatever its extractor stored: a heading breadcrumb for Markdown and Office, a row
/// range for CSV, `(image)` for an image indexed by name. It is passed through unfiltered by
/// choice: an allow-list of "worthy" location shapes is a policy table with no owner, and every
/// extractor added later falls out of it silently. Length is the one thing that IS bounded, see
/// [`MAX_PROVENANCE_CHARS`].
pub fn rerank_passage(path: &str, location: &str, body: &str) -> String {
    let normalized = path.replace('\\', "/");
    let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    if let Some(file) = parts.pop() {
        // Drop a real extension only, and only when a name is left behind: `guide.pdf` loses
        // `.pdf`, while `report.v1_5` (underscore), `.gitignore` (too long), `file.` (nothing after
        // the dot) and `.png` (nothing before it) all keep their basename whole. Without the
        // `stem` check, a basename that is only an extension would collapse to an empty segment —
        // dropping the provenance entirely, or leaving a doubled separator before the location.
        let stem = file
            .rsplit_once('.')
            .filter(|(stem, ext)| {
                !stem.is_empty()
                    && !ext.is_empty()
                    && ext.len() <= 5
                    && ext.chars().all(|c| c.is_ascii_alphanumeric())
            })
            .map_or(file, |(stem, _)| stem);
        parts.push(stem);
    }
    let mut head = parts.join(" / ");
    if !location.is_empty() {
        if !head.is_empty() {
            head.push_str(" > ");
        }
        head.push_str(location);
    }
    if head.is_empty() {
        return body.to_string();
    }
    // The provenance must never crowd out the text it introduces. The scorer's window is fixed
    // (512 tokens) and truncation drops the END of the passage, while a heading breadcrumb has no
    // depth limit of its own — it is as deep as the document. Cutting here eats the deepest
    // headings first and always leaves the document path, the part the measurement credits.
    if let Some((cut, _)) = head.char_indices().nth(MAX_PROVENANCE_CHARS) {
        head.truncate(cut);
        head.push('…');
    }
    format!("{head}\n\n{body}")
}

/// Reorder `pool` by a fresh cross-encoder score and keep the top `top_n`. The returned hits carry
/// the RERANK score in `RankedHit.score` (so the trace + eval `ranked_sources` reflect rerank order —
/// see plan Global Constraints). Fail-open: any scorer error or a score/length mismatch returns the
/// BM25 order (trimmed to `top_n`) unchanged, together with a [`RerankOutcome::FailedOpen`] naming
/// the cause. Pure — logging the failure belongs to the caller (see [`retrieve_with`]).
pub fn rerank_hits(
    idx: &DocIndex,
    query: &str,
    pool: Vec<RankedHit>,
    rr: &dyn Reranker,
    top_n: usize,
) -> (Vec<RankedHit>, RerankOutcome) {
    // Chunk text from the index's stored body (no disk re-parse). An unresolvable chunk gets "" and
    // sinks; it never panics.
    let texts: Vec<String> = pool
        .iter()
        .map(|h| {
            idx.read_chunk_by_ord(&h.path, h.ord)
                .ok()
                .flatten()
                .map(|c| rerank_passage(&h.path, &h.location, &c.body))
                .unwrap_or_default()
        })
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let reason = match rr.rerank(query, &refs) {
        Ok(scores) if scores.len() == pool.len() && scores.iter().all(|s| s.is_finite()) => {
            let logits = rr.emits_logits();
            let mut order: Vec<usize> = (0..pool.len()).collect();
            // Descending by score; stable so equal scores keep BM25 order.
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let hits = order
                .into_iter()
                .take(top_n)
                .map(|i| {
                    let mut h = pool[i].clone();
                    h.score = scores[i]; // rerank logit becomes the reported score
                                         // rel_bm25 is NOT touched: it is the pool's reading and stays so.
                    h.rel_rerank = logits.then(|| sigmoid(scores[i]));
                    h
                })
                .collect();
            return (hits, RerankOutcome::Applied);
        }
        // A non-finite score would sort unpredictably (`total_cmp` orders NaN, but not usefully)
        // and would become a NaN probability. Fail open instead, naming the value.
        Ok(scores) if scores.len() == pool.len() => {
            let bad = scores
                .iter()
                .find(|s| !s.is_finite())
                .copied()
                .unwrap_or(f32::NAN);
            format!(
                "scorer returned a non-finite score ({bad}) for a pool of {}",
                pool.len()
            )
        }
        Ok(scores) => format!(
            "scorer returned {} score(s) for a pool of {}",
            scores.len(),
            pool.len()
        ),
        Err(e) => e.to_string(),
    };
    // Fail-open: BM25 order, with the cause carried out rather than dropped.
    let hits = pool.into_iter().take(top_n).collect();
    (hits, RerankOutcome::FailedOpen(reason))
}

/// Fetch + (optionally) rerank. With `reranker = None` this is `search_filtered(limit)` verbatim
/// (today's path). With a reranker, fetch `pool_size.max(limit)` candidates, rerank, and trim to
/// `limit`. Returns the hits plus a `RerankInfo` for the caller's trace.
#[allow(clippy::too_many_arguments)]
pub fn retrieve_with(
    idx: &DocIndex,
    query: &str,
    limit: usize,
    glob: Option<&str>,
    file_type: Option<&str>,
    scope: Option<&str>,
    reranker: Option<&dyn Reranker>,
    pool_size: usize,
) -> anyhow::Result<(Vec<RankedHit>, RerankInfo)> {
    match reranker {
        Some(rr) => {
            let pool =
                idx.search_filtered(query, pool_size.max(limit).max(1), glob, file_type, scope)?;
            let n = pool.len();
            let (hits, outcome) = rerank_hits(idx, query, pool, rr, limit.max(1));
            // A reranker that dies at runtime used to be invisible: the hits silently became BM25
            // while `reranked` still said true. Say it on stderr AND in the info, so neither a user
            // watching a session nor a measurement harness mistakes BM25 order for rerank order.
            let fallback = match outcome {
                RerankOutcome::Applied => None,
                RerankOutcome::FailedOpen(reason) => {
                    eprintln!("rerank failed, falling back to BM25 order: {reason}");
                    Some(reason)
                }
            };
            if fallback.is_none() && !rr.emits_logits() {
                // Once per process: the list IS in rerank order, but no probability travels with
                // it, so a consumer looking for `rel_rerank` knows why it is absent.
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    eprintln!(
                        "rerank: this backend returns a normalized score, not a logit, so hits carry no rel_rerank (order is unaffected)"
                    );
                });
            }
            Ok((
                hits,
                RerankInfo {
                    reranked: fallback.is_none(),
                    pool: n,
                    fallback,
                },
            ))
        }
        None => {
            let hits = idx.search_filtered(query, limit.max(1), glob, file_type, scope)?;
            let n = hits.len();
            Ok((
                hits,
                RerankInfo {
                    reranked: false,
                    pool: n,
                    fallback: None,
                },
            ))
        }
    }
}

/// Config-driven retrieval: resolves `[rerank]` from `glossa_dir` and either reranks the BM25 pool
/// or returns plain BM25 (the default). The single entry shared by `tools::search`, the eval search
/// arm, and the `kb search` CLI so all three agree on order.
pub fn retrieve(
    idx: &DocIndex,
    glossa_dir: &Path,
    query: &str,
    limit: usize,
    glob: Option<&str>,
    file_type: Option<&str>,
    scope: Option<&str>,
) -> anyhow::Result<(Vec<RankedHit>, RerankInfo)> {
    let cfg = RerankConfig::resolve(glossa_dir);
    let reranker = resolve_reranker(&cfg);
    retrieve_with(
        idx,
        query,
        limit,
        glob,
        file_type,
        scope,
        reranker.as_deref(),
        cfg.pool_size,
    )
}

/// Say when a compute knob was configured on a path that cannot honour it.
///
/// Batching and threading belong to the SERVER for every `scorer = "http"` backend, so a
/// client-side `batch_tokens` / `intra_threads` does nothing there. Dropping it in silence is how
/// the device keys on this path became a trap; this is one line on stderr instead.
///
/// Gated like its call sites: without `http-scorer` there is no remote path to warn about, and an
/// ungated definition is dead code in the no-default-features build.
#[cfg(feature = "http-scorer")]
fn warn_if_compute_knobs_inert(cfg: &RerankConfig) {
    if crate::retrieve::config::compute_knobs_inert(
        cfg.scorer.as_deref(),
        cfg.batch_tokens,
        cfg.intra_threads,
    ) {
        eprintln!(
            "[rerank] batch_tokens / intra_threads are in_process settings and do nothing with \
             scorer = \"http\" (backend {}); tune that server instead",
            cfg.backend
        );
    }
}

/// Build the in-process reranker from `[rerank]`, or `None` (=> plain BM25, fail-open). Any load
/// error (bad path, corrupt export, missing runtime) logs and downgrades to `None`.
#[cfg(all(
    any(feature = "nli", feature = "nli-dynamic"),
    not(feature = "nli-burn")
))]
pub fn resolve_reranker(cfg: &RerankConfig) -> Option<Box<dyn Reranker>> {
    #[cfg(feature = "http-scorer")]
    if cfg.enabled && cfg.scorer.as_deref() == Some("http") {
        warn_if_compute_knobs_inert(cfg);
        let endpoint = cfg.endpoint.clone()?;
        match crate::http_scorer::client::new_ureq_reranker(
            endpoint,
            cfg.timeout_ms,
            cfg.api_key.clone(),
            &cfg.backend,
            cfg.model.clone(),
        ) {
            Ok(r) => return Some(Box::new(r)),
            Err(e) => {
                eprintln!("remote rerank backend invalid: {e}");
                return None;
            }
        }
    }
    if !cfg.is_active() {
        return None;
    }
    let dir = cfg.model_dir.as_ref()?;
    match glossa_nli::InProcessReranker::load(
        dir,
        &cfg.execution_providers,
        cfg.ep_device,
        cfg.ep_mem_limit_mb,
        cfg.batch_tokens,
        cfg.intra_threads,
    ) {
        Ok(r) => Some(Box::new(r)),
        Err(e) => {
            eprintln!("rerank scorer load failed ({}): {e}", dir.display());
            None
        }
    }
}

/// burn/wgpu engine (mirrors `gate::resolve_scorer`'s burn arm): `[rerank]` active + `model_dir`
/// set => a real `InProcessBurnReranker`. Execution-provider config is ignored — the burn backend
/// selects its own device. Same fail-open contract: any load error downgrades to `None` (plain
/// BM25), never propagated.
#[cfg(feature = "nli-burn")]
pub fn resolve_reranker(cfg: &RerankConfig) -> Option<Box<dyn Reranker>> {
    #[cfg(feature = "http-scorer")]
    if cfg.enabled && cfg.scorer.as_deref() == Some("http") {
        warn_if_compute_knobs_inert(cfg);
        let endpoint = cfg.endpoint.clone()?;
        match crate::http_scorer::client::new_ureq_reranker(
            endpoint,
            cfg.timeout_ms,
            cfg.api_key.clone(),
            &cfg.backend,
            cfg.model.clone(),
        ) {
            Ok(r) => return Some(Box::new(r)),
            Err(e) => {
                eprintln!("remote rerank backend invalid: {e}");
                return None;
            }
        }
    }
    if !cfg.is_active() {
        return None;
    }
    let dir = cfg.model_dir.as_ref()?;
    // The burn engines take no budget/thread parameter yet, so a configured one does nothing here.
    // Say it: the remote path is not the only place these can be set and ignored, and silence in
    // the second place is the same trap as silence in the first.
    if crate::retrieve::config::any_compute_knob_set(cfg.batch_tokens, cfg.intra_threads) {
        eprintln!(
            "[rerank] batch_tokens / intra_threads are not read by the burn engine; \
             GLOSSA_NLI_BATCH_TOKENS is the only batch control it honours"
        );
    }
    match glossa_nli::InProcessBurnReranker::load(dir) {
        Ok(r) => Some(Box::new(r)),
        Err(e) => {
            eprintln!("rerank (burn) scorer load failed ({}): {e}", dir.display());
            None
        }
    }
}

/// No engine feature on => no in-process reranker even compiled; always plain BM25.
#[cfg(not(any(feature = "nli", feature = "nli-dynamic", feature = "nli-burn")))]
pub fn resolve_reranker(cfg: &RerankConfig) -> Option<Box<dyn Reranker>> {
    #[cfg(feature = "http-scorer")]
    if cfg.enabled && cfg.scorer.as_deref() == Some("http") {
        warn_if_compute_knobs_inert(cfg);
        let endpoint = cfg.endpoint.clone()?;
        match crate::http_scorer::client::new_ureq_reranker(
            endpoint,
            cfg.timeout_ms,
            cfg.api_key.clone(),
            &cfg.backend,
            cfg.model.clone(),
        ) {
            Ok(r) => return Some(Box::new(r)),
            Err(e) => {
                eprintln!("remote rerank backend invalid: {e}");
                return None;
            }
        }
    }
    #[cfg(not(feature = "http-scorer"))]
    let _ = cfg;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Chunk;
    use std::path::PathBuf;

    // Build a 3-chunk doc so search returns 3 hits with known ords (1=alpha, 2=beta, 3=gamma).
    fn idx_with_pages() -> (tempfile::TempDir, DocIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let sec = |loc: &str, t: &str| Chunk {
            doc_path: PathBuf::from("d.md"),
            location: loc.into(),
            file_type: "md".into(),
            text: t.into(),
        };
        idx.write_chunks(&[
            sec("A", "alpha swap"),
            sec("B", "beta swap"),
            sec("C", "gamma swap"),
        ])
        .unwrap();
        (dir, idx)
    }

    struct ByPath;
    impl Reranker for ByPath {
        fn rerank(&self, _q: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>> {
            // Score by a substring marker present in each passage. gamma > beta > alpha, and
            // alpha gets a negative logit to exercise sign handling in the ordering.
            Ok(passages
                .iter()
                .map(|p| {
                    if p.contains("gamma") {
                        3.0
                    } else if p.contains("beta") {
                        2.0
                    } else {
                        -1.0
                    }
                })
                .collect())
        }
    }

    // Scores a passage only when it carries the chunk's provenance ahead of the body, so this
    // double fails unless the prefix actually reaches the scorer.
    struct ByProvenance;
    impl Reranker for ByProvenance {
        fn rerank(&self, _q: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>> {
            Ok(passages
                .iter()
                .map(|p| if p.starts_with("d > C\n\n") { 5.0 } else { 1.0 })
                .collect())
        }
    }

    #[test]
    fn rerank_passage_leads_with_the_document_path() {
        assert_eq!(
            rerank_passage("manuals/plc/setup guide.pdf", "", "the body"),
            "manuals / plc / setup guide\n\nthe body"
        );
    }

    #[test]
    fn rerank_passage_normalizes_windows_separators() {
        assert_eq!(
            rerank_passage(r"manuals\plc\guide.pdf", "", "b"),
            "manuals / plc / guide\n\nb"
        );
    }

    #[test]
    fn rerank_passage_appends_whatever_location_the_chunker_recorded() {
        // The heading breadcrumb for Markdown/Office…
        assert_eq!(
            rerank_passage("notes/ops.md", "Networking > Gateways", "b"),
            "notes / ops > Networking > Gateways\n\nb"
        );
        // …and, deliberately unfiltered, every other extractor's own label. An allow-list of
        // "good" shapes would be a policy table that each new extractor silently falls out of.
        assert_eq!(
            rerank_passage("data/hosts.csv", "rows 201-300", "b"),
            "data / hosts > rows 201-300\n\nb"
        );
        assert_eq!(
            rerank_passage("diagrams/wiring.png", "(image)", "wiring"),
            "diagrams / wiring > (image)\n\nwiring"
        );
    }

    #[test]
    fn rerank_passage_keeps_the_body_verbatim() {
        let body = "  leading and trailing whitespace, and\na newline\n";
        let got = rerank_passage("a/b.md", "H", body);
        assert_eq!(got, format!("a / b > H\n\n{body}"));
        assert!(got.ends_with(body), "body was altered: {got:?}");
    }

    #[test]
    fn rerank_passage_strips_only_a_real_extension() {
        assert_eq!(rerank_passage("readme", "", "b"), "readme\n\nb");
        // `.v1_5` is not an extension (underscore), so the name stays whole.
        assert_eq!(rerank_passage("report.v1_5", "", "b"), "report.v1_5\n\nb");
    }

    #[test]
    fn rerank_passage_keeps_a_basename_that_is_only_an_extension() {
        // `.png` has no name in front of the dot, so stripping would leave an empty segment: the
        // provenance would vanish entirely, or double its separator once a folder is involved.
        assert_eq!(rerank_passage(".png", "", "b"), ".png\n\nb");
        assert_eq!(
            rerank_passage("diagrams/.png", "(image)", "b"),
            "diagrams / .png > (image)\n\nb"
        );
    }

    #[test]
    fn rerank_passage_caps_the_provenance_so_it_cannot_crowd_out_the_body() {
        // A breadcrumb is as deep as its document, and the scorer's window cuts the END of the
        // passage, so an uncapped head would truncate away the body it introduces. Multi-byte
        // headings also prove the cut lands on a character boundary rather than panicking.
        let deep = (1..40)
            .map(|i| format!("Überschrift {i}"))
            .collect::<Vec<_>>()
            .join(" > ");
        let got = rerank_passage("manuals/guide.md", &deep, "the body");
        assert!(
            got.starts_with("manuals / guide > Überschrift 1 > "),
            "document path lost: {got:?}"
        );
        assert!(got.ends_with("\n\nthe body"), "body lost: {got:?}");
        let head = got.split("\n\n").next().unwrap();
        assert!(
            head.chars().count() <= MAX_PROVENANCE_CHARS + 1,
            "head not capped: {} chars",
            head.chars().count()
        );
    }

    #[test]
    fn rerank_passage_without_a_path_is_the_body_alone() {
        assert_eq!(rerank_passage("", "", "just the body"), "just the body");
    }

    #[test]
    fn rerank_hits_scores_the_passage_with_its_provenance() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &ByProvenance, 10);
        assert_eq!(outcome, RerankOutcome::Applied);
        assert_eq!(
            out[0].location, "C",
            "the scorer never saw the provenance prefix"
        );
        assert_eq!(out[0].score, 5.0);
    }

    #[test]
    fn rerank_hits_reorders_by_descending_score() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &ByPath, 10);
        assert_eq!(outcome, RerankOutcome::Applied);
        let bodies: Vec<String> = out
            .iter()
            .map(|h| idx.read_chunk_by_ord(&h.path, h.ord).unwrap().unwrap().body)
            .collect();
        assert!(
            bodies[0].contains("gamma")
                && bodies[1].contains("beta")
                && bodies[2].contains("alpha"),
            "unexpected order: {bodies:?}"
        );
    }

    #[test]
    fn rerank_hits_carries_rerank_score_into_hit_score() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let (out, _) = rerank_hits(&idx, "swap", pool, &ByPath, 10);
        assert_eq!(out[0].score, 3.0); // top hit's score is the rerank logit, not BM25
    }

    #[test]
    fn rerank_hits_trims_to_top_n() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let (out, _) = rerank_hits(&idx, "swap", pool, &ByPath, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn rerank_hits_fails_open_on_length_mismatch() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let expected = pool.clone();
        // MockReranker returns too few scores -> fail open to BM25 order.
        let (out, outcome) =
            rerank_hits(&idx, "swap", pool, &MockReranker { scores: vec![1.0] }, 10);
        assert_eq!(
            out.iter().map(|h| h.ord).collect::<Vec<_>>(),
            expected.iter().map(|h| h.ord).collect::<Vec<_>>()
        );
        // The count mismatch names both numbers: "3 scores for a pool of 1" and its reverse are
        // different defects, and a reason that says only "mismatch" cannot tell them apart.
        let RerankOutcome::FailedOpen(reason) = outcome else {
            panic!("a short score vector must not report Applied");
        };
        assert!(
            reason.contains('1') && reason.contains('3'),
            "reason should carry both lengths: {reason}"
        );
    }

    struct FailingReranker;
    impl Reranker for FailingReranker {
        fn rerank(&self, _q: &str, _p: &[&str]) -> anyhow::Result<Vec<f32>> {
            anyhow::bail!("boom")
        }
    }

    #[test]
    fn rerank_hits_fails_open_on_scorer_error() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let expected = pool.clone();
        // FailingReranker always errors -> fail open to BM25 order.
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &FailingReranker, 10);
        assert_eq!(
            out.iter().map(|h| h.ord).collect::<Vec<_>>(),
            expected.iter().map(|h| h.ord).collect::<Vec<_>>()
        );
        assert_eq!(
            outcome,
            RerankOutcome::FailedOpen("boom".to_string()),
            "the scorer's own error text is what makes a runtime failure diagnosable"
        );
    }

    #[test]
    fn retrieve_with_reports_not_reranked_when_the_scorer_dies() {
        let (_d, idx) = idx_with_pages();
        // The defect this pins: a reranker that loads and then fails on every call (what the
        // DirectML build does) used to return `reranked: true` with BM25 hits, so a broken engine
        // reported success and a measurement harness read its control arm as its treatment arm.
        let (hits, info) = retrieve_with(
            &idx,
            "swap",
            2,
            None,
            None,
            None,
            Some(&FailingReranker),
            50,
        )
        .unwrap();
        let base = idx.search_filtered("swap", 2, None, None, None).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.ord).collect::<Vec<_>>(),
            base.iter().map(|h| h.ord).collect::<Vec<_>>(),
            "serving still fails open to BM25 — that part is deliberate"
        );
        assert!(
            !info.reranked,
            "a failed rerank must not claim to have ranked"
        );
        assert_eq!(info.fallback.as_deref(), Some("boom"));
    }

    #[test]
    fn retrieve_with_reports_reranked_and_no_fallback_when_the_scorer_works() {
        let (_d, idx) = idx_with_pages();
        let (_hits, info) =
            retrieve_with(&idx, "swap", 2, None, None, None, Some(&ByPath), 50).unwrap();
        assert!(info.reranked);
        assert_eq!(info.fallback, None);
    }

    #[test]
    fn retrieve_with_none_reranker_is_identity() {
        let (_d, idx) = idx_with_pages();
        let (out, info) = retrieve_with(&idx, "swap", 2, None, None, None, None, 50).unwrap();
        let base = idx.search_filtered("swap", 2, None, None, None).unwrap();
        assert_eq!(
            out.iter().map(|h| h.ord).collect::<Vec<_>>(),
            base.iter().map(|h| h.ord).collect::<Vec<_>>()
        );
        assert!(!info.reranked);
    }

    #[test]
    fn retrieve_with_pool_smaller_than_limit_still_fills_limit() {
        let (_d, idx) = idx_with_pages();
        // pool_size 1 but limit 3 -> fetch max(1,3)=3, still return up to 3.
        let (out, _info) =
            retrieve_with(&idx, "swap", 3, None, None, None, Some(&ByPath), 1).unwrap();
        assert_eq!(out.len(), 3);
    }

    // Runs identically with the ORT engine feature ON or OFF: with it off `resolve_reranker` is
    // the always-`None` stub; with it on, `cfg.is_active()` hits the function's own early return
    // (an inactive config is off by default). Either way this proves the fail-open default.
    #[test]
    fn resolve_reranker_none_when_inactive() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::retrieve::config::RerankConfig::resolve(dir.path()); // off by default
        assert!(resolve_reranker(&cfg).is_none());
    }

    // Same as above but named for the burn build (runs identically under `--features nli-burn`,
    // where `resolve_reranker` is the burn arm — its own `cfg.is_active()` early return still
    // fires for an inactive config).
    #[test]
    fn resolve_reranker_none_when_inactive_burn_build() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::retrieve::config::RerankConfig::resolve(dir.path());
        assert!(resolve_reranker(&cfg).is_none());
    }

    #[cfg(feature = "http-scorer")]
    #[test]
    fn resolve_reranker_builds_http_when_enabled_http_and_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"http\"\nendpoint=\"http://gpu:8080\"\n",
        )
        .unwrap();
        let cfg = crate::retrieve::config::RerankConfig::resolve(&g);
        assert!(resolve_reranker(&cfg).is_some());
    }

    #[cfg(feature = "http-scorer")]
    #[test]
    fn resolve_reranker_fails_open_to_none_on_unknown_backend() {
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        // http + endpoint set, but a typo'd backend -> invalid wire shape -> fail-open to BM25.
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"http\"\nendpoint=\"http://gpu:8080\"\n\
             backend=\"vlm\"\n",
        )
        .unwrap();
        let cfg = crate::retrieve::config::RerankConfig::resolve(&g);
        assert!(resolve_reranker(&cfg).is_none());
    }

    #[test]
    fn retrieve_without_config_matches_search_filtered() {
        let (_d, idx) = idx_with_pages();
        let dir = tempfile::tempdir().unwrap(); // no [rerank] ontology -> inactive
        let (out, info) = retrieve(&idx, dir.path(), "swap", 2, None, None, None).unwrap();
        let base = idx.search_filtered("swap", 2, None, None, None).unwrap();
        assert_eq!(
            out.iter().map(|h| h.ord).collect::<Vec<_>>(),
            base.iter().map(|h| h.ord).collect::<Vec<_>>()
        );
        assert!(!info.reranked);
    }

    #[test]
    fn resolve_reranker_none_when_enabled_but_model_dir_missing() {
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join(".glossa");
        std::fs::create_dir_all(&g).unwrap();
        // enabled + in_process but points at a nonexistent model dir -> load fails -> None
        // (fail-open).
        std::fs::write(
            g.join("ontology.toml"),
            "[rerank]\nenabled=true\nscorer=\"in_process\"\nmodel_dir=\"/no/such/dir\"\n",
        )
        .unwrap();
        let cfg = crate::retrieve::config::RerankConfig::resolve(&g);
        assert!(resolve_reranker(&cfg).is_none());
    }

    /// The reranker's probability rides on every reranked hit; the native score and the order are
    /// untouched, and the BM25 reading the pool carried in survives the rerank unchanged.
    #[test]
    fn rerank_sets_rel_rerank_and_keeps_rel_bm25_from_the_pool() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let bm25_by_ord: std::collections::HashMap<u64, f32> =
            pool.iter().map(|h| (h.ord, h.rel_bm25)).collect();
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &ByPath, 10);
        assert_eq!(outcome, RerankOutcome::Applied);
        for h in &out {
            let p = h.rel_rerank.expect("reranked by a logit scorer");
            assert!((0.0..=1.0).contains(&p), "{p}");
            assert_eq!(
                p,
                sigmoid(h.score),
                "rel_rerank is sigmoid of the native logit"
            );
            assert_eq!(
                h.rel_bm25, bm25_by_ord[&h.ord],
                "pool value kept, not recomputed"
            );
        }
        // Same order as sorting by the native score (sigmoid is monotone).
        let scores: Vec<f32> = out.iter().map(|h| h.score).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
    }

    #[test]
    fn sigmoid_is_a_probability_that_preserves_order() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(7.35) - 0.9994).abs() < 1e-3, "{}", sigmoid(7.35));
        assert!(sigmoid(-11.04) < 2e-5, "{}", sigmoid(-11.04));
        assert_eq!(sigmoid(20.0), 1.0, "f32 saturates to exactly 1 above ~17");
        // The other end is NOT symmetric: exactly 0.0 needs e^-x to overflow f32 (x <~ -88).
        assert!(
            sigmoid(-20.0) > 0.0 && sigmoid(-20.0) < 1e-8,
            "{}",
            sigmoid(-20.0)
        );
        assert_eq!(
            sigmoid(-100.0),
            0.0,
            "f32 reaches exactly 0 only past the exp overflow"
        );
        assert!(sigmoid(1.0) > sigmoid(0.9));
    }

    /// A scorer whose numbers are not logits (the hosted Cohere/Jina APIs) still orders the pool,
    /// but no probability is invented for it.
    struct NoLogits;
    impl Reranker for NoLogits {
        fn rerank(&self, _q: &str, p: &[&str]) -> anyhow::Result<Vec<f32>> {
            Ok((0..p.len()).map(|i| 0.9 - 0.1 * i as f32).collect())
        }
        fn emits_logits(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_non_logit_scorer_orders_but_leaves_rel_rerank_none() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &NoLogits, 10);
        assert_eq!(outcome, RerankOutcome::Applied);
        assert!(out.iter().all(|h| h.rel_rerank.is_none()), "{out:?}");
        assert!(out.iter().all(|h| h.rel_bm25 > 0.0));
    }

    /// Fail-open keeps the BM25 pool's readings and sets nothing.
    #[test]
    fn fail_open_keeps_rel_bm25_and_sets_no_rel_rerank() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let expected: Vec<(u64, f32)> = pool.iter().map(|h| (h.ord, h.rel_bm25)).collect();
        let short = MockReranker { scores: vec![1.0] }; // wrong count -> fail-open
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &short, 10);
        assert!(matches!(outcome, RerankOutcome::FailedOpen(_)));
        assert_eq!(
            out.iter().map(|h| (h.ord, h.rel_bm25)).collect::<Vec<_>>(),
            expected
        );
        assert!(out.iter().all(|h| h.rel_rerank.is_none()));
    }

    /// A NaN must never reach the sort or become a probability.
    #[test]
    fn a_non_finite_score_fails_the_rerank_open() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let n = pool.len();
        assert!(n >= 2, "fixture must give a pool to corrupt");
        let mut scores = vec![1.0; n];
        scores[0] = f32::NAN;
        let bad = MockReranker { scores };
        let expected_ords: Vec<u64> = pool.iter().map(|h| h.ord).collect();
        let (out, outcome) = rerank_hits(&idx, "swap", pool, &bad, 10);
        match outcome {
            RerankOutcome::FailedOpen(reason) => {
                assert!(reason.contains("non-finite"), "{reason}")
            }
            other => panic!("expected fail-open, got {other:?}"),
        }
        assert_eq!(
            out.iter().map(|h| h.ord).collect::<Vec<_>>(),
            expected_ords,
            "BM25 order kept"
        );
        assert!(out.iter().all(|h| h.rel_rerank.is_none()));
    }
}
