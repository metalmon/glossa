use crate::index::store::{DocIndex, RankedHit};
use crate::retrieve::config::RerankConfig;
use std::path::Path;

/// Backend-agnostic cross-encoder scorer: one relevance score per passage, same length + order as
/// `passages`. Higher = more relevant. Implemented in-process (`retrieve::rerank_engine`) and, later,
/// over HTTP.
pub trait Reranker {
    fn rerank(&self, query: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>>;
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
                .map(|c| c.body)
                .unwrap_or_default()
        })
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let reason = match rr.rerank(query, &refs) {
        Ok(scores) if scores.len() == pool.len() => {
            let mut order: Vec<usize> = (0..pool.len()).collect();
            // Descending by score; stable so equal scores keep BM25 order.
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let hits = order
                .into_iter()
                .take(top_n)
                .map(|i| {
                    let mut h = pool[i].clone();
                    h.score = scores[i]; // rerank logit becomes the reported score
                    h
                })
                .collect();
            return (hits, RerankOutcome::Applied);
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
}
