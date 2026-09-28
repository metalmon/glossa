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
    pub reranked: bool,
    pub pool: usize,
}

/// Reorder `pool` by a fresh cross-encoder score and keep the top `top_n`. The returned hits carry
/// the RERANK score in `RankedHit.score` (so the trace + eval `ranked_sources` reflect rerank order —
/// see plan Global Constraints). Fail-open: any scorer error or a score/length mismatch returns the
/// BM25 order (trimmed to `top_n`) unchanged.
pub fn rerank_hits(
    idx: &DocIndex,
    query: &str,
    pool: Vec<RankedHit>,
    rr: &dyn Reranker,
    top_n: usize,
) -> Vec<RankedHit> {
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
    match rr.rerank(query, &refs) {
        Ok(scores) if scores.len() == pool.len() => {
            let mut order: Vec<usize> = (0..pool.len()).collect();
            // Descending by score; stable so equal scores keep BM25 order.
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            order
                .into_iter()
                .take(top_n)
                .map(|i| {
                    let mut h = pool[i].clone();
                    h.score = scores[i]; // rerank logit becomes the reported score
                    h
                })
                .collect()
        }
        _ => pool.into_iter().take(top_n).collect(), // fail-open: BM25 order
    }
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
            let hits = rerank_hits(idx, query, pool, rr, limit.max(1));
            Ok((
                hits,
                RerankInfo {
                    reranked: true,
                    pool: n,
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

/// Build the in-process reranker from `[rerank]`, or `None` (=> plain BM25, fail-open). Any load
/// error (bad path, corrupt export, missing runtime) logs and downgrades to `None`.
#[cfg(all(
    any(feature = "nli", feature = "nli-dynamic"),
    not(feature = "nli-burn")
))]
pub fn resolve_reranker(cfg: &RerankConfig) -> Option<Box<dyn Reranker>> {
    if !cfg.is_active() {
        return None;
    }
    let dir = cfg.model_dir.as_ref()?;
    match glossa_nli::InProcessReranker::load(
        dir,
        &cfg.execution_providers,
        cfg.ep_device,
        cfg.ep_mem_limit_mb,
    ) {
        Ok(r) => Some(Box::new(r)),
        Err(e) => {
            eprintln!("rerank scorer load failed ({}): {e}", dir.display());
            None
        }
    }
}

/// No ORT reranker compiled (burn-only, or no engine feature) => always plain BM25.
#[cfg(not(all(
    any(feature = "nli", feature = "nli-dynamic"),
    not(feature = "nli-burn")
)))]
pub fn resolve_reranker(_cfg: &RerankConfig) -> Option<Box<dyn Reranker>> {
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
        let out = rerank_hits(&idx, "swap", pool, &ByPath, 10);
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
        let out = rerank_hits(&idx, "swap", pool, &ByPath, 10);
        assert_eq!(out[0].score, 3.0); // top hit's score is the rerank logit, not BM25
    }

    #[test]
    fn rerank_hits_trims_to_top_n() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let out = rerank_hits(&idx, "swap", pool, &ByPath, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn rerank_hits_fails_open_on_length_mismatch() {
        let (_d, idx) = idx_with_pages();
        let pool = idx.search_filtered("swap", 10, None, None, None).unwrap();
        let expected = pool.clone();
        // MockReranker returns too few scores -> fail open to BM25 order.
        let out = rerank_hits(&idx, "swap", pool, &MockReranker { scores: vec![1.0] }, 10);
        assert_eq!(
            out.iter().map(|h| h.ord).collect::<Vec<_>>(),
            expected.iter().map(|h| h.ord).collect::<Vec<_>>()
        );
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
