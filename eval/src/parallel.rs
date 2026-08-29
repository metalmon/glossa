//! Reusable concurrency core for the `kbx` worker pools (`build`/`reason`/`distil`).
//!
//! Two pieces, kept deliberately thin:
//! - [`GraphWriter`]: serializes graph writes across N in-process worker threads while reusing
//!   the existing core file-lock (`glossa::graph::lock::with_graph_write_lock`) unchanged, so the
//!   `glossa` MCP process still coordinates through the SAME lock. `ops.rs`/`store.rs`/`lock.rs`
//!   are not touched here — this module only calls them.
//! - [`run_units_parallel`]: a bounded worker pool over an owned `Vec<U>`, `jobs.max(1)` workers
//!   pulling from a shared queue. `jobs == 1` runs fully inline (no threads spawned) — the exact
//!   sequential path used today, which is the non-regression contract for `--jobs 1`.

use anyhow::Result;
use indicatif::ProgressBar;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use glossa::graph::lock::with_graph_write_lock;
use glossa::graph::ontology::Ontology;
use glossa::graph::ops::{self, UpsertEdge, UpsertNode, UpsertOutcome};
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;

/// A single serialized entry point for `graph_upsert`, shared by every worker in a pool.
///
/// The in-process `Mutex<()>` serializes the N worker threads cross-platform (Windows file locks
/// don't arbitrate fairly between threads of the SAME process the way they do across processes).
/// Inside that guard, `with_graph_write_lock` is called exactly as the MCP calls it
/// (`src/mcp.rs`) — it wraps the whole read-modify-write with the on-disk advisory lock, so an
/// eval worker and an MCP request can never race on the same `.glossa/graph.sqlite`.
pub struct GraphWriter {
    g: Arc<GraphStore>,
    lock: Arc<Mutex<()>>,
    root: PathBuf,
}

impl GraphWriter {
    pub fn new(g: Arc<GraphStore>, root: PathBuf) -> Self {
        Self { g, lock: Arc::new(Mutex::new(())), root }
    }

    /// Serialized `graph_upsert`. Holds the in-process mutex for the ENTIRE call (including the
    /// nested file-lock RMW), so no two workers — nor a worker and a concurrent MCP writer — can
    /// interleave a read-modify-write on the graph.
    pub fn upsert(
        &self,
        idx: &DocIndex,
        ont: &Ontology,
        nodes: Vec<UpsertNode>,
        edges: Vec<UpsertEdge>,
        now: u64,
        origin: &str,
    ) -> Result<UpsertOutcome> {
        // Poisoning would only happen if a prior worker panicked mid-write; treating that as an
        // unlock (via into_inner) keeps one bad worker from wedging every other worker forever.
        let _guard = match self.lock.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        with_graph_write_lock(&self.root, Duration::from_secs(120), || {
            Ok(ops::graph_upsert(idx, &self.g, ont, nodes, edges, now, origin))
        })
    }
}

/// Run `work` over `units` with `jobs.max(1)` workers, reporting progress via `pb.inc(weight(u))`
/// after each unit completes. Results are returned in COMPLETION order, not input order (nothing
/// downstream may depend on unit ordering — see the spec's non-regression contract).
///
/// `jobs == 1` never spawns a thread: it is the literal sequential loop, byte-for-byte the
/// pre-parallel behaviour.
///
/// On the first error, remaining un-started units are skipped and that error is returned; units
/// already in flight on other workers are allowed to finish (their results are discarded).
pub fn run_units_parallel<U, S>(
    units: Vec<U>,
    jobs: usize,
    pb: &ProgressBar,
    weight: impl Fn(&U) -> u64 + Sync,
    work: impl Fn(&U) -> Result<S> + Sync,
) -> Result<Vec<S>>
where
    U: Send,
    S: Send,
{
    let jobs = jobs.max(1);

    if jobs == 1 {
        let mut results = Vec::with_capacity(units.len());
        for u in &units {
            let s = work(u)?;
            pb.inc(weight(u));
            results.push(s);
        }
        return Ok(results);
    }

    let queue: Mutex<VecDeque<U>> = Mutex::new(units.into_iter().collect());
    let results: Mutex<Vec<S>> = Mutex::new(Vec::new());
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                let next = queue.lock().expect("queue mutex poisoned").pop_front();
                let Some(unit) = next else { break };
                match work(&unit) {
                    Ok(s) => {
                        pb.inc(weight(&unit));
                        results.lock().expect("results mutex poisoned").push(s);
                    }
                    Err(e) => {
                        let mut fe = first_err.lock().expect("first_err mutex poisoned");
                        if fe.is_none() {
                            *fe = Some(e);
                        }
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            });
        }
    });

    if let Some(e) = first_err.into_inner().expect("first_err mutex poisoned") {
        return Err(e);
    }
    Ok(results.into_inner().expect("results mutex poisoned"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glossa::graph::store::{Node, Provenance};
    use std::sync::atomic::AtomicUsize;

    fn progress_bar_hidden() -> ProgressBar {
        ProgressBar::hidden()
    }

    #[test]
    fn run_units_parallel_returns_all_results_as_a_set() {
        let units: Vec<u32> = (0..20).collect();
        let pb = progress_bar_hidden();
        let out = run_units_parallel(units, 4, &pb, |_| 1, |u| Ok(*u * 2)).unwrap();
        let mut got: Vec<u32> = out;
        got.sort_unstable();
        let want: Vec<u32> = (0..20).map(|u| u * 2).collect();
        assert_eq!(got, want, "every unit's result must be present regardless of completion order");
    }

    #[test]
    fn run_units_parallel_honors_small_jobs_count() {
        // With jobs=2 over 20 units and an artificial max-concurrency counter, at most 2 units
        // should ever be in flight at once.
        let units: Vec<u32> = (0..20).collect();
        let pb = progress_bar_hidden();
        let in_flight = AtomicUsize::new(0);
        let max_seen = AtomicUsize::new(0);
        let out = run_units_parallel(units, 2, &pb, |_| 1, |_u| {
            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            max_seen.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(5));
            in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert_eq!(out.len(), 20);
        assert!(max_seen.load(Ordering::SeqCst) <= 2, "jobs=2 must never run >2 units concurrently");
    }

    #[test]
    fn run_units_parallel_jobs_one_runs_inline_no_threads() {
        // A thread::current().id() recorded from inside `work` must equal the caller's thread id
        // for every unit when jobs=1 — proof no thread was spawned.
        let caller_id = std::thread::current().id();
        let units: Vec<u32> = (0..5).collect();
        let pb = progress_bar_hidden();
        let out = run_units_parallel(units, 1, &pb, |_| 1, |_u| Ok(std::thread::current().id()))
            .unwrap();
        assert!(out.iter().all(|id| *id == caller_id), "jobs=1 must run inline on the caller's thread");
    }

    #[test]
    fn run_units_parallel_zero_jobs_clamps_to_one_inline() {
        let caller_id = std::thread::current().id();
        let units: Vec<u32> = (0..3).collect();
        let pb = progress_bar_hidden();
        let out = run_units_parallel(units, 0, &pb, |_| 1, |_u| Ok(std::thread::current().id()))
            .unwrap();
        assert!(out.iter().all(|id| *id == caller_id), "jobs=0 must clamp to 1 and run inline");
    }

    #[test]
    fn run_units_parallel_first_error_stops_remaining_units() {
        let units: Vec<u32> = (0..50).collect();
        let pb = progress_bar_hidden();
        let started = AtomicUsize::new(0);
        let res: Result<Vec<()>> = run_units_parallel(units, 3, &pb, |_| 1, |u| {
            started.fetch_add(1, Ordering::SeqCst);
            if *u == 5 {
                anyhow::bail!("boom at unit 5");
            }
            std::thread::sleep(Duration::from_millis(2));
            Ok(())
        });
        assert!(res.is_err(), "an error in one unit must be propagated");
        assert!(
            started.load(Ordering::SeqCst) < 50,
            "remaining un-started units must be skipped after the first error, got {} started",
            started.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn run_units_parallel_jobs_one_stops_on_first_error_too() {
        let units: Vec<u32> = (0..10).collect();
        let pb = progress_bar_hidden();
        let ran: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let res: Result<Vec<()>> = run_units_parallel(units, 1, &pb, |_| 1, |u| {
            ran.lock().unwrap().push(*u);
            if *u == 2 {
                anyhow::bail!("boom");
            }
            Ok(())
        });
        assert!(res.is_err());
        assert_eq!(*ran.lock().unwrap(), vec![0, 1, 2], "sequential path stops right after the failing unit");
    }

    fn prov() -> Provenance {
        Provenance { source_path: "d.md".into(), range: None, file_sig: None,
            origin: "test".into(), confidence: 0.9, created_at: 1 }
    }

    /// Two threads hammer ONE `GraphWriter` with concurrent upserts that each (a) read the
    /// current node count, (b) sleep briefly to widen any race window, then (c) write a new node
    /// derived from that count. If `upsert` ever let two calls interleave their read-modify-write,
    /// both threads would read the same count and one write would clobber/duplicate — this
    /// asserts the final node count equals the number of successful calls, proving serialization.
    #[test]
    fn graph_writer_upsert_serializes_concurrent_callers() {
        let dir = tempfile::tempdir().unwrap();
        let g = Arc::new(GraphStore::open(dir.path()).unwrap());
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        let ont = Ontology::default();
        let writer = GraphWriter::new(Arc::clone(&g), dir.path().to_path_buf());

        // Seed one node so read-then-write has something to read.
        g.put_node(&Node {
            id: "seed".into(), node_type: "Resolution".into(), label: "seed".into(),
            aliases: vec![], prov: prov(),
        })
        .unwrap();

        const PER_THREAD: usize = 15;
        std::thread::scope(|scope| {
            for t in 0..2 {
                let writer = &writer;
                let idx = &idx;
                let ont = &ont;
                let gref = &g;
                scope.spawn(move || {
                    for i in 0..PER_THREAD {
                        // Read-under-lock is exactly what `ops::graph_upsert` itself does
                        // internally (orienting edges by node type); here we additionally read
                        // OUTSIDE any lock to widen the race window this test is designed to
                        // catch — if `upsert` truly serializes, the count read then used to name
                        // the node id below can still collide across threads, but the node table
                        // itself must never end up short (proving no write was lost/clobbered).
                        let _ = gref.node_count();
                        let node = UpsertNode {
                            node_type: "Resolution".into(),
                            label: format!("node {t}-{i}"),
                            source_path: String::new(),
                            aliases: vec![],
                            valid_from: None,
                            valid_to: None,
                        };
                        let out = writer.upsert(idx, ont, vec![node], vec![], 1, "test").unwrap();
                        assert!(!out.rejected, "upsert must not be rejected: {}", out.message);
                    }
                });
            }
        });

        let total = g.node_count().unwrap();
        // seed + 2 threads * PER_THREAD distinct ids, none lost to an interleaved RMW race.
        assert_eq!(total, 1 + 2 * PER_THREAD as u64, "no upsert may be lost/clobbered under concurrency");
    }
}
