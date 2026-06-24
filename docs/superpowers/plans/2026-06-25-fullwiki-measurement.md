# Fullwiki Measurement Implementation Plan (Plan 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure glossa's retrieval at Wikipedia scale — convert the HotpotQA abstracts corpus into a glossa index and score Recall@k of gold supporting-fact titles.

**Architecture:** A `kb-eval prep-fullwiki` command streams the doubly-bzip2'd `tar.bz2` abstracts archive into one markdown file per inner shard (each article a `# Title` section → a glossa chunk whose `location` is the title). A `kb-eval run --fullwiki <corpus>` mode searches that pre-built shared index instead of building a per-question corpus, and Recall@5/@10/@20 + MRR are computed from the per-question tool-call transcript (gold titles vs ranked hit `location`s).

**Tech Stack:** Rust (the dev-only `eval` crate — C deps allowed here, unlike `glossa`), `bzip2` + `tar` for the archive, existing `glossa` index/trace.

## Global Constraints

- This is **Plan 1 only** (spec components 1–3). The TensorZero `tensorzero` backend is Plan 2 — do not build it here.
- The `eval` crate is dev-only; **C-backed deps (`bzip2`, `tar`) are fine** — the C-free invariant binds only `glossa`, which this plan does not touch.
- **Qwen-only arm** for fullwiki (the `openai` backend) — but the code here is backend-agnostic; no backend changes.
- File-First / never-truncate is not relevant here (we generate the corpus).
- TDD: failing test first, minimal code, pass, commit. Run tests with `cargo test -p kb-eval`.
- Markdown shard format: each article rendered as `# <title>\n<intro>\n` so glossa's markdown extractor makes one chunk per article with `location == title`.

---

## Feasibility spike (run after Task 1, before the full build)

Once Task 1 lands, run the single-shard spike to size the full build:
```
kb-eval prep-fullwiki --archive eval-data/enwiki-abstracts.tar.bz2 --out wiki-corpus --max-shards 1
kb index wiki-corpus           # time this
```
Record: articles in 1 shard, total shard count (printed by prep when run without `--max-shards`, or estimate ~tens of thousands), the index wall-time and `wiki-corpus/.glossa` size. Extrapolate to ~5M articles. If the projected full index time is many hours, raise it with the operator before the overnight build (option: parallelize the indexer — a separate ROADMAP task). This is an operational measurement, not a code task.

---

### Task 1: `prep-fullwiki` — abstracts archive → markdown shards

**Files:**
- Modify: `eval/Cargo.toml` (add `bzip2`, `tar`)
- Create: `eval/src/prep.rs`
- Modify: `eval/src/main.rs` (declare `mod prep;`, add the `PrepFullwiki` subcommand)

**Interfaces:**
- Produces: `prep::prep_fullwiki(archive: &Path, out: &Path, max_shards: Option<usize>) -> anyhow::Result<prep::PrepStats>` where `pub struct PrepStats { pub shards: usize, pub articles: usize }`.

- [ ] **Step 1: Add dependencies**

In `eval/Cargo.toml` under `[dependencies]` add:
```toml
bzip2 = "0.4"
tar = "0.4"
```

- [ ] **Step 2: Declare the module**

In `eval/src/main.rs`, add to the `mod` list (after `mod corpus;`):
```rust
mod prep;
```

- [ ] **Step 3: Write the failing test + implementation**

Create `eval/src/prep.rs`:
```rust
use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tar::Archive;

pub struct PrepStats {
    pub shards: usize,
    pub articles: usize,
}

#[derive(serde::Deserialize)]
struct WikiArticle {
    title: String,
    text: serde_json::Value, // list of sentences, or list of paragraphs (list of lists)
}

/// Convert the HotpotQA abstracts `tar.bz2` into one markdown file per inner `.bz2` shard.
/// Each article becomes a `# <title>` section followed by its intro text, so glossa's markdown
/// extractor yields one chunk per article with `location == title`.
pub fn prep_fullwiki(archive: &Path, out: &Path, max_shards: Option<usize>) -> Result<PrepStats> {
    let file = File::open(archive).with_context(|| format!("open {archive:?}"))?;
    let mut ar = Archive::new(BzDecoder::new(BufReader::new(file))); // outer bz2 -> tar
    fs::create_dir_all(out)?;
    let mut stats = PrepStats { shards: 0, articles: 0 };

    for entry in ar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        if path.extension().and_then(|e| e.to_str()) != Some("bz2") {
            continue; // skip directories / non-shard entries
        }
        if let Some(max) = max_shards {
            if stats.shards >= max {
                break;
            }
        }
        let stem = sanitize_shard_name(&path);
        let md_path = out.join(format!("{stem}.md"));
        let mut w = std::io::BufWriter::new(File::create(&md_path)?);
        let reader = BufReader::new(BzDecoder::new(&mut entry)); // inner bz2 -> JSON lines
        let mut shard_articles = 0usize;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let art: WikiArticle = match serde_json::from_str(&line) {
                Ok(a) => a,
                Err(_) => continue, // skip malformed line, keep going
            };
            let title = art.title.replace(['\n', '\r'], " ");
            let intro = flatten_text(&art.text);
            if title.trim().is_empty() || intro.trim().is_empty() {
                continue;
            }
            writeln!(w, "# {title}")?;
            writeln!(w, "{intro}\n")?;
            shard_articles += 1;
        }
        w.flush()?;
        stats.shards += 1;
        stats.articles += shard_articles;
    }
    Ok(stats)
}

/// `enwiki/AA/wiki_00.bz2` -> `AA_wiki_00`.
fn sanitize_shard_name(path: &Path) -> String {
    let comps: Vec<String> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_string()))
        .collect();
    let n = comps.len();
    let dir = if n >= 2 { comps[n - 2].as_str() } else { "x" };
    let file = comps.last().map(|s| s.trim_end_matches(".bz2")).unwrap_or("shard");
    format!("{dir}_{file}")
}

/// HotpotQA `text` is a list of sentence strings, or a list of paragraphs (each a list of
/// sentences). Flatten to one string.
fn flatten_text(v: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(arr) = v.as_array() {
        for item in arr {
            match item {
                serde_json::Value::String(s) => {
                    out.push_str(s);
                    out.push(' ');
                }
                serde_json::Value::Array(inner) => {
                    for s in inner {
                        if let Some(s) = s.as_str() {
                            out.push_str(s);
                            out.push(' ');
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use std::io::Write as _;

    fn bz(data: &[u8]) -> Vec<u8> {
        let mut e = BzEncoder::new(Vec::new(), Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn converts_nested_archive_to_md_sections() {
        let dir = tempfile::tempdir().unwrap();
        // inner shard: two JSON-line articles (one flat text, one nested paragraphs)
        let jsonl = concat!(
            r#"{"title":"Alpha","text":["A1.","A2."]}"#, "\n",
            r#"{"title":"Beta","text":[["B1.","B2."]]}"#, "\n",
        );
        let inner_bz = bz(jsonl.as_bytes());
        // tar containing enwiki/AA/wiki_00.bz2
        let mut tar_buf = Vec::new();
        {
            let mut tb = tar::Builder::new(&mut tar_buf);
            let mut hdr = tar::Header::new_gnu();
            hdr.set_size(inner_bz.len() as u64);
            hdr.set_mode(0o644);
            hdr.set_cksum();
            tb.append_data(&mut hdr, "enwiki/AA/wiki_00.bz2", &inner_bz[..]).unwrap();
            tb.finish().unwrap();
        }
        let archive = dir.path().join("abstracts.tar.bz2");
        std::fs::write(&archive, bz(&tar_buf)).unwrap();

        let out = dir.path().join("corpus");
        let stats = prep_fullwiki(&archive, &out, None).unwrap();
        assert_eq!(stats.shards, 1);
        assert_eq!(stats.articles, 2);

        let md = std::fs::read_to_string(out.join("AA_wiki_00.md")).unwrap();
        assert!(md.contains("# Alpha") && md.contains("A1. A2."));
        assert!(md.contains("# Beta") && md.contains("B1. B2."));
    }

    #[test]
    fn max_shards_limits_output() {
        let dir = tempfile::tempdir().unwrap();
        let inner_bz = bz(br#"{"title":"X","text":["x."]}"#);
        let mut tar_buf = Vec::new();
        {
            let mut tb = tar::Builder::new(&mut tar_buf);
            for name in ["enwiki/AA/wiki_00.bz2", "enwiki/AA/wiki_01.bz2"] {
                let mut hdr = tar::Header::new_gnu();
                hdr.set_size(inner_bz.len() as u64);
                hdr.set_mode(0o644);
                hdr.set_cksum();
                tb.append_data(&mut hdr, name, &inner_bz[..]).unwrap();
            }
            tb.finish().unwrap();
        }
        let archive = dir.path().join("a.tar.bz2");
        std::fs::write(&archive, bz(&tar_buf)).unwrap();
        let out = dir.path().join("c");
        let stats = prep_fullwiki(&archive, &out, Some(1)).unwrap();
        assert_eq!(stats.shards, 1, "max_shards=1 stops after one shard");
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p kb-eval --lib prep`
Expected: FAIL before the file exists, PASS after (2 tests).

- [ ] **Step 5: Add the `PrepFullwiki` subcommand**

In `eval/src/main.rs`, add a variant to `enum Cmd` (after `Run { ... }`):
```rust
    /// Convert the HotpotQA abstracts tar.bz2 into a glossa-indexable markdown corpus.
    PrepFullwiki {
        /// Path to the `...-abstracts.tar.bz2` archive.
        #[arg(long)]
        archive: PathBuf,
        /// Output directory for the markdown shards.
        #[arg(long)]
        out: PathBuf,
        /// Only convert the first N shards (feasibility spike).
        #[arg(long)]
        max_shards: Option<usize>,
    },
```
And add a match arm in `main()` (alongside `Cmd::Run`):
```rust
        Cmd::PrepFullwiki { archive, out, max_shards } => {
            let stats = prep::prep_fullwiki(&archive, &out, max_shards)?;
            println!("prep-fullwiki: {} shard(s), {} article(s) -> {}", stats.shards, stats.articles, out.display());
            Ok(())
        }
```

- [ ] **Step 6: Verify the binary builds + the suite passes**

Run: `cargo test -p kb-eval`
Expected: PASS (prep tests + existing tests). `cargo build -p kb-eval --release` succeeds.

- [ ] **Step 7: Commit**
```bash
git add eval/Cargo.toml eval/src/prep.rs eval/src/main.rs
git commit -m "feat(eval): prep-fullwiki — abstracts tar.bz2 -> markdown shards"
```

---

### Task 2: Recall@k + MRR scoring

**Files:**
- Modify: `eval/src/score.rs`

**Interfaces:**
- Consumes: `glossa::trace::TraceEntry`, existing `score::normalize`.
- Produces: `score::ranked_titles(transcript: &[glossa::trace::TraceEntry]) -> Vec<String>`, `score::recall_at_k(ranked: &[String], gold: &[String], k: usize) -> f32`, `score::mrr(ranked: &[String], gold: &[String]) -> f32`.

- [ ] **Step 1: Write the failing tests**

Add to `eval/src/score.rs` (the file already has `normalize`):
```rust
/// Distinct article titles a question's searches surfaced, best-rank-first. Search-result hits carry
/// the article title in their `location` field; ties are broken by first occurrence across searches.
pub fn ranked_titles(transcript: &[glossa::trace::TraceEntry]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for e in transcript {
        if e.tool != "search" {
            continue;
        }
        if let Some(arr) = e.result.as_array() {
            for hit in arr {
                if let Some(loc) = hit.get("location").and_then(|v| v.as_str()) {
                    if seen.insert(normalize(loc)) {
                        out.push(loc.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Fraction of gold titles found within the top-k of the merged ranked list.
pub fn recall_at_k(ranked: &[String], gold: &[String], k: usize) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let top: Vec<String> = ranked.iter().take(k).map(|t| normalize(t)).collect();
    let hit = gold.iter().filter(|g| top.contains(&normalize(g))).count();
    hit as f32 / gold.len() as f32
}

/// Reciprocal rank of the first gold title in the merged ranked list (0 if none).
pub fn mrr(ranked: &[String], gold: &[String]) -> f32 {
    if gold.is_empty() {
        return 0.0;
    }
    let goldn: Vec<String> = gold.iter().map(|g| normalize(g)).collect();
    for (i, t) in ranked.iter().enumerate() {
        if goldn.contains(&normalize(t)) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

#[cfg(test)]
mod retrieval_at_k_tests {
    use super::*;
    use glossa::trace::TraceEntry;

    #[test]
    fn ranked_titles_dedups_across_searches() {
        let mk = |hits: serde_json::Value| TraceEntry {
            ts_ms: 0, tool: "search".into(), args: serde_json::json!({}), result: hits,
        };
        let tr = vec![
            mk(serde_json::json!([{"location":"A","score":2.0},{"location":"B","score":1.0}])),
            mk(serde_json::json!([{"location":"B","score":3.0},{"location":"C","score":1.0}])),
        ];
        assert_eq!(ranked_titles(&tr), vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }

    #[test]
    fn recall_and_mrr() {
        let ranked = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
        let gold = vec!["C".to_string(), "E".to_string()];
        assert!((recall_at_k(&ranked, &gold, 2) - 0.0).abs() < 1e-6); // C is rank 3
        assert!((recall_at_k(&ranked, &gold, 3) - 0.5).abs() < 1e-6);
        assert!((mrr(&ranked, &gold) - (1.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn matching_is_normalized() {
        let ranked = vec!["The Beatles".to_string()];
        assert_eq!(recall_at_k(&ranked, &["the beatles".to_string()], 1), 1.0);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p kb-eval --lib score`
Expected: FAIL before the functions exist, PASS after (existing score tests + 3 new).

- [ ] **Step 3: Commit**
```bash
git add eval/src/score.rs
git commit -m "feat(eval): Recall@k + MRR scoring from the tool-call transcript"
```

---

### Task 3: `--fullwiki` run mode + Recall@k in the report

**Files:**
- Modify: `eval/src/run.rs` (fullwiki gating, Recall@k per row + aggregate)
- Modify: `eval/src/main.rs` (`--fullwiki` flag, thread it into `run_eval`, print Recall@k)

**Interfaces:**
- Consumes: `score::{ranked_titles, recall_at_k, mrr}`, `corpus`, `trace_read`.
- Produces: `run::run_eval(dataset_path, backend, backend_name, limit, kb_bin, work, fullwiki: Option<&Path>) -> Report` with new `Row`/`Report` fields `recall_at_5/10/20`, `mrr` (+ aggregate means).

- [ ] **Step 1: Extend `Row` and `Report` in `eval/src/run.rs`**

Add fields to `Row` (after `retrieval_recall`):
```rust
    pub recall_at_5: f32,
    pub recall_at_10: f32,
    pub recall_at_20: f32,
    pub mrr: f32,
```
Add to `Report` (after `recall_mean`):
```rust
    pub recall_at_5_mean: f32,
    pub recall_at_10_mean: f32,
    pub recall_at_20_mean: f32,
    pub mrr_mean: f32,
```

- [ ] **Step 2: Thread fullwiki + compute Recall@k in `eval_one`/`run_eval`**

Change `run_eval`'s signature and body:
```rust
pub fn run_eval(
    dataset_path: &Path,
    backend: &dyn AgentBackend,
    backend_name: &str,
    limit: usize,
    kb_bin: &str,
    work: &Path,
    fullwiki: Option<&Path>,
) -> anyhow::Result<Report> {
    let json = std::fs::read_to_string(dataset_path)?;
    let mut questions = dataset::parse_hotpot(&json)?;
    if limit > 0 && questions.len() > limit {
        questions.truncate(limit);
    }
    let eff_work = fullwiki.unwrap_or(work);
    let rows: Vec<Row> = questions
        .iter()
        .map(|q| eval_one(backend, q, kb_bin, eff_work, fullwiki.is_some()))
        .collect();
    let n = rows.len().max(1) as f32;
    let em_mean = rows.iter().filter(|r| r.em).count() as f32 / n;
    let f1_mean = rows.iter().map(|r| r.f1).sum::<f32>() / n;
    let recall_mean = rows.iter().map(|r| r.retrieval_recall).sum::<f32>() / n;
    let recall_at_5_mean = rows.iter().map(|r| r.recall_at_5).sum::<f32>() / n;
    let recall_at_10_mean = rows.iter().map(|r| r.recall_at_10).sum::<f32>() / n;
    let recall_at_20_mean = rows.iter().map(|r| r.recall_at_20).sum::<f32>() / n;
    let mrr_mean = rows.iter().map(|r| r.mrr).sum::<f32>() / n;
    Ok(Report {
        backend: backend_name.to_string(), rows, em_mean, f1_mean, recall_mean,
        recall_at_5_mean, recall_at_10_mean, recall_at_20_mean, mrr_mean,
    })
}
```
Change `eval_one` to take the `fullwiki: bool` flag and compute Recall@k. Replace the function with:
```rust
fn eval_one(backend: &dyn AgentBackend, q: &dataset::Question, kb_bin: &str, work: &Path, fullwiki: bool) -> Row {
    let base = Row {
        id: q.id.clone(), question: q.question.clone(), gold: q.answer.clone(),
        pred: String::new(), em: false, f1: 0.0, retrieval_recall: 0.0, failed: None,
        transcript: Vec::new(),
        recall_at_5: 0.0, recall_at_10: 0.0, recall_at_20: 0.0, mrr: 0.0,
    };
    // In fullwiki mode the shared corpus is pre-built; do NOT write/index/clear per question.
    if backend.needs_corpus() && !fullwiki {
        if let Err(e) = corpus::write_corpus(work, q).and_then(|_| corpus::index(work, kb_bin)) {
            return Row { failed: Some(format!("corpus: {e}")), ..base };
        }
    }
    let t0 = now_ms();
    let pred = match backend.answer(work, q) {
        Ok(p) => p,
        Err(e) => return Row { failed: Some(format!("backend: {e}")), ..base },
    };
    let t1 = now_ms();
    let entries = if backend.needs_corpus() {
        let dir = work.join(".glossa").join("traces");
        trace_read::read_window(&dir, t0, t1).unwrap_or_default()
    } else {
        Vec::new()
    };
    let recall = if backend.needs_corpus() {
        score::retrieval_recall(&trace_read::seen_files(&entries), &q.supporting_titles)
    } else {
        0.0
    };
    let titles = score::ranked_titles(&entries);
    Row {
        em: score::exact_match(&pred, &q.answer),
        f1: score::token_f1(&pred, &q.answer),
        retrieval_recall: recall,
        recall_at_5: score::recall_at_k(&titles, &q.supporting_titles, 5),
        recall_at_10: score::recall_at_k(&titles, &q.supporting_titles, 10),
        recall_at_20: score::recall_at_k(&titles, &q.supporting_titles, 20),
        mrr: score::mrr(&titles, &q.supporting_titles),
        pred,
        transcript: entries,
        ..base
    }
}
```

- [ ] **Step 3: Add the `--fullwiki` flag and wire it in `eval/src/main.rs`**

Add to the `Run` variant fields:
```rust
        /// Fullwiki mode: search this pre-built shared corpus index (skip per-question corpus build).
        #[arg(long)]
        fullwiki: Option<PathBuf>,
```
Add `fullwiki` to the `Cmd::Run { ... }` destructuring, pass it to `run_eval`:
```rust
            let report = run::run_eval(&dataset, be.as_ref(), &name, limit, &kb_bin, &work, fullwiki.as_deref())?;
```
Extend the summary print:
```rust
            println!(
                "backend={} questions={} EM={:.3} F1={:.3} recall={:.3} R@5={:.3} R@10={:.3} R@20={:.3} MRR={:.3}\nwrote {}",
                report.backend, report.rows.len(), report.em_mean, report.f1_mean, report.recall_mean,
                report.recall_at_5_mean, report.recall_at_10_mean, report.recall_at_20_mean, report.mrr_mean, json_path
            );
```

- [ ] **Step 4: Confirm the mock gate still passes (no edit needed)**

`eval/tests/mock_e2e.rs` shells out to the binary via `assert_cmd::Command::cargo_bin("kb-eval")` — it does NOT call `run_eval` directly — so the new *optional* `--fullwiki` flag needs no change to it.

Run: `cargo test -p kb-eval --test mock_e2e`
Expected: PASS (the mock distractor path is unchanged; `--fullwiki` defaults to `None`).

- [ ] **Step 5: Add a fullwiki-mode unit test in `run.rs`**

`kb-eval` is a **binary-only crate** (modules are declared in `main.rs`; there is no `lib.rs`), so the wiring test lives IN-crate as a unit test in `eval/src/run.rs`, where `run_eval` and `MockBackend` are directly accessible:
```rust
#[cfg(test)]
mod fullwiki_tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use std::collections::HashMap;

    #[test]
    fn fullwiki_mode_does_not_build_per_question_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("wiki");
        std::fs::create_dir_all(&corpus).unwrap();
        // a sentinel a per-question write_corpus clear would remove
        std::fs::write(corpus.join("Sentinel.md"), b"# Sentinel\nkeep me\n").unwrap();

        let dataset = dir.path().join("d.json");
        std::fs::write(&dataset, br#"[{"_id":"q1","question":"Who?","answer":"Bob",
            "context":[["Bob Page",["b."]]],"supporting_facts":[["Bob Page",0]]}]"#).unwrap();

        let be = MockBackend { canned: HashMap::new() };
        let report = run_eval(&dataset, &be, "mock", 0, "kb", dir.path(), Some(corpus.as_path())).unwrap();

        assert_eq!(report.rows.len(), 1);
        assert!(corpus.join("Sentinel.md").exists(), "fullwiki must NOT clear the shared corpus");
        assert_eq!(report.recall_at_5_mean, 0.0); // empty mock transcript
    }
}
```
`MockBackend::needs_corpus()` is `false`, so no model/network and no per-question indexing runs; `tempfile` is a dev-dependency (fine under `#[cfg(test)]`).

Run: `cargo test -p kb-eval --lib run`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `cargo test -p kb-eval`
Expected: PASS (prep, score, mock_e2e, fullwiki wiring).

- [ ] **Step 7: Commit**
```bash
git add eval/src/run.rs eval/src/main.rs
git commit -m "feat(eval): --fullwiki shared-index run mode + Recall@k/MRR in the report"
```

---

## Notes for the implementer

- The download of `eval-data/enwiki-abstracts.tar.bz2` (1.55 GB) runs separately; do not block on it. The Task 1 tests use a tiny synthetic archive built in-test, so all of Plan 1 is implementable and testable without the real archive.
- After all tasks land: run the **feasibility spike** (above), then the operator decides on the full 5M build.
