//! `kbx dataset` operations over `dataset.toml`-shape files: `stat`, `merge`, `validate`, `dedup`,
//! `sample`. Every op reads through [`crate::dataset_toml::parse_dataset_toml`] (the single case
//! parser) and writes through [`write_cases`] — a full-fidelity `[[case]]` serializer that
//! ROUND-TRIPS every field the parser understands (id/question/answer/aliases/tags/hop_type/
//! needs_graph/source/answerable), so merge/dedup never silently drop a field.
//!
//! The logic here is pure and testable — no clap, no stdout formatting, no wall clock (sampling is
//! seeded). `kbx.rs` stays thin: parse args -> call one function here -> print.

use crate::backend::openai::OpenAiBackend;
use crate::dataset::Question;
use crate::dataset_toml::parse_dataset_toml;
use anyhow::{Context, Result};
use glossa::graph::store::GraphStore;
use glossa::index::store::DocIndex;
use glossa::tools::abstention::{coverage_uncovered, covered};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

/// One `[[case]]` in full fidelity — every field `parse_dataset_toml` reads back. Empty
/// collections/strings and the `answerable=true` default are skipped on write so a round-tripped
/// file stays as terse as it started (an absent field re-parses to the same default).
#[derive(Debug, Clone, Serialize)]
pub struct Case {
    pub id: String,
    pub question: String,
    pub answer: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub hop_type: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub needs_graph: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source: Vec<String>,
    #[serde(skip_serializing_if = "is_true")]
    pub answerable: bool,
    /// Deliberate abstention test, orthogonal to `answerable`/`hop_type`/`question` (never
    /// conflated with them). Defaults to `false`; omitted on write when `false` so an absent
    /// `[[case]]` key re-parses to the same default (byte-clean round-trip, like `answerable`).
    #[serde(skip_serializing_if = "is_false")]
    pub abstention: bool,
    /// Optional distilled/canonicalized restatement of `question`. Omitted on write when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distilled_query: Option<String>,
}

/// `#[serde(skip_serializing_if)]` predicate: omit `answerable` when it holds its `true` default
/// (an absent `[[case]]` key re-parses to `true`, so the round-trip is lossless).
fn is_true(b: &bool) -> bool {
    *b
}

/// `#[serde(skip_serializing_if)]` predicate: omit `abstention` when it holds its `false` default
/// (an absent `[[case]]` key re-parses to `false`, so the round-trip is lossless).
fn is_false(b: &bool) -> bool {
    !*b
}

impl Case {
    /// Lift a parsed [`Question`] into a serializable `Case` (the non-case eval fields —
    /// `paragraphs`/`supporting_titles` — are not `[[case]]` keys and are dropped).
    pub fn from_question(q: &Question) -> Self {
        Case {
            id: q.id.clone(),
            question: q.question.clone(),
            answer: q.answer.clone(),
            aliases: q.answer_aliases.clone(),
            tags: q.tags.clone(),
            hop_type: q.hop_type.clone(),
            needs_graph: q.needs_graph.clone(),
            source: q.source.clone(),
            answerable: q.answerable,
            abstention: q.abstention,
            distilled_query: q.distilled_query.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct CaseFile {
    case: Vec<Case>,
}

/// Serialize `cases` as `[[case]]` blocks and write them to `path` (created/truncated), creating
/// the parent dir if needed. The inverse of `parse_dataset_toml` — full-fidelity round-trip.
pub fn write_cases(path: &Path, cases: &[Case]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let file = CaseFile {
        case: cases.to_vec(),
    };
    let text = toml::to_string_pretty(&file).context("serializing dataset.toml")?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Parse a `dataset.toml` file into full-fidelity [`Case`]s (through the single shared parser).
pub fn load_cases(path: &Path) -> Result<Vec<Case>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let qs = parse_dataset_toml(&text)?;
    Ok(qs.iter().map(Case::from_question).collect())
}

/// Normalize a question/answer for dedup + duplicate counting: trim, collapse inner whitespace
/// runs to a single space, lowercase. Two cases whose questions normalize equal are "the same
/// question" for merge/dedup/stat purposes.
pub fn normalize(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

// ---------------------------------------------------------------------------------------------
// stat
// ---------------------------------------------------------------------------------------------

/// Char-length min/median/max over a set of strings (median = middle of the sorted lengths, the
/// mean of the two middles for an even count). All zero for an empty set.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LenStats {
    pub min: usize,
    pub median: usize,
    pub max: usize,
}

fn len_stats(lens: &[usize]) -> LenStats {
    if lens.is_empty() {
        return LenStats::default();
    }
    let mut v = lens.to_vec();
    v.sort_unstable();
    let n = v.len();
    let median = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    };
    LenStats {
        min: v[0],
        median,
        max: v[n - 1],
    }
}

/// Read-only summary of a dataset (see [`compute_stat`]). Tests assert on this, not on stdout.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Stat {
    pub total: usize,
    /// hop_type breakdown.
    pub lexical: usize,
    pub multihop: usize,
    pub untyped: usize,
    /// answerable (default true when absent) vs explicit false.
    pub answerable: usize,
    /// Total non-answerable count (manually-authored `answerable=false` cases; orthogonal to
    /// `abstention` — a case can be `answerable=true` yet still carry `abstention=true`).
    pub unanswerable: usize,
    /// Count of cases with `abstention=true` (set by `kbx dataset gate-mark`'s coverage-abstention
    /// model pass), regardless of `answerable`.
    pub abstention_flagged: usize,
    /// Crossbreak of `abstention_flagged`: `answerable && abstention` — an originally-answerable
    /// case the coverage gate would incorrectly abstain on. The dataset-quality signal to watch:
    /// a non-zero count here means the gate's `k` threshold (or the corpus's coverage) needs
    /// tuning, since a real question is being blocked.
    pub false_abstain: usize,
    /// Crossbreak of `abstention_flagged`: `!answerable && abstention` — an already-unanswerable
    /// case the gate ALSO correctly flags, confirming the gate agrees with the manual label.
    pub flagged_unanswerable: usize,
    /// needs_graph value -> count (empty value keyed as "(unset)").
    pub needs_graph: BTreeMap<String, usize>,
    /// alias coverage.
    pub with_aliases: usize,
    pub without_aliases: usize,
    /// extra occurrences of a normalized question/answer (total - distinct).
    pub dup_questions: usize,
    pub dup_answers: usize,
    /// cases with a blank (after trim) question OR answer.
    pub blank: usize,
    pub q_len: LenStats,
    pub a_len: LenStats,
}

/// Compute a [`Stat`] over `cases`. Pure — no IO.
pub fn compute_stat(cases: &[Case]) -> Stat {
    let mut s = Stat {
        total: cases.len(),
        ..Default::default()
    };
    let mut q_seen: HashSet<String> = HashSet::new();
    let mut a_seen: HashSet<String> = HashSet::new();
    let mut q_lens: Vec<usize> = Vec::with_capacity(cases.len());
    let mut a_lens: Vec<usize> = Vec::with_capacity(cases.len());
    for c in cases {
        match c.hop_type.as_str() {
            "lexical" => s.lexical += 1,
            "multihop" => s.multihop += 1,
            _ => s.untyped += 1,
        }
        if c.answerable {
            s.answerable += 1;
        } else {
            s.unanswerable += 1;
        }
        if c.abstention {
            s.abstention_flagged += 1;
            if c.answerable {
                s.false_abstain += 1;
            } else {
                s.flagged_unanswerable += 1;
            }
        }
        let ng = if c.needs_graph.is_empty() {
            "(unset)".to_string()
        } else {
            c.needs_graph.clone()
        };
        *s.needs_graph.entry(ng).or_default() += 1;
        if c.aliases.is_empty() {
            s.without_aliases += 1;
        } else {
            s.with_aliases += 1;
        }
        if !q_seen.insert(normalize(&c.question)) {
            s.dup_questions += 1;
        }
        if !a_seen.insert(normalize(&c.answer)) {
            s.dup_answers += 1;
        }
        if c.question.trim().is_empty() || c.answer.trim().is_empty() {
            s.blank += 1;
        }
        q_lens.push(c.question.chars().count());
        a_lens.push(c.answer.chars().count());
    }
    s.q_len = len_stats(&q_lens);
    s.a_len = len_stats(&a_lens);
    s
}

// ---------------------------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------------------------

/// One problem found by [`validate_cases`], tagged with the offending case id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub id: String,
    pub problem: String,
}

/// Structural checks over a parsed dataset: non-empty question AND answer (after trim), `hop_type`
/// in {"", "lexical", "multihop"}, and unique ids. Returns every issue found (empty = clean).
/// A TOML/parse error is caught earlier, when the file is loaded.
pub fn validate_cases(cases: &[Case]) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let mut seen_ids: HashSet<&str> = HashSet::new();
    for c in cases {
        if c.question.trim().is_empty() {
            issues.push(ValidationIssue {
                id: c.id.clone(),
                problem: "empty question".into(),
            });
        }
        if c.answer.trim().is_empty() {
            issues.push(ValidationIssue {
                id: c.id.clone(),
                problem: "empty answer".into(),
            });
        }
        if !matches!(c.hop_type.as_str(), "" | "lexical" | "multihop") {
            issues.push(ValidationIssue {
                id: c.id.clone(),
                problem: format!(
                    "invalid hop_type {:?} (want \"\"|lexical|multihop)",
                    c.hop_type
                ),
            });
        }
        if !seen_ids.insert(c.id.as_str()) {
            issues.push(ValidationIssue {
                id: c.id.clone(),
                problem: "duplicate id".into(),
            });
        }
    }
    issues
}

// ---------------------------------------------------------------------------------------------
// dedup
// ---------------------------------------------------------------------------------------------

/// Drop duplicate cases by normalized question, keeping the FIRST occurrence. Returns
/// `(kept, removed)` — order-preserving over the kept set.
pub fn dedup_cases(cases: &[Case]) -> (Vec<Case>, usize) {
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<Case> = Vec::with_capacity(cases.len());
    for c in cases {
        if seen.insert(normalize(&c.question)) {
            kept.push(c.clone());
        }
    }
    let removed = cases.len() - kept.len();
    (kept, removed)
}

// ---------------------------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------------------------

/// A stable, collision-free id derived from `id`: `id` if free, else `id-1`, `id-2`, … — the first
/// suffix not already in `used`. The chosen id is inserted into `used`.
fn unique_id(id: &str, used: &mut HashSet<String>) -> String {
    if used.insert(id.to_string()) {
        return id.to_string();
    }
    let mut n = 1usize;
    loop {
        let candidate = format!("{id}-{n}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

/// Counts returned by [`merge_cases`] / `kbx dataset merge`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MergeSummary {
    pub from: usize,
    pub added: usize,
    pub skipped_dup: usize,
    pub total: usize,
}

/// Merge `src` into `dst` (pure core of `kbx dataset merge`): append each src case whose normalized
/// question is not already present in dst OR earlier in src (dedup within the incoming batch too);
/// re-id any incoming case whose id collides with an existing id (dst or an already-added src id)
/// via [`unique_id`]. Returns `(merged, summary)` — all fields preserved.
pub fn merge_cases(dst: &[Case], src: &[Case]) -> (Vec<Case>, MergeSummary) {
    let mut merged: Vec<Case> = dst.to_vec();
    let mut seen_q: HashSet<String> = dst.iter().map(|c| normalize(&c.question)).collect();
    let mut used_ids: HashSet<String> = dst.iter().map(|c| c.id.clone()).collect();
    let mut summary = MergeSummary {
        from: src.len(),
        ..Default::default()
    };
    for c in src {
        let nq = normalize(&c.question);
        if !seen_q.insert(nq) {
            summary.skipped_dup += 1;
            continue;
        }
        let mut c = c.clone();
        c.id = unique_id(&c.id, &mut used_ids);
        merged.push(c);
        summary.added += 1;
    }
    summary.total = merged.len();
    (merged, summary)
}

/// `kbx dataset merge` end-to-end: load both files, [`merge_cases`], back `into` up to
/// `<into>.bak`, then write the merged set back to `into`. Returns the summary to print.
pub fn merge_files(from: &Path, into: &Path) -> Result<MergeSummary> {
    let dst = load_cases(into)?;
    let src = load_cases(from)?;
    let (merged, summary) = merge_cases(&dst, &src);
    // Back up the pre-merge destination before overwriting it.
    let bak = backup_path(into);
    std::fs::copy(into, &bak)
        .with_context(|| format!("backing up {} to {}", into.display(), bak.display()))?;
    write_cases(into, &merged)?;
    Ok(summary)
}

/// `<path>.bak` — the backup sibling merge/dedup write before overwriting `path`.
pub fn backup_path(path: &Path) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".bak");
    std::path::PathBuf::from(s)
}

/// `kbx dataset dedup` end-to-end: load, [`dedup_cases`], back up to `<file>.bak`, write back.
/// Returns `(removed, new_total)`.
pub fn dedup_file(file: &Path) -> Result<(usize, usize)> {
    let cases = load_cases(file)?;
    let (kept, removed) = dedup_cases(&cases);
    let bak = backup_path(file);
    std::fs::copy(file, &bak)
        .with_context(|| format!("backing up {} to {}", file.display(), bak.display()))?;
    let total = kept.len();
    write_cases(file, &kept)?;
    Ok((removed, total))
}

// ---------------------------------------------------------------------------------------------
// sample
// ---------------------------------------------------------------------------------------------

/// Choose `n` cases with a SEEDED RNG (reproducible for a given `seed`). `n >= total` returns every
/// case in original order; otherwise a seeded sample of `n` distinct cases (in the RNG's draw
/// order). Model-free and clock-free.
pub fn sample_cases(cases: &[Case], n: usize, seed: u64) -> Vec<Case> {
    let len = cases.len();
    if n == 0 || len == 0 {
        return Vec::new();
    }
    if n >= len {
        return cases.to_vec();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    rand::seq::index::sample(&mut rng, len, n)
        .into_iter()
        .map(|i| cases[i].clone())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// gate-mark (coverage-abstention model pass)
// ---------------------------------------------------------------------------------------------

/// Set `abstention` on every case from its (already-distilled) query's coverage: `abstention =
/// true` when at least `k` of the query's distinctive terms are uncovered by the corpus
/// (`coverage_uncovered(query, covered, k).len() >= k`; `covered` wraps
/// [`glossa::tools::abstention::covered`] over `g`/`idx` — the SAME threshold the runtime
/// coverage-abstention gate itself fires on). The query judged is `case.distilled_query` when
/// present, else the raw `case.question` — this function assumes the caller already populated
/// `distilled_query` via [`distill_question`] (or left it `None` for a question that doesn't need
/// distilling); it does not call the model itself.
///
/// Touches ONLY `abstention` — `hop_type`, `answerable`, `question`, and `distilled_query` are
/// left byte-identical, unlike the old destructive `gate_mark` this replaces (which used to flip
/// `answerable`/`hop_type` and stamp `gated`/`orig_hop:` tags). `abstention` is an orthogonal
/// signal: whether the runtime gate would abstain, independent of whether the case is manually
/// marked answerable. Pure: no IO — `g`/`idx` are read-only lookups threaded through to `covered`;
/// the caller owns opening them and writing the mutated `cases` back out.
pub fn mark_abstention(cases: &mut [Case], g: &GraphStore, idx: &DocIndex, k: usize) {
    for c in cases.iter_mut() {
        let query = c.distilled_query.as_deref().unwrap_or(&c.question);
        let uncovered = coverage_uncovered(query, &|t: &str| covered(t, idx, g), k);
        c.abstention = uncovered.len() >= k;
    }
}

/// The absent (uncovered) distinctive terms for `query`, per the same [`coverage_uncovered`] call
/// [`mark_abstention`] makes — reused by `kbx dataset gate-mark`'s `coverage-gaps.md` report so it
/// doesn't need to re-derive the closure/threshold logic at the CLI layer.
pub fn absent_terms(query: &str, g: &GraphStore, idx: &DocIndex, k: usize) -> Vec<String> {
    coverage_uncovered(query, &|t: &str| covered(t, idx, g), k)
        .into_iter()
        .map(|a| a.term)
        .collect()
}

/// Build the two seed messages for one `distill_question` call: `prompt_tmpl` (the behavior-guide
/// instructions, e.g. `question_distill.md`) verbatim as the system message, `question` (the raw
/// ticket text) verbatim as the user message — the same system+user split every other single-shot
/// call site in this crate uses (`judge.rs`, `backend/user_sim.rs`, `distil/gen.rs`'s leak check).
/// Pure and separately testable from the network call itself.
fn distill_messages(prompt_tmpl: &str, question: &str) -> Vec<Value> {
    vec![
        json!({ "role": "system", "content": prompt_tmpl }),
        json!({ "role": "user", "content": question }),
    ]
}

/// Extract the canonical information need / key search terms from a raw support-ticket question
/// (greeting, signature, and contact info stripped by the MODEL, not any hardcoded list) — one
/// temp-0 completion via `backend`'s configured endpoint. `prompt_tmpl` is the loaded
/// `question_distill.md` behavior-guide text (loading mechanism — workspace override vs embedded
/// default — is the caller's concern, mirroring how `builder.md`/`reason.md`/etc. are loaded
/// elsewhere in this crate); `question` is the raw ticket text, interpolated verbatim into the
/// user message by [`distill_messages`].
///
/// Mirrors the single-shot call pattern every other one-off model call in this crate uses
/// (`judge.rs::judge`, `backend/user_sim.rs`, `distil/gen.rs`'s leak check): build a `[system,
/// user]` message pair, drive it through [`crate::backend::openai::chat_once_resampled`] (the
/// provider-neutral degenerate-resample wrapper every production one-off call goes through), and
/// read back `.content`. Deviates from those call sites in one respect: `distill_question` forces
/// `temperature = Some(0.0)` on its own `Endpoint` copy (via [`OpenAiBackend::endpoint_config`])
/// rather than deferring to the backend's configured reader temperature — this call always wants
/// the deterministic canonical restatement, not the reader's sampling behavior (`KB_EVAL_TEMP` can
/// still override it, exactly as `Endpoint::resolve_temperature` does uniformly everywhere else).
///
/// Returns the model's raw text trimmed. No mock exists for `OpenAiBackend`'s HTTP-backed model
/// call (`MockBackend` only stands in for the `AgentBackend::answer` trait, not this crate's
/// single-shot completions), so this function itself is exercised by Task 5's gate-mark
/// integration; the pure interpolation this depends on is covered by
/// `distill_messages_embeds_template_and_question` below.
pub fn distill_question(
    backend: &OpenAiBackend,
    prompt_tmpl: &str,
    question: &str,
) -> Result<String> {
    let messages = distill_messages(prompt_tmpl, question);
    let mut ep = backend.endpoint_config();
    ep.temperature = Some(0.0);
    let msg = crate::backend::openai::chat_once_resampled(&ep, &messages)
        .context("question-distill endpoint request failed")?;
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    Ok(content.trim().to_string())
}

/// Embedded default `question_distill.md` — the same behavior-guide text `kbx init` would
/// scaffold, baked in so `kbx dataset gate-mark` works even on a workspace that predates this
/// prompt file (no rebuild needed to ship the default; mirrors `scaffold.rs`'s `include_str!`
/// pattern for `builder.md`/`judge.md`/etc.).
const DEFAULT_QUESTION_DISTILL_MD: &str = include_str!("../templates/question_distill.md");

/// Load the `question_distill.md` behavior-guide text for `distill_question`'s system prompt: a
/// workspace override at `<corpus_root>/.glossa/kbx/question_distill.md` when present (so an
/// operator can tune the distillation instructions without a rebuild, exactly like `builder.md`),
/// else [`DEFAULT_QUESTION_DISTILL_MD`]. Unlike `builder.md` (which `kbx init` always scaffolds
/// and `run_build` requires present), this prompt has no dedicated `KbxPaths` field yet, so the
/// override path is resolved directly here rather than through `workspace::KbxPaths`.
pub fn load_question_distill_tmpl(corpus_root: &Path) -> Result<String> {
    let override_path = corpus_root
        .join(".glossa")
        .join("kbx")
        .join("question_distill.md");
    if override_path.is_file() {
        std::fs::read_to_string(&override_path)
            .with_context(|| format!("reading {}", override_path.display()))
    } else {
        Ok(DEFAULT_QUESTION_DISTILL_MD.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str, q: &str, a: &str) -> Case {
        Case {
            id: id.into(),
            question: q.into(),
            answer: a.into(),
            aliases: Vec::new(),
            tags: Vec::new(),
            hop_type: String::new(),
            needs_graph: String::new(),
            source: Vec::new(),
            answerable: true,
            abstention: false,
            distilled_query: None,
        }
    }

    #[test]
    fn distill_messages_embeds_template_and_question() {
        // Pure interpolation check: the template text lands verbatim in the system message, the
        // raw ticket question lands verbatim in the user message — no mangling, no reordering.
        // (The model-call half of `distill_question` needs a live endpoint and is covered by
        // Task 5's gate-mark integration; see that fn's doc comment.)
        let tmpl = "Extract the distilled query. Ignore greetings and signatures.";
        let question = "Good day! What is the max input voltage? Best regards, Alex";
        let messages = distill_messages(tmpl, question);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], tmpl);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], question);
    }

    #[test]
    fn write_cases_round_trips_every_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.toml");
        let cases = vec![Case {
            id: "c0".into(),
            question: "  Q one? ".into(),
            answer: "answer one".into(),
            aliases: vec!["a1".into(), "a2".into()],
            tags: vec!["net".into()],
            hop_type: "multihop".into(),
            needs_graph: "yes".into(),
            source: vec!["a.pdf#p.1".into(), "b.pdf#p.2".into()],
            answerable: false,
            abstention: true,
            distilled_query: Some("configure profibus maxTsdr".into()),
        }];
        write_cases(&path, &cases).unwrap();

        let back = load_cases(&path).unwrap();
        assert_eq!(back.len(), 1);
        let c = &back[0];
        assert_eq!(c.id, "c0");
        assert_eq!(c.question, "  Q one? ");
        assert_eq!(c.answer, "answer one");
        assert_eq!(c.aliases, vec!["a1".to_string(), "a2".to_string()]);
        assert_eq!(c.tags, vec!["net".to_string()]);
        assert_eq!(c.hop_type, "multihop");
        assert_eq!(c.needs_graph, "yes");
        assert_eq!(
            c.source,
            vec!["a.pdf#p.1".to_string(), "b.pdf#p.2".to_string()]
        );
        assert!(!c.answerable, "answerable=false survives the round-trip");
        assert!(c.abstention, "abstention=true survives the round-trip");
        assert_eq!(
            c.distilled_query.as_deref(),
            Some("configure profibus maxTsdr"),
            "distilled_query survives the round-trip"
        );
    }

    #[test]
    fn write_cases_defaults_survive_when_omitted() {
        // A minimal case (all optionals empty, answerable true) round-trips to the same defaults.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.toml");
        write_cases(&path, &[case("c0", "Q?", "A")]).unwrap();
        let back = load_cases(&path).unwrap();
        assert!(back[0].aliases.is_empty() && back[0].tags.is_empty());
        assert!(back[0].hop_type.is_empty() && back[0].needs_graph.is_empty());
        assert!(back[0].source.is_empty());
        assert!(back[0].answerable, "absent answerable re-parses to true");
        assert!(!back[0].abstention, "absent abstention re-parses to false");
        assert!(
            back[0].distilled_query.is_none(),
            "absent distilled_query re-parses to None"
        );
    }

    #[test]
    fn write_cases_omits_abstention_and_distilled_query_when_default() {
        // A case with abstention=false / distilled_query=None (the defaults) must OMIT both keys
        // from the written TOML -- byte-clean, exactly like `answerable=true` is omitted.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.toml");
        write_cases(&path, &[case("c0", "Q?", "A")]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("abstention"),
            "abstention=false must be omitted from the written file: {text}"
        );
        assert!(
            !text.contains("distilled_query"),
            "distilled_query=None must be omitted from the written file: {text}"
        );
    }

    #[test]
    fn normalize_collapses_whitespace_and_case() {
        assert_eq!(normalize("  Foo   BAR  "), "foo bar");
        assert_eq!(normalize("Foo\tBar\nBaz"), "foo bar baz");
    }

    #[test]
    fn dedup_keeps_first_of_normalized_duplicates() {
        let cases = vec![
            case("a", "What is X?", "1"),
            case("b", "  what   is x? ", "2"), // normalized-equal to a -> dropped
            case("c", "Different?", "3"),
        ];
        let (kept, removed) = dedup_cases(&cases);
        assert_eq!(removed, 1);
        let ids: Vec<&str> = kept.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "c"], "first occurrence kept");
    }

    #[test]
    fn merge_dedups_reids_and_counts() {
        let dst = vec![case("q0", "Existing?", "e"), case("q1", "Second?", "s")];
        let src = vec![
            case("q0", "Brand new?", "n"),    // id collides with dst q0 -> re-id
            case("dup", "  existing? ", "x"), // normalized dup of dst -> skipped
            case("q9", "Another new?", "y"),  // clean add
        ];
        let (merged, summary) = merge_cases(&dst, &src);
        assert_eq!(summary.from, 3);
        assert_eq!(summary.added, 2);
        assert_eq!(summary.skipped_dup, 1);
        assert_eq!(summary.total, 4);
        assert_eq!(merged.len(), 4);
        // The colliding incoming id was re-id'd, not overwritten; dst's original q0 is intact.
        let ids: Vec<&str> = merged.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids[0], "q0");
        assert!(
            ids.contains(&"q0-1"),
            "re-id'd incoming id present: {ids:?}"
        );
        // No id appears twice.
        let uniq: HashSet<&&str> = ids.iter().collect();
        assert_eq!(uniq.len(), ids.len(), "all ids unique: {ids:?}");
    }

    #[test]
    fn merge_files_backs_up_and_preserves_fields() {
        let dir = tempfile::tempdir().unwrap();
        let into = dir.path().join("dst.toml");
        let from = dir.path().join("src.toml");
        write_cases(&into, &[case("q0", "Existing?", "e")]).unwrap();
        let mut incoming = case("q0", "Fresh?", "f");
        incoming.tags = vec!["t".into()];
        incoming.hop_type = "lexical".into();
        write_cases(&from, &[incoming]).unwrap();

        let summary = merge_files(&from, &into).unwrap();
        assert_eq!(summary.added, 1);
        assert!(backup_path(&into).exists(), "dst.bak was written");

        let merged = load_cases(&into).unwrap();
        assert_eq!(merged.len(), 2);
        let fresh = merged.iter().find(|c| c.question == "Fresh?").unwrap();
        assert_eq!(
            fresh.tags,
            vec!["t".to_string()],
            "fields preserved on merge"
        );
        assert_eq!(fresh.hop_type, "lexical");
        assert_ne!(fresh.id, "q0", "collision re-id'd");
    }

    #[test]
    fn validate_flags_empty_bad_hop_and_dup_ids_but_passes_clean() {
        let clean = vec![
            {
                let mut c = case("a", "Q?", "A");
                c.hop_type = "lexical".into();
                c
            },
            case("b", "Q2?", "A2"),
        ];
        assert!(
            validate_cases(&clean).is_empty(),
            "clean file has no issues"
        );

        let dirty = vec![
            case("x", "  ", "A"),   // empty question
            case("y", "Q?", "   "), // empty answer
            {
                let mut c = case("z", "Q?", "A");
                c.hop_type = "triple".into(); // bad hop_type
                c
            },
            case("x", "Dup id?", "A"), // duplicate id
        ];
        let issues = validate_cases(&dirty);
        let probs: Vec<(&str, &str)> = issues
            .iter()
            .map(|i| (i.id.as_str(), i.problem.as_str()))
            .collect();
        assert!(probs.contains(&("x", "empty question")));
        assert!(probs.contains(&("y", "empty answer")));
        assert!(probs
            .iter()
            .any(|(id, p)| *id == "z" && p.contains("invalid hop_type")));
        assert!(probs.contains(&("x", "duplicate id")));
    }

    #[test]
    fn compute_stat_counts_correctly() {
        let cases = vec![
            {
                let mut c = case("a", "Q one?", "Ans");
                c.hop_type = "lexical".into();
                c.aliases = vec!["x".into()];
                c.needs_graph = "yes".into();
                c
            },
            {
                let mut c = case("b", "Q two longer?", "Another");
                c.hop_type = "multihop".into();
                c.answerable = false;
                c
            },
            case("c", "  q one? ", "Ans"), // normalized-dup question AND dup answer, untyped
            {
                // A gate-mark'd case that agrees with its manual unanswerable label.
                let mut c = case("d", "Q four?", "Ans four");
                c.hop_type = "unanswerable".into();
                c.answerable = false;
                c.abstention = true;
                c
            },
            {
                // A gate-mark'd FALSE ABSTAIN: manually answerable, but the coverage gate would
                // block it -- the dataset-quality signal `false_abstain` exists to surface.
                let mut c = case("e", "Q five?", "Ans five");
                c.hop_type = "lexical".into();
                c.abstention = true;
                c
            },
        ];
        let s = compute_stat(&cases);
        assert_eq!(s.total, 5);
        assert_eq!(s.lexical, 2, "a and e");
        assert_eq!(s.multihop, 1);
        assert_eq!(
            s.untyped, 2,
            "c is untyped, d's hop_type is \"unanswerable\" (also untyped)"
        );
        assert_eq!(s.answerable, 3, "a, c, e");
        assert_eq!(s.unanswerable, 2, "b (manual) + d (manual)");
        assert_eq!(s.abstention_flagged, 2, "d and e both carry abstention=true");
        assert_eq!(s.false_abstain, 1, "e: answerable=true but abstention=true");
        assert_eq!(
            s.flagged_unanswerable, 1,
            "d: answerable=false and abstention=true"
        );
        assert_eq!(s.with_aliases, 1);
        assert_eq!(s.without_aliases, 4);
        assert_eq!(s.dup_questions, 1, "c duplicates a's normalized question");
        assert_eq!(s.dup_answers, 1, "c duplicates a's answer");
        assert_eq!(s.blank, 0);
        assert_eq!(s.needs_graph.get("yes"), Some(&1));
        assert_eq!(s.needs_graph.get("(unset)"), Some(&4));
    }

    /// A minimal on-disk `GraphStore`+`DocIndex` fixture where only "profibus" is covered (indexed
    /// via a BM25 chunk) — mirrors `src/tools/abstention.rs`'s own `covered_by_bm25_hit` fixture,
    /// the existing pattern for constructing these two stores in-memory-equivalent for a test.
    fn covered_profibus_fixture() -> (tempfile::TempDir, DocIndex, GraphStore) {
        let dir = tempfile::tempdir().unwrap();
        let idx = DocIndex::open_or_create(dir.path()).unwrap();
        idx.write_chunks(&[glossa::model::Chunk {
            doc_path: "profibus.pdf".into(),
            location: "p.1".into(),
            file_type: "pdf".into(),
            text: "profibus maxTsdr timeout".into(),
        }])
        .unwrap();
        let g = GraphStore::open(dir.path()).unwrap();
        (dir, idx, g)
    }

    #[test]
    fn mark_abstention_flags_uncovered_and_clears_covered_touching_only_abstention() {
        let (_dir, idx, g) = covered_profibus_fixture();
        // A case whose DISTILLED query has an uncovered distinctive term -> abstention=true. The
        // raw `question` is deliberately covered ("profibus") so this also proves the distilled
        // query -- not the raw question -- is what's judged.
        let mut uncovered_case = case("c1", "profibus ticket", "some answer");
        uncovered_case.hop_type = "multihop".into();
        uncovered_case.answerable = true;
        uncovered_case.distilled_query = Some("What is Zylophon?".into());
        // A case whose distilled query is fully covered ("profibus" is indexed) -> abstention
        // stays false.
        let mut covered_case = case("c2", "raw q", "ans2");
        covered_case.hop_type = "lexical".into();
        covered_case.distilled_query = Some("What is profibus?".into());
        let mut cases = vec![uncovered_case, covered_case];

        mark_abstention(&mut cases, &g, &idx, 1);

        let c1 = cases.iter().find(|c| c.id == "c1").unwrap();
        assert!(c1.abstention, "uncovered distilled term -> abstention=true");
        // Only `abstention` changed -- hop_type/answerable/question/distilled_query untouched.
        assert_eq!(c1.hop_type, "multihop");
        assert!(c1.answerable);
        assert_eq!(c1.question, "profibus ticket");
        assert_eq!(c1.distilled_query.as_deref(), Some("What is Zylophon?"));

        let c2 = cases.iter().find(|c| c.id == "c2").unwrap();
        assert!(!c2.abstention, "fully-covered distilled query -> abstention=false");
        assert_eq!(c2.hop_type, "lexical");
        assert!(c2.answerable);
        assert_eq!(c2.question, "raw q");
        assert_eq!(c2.distilled_query.as_deref(), Some("What is profibus?"));
    }

    #[test]
    fn mark_abstention_falls_back_to_question_when_no_distilled_query() {
        let (_dir, idx, g) = covered_profibus_fixture();
        // No `distilled_query` set -> the raw `question` is judged directly.
        let mut cases = vec![case("c1", "What is Zylophon?", "ans")];
        mark_abstention(&mut cases, &g, &idx, 1);
        assert!(cases[0].abstention, "raw question's uncovered term flags it");
        assert!(cases[0].distilled_query.is_none(), "still untouched");
    }

    #[test]
    fn sample_is_seeded_and_reproducible() {
        let cases: Vec<Case> = (0..10)
            .map(|i| case(&format!("id{i}"), &format!("Q{i}?"), "A"))
            .collect();
        let a = sample_cases(&cases, 3, 42);
        let b = sample_cases(&cases, 3, 42);
        assert_eq!(a.len(), 3);
        let ids_a: Vec<&str> = a.iter().map(|c| c.id.as_str()).collect();
        let ids_b: Vec<&str> = b.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids_a, ids_b, "same seed -> same sample");
        // n >= total returns everything, in original order.
        let all = sample_cases(&cases, 99, 0);
        assert_eq!(all.len(), 10);
        assert_eq!(all[0].id, "id0");
    }
}
