//! `kbx dataset` operations over `dataset.toml`-shape files: `stat`, `merge`, `validate`, `dedup`,
//! `sample`. Every op reads through [`load_cases`] and writes through [`write_cases`], which
//! round-trip a `[[case]]` WITHOUT LOSS: the nine fields the eval pipeline models
//! (id/question/answer/aliases/tags/hop_type/needs_graph/source/answerable) plus every other key
//! the file happens to carry, kept verbatim in [`Case::extra`].
//!
//! That last part is the point. These ops rewrite the whole file, so anything the serializer does
//! not emit is DELETED — and this module used to promise fidelity only for "every field the parser
//! understands", which is a different and much smaller set. A dataset carrying `distilled_query`
//! or a per-case `abstention` flag lost them to a `dedup` that was asked only to remove
//! duplicates.
//!
//! The logic here is pure and testable — no clap, no stdout formatting, no wall clock (sampling is
//! seeded). `kbx.rs` stays thin: parse args -> call one function here -> print.

use crate::dataset::Question;
use crate::dataset_toml::parse_dataset_toml;
use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

/// One `[[case]]`, round-tripped WITHOUT LOSS — including keys this crate does not model.
///
/// The nine named fields are the ones the eval pipeline reads. Everything else a dataset carries
/// lands in [`Case::extra`] and is written back verbatim. That is not a nicety: `dedup` and
/// `merge` rewrite the whole file through [`write_cases`], so a key absent from this struct used
/// to be DELETED from the user's dataset by a command that claims only to remove duplicates. Real
/// datasets carry such keys — `distilled_query`, a per-case `abstention` flag — and losing them is
/// losing hand-written work, silently.
///
/// Empty collections/strings and the `answerable = true` default are skipped on write, so a
/// round-tripped file stays as terse as it started (an absent field re-parses to the same default).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub question: String,
    pub answer: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub hop_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub needs_graph: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source: Vec<String>,
    #[serde(default = "default_answerable", skip_serializing_if = "is_true")]
    pub answerable: bool,
    /// Every `[[case]]` key this crate does not model, kept so a rewrite gives it back. Flattened,
    /// so it is invisible in the file: these keys sit at case level exactly where they started.
    #[serde(flatten, default, skip_serializing_if = "toml::Table::is_empty")]
    pub extra: toml::Table,
}

/// serde `default` for `answerable`: an absent key means true, matching `parse_dataset_toml`.
fn default_answerable() -> bool {
    true
}

/// `#[serde(skip_serializing_if)]` predicate: omit `answerable` when it holds its `true` default
/// (an absent `[[case]]` key re-parses to `true`, so the round-trip is lossless).
fn is_true(b: &bool) -> bool {
    *b
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
            // A `Question` has already lost whatever the parser did not model, so nothing can be
            // recovered here. This path builds NEW cases (distil); the preserving path is
            // `load_cases`, which reads the file straight into `Case`.
            extra: toml::Table::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CaseFile {
    #[serde(default)]
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

/// Parse a `dataset.toml` file into lossless [`Case`]s.
///
/// Reads the TOML straight into `Case` rather than going through `parse_dataset_toml`'s
/// `Question`: `Question` models only what the eval pipeline consumes, so routing through it threw
/// away every other key before `write_cases` could put it back. The shared parser is still the
/// authority for what a case MEANS — it is validated against here — but it is not the right shape
/// for a read-modify-write.
pub fn load_cases(path: &Path) -> Result<Vec<Case>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // Parse through the shared parser too, purely so a file that the eval pipeline would reject
    // fails here with the same message instead of passing dedup and failing later.
    parse_dataset_toml(&text)?;
    let file: CaseFile = toml::from_str(&text).context("parsing dataset.toml")?;
    Ok(file.case)
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
    pub unanswerable: usize,
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
/// in {"", "lexical", "multihop"}, `needs_graph` in {"", "yes", "no", "maybe"}, and unique ids.
/// Returns every issue found (empty = clean). A TOML/parse error is caught earlier, when the file
/// is loaded.
///
/// These two value sets are the authority for the template and the field docs: an out-of-range
/// value used to pass validation and then vanish into the report's "(untyped)"/"(unset)" bucket,
/// so a typo cost a slice silently.
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
        // `needs_graph` was documented as `yes|no|maybe` and never checked, so a typo became
        // "(unset)" in the report's by-type section — the slice silently lost a case instead of
        // naming the problem. Same shape as `hop_type` above, and the same reason for checking.
        if !matches!(c.needs_graph.as_str(), "" | "yes" | "no" | "maybe") {
            issues.push(ValidationIssue {
                id: c.id.clone(),
                problem: format!(
                    "invalid needs_graph {:?} (want \"\"|yes|no|maybe)",
                    c.needs_graph
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported data loss, pinned: `dedup` rewrites the file, so any key the serializer does
    /// not emit is deleted from the user's dataset. Real datasets carry `distilled_query` and a
    /// per-case `abstention` flag; neither is modelled here, and both used to vanish from a
    /// command asked only to remove duplicates.
    #[test]
    fn a_rewrite_keeps_keys_this_crate_does_not_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset.toml");
        std::fs::write(
            &path,
            r#"
[[case]]
id = "a"
question = "Q1?"
answer = "A1"
hop_type = "lexical"
distilled_query = "short side, fits the NLI window"
abstention = true

[[case]]
id = "b"
question = "Q1?"
answer = "A1"
reviewed_by = "ops"
confidence = 3
"#,
        )
        .unwrap();

        let loaded = load_cases(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded[0]
                .extra
                .get("distilled_query")
                .and_then(|v| v.as_str()),
            Some("short side, fits the NLI window"),
            "an unmodelled key must survive the READ"
        );
        assert_eq!(
            loaded[0]
                .extra
                .get("abstention")
                .and_then(toml::Value::as_bool),
            Some(true)
        );
        assert!(
            !loaded[0].extra.contains_key("hop_type"),
            "a modelled field belongs in its own slot, not in extra"
        );

        // Dedup drops the duplicate question and rewrites the whole file — the destructive path.
        let (kept, removed) = dedup_cases(&loaded);
        assert_eq!(kept.len(), 1, "the two cases share a question");
        assert_eq!(removed, 1);
        write_cases(&path, &kept).unwrap();

        let after = std::fs::read_to_string(&path).unwrap();
        assert!(
            after.contains("distilled_query"),
            "the rewrite deleted a hand-written field:
{after}"
        );
        assert!(
            after.contains("abstention"),
            "same, for the abstention flag"
        );

        let reread = load_cases(&path).unwrap();
        assert_eq!(
            reread[0]
                .extra
                .get("distilled_query")
                .and_then(|v| v.as_str()),
            Some("short side, fits the NLI window"),
            "and it must come back identical on the next read"
        );
    }

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
            extra: toml::Table::new(),
        }
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
            extra: toml::Table::new(),
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
            {
                // Documented as yes|no|maybe and never checked, so a typo passed validation and
                // then disappeared into the report's "(unset)" bucket — the slice lost a case
                // without saying so.
                let mut c = case("w", "Q?", "A");
                c.needs_graph = "probably".into();
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
        assert!(probs
            .iter()
            .any(|(id, p)| *id == "w" && p.contains("invalid needs_graph")));
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
        ];
        let s = compute_stat(&cases);
        assert_eq!(s.total, 3);
        assert_eq!(s.lexical, 1);
        assert_eq!(s.multihop, 1);
        assert_eq!(s.untyped, 1);
        assert_eq!(s.answerable, 2);
        assert_eq!(s.unanswerable, 1);
        assert_eq!(s.with_aliases, 1);
        assert_eq!(s.without_aliases, 2);
        assert_eq!(s.dup_questions, 1, "c duplicates a's normalized question");
        assert_eq!(s.dup_answers, 1, "c duplicates a's answer");
        assert_eq!(s.blank, 0);
        assert_eq!(s.needs_graph.get("yes"), Some(&1));
        assert_eq!(s.needs_graph.get("(unset)"), Some(&2));
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
