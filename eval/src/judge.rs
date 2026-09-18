//! File-prompt LLM judge: a `judge.md` system prompt drives an OpenAI-compatible endpoint to
//! grade one case (question/gold/answer) as correct/partial/wrong, via a fixed `VERDICT:` line
//! the harness parses back out. Reuses the same chat client the agent backend drives
//! (`backend::openai::chat_once`) so judge calls hit the endpoint the same way.

use crate::lab::Endpoint;
use anyhow::Context;
use glossa::index::store::DocIndex;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Verdict {
    Correct,
    Partial,
    Wrong,
    Unscored,
}

#[derive(Debug, Clone)]
pub struct Judgement {
    pub verdict: Verdict,
    pub reason: String,
    pub raw: String,
}

/// Parse the LAST `VERDICT:` occurrence in `reply` (case-insensitive), whatever precedes it becomes
/// the `reason`. The marker is matched ANYWHERE — not only at the start of a line — because models
/// frequently inline it after the reason on the same line ("reason. VERDICT: wrong"); requiring a
/// line start silently dropped those to `Unscored`. No `VERDICT:` at all → `Unscored`. An
/// unrecognized value after `VERDICT:` also falls back to `Unscored` (raw reply is always carried).
pub fn parse_verdict(reply: &str) -> Judgement {
    const MARKER: &str = "verdict:";
    // ASCII-only lowercasing: `to_lowercase()` can change a char's byte length for some Unicode
    // (e.g. İ U+0130 -> 2-char/3-byte), which would desync `pos` (an index into the lowercased
    // string) from `reply`'s own byte offsets and panic on a non-char-boundary slice below.
    // `to_ascii_lowercase()` is always 1:1 in byte length, and the marker itself is pure ASCII, so
    // matching still works identically for every reply that actually contains "VERDICT:"/"verdict:".
    let lower = reply.to_ascii_lowercase();
    let (verdict, reason) = match lower.rfind(MARKER) {
        Some(pos) => {
            // The verdict word is the first alphabetic token after the marker (stops at the newline
            // / punctuation / the `correct|partial|wrong` menu separators the prompt uses).
            let after = &reply[pos + MARKER.len()..];
            let token: String = after
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphabetic())
                .flat_map(char::to_lowercase)
                .collect();
            let v = match token.as_str() {
                "correct" => Verdict::Correct,
                "partial" => Verdict::Partial,
                "wrong" => Verdict::Wrong,
                _ => Verdict::Unscored,
            };
            (v, reply[..pos].trim().to_string())
        }
        None => (Verdict::Unscored, reply.trim().to_string()),
    };
    Judgement {
        verdict,
        reason,
        raw: reply.to_string(),
    }
}

/// Split a corpus chunk ref `"<path>#<location>"` into `(path, location)` on the LAST `#`
/// (`rsplit_once`), so a path that itself contains `#` still resolves against its final locator.
/// A ref with no `#`, or with an empty path/location, cannot address a chunk → `None` (that ref is
/// skipped; it never blocks the others).
fn parse_ref(reference: &str) -> Option<(&str, &str)> {
    let (path, loc) = reference.trim().rsplit_once('#')?;
    let (path, loc) = (path.trim(), loc.trim());
    if path.is_empty() || loc.is_empty() {
        None
    } else {
        Some((path, loc))
    }
}

/// Load the source chunks named by `source` from the corpus `idx`, returning `(ref, text)` pairs
/// in input order. Refs that don't parse or whose chunk can't be read are silently skipped. When
/// `idx` is `None` or `source` is empty, the result is empty — the caller then omits the EVIDENCE
/// block entirely, so the judge prompt is byte-identical to the gold-only form.
fn load_evidence(source: &[String], idx: Option<&DocIndex>) -> Vec<(String, String)> {
    let Some(idx) = idx else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for reference in source {
        let Some((path, loc)) = parse_ref(reference) else {
            continue;
        };
        if let Ok(Some(text)) = idx.read_chunk(path, loc) {
            out.push((reference.trim().to_string(), text));
        }
    }
    out
}

/// Format loaded `(ref, text)` snippets into an `EVIDENCE:` block, each snippet labeled with its
/// `path#location`. Empty input → `None`, so the caller drops the block and the prompt stays
/// gold-only. The returned block carries no trailing newline (it is placed between `GOLD:` and
/// `ANSWER:` lines by the caller).
fn evidence_block(snippets: &[(String, String)]) -> Option<String> {
    if snippets.is_empty() {
        return None;
    }
    let mut s = String::from("EVIDENCE:");
    for (reference, text) in snippets {
        s.push_str(&format!("\n[{reference}]\n{}", text.trim()));
    }
    Some(s)
}

/// The grading rules parsed out of a sectioned `judge.md`. `preamble` is the general framing sent
/// as the SYSTEM message; the other three are per-case rule blocks `build_user` places in the USER
/// message next to the data (the judge model reliably follows a rule placed there, not one buried
/// only in the system prompt). A legacy UNSECTIONED `judge.md` (no `[[...]]` markers) parses with
/// its whole text landing in `preamble` and every rule empty — see `split_judge_sections`.
struct Sections {
    preamble: String,
    dialogue: String,
    answerable: String,
    abstention: String,
}

/// Split a sectioned `judge.md` on the marker lines `[[DIALOGUE]]`, `[[ANSWERABLE]]`,
/// `[[ABSTENTION]]` (each on its own line). Text before the first marker is the PREAMBLE. Missing
/// markers yield empty rule sections, so a legacy unsectioned `judge.md` still parses (all its text
/// lands in `preamble`; `build_user` falls back to the historical hardcoded rule text in that case —
/// see its doc comment).
fn split_judge_sections(md: &str) -> Sections {
    let mut cur = 0u8; // 0=preamble 1=dialogue 2=answerable 3=abstention
    let (mut pre, mut dlg, mut ans, mut abs) =
        (String::new(), String::new(), String::new(), String::new());
    for line in md.lines() {
        match line.trim() {
            "[[DIALOGUE]]" => {
                cur = 1;
                continue;
            }
            "[[ANSWERABLE]]" => {
                cur = 2;
                continue;
            }
            "[[ABSTENTION]]" => {
                cur = 3;
                continue;
            }
            _ => {}
        }
        let bucket = match cur {
            1 => &mut dlg,
            2 => &mut ans,
            3 => &mut abs,
            _ => &mut pre,
        };
        bucket.push_str(line);
        bucket.push('\n');
    }
    Sections {
        preamble: pre.trim().to_string(),
        dialogue: dlg.trim().to_string(),
        answerable: ans.trim().to_string(),
        abstention: abs.trim().to_string(),
    }
}

/// Format the captured reader<->user_sim `(role, text)` turns into a `DIALOGUE:` block. Empty input
/// -> `None`, so the caller omits the block and the message is byte-identical to a no-dialogue case.
fn dialogue_block(dialogue: &[(String, String)]) -> Option<String> {
    if dialogue.is_empty() {
        return None;
    }
    let mut s = String::from("DIALOGUE:");
    for (role, text) in dialogue {
        s.push_str(&format!("\n[{role}] {}", text.trim()));
    }
    Some(s)
}

/// Fallback abstention-case grading rule used when `judge_md` has no `[[ABSTENTION]]` section (a
/// legacy unsectioned prompt) — the exact rubric that shipped hardcoded in Rust before sectioning,
/// so an old `judge.md` keeps grading unanswerable cases identically to before this change.
const LEGACY_ABSTENTION_RULE: &str = "Grade `correct` if the ANSWER appropriately declines or \
     states there is no answer; `wrong` if it gives a substantive or fabricated technical answer \
     as if it knew; `partial` if it declines but still adds unsupported specific claims.";

/// Build the judge user message. With `evidence`, the block is injected between `GOLD:` and
/// `ANSWER:`. Without it (`None`), the message is byte-identical to the historical gold-only form.
///
/// When `answerable` is `false` the case is an ABSTENTION test: the question cannot be answered from
/// the knowledge base (out of scope / not covered / a routing or non-technical request), so there is
/// no gold text to compare against. The correct behavior is for the reader to DECLINE — say it has no
/// answer / the info isn't in the KB / route to a human — without inventing a technical answer. The
/// message then carries the abstention rubric (`sections.abstention`, or `LEGACY_ABSTENTION_RULE`
/// when that section is empty) instead of GOLD/EVIDENCE. For an answerable case the rubric is
/// `sections.answerable` (empty for a legacy unsectioned prompt — the general grading rule then lives
/// only in the system-side preamble, exactly as it did historically).
///
/// When `dialogue` is non-empty (a user_sim conversation actually happened) a `DIALOGUE:` block is
/// inserted right after ANSWER, and — when `sections.dialogue` is present — its dialogue-specific
/// rule is appended at the very end, so the judge grades the reader's actual substantive answer
/// across the conversation rather than a closing pleasantry.
///
/// BACKWARD COMPATIBILITY (required): with an unsectioned `judge_md` (so `sections.answerable` and
/// `sections.dialogue` are empty) and empty `dialogue`, every new field above contributes nothing and
/// this reproduces the pre-dialogue message byte-for-byte — the `abstain_note`/reply-line ORDER is
/// therefore kept exactly as before (reply line, then `abstain_note`) rather than reordered, since
/// only that ordering degenerates correctly when nothing new is present.
#[allow(clippy::too_many_arguments)]
fn build_user(
    question: &str,
    gold: &str,
    answer: &str,
    evidence: Option<&str>,
    answerable: bool,
    credit_abstention: bool,
    dialogue: &[(String, String)],
    sections: &Sections,
) -> String {
    let reply = "Reply with one line reason then `VERDICT: correct|partial|wrong`.";
    // DIALOGUE data block (empty unless a user_sim conversation happened) and its rule (empty unless
    // BOTH a dialogue happened AND the template supplies a [[DIALOGUE]] section).
    let dlg_data = match dialogue_block(dialogue) {
        Some(d) => format!("{d}\n"),
        None => String::new(),
    };
    let dlg_rule = if dialogue.is_empty() || sections.dialogue.is_empty() {
        String::new()
    } else {
        format!("\n{}", sections.dialogue)
    };
    // The abstention branch places `dlg_rule` BEFORE the reply line, so it needs a trailing
    // separator when present (otherwise the rule glues onto "Reply with…"); the answerable branch
    // places `dlg_rule` last and needs none. Empty when `dlg_rule` is empty, so the legacy /
    // no-dialogue message stays byte-identical.
    let dlg_sep = if dlg_rule.is_empty() { "" } else { "\n" };

    if !answerable {
        let abs_rule: &str = if sections.abstention.is_empty() {
            LEGACY_ABSTENTION_RULE
        } else {
            &sections.abstention
        };
        return format!(
            "QUESTION: {question}\n\
             NOTE: This question CANNOT be answered from the knowledge base — it is out of scope, not \
             covered by the corpus, or a non-technical / routing request. There is no gold answer.\n\
             The CORRECT behavior is for the assistant to state it has no answer / the information is \
             not in the knowledge base, or to decline or route to a human — WITHOUT inventing a \
             technical answer.\n\
             ANSWER: {answer}\n\
             {dlg_data}{abs_rule}\n\
             {dlg_rule}{dlg_sep}{reply}"
        );
    }
    // Safety-first only: a decline on an ANSWERABLE question is a SAFE miss, not a fabrication — grade
    // it `partial` rather than `wrong`, so the reward teaches "decline when unsure" over "guess wrong".
    // Under the default (balanced) policy this note is omitted and a decline grades `wrong` (a miss),
    // byte-identical to the historical prompt.
    let abstain_note = if credit_abstention {
        "\nNote: if the ANSWER declines / says the information is not in the knowledge base instead of \
         answering, grade `partial` (a safe miss), NOT `wrong`; reserve `wrong` for an INCORRECT \
         substantive answer."
    } else {
        ""
    };
    let gold_ev = match evidence {
        Some(ev) => format!("GOLD: {gold}\n{ev}\n"),
        None => format!("GOLD: {gold}\n"),
    };
    // `sections.answerable` is empty for a legacy unsectioned judge.md, so `ans_rule` contributes
    // nothing then — the general grading rule lives only in the system preamble, as it always has.
    let ans_rule = if sections.answerable.is_empty() {
        String::new()
    } else {
        format!("{}\n", sections.answerable)
    };
    format!(
        "QUESTION: {question}\n{gold_ev}ANSWER: {answer}\n\
         {dlg_data}{ans_rule}{reply}{abstain_note}{dlg_rule}"
    )
}

/// Default number of independent samples `judge()` draws per case before taking the majority
/// verdict (see `majority_verdict`). Voting is INTRINSIC to `judge()` — every caller (eval's
/// `kbx.rs` and train's `gepa_graph.rs`) votes automatically; there is no per-call opt-out or
/// per-call votes argument to forget. Override the ONE knob for a run via the `KB_EVAL_JUDGE_VOTES`
/// env var (mirrors the existing `KB_EVAL_TEMP` override — see `Endpoint::resolve_temperature`);
/// an unset or unparseable value falls back to this default.
const DEFAULT_JUDGE_VOTES: usize = 5;

/// Resolve how many times `judge()` samples the endpoint before taking the majority verdict:
/// `KB_EVAL_JUDGE_VOTES` env var if set and it parses to a `usize`, else `DEFAULT_JUDGE_VOTES`.
/// Always at least 1 (a caller setting `KB_EVAL_JUDGE_VOTES=0` still gets a single sample, not zero
/// judge calls).
pub(crate) fn resolve_judge_votes() -> usize {
    std::env::var("KB_EVAL_JUDGE_VOTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_JUDGE_VOTES)
        .max(1)
}

/// Judge one case. The (optionally sectioned — see `split_judge_sections`) `judge_md` supplies the
/// system-side PREAMBLE plus the per-case grading RULE that `build_user` places in the USER message
/// next to the data. When `source` names corpus chunks and `idx` is supplied, their text is injected
/// as an `EVIDENCE:` block between GOLD and ANSWER so a correct answer that EXCEEDS the terse gold is
/// credited, not penalized (see `evidence_block`). When `source` is empty, `idx` is `None`, or no ref
/// loads, the EVIDENCE block is omitted and the prompt is byte-identical to the gold-only form.
/// `dialogue` (empty unless a user_sim conversation happened — see `backend::openai::
/// take_reader_dialogue`) adds a `DIALOGUE:` block plus its rule so the judge grades the reader's
/// actual substantive answer across the conversation, not a closing pleasantry.
///
/// The endpoint is sampled `resolve_judge_votes()` times (default `DEFAULT_JUDGE_VOTES`, always at
/// least 1) with the SAME message, and the majority verdict across those samples is returned (see
/// `majority_verdict`) — this tames the judge's run-to-run non-determinism (the model flips a
/// borderline verdict between otherwise-identical calls even at temperature 0). Each individual
/// sample still goes through `chat_once_resampled`, which separately guards against a degenerate
/// (empty/truncated) single reply; voting is an orthogonal layer on top of that.
#[allow(clippy::too_many_arguments)]
pub fn judge(
    ep: &Endpoint,
    judge_md: &str,
    question: &str,
    gold: &str,
    answer: &str,
    source: &[String],
    answerable: bool,
    credit_abstention: bool,
    idx: Option<&DocIndex>,
    dialogue: &[(String, String)],
) -> anyhow::Result<Judgement> {
    // Trim the embedded fields so the judge message stays tidy and never ends on a stray newline
    // (some strict providers reject a message ending in `\n` — see prompt::user_prompt).
    let (question, gold, answer) = (question.trim(), gold.trim(), answer.trim());
    let sections = split_judge_sections(judge_md);
    let snippets = load_evidence(source, idx);
    let evidence = evidence_block(&snippets);
    let user = build_user(
        question,
        gold,
        answer,
        evidence.as_deref(),
        answerable,
        credit_abstention,
        dialogue,
        &sections,
    );
    // The messages are built ONCE and resampled `votes` times below — voting stays intrinsic to
    // `judge()` (no separate `votes` parameter), unaffected by the dialogue/sectioning changes above.
    // System content is `sections.preamble`. For a legacy (unsectioned) judge.md that is the whole
    // file with leading/trailing whitespace trimmed and line endings normalized to `\n` (see
    // `split_judge_sections`) — a CONSCIOUS, accepted difference from sending the raw `judge_md`
    // byte-for-byte: it is functionally inert for the grader (whitespace/CRLF only), while the USER
    // message stays byte-identical under legacy+no-dialogue (pinned by `build_user_without_dialogue_matches_legacy`).
    let messages = vec![
        json!({ "role": "system", "content": sections.preamble }),
        json!({ "role": "user", "content": user }),
    ];
    let votes = resolve_judge_votes();
    let mut ballots = Vec::with_capacity(votes);
    for _ in 0..votes {
        let msg = crate::backend::openai::chat_once_resampled(ep, &messages)
            .context("judge endpoint request failed")?;
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        ballots.push(parse_verdict(content));
    }
    Ok(majority_verdict(&ballots))
}

/// The majority `Judgement` over `ballots`: the verdict with the most votes wins; ties break by
/// severity (`Wrong > Partial > Correct > Unscored`) so a split never silently favors a pass. The
/// returned reason is that of the first ballot carrying the winning verdict. Empty input → `Unscored`.
fn majority_verdict(ballots: &[Judgement]) -> Judgement {
    use std::collections::HashMap;
    if ballots.is_empty() {
        return parse_verdict("");
    }
    let mut counts: HashMap<Verdict, usize> = HashMap::new();
    for j in ballots {
        *counts.entry(j.verdict).or_default() += 1;
    }
    let severity = |v: &Verdict| match v {
        Verdict::Wrong => 3,
        Verdict::Partial => 2,
        Verdict::Correct => 1,
        Verdict::Unscored => 0,
    };
    let winner = *counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| severity(a.0).cmp(&severity(b.0))))
        .map(|(v, _)| v)
        .expect("non-empty counts");
    ballots
        .iter()
        .find(|j| j.verdict == winner)
        .cloned()
        .expect("winner came from the ballots")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_parsing() {
        assert!(matches!(
            parse_verdict("reason...\nVERDICT: correct").verdict,
            Verdict::Correct
        ));
        assert!(matches!(
            parse_verdict("VERDICT: Partial").verdict,
            Verdict::Partial
        ));
        assert!(matches!(
            parse_verdict("blah\nverdict: WRONG\n").verdict,
            Verdict::Wrong
        ));
        assert!(matches!(
            parse_verdict("no verdict here").verdict,
            Verdict::Unscored
        ));
        // last VERDICT wins
        assert!(matches!(
            parse_verdict("VERDICT: wrong\nVERDICT: correct").verdict,
            Verdict::Correct
        ));
        // INLINE verdict on the same line as the reason (not only at line start): must still
        // parse, and the reason is everything before the marker.
        let inline = parse_verdict("The answer is empty, providing no information. VERDICT: wrong");
        assert!(matches!(inline.verdict, Verdict::Wrong));
        assert_eq!(
            inline.reason,
            "The answer is empty, providing no information."
        );
        // trailing punctuation / menu separators after the word don't break it
        assert!(matches!(
            parse_verdict("ok. Verdict: correct.").verdict,
            Verdict::Correct
        ));
    }

    /// Regression: a non-ASCII char BEFORE the marker must not desync the byte offset computed
    /// from the lowercased copy against the original `reply` (e.g. a naive `to_lowercase()` can
    /// grow some Unicode chars, like İ U+0130 -> 2-char/3-byte, shifting byte-length). Must parse
    /// without panicking and still find the verdict.
    #[test]
    fn parse_verdict_non_ascii_before_marker_does_not_panic() {
        let j = parse_verdict("café VERDICT: correct");
        assert!(matches!(j.verdict, Verdict::Correct));
        assert_eq!(j.reason, "café");
        // The specific char known to change byte length under full Unicode lowercasing.
        let j2 = parse_verdict("İstanbul café review. VERDICT: wrong");
        assert!(matches!(j2.verdict, Verdict::Wrong));
    }

    #[test]
    fn majority_verdict_takes_mode_then_severity_on_tie() {
        let j = |v: &str| parse_verdict(&format!("reason\nVERDICT: {v}"));
        // Clear mode: 2 correct beats 1 wrong.
        assert_eq!(
            majority_verdict(&[j("correct"), j("correct"), j("wrong")]).verdict,
            Verdict::Correct
        );
        // Three-way tie -> severity: Wrong wins.
        assert_eq!(
            majority_verdict(&[j("correct"), j("partial"), j("wrong")]).verdict,
            Verdict::Wrong
        );
        // Empty -> Unscored.
        assert_eq!(majority_verdict(&[]).verdict, Verdict::Unscored);
    }

    #[test]
    fn ref_parsing_rsplits_on_last_hash() {
        assert_eq!(parse_ref("a.pdf#p.1"), Some(("a.pdf", "p.1")));
        // rsplit on the LAST '#': a path containing '#' still resolves against its final locator.
        assert_eq!(parse_ref("dir/a#b.pdf#p.2"), Some(("dir/a#b.pdf", "p.2")));
        // Surrounding whitespace is trimmed off both sides.
        assert_eq!(parse_ref("  a.pdf # p.1 "), Some(("a.pdf", "p.1")));
        // A path without '#' cannot address a chunk location → None (skipped, gold-only).
        assert_eq!(parse_ref("a.pdf"), None);
        // Empty path or empty location → None.
        assert_eq!(parse_ref("#p.1"), None);
        assert_eq!(parse_ref("a.pdf#"), None);
    }

    #[test]
    fn evidence_block_labels_each_snippet_or_none_when_empty() {
        // Empty → None, so the caller omits the block entirely (gold-only prompt).
        assert!(evidence_block(&[]).is_none());
        let snippets = vec![
            ("a.pdf#p.1".to_string(), "chunk one".to_string()),
            ("b.pdf#p.2".to_string(), "chunk two".to_string()),
        ];
        assert_eq!(
            evidence_block(&snippets).unwrap(),
            "EVIDENCE:\n[a.pdf#p.1]\nchunk one\n[b.pdf#p.2]\nchunk two"
        );
    }

    /// A legacy, entirely UNSECTIONED `judge.md` (no `[[...]]` markers at all): the whole blob lands
    /// in `preamble`, every rule section is empty. `build_user` must fall back to the pre-sectioning
    /// hardcoded behavior when given this — see `build_user_without_dialogue_matches_legacy`.
    fn legacy_sections() -> Sections {
        split_judge_sections("You are a grading judge.")
    }

    /// A minimal sectioned judge.md for build_user tests — each rule a distinct sentinel string so
    /// tests can assert which one landed in the message.
    fn test_sections() -> Sections {
        split_judge_sections(
            "PREAMBLE-TEXT\n[[DIALOGUE]]\nDIALOGUE-RULE\n[[ANSWERABLE]]\nANSWERABLE-RULE\n[[ABSTENTION]]\nABSTENTION-RULE\n",
        )
    }

    #[test]
    fn split_judge_sections_extracts_four_parts_or_falls_back_to_legacy() {
        let s = test_sections();
        assert_eq!(s.preamble, "PREAMBLE-TEXT");
        assert_eq!(s.dialogue, "DIALOGUE-RULE");
        assert_eq!(s.answerable, "ANSWERABLE-RULE");
        assert_eq!(s.abstention, "ABSTENTION-RULE");
        // Missing markers -> everything is preamble, rule sections empty (legacy prompt still parses).
        let legacy = legacy_sections();
        assert_eq!(legacy.preamble, "You are a grading judge.");
        assert!(
            legacy.dialogue.is_empty()
                && legacy.answerable.is_empty()
                && legacy.abstention.is_empty()
        );
    }

    #[test]
    fn split_judge_sections_marker_as_first_line_yields_empty_preamble() {
        // A prompt that opens directly on a marker has no preamble text before it.
        let s = split_judge_sections("[[ANSWERABLE]]\nONLY-RULE\n");
        assert_eq!(s.preamble, "");
        assert_eq!(s.answerable, "ONLY-RULE");
        assert!(s.dialogue.is_empty() && s.abstention.is_empty());
    }

    #[test]
    fn split_judge_sections_only_some_markers_present() {
        // Only ANSWERABLE is declared; the absent DIALOGUE/ABSTENTION sections stay empty (build_user
        // then falls back to LEGACY_ABSTENTION_RULE and omits the dialogue rule).
        let s = split_judge_sections("PRE\n[[ANSWERABLE]]\nANS\n");
        assert_eq!(s.preamble, "PRE");
        assert_eq!(s.answerable, "ANS");
        assert!(s.dialogue.is_empty() && s.abstention.is_empty());
    }

    #[test]
    fn dialogue_block_empty_is_none_and_two_turns_are_ordered() {
        assert!(dialogue_block(&[]).is_none());
        let dlg = vec![
            ("assistant".to_string(), "the real answer".to_string()),
            ("user".to_string(), "ok thanks".to_string()),
        ];
        let b = dialogue_block(&dlg).unwrap();
        assert!(b.starts_with("DIALOGUE:"));
        let asst_at = b.find("[assistant] the real answer").unwrap();
        let user_at = b.find("[user] ok thanks").unwrap();
        assert!(asst_at < user_at, "turns stay in input order");
    }

    #[test]
    fn user_prompt_with_evidence_injects_block_between_gold_and_answer() {
        let s = legacy_sections();
        // Stubbed chunk text (mock) — no corpus, no network.
        let snippets = vec![("a.pdf#p.1".to_string(), "stub evidence text".to_string())];
        let ev = evidence_block(&snippets);
        let prompt = build_user("Q?", "G", "A", ev.as_deref(), true, false, &[], &s);
        assert!(prompt.contains("EVIDENCE:\n[a.pdf#p.1]\nstub evidence text"));
        // Block sits between GOLD and ANSWER.
        let gold_at = prompt.find("GOLD: G").unwrap();
        let ev_at = prompt.find("EVIDENCE:").unwrap();
        let ans_at = prompt.find("ANSWER: A").unwrap();
        assert!(gold_at < ev_at && ev_at < ans_at);
    }

    #[test]
    fn build_user_unanswerable_uses_abstention_rubric_not_gold() {
        let s = legacy_sections();
        // answerable=false → abstention rubric, no GOLD/EVIDENCE (there is no gold to compare).
        let u = build_user(
            "Q?",
            "",
            "not in the knowledge base",
            None,
            false,
            false,
            &[],
            &s,
        );
        assert!(
            u.contains("CANNOT be answered"),
            "carries the abstention note"
        );
        assert!(
            u.contains("declines"),
            "grades on declining, not on gold match"
        );
        assert!(
            !u.contains("GOLD:"),
            "unanswerable prompt must not carry a GOLD line"
        );
        assert!(u.contains("ANSWER: not in the knowledge base"));
        // answerable=true still emits the gold-anchored form.
        let a = build_user("Q?", "G", "A", None, true, false, &[], &s);
        assert!(a.contains("GOLD: G"));
        assert!(!a.contains("CANNOT be answered"));
    }

    #[test]
    fn build_user_credit_abstention_adds_safe_miss_note_only_when_enabled() {
        let s = legacy_sections();
        // Off (credit_abstention=false): no safe-miss note — a decline stays a miss (`wrong`).
        let balanced = build_user("Q?", "G", "A", None, true, false, &[], &s);
        assert!(!balanced.contains("safe miss"));
        // On (credit_abstention=true): the answerable prompt tells the judge to grade a
        // decline as `partial`, not `wrong`.
        let safety = build_user("Q?", "G", "A", None, true, true, &[], &s);
        assert!(safety.contains("safe miss"));
        // The note is answerable-only: an unanswerable prompt is unaffected by credit_abstention.
        assert_eq!(
            build_user("Q?", "", "A", None, false, false, &[], &s),
            build_user("Q?", "", "A", None, false, true, &[], &s)
        );
    }

    #[test]
    fn user_prompt_without_source_is_byte_identical_to_gold_only() {
        let s = legacy_sections();
        // With no evidence, build_user must reproduce the historical gold-only message byte-for-byte.
        let got = build_user("Q?", "G", "A", None, true, false, &[], &s);
        let expected = format!(
            "QUESTION: {}\nGOLD: {}\nANSWER: {}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`.",
            "Q?", "G", "A"
        );
        assert_eq!(got, expected);
        // An empty `source` with no index loads no evidence → block omitted → gold-only path.
        assert!(evidence_block(&load_evidence(&[], None)).is_none());
    }

    /// Pins backward-compat end to end: for an unsectioned `judge.md` and empty `dialogue`, the
    /// built user message equals the pre-dialogue assembly (no DIALOGUE block, no extra rule) for
    /// BOTH the answerable and abstention branches, regardless of `credit_abstention`.
    #[test]
    fn build_user_without_dialogue_matches_legacy() {
        let s = legacy_sections();
        for credit_abstention in [false, true] {
            let answerable_expected = {
                let abstain_note = if credit_abstention {
                    "\nNote: if the ANSWER declines / says the information is not in the knowledge base instead of \
                     answering, grade `partial` (a safe miss), NOT `wrong`; reserve `wrong` for an INCORRECT \
                     substantive answer."
                } else {
                    ""
                };
                format!(
                    "QUESTION: Q?\nGOLD: G\nANSWER: A\n\
                     Reply with one line reason then `VERDICT: correct|partial|wrong`.{abstain_note}"
                )
            };
            assert_eq!(
                build_user("Q?", "G", "A", None, true, credit_abstention, &[], &s),
                answerable_expected
            );

            let abstention_expected = "QUESTION: Q?\n\
                 NOTE: This question CANNOT be answered from the knowledge base — it is out of scope, not \
                 covered by the corpus, or a non-technical / routing request. There is no gold answer.\n\
                 The CORRECT behavior is for the assistant to state it has no answer / the information is \
                 not in the knowledge base, or to decline or route to a human — WITHOUT inventing a \
                 technical answer.\n\
                 ANSWER: A\n\
                 Grade `correct` if the ANSWER appropriately declines or states there is no answer; \
                 `wrong` if it gives a substantive or fabricated technical answer as if it knew; \
                 `partial` if it declines but still adds unsupported specific claims.\n\
                 Reply with one line reason then `VERDICT: correct|partial|wrong`."
                .to_string();
            assert_eq!(
                build_user("Q?", "", "A", None, false, credit_abstention, &[], &s),
                abstention_expected
            );
        }
        // No DIALOGUE block, no dialogue rule, in either branch.
        let a = build_user("Q?", "G", "A", None, true, false, &[], &s);
        let u = build_user("Q?", "", "A", None, false, false, &[], &s);
        assert!(!a.contains("DIALOGUE:") && !u.contains("DIALOGUE:"));
    }

    /// With a sectioned judge.md and a non-empty dialogue, the message carries BOTH the DIALOGUE
    /// block and the dialogue rule; with empty dialogue neither appears, even though the section
    /// exists — the block/rule are dialogue-gated, not just section-gated.
    #[test]
    fn build_user_with_dialogue_adds_block_and_rule() {
        let s = test_sections();
        let dlg = vec![
            ("assistant".to_string(), "the real answer".to_string()),
            ("user".to_string(), "thanks".to_string()),
        ];
        let with_dlg = build_user("Q?", "G", "A", None, true, false, &dlg, &s);
        assert!(with_dlg.contains("DIALOGUE:"));
        assert!(with_dlg.contains("[assistant] the real answer"));
        assert!(with_dlg.contains("DIALOGUE-RULE"));
        assert!(with_dlg.contains("ANSWERABLE-RULE"));

        let without_dlg = build_user("Q?", "G", "A", None, true, false, &[], &s);
        assert!(!without_dlg.contains("DIALOGUE:"));
        assert!(!without_dlg.contains("DIALOGUE-RULE"));
        assert!(without_dlg.contains("ANSWERABLE-RULE"));

        // Same for the abstention branch — and here the dialogue rule sits BEFORE the reply line,
        // so pin that it is SEPARATED from it (a regression guard: a missing separator once glued
        // the rule onto "Reply with…" on the unanswerable+dialogue path).
        let with_dlg_abs = build_user("Q?", "", "A", None, false, false, &dlg, &s);
        assert!(with_dlg_abs.contains("DIALOGUE:"));
        assert!(with_dlg_abs.contains("DIALOGUE-RULE"));
        assert!(with_dlg_abs.contains("ABSTENTION-RULE"));
        assert!(
            with_dlg_abs.contains("DIALOGUE-RULE\nReply"),
            "dialogue rule must be newline-separated from the reply line: {with_dlg_abs}"
        );
        assert!(
            !with_dlg_abs.contains("DIALOGUE-RULEReply"),
            "dialogue rule must not glue onto the reply line: {with_dlg_abs}"
        );
    }

    // Serializes tests that mutate the process-wide `KB_EVAL_JUDGE_VOTES` env var, so a parallel
    // test run never sees a partially-set value from a sibling test (same pattern as
    // `TOKEN_TEST_LOCK` in `backend::openai` / `RESAMPLE_TEST_LOCK` in `backend::resample`).
    static JUDGE_VOTES_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// End-to-end: `judge()` actually samples the endpoint `KB_EVAL_JUDGE_VOTES` times (not once)
    /// and returns the MAJORITY verdict across those samples, not the first/last one. A mock HTTP
    /// server (same pattern as `backend::openai`'s `chat_http` integration tests) serves three
    /// distinct replies — wrong, correct, correct — for one `judge()` call; 2-of-3 is `correct`,
    /// which must win even though the FIRST sample was `wrong`.
    #[test]
    fn judge_samples_n_times_and_returns_the_majority_verdict() {
        let _g = JUDGE_VOTES_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        use std::io::{Read, Write};
        use std::net::TcpListener;

        std::env::set_var("KB_EVAL_JUDGE_VOTES", "3");

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        // 1 wrong, 2 correct: the majority (correct) must win despite the first ballot being wrong.
        let replies = [
            "the answer misses the key fact. VERDICT: wrong",
            "the answer matches the gold. VERDICT: correct",
            "the answer matches the gold. VERDICT: correct",
        ];
        let server = std::thread::spawn(move || {
            for reply in replies {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).unwrap();
                let body = serde_json::json!({
                    "choices": [{
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }]
                })
                .to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).unwrap();
            }
        });

        let ep = Endpoint {
            endpoint: format!("http://127.0.0.1:{port}/v1/chat/completions"),
            model: "m".to_string(),
            api_key: String::new(),
            api_key_env: String::new(),
            timeout_secs: 5,
            api: crate::lab::ApiKind::default(),
            temperature: None,
            rate_limit: None,
            fallback: Vec::new(),
            function_name: None,
            feedback_score_metric: None,
            feedback_bool_metric: None,
            headers: std::collections::BTreeMap::new(),
        };

        let result = judge(
            &ep,
            "You are a grading judge.",
            "What is the capital of France?",
            "Paris",
            "Paris",
            &[],
            true,
            false,
            None,
            &[],
        )
        .unwrap();

        server.join().unwrap();
        std::env::remove_var("KB_EVAL_JUDGE_VOTES");

        assert_eq!(
            result.verdict,
            Verdict::Correct,
            "2-of-3 correct ballots must win the majority, not the first (wrong) sample"
        );
    }
}
