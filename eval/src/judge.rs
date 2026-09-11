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
    let lower = reply.to_lowercase();
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

/// The grading rules parsed out of a sectioned `judge.md`. `preamble` is light framing sent as the
/// SYSTEM message; the other three are the per-case rule blocks placed in the USER message next to
/// the data (the judge model reliably follows a rule in the user message, not one buried in the
/// system prompt — see the design notes on the plan).
struct Sections {
    preamble: String,
    dialogue: String,
    answerable: String,
    abstention: String,
}

/// Split a sectioned `judge.md` on the marker lines `[[DIALOGUE]]`, `[[ANSWERABLE]]`, `[[ABSTENTION]]`.
/// Text before the first marker is the PREAMBLE. Missing markers yield empty sections, so a legacy
/// un-sectioned `judge.md` still parses (all its text lands in `preamble`, the user-side rule blocks
/// are empty, and grading falls back to whatever the preamble says).
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

/// Format the captured reader↔user_sim `(role, text)` turns into a `DIALOGUE:` block. Empty input →
/// `None`, so the caller omits the block and the message is byte-identical to a no-dialogue case.
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

/// Build the judge user message: the case DATA first, then (optionally) a `DIALOGUE:` block, then
/// the case's grading RULE (`sections.answerable` or `sections.abstention`), then the reply line.
/// The rule sits AFTER the data on purpose — the judge model follows a rule placed there, not one
/// left only in the system prompt. When `evidence` is present it is injected between `GOLD:` and
/// `ANSWER:`. When `answerable` is `false` there is no GOLD/EVIDENCE and the abstention rule applies.
/// When `dialogue` is non-empty (a user_sim conversation happened) the `DIALOGUE:` block and the
/// dialogue rule are added so the judge grades the reader's actual answer, not a closing pleasantry.
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
    // DIALOGUE block goes right after the ANSWER; its grading rule (if the section is present) trails
    // the case rule. Both appear only when a dialogue was actually captured.
    let dlg_data = match dialogue_block(dialogue) {
        Some(d) => format!("{d}\n"),
        None => String::new(),
    };
    let dlg_rule = if dialogue.is_empty() || sections.dialogue.is_empty() {
        String::new()
    } else {
        format!("\n{}", sections.dialogue)
    };

    if !answerable {
        let abs_rule = &sections.abstention;
        return format!(
            "QUESTION: {question}\n\
             NOTE: This question CANNOT be answered from the knowledge base; there is no GOLD.\n\
             ANSWER: {answer}\n\
             {dlg_data}{abs_rule}{dlg_rule}\n\
             {reply}"
        );
    }

    // Safety-first only: a decline on an ANSWERABLE question is a SAFE miss, not a fabrication — grade
    // it `partial` rather than `wrong`. Under the default (balanced) policy this note is omitted.
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
    let ans_rule = &sections.answerable;
    format!(
        "QUESTION: {question}\n{gold_ev}ANSWER: {answer}\n\
         {dlg_data}{ans_rule}{abstain_note}{dlg_rule}\n\
         {reply}"
    )
}

/// Judge one case. The sectioned `judge_md` (see `split_judge_sections`) supplies the system
/// PREAMBLE plus the per-case grading RULE that `build_user` places in the USER message next to the
/// data. When `source` names corpus chunks and `idx` is supplied, their text is injected as an
/// `EVIDENCE:` block between GOLD and ANSWER so a correct answer that EXCEEDS the terse gold is
/// credited, not penalized (see `evidence_block`). `dialogue` (empty unless a user_sim conversation
/// happened) adds a `DIALOGUE:` block so the judge grades the reader's actual answer. The endpoint
/// is sampled `votes` times and the majority verdict is returned (see `majority_verdict`), taming
/// the judge's run-to-run non-determinism.
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
    votes: usize,
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
    // The grading rule now lives in the USER message (next to the data); the system message carries
    // only the preamble framing. Vote over `votes` samples to tame the judge's run-to-run noise
    // (the model is non-deterministic even at temp 0); a single sample flaps on borderline cases.
    let messages = vec![
        json!({ "role": "system", "content": sections.preamble }),
        json!({ "role": "user", "content": user }),
    ];
    let mut ballots = Vec::with_capacity(votes.max(1));
    for _ in 0..votes.max(1) {
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
        // INLINE verdict on the same line as the reason (the empty-answer regression): must still
        // parse, and the reason is everything before the marker.
        let inline = parse_verdict("The answer is empty, providing no information. VERDICT: wrong");
        assert!(matches!(inline.verdict, Verdict::Wrong));
        assert_eq!(inline.reason, "The answer is empty, providing no information.");
        // trailing punctuation / menu separators after the word don't break it
        assert!(matches!(parse_verdict("ok. Verdict: correct.").verdict, Verdict::Correct));
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

    /// A minimal sectioned judge.md for build_user tests — each section a distinct sentinel so
    /// tests can assert which one landed in the message.
    fn test_sections() -> Sections {
        split_judge_sections(
            "PREAMBLE-TEXT\n[[DIALOGUE]]\nDIALOGUE-RULE\n[[ANSWERABLE]]\nANSWERABLE-RULE\n[[ABSTENTION]]\nABSTENTION-RULE\n",
        )
    }

    #[test]
    fn split_judge_sections_extracts_four_parts() {
        let s = test_sections();
        assert_eq!(s.preamble, "PREAMBLE-TEXT");
        assert_eq!(s.dialogue, "DIALOGUE-RULE");
        assert_eq!(s.answerable, "ANSWERABLE-RULE");
        assert_eq!(s.abstention, "ABSTENTION-RULE");
        // Missing markers -> everything is preamble, rule sections empty (legacy prompt still parses).
        let legacy = split_judge_sections("just one blob of rules");
        assert_eq!(legacy.preamble, "just one blob of rules");
        assert!(legacy.answerable.is_empty() && legacy.abstention.is_empty());
    }

    #[test]
    fn user_prompt_with_evidence_injects_block_between_gold_and_answer() {
        let s = test_sections();
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
    fn build_user_answerable_uses_answerable_section_after_data() {
        let s = test_sections();
        let u = build_user("Q?", "G", "A", None, true, false, &[], &s);
        assert!(u.contains("GOLD: G"));
        assert!(u.contains("ANSWERABLE-RULE"));
        assert!(!u.contains("ABSTENTION-RULE"));
        // The rule sits AFTER the data (that placement is what the judge model actually follows).
        assert!(u.find("ANSWER: A").unwrap() < u.find("ANSWERABLE-RULE").unwrap());
    }

    #[test]
    fn build_user_unanswerable_uses_abstention_section_no_gold() {
        let s = test_sections();
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
            "carries the unanswerable note"
        );
        assert!(u.contains("ABSTENTION-RULE"), "uses the abstention section");
        assert!(!u.contains("ANSWERABLE-RULE"));
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
    fn build_user_injects_dialogue_block_and_rule_when_present() {
        let s = test_sections();
        let dlg = vec![
            ("assistant".to_string(), "real answer here".to_string()),
            ("user".to_string(), "thanks".to_string()),
        ];
        let u = build_user("Q?", "G", "A", None, true, false, &dlg, &s);
        assert!(u.contains("DIALOGUE:"));
        assert!(u.contains("[assistant] real answer here"));
        assert!(u.contains("DIALOGUE-RULE"));
        // No dialogue -> neither the block nor its rule appear.
        let u2 = build_user("Q?", "G", "A", None, true, false, &[], &s);
        assert!(!u2.contains("DIALOGUE:"));
        assert!(!u2.contains("DIALOGUE-RULE"));
    }

    #[test]
    fn build_user_credit_abstention_adds_safe_miss_note_only_when_enabled() {
        let s = test_sections();
        let balanced = build_user("Q?", "G", "A", None, true, false, &[], &s);
        assert!(!balanced.contains("safe miss"));
        let safety = build_user("Q?", "G", "A", None, true, true, &[], &s);
        assert!(safety.contains("safe miss"));
        // The note is answerable-only: an unanswerable prompt is unaffected by credit_abstention.
        assert_eq!(
            build_user("Q?", "", "A", None, false, false, &[], &s),
            build_user("Q?", "", "A", None, false, true, &[], &s)
        );
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
}
