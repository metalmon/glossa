//! File-prompt LLM judge: a `judge.md` system prompt drives an OpenAI-compatible endpoint to
//! grade one case (question/gold/answer) as correct/partial/wrong, via a fixed `VERDICT:` line
//! the harness parses back out. Reuses the same chat client the agent backend drives
//! (`backend::openai::chat_once`) so judge calls hit the endpoint the same way.

use crate::backend::openai::chat_once;
use crate::lab::Endpoint;
use anyhow::Context;
use glossa::index::store::DocIndex;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Parse the LAST `VERDICT:` line in `reply` (case-insensitive), whatever precedes it becomes the
/// `reason`. No `VERDICT:` line at all → `Unscored`. An unrecognized value after `VERDICT:` also
/// falls back to `Unscored` (but still carries the raw reply so a caller can see what happened).
pub fn parse_verdict(reply: &str) -> Judgement {
    let mut verdict = Verdict::Unscored;
    for line in reply.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_lowercase();
        if let Some(rest) = lower.strip_prefix("verdict:") {
            verdict = match rest.trim() {
                "correct" => Verdict::Correct,
                "partial" => Verdict::Partial,
                "wrong" => Verdict::Wrong,
                _ => Verdict::Unscored,
            };
        }
    }
    // Reason: everything before the first VERDICT: line, joined, trimmed. Falls back to the
    // whole reply when there's no VERDICT: line to anchor on.
    let cut = reply
        .lines()
        .position(|l| l.trim().to_lowercase().starts_with("verdict:"));
    let reason = match cut {
        Some(i) => reply
            .lines()
            .take(i)
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string(),
        None => reply.trim().to_string(),
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

/// Build the judge user message. With `evidence`, the block is injected between `GOLD:` and
/// `ANSWER:`. Without it (`None`), the message is byte-identical to the historical gold-only form.
fn build_user(question: &str, gold: &str, answer: &str, evidence: Option<&str>) -> String {
    match evidence {
        Some(ev) => format!(
            "QUESTION: {question}\nGOLD: {gold}\n{ev}\nANSWER: {answer}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`."
        ),
        None => format!(
            "QUESTION: {question}\nGOLD: {gold}\nANSWER: {answer}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`."
        ),
    }
}

/// Judge one case: system = `judge_md` (the file-prompt), user = the fixed QUESTION/GOLD/ANSWER
/// block. When `source` names corpus chunks and `idx` is supplied, their text is injected as an
/// `EVIDENCE:` block between GOLD and ANSWER so a correct answer that EXCEEDS the terse gold is
/// credited, not penalized (see `evidence_block`). When `source` is empty, `idx` is `None`, or no
/// ref loads, the EVIDENCE block is omitted and the prompt is byte-identical to the gold-only form.
/// Posts to `ep` via `chat_once` and parses the reply with `parse_verdict`.
pub fn judge(
    ep: &Endpoint,
    judge_md: &str,
    question: &str,
    gold: &str,
    answer: &str,
    source: &[String],
    idx: Option<&DocIndex>,
) -> anyhow::Result<Judgement> {
    let api_key = ep.resolve_key();
    // Trim the embedded fields so the judge message stays tidy and never ends on a stray newline
    // (some strict providers reject a message ending in `\n` — see prompt::user_prompt).
    let (question, gold, answer) = (question.trim(), gold.trim(), answer.trim());
    let snippets = load_evidence(source, idx);
    let evidence = evidence_block(&snippets);
    let user = build_user(question, gold, answer, evidence.as_deref());
    let messages = vec![
        json!({ "role": "system", "content": judge_md }),
        json!({ "role": "user", "content": user }),
    ];
    let msg = chat_once(
        &ep.endpoint,
        &ep.model,
        &messages,
        api_key.as_deref(),
        ep.timeout_secs,
        ep.resolve_temperature(),
    )
    .context("judge endpoint request failed")?;
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    Ok(parse_verdict(content))
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

    #[test]
    fn user_prompt_with_evidence_injects_block_between_gold_and_answer() {
        // Stubbed chunk text (mock) — no corpus, no network.
        let snippets = vec![("a.pdf#p.1".to_string(), "stub evidence text".to_string())];
        let ev = evidence_block(&snippets);
        let prompt = build_user("Q?", "G", "A", ev.as_deref());
        assert!(prompt.contains("EVIDENCE:\n[a.pdf#p.1]\nstub evidence text"));
        // Block sits between GOLD and ANSWER.
        let gold_at = prompt.find("GOLD: G").unwrap();
        let ev_at = prompt.find("EVIDENCE:").unwrap();
        let ans_at = prompt.find("ANSWER: A").unwrap();
        assert!(gold_at < ev_at && ev_at < ans_at);
    }

    #[test]
    fn user_prompt_without_source_is_byte_identical_to_gold_only() {
        // With no evidence, build_user must reproduce the historical gold-only message byte-for-byte.
        let got = build_user("Q?", "G", "A", None);
        let expected = format!(
            "QUESTION: {}\nGOLD: {}\nANSWER: {}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`.",
            "Q?", "G", "A"
        );
        assert_eq!(got, expected);
        // An empty `source` with no index loads no evidence → block omitted → gold-only path.
        assert!(evidence_block(&load_evidence(&[], None)).is_none());
    }
}
