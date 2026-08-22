//! File-prompt LLM judge: a `judge.md` system prompt drives an OpenAI-compatible endpoint to
//! grade one case (question/gold/answer) as correct/partial/wrong, via a fixed `VERDICT:` line
//! the harness parses back out. Reuses the same chat client the agent backend drives
//! (`backend::openai::chat_once`) so judge calls hit the endpoint the same way.

use crate::backend::openai::chat_once;
use crate::lab::Endpoint;
use anyhow::Context;
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        Some(i) => reply.lines().take(i).collect::<Vec<_>>().join("\n").trim().to_string(),
        None => reply.trim().to_string(),
    };
    Judgement {
        verdict,
        reason,
        raw: reply.to_string(),
    }
}

/// Judge one case: system = `judge_md` (the file-prompt), user = the fixed
/// QUESTION/GOLD/ANSWER block. Posts to `ep` via `chat_once` and parses the reply with
/// `parse_verdict`.
pub fn judge(
    ep: &Endpoint,
    judge_md: &str,
    question: &str,
    gold: &str,
    answer: &str,
) -> anyhow::Result<Judgement> {
    let api_key = ep.resolve_key();
    // Trim the embedded fields so the judge message stays tidy and never ends on a stray newline
    // (some strict providers reject a message ending in `\n` — see prompt::user_prompt).
    let (question, gold, answer) = (question.trim(), gold.trim(), answer.trim());
    let user = format!(
        "QUESTION: {question}\nGOLD: {gold}\nANSWER: {answer}\n\
         Reply with one line reason then `VERDICT: correct|partial|wrong`."
    );
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
        assert!(matches!(parse_verdict("reason...\nVERDICT: correct").verdict, Verdict::Correct));
        assert!(matches!(parse_verdict("VERDICT: Partial").verdict, Verdict::Partial));
        assert!(matches!(parse_verdict("blah\nverdict: WRONG\n").verdict, Verdict::Wrong));
        assert!(matches!(parse_verdict("no verdict here").verdict, Verdict::Unscored));
        // last VERDICT wins
        assert!(matches!(
            parse_verdict("VERDICT: wrong\nVERDICT: correct").verdict,
            Verdict::Correct
        ));
    }
}
