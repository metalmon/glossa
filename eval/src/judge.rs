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

/// Build the judge user message. With `evidence`, the block is injected between `GOLD:` and
/// `ANSWER:`. Without it (`None`), the message is byte-identical to the historical gold-only form.
///
/// When `answerable` is `false` the case is an ABSTENTION test: the question cannot be answered from
/// the knowledge base (out of scope / not covered / a routing or non-technical request), so there is
/// no gold text to compare against. The correct behavior is for the reader to DECLINE — say it has no
/// answer / the info isn't in the KB / route to a human — without inventing a technical answer. The
/// message then carries the abstention rubric instead of GOLD/EVIDENCE.
fn build_user(
    question: &str,
    gold: &str,
    answer: &str,
    evidence: Option<&str>,
    answerable: bool,
    credit_abstention: bool,
) -> String {
    if !answerable {
        return format!(
            "QUESTION: {question}\n\
             NOTE: This question CANNOT be answered from the knowledge base — it is out of scope, not \
             covered by the corpus, or a non-technical / routing request. There is no gold answer.\n\
             The CORRECT behavior is for the assistant to state it has no answer / the information is \
             not in the knowledge base, or to decline or route to a human — WITHOUT inventing a \
             technical answer.\n\
             ANSWER: {answer}\n\
             Grade `correct` if the ANSWER appropriately declines or states there is no answer; \
             `wrong` if it gives a substantive or fabricated technical answer as if it knew; \
             `partial` if it declines but still adds unsupported specific claims.\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`."
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
    match evidence {
        Some(ev) => format!(
            "QUESTION: {question}\nGOLD: {gold}\n{ev}\nANSWER: {answer}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`.{abstain_note}"
        ),
        None => format!(
            "QUESTION: {question}\nGOLD: {gold}\nANSWER: {answer}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`.{abstain_note}"
        ),
    }
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
fn resolve_judge_votes() -> usize {
    std::env::var("KB_EVAL_JUDGE_VOTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_JUDGE_VOTES)
        .max(1)
}

/// Judge one case: system = `judge_md` (the file-prompt), user = the fixed QUESTION/GOLD/ANSWER
/// block. When `source` names corpus chunks and `idx` is supplied, their text is injected as an
/// `EVIDENCE:` block between GOLD and ANSWER so a correct answer that EXCEEDS the terse gold is
/// credited, not penalized (see `evidence_block`). When `source` is empty, `idx` is `None`, or no
/// ref loads, the EVIDENCE block is omitted and the prompt is byte-identical to the gold-only form.
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
) -> anyhow::Result<Judgement> {
    // Trim the embedded fields so the judge message stays tidy and never ends on a stray newline
    // (some strict providers reject a message ending in `\n` — see prompt::user_prompt).
    let (question, gold, answer) = (question.trim(), gold.trim(), answer.trim());
    let snippets = load_evidence(source, idx);
    let evidence = evidence_block(&snippets);
    let user = build_user(
        question,
        gold,
        answer,
        evidence.as_deref(),
        answerable,
        credit_abstention,
    );
    let messages = vec![
        json!({ "role": "system", "content": judge_md }),
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

    #[test]
    fn user_prompt_with_evidence_injects_block_between_gold_and_answer() {
        // Stubbed chunk text (mock) — no corpus, no network.
        let snippets = vec![("a.pdf#p.1".to_string(), "stub evidence text".to_string())];
        let ev = evidence_block(&snippets);
        let prompt = build_user("Q?", "G", "A", ev.as_deref(), true, false);
        assert!(prompt.contains("EVIDENCE:\n[a.pdf#p.1]\nstub evidence text"));
        // Block sits between GOLD and ANSWER.
        let gold_at = prompt.find("GOLD: G").unwrap();
        let ev_at = prompt.find("EVIDENCE:").unwrap();
        let ans_at = prompt.find("ANSWER: A").unwrap();
        assert!(gold_at < ev_at && ev_at < ans_at);
    }

    #[test]
    fn build_user_unanswerable_uses_abstention_rubric_not_gold() {
        // answerable=false → abstention rubric, no GOLD/EVIDENCE (there is no gold to compare).
        let u = build_user("Q?", "", "not in the knowledge base", None, false, false);
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
        let a = build_user("Q?", "G", "A", None, true, false);
        assert!(a.contains("GOLD: G"));
        assert!(!a.contains("CANNOT be answered"));
    }

    #[test]
    fn build_user_credit_abstention_adds_safe_miss_note_only_when_enabled() {
        // Off (credit_abstention=false): no safe-miss note — a decline stays a miss (`wrong`).
        let balanced = build_user("Q?", "G", "A", None, true, false);
        assert!(!balanced.contains("safe miss"));
        // On (credit_abstention=true): the answerable prompt tells the judge to grade a
        // decline as `partial`, not `wrong`.
        let safety = build_user("Q?", "G", "A", None, true, true);
        assert!(safety.contains("safe miss"));
        // The note is answerable-only: an unanswerable prompt is unaffected by credit_abstention.
        assert_eq!(
            build_user("Q?", "", "A", None, false, false),
            build_user("Q?", "", "A", None, false, true)
        );
    }

    #[test]
    fn user_prompt_without_source_is_byte_identical_to_gold_only() {
        // With no evidence, build_user must reproduce the historical gold-only message byte-for-byte.
        let got = build_user("Q?", "G", "A", None, true, false);
        let expected = format!(
            "QUESTION: {}\nGOLD: {}\nANSWER: {}\n\
             Reply with one line reason then `VERDICT: correct|partial|wrong`.",
            "Q?", "G", "A"
        );
        assert_eq!(got, expected);
        // An empty `source` with no index loads no evidence → block omitted → gold-only path.
        assert!(evidence_block(&load_evidence(&[], None)).is_none());
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
        let _g = JUDGE_VOTES_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
