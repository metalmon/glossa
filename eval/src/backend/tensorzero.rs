use serde_json::{json, Value};

pub struct EpisodeOutcome {
    pub answer: String,
    pub episode_id: Option<String>,
    pub surfaced_titles: Vec<String>,
}

/// A single TensorZero `/inference` turn result.
pub struct TzTurn {
    pub content: Vec<Value>, // TZ content blocks: {type:"tool_call"|"text"|"thought", ...}
    pub episode_id: String,
}

/// Drive a TensorZero episode to a final text answer.
///
/// `chat(messages, episode_id)` performs one `/inference` call (episode_id is None on the first turn,
/// then the id returned by the first turn). When the returned content has `tool_call` blocks, each is
/// executed via `exec(name, args)` and fed back as a `tool_result` content block; otherwise the
/// concatenated `text` blocks are the answer.
pub fn run_episode<C, X>(
    mut chat: C,
    user_question: &str,
    mut exec: X,
    max_rounds: usize,
) -> anyhow::Result<EpisodeOutcome>
where
    C: FnMut(&[Value], Option<&str>) -> anyhow::Result<TzTurn>,
    X: FnMut(&str, &Value) -> (String, Vec<String>),
{
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": user_question })];
    let mut episode_id: Option<String> = None;
    let mut surfaced_titles: Vec<String> = Vec::new();

    for _ in 0..max_rounds {
        let turn = chat(&messages, episode_id.as_deref())?;
        episode_id = Some(turn.episode_id.clone());

        let tool_calls: Vec<&Value> = turn
            .content
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_call"))
            .collect();

        if tool_calls.is_empty() {
            let answer: String = turn
                .content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("");
            return Ok(EpisodeOutcome { answer, episode_id, surfaced_titles });
        }

        for call in &tool_calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = call.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            // echo the assistant tool_call, then the tool_result (result MUST be a string)
            messages.push(json!({
                "role": "assistant",
                "content": [{ "type": "tool_call", "id": id, "name": name, "arguments": args }]
            }));
            let (result, titles) = exec(name, &args);
            surfaced_titles.extend(titles);
            messages.push(json!({
                "role": "user",
                "content": [{ "type": "tool_result", "id": id, "name": name, "result": result }]
            }));
        }
    }
    // Out of rounds: best-effort empty answer (the report still scores it).
    Ok(EpisodeOutcome { answer: String::new(), episode_id, surfaced_titles })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn episode_executes_tool_then_answers() {
        let round = RefCell::new(0usize);
        let chat = |_msgs: &[Value], _eid: Option<&str>| {
            let mut r = round.borrow_mut();
            *r += 1;
            if *r == 1 {
                Ok(TzTurn {
                    content: vec![json!({ "type": "tool_call", "id": "c1", "name": "search", "arguments": { "query": "corliss" } })],
                    episode_id: "ep1".into(),
                })
            } else {
                Ok(TzTurn {
                    content: vec![json!({ "type": "text", "text": "ANSWER: Chief of Protocol" })],
                    episode_id: "ep1".into(),
                })
            }
        };
        let exec = |name: &str, args: &Value| {
            assert_eq!(name, "search");
            assert_eq!(args["query"], "corliss");
            ("Meet_Corliss_Archer.md:Meet Corliss Archer: …  [9.0]".to_string(),
             vec!["Meet Corliss Archer".to_string()])
        };
        let out = run_episode(chat, "Question: ...", exec, 8).unwrap();
        assert_eq!(out.answer, "ANSWER: Chief of Protocol");
        assert_eq!(out.episode_id.as_deref(), Some("ep1"));
        assert_eq!(out.surfaced_titles, vec!["Meet Corliss Archer".to_string()]);
    }

    #[test]
    fn episode_returns_direct_answer() {
        let chat = |_: &[Value], _: Option<&str>| Ok(TzTurn {
            content: vec![json!({ "type": "text", "text": "ANSWER: yes" })],
            episode_id: "e".into(),
        });
        let exec = |_: &str, _: &Value| (String::new(), Vec::new());
        let out = run_episode(chat, "q", exec, 4).unwrap();
        assert_eq!(out.answer, "ANSWER: yes");
    }
}
