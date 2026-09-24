//! `--vision` image threading for the answer path: the pure `vision_user_message` builder and the
//! `VisionTransport` wrapper that appends it right after a round's tool results. Split out of
//! `backend::openai`; the OpenAI-compatible `/v1/chat/completions` shape has no image slot on a
//! `role:"tool"` message, so images ride in a follow-up `role:"user"` message.

use crate::backend::transport::{ChatTransport, TurnReply};
use glossa::read::DocImage;
use serde_json::{json, Value};
use std::rc::Rc;

/// Build the vision-input user message for a set of images returned by a tool call this round, or
/// `None` when there are none. `--vision`-only mechanism (see `run_agent_loop`): the OpenAI-
/// compatible `/v1/chat/completions` shape has no image slot on a `role:"tool"` message, so images
/// ride in a FOLLOW-UP `role:"user"` message whose `content` is an array — one leading text part
/// plus one `image_url` part per image, each a `data:image/jpeg;base64,<payload>` URI.
///
/// Every image is normalized through [`glossa::read::to_jpeg`] first (JPEG passes through
/// untouched; anything else is decoded and re-encoded) and base64-encoded with the STANDARD
/// (padded, unwrapped) alphabet — canonical, no embedded whitespace/newlines, since a malformed
/// `image_url.url` 400s on our opencode-zen endpoint.
///
/// NO per-turn image cap (by request): every image the tool call returned is fed. A read of a
/// large multi-page datasheet can surface many, so mind the request payload size / endpoint limits
/// — `to_jpeg` bounds each image's SIZE, but there is no bound on the COUNT.
pub(crate) fn vision_user_message(images: &[DocImage]) -> Option<Value> {
    if images.is_empty() {
        return None;
    }
    use base64::Engine as _;
    let mut content = vec![json!({
        "type": "text",
        "text": format!("Images from that read ({}):", images.len())
    })];
    for img in images {
        let jpeg = glossa::read::to_jpeg(img.clone());
        let payload = base64::engine::general_purpose::STANDARD.encode(&jpeg.bytes);
        content.push(json!({
            "type": "image_url",
            "image_url": { "url": format!("data:image/jpeg;base64,{payload}") }
        }));
    }
    Some(json!({ "role": "user", "content": Value::Array(content) }))
}

/// Answer-path `--vision` wrapper around the reader's real `ChatTransport`. Mirrors the
/// build/extract path's `ClosureTransport`: images the round's `exec` surfaced are buffered in
/// `pending_images`, and — right after the inner transport pushes its `role:"tool"` result(s) —
/// drained into ONE follow-up `role:"user"` image message (`vision_user_message`), because the
/// OpenAI-compatible endpoint has no image slot on a tool message. Every other method delegates to
/// `inner` unchanged, so behavior is identical apart from that appended image message.
///
/// Constructed ONLY when `--vision` is on AND the reader speaks the OpenAI-compatible chat API
/// (`vision_user_message` emits that provider's `image_url` data-URI shape). The non-vision answer
/// path — and any non-OpenAI api kind — drives the inner transport directly, so the transcript is
/// byte-identical to before this wrapper existed.
pub(crate) struct VisionTransport<'a> {
    pub(crate) inner: &'a dyn ChatTransport,
    pub(crate) pending_images: Rc<std::cell::RefCell<Vec<DocImage>>>,
}

impl ChatTransport for VisionTransport<'_> {
    fn tools_schema(&self, ctx: &glossa::tools::registry::ToolContext) -> Value {
        self.inner.tools_schema(ctx)
    }

    fn call(
        &self,
        ep: &crate::lab::Endpoint,
        system: Option<&str>,
        messages: &[Value],
        tools: Option<&Value>,
        temperature: Option<f64>,
    ) -> anyhow::Result<TurnReply> {
        self.inner.call(ep, system, messages, tools, temperature)
    }

    fn push_assistant_turn(&self, messages: &mut Vec<Value>, reply: &TurnReply) {
        self.inner.push_assistant_turn(messages, reply);
    }

    fn push_tool_results(&self, messages: &mut Vec<Value>, results: &[(String, String)]) {
        // Inner transport owns the `role:"tool"` result shape; we only append the vision message.
        self.inner.push_tool_results(messages, results);
        let images: Vec<DocImage> = self.pending_images.borrow_mut().drain(..).collect();
        if let Some(img_msg) = vision_user_message(&images) {
            messages.push(img_msg);
        }
    }

    fn post_feedback(&self, episode_id: &str, metrics: &[(&str, Value)]) {
        self.inner.post_feedback(episode_id, metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub_image(tag: u8) -> DocImage {
        // mime "image/jpeg" short-circuits `to_jpeg` (returns the bytes unchanged, no real JPEG
        // decode needed) — exactly what a stubbed unit test wants: no fixture image file, no
        // `image` crate round-trip, just distinct bytes per stub so multiple images are
        // distinguishable in the encoded output.
        DocImage {
            mime: "image/jpeg".to_string(),
            bytes: vec![0xFF, 0xD8, 0xFF, tag], // fake JPEG-ish bytes, tag makes each stub unique
        }
    }

    #[test]
    fn vision_message_none_when_no_images() {
        assert!(vision_user_message(&[]).is_none());
    }

    #[test]
    fn vision_message_builds_canonical_data_uri_content_array() {
        let img = stub_image(1);
        let msg =
            vision_user_message(std::slice::from_ref(&img)).expect("one image -> Some(message)");
        assert_eq!(msg["role"], "user");
        let content = msg["content"].as_array().expect("content must be an array");
        assert_eq!(
            content.len(),
            2,
            "one text part + one image_url part: {content:?}"
        );
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"]
            .as_str()
            .expect("image_url.url string");
        assert!(
            url.starts_with("data:image/jpeg;base64,"),
            "must be a JPEG data URI, got: {url}"
        );
        let payload = url.strip_prefix("data:image/jpeg;base64,").unwrap();
        assert!(
            !payload.contains('\n') && !payload.contains(' '),
            "base64 payload must be canonical (no embedded whitespace/newlines): {payload:?}"
        );
        use base64::Engine as _;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .expect("payload must be valid standard base64");
        assert_eq!(
            decoded, img.bytes,
            "decoded payload must round-trip the original JPEG bytes"
        );
    }

    #[test]
    fn vision_message_feeds_all_images_uncapped() {
        // No per-turn cap (by request): every image the tool call returned is fed, however many.
        let images: Vec<DocImage> = (0..6).map(stub_image).collect();
        let msg = vision_user_message(&images).expect("non-empty -> Some(message)");
        let content = msg["content"].as_array().unwrap();
        let image_parts = content.iter().filter(|p| p["type"] == "image_url").count();
        assert_eq!(
            image_parts,
            images.len(),
            "every image must be fed — no per-turn cap: {content:?}"
        );
        assert!(
            content[0]["text"]
                .as_str()
                .unwrap()
                .contains(&images.len().to_string()),
            "lead text states the image count: {content:?}"
        );
    }

    /// Answer-path per-round assembly (Task 7): the reader's real transport pushes its `role:"tool"`
    /// result, then — under `--vision` — `VisionTransport` appends the follow-up `role:"user"` image
    /// message. This is the exact production wrapper the answer loop installs; testing it directly
    /// mirrors how `vision_user_message` itself is unit-tested (the live loop is transport-driven).
    fn build_round_messages_with_tool_images(images: &[DocImage], vision: bool) -> Vec<Value> {
        use crate::backend::transport::openai::OpenAiTransport;
        let results = [("call_1".to_string(), "(scanned page text)".to_string())];
        let mut messages: Vec<Value> = Vec::new();
        if vision {
            // The same wrapper `answer_capturing` installs under `--vision`: inner transport pushes
            // the tool result, then the buffered images ride in a follow-up user message.
            let vt = VisionTransport {
                inner: &OpenAiTransport,
                pending_images: Rc::new(std::cell::RefCell::new(images.to_vec())),
            };
            vt.push_tool_results(&mut messages, &results);
        } else {
            // Off `--vision`: the raw transport, no wrapper — the pre-vision transcript shape.
            OpenAiTransport.push_tool_results(&mut messages, &results);
        }
        messages
    }

    #[test]
    fn answering_loop_emits_vision_user_message_when_vision_on() {
        // Vision on: a `read` that returns an image produces a following `role:"user"` message
        // carrying an `image_url` data URI, right after the tool result.
        let img = stub_image(7);
        let msgs = build_round_messages_with_tool_images(std::slice::from_ref(&img), true);
        let user_img = msgs
            .iter()
            .rev()
            .find(|m| m["role"] == "user")
            .expect("vision-on must append a role:\"user\" image message");
        let parts = user_img["content"]
            .as_array()
            .expect("image message content must be an array");
        assert!(
            parts.iter().any(|p| p["type"] == "image_url"
                && p["image_url"]["url"]
                    .as_str()
                    .unwrap_or("")
                    .starts_with("data:image/jpeg;base64,")),
            "expected an image_url data URI part, got: {parts:?}"
        );

        // Vision off: no image message is emitted (byte-identical to the pre-vision transcript).
        let msgs_off =
            build_round_messages_with_tool_images(std::slice::from_ref(&stub_image(7)), false);
        assert!(
            !msgs_off.iter().any(|m| m["role"] == "user"
                && m["content"]
                    .as_array()
                    .map(|a| a.iter().any(|p| p["type"] == "image_url"))
                    .unwrap_or(false)),
            "vision-off must not emit any image_url user message: {msgs_off:?}"
        );
    }
}
