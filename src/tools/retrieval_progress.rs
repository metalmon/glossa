//! Per-EPISODE retrieval-plateau tracker (moved from `kb-eval`'s `backend::glossa_tools` — pure
//! relocation, no behavior change). Shared by any caller (eval harness, and — in a later step —
//! the MCP server) that wants to detect a reader whose searches have stopped surfacing new ground.

use regex::Regex;
use std::collections::{HashSet, VecDeque};
use std::sync::OnceLock;

/// Per-EPISODE retrieval-plateau tracker. A reader that spirals — issuing search after search that
/// each land on already-seen ground — never actually gains information, but the loop's
/// unproductive-streak guard (`unproductive_steer`) resets on ANY single new id, so a spiral of
/// varied slightly-productive probes evades it. This tracker measures a WINDOWED marginal-
/// information-gain PLATEAU instead: it records the number of NEW result-ids each retrieval call
/// surfaces into a sliding window of the last [`RetrievalProgress::W`] calls, and once enough calls
/// have run with a cumulative body of results, fires ONCE when that window's total new-id count
/// falls to (or below) [`RetrievalProgress::G`].
///
/// What it emits is a NEUTRAL OBSERVATION appended to the tool result (see [`RetrievalProgress::
/// observe`]) — a factual "gain has plateaued" marker with counts — NOT an instruction. The POLICY
/// (answer now / stop searching / change approach) is deliberately left to the reader PROMPT / GEPA:
/// baking a directive into the shared tool layer would break its reuse and pre-empt the very thing
/// the prompt is being optimized to decide.
///
/// One tracker lives per EPISODE (one eval case / one train rollout); callers that don't opt in
/// simply never construct one, so the tool result is byte-identical to today. Only RETRIEVAL tool
/// calls (the id-surfacing ones — see [`is_retrieval_tool`]) feed `observe`; write/non-retrieval
/// tools are skipped so their zero-id results don't spuriously drain the window.
#[derive(Debug, Default)]
pub struct RetrievalProgress {
    /// Every distinct result-id seen so far this episode (the cumulative retrieval footprint).
    seen: HashSet<String>,
    /// New-id count for each of the last `W` retrieval calls (front = oldest). A plateau is a
    /// window whose SUM has fallen to `<= G`.
    window: VecDeque<usize>,
    /// True once the plateau marker has been emitted for the CURRENT plateau; reset to false as
    /// soon as a later call surfaces genuinely new ids, so a second, distinct plateau can fire
    /// again rather than the signal being one-shot for the whole episode.
    fired: bool,
    /// Total retrieval calls observed this episode (the `>= M` gate below).
    calls: usize,
}

impl RetrievalProgress {
    /// Sliding-window size: judge the plateau over the last `W` retrieval calls.
    pub const W: usize = 3;
    /// Minimum retrieval calls before a plateau can fire — don't call it early on a reader that
    /// simply hasn't searched much yet.
    pub const M: usize = 4;
    /// Minimum cumulative distinct ids before a plateau can fire — a reader that has surfaced
    /// almost nothing isn't "plateaued", it just hasn't found the corpus yet.
    pub const E: usize = 3;
    /// New-id budget over the window: `sum(window) <= G` is the plateau condition. `0` = zero new
    /// ids across the last `W` calls.
    pub const G: usize = 0;

    pub fn new() -> Self {
        Self::default()
    }

    /// Record one RETRIEVAL call's surfaced `new_ids` and return `Some(marker)` exactly once when
    /// the window-of-last-`W` marginal gain has plateaued (`calls >= M` AND `seen >= E` AND a full
    /// window AND `sum(window) <= G`), else `None`. The marker is a neutral factual observation with
    /// counts — no imperative. Firing flips `fired` so it emits once per plateau; a later call that
    /// brings genuinely new ids clears `fired` so a subsequent plateau can fire again.
    pub fn observe(&mut self, new_ids: &[String]) -> Option<String> {
        self.calls += 1;
        let mut new_count = 0usize;
        for id in new_ids {
            if self.seen.insert(id.clone()) {
                new_count += 1;
            }
        }
        self.window.push_back(new_count);
        if self.window.len() > Self::W {
            self.window.pop_front();
        }
        // Genuinely new ground re-arms the signal for a future, distinct plateau.
        if new_count > 0 {
            self.fired = false;
        }
        let window_sum: usize = self.window.iter().sum();
        let plateaued = !self.fired
            && self.calls >= Self::M
            && self.seen.len() >= Self::E
            && self.window.len() >= Self::W
            && window_sum <= Self::G;
        if plateaued {
            self.fired = true;
            return Some(format!(
                "\n\n[retrieval: {} unique results gathered; {} new over the last {} searches — gain has plateaued]",
                self.seen.len(),
                window_sum,
                self.window.len(),
            ));
        }
        None
    }
}

/// Whether a tool NAME surfaces retrieval result-ids that should feed [`RetrievalProgress`] — the
/// id-producing reader tools (`search`/`glossary`/`related`/`neighbors`/`reach`/`sql`/`read`). Write
/// and non-id tools (`glob`/`grep`/`get_source_file`/`graph_stats`/`resolve`, and any unknown name)
/// are excluded so their zero-id results never spuriously count as "no new" and drain the plateau
/// window. `read` qualifies because its caller substitutes its `path` arg as the surfaced id (see
/// `openai::execute_tool` / `gepa_graph::rollout_one`), even though `exec` returns none for it.
pub fn is_retrieval_tool(name: &str) -> bool {
    matches!(
        name,
        "search" | "glossary" | "related" | "neighbors" | "reach" | "sql" | "read"
    )
}

/// Stable node identifiers surfaced by a graph tool's rendered body, for the unproductive-streak
/// novelty tracker in `openai::run_agent_loop`. Every graph-tool renderer in `glossa::tools`
/// (glossary's main hits AND its chain hops, related, neighbors, reach, and sql's
/// id-column handles) funnels a grounded node through `tools::node_ref`'s anchor in ONE of two
/// forms, both built from the SAME glued `<path>#<ord>` token (`tools.rs:532`):
///   - **entity/reasoning node**: `tools::read_anchor` wraps it as `— read <path>#<ord> · <label>`
///     (only emitted when the node has an outgoing MENTIONS edge to a section).
///   - **structural node (Section/Document)**: `tools::endpoint_ref`/`node_ref` print the anchor
///     BARE, with no "read" word at all — `<path>#<ord> · <label>` for a Section (glossary's
///     exact-title stub at `tools.rs:765`, `edge_line`/neighbors at `tools.rs:1018`,
///     `render_reach_chain` at `tools.rs:1190`), or `<path>  (document)` — no ord — for a Document.
/// The first fix here only matched the "— read" form, so a totally normal move — `neighbors` on a
/// Document returning its child Sections, or several glossary/reach calls landing on different
/// Section/Document nodes — surfaced ZERO ids per call and falsely tripped the streak on a reader
/// making real (structural) progress. The regex below matches the glued `<path>#<ord>` (or the
/// Document's `<path>  (document)`) regardless of whether "— read " precedes it, so both forms count.
/// It's still unambiguous against a glossary chain-hop line (`edge_type  [node_type]  label`, no
/// bare id at all — see `tools::chain_lines`): that line has no `#<ord>` or `(document)` token, so
/// it never matches — a generic `<token>  [Type]` scan would have misread the EDGE TYPE as a
/// stable node id instead, collapsing distinct endpoints reached via the same relation into one
/// false "already seen".
/// The id is `path#ord` (Section) or bare `path` (Document) — the same shape a `read` call's own
/// id takes for the no-`n` case. An ungrounded, non-structural node (no MENTIONS edge, so no
/// anchor at all) contributes nothing from that line; that's fine, the streak only needs SOME
/// calls in a burst to register progress, not every line.
pub fn extract_node_ids(body: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    // The ordinal branch requires the GLUED `path#n` anchor (zero spaces) — `tools::node_ref`
    // emits exactly this tight form for every real anchor. Tolerating `\s*` here let a node's own
    // label text (e.g. "issue #42") spuriously match as an anchor, causing false novelty for the
    // loop detector; requiring glued-only fixes that. The bare `(document)` anchor is untouched
    // and still requires the wider `\s{2,}` gap so a path token's own text doesn't
    // false-positive into a document match.
    let re = RE.get_or_init(|| {
        Regex::new(r"(\S+?)(?:#(\d+)|\s{2,}\(document\))").expect("valid regex")
    });
    re.captures_iter(body)
        .map(|c| match c.get(2) {
            Some(ord) => format!("{}#{}", &c[1], ord.as_str()),
            None => c[1].to_string(),
        })
        .collect()
}

/// Which condition fired on a given [`ReaderSignals::observe`] call, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    /// The exact same (tool, args) call as the immediately-previous observed call.
    Repeat,
    /// `STREAK_K` consecutive VARIED calls (distinct keys) each surfaced zero new ids.
    Streak,
    /// The windowed marginal-information-gain plateau (same condition as [`RetrievalProgress`]).
    Plateau,
}

/// What the CALLER should do with the tool result body, given collapsed/ok novelty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultRender {
    /// Pass the body through unchanged (novelty is fine).
    Full,
    /// Novelty collapsed: keep ONLY results whose id is NOT already seen, then append `marker`.
    /// (For a hit-list tool the caller filters hits by id; if it has no per-id structure it should
    /// fall back to `ReplaceWith`.) `omitted` = count of already-seen ids in THIS call.
    OnlyNew { marker: String, omitted: usize },
    /// Replace the body entirely with `marker` (nothing new worth re-dumping / exact repeat).
    ReplaceWith { marker: String },
}

/// Result of one [`ReaderSignals::observe`] call: which signal (if any) fired, what the caller
/// should render, and the neutral marker text (for capture/logging) when one was emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// `None` when nothing fired this call (render is always `Full` in that case).
    pub kind: Option<SignalKind>,
    /// What the caller should do with the tool result body.
    pub render: ResultRender,
    /// The neutral marker text, for capture; `None` exactly when `render == Full`.
    pub marker: Option<String>,
}

/// Unified per-SESSION reader-progress tracker: a superset of [`RetrievalProgress`] that also
/// detects an exact-repeat call and a short unproductive STREAK, on top of the same windowed
/// PLATEAU. One tracker lives per session/episode; a caller that only wants the original
/// repeat+streak-free behavior keeps using [`RetrievalProgress`] directly — this type is additive,
/// not a replacement (see the module-level callers, which are migrated in a later step).
///
/// All three signals emit the same kind of thing: a NEUTRAL factual observation, never a
/// directive — see the marker text in `repeat_marker`/`streak_marker`/`plateau_marker`. The POLICY
/// (answer now / stop searching / change approach) stays the reader prompt's job.
#[derive(Debug, Default)]
pub struct ReaderSignals {
    /// Every distinct result-id seen so far this session (cumulative retrieval footprint).
    seen: HashSet<String>,
    /// The (tool,args) identity key of the immediately-previous observed call, for repeat
    /// detection.
    last_key: Option<String>,
    /// New-id count for each of the last `W` (non-repeat) retrieval calls (front = oldest).
    window: VecDeque<usize>,
    /// True once the plateau marker has been emitted for the CURRENT plateau; cleared as soon as
    /// a later call surfaces genuinely new ids, so a second, distinct plateau can fire again.
    fired_plateau: bool,
    /// Total non-repeat retrieval calls observed this session (the `>= M` gate).
    calls: usize,
    /// Whether the plateau signal is armed at all (a caller mirroring today's loop, which only
    /// wants repeat+streak, constructs with `with_plateau(false)`).
    plateau_enabled: bool,
    /// Consecutive zero-new-id calls (VARIED keys only — a repeat doesn't touch this), for the
    /// streak signal. Resets to 0 both when a call brings new ids AND right after the streak
    /// fires (so it fires once per streak, then re-arms).
    streak: usize,
}

impl ReaderSignals {
    /// Consecutive zero-new-id (varied-key) calls before the streak signal fires. This mirrors
    /// `agent_loop::UNPRODUCTIVE_STREAK_K` (K=3); core cannot depend on the eval crate, so the
    /// number is redefined here — keep the two in sync by hand if either changes.
    pub const STREAK_K: usize = 3;
    /// Sliding-window size — reused verbatim from [`RetrievalProgress::W`].
    pub const W: usize = RetrievalProgress::W;
    /// Minimum non-repeat retrieval calls before a plateau can fire — reused from
    /// [`RetrievalProgress::M`].
    pub const M: usize = RetrievalProgress::M;
    /// Minimum cumulative distinct ids before a plateau can fire — reused from
    /// [`RetrievalProgress::E`].
    pub const E: usize = RetrievalProgress::E;
    /// New-id budget over the window — reused from [`RetrievalProgress::G`].
    pub const G: usize = RetrievalProgress::G;

    /// New tracker with the plateau signal ARMED (in addition to repeat+streak, which are always
    /// on).
    pub fn new() -> Self {
        Self {
            plateau_enabled: true,
            ..Self::default()
        }
    }

    /// New tracker with the plateau signal explicitly enabled/disabled. A caller that only wants
    /// repeat+streak (mirroring today's loop's unproductive-streak guard) passes `false`.
    pub fn with_plateau(enabled: bool) -> Self {
        Self {
            plateau_enabled: enabled,
            ..Self::default()
        }
    }

    /// Record one retrieval call (`tool` name — gated by the caller with [`is_retrieval_tool`]
    /// before calling this; not otherwise inspected here) identified by a stable `key` (the
    /// caller's `format!("{tool}:{compact_args}")`, or equivalent) and the ids it surfaced, and
    /// return the [`Outcome`] the caller should act on.
    ///
    /// Precedence (first match wins for `kind`; state is ALWAYS updated regardless of which — or
    /// whether any — signal fires):
    /// 1. **Repeat** — `key` is identical to the immediately-previous observed key. An identical
    ///    call surfaces nothing new by construction, so its ids are NOT folded into `seen`/
    ///    `window` (a following genuinely-new call still counts in full).
    /// 2. **Streak** — `STREAK_K` consecutive VARIED calls each surfacing zero new ids.
    /// 3. **Plateau** (only when armed, and only if Streak didn't already fire this call) — the
    ///    `W` calls immediately BEFORE this one already ground to a halt (`sum <= G`), `>= M`
    ///    total calls and `>= E` distinct ids have accumulated. Gating on the PRE-existing window
    ///    (rather than one that already includes this call's own contribution) is what makes both
    ///    renders reachable under `G == 0`: if THIS call still turned up a few new ids, that's
    ///    `OnlyNew` (the marker reports the plateaued trend that led up to it, while the caller
    ///    still keeps what's genuinely new); if it turned up nothing either, `ReplaceWith`.
    pub fn observe(&mut self, tool: &str, key: &str, ids: &[String]) -> Outcome {
        let _ = tool; // caller already gated with `is_retrieval_tool`; not otherwise needed here.

        // 1. Repeat: identical (tool,args) key as last time. Don't touch seen/window/streak with
        // this call's ids — an identical call can't surface anything genuinely new.
        if self.last_key.as_deref() == Some(key) {
            self.calls += 1;
            self.last_key = Some(key.to_string());
            let marker = repeat_marker();
            return Outcome {
                kind: Some(SignalKind::Repeat),
                render: ResultRender::ReplaceWith { marker: marker.clone() },
                marker: Some(marker),
            };
        }

        // Snapshot the window's state BEFORE this call's own contribution — the plateau gate
        // below judges the trend over the W calls that led UP TO this one (see doc comment).
        let prior_window_sum: usize = self.window.iter().sum();
        let prior_window_len = self.window.len();

        let mut new_count = 0usize;
        for id in ids {
            if self.seen.insert(id.clone()) {
                new_count += 1;
            }
        }
        self.window.push_back(new_count);
        if self.window.len() > Self::W {
            self.window.pop_front();
        }
        self.calls += 1;
        self.last_key = Some(key.to_string());
        // Snapshot fired_plateau as it stood BEFORE this call re-arms it — the Plateau gate below
        // must judge "was armed as of the START of this call", not "just got re-armed by the very
        // same call's own new ids". Without this, a call that both re-arms (new_count > 0) AND
        // still sees a stale, not-yet-drained prior window (sum <= G) would spuriously refire the
        // SAME instant it broke the plateau, instead of only once the window has genuinely drained
        // again on a LATER call.
        let plateau_was_armed = !self.fired_plateau;
        if new_count > 0 {
            // Genuinely new ground re-arms both the streak counter and a future plateau.
            self.fired_plateau = false;
            self.streak = 0;
        } else {
            self.streak += 1;
        }

        // 2. Streak: K consecutive (varied-key) zero-new calls, this one included.
        if self.streak >= Self::STREAK_K {
            self.streak = 0;
            let marker = streak_marker();
            return Outcome {
                kind: Some(SignalKind::Streak),
                render: ResultRender::ReplaceWith { marker: marker.clone() },
                marker: Some(marker),
            };
        }

        // 3. Plateau (armed + not already claimed by Streak this call).
        if self.plateau_enabled
            && plateau_was_armed
            && self.calls >= Self::M
            && self.seen.len() >= Self::E
            && prior_window_len >= Self::W
            && prior_window_sum <= Self::G
        {
            self.fired_plateau = true;
            let marker = plateau_marker(self.seen.len(), prior_window_sum, prior_window_len);
            let render = if new_count > 0 {
                ResultRender::OnlyNew {
                    marker: marker.clone(),
                    omitted: ids.len() - new_count,
                }
            } else {
                ResultRender::ReplaceWith { marker: marker.clone() }
            };
            return Outcome {
                kind: Some(SignalKind::Plateau),
                render,
                marker: Some(marker),
            };
        }

        Outcome {
            kind: None,
            render: ResultRender::Full,
            marker: None,
        }
    }
}

/// Neutral repeat marker: the exact same (tool,args) call already ran this session.
fn repeat_marker() -> String {
    "\n\n[retrieval: identical query already run this session — result unchanged]".to_string()
}

/// Neutral streak marker: `STREAK_K` consecutive searches surfaced no new information.
fn streak_marker() -> String {
    format!(
        "\n\n[retrieval: last {} searches surfaced no new information]",
        ReaderSignals::STREAK_K
    )
}

/// Neutral plateau marker — same wording as [`RetrievalProgress::observe`]'s, verbatim.
fn plateau_marker(seen: usize, window_sum: usize, window_len: usize) -> String {
    format!(
        "\n\n[retrieval: {seen} unique results gathered; {window_sum} new over the last {window_len} searches — gain has plateaued]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// A productive multi-hop — every call surfaces a genuinely NEW id — must NEVER fire: the
    /// window sum stays >= 1 > G on every call, so `observe` returns `None` throughout, even well
    /// past the M-call and E-id thresholds.
    #[test]
    fn plateau_never_fires_on_a_productive_multihop() {
        let mut p = RetrievalProgress::new();
        for i in 0..12 {
            let id = format!("doc.md#{i}");
            assert_eq!(
                p.observe(&[id.clone()]),
                None,
                "a call that adds a new id ({id}) must not plateau"
            );
        }
    }

    /// The plateau fires only once `calls >= M`, `seen >= E`, and the last-`W` window sum `<= G`
    /// all hold. Call 1 seeds E distinct ids; calls 2..=M add nothing, so by call M the window is
    /// all-zero and the marker fires — not before.
    #[test]
    fn plateau_fires_once_thresholds_met_not_before() {
        let mut p = RetrievalProgress::new();
        // Call 1: E distinct ids in one shot -> seen >= E, but calls < M so no fire yet.
        assert_eq!(p.observe(&ids(&["a", "b", "c"])), None, "call 1: below M");
        // Calls 2..M-1: nothing new, still below the M-call gate.
        for _ in 0..(RetrievalProgress::M - 2) {
            assert_eq!(p.observe(&[]), None, "no-new call below M must stay quiet");
        }
        // Call M: window is now all-zero over the last W calls, seen >= E, calls == M -> FIRE.
        let marker = p
            .observe(&[])
            .expect("plateau must fire once M/E/window-sum thresholds are all met");
        assert!(marker.contains("plateaued"), "marker: {marker}");
        assert!(marker.contains("3 unique results gathered"), "counts in marker: {marker}");
    }

    /// Fires ONCE per plateau: after firing it stays quiet on further no-new calls, then a call
    /// that brings genuinely new ids re-arms it so a SECOND, distinct plateau fires again.
    #[test]
    fn plateau_fires_once_then_rearms_on_new_ids() {
        let mut p = RetrievalProgress::new();
        p.observe(&ids(&["a", "b", "c"])); // seed E ids (call 1)
        for _ in 0..(RetrievalProgress::M - 2) {
            p.observe(&[]);
        }
        assert!(p.observe(&[]).is_some(), "first plateau fires at call M");
        // Still plateaued, but already fired -> silent.
        assert_eq!(p.observe(&[]), None, "second no-new call must not re-fire the same plateau");
        assert_eq!(p.observe(&[]), None, "still silent while plateaued");
        // A genuinely new id re-arms the signal and refills the window with a non-zero count.
        assert_eq!(p.observe(&ids(&["d"])), None, "new id: window sum > G, no fire, re-armed");
        // Drain the window back to all-zero over W calls -> a fresh plateau fires again.
        for _ in 0..(RetrievalProgress::W - 1) {
            assert_eq!(p.observe(&[]), None, "window not yet fully drained");
        }
        assert!(p.observe(&[]).is_some(), "a distinct later plateau must fire again");
    }

    /// The marker is a NEUTRAL observation: it states the counts and that gain has plateaued, and
    /// carries none of the imperative/policy words that belong in the prompt (that's GEPA's job).
    #[test]
    fn plateau_marker_is_a_neutral_observation_not_a_directive() {
        let mut p = RetrievalProgress::new();
        p.observe(&ids(&["a", "b", "c"]));
        for _ in 0..(RetrievalProgress::M - 2) {
            p.observe(&[]);
        }
        let marker = p.observe(&[]).expect("plateau fires").to_lowercase();
        for imperative in ["stop", "answer", "must", "commit", "should", "change approach", "give up"] {
            assert!(
                !marker.contains(imperative),
                "neutral marker must not contain the directive word {imperative:?}: {marker}"
            );
        }
    }

    /// `is_retrieval_tool` gates which tools feed the tracker: the id-surfacing reader tools do,
    /// write/non-id tools don't (so their zero-id results can't drain the plateau window).
    #[test]
    fn is_retrieval_tool_selects_id_surfacing_tools() {
        for t in ["search", "glossary", "related", "neighbors", "reach", "sql", "read"] {
            assert!(is_retrieval_tool(t), "{t} surfaces ids");
        }
        for t in ["glob", "grep", "get_source_file", "graph_stats", "resolve", "unknown"] {
            assert!(!is_retrieval_tool(t), "{t} must not feed the tracker");
        }
    }

    /// Regression guard for fix round 2: the entity-node "— read" anchor and the BARE structural
    /// (Section/Document) anchor must both be captured — the first version of this regex only
    /// matched the "— read" form, so a Section/Document endpoint (rendered by `endpoint_ref`/
    /// `node_ref` with no "read" word at all) surfaced zero ids.
    #[test]
    fn extract_node_ids_matches_entity_anchor_and_bare_structural_forms() {
        // Entity/reasoning node: `tools::read_anchor`'s "— read <path>#<ord> · <label>" (glued,
        // no space — the real form `tools::node_ref` emits).
        let entity_line = "n1  [Entity]  Some Fact   — read doc.md#3 · SecTitle";
        assert_eq!(extract_node_ids(entity_line), vec!["doc.md#3".to_string()]);

        // Bare structural Section endpoint (edge_line / render_reach_chain / glossary's
        // exact-title stub) — no "read" word, just the raw `node_ref` anchor.
        let section_line = "CONTAINS       ->  doc.md#2 · SecB";
        assert_eq!(extract_node_ids(section_line), vec!["doc.md#2".to_string()]);

        // Bare structural Document endpoint — no ord at all.
        let doc_line = "doc.md  (document)";
        assert_eq!(extract_node_ids(doc_line), vec!["doc.md".to_string()]);

        // A line with neither anchor form contributes nothing.
        assert!(extract_node_ids("REFERENCES      ->  fact-9  [Entity]  no anchor here").is_empty());

        // A single space before `#n` inside a node's own label text (NOT a real glued anchor)
        // must NOT spuriously match — this is the false-novelty bug the tightened regex fixes.
        assert!(
            extract_node_ids("n2  [Entity]  reported issue #42 with no read anchor").is_empty(),
            "a spaced '#n' inside label text must not be mistaken for a glued anchor"
        );

        // Multiple anchors on one body (e.g. several neighbors lines) all extract.
        let multi = format!("{section_line}\n{entity_line}\n{doc_line}");
        let mut ids = extract_node_ids(&multi);
        ids.sort();
        let mut want = vec!["doc.md".to_string(), "doc.md#2".to_string(), "doc.md#3".to_string()];
        want.sort();
        assert_eq!(ids, want);
    }

    // ---- ReaderSignals ----------------------------------------------------------------------

    /// A repeat (identical consecutive key) fires ReplaceWith, and its ids are NOT folded into
    /// `seen`/`window` — a following call with the SAME ids under a NEW key still counts them as
    /// genuinely new.
    #[test]
    fn reader_signals_repeat_fires_and_does_not_consume_novelty() {
        let mut r = ReaderSignals::new();
        let out1 = r.observe("search", "search:q1", &ids(&["a", "b"]));
        assert_eq!(out1.kind, None, "first call: nothing to repeat yet");

        let out2 = r.observe("search", "search:q1", &ids(&["a", "b"]));
        assert_eq!(out2.kind, Some(SignalKind::Repeat));
        assert_eq!(
            out2.render,
            ResultRender::ReplaceWith { marker: repeat_marker() }
        );
        assert_eq!(out2.marker, Some(repeat_marker()));

        // A later call under a DIFFERENT key with the SAME ids still counts them as new, proving
        // the repeat call never touched `seen`.
        let out3 = r.observe("search", "search:q2", &ids(&["a", "b"]));
        assert_eq!(out3.kind, None, "genuinely new ids -> no signal");
        assert_eq!(out3.render, ResultRender::Full);
    }

    /// Streak fires at exactly STREAK_K consecutive zero-new VARIED calls, exactly once, then
    /// re-arms once a later call brings a genuinely new id.
    #[test]
    fn reader_signals_streak_fires_once_then_rearms() {
        let mut r = ReaderSignals::new();
        // Seed some ids so the calls below are genuinely varied but yield zero NEW ids.
        assert_eq!(r.observe("search", "k0", &ids(&["a"])).kind, None);

        let mut fired = 0;
        for i in 0..ReaderSignals::STREAK_K {
            let out = r.observe("search", &format!("k{}", i + 1), &[]);
            if out.kind == Some(SignalKind::Streak) {
                fired += 1;
                assert_eq!(i, ReaderSignals::STREAK_K - 1, "streak must fire on the Kth call");
                assert_eq!(out.render, ResultRender::ReplaceWith { marker: streak_marker() });
                assert_eq!(out.marker, Some(streak_marker()));
            } else {
                assert_eq!(out.kind, None, "no signal before the Kth zero-new call");
            }
        }
        assert_eq!(fired, 1, "streak must fire exactly once across the K calls");

        // Immediately after firing, a further zero-new call must NOT re-fire (counter reset).
        assert_eq!(r.observe("search", "k-quiet", &[]).kind, None);

        // A genuinely new id re-arms the counter from zero.
        assert_eq!(r.observe("search", "k-new", &ids(&["b"])).kind, None);

        // A second streak of K zero-new varied calls fires again.
        let mut fired2 = 0;
        for i in 0..ReaderSignals::STREAK_K {
            let out = r.observe("search", &format!("k2-{i}"), &[]);
            if out.kind == Some(SignalKind::Streak) {
                fired2 += 1;
            }
        }
        assert_eq!(fired2, 1, "a distinct later streak fires again");
    }

    /// Plateau fires once the W/M/E/G thresholds hold, rendering OnlyNew when the firing call
    /// still turned up a few new ids, ReplaceWith when it turned up none; it re-arms after a
    /// later call brings genuinely new ground.
    #[test]
    fn reader_signals_plateau_fires_replace_with_on_zero_new() {
        let mut r = ReaderSignals::new();
        // Call 1: seed >= E distinct ids.
        assert_eq!(r.observe("search", "k1", &ids(&["a", "b", "c"])).kind, None);
        // Calls 2, 3: zero-new, fill the window to W with an all-zero trailing history.
        assert_eq!(r.observe("search", "k2", &[]).kind, None);
        assert_eq!(r.observe("search", "k3", &[]).kind, None);
        // Call 4: calls>=M, seen>=E, the PRIOR window [3,0,0] has len 3 but sum 3 -- doesn't
        // satisfy G yet, but it's also this call's zero-new that ticks the streak to K=3, so
        // Streak fires here instead (Streak takes precedence over Plateau in the SAME call).
        let out4 = r.observe("search", "k4", &[]);
        assert_eq!(out4.kind, Some(SignalKind::Streak));
        // Call 5: prior window is now all-zero ([0,0,0], the stale "3" evicted at call 4) and
        // this call itself is zero-new -> Plateau fires with ReplaceWith.
        let out5 = r.observe("search", "k5", &[]);
        assert_eq!(out5.kind, Some(SignalKind::Plateau));
        assert_eq!(
            out5.render,
            ResultRender::ReplaceWith {
                marker: plateau_marker(3, 0, 3)
            }
        );
        assert_eq!(out5.marker, Some(plateau_marker(3, 0, 3)));

        // Re-arm: a later call with a genuinely new id clears fired_plateau; draining the window
        // back to all-zero over W calls fires a fresh, distinct plateau again. The new id at k6
        // lingers in the window for a few calls before it's fully evicted (and a Streak may claim
        // one of the intervening zero-new calls first, same as k4 above) — so scan forward for the
        // next Plateau rather than asserting one exact index.
        assert_eq!(r.observe("search", "k6", &ids(&["d"])).kind, None, "re-arming call itself must not immediately refire");
        let mut saw_plateau_again = false;
        for i in 7..15 {
            let out = r.observe("search", &format!("k{i}"), &[]);
            if out.kind == Some(SignalKind::Plateau) {
                saw_plateau_again = true;
                break;
            }
        }
        assert!(saw_plateau_again, "a distinct later plateau must fire again once the window redrains");
    }

    /// Plateau renders OnlyNew (keeping the omitted count) when the firing call itself still
    /// surfaced some new ids, while the marker reports the plateaued trend leading up to it.
    #[test]
    fn reader_signals_plateau_fires_only_new_when_call_still_has_new_ids() {
        let mut r = ReaderSignals::new();
        assert_eq!(r.observe("search", "k1", &ids(&["a", "b", "c"])).kind, None);
        assert_eq!(r.observe("search", "k2", &[]).kind, None);
        assert_eq!(r.observe("search", "k3", &[]).kind, None);
        // Call 4 is zero-new too, so the streak claims it first (see the ReplaceWith test above
        // for why); state still updates, leaving the window all-zero afterward.
        assert_eq!(r.observe("search", "k4", &[]).kind, Some(SignalKind::Streak));
        // Call 5: prior window all-zero -> Plateau gate holds; THIS call brings 1 new + 1
        // already-seen id -> OnlyNew{omitted: 1}.
        let out5 = r.observe("search", "k5", &ids(&["d", "a"]));
        assert_eq!(out5.kind, Some(SignalKind::Plateau));
        assert_eq!(
            out5.render,
            ResultRender::OnlyNew {
                marker: plateau_marker(4, 0, 3),
                omitted: 1,
            }
        );
        assert_eq!(out5.marker, Some(plateau_marker(4, 0, 3)));
    }

    /// `with_plateau(false)` disarms the plateau signal entirely; repeat and streak still fire.
    #[test]
    fn reader_signals_with_plateau_false_disables_only_plateau() {
        let mut r = ReaderSignals::with_plateau(false);
        assert_eq!(r.observe("search", "k1", &ids(&["a", "b", "c"])).kind, None);
        for i in 0..20 {
            let out = r.observe("search", &format!("k{}", i + 2), &[]);
            assert_ne!(
                out.kind,
                Some(SignalKind::Plateau),
                "plateau must never fire when disarmed"
            );
        }

        // Repeat still fires when disarmed.
        let mut r2 = ReaderSignals::with_plateau(false);
        r2.observe("search", "same", &ids(&["a"]));
        assert_eq!(r2.observe("search", "same", &ids(&["a"])).kind, Some(SignalKind::Repeat));

        // Streak still fires when disarmed.
        let mut r3 = ReaderSignals::with_plateau(false);
        r3.observe("search", "k0", &ids(&["a"]));
        let mut fired = false;
        for i in 0..ReaderSignals::STREAK_K {
            if r3.observe("search", &format!("k{}", i + 1), &[]).kind == Some(SignalKind::Streak) {
                fired = true;
            }
        }
        assert!(fired, "streak must still fire when plateau is disarmed");
    }

    /// `Outcome.marker` is `Some` exactly when `render != Full`, `None` exactly when `render ==
    /// Full`.
    #[test]
    fn reader_signals_marker_matches_render_fullness() {
        let mut r = ReaderSignals::new();
        let out = r.observe("search", "k1", &ids(&["a"]));
        assert_eq!(out.render, ResultRender::Full);
        assert_eq!(out.marker, None);

        let repeat = r.observe("search", "k1", &ids(&["a"]));
        assert_ne!(repeat.render, ResultRender::Full);
        assert!(repeat.marker.is_some());
    }

    /// A productive multi-hop — every call under a distinct key surfaces a genuinely new id —
    /// never fires ANY signal, well past the M-call/E-id thresholds.
    #[test]
    fn reader_signals_never_fires_on_a_productive_multihop() {
        let mut r = ReaderSignals::new();
        for i in 0..12 {
            let id = format!("doc.md#{i}");
            let out = r.observe("search", &format!("k{i}"), &ids(&[&id]));
            assert_eq!(out.kind, None, "a call that adds a new id ({id}) must not fire");
            assert_eq!(out.render, ResultRender::Full);
        }
    }
}
