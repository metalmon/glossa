//! Shared helpers for the "docs vs code" drift check: does a `.md` file name a `kb`/`kbx`
//! flag, subcommand, or MCP tool that doesn't actually exist?
//!
//! This module only extracts and matches tokens — it knows nothing about `clap` or the MCP
//! registry, so it's usable from both binaries (which can't see each other's `Cli`, since each
//! lives in its own crate/binary) and from `tests/` (which sees the MCP tool catalog but not
//! either `Cli`). The three call sites each supply their own "known names" set and walk their
//! own source of truth; this module just answers "what does the doc claim, and where".
//!
//! Kept in small functions with their own unit tests (below) rather than one opaque pass, per
//! the project's testing norms: a failure here should point at which half broke — file
//! discovery, line extraction, or token matching — not just "the drift test failed".

use std::path::{Path, PathBuf};

/// Walk `roots` (directories or single files) collecting every `*.md` file the `ignore` crate's
/// gitignore-aware walker would surface — the same walker ripgrep uses, so anything gitignored
/// (e.g. `docs/superpowers/`) is excluded for free when the walk happens inside a git checkout.
pub fn find_markdown_files(roots: &[&Path]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in roots {
        if root.is_file() {
            if is_markdown(root) {
                out.push(root.to_path_buf());
            }
            continue;
        }
        if !root.is_dir() {
            continue;
        }
        for entry in ignore::WalkBuilder::new(root).build().flatten() {
            let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
            if is_file && is_markdown(entry.path()) {
                out.push(entry.path().to_path_buf());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn is_markdown(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("md")
}

/// One token a doc claims exists in a `kb`/`kbx` command context, with its 1-based source line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedToken {
    pub line: usize,
    pub kind: TokenKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// A `--long-flag` token, checked at any position in the command line.
    Flag,
    /// The bare word immediately after the command word — checked only at this one depth. See
    /// `scan_command_line` for why deeper words aren't checked.
    Subcommand,
}

/// Scan one markdown file's text for `command_word` ("kb" or "kbx") invocations — both fenced
/// code blocks and single-backtick inline spans — and return every claimed flag/subcommand token.
pub fn extract_claims(text: &str, command_word: &str) -> Vec<ClaimedToken> {
    let mut claims = Vec::new();
    let mut in_fence = false;
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        if raw_line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            scan_command_line(raw_line, command_word, line_no, &mut claims);
            continue;
        }
        for span in inline_code_spans(raw_line) {
            scan_command_line(span, command_word, line_no, &mut claims);
        }
    }
    claims
}

/// Every single-backtick inline code span on a line (fenced blocks are handled by the caller's
/// own fence state machine, not this function).
fn inline_code_spans(line: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('`') else { break };
        spans.push(&after[..end]);
        rest = &after[end + 1..];
    }
    spans
}

/// If `line` (a fenced-block line or an inline-span's contents) is a `command_word` invocation —
/// its first whitespace-separated token, minus a leading shell prompt char, equals `command_word`
/// exactly — record its claims: every `--flag` token at any position, and the bare word
/// immediately following the command word (if any) as a subcommand candidate.
///
/// Subcommands are checked only one level deep. Past that, a bare word is ambiguous: `kb mcp
/// somecorpus` has a positional corpus path right where a nested subcommand could also go, and
/// telling the two apart needs a real parse of each subcommand's arity. Flags don't have that
/// ambiguity (always introduced by `--`), so they're checked wherever they appear on the line.
fn scan_command_line(line: &str, command_word: &str, line_no: usize, out: &mut Vec<ClaimedToken>) {
    let mut tokens = line.split_whitespace().peekable();
    // A doc sometimes shows a shell prompt as its own token before the command ("$ kb index
    // --force"); skip at most one of those so it doesn't masquerade as (or hide) the command word.
    if matches!(tokens.peek(), Some(&"$") | Some(&">") | Some(&"#")) {
        tokens.next();
    }
    let Some(first) = tokens.next() else { return };
    if first != command_word {
        return;
    }
    let mut depth0_checked = false;
    for tok in tokens {
        if let Some(flag) = tok.strip_prefix("--") {
            if let Some(name) = flag_name(flag) {
                out.push(ClaimedToken {
                    line: line_no,
                    kind: TokenKind::Flag,
                    text: name,
                });
            }
            depth0_checked = true;
            continue;
        }
        if !depth0_checked {
            depth0_checked = true;
            if is_bare_identifier(tok) {
                out.push(ClaimedToken {
                    line: line_no,
                    kind: TokenKind::Subcommand,
                    text: tok.to_string(),
                });
            }
        }
    }
}

/// `--flag=value` / `--flag:value` -> `flag`; trims trailing punctuation a sentence may have run
/// into the span (e.g. a closing paren). `None` if nothing alphanumeric is left.
fn flag_name(flag: &str) -> Option<String> {
    let name = flag.split(['=', ':']).next().unwrap_or(flag);
    let name = name.trim_end_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
    (!name.is_empty()).then(|| name.to_string())
}

/// A plain lowercase CLI-identifier shape (`graph-doctor`, `service`, `status`) — excludes
/// placeholders (`<path>`), quoted values (`"pattern"`), and paths, which never match this.
fn is_bare_identifier(tok: &str) -> bool {
    let mut chars = tok.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Extract the first-column backtick identifier of every markdown table row inside one section
/// (from a heading line equal to `section_heading` up to the next heading of the same-or-shallower
/// level), e.g. the `docs/mcp.md` `## Tools` table's `| \`search\` | ... |` rows. Narrow and
/// mechanical on purpose: a bare backtick identifier can be any word in prose, but as a table's
/// own first column under a named section it's unambiguously claiming to be the thing that table
/// enumerates.
pub fn extract_table_first_column(text: &str, section_heading: &str) -> Vec<(usize, String)> {
    let target_level = section_heading.chars().take_while(|&c| c == '#').count();
    let mut out = Vec::new();
    let mut in_section = false;
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed == section_heading {
            in_section = true;
            continue;
        }
        if !in_section {
            continue;
        }
        let level = trimmed.chars().take_while(|&c| c == '#').count();
        if level > 0 && level <= target_level {
            break; // next section at or above our level: we're done
        }
        if let Some(name) = table_row_first_backtick_identifier(trimmed) {
            out.push((idx + 1, name));
        }
    }
    out
}

/// `| \`name\` | ... |` -> `Some("name")`, only when the first cell is EXACTLY one bare
/// identifier in backticks (not e.g. `` `resolve` / `delete_by_source` `` or `` `[verify].enabled` ``).
fn table_row_first_backtick_identifier(row: &str) -> Option<String> {
    let rest = row.strip_prefix('|')?.trim_start();
    let rest = rest.strip_prefix('`')?;
    let end = rest.find('`')?;
    let cell = &rest[..end];
    let after = rest[end + 1..].trim_start();
    if !after.starts_with('|') {
        return None;
    }
    is_bare_identifier(cell).then(|| cell.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_code_spans_finds_each_backtick_pair() {
        let line = "Use `kb search --format rg` or `kb grep`.";
        assert_eq!(
            inline_code_spans(line),
            vec!["kb search --format rg", "kb grep"]
        );
    }

    #[test]
    fn inline_code_spans_ignores_unrelated_prose() {
        assert_eq!(inline_code_spans("no code spans here"), Vec::<&str>::new());
    }

    #[test]
    fn scan_command_line_ignores_lines_whose_command_word_differs() {
        let mut out = Vec::new();
        scan_command_line("gh run view --log", "kb", 1, &mut out);
        scan_command_line("cargo clippy -D warnings", "kb", 2, &mut out);
        scan_command_line("rclone --vfs-cache-mode writes", "kb", 3, &mut out);
        assert!(
            out.is_empty(),
            "non-kb command lines must not be scanned: {out:?}"
        );
    }

    #[test]
    fn scan_command_line_strips_a_leading_shell_prompt() {
        let mut out = Vec::new();
        scan_command_line("$ kb index --force", "kb", 1, &mut out);
        assert_eq!(out.len(), 2); // "index" subcommand + "--force" flag
    }

    #[test]
    fn scan_command_line_collects_subcommand_and_flags() {
        let mut out = Vec::new();
        scan_command_line(
            "kb search \"foo\" --ignore-case --limit=10",
            "kb",
            5,
            &mut out,
        );
        assert_eq!(
            out,
            vec![
                ClaimedToken {
                    line: 5,
                    kind: TokenKind::Subcommand,
                    text: "search".into()
                },
                ClaimedToken {
                    line: 5,
                    kind: TokenKind::Flag,
                    text: "ignore-case".into()
                },
                ClaimedToken {
                    line: 5,
                    kind: TokenKind::Flag,
                    text: "limit".into()
                },
            ]
        );
    }

    #[test]
    fn scan_command_line_does_not_treat_a_positional_value_as_a_deeper_subcommand() {
        // "somecorpus" sits right where a nested subcommand could also go; only "mcp" (depth 0)
        // is checked, matching the documented under-reach.
        let mut out = Vec::new();
        scan_command_line("kb mcp somecorpus --profile editor", "kb", 1, &mut out);
        assert_eq!(
            out,
            vec![
                ClaimedToken {
                    line: 1,
                    kind: TokenKind::Subcommand,
                    text: "mcp".into()
                },
                ClaimedToken {
                    line: 1,
                    kind: TokenKind::Flag,
                    text: "profile".into()
                },
            ]
        );
    }

    #[test]
    fn scan_command_line_skips_subcommand_check_once_a_flag_leads() {
        // A global flag before the subcommand (`kb --root x search ...`) makes the depth-0 slot a
        // flag, not a bare word; we deliberately don't try to recover the real subcommand after it.
        let mut out = Vec::new();
        scan_command_line("kb --root x search foo", "kb", 1, &mut out);
        assert_eq!(
            out,
            vec![ClaimedToken {
                line: 1,
                kind: TokenKind::Flag,
                text: "root".into()
            }]
        );
    }

    #[test]
    fn extract_claims_only_scans_inside_fences_and_inline_spans() {
        let text = "\
Some prose mentioning kb search is not code.

```bash
kb search --word-regexp foo
```

Inline: `kb read --help`.
";
        let claims = extract_claims(text, "kb");
        let flags: Vec<_> = claims
            .iter()
            .filter(|c| c.kind == TokenKind::Flag)
            .map(|c| c.text.as_str())
            .collect();
        assert_eq!(flags, vec!["word-regexp", "help"]);
    }

    #[test]
    fn extract_table_first_column_stops_at_the_next_heading() {
        let text = "\
## Tools

| Tool | Purpose |
|------|---------|
| `search` | find things |
| `graph_upsert` | write a node |

## Next section

| `not_a_tool` | should not be collected |
";
        let rows = extract_table_first_column(text, "## Tools");
        assert_eq!(
            rows,
            vec![(5, "search".to_string()), (6, "graph_upsert".to_string())]
        );
    }

    #[test]
    fn extract_table_first_column_rejects_multi_token_or_bracketed_cells() {
        let text = "\
## Tools

| `resolve` / `delete_by_source` | not a single bare name |
| `[verify].enabled` | a config key, not a tool |
| `search` | a real one |
";
        let rows = extract_table_first_column(text, "## Tools");
        assert_eq!(rows, vec![(5, "search".to_string())]);
    }

    #[test]
    fn find_markdown_files_collects_a_single_file_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("README.md");
        std::fs::write(&file, "# hi").unwrap();
        let found = find_markdown_files(&[file.as_path()]);
        assert_eq!(found, vec![file]);
    }

    #[test]
    fn find_markdown_files_skips_non_markdown_and_missing_roots() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "x").unwrap();
        let missing = dir.path().join("does-not-exist");
        let found = find_markdown_files(&[dir.path(), missing.as_path()]);
        assert!(found.is_empty());
    }
}
