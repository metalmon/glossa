//! Default `.ignore` seeding.
//!
//! glossa walks EVERY file and reads unknown types as text (see `extract::extract_file`), so a
//! corpus that happens to contain installers or archives (a 3 GB `.exe`, a `.zip`) gets slurped
//! into memory and indexed as binary garbage — the dominant cost of a first index on a real mixed
//! folder. Rather than hard-code a blacklist of binary extensions, we seed a `.ignore` that
//! *whitelists* the types we actually extract, using the gitignore idiom "ignore everything, then
//! re-include": `*` drops all files, `!*/` keeps directories so the walk still descends, and each
//! `!*.ext` re-includes a supported type. The whole policy lives in this one editable file — add
//! `!*.myext` to include a non-standard type, delete a line to exclude one. glossa's walk already
//! honours `.ignore` (ripgrep's `ignore` crate), so no walk-side code is needed.

use std::path::{Path, PathBuf};

/// The seeded whitelist. Kept in sync with the extractors in `walk::extractors` + the text/csv/html
/// fallbacks in `extract::extract_file`. Editable by the user after seeding.
pub const DEFAULT_IGNORE: &str = "\
# glossa: index only supported document types. This file is a WHITELIST via the gitignore idiom:
#   `*` ignores everything, `!*/` keeps directories so the walk descends, each `!*.ext` re-includes
#   a type glossa can extract. Edit freely: add `!*.myext` to include a type, delete a line to drop
#   one. Delete this file entirely to go back to indexing every file.
*
!*/

# documents
!*.pdf
!*.docx
!*.doc
!*.xlsx
!*.xls
!*.pptx
!*.ppt
!*.odt
!*.ods
!*.odp
!*.md
!*.markdown
!*.csv
!*.tsv
!*.html
!*.htm

# images (indexed by filename)
!*.png
!*.jpg
!*.jpeg
!*.gif
!*.webp
!*.bmp
!*.tif
!*.tiff

# text & code
!*.txt
!*.log
!*.rst
!*.json
!*.yaml
!*.yml
!*.toml
!*.xml
!*.ini
!*.cfg
!*.conf
!*.rs
!*.py
!*.js
!*.ts
!*.tsx
!*.jsx
!*.java
!*.kt
!*.c
!*.h
!*.cpp
!*.cc
!*.hpp
!*.cs
!*.go
!*.rb
!*.php
!*.sh
!*.bash
!*.ps1
!*.sql
!*.r
!*.lua
!*.swift
";

/// Write the default whitelist `.ignore` at `root`, but ONLY when the corpus has no ignore file of
/// its own (`.ignore` or `.gitignore`) — never clobber a user's existing setup. Returns the written
/// path, or `None` if one already existed (or the write failed). Idempotent: a second call is a
/// no-op once `.ignore` exists.
pub fn seed_if_absent(root: &Path) -> Option<PathBuf> {
    let dot_ignore = root.join(".ignore");
    let git_ignore = root.join(".gitignore");
    if dot_ignore.exists() || git_ignore.exists() {
        return None;
    }
    std::fs::write(&dot_ignore, DEFAULT_IGNORE).ok()?;
    Some(dot_ignore)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_a_whitelist_ignore_when_none_present() {
        let d = tempfile::tempdir().unwrap();
        let p = seed_if_absent(d.path()).expect("seeds when no ignore file exists");
        assert_eq!(p, d.path().join(".ignore"));
        let content = std::fs::read_to_string(&p).unwrap();
        // The gitignore-whitelist idiom must be intact.
        assert!(content.contains("\n*\n"), "ignores everything first");
        assert!(content.contains("!*/"), "keeps directories so the walk descends");
        assert!(content.contains("!*.pdf") && content.contains("!*.docx"));
        assert!(content.contains("!*.png"), "images stay whitelisted");
    }

    #[test]
    fn is_idempotent_and_never_clobbers_existing() {
        let d = tempfile::tempdir().unwrap();
        seed_if_absent(d.path()).unwrap();
        // Second call is a no-op — the file already exists.
        assert!(seed_if_absent(d.path()).is_none());
        // A user edit must survive a later seed attempt.
        std::fs::write(d.path().join(".ignore"), "*\n!*.pdf\n").unwrap();
        assert!(seed_if_absent(d.path()).is_none());
        assert_eq!(
            std::fs::read_to_string(d.path().join(".ignore")).unwrap(),
            "*\n!*.pdf\n"
        );
    }

    #[test]
    fn does_not_seed_when_a_gitignore_exists() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(".gitignore"), "target/\n").unwrap();
        assert!(seed_if_absent(d.path()).is_none(), "respects an existing .gitignore");
        assert!(!d.path().join(".ignore").exists());
    }
}
