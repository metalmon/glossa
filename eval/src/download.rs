//! `kbx download`: fetch the ONNX NLI model + tokenizer from a HuggingFace repo into a local
//! `model_dir` (the dir `[verify.nli].model_dir` then points at). Split into a PURE URL-building
//! core (unit-tested, no network) and a thin blocking fetch on top of `ureq` (not network-tested —
//! CI has no network; see the module tests for what IS covered).

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// HuggingFace "resolve" URL for one file in a repo at a given revision:
/// `https://huggingface.co/<repo>/resolve/<revision>/<file>`. Leading/trailing slashes on `repo`
/// and `file` are trimmed so callers can pass either `"org/repo"` or `"org/repo/"` and either
/// `"file.bin"` or `"/file.bin"` without producing a doubled-slash URL. `revision` is used as
/// given (callers default it to `"main"`, e.g. via the CLI's `--revision` default).
pub fn hf_resolve_url(repo: &str, revision: &str, file: &str) -> String {
    let repo = repo.trim_matches('/');
    let file = file.trim_matches('/');
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

/// Download `files` from `repo`@`revision` into `to_dir` (created if absent), streaming each
/// response body straight to disk (no full in-memory buffer). Returns `(path, bytes)` per file, in
/// the same order as `files`.
///
/// `expected_sha256`, when it has an entry for a given file name, would be checked after that file
/// is written (hex, case-insensitive), with a mismatch deleting the partial file and returning an
/// error — but this crate does not (yet) depend on `sha2`, so checksum verification is SKIPPED
/// entirely (size-only, matching the module-level doc): every downloaded file is checked for
/// non-emptiness (`bytes > 0`) instead. `expected_sha256` is accepted for forward-compatibility
/// with a future `--sha256` flag once `sha2` is added as a dependency; today the CLI always passes
/// `None`.
pub fn download_files(
    repo: &str,
    revision: &str,
    files: &[String],
    to_dir: &Path,
    expected_sha256: Option<&HashMap<String, String>>,
) -> Result<Vec<(PathBuf, u64)>> {
    std::fs::create_dir_all(to_dir)
        .with_context(|| format!("creating download dir {}", to_dir.display()))?;

    let mut out = Vec::with_capacity(files.len());
    for file in files {
        let url = hf_resolve_url(repo, revision, file);
        let dest = to_dir.join(file);

        let resp = ureq::get(&url)
            .call()
            .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
        let mut reader = resp.into_reader();
        let mut out_file =
            std::fs::File::create(&dest).with_context(|| format!("creating {}", dest.display()))?;
        let bytes = std::io::copy(&mut reader, &mut out_file)
            .with_context(|| format!("writing {}", dest.display()))?;

        if bytes == 0 {
            let _ = std::fs::remove_file(&dest);
            bail!("empty download: {url} wrote 0 bytes");
        }

        // Checksum verification is intentionally SKIPPED (size-only) — no `sha2` dependency in
        // this crate today; see the function doc comment. `expected_sha256` is accepted but unused
        // until a future `--sha256` flag adds that dependency.
        let _ = expected_sha256;

        out.push((dest, bytes));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_url_basic() {
        assert_eq!(
            hf_resolve_url("metalmon/rubert-nli-threeway-onnx", "main", "model.onnx"),
            "https://huggingface.co/metalmon/rubert-nli-threeway-onnx/resolve/main/model.onnx"
        );
    }

    #[test]
    fn resolve_url_trims_repo_slashes() {
        assert_eq!(
            hf_resolve_url("metalmon/repo/", "main", "tokenizer.json"),
            "https://huggingface.co/metalmon/repo/resolve/main/tokenizer.json"
        );
    }

    #[test]
    fn resolve_url_trims_leading_repo_slash() {
        assert_eq!(
            hf_resolve_url("/metalmon/repo", "main", "config.json"),
            "https://huggingface.co/metalmon/repo/resolve/main/config.json"
        );
    }

    #[test]
    fn resolve_url_trims_file_slashes() {
        assert_eq!(
            hf_resolve_url("metalmon/repo", "main", "/model.onnx"),
            "https://huggingface.co/metalmon/repo/resolve/main/model.onnx"
        );
    }

    #[test]
    fn resolve_url_non_main_revision() {
        assert_eq!(
            hf_resolve_url("metalmon/repo", "v1.0", "model.onnx"),
            "https://huggingface.co/metalmon/repo/resolve/v1.0/model.onnx"
        );
    }
}
