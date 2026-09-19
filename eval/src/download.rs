//! `kbx download`: fetch the ONNX NLI model + tokenizer from a HuggingFace repo into a local
//! `model_dir` (the dir `[verify.nli].model_dir` then points at). Split into a PURE URL-building
//! core (unit-tested, no network) and a thin blocking fetch on top of `ureq` (not network-tested —
//! CI has no network; see the module tests for what IS covered).

use anyhow::{bail, Context, Result};
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
/// No checksum verification: this crate does not depend on `sha2`, so integrity is checked
/// size-only — every downloaded file is checked for non-emptiness (`bytes > 0`). A future
/// `--sha256` flag can add real verification (and the `sha2` dependency) if needed.
pub fn download_files(
    repo: &str,
    revision: &str,
    files: &[String],
    to_dir: &Path,
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
            .with_context(|| format!("writing {} from {url}", dest.display()))?;

        if bytes == 0 {
            let _ = std::fs::remove_file(&dest);
            bail!("empty download: {url} wrote 0 bytes");
        }

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
