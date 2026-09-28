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
    download_files_tolerant(repo, revision, files, to_dir, &[])
}

/// Like [`download_files`], but a `404` on a file listed in `optional` is skipped (not an error) —
/// used for `model.onnx.data`, which exists only when the fp32 export has external data. Any other
/// error, or a `404` on a non-optional file, still fails.
pub fn download_files_tolerant(
    repo: &str,
    revision: &str,
    files: &[String],
    to_dir: &Path,
    optional: &[&str],
) -> Result<Vec<(PathBuf, u64)>> {
    std::fs::create_dir_all(to_dir)
        .with_context(|| format!("creating download dir {}", to_dir.display()))?;

    let mut out = Vec::with_capacity(files.len());
    for file in files {
        let url = hf_resolve_url(repo, revision, file);
        let dest = to_dir.join(file);

        let resp = match ureq::get(&url).call() {
            Ok(r) => r,
            Err(ureq::Error::Status(404, _)) if optional.contains(&file.as_str()) => {
                continue; // optional file absent on the repo (e.g. single-file fp32 has no .data)
            }
            Err(e) => return Err(anyhow::anyhow!("GET {url}: {e}")),
        };
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

/// A published ONNX precision variant. Each maps to a distinct HuggingFace filename; a downloaded
/// dir is canonicalized so the loaded file is always `model.onnx` (see [`download_variant`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Fp32,
    Fp16,
    Int8,
}

impl Variant {
    /// The repo filename for this variant's ONNX graph.
    pub fn onnx_file(self) -> &'static str {
        match self {
            Variant::Fp32 => "model.onnx",
            Variant::Fp16 => "model.fp16.onnx",
            Variant::Int8 => "model.int8.onnx",
        }
    }
}

/// The HuggingFace files to fetch for a variant: its ONNX (+ `model.onnx.data` for fp32, which may
/// be absent for a single-file fp32 export — treated as optional) plus the shared tokenizer/config.
pub fn variant_files(v: Variant) -> Vec<String> {
    let mut f = vec![v.onnx_file().to_string()];
    if v == Variant::Fp32 {
        f.push("model.onnx.data".to_string());
    }
    f.push("tokenizer.json".to_string());
    f.push("config.json".to_string());
    f
}

/// Rename `<dir>/model.<v>.onnx` -> `<dir>/model.onnx` so the model dir is canonical and the engine's
/// `resolve_model_file` loads it unchanged. No-op for [`Variant::Fp32`] (already `model.onnx`).
pub fn canonicalize_onnx(dir: &Path, v: Variant) -> Result<()> {
    if v == Variant::Fp32 {
        return Ok(());
    }
    let from = dir.join(v.onnx_file());
    let to = dir.join("model.onnx");
    std::fs::rename(&from, &to)
        .with_context(|| format!("renaming {} -> {}", from.display(), to.display()))?;
    Ok(())
}

/// Download one precision `variant` from `repo`@`revision` into `to_dir` and canonicalize the ONNX to
/// `model.onnx`. `model.onnx.data` (fp32 external data) is optional. Returns the `(path, bytes)` of
/// the files actually fetched.
pub fn download_variant(
    repo: &str,
    revision: &str,
    to_dir: &Path,
    variant: Variant,
) -> Result<Vec<(PathBuf, u64)>> {
    let files = variant_files(variant);
    let out = download_files_tolerant(repo, revision, &files, to_dir, &["model.onnx.data"])?;
    canonicalize_onnx(to_dir, variant)?;
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

#[cfg(test)]
mod variant_tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn variant_files_map() {
        assert_eq!(
            variant_files(Variant::Fp32),
            v(&[
                "model.onnx",
                "model.onnx.data",
                "tokenizer.json",
                "config.json"
            ])
        );
        assert_eq!(
            variant_files(Variant::Fp16),
            v(&["model.fp16.onnx", "tokenizer.json", "config.json"])
        );
        assert_eq!(
            variant_files(Variant::Int8),
            v(&["model.int8.onnx", "tokenizer.json", "config.json"])
        );
    }

    #[test]
    fn canonicalize_renames_fp16_to_model_onnx() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("model.fp16.onnx"), b"x").unwrap();
        canonicalize_onnx(d.path(), Variant::Fp16).unwrap();
        assert!(d.path().join("model.onnx").is_file());
        assert!(!d.path().join("model.fp16.onnx").exists());
    }

    #[test]
    fn canonicalize_fp32_is_noop() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("model.onnx"), b"x").unwrap();
        canonicalize_onnx(d.path(), Variant::Fp32).unwrap();
        assert!(d.path().join("model.onnx").is_file());
    }
}
