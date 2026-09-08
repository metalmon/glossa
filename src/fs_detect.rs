//! Injectable network-filesystem detector.
//!
//! The state-dir separation (see `root.rs`) assumes the state directory lives on local disk:
//! file locks, atomic rename and SQLite WAL are unreliable on network filesystems and
//! reintroduce the corruption that separation avoids. This module detects that case behind a
//! trait so the warning logic (Task 9/11) is unit-testable without a real SMB/NFS mount.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsKind {
    Local,
    Network,
    Unknown,
}

/// Injectable so tests can assert the warning logic without a real SMB/NFS mount.
pub trait FsDetector {
    fn classify(&self, path: &std::path::Path) -> FsKind;
}

// These f_type magics and `kind_from_magic` are live on the production target (Linux, via the
// `statfs` call below) and in the unit tests on every target; they are dead only in a non-test
// build on a non-Linux dev host, hence the `allow(dead_code)`.
/// NFS (`statfs.f_type` on a mounted NFS export).
#[allow(dead_code)]
const NFS_SUPER_MAGIC: u64 = 0x6969;
/// SMB (older SMB1/CIFS client magic).
#[allow(dead_code)]
const SMB_SUPER_MAGIC: u64 = 0x517b;
/// CIFS/SMB2 client magic (Linux `cifs.ko`).
#[allow(dead_code)]
const CIFS_MAGIC_NUMBER: u64 = 0xff534d42;
/// FUSE (covers sshfs/rclone-mounted network shares presented as FUSE).
#[allow(dead_code)]
const FUSE_SUPER_MAGIC: u64 = 0x65735546;

/// Pure `f_type` magic -> `FsKind` mapping, factored out of the `statfs` call so it is
/// unit-testable without a syscall (and without `cfg(target_os = "linux")`).
#[allow(dead_code)]
fn kind_from_magic(magic: u64) -> FsKind {
    match magic {
        NFS_SUPER_MAGIC | SMB_SUPER_MAGIC | CIFS_MAGIC_NUMBER | FUSE_SUPER_MAGIC => {
            FsKind::Network
        }
        _ => FsKind::Local,
    }
}

/// Real detector: `statfs` magic on Linux, `Unknown` on every other target (the warning simply
/// never fires there; the production deployment target is Linux).
pub struct SysFsDetector;

impl FsDetector for SysFsDetector {
    fn classify(&self, path: &std::path::Path) -> FsKind {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::ffi::OsStrExt;
            let Ok(cpath) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
                return FsKind::Unknown;
            };
            // SAFETY: zeroed statfs is a valid POD; we only read f_type after a success return.
            let mut st: libc::statfs = unsafe { std::mem::zeroed() };
            if unsafe { libc::statfs(cpath.as_ptr(), &mut st) } != 0 {
                return FsKind::Unknown;
            }
            kind_from_magic(st.f_type as u64)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            FsKind::Unknown
        }
    }
}

/// Returns `Some(warning)` when the state-dir is on a network filesystem; `None` otherwise. Pure
/// over the detector so it is fully unit-testable.
pub fn state_dir_network_warning(
    det: &dyn FsDetector,
    state_base: &std::path::Path,
) -> Option<String> {
    match det.classify(state_base) {
        FsKind::Network => Some(format!(
            "state-dir {} appears to be on a NETWORK filesystem — file locks, atomic rename and \
             SQLite WAL are unreliable there and reintroduce the corruption this separation avoids. \
             Point --state-dir at LOCAL disk.",
            state_base.display()
        )),
        FsKind::Local | FsKind::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    struct Fake(FsKind);
    impl FsDetector for Fake {
        fn classify(&self, _p: &Path) -> FsKind {
            self.0
        }
    }

    #[test]
    fn warns_on_network_state_dir() {
        let w = state_dir_network_warning(&Fake(FsKind::Network), &PathBuf::from("/mnt/x"));
        assert!(w.unwrap().to_lowercase().contains("network"));
    }
    #[test]
    fn silent_on_local_or_unknown() {
        assert!(
            state_dir_network_warning(&Fake(FsKind::Local), Path::new("/var/lib/glossa"))
                .is_none()
        );
        assert!(
            state_dir_network_warning(&Fake(FsKind::Unknown), Path::new("/whatever")).is_none()
        );
    }

    /// Exercises the magic -> `FsKind` mapping directly (no syscall), so it runs on any target
    /// including this non-Linux build.
    #[test]
    fn kind_from_magic_classifies_known_network_fs_types() {
        for magic in [
            NFS_SUPER_MAGIC,
            SMB_SUPER_MAGIC,
            CIFS_MAGIC_NUMBER,
            FUSE_SUPER_MAGIC,
        ] {
            assert_eq!(kind_from_magic(magic), FsKind::Network);
        }
        // ext4 — an arbitrary local filesystem magic, not in the network set.
        assert_eq!(kind_from_magic(0xEF53), FsKind::Local);
    }
}
