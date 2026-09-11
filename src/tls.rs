//! Native TLS + mTLS for the streamable-http MCP listener, behind the opt-in `tls` cargo
//! feature (§3b). The default build never compiles this module and stays plaintext (a
//! reverse-proxy/TLS-terminator in front is the default-supported deployment).
//!
//! Design: `ReloadableTls` holds the live `rustls::ServerConfig` behind an `ArcSwap`. `serve_tls`
//! wraps the bound `TcpListener` in `TlsListener`. A `reload()` (SIGHUP or cert-file mtime change)
//! only ever `store`s a new `Arc<ServerConfig>` -- connections already past their handshake keep
//! the acceptor/session they were built with, so in-flight connections are undisturbed; only new
//! connections observe the reloaded cert. A failed handshake (e.g. a plaintext client hitting the
//! TLS port) is logged and the offending connection is dropped; the listener keeps accepting.
//!
//! FU1 (handshake head-of-line DoS): `TlsListener` does NOT perform the TLS handshake inline in
//! its `accept()` future. A background task owns the raw `TcpListener`, accepts TCP as fast as it
//! allows, and spawns a SEPARATE task per connection to drive that connection's handshake under
//! `handshake_timeout_from_env()`'s deadline; only a completed handshake is handed to
//! `accept()` (via a channel). A client that opens TCP and stalls the handshake therefore only
//! ties up its own spawned task -- it cannot block new TCP accepts or other pending handshakes.
//!
//! Round-1 fix follow-ups (C1/I1/I3, see `fu1-fix-round1.md`) hardened this further:
//!
//! - **C1 (pre-auth handshake bound).** The background task spawns one handshake task per raw TCP
//!   accept with NO bound of its own -- unbounded when `GLOSSA_MCP_MAX_CONNECTIONS` is unset (the
//!   default TLS deployment), letting a pre-auth TCP flood exhaust FDs/memory. A dedicated
//!   `tokio::sync::Semaphore` (`GLOSSA_MCP_MAX_HANDSHAKES`, default
//!   [`DEFAULT_MAX_HANDSHAKES`]) now bounds concurrent IN-FLIGHT handshakes, independent of and
//!   in addition to the connection cap, applying whether or not that cap is set. `try_acquire`
//!   (never `.await`) is used in the accept loop itself, so an exhausted bound SHEDS the new
//!   connection immediately instead of queuing (queuing there would re-create FU1's head-of-line
//!   blocking at this new semaphore). The permit is held by the spawned handshake task and
//!   released on every exit path (success, TLS error, timeout).
//! - **I1 (connection cap now counts ESTABLISHED sessions).** `GLOSSA_MCP_MAX_CONNECTIONS`'s
//!   permit is acquired ONLY after a successful handshake, immediately before handoff to the
//!   serve loop -- NOT at raw accept. This is a deliberate divergence from the plaintext
//!   `conn_cap::CappedListener` (which caps at accept, before any handshake): over TLS, a
//!   pre-auth flood can only ever contend the self-clearing, timeout-bounded handshake semaphore
//!   above, never the connection cap, so legitimate authenticated clients are never starved out
//!   by anonymous stalled handshakes. Total FDs are still bounded either way (by
//!   `max_handshakes + max_connections`, both finite). When the cap is unset there is no
//!   post-handshake acquire at all -- unchanged, no cap.
//! - **I3 (no silent deafness).** A raw `accept()` error is now logged (`tracing::warn!`) with a
//!   short bounded backoff before the loop retries, instead of the previous design's failure mode
//!   of parking forever with no signal. If the background accept task itself ever ends
//!   unexpectedly (only reachable via a panic; the loop otherwise never exits), `TlsListener`'s
//!   `accept()` now logs `tracing::error!` before parking, instead of doing so silently.
//!
//! Round-2 fix follow-ups (N1/M1, see `fu1-fix-round2.md`) closed a further, reverse re-coupling
//! that round 1 introduced, plus escalation for persistent accept() failures:
//!
//! - **N1 (connection-cap acquire is now NON-BLOCKING; full cap SHEDS, never queues).** Round 1's
//!   I1 fix acquired the connection-cap permit with an AWAITED `acquire_owned()` while the
//!   spawned handshake task still held its (now-pointless, since the handshake already finished)
//!   pre-auth handshake permit. With long-lived sessions (the normal case for streamable-http/SSE
//!   MCP) and the cap set, enough post-handshake connections parked on a full cap would consume
//!   every pre-auth handshake permit too -- re-coupling the two bounds in the OPPOSITE direction
//!   from I1's original bug, starving new clients at the pre-auth gate instead of the cap.
//!   Dropping the handshake permit before an awaited cap-acquire was rejected as a fix: it would
//!   leave an unbounded number of fully-handshaken connections parked on the cap holding NO
//!   permit at all -- an app-level-unbounded queue, i.e. a post-handshake variant of C1. The
//!   actual fix makes the cap acquire `try_acquire_owned()` (never `.await`): `Ok` proceeds
//!   exactly as before; `Err` (cap full) SHEDS the connection immediately (closes it, logs a
//!   warning) instead of queuing. Over TLS, therefore, a full connection cap SHEDS newly
//!   handshaken connections rather than queuing them -- UNLIKE the plaintext
//!   `conn_cap::CappedListener`, which queues excess in the bounded kernel accept backlog. Total
//!   FDs stay bounded by `max_handshakes + max_connections` either way; the two bounds are now
//!   independent in BOTH directions (pre-auth floods can't starve the cap, I1; cap exhaustion
//!   can't starve pre-auth handshakes, N1).
//! - **M1 (accept() error escalation).** A persistent run of `accept()` failures used to produce
//!   an unbounded `warn!` flood indistinguishable from ordinary transient churn. `accept_loop` now
//!   tracks consecutive failures (reset to 0 on any success) and emits ONE `tracing::error!` the
//!   moment the count crosses a fixed threshold (10) -- a single clear operator/watchdog signal,
//!   not a second flood -- while still never exiting the loop (never-deafen, I3, is unaffected).
#![cfg(feature = "tls")]

use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::{Context as _, Result};
use arc_swap::ArcSwap;
use axum::serve::Listener;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use rustls::{RootCertStore, ServerConfig};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_rustls::{server::TlsStream, TlsAcceptor};

/// Cert/key (+ optional client-CA for mTLS) file paths, resolved once from CLI flags/env at
/// startup. `reload()` re-reads these SAME paths, so a cert rotation that keeps the path stable
/// (the common `certbot`/ACME renewal pattern) is picked up without a restart.
#[derive(Clone, Debug)]
pub struct TlsFiles {
    pub cert: PathBuf,
    pub key: PathBuf,
    pub client_ca: Option<PathBuf>,
}

fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let f = File::open(path).with_context(|| format!("open TLS cert file {}", path.display()))?;
    let mut rd = BufReader::new(f);
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut rd)
        .collect::<Result<Vec<_>, io::Error>>()
        .with_context(|| format!("parse PEM certificates from {}", path.display()))?;
    if certs.is_empty() {
        anyhow::bail!("no certificates found in {}", path.display());
    }
    Ok(certs)
}

fn load_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    let f = File::open(path).with_context(|| format!("open TLS key file {}", path.display()))?;
    let mut rd = BufReader::new(f);
    rustls_pemfile::private_key(&mut rd)
        .with_context(|| format!("parse PEM private key from {}", path.display()))?
        .ok_or_else(|| anyhow::anyhow!("no private key found in {}", path.display()))
}

/// Load `f.cert`/`f.key` and build a rustls `ServerConfig`. When `f.client_ca` is set, installs a
/// `WebPkiClientVerifier` over that CA's certificates: client certs are REQUIRED and verified
/// against it (mTLS). Otherwise no client authentication is performed.
pub fn build_server_config(f: &TlsFiles) -> Result<Arc<ServerConfig>> {
    let certs = load_certs(&f.cert)?;
    let key = load_key(&f.key)?;
    let builder = ServerConfig::builder();
    let config = match &f.client_ca {
        Some(ca_path) => {
            let ca_certs = load_certs(ca_path)?;
            let mut roots = RootCertStore::empty();
            for c in ca_certs {
                roots
                    .add(c)
                    .context("add client-CA certificate to the mTLS root store")?;
            }
            let verifier = WebPkiClientVerifier::builder(Arc::new(roots))
                .build()
                .context("build mTLS client-certificate verifier")?;
            builder
                .with_client_cert_verifier(verifier)
                .with_single_cert(certs, key)
                .context("install server certificate/key (mTLS)")?
        }
        None => builder
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .context("install server certificate/key")?,
    };
    Ok(Arc::new(config))
}

fn file_mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// A snapshot of the TLS input files' mtimes, used by the reload-poll task to detect a cert
/// rotation without re-reading/parsing the files on every tick.
type FileMtimes = (
    Option<std::time::SystemTime>,
    Option<std::time::SystemTime>,
    Option<std::time::SystemTime>,
);

/// The live TLS server config, reloadable in place. New connections pick up whatever `current()`
/// returns at THEIR accept time; already-established connections are unaffected by a later
/// `reload()` (see module docs).
pub struct ReloadableTls {
    cfg: ArcSwap<ServerConfig>,
    files: TlsFiles,
}

impl ReloadableTls {
    /// Build the initial config from `files` (fails the same way `build_server_config` does).
    pub fn new(files: TlsFiles) -> Result<Self> {
        let cfg = build_server_config(&files)?;
        Ok(Self {
            cfg: ArcSwap::new(cfg),
            files,
        })
    }

    /// Re-read `self.files` from disk and, on success, swap in the new config for future
    /// connections. On failure the PREVIOUS config keeps serving (a bad/partially-written cert
    /// file never takes the listener down); the error is returned for the caller to log.
    pub fn reload(&self) -> Result<()> {
        let cfg = build_server_config(&self.files)?;
        self.cfg.store(cfg);
        Ok(())
    }

    /// The config in effect right now.
    pub fn current(&self) -> Arc<ServerConfig> {
        self.cfg.load_full()
    }

    fn file_mtimes(&self) -> FileMtimes {
        (
            file_mtime(&self.files.cert),
            file_mtime(&self.files.key),
            self.files.client_ca.as_deref().and_then(file_mtime),
        )
    }
}

/// mtime-poll `reloadable`'s cert/key/client-ca files (~5s, cross-platform incl. Windows where
/// there is no SIGHUP) and `reload()` on change. Mirrors `logreload::spawn_poll`'s shape. Stops
/// with `cancel`.
pub fn spawn_reload_poll(
    reloadable: Arc<ReloadableTls>,
    cancel: tokio_util::sync::CancellationToken,
) {
    tokio::spawn(async move {
        let mut last = reloadable.file_mtimes();
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tick.tick() => {
                    let cur = reloadable.file_mtimes();
                    if cur != last {
                        last = cur;
                        match reloadable.reload() {
                            Ok(()) => tracing::info!("TLS cert/key reloaded (file change detected)"),
                            Err(e) => tracing::warn!("TLS reload failed, keeping the previous cert: {e:#}"),
                        }
                    }
                }
            }
        }
    });
}

/// How many successfully-handshaken connections `TlsListener`'s background accept task (FU1) may
/// hold in its hand-off channel before axum's serve loop has picked them up. Generous enough that
/// a burst of concurrent handshake completions isn't throttled by this hand-off; bounded so it
/// can't itself become an unbounded buffer.
const HANDSHAKE_CHANNEL_CAPACITY: usize = 64;

/// Default per-handshake timeout (FU1, anti head-of-line-blocking DoS). Override via
/// `GLOSSA_MCP_TLS_HANDSHAKE_TIMEOUT_SECS` (seconds; 0, unset, or unparseable falls back to this).
pub const DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECS: u64 = 10;

/// Resolve the per-handshake timeout `TlsListener` enforces on every individual TLS handshake
/// (FU1), from `GLOSSA_MCP_TLS_HANDSHAKE_TIMEOUT_SECS` with a sane built-in default.
pub fn handshake_timeout_from_env() -> Duration {
    std::env::var("GLOSSA_MCP_TLS_HANDSHAKE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_TLS_HANDSHAKE_TIMEOUT_SECS))
}

/// Default bound on concurrent IN-FLIGHT (not-yet-completed) TLS handshakes (C1, round-1 fix).
/// Applies whether or not `GLOSSA_MCP_MAX_CONNECTIONS` is set -- this is what protects the
/// default (uncapped) TLS deployment from a pre-auth TCP-flood FD/memory exhaustion. Override via
/// `GLOSSA_MCP_MAX_HANDSHAKES` (0, unset, or unparseable falls back to this).
pub const DEFAULT_MAX_HANDSHAKES: usize = 256;

/// Resolve the pre-auth handshake concurrency bound from `GLOSSA_MCP_MAX_HANDSHAKES`, falling
/// back to [`DEFAULT_MAX_HANDSHAKES`].
pub fn max_handshakes_from_env() -> usize {
    std::env::var("GLOSSA_MCP_MAX_HANDSHAKES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_HANDSHAKES)
}

/// A raw IO paired with an OPTIONAL connection-cap permit, held for as long as this value lives.
/// Unlike `conn_cap::PermitGuardedIo` (always-present permit, acquired at raw accept -- correct
/// for the plaintext path), the permit here is acquired AFTER a successful TLS handshake (I1,
/// round-1 fix) and is simply absent (`None`) when no cap is configured. Dropping this value (any
/// exit path -- connection close, error, channel-full shed) drops the permit, if any.
struct SessionCapped<Io> {
    io: Io,
    _permit: Option<OwnedSemaphorePermit>,
}

impl<Io: AsyncRead + AsyncWrite + Unpin> AsyncRead for SessionCapped<Io> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_read(cx, buf)
    }
}

impl<Io: AsyncRead + AsyncWrite + Unpin> AsyncWrite for SessionCapped<Io> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().io).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_shutdown(cx)
    }
}

/// A TLS connection's stream, ready to be handed to hyper/axum like any other `AsyncRead +
/// AsyncWrite` transport. Wraps the raw `TlsStream<TcpStream>` together with its (optional,
/// post-handshake) connection-cap permit -- see `SessionCapped`.
pub struct TlsIo(SessionCapped<TlsStream<TcpStream>>);

impl AsyncRead for TlsIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_shutdown(cx)
    }
}

type SocketAddr = std::net::SocketAddr;

/// The background loop `TlsListener` spawns (see its docs, FU1/C1/I1/I3): accept raw TCP off
/// `inner` as fast as it allows, bound concurrent in-flight handshakes with `max_handshakes`
/// (C1), and hand each admitted connection's handshake to its OWN spawned task -- under
/// `handshake_timeout` -- so a stalled handshake can never hold up the next accept. Successful
/// handshakes acquire the (optional) connection-cap permit from `conn_cap` -- AFTER the handshake
/// (I1) -- then are sent down `tx`; failures/timeouts/shed connections are logged and dropped.
/// Runs until its `JoinHandle` is aborted (`TlsListener::drop`) or it panics.
///
/// `fault_inject`, while `> 0`, makes each raw-accept iteration fail with a synthetic error
/// (decrementing by one) instead of calling the real `accept()` -- a test-only seam (I3's
/// `accept_error_does_not_deafen_listener`, M1's `accept_error_escalates_after_threshold`) for
/// exercising the warn+backoff+continue (and error!-escalation) path with a real, reproducible
/// error. Production call sites (`TlsListener::new`) pass a counter that is never incremented.
async fn accept_loop(
    inner: TcpListener,
    reloadable: Arc<ReloadableTls>,
    handshake_timeout: Duration,
    max_handshakes: usize,
    conn_cap: Option<Arc<Semaphore>>,
    tx: mpsc::Sender<(TlsIo, SocketAddr)>,
    fault_inject: Arc<AtomicUsize>,
) {
    // M1: a persistent accept() error must escalate beyond an unbounded warn! flood so it's
    // distinguishable from transient churn -- reset to 0 on every successful accept; a single
    // error! fires only at the exact moment the count CROSSES the threshold (not on every
    // iteration past it), so a stuck listener produces one clear signal, not a second flood.
    const ACCEPT_ERROR_ESCALATION_THRESHOLD: u32 = 10;
    let mut consecutive_accept_failures: u32 = 0;
    let handshake_sem = Arc::new(Semaphore::new(max_handshakes.max(1)));
    loop {
        // I3: a real `io::Result` from the raw listener (unlike `axum::serve::Listener`'s own
        // impl for `TcpListener`, which retries transient errors internally and never surfaces
        // them) so a persistent failure (e.g. EMFILE under FD pressure) is observable instead of
        // silently vanishing. `fault_inject` lets a test force exactly this branch deterministically.
        let accept_result = if fault_inject.load(Ordering::SeqCst) > 0 {
            fault_inject.fetch_sub(1, Ordering::SeqCst);
            Err(io::Error::new(
                io::ErrorKind::Other,
                "injected test failure (I3/M1 regression test)",
            ))
        } else {
            inner.accept().await
        };
        let (tcp, addr) = match accept_result {
            Ok(pair) => {
                consecutive_accept_failures = 0;
                pair
            }
            Err(e) => {
                consecutive_accept_failures += 1;
                tracing::warn!(
                    "TLS listener accept() failed ({e}); backing off briefly and retrying (a \
                     transient error must never permanently deafen the listener) \
                     [{consecutive_accept_failures} consecutive failure(s)]"
                );
                if consecutive_accept_failures == ACCEPT_ERROR_ESCALATION_THRESHOLD {
                    tracing::error!(
                        "TLS listener accept() has failed {ACCEPT_ERROR_ESCALATION_THRESHOLD} \
                         times in a row; still retrying (never deafening), but this likely \
                         indicates a persistent problem (e.g. FD exhaustion) that needs operator \
                         attention"
                    );
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };

        // C1: pre-auth handshake bound, independent of (and applied regardless of) the
        // connection cap -- protects the default, uncapped TLS deployment from an unbounded
        // pre-auth TCP flood. `try_acquire` (never `.await`) so an exhausted bound SHEDS this
        // connection immediately instead of queuing -- queuing here would re-create FU1's
        // head-of-line blocking at this new semaphore.
        let handshake_permit = match handshake_sem.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                tracing::warn!(
                    "TLS pre-auth handshake limit ({max_handshakes}) reached, shedding a new \
                     connection from {addr} before its handshake"
                );
                continue; // `tcp` drops here, closing the raw socket.
            }
        };

        let acceptor = TlsAcceptor::from(reloadable.current());
        let tx = tx.clone();
        let conn_cap = conn_cap.clone();
        tokio::spawn(async move {
            // Held for this whole task; released on EVERY exit path below (success, TLS error,
            // timeout) purely by falling out of scope -- bounds in-flight handshakes only.
            let _handshake_permit = handshake_permit;
            match tokio::time::timeout(handshake_timeout, acceptor.accept(tcp)).await {
                Ok(Ok(tls)) => {
                    // I1/N1: the connection-cap permit (when the cap is configured) is acquired
                    // ONLY now, after a successful handshake -- see module docs for why this
                    // deliberately diverges from the plaintext `conn_cap::CappedListener` (which
                    // caps at raw accept). `try_acquire_owned` (NON-blocking, N1 fix): an earlier
                    // version `.await`ed this acquire while still holding `_handshake_permit`,
                    // which re-coupled the two bounds in REVERSE -- with long-lived sessions
                    // (normal for streamable-http/SSE), enough post-handshake connections parked
                    // on a full cap would consume every pre-auth handshake permit too, starving
                    // new clients at the pre-auth gate. Trying instead of awaiting means: a cap
                    // permit is either free right now (take it) or not (SHED -- close this
                    // now-authenticated connection immediately rather than queue it holding a
                    // handshake permit). Over TLS, a full connection cap therefore SHEDS newly
                    // handshaken connections instead of queuing them (unlike the plaintext
                    // `CappedListener`, which queues in the bounded kernel backlog) -- the
                    // trade-off that keeps both bounds independent in both directions and total
                    // FDs bounded by `max_handshakes + max_connections`.
                    let permit = match &conn_cap {
                        Some(sem) => match sem.clone().try_acquire_owned() {
                            Ok(p) => Some(p),
                            Err(_) => {
                                tracing::warn!(
                                    "TLS connection cap reached; shedding the now-handshaken \
                                     connection from {addr} instead of queuing it"
                                );
                                return; // `tls` (and `_handshake_permit`) drop here.
                            }
                        },
                        None => None,
                    };
                    let io = TlsIo(SessionCapped {
                        io: tls,
                        _permit: permit,
                    });
                    // The receiver only goes away once the whole `TlsListener` is dropped (serve
                    // loop shutting down) -- a failed send here just lost the race against
                    // shutdown; nothing to log or recover. The permit (if any) is dropped with
                    // `io` on this path too, so no leak.
                    let _ = tx.send((io, addr)).await;
                }
                Ok(Err(e)) => {
                    tracing::warn!("TLS handshake with {addr} failed, dropping connection: {e}");
                }
                Err(_) => {
                    tracing::warn!(
                        "TLS handshake with {addr} exceeded the {handshake_timeout:?} handshake \
                         timeout, dropping connection"
                    );
                }
            }
        });
    }
}

/// An `axum::serve::Listener` that terminates TLS WITHOUT letting a stalled handshake block new
/// acceptance (FU1, anti head-of-line DoS), while bounding pre-auth handshake concurrency (C1)
/// and applying the connection cap to established sessions only (I1). Construction spawns a
/// background task (`accept_loop`) that owns the raw `TcpListener` and performs each connection's
/// TLS handshake in its own further-spawned task under a timeout; only a completed, admitted
/// handshake is handed to `accept()` over an internal channel. A `reload()` on the `ReloadableTls`
/// applies to the NEXT handshake the background task starts, without dropping the listener or any
/// connection already past its handshake. Dropping `TlsListener` aborts the background task.
pub struct TlsListener {
    rx: mpsc::Receiver<(TlsIo, SocketAddr)>,
    accept_task: tokio::task::JoinHandle<()>,
    local_addr: SocketAddr,
}

impl Drop for TlsListener {
    fn drop(&mut self) {
        // Stop the background accept loop. Any handshake tasks it already spawned keep running
        // independently to completion/timeout -- they only hold a channel `Sender`, so the
        // `Receiver` disappearing just makes their eventual `send` a harmless no-op (and drops
        // their handshake/conn-cap permits normally when the task ends).
        self.accept_task.abort();
    }
}

impl TlsListener {
    /// Wrap a bound `TcpListener`, terminating TLS off the accept path (see struct docs).
    /// `handshake_timeout` bounds every individual handshake attempt (FU1); pass
    /// `handshake_timeout_from_env()` for the standard env-overridable default. `max_handshakes`
    /// bounds concurrent in-flight handshakes (C1); pass `max_handshakes_from_env()`.
    /// `max_connections`, when `Some`, caps concurrent ESTABLISHED sessions (I1) -- a connection
    /// that finishes its handshake while the cap is full is SHED (closed), not queued (N1);
    /// `None` means no cap, unchanged from before FU2.
    pub fn new(
        inner: TcpListener,
        reloadable: Arc<ReloadableTls>,
        handshake_timeout: Duration,
        max_handshakes: usize,
        max_connections: Option<usize>,
    ) -> Self {
        let local_addr = inner
            .local_addr()
            .expect("the bound listener handed to TlsListener::new always has a local addr");
        let (tx, rx) = mpsc::channel(HANDSHAKE_CHANNEL_CAPACITY);
        let conn_cap = max_connections.map(|n| Arc::new(Semaphore::new(n)));
        let accept_task = tokio::spawn(accept_loop(
            inner,
            reloadable,
            handshake_timeout,
            max_handshakes,
            conn_cap,
            tx,
            Arc::new(AtomicUsize::new(0)), // production: fault injection never fires.
        ));
        Self {
            rx,
            accept_task,
            local_addr,
        }
    }
}

impl Listener for TlsListener {
    type Io = TlsIo;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.rx.recv().await {
                Some(pair) => return pair,
                // I3: the accept-loop task ended without us aborting it (it never returns on its
                // own outside a panic -- see its docs, transient accept() errors now retry
                // in-place instead of ending the loop). Nothing left to accept from -- park
                // forever rather than spin, but LOUDLY: this is the "listener gone" condition, so
                // log it at error! instead of the previous silent park.
                None => {
                    tracing::error!(
                        "TLS accept-loop background task ended unexpectedly (this should only \
                         happen after a panic); no new TLS connections can be accepted"
                    );
                    std::future::pending::<()>().await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local_addr)
    }
}

/// Serve `app` over HTTPS on `listener`. TLS handshakes run off the accept path under
/// `handshake_timeout` (FU1), bounded to `max_handshakes` concurrent in-flight attempts (C1,
/// independent of and applied regardless of `max_connections`). When `max_connections` is `Some`,
/// that many ESTABLISHED (post-handshake) sessions may be open at once (I1); a connection that
/// finishes its handshake while the cap is already full is SHED (closed), not queued (N1), so cap
/// exhaustion can never starve the pre-auth handshake bound in reverse. `None` means no cap,
/// unchanged. Graceful shutdown mirrors the plaintext path: draining stops when `cancel` fires.
pub async fn serve_tls(
    listener: TcpListener,
    app: axum::Router,
    reloadable: Arc<ReloadableTls>,
    cancel: tokio_util::sync::CancellationToken,
    handshake_timeout: Duration,
    max_handshakes: usize,
    max_connections: Option<usize>,
) -> Result<()> {
    axum::serve(
        TlsListener::new(
            listener,
            reloadable,
            handshake_timeout,
            max_handshakes,
            max_connections,
        ),
        app,
    )
    .with_graceful_shutdown(async move { cancel.cancelled().await })
    .await
    .context("HTTPS serve loop")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{
        BasicConstraints, CertificateParams, DistinguishedName, DnType, IsCa, KeyPair, SanType,
    };
    use std::net::SocketAddr;
    use tokio::net::TcpStream;

    /// A self-signed CA cert/key pair, PEM-encoded.
    struct TestCa {
        cert_pem: String,
        cert: rcgen::Certificate,
        key: KeyPair,
    }

    fn make_ca() -> TestCa {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "glossa-test-ca");
        params.distinguished_name = dn;
        let cert = params.self_signed(&key).unwrap();
        let cert_pem = cert.pem();
        TestCa {
            cert_pem,
            cert,
            key,
        }
    }

    /// A leaf cert for `127.0.0.1`, signed by `ca` (server or client role -- same shape, callers
    /// set `eku` appropriately via `server`).
    fn make_leaf(ca: &TestCa, common_name: &str) -> (String, String) {
        let key = KeyPair::generate().unwrap();
        let mut params =
            CertificateParams::new(vec!["127.0.0.1".to_string(), "localhost".to_string()]).unwrap();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, common_name);
        params.distinguished_name = dn;
        params
            .subject_alt_names
            .push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
        let cert = params.signed_by(&key, &ca.cert, &ca.key).unwrap();
        (cert.pem(), key.serialize_pem())
    }

    /// Writes `contents` to `path`, creating parent dirs as needed.
    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    fn client_root_store(ca_pem: &str) -> RootCertStore {
        let mut roots = RootCertStore::empty();
        let mut rd = BufReader::new(ca_pem.as_bytes());
        for cert in rustls_pemfile::certs(&mut rd) {
            roots.add(cert.unwrap()).unwrap();
        }
        roots
    }

    #[test]
    fn build_server_config_loads_cert_and_key() {
        let dir = tempfile::tempdir().unwrap();
        let ca = make_ca();
        let (leaf_pem, key_pem) = make_leaf(&ca, "glossa-test-server");
        let cert_path = dir.path().join("server.pem");
        let key_path = dir.path().join("server.key");
        write(&cert_path, &leaf_pem);
        write(&key_path, &key_pem);
        let files = TlsFiles {
            cert: cert_path,
            key: key_path,
            client_ca: None,
        };
        build_server_config(&files).expect("valid cert/key builds a ServerConfig");
    }

    #[test]
    fn build_server_config_with_client_ca_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let ca = make_ca();
        let (leaf_pem, key_pem) = make_leaf(&ca, "glossa-test-server");
        let cert_path = dir.path().join("server.pem");
        let key_path = dir.path().join("server.key");
        let ca_path = dir.path().join("ca.pem");
        write(&cert_path, &leaf_pem);
        write(&key_path, &key_pem);
        write(&ca_path, &ca.cert_pem);
        let files = TlsFiles {
            cert: cert_path,
            key: key_path,
            client_ca: Some(ca_path),
        };
        // Full mTLS behavior (require + verify) is exercised end-to-end by the handshake tests
        // below; this asserts the config itself builds without error.
        build_server_config(&files).expect("mTLS config builds with a client CA installed");
    }

    #[test]
    fn build_server_config_rejects_missing_cert_file() {
        let dir = tempfile::tempdir().unwrap();
        let files = TlsFiles {
            cert: dir.path().join("does-not-exist.pem"),
            key: dir.path().join("does-not-exist.key"),
            client_ca: None,
        };
        assert!(build_server_config(&files).is_err());
    }

    /// Bind an ephemeral port, serve `serve_tls` in the background, and return (addr, ca_pem,
    /// cert_path, key_path, reloadable, cancel) so callers can drive handshakes and reloads.
    /// `handshake_timeout`/`max_handshakes`/`max_connections` are passed straight through to
    /// `serve_tls` so FU1/FU2/C1/I1 tests can exercise a short timeout / a small bound / a small
    /// cap without touching the other tests' shape.
    async fn spawn_test_server(
        client_ca_pem: Option<&str>,
        handshake_timeout: Duration,
        max_handshakes: usize,
        max_connections: Option<usize>,
    ) -> (
        SocketAddr,
        TestCa,
        PathBuf,
        PathBuf,
        Arc<ReloadableTls>,
        tokio_util::sync::CancellationToken,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let ca = make_ca();
        let (leaf_pem, key_pem) = make_leaf(&ca, "glossa-test-server-1");
        let cert_path = dir.path().join("server.pem");
        let key_path = dir.path().join("server.key");
        write(&cert_path, &leaf_pem);
        write(&key_path, &key_pem);
        let client_ca_path = client_ca_pem.map(|pem| {
            let p = dir.path().join("client-ca.pem");
            write(&p, pem);
            p
        });
        let files = TlsFiles {
            cert: cert_path.clone(),
            key: key_path.clone(),
            client_ca: client_ca_path,
        };
        let reloadable = Arc::new(ReloadableTls::new(files).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        let app = axum::Router::new().route("/ping", axum::routing::get(|| async { "pong" }));
        let serve_reloadable = reloadable.clone();
        let serve_cancel = cancel.clone();
        tokio::spawn(async move {
            let _ = serve_tls(
                listener,
                app,
                serve_reloadable,
                serve_cancel,
                handshake_timeout,
                max_handshakes,
                max_connections,
            )
            .await;
        });
        (addr, ca, cert_path, key_path, reloadable, cancel, dir)
    }

    fn client_config(roots: RootCertStore) -> Arc<rustls::ClientConfig> {
        Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    }

    fn client_config_with_cert(
        roots: RootCertStore,
        cert_pem: &str,
        key_pem: &str,
    ) -> Arc<rustls::ClientConfig> {
        let mut rd = BufReader::new(cert_pem.as_bytes());
        let certs: Vec<_> = rustls_pemfile::certs(&mut rd)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut kd = BufReader::new(key_pem.as_bytes());
        let key = rustls_pemfile::private_key(&mut kd).unwrap().unwrap();
        Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_client_auth_cert(certs, key)
                .unwrap(),
        )
    }

    async fn https_get(addr: SocketAddr, client_cfg: Arc<rustls::ClientConfig>) -> Result<String> {
        let connector = tokio_rustls::TlsConnector::from(client_cfg);
        let tcp = TcpStream::connect(addr).await.context("tcp connect")?;
        let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let mut tls = connector
            .connect(server_name, tcp)
            .await
            .context("tls handshake")?;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        tls.write_all(b"GET /ping HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .await
            .context("write request")?;
        let mut resp = Vec::new();
        tls.read_to_end(&mut resp).await.context("read response")?;
        Ok(String::from_utf8_lossy(&resp).to_string())
    }

    #[tokio::test]
    async fn tls_handshake_succeeds_and_serves_the_app() {
        let (addr, ca, _cert, _key, _reloadable, cancel, _dir) =
            spawn_test_server(None, Duration::from_secs(5), DEFAULT_MAX_HANDSHAKES, None).await;
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let resp = https_get(addr, cfg)
            .await
            .expect("TLS handshake + HTTP roundtrip succeeds");
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn plaintext_client_is_rejected_by_the_tls_port() {
        let (addr, _ca, _cert, _key, _reloadable, cancel, _dir) =
            spawn_test_server(None, Duration::from_secs(5), DEFAULT_MAX_HANDSHAKES, None).await;
        let mut tcp = TcpStream::connect(addr).await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // A plaintext HTTP request sent straight at the TLS port is not a valid TLS ClientHello --
        // the server's handshake fails and it closes/resets the connection instead of ever
        // reaching the app.
        let _ = tcp
            .write_all(b"GET /ping HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .await;
        let mut buf = [0u8; 64];
        let read = tokio::time::timeout(std::time::Duration::from_secs(2), tcp.read(&mut buf))
            .await
            .expect("server responds (by closing) within the timeout, not hanging");
        // The connection must be rejected: either a clean EOF (Ok(0)), a reset (Err), or rustls
        // sending its own short fatal TLS alert record (a handful of raw bytes -- content-type +
        // version + length + a 2-byte alert body, NOT a parseable HTTP response) before closing.
        // What must NOT happen is the plaintext request being answered as if the app saw it.
        match read {
            Ok(0) => {}
            Ok(n) => {
                let body = String::from_utf8_lossy(&buf[..n]);
                assert!(
                    !body.starts_with("HTTP/"),
                    "a plaintext client must never get a real HTTP response back, got: {body:?}"
                );
            }
            Err(_) => {}
        }
        cancel.cancel();
    }

    #[tokio::test]
    async fn mtls_rejects_missing_client_cert_and_accepts_valid_one() {
        let ca = make_ca();
        let (addr, server_ca, _cert, _key, _reloadable, cancel, _dir) = {
            let dir = tempfile::tempdir().unwrap();
            let (leaf_pem, key_pem) = make_leaf(&ca, "glossa-test-mtls-server");
            let cert_path = dir.path().join("server.pem");
            let key_path = dir.path().join("server.key");
            let client_ca_path = dir.path().join("client-ca.pem");
            write(&cert_path, &leaf_pem);
            write(&key_path, &key_pem);
            write(&client_ca_path, &ca.cert_pem);
            let files = TlsFiles {
                cert: cert_path.clone(),
                key: key_path.clone(),
                client_ca: Some(client_ca_path),
            };
            let reloadable = Arc::new(ReloadableTls::new(files).unwrap());
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let cancel = tokio_util::sync::CancellationToken::new();
            let app = axum::Router::new().route("/ping", axum::routing::get(|| async { "pong" }));
            let serve_reloadable = reloadable.clone();
            let serve_cancel = cancel.clone();
            tokio::spawn(async move {
                let _ = serve_tls(
                    listener,
                    app,
                    serve_reloadable,
                    serve_cancel,
                    Duration::from_secs(5),
                    DEFAULT_MAX_HANDSHAKES,
                    None,
                )
                .await;
            });
            (addr, ca, cert_path, key_path, reloadable, cancel, dir)
        };

        // No client cert: the server requires one (mTLS) -> handshake must fail.
        let no_cert_cfg = client_config(client_root_store(&server_ca.cert_pem));
        let no_cert_result = https_get(addr, no_cert_cfg).await;
        assert!(
            no_cert_result.is_err(),
            "a client with no cert must be rejected under mTLS, got: {no_cert_result:?}"
        );

        // Valid client cert signed by the trusted client-CA -> handshake must succeed.
        let (client_cert_pem, client_key_pem) = make_leaf(&server_ca, "glossa-test-client");
        let with_cert_cfg = client_config_with_cert(
            client_root_store(&server_ca.cert_pem),
            &client_cert_pem,
            &client_key_pem,
        );
        let resp = https_get(addr, with_cert_cfg)
            .await
            .expect("a valid client cert signed by the trusted CA is accepted");
        assert!(resp.contains("pong"));
        cancel.cancel();
    }

    #[tokio::test]
    async fn reload_serves_the_new_cert_without_dropping_the_listener() {
        let (addr, ca, cert_path, key_path, reloadable, cancel, _dir) =
            spawn_test_server(None, Duration::from_secs(5), DEFAULT_MAX_HANDSHAKES, None).await;

        // First connection: served by the original cert (CN "glossa-test-server-1").
        let cfg = client_config(client_root_store(&ca.cert_pem));
        https_get(addr, cfg.clone())
            .await
            .expect("first connection succeeds against the original cert");

        // Rotate the cert file in place (same path) and reload.
        let (new_leaf_pem, new_key_pem) = make_leaf(&ca, "glossa-test-server-2");
        write(&cert_path, &new_leaf_pem);
        write(&key_path, &new_key_pem);
        reloadable
            .reload()
            .expect("reload picks up the rotated cert/key");

        // A fresh connection after reload must still succeed -- the listener kept running and
        // the new cert (still signed by the same trusted CA) is accepted.
        let resp = https_get(addr, cfg)
            .await
            .expect("a NEW connection after reload succeeds without restarting the listener");
        assert!(resp.contains("pong"));
        cancel.cancel();
    }

    #[tokio::test]
    async fn bearer_token_middleware_still_enforced_over_tls() {
        // `bearer_auth_layer` itself lives in the `kb` binary crate (main.rs), not this library
        // module, so it can't be linked into a lib-crate test directly. This proves the thing
        // task 8 is actually responsible for: an axum auth middleware in front of the app is
        // still evaluated normally when the app is served over `serve_tls` -- TLS termination
        // does not bypass or short-circuit router-level middleware.
        let dir = tempfile::tempdir().unwrap();
        let ca = make_ca();
        let (leaf_pem, key_pem) = make_leaf(&ca, "glossa-test-auth-server");
        let cert_path = dir.path().join("server.pem");
        let key_path = dir.path().join("server.key");
        write(&cert_path, &leaf_pem);
        write(&key_path, &key_pem);
        let files = TlsFiles {
            cert: cert_path,
            key: key_path,
            client_ca: None,
        };
        let reloadable = Arc::new(ReloadableTls::new(files).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        const TOKEN: &str = "secret-token";
        async fn require_bearer(
            req: axum::extract::Request,
            next: axum::middleware::Next,
        ) -> axum::response::Response {
            use axum::http::{header, StatusCode};
            let ok = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                == Some("Bearer secret-token");
            if ok {
                next.run(req).await
            } else {
                StatusCode::UNAUTHORIZED.into_response()
            }
        }
        use axum::response::IntoResponse;
        let app = axum::Router::new()
            .route("/mcp", axum::routing::get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(require_bearer));
        let serve_cancel = cancel.clone();
        tokio::spawn(async move {
            let _ = serve_tls(
                listener,
                app,
                reloadable,
                serve_cancel,
                Duration::from_secs(5),
                DEFAULT_MAX_HANDSHAKES,
                None,
            )
            .await;
        });

        let cfg = client_config(client_root_store(&ca.cert_pem));
        let connector = tokio_rustls::TlsConnector::from(cfg);

        // Without the token: 401.
        let tcp = TcpStream::connect(addr).await.unwrap();
        let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let mut tls = connector.connect(server_name, tcp).await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        tls.write_all(b"GET /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut resp = Vec::new();
        tls.read_to_end(&mut resp).await.unwrap();
        let resp = String::from_utf8_lossy(&resp);
        assert!(
            resp.starts_with("HTTP/1.1 401"),
            "no bearer token -> 401, got: {resp}"
        );

        // With the token: 200.
        let tcp = TcpStream::connect(addr).await.unwrap();
        let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let mut tls = connector.connect(server_name, tcp).await.unwrap();
        tls.write_all(
            format!(
                "GET /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {TOKEN}\r\nConnection: close\r\n\r\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
        let mut resp = Vec::new();
        tls.read_to_end(&mut resp).await.unwrap();
        let resp = String::from_utf8_lossy(&resp);
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "valid bearer token -> 200, got: {resp}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn stalled_handshake_does_not_block_a_second_connections_handshake() {
        // FU1 regression: the OLD `TlsListener::accept()` awaited `acceptor.accept(tcp)` INLINE,
        // so a client that opened TCP and never completed its TLS ClientHello would block ALL new
        // connections behind it (pre-auth head-of-line DoS). The fix moves each handshake off the
        // accept path into its own spawned task, so a stalled handshake only ties up that task.
        // A generous handshake_timeout keeps the stalled connection's own timeout from firing
        // during this test -- what's under test is concurrency, not the timeout knob (see the
        // dedicated timeout test below).
        let (addr, ca, _cert, _key, _reloadable, cancel, _dir) =
            spawn_test_server(None, Duration::from_secs(30), DEFAULT_MAX_HANDSHAKES, None).await;

        // First client: completes the TCP handshake, then sends nothing -- its TLS handshake
        // never progresses. Held for the rest of the test so it stays "stalled".
        let _stalled = TcpStream::connect(addr).await.unwrap();

        // Second client: a normal, well-behaved TLS handshake. With the old inline design this
        // would hang forever behind the stalled connection above; with the fix it completes
        // promptly regardless.
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let resp = tokio::time::timeout(Duration::from_secs(5), https_get(addr, cfg))
            .await
            .expect("a well-behaved second connection must not be blocked by a stalled handshake")
            .expect("TLS handshake + HTTP roundtrip succeeds");
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn stalled_handshake_times_out_and_is_dropped() {
        // FU1's timeout knob: a handshake that never progresses must not be held open forever --
        // it gets dropped once `handshake_timeout` elapses, freeing whatever it held (e.g. a
        // connection-cap permit, FU2).
        let short_timeout = Duration::from_millis(300);
        let (addr, _ca, _cert, _key, _reloadable, cancel, _dir) =
            spawn_test_server(None, short_timeout, DEFAULT_MAX_HANDSHAKES, None).await;

        let mut stalled = TcpStream::connect(addr).await.unwrap();
        use tokio::io::AsyncReadExt;
        // The server never receives a ClientHello, so its handshake stalls; once the timeout
        // fires the background task drops the raw TCP connection. The client must observe that
        // close well within a small multiple of `short_timeout`, not hang indefinitely.
        let mut buf = [0u8; 16];
        let read = tokio::time::timeout(Duration::from_secs(3), stalled.read(&mut buf)).await;
        match read {
            Ok(Ok(0)) => {}  // clean EOF: server closed its end after the timeout fired
            Ok(Err(_)) => {} // reset
            other => {
                panic!("a stalled handshake must be dropped after its timeout, got: {other:?}")
            }
        }
        cancel.cancel();
    }

    #[tokio::test]
    async fn connection_cap_applies_over_tls_when_set() {
        // FU2 regression: `GLOSSA_MCP_MAX_CONNECTIONS` used to be silently bypassed by
        // `serve_tls` (it returned before the plaintext path's `CappedListener` ever applied).
        // `accept_loop` now acquires the SAME kind of cap permit after a successful handshake
        // (I1: post-handshake, not at raw accept -- see `conn_cap_not_consumed_by_stalled_preauth`
        // for the regression guard on THAT ordering), so the cap still holds here. The acquire is
        // non-blocking (N1: `try_acquire_owned`, not an awaited queue -- see
        // `cap_full_sheds_without_consuming_handshake_permits` for that regression guard), so a
        // full cap SHEDS the second connection promptly rather than hanging it.
        let (addr, ca, _cert, _key, _reloadable, cancel, _dir) = spawn_test_server(
            None,
            Duration::from_secs(5),
            DEFAULT_MAX_HANDSHAKES,
            Some(1),
        )
        .await;
        let cfg = client_config(client_root_store(&ca.cert_pem));

        // First connection: admitted immediately (cap = 1, nothing else open yet), and kept OPEN
        // -- holding the cap's one permit -- by never dropping the client-side TLS stream.
        let connector = tokio_rustls::TlsConnector::from(cfg.clone());
        let tcp = TcpStream::connect(addr).await.unwrap();
        let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let first = connector
            .connect(server_name, tcp)
            .await
            .expect("first connection is admitted (under the cap)");

        // Second connection: its OWN TLS handshake can complete (the cap no longer gates the raw
        // accept/handshake, only the post-handshake handoff -- see I1), but with the cap at 1 and
        // the first connection still open, the server-side task's non-blocking cap-acquire fails
        // and it SHEDS the connection immediately (N1) -- so the full request/response round trip
        // must fail FAST (well within the timeout), not hang until the timeout elapses.
        let shed = tokio::time::timeout(Duration::from_millis(800), https_get(addr, cfg.clone()))
            .await
            .expect("a shed connection fails fast, well within the timeout -- it must not hang");
        assert!(
            shed.is_err(),
            "a second connection must be shed (rejected) by the connection cap while the first \
             is open, got: {shed:?}"
        );

        // Freeing the first connection's permit lets a fresh second connection through.
        drop(first);
        let resp = tokio::time::timeout(Duration::from_secs(5), https_get(addr, cfg))
            .await
            .expect("a new connection proceeds once the cap's permit is released")
            .expect("TLS handshake + HTTP roundtrip succeeds");
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn preauth_semaphore_bounds_concurrent_handshakes() {
        // C1 regression: `accept_loop` used to spawn one handshake task per raw TCP accept with
        // NO bound of its own -- unbounded when the connection cap is unset (the default TLS
        // deployment), letting a pre-auth flood exhaust FDs/memory. A dedicated pre-auth
        // handshake semaphore now bounds concurrent in-flight handshakes independent of the
        // (here, unset) connection cap.
        //
        // A short handshake_timeout (rather than a long one) lets this test ALSO prove the bound
        // self-clears: once the first stalled handshake's timeout fires and releases its permit,
        // a legitimate client is admitted.
        let short_timeout = Duration::from_millis(400);
        let (addr, ca, _cert, _key, _reloadable, cancel, _dir) =
            spawn_test_server(None, short_timeout, 1, None).await;

        // First connection: opens TCP, sends nothing -- consumes the sole pre-auth permit for the
        // whole handshake_timeout window.
        let _stalled1 = TcpStream::connect(addr).await.unwrap();
        // Give the accept loop a moment to accept it and acquire the permit.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Second connection: also never sends a ClientHello. With the pre-auth bound exhausted,
        // `accept_loop` must SHED it immediately (try_acquire, not queue) -- the client observes
        // a close well within `short_timeout`, proving it was shed rather than queued behind the
        // first connection's own timeout.
        let mut stalled2 = TcpStream::connect(addr).await.unwrap();
        use tokio::io::AsyncReadExt;
        let read2 =
            tokio::time::timeout(Duration::from_millis(250), stalled2.read(&mut [0u8; 16])).await;
        match read2 {
            Ok(Ok(0)) => {}  // shed: clean EOF
            Ok(Err(_)) => {} // shed: reset
            other => panic!(
                "a second connection must be shed promptly when the pre-auth handshake limit is \
                 reached (not queued behind the first), got: {other:?}"
            ),
        }
        // The accept loop itself was never blocked by either connection above (both were handled
        // without an `.await` on the pre-auth semaphore) -- once the first stalled handshake's
        // timeout releases its permit, a legitimate client is admitted for a real handshake.
        // Retried (rather than a single attempt) since exactly when `short_timeout` finishes
        // elapsing relative to this test's own progress isn't precisely deterministic -- an
        // attempt that lands in the remaining pre-release window would itself be shed, which is
        // correct/expected admission-control behavior, not a bug; what's under test is that
        // admission resumes soon after release, not the exact millisecond it does.
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        let resp = loop {
            match tokio::time::timeout(Duration::from_millis(500), https_get(addr, cfg.clone()))
                .await
            {
                Ok(Ok(resp)) => break resp,
                _ if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Ok(Err(e)) => panic!(
                    "a new handshake must be admitted once the pre-auth permit is released, \
                     got: {e}"
                ),
                Err(_) => panic!(
                    "a new handshake was never admitted within the deadline after the pre-auth \
                     permit should have been released"
                ),
            }
        };
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn conn_cap_not_consumed_by_stalled_preauth() {
        // I1 regression: the connection-cap permit must be acquired AFTER a successful handshake,
        // not before/at raw accept -- otherwise a stalled pre-auth handshake would occupy the
        // sole connection-cap permit for its whole handshake_timeout window and starve every
        // legitimate client. A generous handshake_timeout and pre-auth bound keep THOSE
        // mechanisms from interfering -- only the conn-cap ordering is under test here.
        let (addr, ca, _cert, _key, _reloadable, cancel, _dir) = spawn_test_server(
            None,
            Duration::from_secs(30),
            DEFAULT_MAX_HANDSHAKES,
            Some(1),
        )
        .await;

        // A stalled pre-auth connection: opens TCP, never sends a ClientHello. Under the OLD
        // (broken) design this would hold the connection cap's one permit for the whole
        // handshake_timeout window.
        let _stalled = TcpStream::connect(addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        // A legitimate second client must still complete its handshake AND obtain the sole
        // connection-cap permit (i.e. get a real app response) promptly -- proving the stalled
        // connection above did not consume it.
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let resp = tokio::time::timeout(Duration::from_secs(5), https_get(addr, cfg))
            .await
            .expect(
                "a legitimate connection must not be starved out of the connection cap by a \
                 stalled pre-auth handshake",
            )
            .expect("TLS handshake + HTTP roundtrip succeeds");
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn cap_full_sheds_without_consuming_handshake_permits() {
        // N1 regression (the REVERSE of `conn_cap_not_consumed_by_stalled_preauth`, I1's guard):
        // round-1's I1 fix acquired the connection-cap permit with an AWAITED `acquire_owned()`
        // while the spawned handshake task still held its (by-then-pointless) pre-auth handshake
        // permit. With the cap full and a long-lived session holding it forever (the normal
        // streamable-http/SSE case), a connection that finishes its handshake would park on the
        // cap FOREVER, still holding its handshake permit -- eventually exhausting the whole
        // pre-auth handshake bound and starving brand-new clients at the pre-auth gate instead.
        // The fix (`try_acquire_owned`, shed on full) must ensure: (a) a connection that finishes
        // its handshake while the cap is already full is shed PROMPTLY, not parked; (b) doing so
        // releases its handshake permit immediately, so a brand-new connection's admission at the
        // pre-auth gate is never affected by how full the (independent) connection cap is.
        let max_handshakes = 1;
        let (addr, ca, _cert, _key, _reloadable, cancel, _dir) =
            spawn_test_server(None, Duration::from_secs(30), max_handshakes, Some(1)).await;
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let connector = tokio_rustls::TlsConnector::from(cfg);

        // Holder: a real, long-lived session that completes its handshake (releasing its OWN
        // handshake permit immediately afterward, like any successful handshake) and then holds
        // the sole connection-cap permit forever by simply never being dropped.
        let tcp = TcpStream::connect(addr).await.unwrap();
        let server_name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let _holder = connector
            .connect(server_name, tcp)
            .await
            .expect("the holder connection is admitted (the cap starts empty)");

        // Excess connection: its OWN handshake must still succeed -- pre-auth capacity is
        // available (the holder already released its handshake permit) -- even though the
        // connection cap is now full and this connection will be shed right after.
        let tcp2 = TcpStream::connect(addr).await.unwrap();
        let server_name2 = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        let mut excess = connector.connect(server_name2, tcp2).await.expect(
            "a connection's OWN handshake must succeed regardless of connection-cap fullness",
        );

        // (a) The excess connection must be SHED promptly once its post-handshake cap-acquire
        // fails -- not parked waiting for a slot that will never free.
        use tokio::io::AsyncReadExt;
        let mut buf = [0u8; 16];
        let shed = tokio::time::timeout(Duration::from_secs(2), excess.read(&mut buf)).await;
        match shed {
            Ok(Ok(0)) => {}  // clean EOF
            Ok(Err(_)) => {} // reset / unexpected-eof from the abrupt close
            other => panic!(
                "a connection must be SHED promptly when it finishes its handshake with the \
                 connection cap already full (not parked waiting for a slot), got: {other:?}"
            ),
        }

        // (b) A BRAND-NEW connection's handshake must still be admitted at the pre-auth gate --
        // proving the excess connection's shed released its handshake permit rather than leaking
        // it while parked. Against the pre-fix (awaited-acquire) design this step times out: the
        // excess connection's task would still be stuck awaiting the never-freed cap permit,
        // still holding the sole `max_handshakes = 1` permit, so this handshake would never even
        // start.
        let tcp3 = TcpStream::connect(addr).await.unwrap();
        let server_name3 = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            connector.connect(server_name3, tcp3),
        )
        .await
        .expect(
            "a brand-new connection's handshake must be admitted promptly -- connection-cap \
                 exhaustion must never consume pre-auth handshake capacity",
        )
        .expect("the fresh connection's TLS handshake itself succeeds");

        cancel.cancel();
    }

    /// Like `spawn_test_server`, but calls `accept_loop` directly (bypassing `TlsListener::new`,
    /// which never exposes fault injection) with `fault_count` forced synthetic accept()
    /// failures before real accepts resume -- the seam shared by the I3 and M1 regression tests.
    async fn spawn_test_server_with_fault(
        fault_count: usize,
    ) -> (
        SocketAddr,
        TestCa,
        tokio_util::sync::CancellationToken,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let ca = make_ca();
        let (leaf_pem, key_pem) = make_leaf(&ca, "glossa-test-accept-error-server");
        let cert_path = dir.path().join("server.pem");
        let key_path = dir.path().join("server.key");
        write(&cert_path, &leaf_pem);
        write(&key_path, &key_pem);
        let files = TlsFiles {
            cert: cert_path,
            key: key_path,
            client_ca: None,
        };
        let reloadable = Arc::new(ReloadableTls::new(files).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel(HANDSHAKE_CHANNEL_CAPACITY);
        let fault = Arc::new(AtomicUsize::new(fault_count));
        let accept_task = tokio::spawn(accept_loop(
            listener,
            reloadable,
            Duration::from_secs(5),
            DEFAULT_MAX_HANDSHAKES,
            None,
            tx,
            fault,
        ));
        let tls_listener = TlsListener {
            rx,
            accept_task,
            local_addr: addr,
        };
        let app = axum::Router::new().route("/ping", axum::routing::get(|| async { "pong" }));
        let cancel = tokio_util::sync::CancellationToken::new();
        let serve_cancel = cancel.clone();
        tokio::spawn(async move {
            let _ = axum::serve(tls_listener, app)
                .with_graceful_shutdown(async move { serve_cancel.cancelled().await })
                .await;
        });
        (addr, ca, cancel, dir)
    }

    #[tokio::test]
    async fn accept_error_does_not_deafen_listener() {
        // I3 regression: a transient accept() error (e.g. EMFILE under FD pressure) must not
        // permanently deafen the listener. Exactly ONE synthetic accept() failure is forced
        // before real accepts resume -- the cheapest reliable seam for a real, reproducible error
        // without depending on actually exhausting OS file descriptors.
        let (addr, ca, cancel, _dir) = spawn_test_server_with_fault(1).await;

        // A real client connection, made right after the server starts: it races the injected
        // failure. With the I3 fix (warn + short backoff + continue) the loop recovers in well
        // under a second and this connection still completes; with the old silent-`pending()`
        // bug it would hang forever.
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let resp = tokio::time::timeout(Duration::from_secs(3), https_get(addr, cfg))
            .await
            .expect(
                "the listener must keep accepting after a transient accept() error, not deafen \
                 permanently",
            )
            .expect("TLS handshake + HTTP roundtrip succeeds");
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );
        cancel.cancel();
    }

    /// A `tracing::Subscriber` writer that captures formatted log lines into a shared buffer, so a
    /// test can assert an `ERROR`-level event fired without depending on message text matching.
    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn accept_error_escalates_after_threshold() {
        // M1 regression: a persistent (not just transient) run of accept() errors must escalate
        // beyond the per-failure warn! (which would otherwise flood forever with no signal that
        // this is different from ordinary transient churn). A thread-local subscriber captures
        // formatted log output for the duration of this test (`#[tokio::test]` defaults to a
        // current-thread runtime, so the accept-loop task spawned below runs on this same
        // thread -- the capture applies to it too) so the escalation can be asserted directly,
        // without depending on exact message text.
        let captured = CapturedLogs::default();
        let make_writer = {
            let captured = captured.clone();
            move || captured.clone()
        };
        let subscriber = tracing_subscriber::fmt()
            .with_writer(make_writer)
            .with_max_level(tracing::Level::ERROR)
            .without_time()
            .with_target(false)
            .finish();
        let _log_guard = tracing::subscriber::set_default(subscriber);

        // Exactly the escalation threshold's worth of consecutive synthetic accept() failures,
        // then real accepts resume.
        let (addr, ca, cancel, _dir) = spawn_test_server_with_fault(10).await;

        // The loop must still be accepting afterward -- never-deafen holds regardless of how many
        // consecutive failures preceded a real accept. Each injected failure costs a 100ms
        // backoff, so ~1s of delay is expected before the real accept resumes; the timeout below
        // gives a generous margin over that.
        let cfg = client_config(client_root_store(&ca.cert_pem));
        let resp = tokio::time::timeout(Duration::from_secs(5), https_get(addr, cfg))
            .await
            .expect(
                "the listener must keep accepting after 10 consecutive accept() errors, not \
                 deafen",
            )
            .expect("TLS handshake + HTTP roundtrip succeeds");
        assert!(
            resp.contains("pong"),
            "response should reach the app: {resp}"
        );

        // An ERROR-level event must have fired exactly at the threshold crossing.
        drop(_log_guard);
        let logs = String::from_utf8_lossy(&captured.0.lock().unwrap()).into_owned();
        assert!(
            !logs.is_empty(),
            "10 consecutive accept() failures must escalate to an ERROR-level log, got no \
             captured ERROR output"
        );
        cancel.cancel();
    }
}
