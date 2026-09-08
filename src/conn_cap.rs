//! Optional listener-level concurrent-connection cap (§3c, anti-slowloris). Off unless
//! `GLOSSA_MCP_MAX_CONNECTIONS` is set. Wraps a listener; each accepted connection holds a
//! semaphore permit for its lifetime, so the (permit+1)-th connection's `accept()` doesn't
//! resolve until an existing connection closes and releases its permit -- the OS backlog
//! absorbs the excess instead of the process spinning up unbounded per-connection state.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::serve::Listener;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A `Listener` wrapper that bounds the number of concurrently-open connections accepted from
/// `inner`. Generic over the wrapped listener so it composes with anything implementing
/// `axum::serve::Listener` -- in practice `tokio::net::TcpListener` (the plaintext path).
///
/// The native-TLS path (`tls` feature, `src/tls.rs`) does NOT wrap this type: its connection cap
/// is a separate `tokio::sync::Semaphore` acquired AFTER a successful handshake rather than at
/// raw accept (see `tls.rs` module docs, round-1 fix I1) -- capping at accept, as this type does,
/// would let a stalled pre-auth TLS handshake occupy the cap for its whole timeout window and
/// starve legitimate clients. Both still bound total concurrently-open connections, but they
/// differ in more than just WHEN the permit is taken: this type QUEUES the `(n+1)`-th connection
/// (its `accept()` simply doesn't resolve until a permit frees, with the OS backlog absorbing the
/// wait), while the TLS path SHEDS (closes) a newly-handshaken connection outright when the cap is
/// full, rather than awaiting a permit (round-2 fix N1) -- an awaited acquire there would let
/// enough parked post-handshake connections exhaust the separate pre-auth handshake bound too,
/// re-coupling the two budgets in reverse. Queuing is correct for this type (nothing else is at
/// stake pre-handshake); shedding is correct for the TLS path (an awaited queue there would risk
/// unbounded post-handshake FDs in limbo, or the reverse starvation above).
pub struct CappedListener<L> {
    inner: L,
    sem: Arc<Semaphore>,
}

impl<L> CappedListener<L> {
    /// Wrap `inner`, admitting at most `max_connections` concurrently-open connections.
    pub fn new(inner: L, max_connections: usize) -> Self {
        Self {
            inner,
            sem: Arc::new(Semaphore::new(max_connections)),
        }
    }
}

/// An accepted connection's I/O stream tied to the semaphore permit that admitted it. Dropping
/// the stream (connection close, either side) drops the permit, which is what lets the next
/// queued `accept()` proceed.
pub struct PermitGuardedIo<Io> {
    io: Io,
    _permit: OwnedSemaphorePermit,
}

impl<Io: AsyncRead + Unpin> AsyncRead for PermitGuardedIo<Io> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_read(cx, buf)
    }
}

impl<Io: AsyncWrite + Unpin> AsyncWrite for PermitGuardedIo<Io> {
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

impl<L: Listener> Listener for CappedListener<L> {
    type Io = PermitGuardedIo<L::Io>;
    type Addr = L::Addr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        // Blocks here (not inside `inner.accept()`) once `max_connections` connections are
        // live, so the underlying listener simply doesn't get polled for a new connection until
        // one frees up -- unaccepted SYNs queue in the OS backlog instead of the process
        // admitting unbounded concurrent connections. The semaphore is only ever closed by
        // dropping this whole `CappedListener`, which also stops anyone calling `accept()`.
        let permit = self
            .sem
            .clone()
            .acquire_owned()
            .await
            .expect("connection-cap semaphore is never closed while the listener is alive");
        // `Listener::accept` implementations (e.g. axum's own for `TcpListener`) already loop
        // internally on transient accept errors, logging and retrying -- so a single await here
        // is enough to get a live connection back.
        let (io, addr) = self.inner.accept().await;
        (
            PermitGuardedIo {
                io,
                _permit: permit,
            },
            addr,
        )
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

// NOTE (flagged deviation, see task-7-report.md): axum only ships `Connected<IncomingStream<'_,
// L>> for SocketAddr` for its OWN listener types (`TcpListener`/`UnixListener`); bridging it for
// a custom `Listener` wrapper like `CappedListener` from outside axum's crate is not possible --
// `Connected` and `IncomingStream` are both foreign here, and Rust's orphan rule rejects an impl
// whose only local type is nested inside another foreign generic. Practical effect: when the
// connection-cap guard (this module) and the per-IP rate-limit guard are BOTH enabled,
// `serve_streamable_http` serves without `into_make_service_with_connect_info` (see the listener
// match there), so `ConnectInfo` is never populated -- the rate-limit guard's peer-IP fallback
// only fires for requests that carry a trusted X-Forwarded-For/X-Real-Ip/Forwarded header.
// Requests with neither do NOT get rejected: `main.rs`'s `FailOpenIpKeyExtractor` degrades them
// to one shared global rate-limit bucket instead of failing closed (see its doc comment) --
// `tower_governor`'s own extractors would otherwise turn an unextractable key into an outright
// rejection of every such request, which would make this combination -- the exact
// "native-TLS-without-proxy hardening" scenario these guards exist for -- reject all /mcp
// traffic.
