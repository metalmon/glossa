//! `kbi` internals: server state + startup (one session per model), HTTP handlers +
//! router, and the auth/interlock/overload guard.
pub mod cli;
pub mod guard;
pub mod handlers;
pub mod state;
