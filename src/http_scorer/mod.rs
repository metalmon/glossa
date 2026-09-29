//! Remote HTTP scorer: the TEI/Jina wire contract (`wire`, always compiled) shared by the
//! `inference-server` (kb-eval) and the sync HTTP clients (`client`, feature `http-scorer`).
pub mod wire;

#[cfg(feature = "http-scorer")]
pub mod client;
