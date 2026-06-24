pub mod prompt;
pub mod mock;
pub mod claude;
pub mod qwen;

use crate::dataset::Question;
use std::path::Path;

pub trait AgentBackend {
    fn needs_corpus(&self) -> bool;
    fn answer(&self, work: &Path, q: &Question) -> anyhow::Result<String>;
}
