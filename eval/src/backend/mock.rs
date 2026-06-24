use super::AgentBackend;
use crate::dataset::Question;
use std::collections::HashMap;
use std::path::Path;

pub struct MockBackend {
    pub canned: HashMap<String, String>,
}

impl AgentBackend for MockBackend {
    fn needs_corpus(&self) -> bool {
        false
    }
    fn answer(&self, _work: &Path, q: &Question) -> anyhow::Result<String> {
        Ok(self.canned.get(&q.id).cloned().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Question;

    #[test]
    fn mock_returns_canned() {
        let mut canned = HashMap::new();
        canned.insert("q1".to_string(), "Bob Page".to_string());
        let b = MockBackend { canned };
        let q = Question { id: "q1".into(), question: "?".into(), answer: "".into(), paragraphs: vec![], supporting_titles: vec![] };
        assert_eq!(b.answer(Path::new("."), &q).unwrap(), "Bob Page");
        assert!(!b.needs_corpus());
    }
}
