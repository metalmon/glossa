use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// token → number of corpus chunks containing it, plus the chunk count. Persisted at `.glossa/df`.
pub struct DfTable {
    pub n_chunks: u32,
    map: HashMap<String, u32>,
}

impl Default for DfTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DfTable {
    pub fn new() -> Self {
        DfTable {
            n_chunks: 0,
            map: HashMap::new(),
        }
    }

    pub fn add_chunk(&mut self, tokens: &[String]) {
        self.n_chunks += 1;
        let mut seen = std::collections::HashSet::new();
        for tok in tokens {
            if seen.insert(tok.as_str()) {
                *self.map.entry(tok.clone()).or_insert(0) += 1;
            }
        }
    }

    pub fn df(&self, token: &str) -> u32 {
        self.map.get(token).copied().unwrap_or(0)
    }

    /// Number of distinct tokens tracked (vocabulary size) — for progress/status messages.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn is_rare(&self, token: &str, rare_df_frac: f32) -> bool {
        let d = self.df(token);
        d == 0 || (d as f32) <= rare_df_frac * (self.n_chunks as f32)
    }

    pub fn sidecar_path(glossa_dir: &Path) -> PathBuf {
        glossa_dir.join("df")
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        // line format: "<n_chunks>\n" then "<df>\t<token>\n" per entry
        let mut s = format!("{}\n", self.n_chunks);
        for (tok, d) in &self.map {
            s.push_str(&format!("{d}\t{tok}\n"));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        crate::index::manifest::atomic_write(path, s.as_bytes())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut lines = raw.lines();
        let n_chunks = lines
            .next()
            .and_then(|l| l.trim().parse().ok())
            .unwrap_or(0);
        let mut map = HashMap::new();
        for l in lines {
            if let Some((d, tok)) = l.split_once('\t') {
                if let Ok(d) = d.parse::<u32>() {
                    map.insert(tok.to_string(), d);
                }
            }
        }
        Ok(DfTable { n_chunks, map })
    }
}

#[cfg(test)]
mod tests {
    use super::DfTable;
    use crate::gate::token::tokenize;

    #[test]
    fn rarity_and_roundtrip() {
        let mut t = DfTable::new();
        for page in [
            "network modbus setup",
            "modbus and modbus again",
            "channel calibration pp.19.00.00.00",
        ] {
            t.add_chunk(&tokenize(page));
        }
        assert_eq!(t.n_chunks, 3);
        assert_eq!(t.df("modbus"), 2); // on 2 of 3 chunks
        assert_eq!(t.df("pp.19.00.00.00"), 1); // rare code
        assert!(t.is_rare("pp.19.00.00.00", 0.5)); // 1 <= 0.5*3
        assert!(!t.is_rare("modbus", 0.5)); // 2 > 1.5
        assert!(t.is_rare("neverseen", 0.5)); // df==0 ⇒ rare
        let dir = tempfile::tempdir().unwrap();
        let p = DfTable::sidecar_path(dir.path());
        t.save(&p).unwrap();
        let back = DfTable::load(&p).unwrap();
        assert_eq!(back.n_chunks, 3);
        assert_eq!(back.df("modbus"), 2);
    }
}
