use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_path: PathBuf,
    pub location: String,
    pub file_type: String,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_constructs_and_compares() {
        let a = Chunk {
            doc_path: PathBuf::from("a.md"),
            location: "Intro".into(),
            file_type: "md".into(),
            text: "hello".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
