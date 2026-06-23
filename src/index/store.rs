use crate::index::multilang::{default_detector, multilang_analyzer};
use anyhow::Context;
use std::path::Path;
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, STRING};
use tantivy::{Index, TantivyError};

#[derive(Clone, Copy)]
pub struct Fields {
    pub body: Field,
    pub path: Field,
    pub location: Field,
    pub file_type: Field,
}

pub struct DocIndex {
    pub index: Index,
    pub fields: Fields,
}

fn build_schema() -> (Schema, Fields) {
    let mut sb = Schema::builder();
    let body_opts = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("multilang")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let body = sb.add_text_field("body", body_opts);
    let path = sb.add_text_field("path", STRING | STORED);
    let location = sb.add_text_field("location", STRING | STORED);
    let file_type = sb.add_text_field("file_type", STRING | STORED);
    (sb.build(), Fields { body, path, location, file_type })
}

fn index_dir(dir: &Path) -> std::path::PathBuf {
    dir.join(".glossa").join("index")
}

impl DocIndex {
    pub fn open_or_create(dir: &Path) -> anyhow::Result<DocIndex> {
        let (schema, fields) = build_schema();
        let idx_path = index_dir(dir);
        std::fs::create_dir_all(&idx_path).with_context(|| format!("create {idx_path:?}"))?;
        let index = match Index::create_in_dir(&idx_path, schema.clone()) {
            Ok(i) => i,
            Err(TantivyError::IndexAlreadyExists) => Index::open_in_dir(&idx_path)?,
            Err(e) => return Err(e.into()),
        };
        index
            .tokenizers()
            .register("multilang", multilang_analyzer(default_detector()));
        Ok(DocIndex { index, fields })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_then_reopens_index() {
        let dir = tempfile::tempdir().unwrap();
        let a = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(a.index.schema().get_field("body").is_ok());
        drop(a);
        // Second call must open the existing index, not error.
        let b = DocIndex::open_or_create(dir.path()).unwrap();
        assert!(b.index.schema().get_field("path").is_ok());
    }
}
