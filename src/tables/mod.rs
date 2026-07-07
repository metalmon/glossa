//! Agent-authored limit tables (`.glossa/notes/<doc>/*.csp`) and constraint compilation.

pub(crate) mod csv;

#[cfg(feature = "notebook")]
pub use crate::notebook::{
    list_note_paths, mirror_dir_for_doc, normalize_note_file, notes_root, resolve_note_by_document,
    resolve_note_by_path, NoteEntry, NotePath,
};

#[cfg(feature = "constraint")]
mod capabilities;
#[cfg(feature = "constraint")]
mod compile;
#[cfg(feature = "constraint")]
mod coverage;
#[cfg(feature = "constraint")]
mod wiring;

#[cfg(feature = "constraint")]
pub use compile::tables_to_graph;
#[cfg(feature = "constraint")]
pub use coverage::{count_csp_files, csp_column_values};
