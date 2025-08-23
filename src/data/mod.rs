//! Data handling and preprocessing module
//!
//! This module provides comprehensive data loading, preprocessing, and batching capabilities
//! for training and inference pipelines.

pub mod async_loader;
pub mod batch;
pub mod jsonl;
pub mod stream;

// Re-export common types
pub use async_loader::{AsyncJsonlReader, AsyncLoaderConfig, BatchStream, DataPipeline};
pub use batch::{Batch, BatchConfig, BatchLoader};
pub use jsonl::{JsonlReader, JsonlRecord, JsonlWriter};
pub use stream::{DataStream, StreamProcessor};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common data record structure for text retrieval tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextRecord {
    /// Unique identifier for the record
    pub id: String,

    /// Main text content
    pub text: String,

    /// Optional title or short description
    pub title: Option<String>,

    /// Additional metadata fields
    pub metadata: HashMap<String, serde_json::Value>,

    /// Optional embedding vector (for pre-computed embeddings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl TextRecord {
    /// Create a new text record
    pub fn new<S: Into<String>>(id: S, text: S) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            title: None,
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    /// Create a new text record with title
    pub fn with_title<S: Into<String>>(id: S, text: S, title: S) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            title: Some(title.into()),
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    /// Add metadata to the record
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set the embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Get the display text (title if available, otherwise text)
    pub fn display_text(&self) -> &str {
        self.title.as_deref().unwrap_or(&self.text)
    }
}

/// Query-document pair for retrieval training
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryDocPair {
    /// Query text
    pub query: String,

    /// Positive document
    pub positive_doc: String,

    /// Optional negative documents for contrastive learning
    pub negative_docs: Vec<String>,

    /// Optional relevance score
    pub relevance_score: Option<f32>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl QueryDocPair {
    /// Create a new query-document pair
    pub fn new<S: Into<String>>(query: S, positive_doc: S) -> Self {
        Self {
            query: query.into(),
            positive_doc: positive_doc.into(),
            negative_docs: Vec::new(),
            relevance_score: None,
            metadata: HashMap::new(),
        }
    }

    /// Add negative documents
    pub fn with_negatives(mut self, negatives: Vec<String>) -> Self {
        self.negative_docs = negatives;
        self
    }

    /// Set relevance score
    pub fn with_relevance_score(mut self, score: f32) -> Self {
        self.relevance_score = Some(score);
        self
    }
}

/// Data statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStats {
    /// Total number of records
    pub total_records: usize,

    /// Average text length
    pub avg_text_length: f64,

    /// Maximum text length
    pub max_text_length: usize,

    /// Minimum text length
    pub min_text_length: usize,

    /// Number of records with embeddings
    pub records_with_embeddings: usize,
}

impl DataStats {
    /// Create empty statistics
    pub fn new() -> Self {
        Self {
            total_records: 0,
            avg_text_length: 0.0,
            max_text_length: 0,
            min_text_length: usize::MAX,
            records_with_embeddings: 0,
        }
    }

    /// Update statistics with a new record
    pub fn update(&mut self, record: &TextRecord) {
        let text_length = record.text.len();

        self.total_records += 1;
        self.max_text_length = self.max_text_length.max(text_length);
        self.min_text_length = self.min_text_length.min(text_length);

        if record.embedding.is_some() {
            self.records_with_embeddings += 1;
        }

        // Update average length incrementally
        let delta = text_length as f64 - self.avg_text_length;
        self.avg_text_length += delta / self.total_records as f64;
    }

    /// Finalize statistics calculation
    pub fn finalize(&mut self) {
        if self.total_records == 0 {
            self.min_text_length = 0;
        }
    }
}

impl Default for DataStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_record_creation() {
        let record = TextRecord::new("1", "test text");
        assert_eq!(record.id, "1");
        assert_eq!(record.text, "test text");
        assert_eq!(record.title, None);
    }

    #[test]
    fn test_text_record_with_title() {
        let record = TextRecord::with_title("1", "test text", "Test Title");
        assert_eq!(record.display_text(), "Test Title");
    }

    #[test]
    fn test_query_doc_pair() {
        let pair = QueryDocPair::new("what is rust?", "Rust is a systems programming language");
        assert_eq!(pair.query, "what is rust?");
        assert_eq!(pair.positive_doc, "Rust is a systems programming language");
        assert!(pair.negative_docs.is_empty());
    }

    #[test]
    fn test_data_stats() {
        let mut stats = DataStats::new();
        let record1 = TextRecord::new("1", "short");
        let record2 = TextRecord::new("2", "much longer text");

        stats.update(&record1);
        stats.update(&record2);

        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.max_text_length, 16);
        assert_eq!(stats.min_text_length, 5);
    }
}
