//! Batching utilities for efficient data processing
//!
//! This module provides functionality for creating and managing batches of data
//! for training and inference operations.

use crate::data::TextRecord;
use crate::error::{MlxRetrievalError, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::mpsc;

/// Configuration for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Number of samples per batch
    pub batch_size: usize,

    /// Maximum sequence length for tokenization
    pub max_seq_length: usize,

    /// Whether to shuffle batches
    pub shuffle: bool,

    /// Whether to drop the last incomplete batch
    pub drop_last: bool,

    /// Number of worker threads for parallel processing
    pub num_workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_seq_length: 512,
            shuffle: true,
            drop_last: false,
            num_workers: 4,
        }
    }
}

/// A batch of training data
#[derive(Debug, Clone)]
pub struct Batch {
    /// Batch of input IDs (tokenized text)
    pub input_ids: Array,

    /// Attention masks for the inputs
    pub attention_mask: Array,

    /// Optional labels for supervised learning
    pub labels: Option<Array>,

    /// Batch metadata
    pub metadata: BatchMetadata,
}

/// Metadata associated with a batch
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Number of samples in the batch
    pub batch_size: usize,

    /// Actual sequence lengths before padding
    pub sequence_lengths: Vec<usize>,

    /// Original record IDs
    pub record_ids: Vec<String>,

    /// Batch index for tracking
    pub batch_index: usize,
}

impl Batch {
    /// Create a new batch
    pub fn new(
        input_ids: Array,
        attention_mask: Array,
        labels: Option<Array>,
        metadata: BatchMetadata,
    ) -> Result<Self> {
        // Validate dimensions
        let input_shape = input_ids.shape();
        let mask_shape = attention_mask.shape();

        if input_shape != mask_shape {
            return Err(MlxRetrievalError::invalid_input(format!(
                "Input IDs shape {input_shape:?} does not match attention mask shape {mask_shape:?}"
            )));
        }

        if let Some(ref label_array) = labels {
            let label_shape = label_array.shape();
            if label_shape[0] != input_shape[0] {
                return Err(MlxRetrievalError::invalid_input(format!(
                    "Labels batch size {} does not match input batch size {}",
                    label_shape[0], input_shape[0]
                )));
            }
        }

        Ok(Self {
            input_ids,
            attention_mask,
            labels,
            metadata,
        })
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.metadata.batch_size
    }

    /// Get the sequence length
    pub fn seq_length(&self) -> usize {
        self.input_ids.shape().get(1).copied().unwrap_or(0) as usize
    }

    /// Move batch to a specific device (for future device management)
    pub fn to_device(self, _device: &str) -> Result<Self> {
        // MLX handles device placement automatically
        // This is a placeholder for future device management
        Ok(self)
    }
}

/// Batch loader for creating batches from data
pub struct BatchLoader {
    config: BatchConfig,
    buffer: VecDeque<TextRecord>,
    batch_index: usize,
}

impl BatchLoader {
    /// Create a new batch loader
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            batch_index: 0,
        }
    }

    /// Add records to the buffer
    pub fn add_records(&mut self, records: Vec<TextRecord>) {
        if self.config.shuffle {
            // Simple shuffle - in production, use proper randomization
            let mut records = records;
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            records.shuffle(&mut rng);
            self.buffer.extend(records);
        } else {
            self.buffer.extend(records);
        }
    }

    /// Try to create a batch from buffered records
    pub fn try_create_batch(&mut self) -> Result<Option<Batch>> {
        if self.buffer.len() < self.config.batch_size && !self.config.drop_last {
            if self.buffer.is_empty() {
                return Ok(None);
            }
        } else if self.buffer.len() < self.config.batch_size {
            return Ok(None);
        }

        let batch_size = std::cmp::min(self.config.batch_size, self.buffer.len());
        let records: Vec<_> = (0..batch_size)
            .map(|_| self.buffer.pop_front().unwrap())
            .collect();

        self.create_batch_from_records(records)
    }

    /// Create a batch from a set of records
    fn create_batch_from_records(&mut self, records: Vec<TextRecord>) -> Result<Option<Batch>> {
        let batch_size = records.len();
        if batch_size == 0 {
            return Ok(None);
        }

        // For now, create dummy arrays - in production, this would use tokenization
        let seq_length = self.config.max_seq_length;

        // Create input_ids array (batch_size, seq_length)
        let input_ids = Array::zeros::<i32>(&[batch_size as i32, seq_length as i32])
            .map_err(MlxRetrievalError::from)?;

        // Create attention_mask array (batch_size, seq_length)
        let attention_mask = Array::ones::<i32>(&[batch_size as i32, seq_length as i32])
            .map_err(MlxRetrievalError::from)?;

        // Collect metadata
        let record_ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
        let sequence_lengths: Vec<usize> = records
            .iter()
            .map(|r| r.text.len().min(self.config.max_seq_length))
            .collect();

        let metadata = BatchMetadata {
            batch_size,
            sequence_lengths,
            record_ids,
            batch_index: self.batch_index,
        };

        self.batch_index += 1;

        let batch = Batch::new(input_ids, attention_mask, None, metadata)?;
        Ok(Some(batch))
    }

    /// Check if there are more batches available
    pub fn has_batches(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of remaining records in buffer
    pub fn remaining_count(&self) -> usize {
        self.buffer.len()
    }
}

/// Async batch producer for streaming data processing
pub struct BatchProducer {
    config: BatchConfig,
    sender: mpsc::Sender<Result<Batch>>,
    receiver: mpsc::Receiver<Result<Batch>>,
}

impl BatchProducer {
    /// Create a new batch producer
    pub fn new(config: BatchConfig, buffer_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel(buffer_size);

        Self {
            config,
            sender,
            receiver,
        }
    }

    /// Get the receiver for consuming batches
    pub fn receiver(self) -> mpsc::Receiver<Result<Batch>> {
        self.receiver
    }

    /// Process records asynchronously and produce batches
    pub async fn process_records(&self, records: Vec<TextRecord>) -> Result<()> {
        let mut loader = BatchLoader::new(self.config.clone());
        loader.add_records(records);

        while let Some(batch) = loader.try_create_batch()? {
            if self.sender.send(Ok(batch)).await.is_err() {
                break; // Receiver dropped
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_seq_length, 512);
        assert!(config.shuffle);
    }

    #[test]
    fn test_batch_loader() -> Result<()> {
        let config = BatchConfig {
            batch_size: 2,
            max_seq_length: 128,
            shuffle: false,
            drop_last: false,
            num_workers: 1,
        };

        let mut loader = BatchLoader::new(config);

        let records = vec![
            TextRecord::new("1", "first text"),
            TextRecord::new("2", "second text"),
            TextRecord::new("3", "third text"),
        ];

        loader.add_records(records);

        // Should create first batch with 2 records
        let batch1 = loader.try_create_batch()?.unwrap();
        assert_eq!(batch1.batch_size(), 2);

        // Should create second batch with 1 record
        let batch2 = loader.try_create_batch()?.unwrap();
        assert_eq!(batch2.batch_size(), 1);

        // No more batches
        assert!(loader.try_create_batch()?.is_none());

        Ok(())
    }

    #[test]
    fn test_batch_metadata() {
        let metadata = BatchMetadata {
            batch_size: 2,
            sequence_lengths: vec![10, 15],
            record_ids: vec!["1".to_string(), "2".to_string()],
            batch_index: 0,
        };

        assert_eq!(metadata.batch_size, 2);
        assert_eq!(metadata.sequence_lengths.len(), 2);
        assert_eq!(metadata.record_ids.len(), 2);
    }
}
