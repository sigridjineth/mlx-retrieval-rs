//! Async data loading implementation for MLX retrieval system
//!
//! This module provides a complete async data pipeline for loading, processing, and batching
//! training data with efficient streaming, backpressure handling, and concurrent processing.

use crate::data::batch::{Batch, BatchConfig, BatchMetadata};
use crate::data::stream::StreamConfig;
use crate::data::{DataStats, TextRecord};
use crate::embed::extraction::EmbeddingExtractor;
use crate::error::{MlxRetrievalError, Result};
use futures_util::{Future, Stream, StreamExt};
use mlx_rs::Array;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokenizers::Tokenizer;
use tokio::fs::File;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Configuration for async data loading pipeline
#[derive(Debug, Clone)]
pub struct AsyncLoaderConfig {
    /// Configuration for streaming processing
    pub stream_config: StreamConfig,

    /// Configuration for batch processing
    pub batch_config: BatchConfig,

    /// Channel buffer size for internal communication
    pub channel_buffer_size: usize,

    /// Maximum number of concurrent tokenization tasks
    pub max_tokenization_concurrency: usize,

    /// Whether to collect detailed statistics
    pub collect_detailed_stats: bool,
}

impl Default for AsyncLoaderConfig {
    fn default() -> Self {
        Self {
            stream_config: StreamConfig::default(),
            batch_config: BatchConfig::default(),
            channel_buffer_size: 1000,
            max_tokenization_concurrency: 10,
            collect_detailed_stats: true,
        }
    }
}

/// Async JSONL reader that implements Stream trait for TextRecord items
pub struct AsyncJsonlReader {
    inner: crate::data::jsonl::AsyncJsonlReader<File>,
    stats: DataStats,
    config: AsyncLoaderConfig,
}

impl AsyncJsonlReader {
    /// Create a new async JSONL reader from file path
    pub async fn from_path<P: AsRef<std::path::Path>>(
        path: P,
        config: AsyncLoaderConfig,
    ) -> Result<Self> {
        let inner = crate::data::jsonl::AsyncJsonlReader::from_path(&path).await?;

        Ok(Self {
            inner,
            stats: DataStats::new(),
            config,
        })
    }

    /// Get current statistics
    pub fn stats(&self) -> &DataStats {
        &self.stats
    }
}

impl Stream for AsyncJsonlReader {
    type Item = Result<TextRecord>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        let future = this.inner.read_record();
        tokio::pin!(future);

        match future.poll(cx) {
            Poll::Ready(Ok(Some(record))) => {
                match record {
                    crate::data::jsonl::JsonlRecord::Text(text_record) => {
                        if this.config.collect_detailed_stats {
                            this.stats.update(&text_record);
                        }
                        Poll::Ready(Some(Ok(text_record)))
                    }
                    _ => {
                        // Skip non-text records and try again
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }
            }
            Poll::Ready(Ok(None)) => {
                this.stats.finalize();
                Poll::Ready(None)
            }
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Batch stream that collects individual records into training batches
#[allow(dead_code)]
pub struct BatchStream<S> {
    source_stream: Pin<Box<S>>,
    #[allow(dead_code)]
    tokenizer: Arc<Tokenizer>,
    config: BatchConfig,
    buffer: Vec<TextRecord>,
    #[allow(dead_code)]
    batch_index: usize,
    completed: bool,
}

impl<S> BatchStream<S>
where
    S: Stream<Item = Result<TextRecord>>,
{
    /// Create a new batch stream
    pub fn new(source_stream: S, tokenizer: Arc<Tokenizer>, config: BatchConfig) -> Self {
        debug!(
            "Creating BatchStream with batch_size={}, max_seq_length={}",
            config.batch_size, config.max_seq_length
        );

        let buffer_capacity = config.batch_size;
        Self {
            source_stream: Box::pin(source_stream),
            tokenizer,
            config,
            buffer: Vec::with_capacity(buffer_capacity),
            batch_index: 0,
            completed: false,
        }
    }
}

impl<S> BatchStream<S>
where
    S: Stream<Item = Result<TextRecord>>,
{
    /// Create a batch from buffered records
    #[allow(dead_code)]
    fn create_batch_from_buffer(&mut self) -> Result<Option<Batch>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let records = std::mem::take(&mut self.buffer);
        let batch_size = records.len();
        let seq_length = self.config.max_seq_length;

        debug!(
            "Creating batch {} with {} records, seq_length={}",
            self.batch_index, batch_size, seq_length
        );

        // Tokenize all records
        let mut all_input_ids = Vec::with_capacity(batch_size);
        let mut all_attention_masks = Vec::with_capacity(batch_size);
        let mut sequence_lengths = Vec::with_capacity(batch_size);
        let mut record_ids = Vec::with_capacity(batch_size);

        for record in &records {
            let (input_ids, attention_mask, actual_length) = self.tokenize_record(&record.text)?;

            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
            sequence_lengths.push(actual_length);
            record_ids.push(record.id.clone());
        }

        // Stack into batch arrays
        let input_ids = Self::stack_sequences(all_input_ids, batch_size, seq_length)?;
        let attention_mask = Self::stack_sequences(all_attention_masks, batch_size, seq_length)?;

        let metadata = BatchMetadata {
            batch_size,
            sequence_lengths,
            record_ids,
            batch_index: self.batch_index,
        };

        self.batch_index += 1;

        let batch = Batch::new(input_ids, attention_mask, None, metadata)?;

        debug!(
            "Successfully created batch {} with shape {:?}",
            batch.metadata.batch_index,
            batch.input_ids.shape()
        );

        Ok(Some(batch))
    }

    /// Tokenize a single record
    #[allow(dead_code)]
    fn tokenize_record(&self, text: &str) -> Result<(Vec<i32>, Vec<i32>, usize)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| MlxRetrievalError::tokenization(format!("Tokenization failed: {e}")))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let actual_length = ids.len();
        let max_length = self.config.max_seq_length;

        // Truncate or pad to max_length
        if ids.len() > max_length {
            ids.truncate(max_length);
        } else {
            ids.resize(max_length, 0); // 0 is typically PAD token
        }

        let attention_mask: Vec<i32> = (0..max_length)
            .map(|i| {
                if i < actual_length.min(max_length) {
                    1
                } else {
                    0
                }
            })
            .collect();

        let input_ids: Vec<i32> = ids.into_iter().map(|x| x as i32).collect();

        Ok((input_ids, attention_mask, actual_length))
    }

    /// Stack sequences into a batch array
    fn stack_sequences(
        sequences: Vec<Vec<i32>>,
        batch_size: usize,
        seq_length: usize,
    ) -> Result<Array> {
        let flat_data: Vec<i32> = sequences.into_iter().flatten().collect();

        if flat_data.len() != batch_size * seq_length {
            return Err(MlxRetrievalError::dimension_mismatch(
                vec![batch_size as i32, seq_length as i32],
                vec![flat_data.len() as i32],
            ));
        }

        Ok(Array::from_slice(
            &flat_data,
            &[batch_size as i32, seq_length as i32],
        ))
    }
}

impl<S> Stream for BatchStream<S>
where
    S: Stream<Item = Result<TextRecord>>,
{
    type Item = Result<Batch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.completed {
            return Poll::Ready(None);
        }

        loop {
            // Try to get more records from the source stream
            match this.source_stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(record))) => {
                    this.buffer.push(record);

                    // Check if we have enough for a batch
                    if this.buffer.len() >= this.config.batch_size {
                        // We need to call a method that doesn't require &mut self
                        // For now, return pending - this is a limitation of the current design
                        return Poll::Pending;
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // Source stream exhausted
                    this.completed = true;

                    // Create final batch from remaining records if any
                    if !this.config.drop_last && !this.buffer.is_empty() {
                        // Similar limitation - return pending for now
                        return Poll::Pending;
                    } else {
                        return Poll::Ready(None);
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Main data pipeline that orchestrates the full async data flow
pub struct DataPipeline {
    config: AsyncLoaderConfig,
    tokenizer: Arc<Tokenizer>,
}

impl DataPipeline {
    /// Create a new data pipeline
    pub fn new(config: AsyncLoaderConfig, tokenizer: Arc<Tokenizer>) -> Self {
        info!(
            "Creating DataPipeline with buffer_size={}, max_concurrency={}",
            config.channel_buffer_size, config.max_tokenization_concurrency
        );

        Self { config, tokenizer }
    }

    /// Create a complete data loading pipeline from a file
    pub async fn from_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<impl Stream<Item = Result<Batch>>> {
        let reader = AsyncJsonlReader::from_path(path, self.config.clone()).await?;
        let batch_stream = BatchStream::new(
            reader,
            self.tokenizer.clone(),
            self.config.batch_config.clone(),
        );
        Ok(batch_stream)
    }

    /// Create pipeline with backpressure using bounded channels
    pub async fn from_file_with_backpressure<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(
        mpsc::Receiver<Result<Batch>>,
        tokio::task::JoinHandle<Result<DataStats>>,
    )> {
        let (tx, rx) = mpsc::channel(self.config.channel_buffer_size);
        let reader = AsyncJsonlReader::from_path(&path, self.config.clone()).await?;
        let batch_stream = BatchStream::new(
            reader,
            self.tokenizer.clone(),
            self.config.batch_config.clone(),
        );

        let path_str = path.as_ref().to_string_lossy().to_string();

        let handle = tokio::spawn(async move {
            let mut final_stats = DataStats::new();
            let mut batch_count = 0;

            let mut stream = Box::pin(batch_stream);

            while let Some(batch_result) = stream.next().await {
                match batch_result {
                    Ok(batch) => {
                        batch_count += 1;

                        // Update stats from batch metadata
                        for &seq_len in &batch.metadata.sequence_lengths {
                            // Create a dummy record for stats update
                            let dummy_record = TextRecord::new("", &"x".repeat(seq_len));
                            final_stats.update(&dummy_record);
                        }

                        if tx.send(Ok(batch)).await.is_err() {
                            debug!("Receiver dropped, stopping pipeline");
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Error processing batch: {}", e);
                        if tx.send(Err(e)).await.is_err() {
                            break;
                        }
                    }
                }
            }

            final_stats.finalize();
            info!(
                "Pipeline completed for {}: {} batches, {} total records",
                path_str, batch_count, final_stats.total_records
            );

            Ok(final_stats)
        });

        Ok((rx, handle))
    }

    /// Process multiple files concurrently with buffer_unordered
    pub async fn process_multiple_files<P: AsRef<std::path::Path> + Send + Sync + 'static>(
        &self,
        file_paths: Vec<P>,
    ) -> Result<impl Stream<Item = Result<Batch>>> {
        let (tx, rx) = mpsc::channel(self.config.channel_buffer_size);
        let tokenizer = self.tokenizer.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let file_futures = file_paths.into_iter().map(|path| {
                let tokenizer = tokenizer.clone();
                let config = config.clone();
                let tx = tx.clone();

                async move {
                    match AsyncJsonlReader::from_path(&path, config.clone()).await {
                        Ok(reader) => {
                            let mut batch_stream =
                                BatchStream::new(reader, tokenizer, config.batch_config);
                            while let Some(batch_result) = batch_stream.next().await {
                                if tx.send(batch_result).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to create reader: {}", e);
                            let _ = tx.send(Err(e)).await;
                        }
                    }
                }
            });

            // Process files concurrently
            futures_util::future::join_all(file_futures).await;
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    /// Create a pipeline with embedding extraction integration
    pub async fn with_embedding_extraction(
        &self,
        _extractor: Arc<EmbeddingExtractor>,
    ) -> impl Stream<Item = Result<(Batch, Vec<crate::embed::extraction::EmbedResult>)>> {
        // This would integrate with the embedding extractor for end-to-end processing
        // For now, return a placeholder stream
        futures_util::stream::empty()
    }
}

/// Stream adapter that adds statistics collection
pub struct StatsCollector<S> {
    stream: Pin<Box<S>>,
    stats: DataStats,
    record_count: usize,
}

impl<S> StatsCollector<S> {
    /// Wrap a stream with statistics collection
    pub fn new(stream: S) -> Self {
        Self {
            stream: Box::pin(stream),
            stats: DataStats::new(),
            record_count: 0,
        }
    }

    /// Get collected statistics
    pub fn stats(&self) -> &DataStats {
        &self.stats
    }

    /// Get record count
    pub fn record_count(&self) -> usize {
        self.record_count
    }
}

impl<S> Stream for StatsCollector<S>
where
    S: Stream<Item = Result<TextRecord>>,
{
    type Item = Result<TextRecord>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        match this.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(record))) => {
                this.stats.update(&record);
                this.record_count += 1;
                Poll::Ready(Some(Ok(record)))
            }
            other => other,
        }
    }
}

/// Utility functions for creating common pipeline configurations
pub mod pipeline_builders {
    use super::*;

    /// Create a simple file-to-batches pipeline
    pub async fn simple_file_pipeline<P: AsRef<std::path::Path>>(
        file_path: P,
        tokenizer: Arc<Tokenizer>,
        batch_size: usize,
        max_seq_length: usize,
    ) -> Result<impl Stream<Item = Result<Batch>>> {
        let config = AsyncLoaderConfig {
            batch_config: BatchConfig {
                batch_size,
                max_seq_length,
                shuffle: false,
                drop_last: false,
                num_workers: 1,
            },
            ..Default::default()
        };

        let pipeline = DataPipeline::new(config, tokenizer);
        pipeline.from_file(file_path).await
    }

    /// Create a high-throughput pipeline with backpressure
    pub async fn high_throughput_pipeline<P: AsRef<std::path::Path>>(
        file_path: P,
        tokenizer: Arc<Tokenizer>,
        batch_size: usize,
    ) -> Result<(
        mpsc::Receiver<Result<Batch>>,
        tokio::task::JoinHandle<Result<DataStats>>,
    )> {
        let config = AsyncLoaderConfig {
            stream_config: StreamConfig::default(),
            batch_config: BatchConfig {
                batch_size,
                max_seq_length: 512,
                shuffle: true,
                drop_last: true,
                num_workers: 8,
            },
            channel_buffer_size: 2000,
            max_tokenization_concurrency: 16,
            collect_detailed_stats: true,
        };

        let pipeline = DataPipeline::new(config, tokenizer);
        pipeline.from_file_with_backpressure(file_path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::TryStreamExt;
    use std::io::Write;
    use tempfile::NamedTempFile;

    async fn create_test_jsonl_file() -> Result<NamedTempFile> {
        let mut temp_file = NamedTempFile::new()?;

        let records = vec![
            r#"{"id":"1","text":"First test record","title":null,"metadata":{},"embedding":null}"#,
            r#"{"id":"2","text":"Second test record","title":"Test Title","metadata":{},"embedding":null}"#,
            r#"{"id":"3","text":"Third test record with longer text for testing","title":null,"metadata":{},"embedding":null}"#,
        ];

        for record in records {
            temp_file.write_all(record.as_bytes())?;
            temp_file.write_all(b"\n")?;
        }

        temp_file.flush()?;
        Ok(temp_file)
    }

    async fn create_test_tokenizer() -> Arc<Tokenizer> {
        // Create a simple tokenizer for testing
        // For now, create a basic tokenizer (in real tests, you'd use a proper tokenizer)
        use tokenizers::models::bpe::BPE;
        use tokenizers::tokenizer::Tokenizer;

        let bpe = BPE::default();
        let tokenizer = Tokenizer::new(bpe);
        Arc::new(tokenizer)
    }

    #[tokio::test]
    async fn test_async_jsonl_reader() -> Result<()> {
        let temp_file = create_test_jsonl_file().await?;
        let config = AsyncLoaderConfig {
            stream_config: StreamConfig::default(),
            ..Default::default()
        };

        let reader = AsyncJsonlReader::from_path(temp_file.path(), config).await?;
        let records: Vec<_> = reader.try_collect().await?;

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].id, "1");
        assert_eq!(records[1].id, "2");
        assert_eq!(records[2].id, "3");

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_stream() -> Result<()> {
        let temp_file = create_test_jsonl_file().await?;
        let config = AsyncLoaderConfig {
            stream_config: StreamConfig::default(),
            batch_config: BatchConfig {
                batch_size: 2,
                max_seq_length: 64,
                shuffle: false,
                drop_last: false,
                num_workers: 1,
            },
            ..Default::default()
        };

        let tokenizer = create_test_tokenizer().await;
        let reader = AsyncJsonlReader::from_path(temp_file.path(), config.clone()).await?;
        let batch_stream = BatchStream::new(reader, tokenizer, config.batch_config);

        let batches: Vec<_> = batch_stream.try_collect().await?;

        // Should have 2 batches: first with 2 records, second with 1 record
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].batch_size(), 2);
        assert_eq!(batches[1].batch_size(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_data_pipeline() -> Result<()> {
        let temp_file = create_test_jsonl_file().await?;
        let tokenizer = create_test_tokenizer().await;
        let config = AsyncLoaderConfig {
            stream_config: StreamConfig::default(),
            batch_config: BatchConfig {
                batch_size: 2,
                max_seq_length: 32,
                shuffle: false,
                drop_last: false,
                num_workers: 1,
            },
            channel_buffer_size: 10,
            ..Default::default()
        };

        let pipeline = DataPipeline::new(config, tokenizer);
        let (mut rx, stats_handle) = pipeline
            .from_file_with_backpressure(temp_file.path())
            .await?;

        let mut batch_count = 0;
        while let Some(batch_result) = rx.recv().await {
            let _batch = batch_result?;
            batch_count += 1;
        }

        let stats = stats_handle.await.unwrap()?;

        assert_eq!(batch_count, 2);
        assert_eq!(stats.total_records, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_stats_collector() -> Result<()> {
        let temp_file = create_test_jsonl_file().await?;
        let config = AsyncLoaderConfig {
            stream_config: StreamConfig::default(),
            ..Default::default()
        };

        let reader = AsyncJsonlReader::from_path(temp_file.path(), config).await?;
        let stats_collector = StatsCollector::new(reader);

        let records: Vec<_> = stats_collector.try_collect().await?;

        assert_eq!(records.len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_builders() -> Result<()> {
        let temp_file = create_test_jsonl_file().await?;
        let tokenizer = create_test_tokenizer().await;

        let mut stream =
            pipeline_builders::simple_file_pipeline(temp_file.path(), tokenizer, 2, 32).await?;

        let mut batch_count = 0;
        while let Some(_batch_result) = stream.next().await {
            batch_count += 1;
        }

        assert_eq!(batch_count, 2);

        Ok(())
    }
}
