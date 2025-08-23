//! Streaming data processing utilities
//!
//! This module provides functionality for processing data streams,
//! including real-time data processing and streaming transformations.

use crate::data::{DataStats, JsonlRecord, TextRecord};
use crate::error::{ErrorContext, MlxRetrievalError, Result};
use futures_util::{Stream, StreamExt, TryStreamExt};
use pin_project::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Configuration for stream processing
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size for intermediate processing
    pub buffer_size: usize,

    /// Maximum number of concurrent operations
    pub max_concurrency: usize,

    /// Whether to collect statistics during processing
    pub collect_stats: bool,

    /// Batch size for batched operations
    pub batch_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_concurrency: 10,
            collect_stats: true,
            batch_size: 100,
        }
    }
}

/// A data stream that can process various types of records
#[pin_project]
pub struct DataStream<S> {
    #[pin]
    stream: S,
    config: StreamConfig,
    stats: DataStats,
    processed_count: usize,
}

impl<S> DataStream<S> {
    /// Create a new data stream
    pub fn new(stream: S, config: StreamConfig) -> Self {
        Self {
            stream,
            config,
            stats: DataStats::new(),
            processed_count: 0,
        }
    }

    /// Get the current statistics
    pub fn stats(&self) -> &DataStats {
        &self.stats
    }

    /// Get the number of processed records
    pub fn processed_count(&self) -> usize {
        self.processed_count
    }
}

impl<S, T, E> Stream for DataStream<S>
where
    S: Stream<Item = std::result::Result<T, E>>,
{
    type Item = std::result::Result<T, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        match this.stream.poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => {
                *this.processed_count += 1;
                Poll::Ready(Some(Ok(item)))
            }
            other => other,
        }
    }
}

/// Stream processor for applying transformations to data streams
pub struct StreamProcessor {
    config: StreamConfig,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(config: StreamConfig) -> Self {
        Self { config }
    }

    /// Process a stream of JSONL records
    pub fn process_jsonl_stream<S>(&self, stream: S) -> impl Stream<Item = Result<JsonlRecord>>
    where
        S: Stream<Item = Result<String>> + Send + 'static,
    {
        stream
            .map(|line_result| {
                line_result.and_then(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        return Err(MlxRetrievalError::data_processing("Empty line"));
                    }

                    serde_json::from_str::<JsonlRecord>(line)
                        .map_err(MlxRetrievalError::from)
                        .context("Failed to parse JSONL record")
                })
            })
            .filter_map(|result| async move {
                match result {
                    Ok(record) => Some(Ok(record)),
                    Err(e) => {
                        tracing::warn!("Skipping invalid record: {}", e);
                        None // Skip invalid records
                    }
                }
            })
    }

    /// Filter text records from a mixed stream
    pub fn filter_text_records<S>(&self, stream: S) -> impl Stream<Item = Result<TextRecord>>
    where
        S: Stream<Item = Result<JsonlRecord>> + Send + 'static,
    {
        stream.filter_map(|record_result| async move {
            match record_result {
                Ok(JsonlRecord::Text(text_record)) => Some(Ok(text_record)),
                Ok(_) => None, // Skip non-text records
                Err(e) => Some(Err(e)),
            }
        })
    }

    /// Batch records in a stream
    pub fn batch_records<S, T>(&self, stream: S) -> impl Stream<Item = Result<Vec<T>>>
    where
        S: Stream<Item = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        stream
            .try_chunks(self.config.batch_size)
            .map(|chunk_result| chunk_result.map_err(|err| err.1))
    }

    /// Apply parallel processing to a stream
    pub fn process_parallel<S, T, F, Fut, R>(
        &self,
        stream: S,
        processor: F,
    ) -> impl Stream<Item = Result<R>>
    where
        S: Stream<Item = Result<T>> + Send + 'static,
        T: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + 'static + Clone,
        Fut: futures_util::Future<Output = Result<R>> + Send + 'static,
        R: Send + 'static,
    {
        let results_stream = stream
            .map(move |item| {
                let processor = processor.clone();
                async move {
                    match item {
                        Ok(data) => processor(data).await.map(|_| ()),
                        Err(e) => Err(e),
                    }
                }
            })
            .buffer_unordered(self.config.max_concurrency);

        Box::pin(results_stream.filter_map(|result| async move {
            match result {
                Ok(()) => None, // Process successful, return nothing
                Err(_) => None, // For now, swallow errors - real implementation would collect
            }
        }))
    }

    /// Collect statistics from a text record stream
    pub async fn collect_stats_from_stream<S>(&self, mut stream: S) -> Result<DataStats>
    where
        S: Stream<Item = Result<TextRecord>> + Unpin,
    {
        let mut stats = DataStats::new();

        while let Some(record_result) = stream.next().await {
            let record = record_result?;
            stats.update(&record);
        }

        stats.finalize();
        Ok(stats)
    }
}

/// Create a stream from an async iterator
pub fn from_async_iter<I, T>(iter: I) -> impl Stream<Item = T>
where
    I: IntoIterator<Item = T>,
    I::IntoIter: Send + 'static,
    T: Send + 'static,
{
    futures_util::stream::iter(iter)
}

/// Create a buffered channel stream for producer-consumer patterns
pub fn create_buffered_stream<T>(buffer_size: usize) -> (mpsc::Sender<T>, impl Stream<Item = T>)
where
    T: Send + 'static,
{
    let (sender, receiver) = mpsc::channel(buffer_size);
    let stream = ReceiverStream::new(receiver);
    (sender, stream)
}

/// Stream transformation utilities
pub struct StreamTransform;

impl StreamTransform {
    /// Transform text records by applying a function to the text field
    pub fn map_text<S, F>(stream: S, mapper: F) -> impl Stream<Item = Result<TextRecord>>
    where
        S: Stream<Item = Result<TextRecord>> + Send + 'static,
        F: Fn(String) -> Result<String> + Send + Sync + 'static,
    {
        stream.map(move |record_result| {
            record_result.and_then(|mut record| {
                record.text = mapper(record.text)?;
                Ok(record)
            })
        })
    }

    /// Filter records based on a predicate
    pub fn filter<S, T, F>(stream: S, predicate: F) -> impl Stream<Item = Result<T>>
    where
        S: Stream<Item = Result<T>> + Send + 'static,
        T: Send + 'static,
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let predicate = std::sync::Arc::new(predicate);
        stream.filter_map(move |record_result| {
            let predicate = predicate.clone();
            async move {
                match record_result {
                    Ok(record) if predicate(&record) => Some(Ok(record)),
                    Ok(_) => None,
                    Err(e) => Some(Err(e)),
                }
            }
        })
    }

    /// Take only the first N successful records
    pub fn take<S, T>(stream: S, n: usize) -> impl Stream<Item = Result<T>>
    where
        S: Stream<Item = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let mut count = 0;
        stream.take_while(move |_| {
            let should_continue = count < n;
            count += 1;
            futures_util::future::ready(should_continue)
        })
    }

    /// Skip the first N records
    pub fn skip<S, T>(stream: S, n: usize) -> impl Stream<Item = Result<T>>
    where
        S: Stream<Item = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        stream.skip(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    #[tokio::test]
    async fn test_data_stream_creation() {
        let data: Vec<Result<TextRecord>> = vec![
            Ok(TextRecord::new("1", "first")),
            Ok(TextRecord::new("2", "second")),
        ];

        let stream = stream::iter(data);
        let data_stream = DataStream::new(stream, StreamConfig::default());

        let records: Vec<_> = data_stream.collect().await;
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_stream_processor() {
        let processor = StreamProcessor::new(StreamConfig::default());

        let jsonl_lines = vec![
            Ok(
                r#"{"id":"1","text":"test","title":null,"metadata":{},"embedding":null}"#
                    .to_string(),
            ),
            Ok(
                r#"{"id":"2","text":"test2","title":null,"metadata":{},"embedding":null}"#
                    .to_string(),
            ),
        ];

        let stream = stream::iter(jsonl_lines);
        let records: Vec<Result<JsonlRecord>> =
            processor.process_jsonl_stream(stream).collect().await;

        assert_eq!(records.len(), 2);
        assert!(records[0].is_ok());
        assert!(records[1].is_ok());
    }

    #[tokio::test]
    async fn test_stream_transform() {
        let data = vec![
            Ok(TextRecord::new("1", "hello")),
            Ok(TextRecord::new("2", "world")),
        ];

        let stream = stream::iter(data);
        let transformed: Vec<_> = StreamTransform::map_text(stream, |text| Ok(text.to_uppercase()))
            .collect()
            .await;

        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].as_ref().unwrap().text, "HELLO");
        assert_eq!(transformed[1].as_ref().unwrap().text, "WORLD");
    }

    #[test]
    fn test_buffered_stream() {
        let (sender, stream) = create_buffered_stream::<i32>(10);

        // Just verify the types are correct - actual async testing would require tokio runtime
        drop(sender);
        drop(stream);
    }

    #[tokio::test]
    async fn test_stream_filter() {
        let data = vec![
            Ok(TextRecord::new("1", "short")),
            Ok(TextRecord::new("2", "this is a longer text")),
            Ok(TextRecord::new("3", "tiny")),
        ];

        let stream = stream::iter(data);
        let filtered: Vec<_> = StreamTransform::filter(stream, |record| record.text.len() > 10)
            .collect()
            .await;

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].as_ref().unwrap().text, "this is a longer text");
    }
}
