//! JSONL (JSON Lines) format support for data I/O
//!
//! This module provides efficient reading and writing of JSONL files,
//! which are commonly used for storing training data in NLP tasks.

use crate::data::{DataStats, QueryDocPair, TextRecord};
use crate::error::{ErrorContext, MlxRetrievalError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use tokio::fs::File as AsyncFile;
use tokio::io::{
    AsyncBufReadExt, AsyncWriteExt, BufReader as AsyncBufReader, BufWriter as AsyncBufWriter,
};

/// A generic JSONL record that can be deserialized into different types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonlRecord {
    /// Text record for embedding tasks
    Text(TextRecord),
    /// Query-document pair for retrieval tasks
    QueryDoc(QueryDocPair),
    /// Generic JSON object for flexibility
    Generic(serde_json::Value),
}

impl JsonlRecord {
    /// Try to convert to a TextRecord
    pub fn as_text_record(&self) -> Option<&TextRecord> {
        match self {
            JsonlRecord::Text(record) => Some(record),
            _ => None,
        }
    }

    /// Try to convert to a QueryDocPair
    pub fn as_query_doc_pair(&self) -> Option<&QueryDocPair> {
        match self {
            JsonlRecord::QueryDoc(pair) => Some(pair),
            _ => None,
        }
    }

    /// Get the record as a generic JSON value
    #[allow(static_mut_refs)]
    pub fn as_json(&self) -> &serde_json::Value {
        match self {
            JsonlRecord::Generic(value) => value,
            JsonlRecord::Text(record) => {
                // This is a bit inefficient, but provides the interface
                static mut TEMP_VALUE: Option<serde_json::Value> = None;
                unsafe {
                    TEMP_VALUE = Some(serde_json::to_value(record).unwrap_or_default());
                    TEMP_VALUE.as_ref().unwrap()
                }
            }
            JsonlRecord::QueryDoc(pair) => {
                static mut TEMP_VALUE: Option<serde_json::Value> = None;
                unsafe {
                    TEMP_VALUE = Some(serde_json::to_value(pair).unwrap_or_default());
                    TEMP_VALUE.as_ref().unwrap()
                }
            }
        }
    }
}

impl From<TextRecord> for JsonlRecord {
    fn from(record: TextRecord) -> Self {
        JsonlRecord::Text(record)
    }
}

impl From<QueryDocPair> for JsonlRecord {
    fn from(pair: QueryDocPair) -> Self {
        JsonlRecord::QueryDoc(pair)
    }
}

/// JSONL reader for efficient sequential reading
pub struct JsonlReader<R> {
    reader: BufReader<R>,
    line_number: usize,
    stats: DataStats,
}

impl JsonlReader<File> {
    /// Create a new JSONL reader from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)
            .with_context(|| format!("Failed to open JSONL file: {:?}", path.as_ref()))?;
        Ok(Self::new(file))
    }
}

impl<R: std::io::Read> JsonlReader<R> {
    /// Create a new JSONL reader from any Read implementor
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            line_number: 0,
            stats: DataStats::new(),
        }
    }

    /// Read all records from the JSONL file
    pub fn read_all(&mut self) -> Result<Vec<JsonlRecord>> {
        let mut records = Vec::new();

        for record in self {
            records.push(record?);
        }

        Ok(records)
    }

    /// Read only text records, filtering out other types
    pub fn read_text_records(&mut self) -> Result<Vec<TextRecord>> {
        let mut _records: Vec<TextRecord> = Vec::new();

        // Collect all records first, then update stats
        let mut text_records = Vec::new();
        for record in self.by_ref() {
            let record = record?;
            if let JsonlRecord::Text(text_record) = record {
                text_records.push(text_record);
            }
        }

        // Update stats with collected records
        for text_record in &text_records {
            self.stats.update(text_record);
        }

        self.stats.finalize();
        Ok(text_records)
    }

    /// Get statistics about the data read so far
    pub fn stats(&self) -> &DataStats {
        &self.stats
    }

    /// Get the current line number
    pub fn line_number(&self) -> usize {
        self.line_number
    }
}

impl<R: std::io::Read> Iterator for JsonlReader<R> {
    type Item = Result<JsonlRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                self.line_number += 1;
                let line = line.trim();

                if line.is_empty() {
                    return self.next(); // Skip empty lines
                }

                match serde_json::from_str::<JsonlRecord>(line) {
                    Ok(record) => Some(Ok(record)),
                    Err(e) => Some(Err(MlxRetrievalError::generic(
                        &format!("Failed to parse line {}", self.line_number),
                        e,
                    ))),
                }
            }
            Err(e) => Some(Err(MlxRetrievalError::generic(
                &format!("Failed to read line {}", self.line_number + 1),
                e,
            ))),
        }
    }
}

/// JSONL writer for efficient sequential writing
pub struct JsonlWriter<W: std::io::Write> {
    writer: BufWriter<W>,
    written_count: usize,
}

impl JsonlWriter<File> {
    /// Create a new JSONL writer from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(&path)
            .with_context(|| format!("Failed to create JSONL file: {:?}", path.as_ref()))?;
        Ok(Self::new(file))
    }
}

impl<W: std::io::Write> JsonlWriter<W> {
    /// Create a new JSONL writer from any Write implementor
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::new(writer),
            written_count: 0,
        }
    }

    /// Write a single record
    pub fn write_record(&mut self, record: &JsonlRecord) -> Result<()> {
        let json_str =
            serde_json::to_string(record).context("Failed to serialize record to JSON")?;

        writeln!(self.writer, "{json_str}").context("Failed to write record to file")?;

        self.written_count += 1;
        Ok(())
    }

    /// Write multiple records
    pub fn write_records(&mut self, records: &[JsonlRecord]) -> Result<()> {
        for record in records {
            self.write_record(record)?;
        }
        Ok(())
    }

    /// Write text records
    pub fn write_text_records(&mut self, records: &[TextRecord]) -> Result<()> {
        for record in records {
            self.write_record(&JsonlRecord::Text(record.clone()))?;
        }
        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("Failed to flush JSONL writer")
    }

    /// Get the number of records written
    pub fn written_count(&self) -> usize {
        self.written_count
    }

    /// Finish writing and return the inner writer
    pub fn finish(mut self) -> Result<W> {
        self.flush()?;
        self.writer
            .into_inner()
            .map_err(|e| MlxRetrievalError::from(e.into_error()))
    }
}

/// Async JSONL reader for non-blocking I/O
pub struct AsyncJsonlReader<R> {
    reader: AsyncBufReader<R>,
    line_number: usize,
}

impl<R: tokio::io::AsyncRead + Unpin> AsyncJsonlReader<R> {
    /// Create a new async JSONL reader
    pub fn new(reader: R) -> Self {
        Self {
            reader: AsyncBufReader::new(reader),
            line_number: 0,
        }
    }

    /// Read the next record asynchronously
    pub async fn read_record(&mut self) -> Result<Option<JsonlRecord>> {
        let mut line = String::new();

        match self.reader.read_line(&mut line).await {
            Ok(0) => Ok(None), // EOF
            Ok(_) => {
                self.line_number += 1;
                let line = line.trim();

                if line.is_empty() {
                    return Box::pin(self.read_record()).await; // Skip empty lines
                }

                serde_json::from_str::<JsonlRecord>(line)
                    .map(Some)
                    .map_err(MlxRetrievalError::Serialization)
                    .map_err(|e| {
                        MlxRetrievalError::generic(
                            &format!("Failed to parse line {}", self.line_number),
                            e,
                        )
                    })
            }
            Err(e) => Err(MlxRetrievalError::generic(
                &format!("Failed to read line {}", self.line_number + 1),
                e,
            )),
        }
    }

    /// Read all records asynchronously
    pub async fn read_all(&mut self) -> Result<Vec<JsonlRecord>> {
        let mut records = Vec::new();

        while let Some(record) = self.read_record().await? {
            records.push(record);
        }

        Ok(records)
    }
}

impl AsyncJsonlReader<AsyncFile> {
    /// Create an async JSONL reader from a file path
    pub async fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = AsyncFile::open(&path)
            .await
            .with_context(|| format!("Failed to open async JSONL file: {:?}", path.as_ref()))?;
        Ok(Self::new(file))
    }
}

/// Async JSONL writer for non-blocking I/O
pub struct AsyncJsonlWriter<W> {
    writer: AsyncBufWriter<W>,
    written_count: usize,
}

impl<W: tokio::io::AsyncWrite + Unpin> AsyncJsonlWriter<W> {
    /// Create a new async JSONL writer
    pub fn new(writer: W) -> Self {
        Self {
            writer: AsyncBufWriter::new(writer),
            written_count: 0,
        }
    }

    /// Write a single record asynchronously
    pub async fn write_record(&mut self, record: &JsonlRecord) -> Result<()> {
        let json_str =
            serde_json::to_string(record).context("Failed to serialize record to JSON")?;

        self.writer
            .write_all(json_str.as_bytes())
            .await
            .context("Failed to write record bytes")?;
        self.writer
            .write_all(b"\n")
            .await
            .context("Failed to write newline")?;

        self.written_count += 1;
        Ok(())
    }

    /// Flush the writer asynchronously
    pub async fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .await
            .context("Failed to flush async JSONL writer")
    }

    /// Get the number of records written
    pub fn written_count(&self) -> usize {
        self.written_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_jsonl_record_conversion() {
        let text_record = TextRecord::new("1", "test text");
        let jsonl_record = JsonlRecord::from(text_record.clone());

        assert!(jsonl_record.as_text_record().is_some());
        assert_eq!(jsonl_record.as_text_record().unwrap(), &text_record);
    }

    #[test]
    fn test_jsonl_reader_writer() -> Result<()> {
        let records = vec![
            JsonlRecord::Text(TextRecord::new("1", "first text")),
            JsonlRecord::Text(TextRecord::new("2", "second text")),
        ];

        // Write to memory buffer
        let mut buffer = Vec::new();
        {
            let mut writer = JsonlWriter::new(&mut buffer);
            writer.write_records(&records)?;
            writer.flush()?;
        }

        // Read from memory buffer
        let cursor = Cursor::new(buffer);
        let mut reader = JsonlReader::new(cursor);
        let read_records = reader.read_all()?;

        assert_eq!(records.len(), read_records.len());

        Ok(())
    }

    #[tokio::test]
    async fn test_async_jsonl_reader() -> Result<()> {
        let data = r#"{"id":"1","text":"first text","title":null,"metadata":{},"embedding":null}
{"id":"2","text":"second text","title":null,"metadata":{},"embedding":null}"#;

        let cursor = Cursor::new(data.as_bytes());
        let mut reader = AsyncJsonlReader::new(cursor);

        let records = reader.read_all().await?;
        assert_eq!(records.len(), 2);

        Ok(())
    }
}
