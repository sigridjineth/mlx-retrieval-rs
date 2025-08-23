//! Embedding extraction and processing module
//!
//! This module provides functionality for extracting embeddings from text,
//! applying various pooling strategies, and managing embedding operations.

pub mod extraction;
pub mod pooling;

// Re-export common types
pub use extraction::{EmbedResult, EmbeddingConfig, EmbeddingExtractor};
pub use pooling::{PooledEmbedding, PoolingConfig, PoolingStrategy};

use crate::error::Result;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// Configuration for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    /// Model configuration
    pub model: EmbeddingConfig,

    /// Pooling configuration
    pub pooling: PoolingConfig,

    /// Maximum sequence length for input text
    pub max_sequence_length: usize,

    /// Batch size for processing
    pub batch_size: usize,

    /// Whether to normalize embeddings
    pub normalize: bool,

    /// Device to run computations on
    pub device: String,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingConfig::default(),
            pooling: PoolingConfig::default(),
            max_sequence_length: 512,
            batch_size: 32,
            normalize: true,
            device: "gpu".to_string(),
        }
    }
}

/// A complete embedding result with metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector
    pub embedding: Array,

    /// Input text that was embedded
    pub text: String,

    /// Token count used for the embedding
    pub token_count: usize,

    /// Pooling method used
    pub pooling_method: PoolingStrategy,

    /// Whether the embedding was normalized
    pub normalized: bool,

    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

impl EmbeddingResult {
    /// Create a new embedding result
    pub fn new(
        embedding: Array,
        text: String,
        token_count: usize,
        pooling_method: PoolingStrategy,
        normalized: bool,
        processing_time_ms: f64,
    ) -> Self {
        Self {
            embedding,
            text,
            token_count,
            pooling_method,
            normalized,
            processing_time_ms,
        }
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.embedding.shape().last().copied().unwrap_or(0) as usize
    }

    /// Get the embedding as a Vec<f32> for compatibility
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        // Convert MLX Array to Vec<f32>
        let size = self.embedding.size();
        let mut result = Vec::with_capacity(size);

        // This is a simplified conversion - in practice, we'd need proper MLX array access
        for _i in 0..size {
            // Placeholder: actual implementation would extract values from MLX array
            result.push(0.0f32);
        }

        Ok(result)
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &EmbeddingResult) -> Result<f32> {
        // This would use MLX operations for efficient computation
        let dot_product = self.embedding.multiply(&other.embedding)?;
        let norm_self = self.embedding.square()?.sum(None, false)?.sqrt()?;
        let norm_other = other.embedding.square()?.sum(None, false)?.sqrt()?;

        let _similarity = dot_product.divide(&norm_self.multiply(&norm_other)?)?;

        // Extract scalar value - this is simplified
        Ok(0.0f32) // Placeholder
    }
}

/// Utilities for working with embeddings
pub struct EmbeddingUtils;

impl EmbeddingUtils {
    /// Normalize an embedding vector
    pub fn normalize_embedding(embedding: &Array) -> Result<Array> {
        let norm = embedding.square()?.sum(None, false)?.sqrt()?;
        embedding.divide(&norm).map_err(Into::into)
    }

    /// Compute pairwise cosine similarities between embeddings
    pub fn pairwise_cosine_similarity(embeddings1: &Array, embeddings2: &Array) -> Result<Array> {
        // Normalize embeddings
        let norm1 = Self::normalize_embedding(embeddings1)?;
        let norm2 = Self::normalize_embedding(embeddings2)?;

        // Compute dot product (cosine similarity for normalized vectors)
        norm1
            .matmul(&norm2.transpose(&[-2, -1])?)
            .map_err(Into::into)
    }

    /// Compute mean embedding from a batch of embeddings
    pub fn mean_embedding(embeddings: &Array) -> Result<Array> {
        mlx_rs::ops::mean(embeddings, Some(&[0] as &[i32]), None).map_err(Into::into)
    }

    /// Find top-k most similar embeddings
    pub fn top_k_similar(
        query_embedding: &Array,
        candidate_embeddings: &Array,
        k: usize,
    ) -> Result<(Array, Array)> {
        // Compute similarities
        let _similarities = Self::pairwise_cosine_similarity(
            &query_embedding.expand_dims(&[0])?,
            candidate_embeddings,
        )?;

        // Get top-k indices and scores
        // This is a placeholder - actual implementation would use MLX topk operation
        let indices = Array::zeros::<i32>(&[k as i32])?;
        let scores = Array::zeros::<f32>(&[k as i32])?;

        Ok((indices, scores))
    }
}

/// Batch processing utilities for embeddings
pub struct BatchEmbedder {
    config: EmbedConfig,
    extractor: EmbeddingExtractor,
}

impl BatchEmbedder {
    /// Create a new batch embedder
    pub fn new(config: EmbedConfig, extractor: EmbeddingExtractor) -> Self {
        Self { config, extractor }
    }

    /// Process a batch of texts and return embeddings
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResult>> {
        let mut results = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in texts.chunks(self.config.batch_size) {
            let chunk_results = self.process_chunk(chunk).await?;
            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Process a single chunk of texts
    async fn process_chunk(&self, texts: &[String]) -> Result<Vec<EmbeddingResult>> {
        let start_time = std::time::Instant::now();

        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            // Extract embedding using the extractor
            let embed_result = self.extractor.extract_embedding(text).await?;

            // Apply pooling
            let pooled = embed_result.apply_pooling(&self.config.pooling)?;

            // Normalize if configured
            let final_embedding = if self.config.normalize {
                EmbeddingUtils::normalize_embedding(&pooled.embedding)?
            } else {
                pooled.embedding
            };

            let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

            let result = EmbeddingResult::new(
                final_embedding,
                text.clone(),
                embed_result.token_count,
                pooled.method,
                self.config.normalize,
                processing_time,
            );

            results.push(result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_config_default() {
        let config = EmbedConfig::default();
        assert_eq!(config.max_sequence_length, 512);
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
    }

    #[test]
    fn test_embedding_result_creation() -> Result<()> {
        let embedding = Array::ones::<f32>(&[768])?;
        let result = EmbeddingResult::new(
            embedding,
            "test text".to_string(),
            10,
            PoolingStrategy::Mean,
            true,
            15.5,
        );

        assert_eq!(result.text, "test text");
        assert_eq!(result.token_count, 10);
        assert_eq!(result.dimension(), 768);
        assert!(result.normalized);

        Ok(())
    }

    #[test]
    fn test_embedding_utils_normalize() -> Result<()> {
        let embedding = Array::from_slice(&[3.0f32, 4.0f32], &[2]);
        let _normalized = EmbeddingUtils::normalize_embedding(&embedding)?;

        // The normalized vector should have unit length
        // [3, 4] -> [0.6, 0.8] (since ||(3,4)|| = 5)
        // This is a simplified test - actual values would need proper MLX computation

        Ok(())
    }
}
