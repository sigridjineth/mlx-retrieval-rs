//! Pooling strategies for converting token embeddings to sentence embeddings
//!
//! This module implements various pooling strategies commonly used in sentence
//! embedding models to aggregate token-level representations.

use crate::error::Result;
use mlx_rs::{ops::indexing::IndexOp, Array};
use serde::{Deserialize, Serialize};

/// Available pooling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling over all tokens
    Mean,
    /// Mean pooling with attention mask weighting
    MeanWithMask,
    /// Max pooling over all tokens
    Max,
    /// Use CLS token embedding (first token)
    Cls,
    /// Last token pooling
    LastToken,
    /// Weighted average using attention weights
    AttentionWeighted,
    /// Combination of multiple strategies
    Combined,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::MeanWithMask
    }
}

impl std::fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolingStrategy::Mean => write!(f, "mean"),
            PoolingStrategy::MeanWithMask => write!(f, "mean_with_mask"),
            PoolingStrategy::Max => write!(f, "max"),
            PoolingStrategy::Cls => write!(f, "cls"),
            PoolingStrategy::LastToken => write!(f, "last_token"),
            PoolingStrategy::AttentionWeighted => write!(f, "attention_weighted"),
            PoolingStrategy::Combined => write!(f, "combined"),
        }
    }
}

/// Configuration for pooling operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolingConfig {
    /// Primary pooling strategy
    pub strategy: PoolingStrategy,

    /// Whether to normalize the pooled embedding
    pub normalize: bool,

    /// For combined strategy, weights for different methods
    pub combination_weights: Option<CombinationWeights>,

    /// For attention-weighted pooling, layer to use for attention weights
    pub attention_layer: Option<usize>,
}

impl Default for PoolingConfig {
    fn default() -> Self {
        Self {
            strategy: PoolingStrategy::MeanWithMask,
            normalize: true,
            combination_weights: None,
            attention_layer: None,
        }
    }
}

/// Weights for combining different pooling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinationWeights {
    pub mean: f32,
    pub max: f32,
    pub cls: f32,
}

impl Default for CombinationWeights {
    fn default() -> Self {
        Self {
            mean: 0.4,
            max: 0.3,
            cls: 0.3,
        }
    }
}

/// Result of a pooling operation
#[derive(Debug, Clone)]
pub struct PooledEmbedding {
    /// The pooled embedding vector
    pub embedding: Array,

    /// The pooling method used
    pub method: PoolingStrategy,

    /// Whether the embedding was normalized
    pub normalized: bool,

    /// Additional metadata about the pooling operation
    pub metadata: PoolingMetadata,
}

/// Metadata about pooling operation
#[derive(Debug, Clone)]
pub struct PoolingMetadata {
    /// Original sequence length
    pub sequence_length: usize,

    /// Number of valid (non-padded) tokens
    pub valid_tokens: usize,

    /// Embedding dimension
    pub embedding_dim: usize,
}

impl PooledEmbedding {
    /// Create a new pooled embedding
    pub fn new(
        embedding: Array,
        method: PoolingStrategy,
        normalized: bool,
        sequence_length: usize,
        valid_tokens: usize,
        embedding_dim: usize,
    ) -> Self {
        let metadata = PoolingMetadata {
            sequence_length,
            valid_tokens,
            embedding_dim,
        };

        Self {
            embedding,
            method,
            normalized,
            metadata,
        }
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.metadata.embedding_dim
    }
}

/// Apply pooling strategy to token embeddings
pub fn apply_pooling(
    token_embeddings: &Array,
    attention_mask: &Array,
    config: &PoolingConfig,
) -> Result<PooledEmbedding> {
    let sequence_length = token_embeddings.shape().first().copied().unwrap_or(0) as usize;
    let embedding_dim = token_embeddings.shape().last().copied().unwrap_or(0) as usize;
    let valid_tokens = count_valid_tokens(attention_mask)?;

    let pooled = match config.strategy {
        PoolingStrategy::Mean => simple_mean_pooling(token_embeddings)?,
        PoolingStrategy::MeanWithMask => mean_pooling_with_mask(token_embeddings, attention_mask)?,
        PoolingStrategy::Max => max_pooling(token_embeddings)?,
        PoolingStrategy::Cls => cls_pooling(token_embeddings)?,
        PoolingStrategy::LastToken => last_token_pooling(token_embeddings, attention_mask)?,
        PoolingStrategy::AttentionWeighted => {
            attention_weighted_pooling(token_embeddings, attention_mask)?
        }
        PoolingStrategy::Combined => combined_pooling(token_embeddings, attention_mask, config)?,
    };

    let final_embedding = if config.normalize {
        normalize_embedding(&pooled)?
    } else {
        pooled
    };

    Ok(PooledEmbedding::new(
        final_embedding,
        config.strategy,
        config.normalize,
        sequence_length,
        valid_tokens,
        embedding_dim,
    ))
}

/// Simple mean pooling over all tokens
fn simple_mean_pooling(token_embeddings: &Array) -> Result<Array> {
    token_embeddings.mean(&[1], false).map_err(Into::into)
}

/// Mean pooling with attention mask weighting (public function for testing)
pub fn mean_pooling(hidden_states: &Array, attention_mask: &Array) -> Result<Array> {
    mean_pooling_with_mask(hidden_states, attention_mask)
}

/// Mean pooling with attention mask weighting
fn mean_pooling_with_mask(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
    // token_embeddings shape: [B, L, D]
    // attention_mask shape: [B, L]
    // output shape: [B, D]
    // Following the Python reference:
    // s = mx.sum(hidden_states * attention_mask[:, :, None], axis=1)
    // d = mx.sum(attention_mask, axis=1, keepdims=True).clip(1e-9, None)
    // return s / d

    // Step 1: Expand attention mask to match embedding dimensions [B, L, 1]
    // attention_mask[:, :, None] equivalent to expand_dims at axis=-1
    let mask_expanded = attention_mask.expand_dims(&[2])?;

    // Step 2: Element-wise multiply: hidden_states * attention_mask[:, :, None]
    let masked_embeddings = token_embeddings.multiply(&mask_expanded)?;

    // Step 3: Sum over sequence dimension (axis=1): mx.sum(..., axis=1)
    let s = masked_embeddings.sum(&[1], false)?;

    // Step 4: Sum attention mask over sequence dimension with keepdims
    // mx.sum(attention_mask, axis=1, keepdims=True)
    let d = attention_mask.sum(&[1], true)?;

    // Step 5: Clip to avoid division by zero: .clip(1e-9, None)
    let epsilon = Array::from_float(1e-9f32);
    let d_clipped = mlx_rs::ops::maximum(&d, &epsilon)?;

    // Step 6: Divide: s / d
    let pooled_embeddings = s.divide(&d_clipped)?;

    Ok(pooled_embeddings)
}

/// Max pooling over all tokens
fn max_pooling(token_embeddings: &Array) -> Result<Array> {
    token_embeddings.max(&[1], false).map_err(Into::into)
}

/// Use CLS token (first token) as embedding
fn cls_pooling(token_embeddings: &Array) -> Result<Array> {
    // Extract first token (index 0) for all batch items
    Ok(token_embeddings.index((.., 0, ..)))
}

/// Use last valid token as embedding
pub fn last_token_pooling(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
    let batch_size = token_embeddings.shape()[0];
    let embedding_dim = token_embeddings.shape()[2];

    // Find the last valid token position for each batch item
    let _mask_sum = attention_mask.sum(&[1], false)?;

    // Create result array
    let mut last_tokens = Vec::new();

    for i in 0..batch_size {
        // Get attention mask for this batch item
        let mask_i = attention_mask.index((i, ..));

        // Find last valid token position
        // Sum the mask to get the number of valid tokens
        let num_valid = mask_i.sum(&[], false)?.item::<f32>() as usize;

        // Last valid token is at position (num_valid - 1)
        // Ensure we don't go below 0
        let last_idx = if num_valid > 0 { num_valid - 1 } else { 0 };

        // Extract embedding at last valid position
        let last_token = token_embeddings.index((i, last_idx as i32, ..));
        last_tokens.push(last_token);
    }

    // Stack all last tokens into a batch
    if last_tokens.is_empty() {
        // Return zeros if no valid tokens found
        Ok(Array::zeros::<f32>(&[batch_size, embedding_dim])?)
    } else {
        mlx_rs::ops::stack(&last_tokens, 0).map_err(Into::into)
    }
}

/// Attention-weighted pooling (simplified version)
fn attention_weighted_pooling(token_embeddings: &Array, attention_mask: &Array) -> Result<Array> {
    // For simplicity, use attention mask as weights
    // In a real implementation, this would use actual attention weights from the model
    let mask_normalized = normalize_attention_weights(attention_mask)?;
    let mask_expanded = mask_normalized.expand_dims(&[2])?;

    // Weighted sum
    let weighted_embeddings =
        token_embeddings.multiply(&mask_expanded.as_dtype(token_embeddings.dtype())?)?;
    weighted_embeddings.sum(&[1], false).map_err(Into::into)
}

/// Combined pooling using multiple strategies
fn combined_pooling(
    token_embeddings: &Array,
    attention_mask: &Array,
    config: &PoolingConfig,
) -> Result<Array> {
    let default_weights = CombinationWeights::default();
    let weights = config
        .combination_weights
        .as_ref()
        .unwrap_or(&default_weights);

    // Compute different pooling methods
    let mean_pooled = mean_pooling_with_mask(token_embeddings, attention_mask)?;
    let max_pooled = max_pooling(token_embeddings)?;
    let cls_pooled = cls_pooling(token_embeddings)?;

    // Combine with weights
    let weighted_mean = mean_pooled.multiply(Array::from_float(weights.mean))?;
    let weighted_max = max_pooled.multiply(Array::from_float(weights.max))?;
    let weighted_cls = cls_pooled.multiply(Array::from_float(weights.cls))?;

    // Sum all components
    let combined = weighted_mean.add(&weighted_max)?.add(&weighted_cls)?;

    Ok(combined)
}

/// Normalize an embedding to unit length
fn normalize_embedding(embedding: &Array) -> Result<Array> {
    let norm = embedding.square()?.sum(&[-1], false)?.sqrt()?;

    // Avoid division by zero
    let epsilon = Array::from_float(1e-8);
    let safe_norm = mlx_rs::ops::maximum(&norm, &epsilon)?;

    embedding.divide(&safe_norm).map_err(Into::into)
}

/// Normalize attention weights to sum to 1
fn normalize_attention_weights(attention_mask: &Array) -> Result<Array> {
    let sum = attention_mask.sum(&[], false)?;
    let epsilon = Array::from_float(1e-8);
    let safe_sum = mlx_rs::ops::maximum(&sum, &epsilon)?;

    attention_mask
        .as_dtype(mlx_rs::Dtype::Float32)?
        .divide(&safe_sum)
        .map_err(Into::into)
}

/// Count valid tokens from attention mask
fn count_valid_tokens(attention_mask: &Array) -> Result<usize> {
    let sum = attention_mask.sum(&[], false)?;
    let sum_val = sum.item::<i32>();
    Ok(sum_val as usize)
}

/// Pooling strategy builder for easy configuration
pub struct PoolingConfigBuilder {
    strategy: PoolingStrategy,
    normalize: bool,
    combination_weights: Option<CombinationWeights>,
    attention_layer: Option<usize>,
}

impl Default for PoolingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolingConfigBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            strategy: PoolingStrategy::MeanWithMask,
            normalize: true,
            combination_weights: None,
            attention_layer: None,
        }
    }

    /// Set the pooling strategy
    pub fn strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set combination weights (for combined strategy)
    pub fn combination_weights(mut self, weights: CombinationWeights) -> Self {
        self.combination_weights = Some(weights);
        self
    }

    /// Set attention layer (for attention-weighted strategy)
    pub fn attention_layer(mut self, layer: usize) -> Self {
        self.attention_layer = Some(layer);
        self
    }

    /// Build the configuration
    pub fn build(self) -> PoolingConfig {
        PoolingConfig {
            strategy: self.strategy,
            normalize: self.normalize,
            combination_weights: self.combination_weights,
            attention_layer: self.attention_layer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    use std::fs;

    #[test]
    fn test_mean_pooling_with_mask_golden() -> Result<()> {
        // Load golden test data
        let golden_data = fs::read_to_string("tests/mean_pooling_with_mask.json")
            .expect("Failed to read golden test file");
        let test_case: serde_json::Value =
            serde_json::from_str(&golden_data).expect("Failed to parse golden test JSON");

        // Extract input data
        let hidden_states_data = test_case["inputs"]["hidden_states"]["data"]
            .as_array()
            .unwrap();
        let attention_mask_data = test_case["inputs"]["attention_mask"]["data"]
            .as_array()
            .unwrap();
        let expected_data = test_case["expected_output"]["pooled_embedding"]["data"]
            .as_array()
            .unwrap();

        // Convert JSON data to flat vectors
        let hidden_states_vec: Vec<f32> = hidden_states_data
            .iter()
            .flat_map(|batch| batch.as_array().unwrap())
            .flat_map(|seq| seq.as_array().unwrap())
            .map(|val| val.as_f64().unwrap() as f32)
            .collect();

        let attention_mask_vec: Vec<f32> = attention_mask_data
            .iter()
            .flat_map(|batch| batch.as_array().unwrap())
            .map(|val| val.as_f64().unwrap() as f32)
            .collect();

        let expected_vec: Vec<f32> = expected_data
            .iter()
            .flat_map(|batch| batch.as_array().unwrap())
            .map(|val| val.as_f64().unwrap() as f32)
            .collect();

        // Create MLX arrays
        let hidden_states = Array::from_slice(&hidden_states_vec, &[4, 6, 8]);
        let attention_mask = Array::from_slice(&attention_mask_vec, &[4, 6]);
        let expected_output = Array::from_slice(&expected_vec, &[4, 8]);

        // Call our mean pooling function
        let result = mean_pooling(&hidden_states, &attention_mask)?;

        // Check shapes match
        assert_eq!(result.shape(), expected_output.shape());

        // Check values are close (within tolerance for floating point)
        let diff = &result - &expected_output;
        let max_diff = diff.abs()?.max(&[], false)?;
        let max_diff_val = max_diff.item::<f32>();
        assert!(
            max_diff_val < 1e-5,
            "Max difference {} exceeds tolerance",
            max_diff_val
        );

        Ok(())
    }

    #[test]
    fn test_pooling_strategy_display() {
        assert_eq!(PoolingStrategy::Mean.to_string(), "mean");
        assert_eq!(PoolingStrategy::MeanWithMask.to_string(), "mean_with_mask");
        assert_eq!(PoolingStrategy::Cls.to_string(), "cls");
    }

    #[test]
    fn test_pooling_config_builder() {
        let config = PoolingConfigBuilder::new()
            .strategy(PoolingStrategy::Max)
            .normalize(false)
            .build();

        assert_eq!(config.strategy, PoolingStrategy::Max);
        assert!(!config.normalize);
    }

    #[test]
    fn test_mean_pooling() -> Result<()> {
        // Create test embeddings: 3 tokens, 4 dimensions
        let embeddings = Array::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, // token 1
                2.0, 3.0, 4.0, 5.0, // token 2
                3.0, 4.0, 5.0, 6.0, // token 3
            ],
            &[3, 4],
        );

        let pooled = simple_mean_pooling(&embeddings)?;

        // Expected mean: [2.0, 3.0, 4.0, 5.0]
        assert_eq!(pooled.shape(), vec![4]);

        Ok(())
    }

    #[test]
    fn test_cls_pooling() -> Result<()> {
        let embeddings = Array::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, // CLS token
                5.0, 6.0, 7.0, 8.0, // token 2
                9.0, 10.0, 11.0, 12.0, // token 3
            ],
            &[3, 4],
        );

        let pooled = cls_pooling(&embeddings)?;

        // Should return first token embedding
        assert_eq!(pooled.shape(), vec![4]);

        Ok(())
    }

    #[test]
    fn test_pooled_embedding_creation() -> Result<()> {
        let embedding = Array::ones::<f32>(&[384]).unwrap();
        let pooled = PooledEmbedding::new(embedding, PoolingStrategy::Mean, true, 10, 8, 384);

        assert_eq!(pooled.method, PoolingStrategy::Mean);
        assert!(pooled.normalized);
        assert_eq!(pooled.dimension(), 384);
        assert_eq!(pooled.metadata.sequence_length, 10);
        assert_eq!(pooled.metadata.valid_tokens, 8);

        Ok(())
    }

    #[test]
    fn test_combination_weights_default() {
        let weights = CombinationWeights::default();
        assert_eq!(weights.mean + weights.max + weights.cls, 1.0);
    }
}
