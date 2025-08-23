//! Embedding extraction from language models
//!
//! This module provides functionality for extracting embeddings from text using
//! various transformer models through the MLX framework.

use crate::embed::pooling::{PooledEmbedding, PoolingConfig};
use crate::error::{MlxRetrievalError, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// Configuration for embedding extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name or path
    pub model_name: String,

    /// Model dimension (embedding size)
    pub model_dim: usize,

    /// Maximum sequence length
    pub max_seq_length: usize,

    /// Whether to use mean pooling by default
    pub use_mean_pooling: bool,

    /// Model-specific parameters
    pub model_params: HashMap<String, serde_json::Value>,

    /// Tokenizer configuration
    pub tokenizer_config: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            model_dim: 384,
            max_seq_length: 512,
            use_mean_pooling: true,
            model_params: HashMap::new(),
            tokenizer_config: None,
        }
    }
}

/// Result of embedding extraction before pooling
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// Raw token embeddings (sequence_length, hidden_dim)
    pub token_embeddings: Array,

    /// Attention mask for valid tokens
    pub attention_mask: Array,

    /// Input IDs that were processed
    pub input_ids: Array,

    /// Number of actual tokens (excluding padding)
    pub token_count: usize,

    /// Special token positions (CLS, SEP, etc.)
    pub special_tokens: HashMap<String, Vec<usize>>,
}

impl EmbedResult {
    /// Create a new embedding result
    pub fn new(
        token_embeddings: Array,
        attention_mask: Array,
        input_ids: Array,
        token_count: usize,
    ) -> Self {
        Self {
            token_embeddings,
            attention_mask,
            input_ids,
            token_count,
            special_tokens: HashMap::new(),
        }
    }

    /// Add special token positions
    pub fn with_special_tokens(mut self, special_tokens: HashMap<String, Vec<usize>>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Get the sequence length
    pub fn seq_length(&self) -> usize {
        self.token_embeddings.shape().first().copied().unwrap_or(0) as usize
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.token_embeddings.shape().last().copied().unwrap_or(0) as usize
    }

    /// Apply pooling to get sentence-level embedding
    pub fn apply_pooling(&self, config: &PoolingConfig) -> Result<PooledEmbedding> {
        crate::embed::pooling::apply_pooling(&self.token_embeddings, &self.attention_mask, config)
    }

    /// Get embeddings for specific tokens
    pub fn get_token_embeddings(&self, token_indices: &[usize]) -> Result<Array> {
        if token_indices.is_empty() {
            return Err(MlxRetrievalError::invalid_input(
                "Token indices cannot be empty",
            ));
        }

        let max_index = token_indices.iter().max().unwrap();
        if *max_index >= self.seq_length() {
            return Err(MlxRetrievalError::invalid_input(format!(
                "Token index {} out of bounds for sequence length {}",
                max_index,
                self.seq_length()
            )));
        }

        // Extract embeddings for specified tokens
        // This is a placeholder - actual implementation would use MLX indexing
        let embedding_dim = self.embedding_dim();
        Array::zeros::<f32>(&[token_indices.len() as i32, embedding_dim as i32]).map_err(Into::into)
    }

    /// Get CLS token embedding (if available)
    pub fn get_cls_embedding(&self) -> Result<Option<Array>> {
        if let Some(cls_positions) = self.special_tokens.get("CLS") {
            if let Some(&first_cls) = cls_positions.first() {
                return Ok(Some(self.get_token_embeddings(&[first_cls])?));
            }
        }
        Ok(None)
    }
}

/// Main embedding extractor
pub struct EmbeddingExtractor {
    config: EmbeddingConfig,
    tokenizer: Option<Tokenizer>,
    model_loaded: bool,
}

impl EmbeddingExtractor {
    /// Create a new embedding extractor
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            tokenizer: None,
            model_loaded: false,
        }
    }

    /// Load the model and tokenizer
    pub async fn load_model(&mut self) -> Result<()> {
        // Load tokenizer
        self.load_tokenizer().await?;

        // Initialize model (placeholder for actual model loading)
        self.model_loaded = true;

        tracing::info!("Loaded embedding model: {}", self.config.model_name);
        Ok(())
    }

    /// Load the tokenizer
    async fn load_tokenizer(&mut self) -> Result<()> {
        // In a real implementation, this would load from HuggingFace Hub
        // For now, we'll create a placeholder

        match &self.config.tokenizer_config {
            Some(tokenizer_path) => {
                self.tokenizer = Some(Tokenizer::from_file(tokenizer_path).map_err(|e| {
                    MlxRetrievalError::tokenization(format!(
                        "Failed to load tokenizer from file: {e}"
                    ))
                })?);
            }
            None => {
                // Would typically download from HuggingFace Hub
                // For now, create a minimal tokenizer setup
                tracing::warn!("No tokenizer config provided, using placeholder");
            }
        }

        Ok(())
    }

    /// Extract embeddings from text
    pub async fn extract_embedding(&self, text: &str) -> Result<EmbedResult> {
        if !self.model_loaded {
            return Err(MlxRetrievalError::model(
                "Model not loaded. Call load_model() first.",
            ));
        }

        // Tokenize input
        let (input_ids, attention_mask, special_tokens) = self.tokenize_text(text)?;

        // Run model inference
        let token_embeddings = self
            .run_model_inference(&input_ids, &attention_mask)
            .await?;

        // Count actual tokens (excluding padding)
        let token_count = self.count_actual_tokens(&attention_mask)?;

        let result = EmbedResult::new(token_embeddings, attention_mask, input_ids, token_count)
            .with_special_tokens(special_tokens);

        Ok(result)
    }

    /// Tokenize input text
    fn tokenize_text(&self, text: &str) -> Result<(Array, Array, HashMap<String, Vec<usize>>)> {
        let max_length = self.config.max_seq_length;

        match &self.tokenizer {
            Some(tokenizer) => {
                let encoding = tokenizer.encode(text, true).map_err(|e| {
                    MlxRetrievalError::tokenization(format!("Failed to tokenize text: {e}"))
                })?;

                let mut ids = encoding.get_ids().to_vec();
                let mut attention_mask = vec![1u32; ids.len()];

                // Truncate or pad to max_length
                if ids.len() > max_length {
                    ids.truncate(max_length);
                    attention_mask.truncate(max_length);
                } else {
                    let pad_len = max_length - ids.len();
                    ids.extend(vec![0u32; pad_len]); // 0 is typically PAD token
                    attention_mask.extend(vec![0u32; pad_len]);
                }

                // Convert to MLX arrays
                let ids_i32: Vec<i32> = ids.into_iter().map(|x| x as i32).collect();
                let input_ids = Array::from_slice(&ids_i32, &[max_length as i32]);

                let attention_mask_i32: Vec<i32> =
                    attention_mask.into_iter().map(|x| x as i32).collect();
                let attention_mask = Array::from_slice(&attention_mask_i32, &[max_length as i32]);

                // Find special tokens (simplified)
                let mut special_tokens = HashMap::new();

                // Look for CLS token (usually ID 101 for BERT-like models)
                if let Some(pos) = encoding.get_ids().iter().position(|&id| id == 101) {
                    special_tokens.insert("CLS".to_string(), vec![pos]);
                }

                // Look for SEP token (usually ID 102)
                let sep_positions: Vec<usize> = encoding
                    .get_ids()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &id)| if id == 102 { Some(i) } else { None })
                    .collect();

                if !sep_positions.is_empty() {
                    special_tokens.insert("SEP".to_string(), sep_positions);
                }

                Ok((input_ids, attention_mask, special_tokens))
            }
            None => {
                // Fallback: create dummy tokens
                let input_ids = Array::ones::<i32>(&[max_length as i32])?;
                let attention_mask = Array::ones::<i32>(&[max_length as i32])?;
                Ok((input_ids, attention_mask, HashMap::new()))
            }
        }
    }

    /// Run model inference to get token embeddings
    async fn run_model_inference(
        &self,
        input_ids: &Array,
        _attention_mask: &Array,
    ) -> Result<Array> {
        // Placeholder for actual model inference
        // In a real implementation, this would:
        // 1. Forward pass through the transformer model
        // 2. Extract hidden states from the last layer
        // 3. Return token-level embeddings

        let seq_length = input_ids.shape()[0];
        let hidden_dim = self.config.model_dim as i32;

        // Create dummy embeddings for now
        let token_embeddings =
            mlx_rs::random::normal::<f32>(&[seq_length, hidden_dim], 0.0, 1.0, None)?;

        Ok(token_embeddings)
    }

    /// Count actual tokens (excluding padding)
    fn count_actual_tokens(&self, attention_mask: &Array) -> Result<usize> {
        // Sum the attention mask to get the number of actual tokens
        let sum = mlx_rs::ops::sum(attention_mask, None, None)?.item::<i32>();

        Ok(sum as usize)
    }

    /// Extract embeddings from multiple texts in batch
    pub async fn extract_batch_embeddings(&self, texts: &[String]) -> Result<Vec<EmbedResult>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let result = self.extract_embedding(text).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.config.model_name.clone(),
            dimension: self.config.model_dim,
            max_sequence_length: self.config.max_seq_length,
            loaded: self.model_loaded,
        }
    }
}

/// Information about the loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,

    /// Embedding dimension
    pub dimension: usize,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Whether the model is loaded
    pub loaded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_dim, 384);
        assert_eq!(config.max_seq_length, 512);
        assert!(config.use_mean_pooling);
    }

    #[tokio::test]
    async fn test_embedding_extractor_creation() {
        let config = EmbeddingConfig::default();
        let extractor = EmbeddingExtractor::new(config);

        let info = extractor.model_info();
        assert_eq!(info.dimension, 384);
        assert!(!info.loaded);
    }

    #[test]
    fn test_embed_result_creation() -> Result<()> {
        let token_embeddings = Array::zeros::<f32>(&[10, 384])?;
        let attention_mask = Array::ones::<i32>(&[10])?;
        let input_ids = Array::ones::<i32>(&[10])?;

        let result = EmbedResult::new(token_embeddings, attention_mask, input_ids, 8);

        assert_eq!(result.seq_length(), 10);
        assert_eq!(result.embedding_dim(), 384);
        assert_eq!(result.token_count, 8);

        Ok(())
    }

    #[tokio::test]
    async fn test_extract_embedding_without_model() {
        let config = EmbeddingConfig::default();
        let extractor = EmbeddingExtractor::new(config);

        let result = extractor.extract_embedding("test text").await;
        assert!(result.is_err());

        // Should fail because model is not loaded
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Model not loaded"));
    }
}
