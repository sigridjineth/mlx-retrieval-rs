//! Loss functions for training embedding models
//!
//! This module provides various loss functions commonly used in training
//! retrieval and embedding models, including contrastive learning approaches.

pub mod hard_negative;
pub mod infonce;
pub mod ntxent;

// Re-export common types
pub use hard_negative::{HardNegativeConfig, HardNegativeLoss, HardNegativeMiningStrategy};
pub use infonce::{InfoNceConfig, InfoNceLoss};
pub use ntxent::{NtXentConfig, NtXentLoss};

use crate::error::Result;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// Configuration for loss computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Type of loss function to use
    pub loss_type: LossType,

    /// Temperature parameter for similarity scaling
    pub temperature: f32,

    /// Whether to normalize embeddings before computing similarity
    pub normalize_embeddings: bool,

    /// Loss-specific configuration
    pub specific_config: LossSpecificConfig,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            loss_type: LossType::InfoNCE,
            temperature: 0.07,
            normalize_embeddings: true,
            specific_config: LossSpecificConfig::InfoNCE(InfoNceConfig::default()),
        }
    }
}

/// Available loss function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    /// InfoNCE (Information Noise Contrastive Estimation)
    InfoNCE,
    /// NT-Xent (Normalized Temperature-scaled Cross Entropy)
    NTXent,
    /// Hard Negative Mining with contrastive loss
    HardNegative,
    /// Multiple Negatives Ranking Loss
    MultipleNegatives,
    /// Triplet Loss
    Triplet,
}

impl std::fmt::Display for LossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossType::InfoNCE => write!(f, "infonce"),
            LossType::NTXent => write!(f, "ntxent"),
            LossType::HardNegative => write!(f, "hard_negative"),
            LossType::MultipleNegatives => write!(f, "multiple_negatives"),
            LossType::Triplet => write!(f, "triplet"),
        }
    }
}

/// Loss-specific configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LossSpecificConfig {
    InfoNCE(InfoNceConfig),
    NTXent(NtXentConfig),
    HardNegative(HardNegativeConfig),
    MultipleNegatives(MultipleNegativesConfig),
    Triplet(TripletConfig),
}

/// Configuration for Multiple Negatives Ranking Loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleNegativesConfig {
    /// Scale factor for the loss
    pub scale: f32,

    /// Whether to use in-batch negatives
    pub use_in_batch_negatives: bool,
}

impl Default for MultipleNegativesConfig {
    fn default() -> Self {
        Self {
            scale: 20.0,
            use_in_batch_negatives: true,
        }
    }
}

/// Configuration for Triplet Loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletConfig {
    /// Margin for triplet loss
    pub margin: f32,

    /// Distance metric to use
    pub distance_metric: DistanceMetric,

    /// Whether to use hard triplet mining
    pub use_hard_mining: bool,
}

impl Default for TripletConfig {
    fn default() -> Self {
        Self {
            margin: 0.2,
            distance_metric: DistanceMetric::Cosine,
            use_hard_mining: false,
        }
    }
}

/// Distance metrics for similarity computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    DotProduct,
    /// Manhattan distance
    Manhattan,
}

/// Result of a loss computation
#[derive(Debug, Clone)]
pub struct LossResult {
    /// The computed loss value
    pub loss: Array,

    /// Additional metrics computed during loss calculation
    pub metrics: LossMetrics,

    /// Gradients (if computed)
    pub gradients: Option<Array>,
}

/// Additional metrics from loss computation
#[derive(Debug, Clone, Default)]
pub struct LossMetrics {
    /// Number of positive pairs
    pub positive_pairs: usize,

    /// Number of negative pairs
    pub negative_pairs: usize,

    /// Average positive similarity
    pub avg_positive_similarity: f32,

    /// Average negative similarity
    pub avg_negative_similarity: f32,

    /// Accuracy (fraction of correct predictions)
    pub accuracy: f32,

    /// Top-k accuracy (if applicable)
    pub top_k_accuracy: Option<f32>,
}

impl LossResult {
    /// Create a new loss result
    pub fn new(loss: Array, metrics: LossMetrics) -> Self {
        Self {
            loss,
            metrics,
            gradients: None,
        }
    }

    /// Add gradients to the result
    pub fn with_gradients(mut self, gradients: Array) -> Self {
        self.gradients = Some(gradients);
        self
    }

    /// Get the scalar loss value
    pub fn scalar_loss(&self) -> f32 {
        self.loss.item::<f32>()
    }
}

/// Main loss function trait
pub trait LossFunction {
    /// Compute loss given query and document embeddings
    fn compute_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
        labels: Option<&Array>,
    ) -> Result<LossResult>;

    /// Get the loss function name
    fn name(&self) -> &'static str;

    /// Get loss-specific configuration
    fn config(&self) -> &dyn std::any::Any;
}

/// Factory for creating loss functions
pub struct LossFactory;

impl LossFactory {
    /// Create a loss function based on configuration
    pub fn create_loss_function(
        config: &LossConfig,
    ) -> Result<Box<dyn LossFunction + Send + Sync>> {
        match config.loss_type {
            LossType::InfoNCE => {
                let infonce_config = match &config.specific_config {
                    LossSpecificConfig::InfoNCE(cfg) => cfg.clone(),
                    _ => InfoNceConfig::default(),
                };
                Ok(Box::new(InfoNceLoss::new(infonce_config)))
            }
            LossType::NTXent => {
                let ntxent_config = match &config.specific_config {
                    LossSpecificConfig::NTXent(cfg) => cfg.clone(),
                    _ => NtXentConfig::default(),
                };
                Ok(Box::new(NtXentLoss::new(ntxent_config)))
            }
            LossType::HardNegative => {
                let hard_neg_config = match &config.specific_config {
                    LossSpecificConfig::HardNegative(cfg) => cfg.clone(),
                    _ => HardNegativeConfig::default(),
                };
                Ok(Box::new(HardNegativeLoss::new(hard_neg_config)))
            }
            LossType::MultipleNegatives => {
                let mn_config = match &config.specific_config {
                    LossSpecificConfig::MultipleNegatives(cfg) => cfg.clone(),
                    _ => MultipleNegativesConfig::default(),
                };
                Ok(Box::new(MultipleNegativesLoss::new(mn_config)))
            }
            LossType::Triplet => {
                let triplet_config = match &config.specific_config {
                    LossSpecificConfig::Triplet(cfg) => cfg.clone(),
                    _ => TripletConfig::default(),
                };
                Ok(Box::new(TripletLoss::new(triplet_config)))
            }
        }
    }
}

/// Multiple Negatives Ranking Loss implementation
pub struct MultipleNegativesLoss {
    config: MultipleNegativesConfig,
}

impl MultipleNegativesLoss {
    /// Create a new Multiple Negatives loss
    pub fn new(config: MultipleNegativesConfig) -> Self {
        Self { config }
    }
}

impl LossFunction for MultipleNegativesLoss {
    fn compute_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
        _labels: Option<&Array>,
    ) -> Result<LossResult> {
        // Compute similarity matrix
        let similarities =
            compute_similarity_matrix(query_embeddings, doc_embeddings, DistanceMetric::Cosine)?;
        let scaled_similarities = similarities.multiply(Array::from(self.config.scale))?;

        // Create labels (diagonal matrix for positive pairs)
        let batch_size = query_embeddings.shape()[0];
        let labels = Array::eye::<f32>(batch_size, None, None)?;

        // Compute cross-entropy loss
        let loss = cross_entropy_loss(&scaled_similarities, &labels)?;

        // Compute metrics
        let metrics = compute_ranking_metrics(&similarities, &labels)?;

        Ok(LossResult::new(loss, metrics))
    }

    fn name(&self) -> &'static str {
        "MultipleNegatives"
    }

    fn config(&self) -> &dyn std::any::Any {
        &self.config
    }
}

/// Triplet Loss implementation
pub struct TripletLoss {
    config: TripletConfig,
}

impl TripletLoss {
    /// Create a new Triplet loss
    pub fn new(config: TripletConfig) -> Self {
        Self { config }
    }
}

impl LossFunction for TripletLoss {
    fn compute_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
        _labels: Option<&Array>,
    ) -> Result<LossResult> {
        let batch_size = query_embeddings.shape()[0] as usize;

        // For triplet loss, we expect pairs of positive and negative documents
        if doc_embeddings.shape()[0] != (batch_size * 2) as i32 {
            return Err(crate::error::MlxRetrievalError::invalid_input(
                "For triplet loss, expect 2 documents per query (positive and negative)",
            ));
        }

        // Split positive and negative embeddings
        let pos_embeddings = doc_embeddings.index((0..(batch_size as i32),));
        let neg_embeddings = doc_embeddings.index(((batch_size as i32)..,));

        // Compute distances
        let pos_distances = compute_distance(
            query_embeddings,
            &pos_embeddings,
            self.config.distance_metric,
        )?;
        let neg_distances = compute_distance(
            query_embeddings,
            &neg_embeddings,
            self.config.distance_metric,
        )?;

        // Triplet loss: max(0, margin + d(a,p) - d(a,n))
        let margin = Array::from(self.config.margin);
        let loss_per_sample = ops::maximum(
            &(pos_distances.clone() - neg_distances.clone() + margin),
            Array::from(0.0),
        )?;
        let loss = ops::mean(&loss_per_sample, None, None)?;

        // Compute metrics
        let metrics = compute_triplet_metrics(&pos_distances, &neg_distances, self.config.margin)?;

        Ok(LossResult::new(loss, metrics))
    }

    fn name(&self) -> &'static str {
        "Triplet"
    }

    fn config(&self) -> &dyn std::any::Any {
        &self.config
    }
}

/// Utility functions for loss computation
pub mod utils {
    use super::*;

    /// Normalize embeddings to unit length
    pub fn normalize_embeddings(embeddings: &Array) -> Result<Array> {
        let norms = ops::sum(&embeddings.square()?, &[-1][..], true)?.sqrt()?;
        let epsilon = Array::from(1e-8);
        let safe_norms = ops::maximum(&norms, &epsilon)?;
        embeddings.divide(&safe_norms).map_err(Into::into)
    }

    /// Compute temperature-scaled similarities
    pub fn temperature_scaled_similarity(
        embeddings1: &Array,
        embeddings2: &Array,
        temperature: f32,
    ) -> Result<Array> {
        let similarities = embeddings1.matmul(&embeddings2.transpose(&[-2, -1])?)?;
        similarities
            .divide(Array::from(temperature))
            .map_err(Into::into)
    }
}

// Helper functions
fn compute_similarity_matrix(
    embeddings1: &Array,
    embeddings2: &Array,
    metric: DistanceMetric,
) -> Result<Array> {
    match metric {
        DistanceMetric::Cosine => {
            let norm1 = utils::normalize_embeddings(embeddings1)?;
            let norm2 = utils::normalize_embeddings(embeddings2)?;
            norm1
                .matmul(&norm2.transpose(&[-2, -1])?)
                .map_err(Into::into)
        }
        DistanceMetric::DotProduct => embeddings1
            .matmul(&embeddings2.transpose(&[-2, -1])?)
            .map_err(Into::into),
        _ => {
            // For other metrics, implement as needed
            embeddings1
                .matmul(&embeddings2.transpose(&[-2, -1])?)
                .map_err(Into::into)
        }
    }
}

fn compute_distance(
    embeddings1: &Array,
    embeddings2: &Array,
    metric: DistanceMetric,
) -> Result<Array> {
    match metric {
        DistanceMetric::Cosine => {
            let similarities = compute_similarity_matrix(embeddings1, embeddings2, metric)?;
            Array::from(1.0).subtract(&similarities).map_err(Into::into)
        }
        DistanceMetric::Euclidean => {
            let diff = embeddings1.subtract(embeddings2)?;
            ops::sum(&diff.square()?, &[-1][..], false)?
                .sqrt()
                .map_err(Into::into)
        }
        _ => {
            // Fallback to euclidean
            let diff = embeddings1.subtract(embeddings2)?;
            ops::sum(&diff.square()?, &[-1][..], false)?
                .sqrt()
                .map_err(Into::into)
        }
    }
}

fn cross_entropy_loss(logits: &Array, labels: &Array) -> Result<Array> {
    // Simplified cross-entropy implementation
    let softmax = softmax(logits)?;
    let log_softmax = softmax.log()?;
    let loss = ops::sum(&labels.multiply(&log_softmax)?, &[-1][..], false)?;
    let neg_loss = &loss * Array::from(-1.0);
    ops::mean(&neg_loss, None, None).map_err(Into::into)
}

fn softmax(logits: &Array) -> Result<Array> {
    let max_logits = ops::max(logits, &[-1][..], true)?;
    let shifted = logits.subtract(&max_logits)?;
    let exp_logits = shifted.exp()?;
    let sum_exp = ops::sum(&exp_logits, &[-1][..], true)?;
    exp_logits.divide(&sum_exp).map_err(Into::into)
}

fn compute_ranking_metrics(_similarities: &Array, _labels: &Array) -> Result<LossMetrics> {
    // Simplified metrics computation
    Ok(LossMetrics {
        positive_pairs: 1,
        negative_pairs: 1,
        avg_positive_similarity: 0.0,
        avg_negative_similarity: 0.0,
        accuracy: 0.0,
        top_k_accuracy: None,
    })
}

fn compute_triplet_metrics(
    pos_distances: &Array,
    neg_distances: &Array,
    _margin: f32,
) -> Result<LossMetrics> {
    // Simplified metrics computation
    Ok(LossMetrics {
        positive_pairs: pos_distances.size(),
        negative_pairs: neg_distances.size(),
        avg_positive_similarity: 0.0,
        avg_negative_similarity: 0.0,
        accuracy: 0.0,
        top_k_accuracy: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_config_default() {
        let config = LossConfig::default();
        assert_eq!(config.loss_type, LossType::InfoNCE);
        assert_eq!(config.temperature, 0.07);
        assert!(config.normalize_embeddings);
    }

    #[test]
    fn test_loss_type_display() {
        assert_eq!(LossType::InfoNCE.to_string(), "infonce");
        assert_eq!(LossType::NTXent.to_string(), "ntxent");
        assert_eq!(LossType::Triplet.to_string(), "triplet");
    }

    #[test]
    fn test_loss_result_creation() -> Result<()> {
        let loss = Array::from(0.5);
        let metrics = LossMetrics::default();
        let result = LossResult::new(loss, metrics);

        assert_eq!(result.scalar_loss(), 0.5);
        assert!(result.gradients.is_none());

        Ok(())
    }

    #[test]
    fn test_normalize_embeddings() -> Result<()> {
        let embeddings = Array::from_slice(&[3.0, 4.0], &[1, 2]);
        let normalized = utils::normalize_embeddings(&embeddings)?;

        // Should have unit norm: [3,4] -> [0.6, 0.8]
        assert_eq!(normalized.shape(), vec![1, 2]);

        Ok(())
    }
}
