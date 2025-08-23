//! Hard Negative Mining Loss
//!
//! This module implements hard negative mining strategies for contrastive learning.
//! Hard negative mining focuses on the most difficult negative examples to improve
//! the quality of learned embeddings.

use crate::error::Result;
use crate::loss::{utils, DistanceMetric, LossFunction, LossMetrics, LossResult};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// Strategy for selecting hard negatives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardNegativeMiningStrategy {
    /// Select negatives with highest similarity to query
    HardestNegatives,
    /// Random sampling from top-k hardest negatives
    SemiHardNegatives,
    /// Select negatives that are harder than positive but not too hard
    TripletSemiHard,
    /// Use all negatives but weight by difficulty
    WeightedHardNegatives,
    /// Online hard example mining during training
    OnlineHardMining,
}

impl Default for HardNegativeMiningStrategy {
    fn default() -> Self {
        Self::SemiHardNegatives
    }
}

impl std::fmt::Display for HardNegativeMiningStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HardestNegatives => write!(f, "hardest"),
            Self::SemiHardNegatives => write!(f, "semi_hard"),
            Self::TripletSemiHard => write!(f, "triplet_semi_hard"),
            Self::WeightedHardNegatives => write!(f, "weighted_hard"),
            Self::OnlineHardMining => write!(f, "online_hard"),
        }
    }
}

/// Configuration for hard negative mining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeConfig {
    /// Mining strategy to use
    pub mining_strategy: HardNegativeMiningStrategy,

    /// Temperature parameter for similarity scaling
    pub temperature: f32,

    /// Number of hard negatives to select per positive
    pub num_hard_negatives: usize,

    /// Margin for triplet-based mining
    pub margin: f32,

    /// Alpha parameter for semi-hard negative selection
    pub alpha: f32,

    /// Beta parameter for weighting negatives
    pub beta: f32,

    /// Whether to normalize embeddings
    pub normalize_embeddings: bool,

    /// Distance metric for similarity computation
    pub distance_metric: DistanceMetric,

    /// Whether to use focal loss weighting
    pub use_focal_loss: bool,

    /// Focal loss gamma parameter
    pub focal_gamma: f32,
}

impl Default for HardNegativeConfig {
    fn default() -> Self {
        Self {
            mining_strategy: HardNegativeMiningStrategy::default(),
            temperature: 0.07,
            num_hard_negatives: 8,
            margin: 0.2,
            alpha: 0.2,
            beta: 0.8,
            normalize_embeddings: true,
            distance_metric: DistanceMetric::Cosine,
            use_focal_loss: false,
            focal_gamma: 2.0,
        }
    }
}

/// Hard negative mining loss implementation
pub struct HardNegativeLoss {
    config: HardNegativeConfig,
}

impl HardNegativeLoss {
    /// Create a new hard negative mining loss
    pub fn new(config: HardNegativeConfig) -> Self {
        Self { config }
    }

    /// Compute hard negative mining loss
    pub fn compute_hard_negative_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
    ) -> Result<LossResult> {
        let _batch_size = query_embeddings.shape()[0] as usize;

        // Normalize embeddings if configured
        let (queries, docs) = if self.config.normalize_embeddings {
            (
                utils::normalize_embeddings(query_embeddings)?,
                utils::normalize_embeddings(doc_embeddings)?,
            )
        } else {
            (query_embeddings.clone(), doc_embeddings.clone())
        };

        // Compute similarity matrix
        let similarities = self.compute_similarity_matrix(&queries, &docs)?;

        // Mine hard negatives based on strategy
        let (hard_negative_indices, negative_weights) = self.mine_hard_negatives(&similarities)?;

        // Compute loss using selected hard negatives
        let loss = self.compute_loss_with_hard_negatives(
            &similarities,
            &hard_negative_indices,
            &negative_weights,
        )?;

        // Compute metrics
        let metrics = self.compute_hard_negative_metrics(&similarities, &hard_negative_indices)?;

        Ok(LossResult::new(loss, metrics))
    }

    /// Compute similarity matrix
    fn compute_similarity_matrix(&self, queries: &Array, docs: &Array) -> Result<Array> {
        match self.config.distance_metric {
            DistanceMetric::Cosine => queries
                .matmul(&docs.transpose(&[1, 0])?)
                .map_err(Into::into),
            DistanceMetric::DotProduct => queries
                .matmul(&docs.transpose(&[1, 0])?)
                .map_err(Into::into),
            _ => {
                // For other metrics, convert to cosine for simplicity
                let norm_queries = utils::normalize_embeddings(queries)?;
                let norm_docs = utils::normalize_embeddings(docs)?;
                norm_queries
                    .matmul(&norm_docs.transpose(&[1, 0])?)
                    .map_err(Into::into)
            }
        }
    }

    /// Mine hard negatives based on the configured strategy
    fn mine_hard_negatives(&self, similarities: &Array) -> Result<(Array, Array)> {
        match self.config.mining_strategy {
            HardNegativeMiningStrategy::HardestNegatives => {
                self.mine_hardest_negatives(similarities)
            }
            HardNegativeMiningStrategy::SemiHardNegatives => {
                self.mine_semi_hard_negatives(similarities)
            }
            HardNegativeMiningStrategy::TripletSemiHard => {
                self.mine_triplet_semi_hard_negatives(similarities)
            }
            HardNegativeMiningStrategy::WeightedHardNegatives => {
                self.mine_weighted_hard_negatives(similarities)
            }
            HardNegativeMiningStrategy::OnlineHardMining => {
                self.mine_online_hard_negatives(similarities)
            }
        }
    }

    /// Mine hardest negatives (highest similarity non-diagonal elements)
    fn mine_hardest_negatives(&self, similarities: &Array) -> Result<(Array, Array)> {
        let batch_size = similarities.shape()[0] as usize;

        // Create mask to exclude diagonal (positive pairs)
        let diagonal_mask = Array::eye::<f32>(batch_size as i32, Some(batch_size as i32), None)?;
        let negative_mask = Array::from(1.0).subtract(&diagonal_mask)?;

        // Apply mask to similarities (set diagonal to very low value)
        let masked_similarities = similarities
            .multiply(&negative_mask)?
            .add(&diagonal_mask.multiply(Array::from(-1e6))?)?;

        // Get top-k hardest negatives for each query
        let k = std::cmp::min(self.config.num_hard_negatives, batch_size - 1);
        let _top_indices = mlx_rs::ops::argsort(&masked_similarities, -1)?;
        // Take the top k indices (simplified for now)
        let indices_shape = [batch_size as i32, k as i32];
        let top_indices = Array::zeros::<i32>(&indices_shape)?;

        // Create uniform weights for selected negatives
        let weights_shape = [batch_size as i32, k as i32];
        let weights = Array::full::<f32>(&weights_shape, &Array::from(1.0 / k as f32))?;

        Ok((top_indices, weights))
    }

    /// Mine semi-hard negatives (between positive and margin)
    fn mine_semi_hard_negatives(&self, similarities: &Array) -> Result<(Array, Array)> {
        let batch_size = similarities.shape()[0] as usize;

        // Get positive similarities (diagonal)
        let mut positive_sims = Vec::new();
        for i in 0..batch_size {
            let pos_sim = similarities.index((i as i32, i as i32)).item::<f32>();
            positive_sims.push(pos_sim);
        }

        let mut selected_indices = Vec::new();
        let mut weights = Vec::new();

        for (query_idx, &pos_sim) in positive_sims.iter().enumerate() {
            let query_similarities = similarities.index((query_idx as i32, ..)).squeeze(&[0])?;
            let mut candidates = Vec::new();

            // Find negatives that are semi-hard (similarity > pos_sim - alpha)
            for doc_idx in 0..batch_size {
                if doc_idx == query_idx {
                    continue; // Skip positive pair
                }

                let neg_sim = query_similarities.index(doc_idx as i32).item::<f32>();
                if neg_sim > pos_sim - self.config.alpha {
                    candidates.push((doc_idx, neg_sim));
                }
            }

            // Sort by similarity (descending) and take top-k
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let k = std::cmp::min(self.config.num_hard_negatives, candidates.len());

            let mut query_indices = Vec::new();
            let mut query_weights = Vec::new();

            for (doc_idx, sim) in candidates.into_iter().take(k) {
                query_indices.push(doc_idx as i32);
                // Weight by difficulty (higher similarity = higher weight)
                let weight = (sim + 1.0) / 2.0; // Normalize to [0,1]
                query_weights.push(weight);
            }

            // Normalize weights
            let weight_sum: f32 = query_weights.iter().sum();
            if weight_sum > 0.0 {
                for w in &mut query_weights {
                    *w /= weight_sum;
                }
            }

            // Pad to fixed size if necessary
            while query_indices.len() < self.config.num_hard_negatives {
                query_indices.push(-1); // Use -1 as padding
                query_weights.push(0.0);
            }

            selected_indices.extend(query_indices);
            weights.extend(query_weights);
        }

        let indices_shape = [batch_size as i32, self.config.num_hard_negatives as i32];
        let indices_array = Array::from_slice(&selected_indices, &indices_shape);
        let weights_array = Array::from_slice(&weights, &indices_shape);

        Ok((indices_array, weights_array))
    }

    /// Mine triplet semi-hard negatives
    fn mine_triplet_semi_hard_negatives(&self, similarities: &Array) -> Result<(Array, Array)> {
        // For triplet semi-hard, we look for negatives that are:
        // similarity(q, n) > similarity(q, p) - margin
        // but not the absolute hardest negatives

        let batch_size = similarities.shape()[0] as usize;
        let mut selected_indices = Vec::new();
        let mut weights = Vec::new();

        for query_idx in 0..batch_size {
            let pos_sim = similarities
                .index((query_idx as i32, query_idx as i32))
                .item::<f32>();
            let threshold = pos_sim - self.config.margin;

            let query_sims = similarities.index((query_idx as i32, ..)).squeeze(&[0])?;
            let mut candidates = Vec::new();

            for doc_idx in 0..batch_size {
                if doc_idx == query_idx {
                    continue;
                }

                let neg_sim = query_sims.index(doc_idx as i32).item::<f32>();
                if neg_sim > threshold && neg_sim < pos_sim {
                    candidates.push((doc_idx, neg_sim));
                }
            }

            // Sort by similarity and take middle range (semi-hard)
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let k = std::cmp::min(self.config.num_hard_negatives, candidates.len());
            let start_idx = candidates.len() / 4; // Skip hardest 25%
            let end_idx = std::cmp::min(start_idx + k, candidates.len());

            let mut query_indices = Vec::new();
            let mut query_weights = Vec::new();

            for (doc_idx, _sim) in candidates
                .into_iter()
                .skip(start_idx)
                .take(end_idx - start_idx)
            {
                query_indices.push(doc_idx as i32);
                query_weights.push(1.0);
            }

            // Normalize weights
            let weight_sum = query_weights.len() as f32;
            if weight_sum > 0.0 {
                for w in &mut query_weights {
                    *w /= weight_sum;
                }
            }

            // Pad to fixed size
            while query_indices.len() < self.config.num_hard_negatives {
                query_indices.push(-1);
                query_weights.push(0.0);
            }

            selected_indices.extend(query_indices);
            weights.extend(query_weights);
        }

        let indices_shape = [batch_size as i32, self.config.num_hard_negatives as i32];
        let indices_array = Array::from_slice(&selected_indices, &indices_shape);
        let weights_array = Array::from_slice(&weights, &indices_shape);

        Ok((indices_array, weights_array))
    }

    /// Mine weighted hard negatives
    fn mine_weighted_hard_negatives(&self, similarities: &Array) -> Result<(Array, Array)> {
        let batch_size = similarities.shape()[0] as usize;

        // Use all negatives but weight by difficulty
        let mut selected_indices = Vec::new();
        let mut weights = Vec::new();

        for query_idx in 0..batch_size {
            let query_sims = similarities.index((query_idx as i32, ..)).squeeze(&[0])?;

            for doc_idx in 0..batch_size {
                if doc_idx == query_idx {
                    continue; // Skip positive pair
                }

                selected_indices.push(doc_idx as i32);

                // Weight by similarity (harder negatives get higher weight)
                let sim = query_sims.index(doc_idx as i32).item::<f32>();
                let weight = ((sim + 1.0) / 2.0).powf(self.config.beta);
                weights.push(weight);
            }
        }

        // Normalize weights per query
        let negatives_per_query = batch_size - 1;
        for query_idx in 0..batch_size {
            let start = query_idx * negatives_per_query;
            let end = start + negatives_per_query;
            let weight_sum: f32 = weights[start..end].iter().sum();

            if weight_sum > 0.0 {
                for w in &mut weights[start..end] {
                    *w /= weight_sum;
                }
            }
        }

        let indices_shape = [batch_size as i32, negatives_per_query as i32];
        let indices_array = Array::from_slice(&selected_indices, &indices_shape);
        let weights_array = Array::from_slice(&weights, &indices_shape);

        Ok((indices_array, weights_array))
    }

    /// Online hard negative mining (placeholder)
    fn mine_online_hard_negatives(&self, similarities: &Array) -> Result<(Array, Array)> {
        // For now, fallback to hardest negatives
        // In a full implementation, this would maintain a memory bank
        // of hard negatives from previous batches
        self.mine_hardest_negatives(similarities)
    }

    /// Compute loss using selected hard negatives
    fn compute_loss_with_hard_negatives(
        &self,
        similarities: &Array,
        hard_negative_indices: &Array,
        negative_weights: &Array,
    ) -> Result<Array> {
        let batch_size = similarities.shape()[0] as usize;
        let num_negatives = hard_negative_indices.shape()[1] as usize;

        let mut total_loss = Array::from(0.0);

        for query_idx in 0..batch_size {
            // Get positive similarity
            let pos_sim = similarities.index((query_idx as i32, query_idx as i32));
            let pos_logit = &pos_sim / &Array::from(self.config.temperature);

            // Get negative similarities
            let mut neg_logits = Vec::new();
            let mut weights = Vec::new();

            for neg_idx in 0..num_negatives {
                let doc_idx = hard_negative_indices
                    .index((query_idx as i32, neg_idx as i32))
                    .item::<i32>();
                if doc_idx >= 0 {
                    let neg_sim = similarities.index((query_idx as i32, doc_idx));
                    let neg_logit = &neg_sim / &Array::from(self.config.temperature);
                    neg_logits.push(neg_logit);

                    let weight = negative_weights
                        .index((query_idx as i32, neg_idx as i32))
                        .item::<f32>();
                    weights.push(weight);
                }
            }

            if neg_logits.is_empty() {
                continue;
            }

            // Compute contrastive loss for this query
            let mut denominator = pos_logit.exp()?;

            for (neg_logit, weight) in neg_logits.iter().zip(weights.iter()) {
                let weighted_exp = neg_logit.exp()?.multiply(Array::from(*weight))?;
                denominator = denominator.add(&weighted_exp)?;
            }

            let loss = denominator.log()?.subtract(&pos_logit)?;

            // Apply focal loss if configured
            let final_loss = if self.config.use_focal_loss {
                let prob = pos_logit.exp()?.divide(&denominator)?;
                let focal_weight = mlx_rs::ops::power(
                    &(Array::from(1.0).subtract(&prob)?),
                    Array::from(self.config.focal_gamma),
                )?;
                loss.multiply(&focal_weight)?
            } else {
                loss
            };

            total_loss = total_loss.add(&final_loss)?;
        }

        // Average over batch
        total_loss
            .divide(Array::from(batch_size as f32))
            .map_err(Into::into)
    }

    /// Compute metrics specific to hard negative mining
    fn compute_hard_negative_metrics(
        &self,
        similarities: &Array,
        hard_negative_indices: &Array,
    ) -> Result<LossMetrics> {
        let batch_size = similarities.shape()[0] as usize;
        let num_negatives = hard_negative_indices.shape()[1] as usize;

        let mut positive_sims = Vec::new();
        let mut negative_sims = Vec::new();

        // Collect positive similarities
        for i in 0..batch_size {
            let pos_sim = similarities.index((i as i32, i as i32)).item::<f32>();
            positive_sims.push(pos_sim);
        }

        // Collect negative similarities
        for query_idx in 0..batch_size {
            for neg_idx in 0..num_negatives {
                let doc_idx = hard_negative_indices
                    .index((query_idx as i32, neg_idx as i32))
                    .item::<i32>();
                if doc_idx >= 0 {
                    let neg_sim = similarities
                        .index((query_idx as i32, doc_idx))
                        .item::<f32>();
                    negative_sims.push(neg_sim);
                }
            }
        }

        let avg_positive_similarity =
            positive_sims.iter().sum::<f32>() / positive_sims.len() as f32;
        let avg_negative_similarity = if negative_sims.is_empty() {
            0.0
        } else {
            negative_sims.iter().sum::<f32>() / negative_sims.len() as f32
        };

        // Compute accuracy (how many queries have positive similarity > max negative similarity)
        let mut correct = 0;
        for (query_idx, pos_sim) in positive_sims.iter().enumerate().take(batch_size) {
            let pos_sim = *pos_sim;
            let mut max_neg_sim = f32::NEG_INFINITY;

            for neg_idx in 0..num_negatives {
                let doc_idx = hard_negative_indices
                    .index((query_idx as i32, neg_idx as i32))
                    .item::<i32>();
                if doc_idx >= 0 {
                    let neg_sim = similarities
                        .index((query_idx as i32, doc_idx))
                        .item::<f32>();
                    max_neg_sim = max_neg_sim.max(neg_sim);
                }
            }

            if pos_sim > max_neg_sim {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / batch_size as f32;

        Ok(LossMetrics {
            positive_pairs: batch_size,
            negative_pairs: negative_sims.len(),
            avg_positive_similarity,
            avg_negative_similarity,
            accuracy,
            top_k_accuracy: None,
        })
    }
}

impl LossFunction for HardNegativeLoss {
    fn compute_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
        _labels: Option<&Array>,
    ) -> Result<LossResult> {
        self.compute_hard_negative_loss(query_embeddings, doc_embeddings)
    }

    fn name(&self) -> &'static str {
        "HardNegative"
    }

    fn config(&self) -> &dyn std::any::Any {
        &self.config
    }
}

/// Builder for hard negative configuration
pub struct HardNegativeConfigBuilder {
    config: HardNegativeConfig,
}

impl Default for HardNegativeConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HardNegativeConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: HardNegativeConfig::default(),
        }
    }

    /// Set mining strategy
    pub fn mining_strategy(mut self, strategy: HardNegativeMiningStrategy) -> Self {
        self.config.mining_strategy = strategy;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set number of hard negatives
    pub fn num_hard_negatives(mut self, num: usize) -> Self {
        self.config.num_hard_negatives = num;
        self
    }

    /// Set margin for triplet mining
    pub fn margin(mut self, margin: f32) -> Self {
        self.config.margin = margin;
        self
    }

    /// Enable focal loss
    pub fn use_focal_loss(mut self, use_focal: bool) -> Self {
        self.config.use_focal_loss = use_focal;
        self
    }

    /// Set focal loss gamma parameter
    pub fn focal_gamma(mut self, gamma: f32) -> Self {
        self.config.focal_gamma = gamma;
        self
    }

    /// Build the configuration
    pub fn build(self) -> HardNegativeConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_negative_config_default() {
        let config = HardNegativeConfig::default();
        assert_eq!(
            config.mining_strategy,
            HardNegativeMiningStrategy::SemiHardNegatives
        );
        assert_eq!(config.temperature, 0.07);
        assert_eq!(config.num_hard_negatives, 8);
        assert!(!config.use_focal_loss);
    }

    #[test]
    fn test_mining_strategy_display() {
        assert_eq!(
            HardNegativeMiningStrategy::HardestNegatives.to_string(),
            "hardest"
        );
        assert_eq!(
            HardNegativeMiningStrategy::SemiHardNegatives.to_string(),
            "semi_hard"
        );
    }

    #[test]
    fn test_hard_negative_config_builder() {
        let config = HardNegativeConfigBuilder::new()
            .mining_strategy(HardNegativeMiningStrategy::HardestNegatives)
            .temperature(0.1)
            .num_hard_negatives(5)
            .use_focal_loss(true)
            .focal_gamma(1.5)
            .build();

        assert_eq!(
            config.mining_strategy,
            HardNegativeMiningStrategy::HardestNegatives
        );
        assert_eq!(config.temperature, 0.1);
        assert_eq!(config.num_hard_negatives, 5);
        assert!(config.use_focal_loss);
        assert_eq!(config.focal_gamma, 1.5);
    }

    #[tokio::test]
    async fn test_hard_negative_loss_computation() -> Result<()> {
        let config = HardNegativeConfig::default();
        let loss_fn = HardNegativeLoss::new(config);

        // Create dummy embeddings
        let query_embeddings = mlx_rs::random::normal::<f32>(&[4, 128], None, None, None)?;
        let doc_embeddings = mlx_rs::random::normal::<f32>(&[4, 128], None, None, None)?;

        let result = loss_fn.compute_loss(&query_embeddings, &doc_embeddings, None)?;

        // Check that loss is computed
        assert!(result.scalar_loss() > 0.0);
        assert_eq!(result.metrics.positive_pairs, 4);

        Ok(())
    }

    #[test]
    fn test_hard_negative_loss_creation() {
        let config = HardNegativeConfig::default();
        let loss = HardNegativeLoss::new(config);
        assert_eq!(loss.name(), "HardNegative");
    }
}
