//! NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
//!
//! This module implements the NT-Xent loss function, which is commonly used in
//! self-supervised contrastive learning methods like SimCLR. It normalizes the
//! temperature-scaled cross entropy loss.

use crate::error::Result;
use crate::loss::{utils, DistanceMetric, LossFunction, LossMetrics, LossResult};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// Configuration for NT-Xent loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtXentConfig {
    /// Temperature parameter for scaling similarities
    pub temperature: f32,

    /// Whether to normalize embeddings before computing similarity
    pub normalize_embeddings: bool,

    /// Distance metric for computing similarities
    pub distance_metric: DistanceMetric,

    /// Whether to use symmetric loss (both directions)
    pub symmetric: bool,

    /// Whether to exclude self-similarity in negative sampling
    pub exclude_self: bool,

    /// Base for logarithm (2.0 for log2, e for ln)
    pub log_base: f32,

    /// Whether to use cosine annealing for temperature
    pub use_temperature_annealing: bool,

    /// Initial temperature for annealing
    pub initial_temperature: f32,

    /// Final temperature for annealing
    pub final_temperature: f32,

    /// Current training step (for annealing)
    pub current_step: usize,

    /// Total training steps (for annealing)
    pub total_steps: usize,
}

impl Default for NtXentConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            normalize_embeddings: true,
            distance_metric: DistanceMetric::Cosine,
            symmetric: true,
            exclude_self: true,
            log_base: std::f32::consts::E,
            use_temperature_annealing: false,
            initial_temperature: 0.5,
            final_temperature: 0.05,
            current_step: 0,
            total_steps: 10000,
        }
    }
}

impl NtXentConfig {
    /// Get the current temperature (considering annealing if enabled)
    pub fn current_temperature(&self) -> f32 {
        if self.use_temperature_annealing && self.total_steps > 0 {
            let progress = (self.current_step as f32) / (self.total_steps as f32);
            let progress = progress.min(1.0);

            // Cosine annealing
            let cos_progress = (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0;
            self.final_temperature
                + (self.initial_temperature - self.final_temperature) * cos_progress
        } else {
            self.temperature
        }
    }

    /// Update the current training step (for temperature annealing)
    pub fn update_step(&mut self, step: usize) {
        self.current_step = step;
    }
}

/// NT-Xent loss implementation
pub struct NtXentLoss {
    config: NtXentConfig,
}

impl NtXentLoss {
    /// Create a new NT-Xent loss function
    pub fn new(config: NtXentConfig) -> Self {
        Self { config }
    }

    /// Compute NT-Xent loss for contrastive learning
    pub fn compute_ntxent_loss(
        &self,
        embeddings1: &Array,
        embeddings2: &Array,
    ) -> Result<LossResult> {
        let batch_size = embeddings1.shape()[0] as usize;

        // Normalize embeddings if configured
        let (emb1, emb2) = if self.config.normalize_embeddings {
            (
                utils::normalize_embeddings(embeddings1)?,
                utils::normalize_embeddings(embeddings2)?,
            )
        } else {
            (embeddings1.clone(), embeddings2.clone())
        };

        // Concatenate embeddings for full similarity matrix
        let all_embeddings = mlx_rs::ops::concatenate(&[&emb1, &emb2], Some(0))?;

        // Compute similarity matrix
        let similarity_matrix = self.compute_similarity_matrix(&all_embeddings)?;

        // Get current temperature
        let temperature = self.config.current_temperature();
        let scaled_similarity = similarity_matrix.divide(Array::from(temperature))?;

        // Create positive pair mask
        let positive_mask = self.create_positive_mask(batch_size)?;

        // Compute NT-Xent loss
        let loss = if self.config.symmetric {
            // Compute loss in both directions and average
            let loss_12 =
                self.compute_directional_loss(&scaled_similarity, &positive_mask, batch_size)?;
            let loss_21 = self.compute_directional_loss(
                &scaled_similarity.transpose(&[1, 0])?,
                &positive_mask.transpose(&[1, 0])?,
                batch_size,
            )?;
            loss_12.add(&loss_21)?.divide(Array::from(2.0))?
        } else {
            self.compute_directional_loss(&scaled_similarity, &positive_mask, batch_size)?
        };

        // Compute metrics
        let metrics =
            self.compute_ntxent_metrics(&similarity_matrix, &positive_mask, batch_size)?;

        Ok(LossResult::new(loss, metrics))
    }

    /// Compute similarity matrix
    fn compute_similarity_matrix(&self, embeddings: &Array) -> Result<Array> {
        match self.config.distance_metric {
            DistanceMetric::Cosine => embeddings
                .matmul(&embeddings.transpose(&[1, 0])?)
                .map_err(Into::into),
            DistanceMetric::DotProduct => embeddings
                .matmul(&embeddings.transpose(&[1, 0])?)
                .map_err(Into::into),
            DistanceMetric::Euclidean => self.compute_euclidean_similarity_matrix(embeddings),
            DistanceMetric::Manhattan => self.compute_manhattan_similarity_matrix(embeddings),
        }
    }

    /// Compute Euclidean distance-based similarity matrix
    fn compute_euclidean_similarity_matrix(&self, embeddings: &Array) -> Result<Array> {
        let _n = embeddings.shape()[0];

        // Compute pairwise squared distances
        let embeddings_expanded_1 = embeddings.expand_dims(&[1])?; // (n, 1, d)
        let embeddings_expanded_2 = embeddings.expand_dims(&[0])?; // (1, n, d)

        let diff = embeddings_expanded_1.subtract(&embeddings_expanded_2)?;
        let squared_distances = diff.square()?.sum(&[-1], false)?;

        // Convert to similarity (negative distance)
        Ok(&squared_distances * Array::from(-1.0))
    }

    /// Compute Manhattan distance-based similarity matrix
    fn compute_manhattan_similarity_matrix(&self, embeddings: &Array) -> Result<Array> {
        let embeddings_expanded_1 = embeddings.expand_dims(&[1])?;
        let embeddings_expanded_2 = embeddings.expand_dims(&[0])?;

        let diff = embeddings_expanded_1.subtract(&embeddings_expanded_2)?;
        let manhattan_distances = diff.abs()?.sum(&[-1], false)?;

        // Convert to similarity (negative distance)
        Ok(&manhattan_distances * Array::from(-1.0))
    }

    /// Create positive pair mask for NT-Xent
    fn create_positive_mask(&self, batch_size: usize) -> Result<Array> {
        let total_size = batch_size * 2;

        // Create mask matrix using array construction
        // For NT-Xent, positive pairs are (i, i+batch_size) and (i+batch_size, i)
        let mut mask_data = vec![0.0f32; total_size * total_size];

        for i in 0..batch_size {
            let i1 = i;
            let i2 = i + batch_size;

            // Set (i1, i2) = 1.0
            mask_data[i1 * total_size + i2] = 1.0;
            // Set (i2, i1) = 1.0
            mask_data[i2 * total_size + i1] = 1.0;
        }

        let mask = Array::from_slice(&mask_data, &[total_size as i32, total_size as i32]);

        Ok(mask)
    }

    /// Compute directional NT-Xent loss
    fn compute_directional_loss(
        &self,
        scaled_similarity: &Array,
        positive_mask: &Array,
        batch_size: usize,
    ) -> Result<Array> {
        let total_size = batch_size * 2;

        // Create mask to exclude self-similarities if configured
        let self_mask = if self.config.exclude_self {
            let eye = Array::eye::<f32>(total_size as i32, None, None)?;
            Array::from(1.0).subtract(&eye)?
        } else {
            Array::ones::<f32>(&[total_size as i32, total_size as i32])?
        };

        // Apply self mask to similarities
        let masked_similarities = scaled_similarity
            .multiply(&self_mask)?
            .add(&Array::eye::<f32>(total_size as i32, None, None)?.multiply(Array::from(-1e9))?)?;

        // Compute log softmax
        let log_softmax = self.log_softmax(&masked_similarities)?;

        // Extract positive log probabilities
        let positive_log_probs = log_softmax.multiply(positive_mask)?;

        // Sum over positive pairs and average over batch
        let loss_per_sample = positive_log_probs.sum(&[-1], false)?;
        let num_positives_per_sample = positive_mask.sum(&[-1], false)?;

        // Average loss per positive pair, then over batch
        let epsilon = Array::from(1e-8);
        let safe_denominators = mlx_rs::ops::maximum(&num_positives_per_sample, &epsilon)?;
        let avg_loss_per_sample = loss_per_sample.divide(&safe_denominators)?;
        let negated_loss = &avg_loss_per_sample * Array::from(-1.0);
        negated_loss.mean(&[], false).map_err(Into::into)
    }

    /// Compute log softmax using specified base
    fn log_softmax(&self, logits: &Array) -> Result<Array> {
        let max_logits = logits.max(&[-1], true)?;
        let shifted_logits = logits.subtract(&max_logits)?;
        let sum_exp = shifted_logits.exp()?.sum(&[-1], true)?;
        let log_sum_exp = sum_exp.log()?;

        let log_softmax = shifted_logits.subtract(&log_sum_exp)?;

        // Convert to specified base if not natural log
        if (self.config.log_base - std::f32::consts::E).abs() > 1e-6 {
            let conversion_factor = Array::from(1.0 / self.config.log_base.ln());
            log_softmax.multiply(&conversion_factor).map_err(Into::into)
        } else {
            Ok(log_softmax)
        }
    }

    /// Compute NT-Xent specific metrics
    fn compute_ntxent_metrics(
        &self,
        similarity_matrix: &Array,
        positive_mask: &Array,
        batch_size: usize,
    ) -> Result<LossMetrics> {
        let total_size = batch_size * 2;

        // Compute average positive and negative similarities
        let positive_similarities = similarity_matrix.multiply(positive_mask)?;
        let negative_mask = Array::from(1.0)
            .subtract(positive_mask)?
            .subtract(&Array::eye::<f32>(total_size as i32, None, None)?)?; // Exclude self-similarities
        let negative_similarities = similarity_matrix.multiply(&negative_mask)?;

        let num_positives = positive_mask.sum(&[], false)?;
        let num_negatives = negative_mask.sum(&[], false)?;

        let avg_positive_sim = positive_similarities
            .sum(&[], false)?
            .divide(&mlx_rs::ops::maximum(&num_positives, Array::from(1.0))?)?;
        let avg_negative_sim = negative_similarities
            .sum(&[], false)?
            .divide(&mlx_rs::ops::maximum(&num_negatives, Array::from(1.0))?)?;

        let avg_pos = avg_positive_sim.item::<f32>();
        let avg_neg = avg_negative_sim.item::<f32>();

        // Compute accuracy: fraction of samples where positive similarity > max negative similarity
        let mut correct = 0;

        for i in 0..total_size {
            let row_similarities = similarity_matrix.index((i as i32,));
            let row_positive_mask = positive_mask.index((i as i32,));
            let row_negative_mask = negative_mask.index((i as i32,));

            // Find maximum positive similarity
            let positive_sims = row_similarities.multiply(&row_positive_mask)?;
            let max_positive = positive_sims.max(&[], false)?;

            // Find maximum negative similarity
            let negative_sims = row_similarities.multiply(&row_negative_mask)?;
            let max_negative = negative_sims.max(&[], false)?;

            if max_positive.item::<f32>() > max_negative.item::<f32>() {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / total_size as f32;

        // Compute top-k accuracy
        let top5_accuracy = self.compute_topk_accuracy(similarity_matrix, positive_mask, 5)?;

        Ok(LossMetrics {
            positive_pairs: num_positives.item::<i32>() as usize,
            negative_pairs: num_negatives.item::<i32>() as usize,
            avg_positive_similarity: avg_pos,
            avg_negative_similarity: avg_neg,
            accuracy,
            top_k_accuracy: Some(top5_accuracy),
        })
    }

    /// Compute top-k accuracy
    fn compute_topk_accuracy(
        &self,
        similarity_matrix: &Array,
        positive_mask: &Array,
        k: i32,
    ) -> Result<f32> {
        let total_size = similarity_matrix.shape()[0];
        let mut correct_count = 0;

        for i in 0..total_size {
            let row_similarities = similarity_matrix.index((i,));
            let row_positive_mask = positive_mask.index((i,));

            // Get top-k indices
            // Use argsort to get top-k indices
            let sorted_indices = mlx_rs::ops::argsort(&row_similarities, -1)?;
            let shape_len = sorted_indices.shape().len();
            let last_dim = sorted_indices.shape()[shape_len - 1];
            let start_idx = last_dim - k;
            let topk_indices = sorted_indices.index((.., start_idx..));

            // Check if any positive pair is in top-k
            let mut found_positive = false;
            for j in 0..k {
                let idx = topk_indices.index((j,)).item::<i32>();
                let is_positive = row_positive_mask.index((idx,)).item::<f32>() > 0.5;
                if is_positive {
                    found_positive = true;
                    break;
                }
            }

            if found_positive {
                correct_count += 1;
            }
        }

        Ok(correct_count as f32 / total_size as f32)
    }

    /// Update configuration for next training step
    pub fn update_config(&mut self, step: usize) {
        self.config.update_step(step);
    }

    /// Get current temperature
    pub fn current_temperature(&self) -> f32 {
        self.config.current_temperature()
    }
}

impl LossFunction for NtXentLoss {
    fn compute_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
        _labels: Option<&Array>,
    ) -> Result<LossResult> {
        self.compute_ntxent_loss(query_embeddings, doc_embeddings)
    }

    fn name(&self) -> &'static str {
        "NT-Xent"
    }

    fn config(&self) -> &dyn std::any::Any {
        &self.config
    }
}

/// Builder for NT-Xent configuration
pub struct NtXentConfigBuilder {
    config: NtXentConfig,
}

impl Default for NtXentConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NtXentConfigBuilder {
    /// Create a new NT-Xent config builder
    pub fn new() -> Self {
        Self {
            config: NtXentConfig::default(),
        }
    }

    /// Set temperature parameter
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set whether to normalize embeddings
    pub fn normalize_embeddings(mut self, normalize: bool) -> Self {
        self.config.normalize_embeddings = normalize;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Set whether to use symmetric loss
    pub fn symmetric(mut self, symmetric: bool) -> Self {
        self.config.symmetric = symmetric;
        self
    }

    /// Set whether to exclude self-similarities
    pub fn exclude_self(mut self, exclude: bool) -> Self {
        self.config.exclude_self = exclude;
        self
    }

    /// Set logarithm base
    pub fn log_base(mut self, base: f32) -> Self {
        self.config.log_base = base;
        self
    }

    /// Enable temperature annealing
    pub fn enable_temperature_annealing(
        mut self,
        initial_temp: f32,
        final_temp: f32,
        total_steps: usize,
    ) -> Self {
        self.config.use_temperature_annealing = true;
        self.config.initial_temperature = initial_temp;
        self.config.final_temperature = final_temp;
        self.config.total_steps = total_steps;
        self
    }

    /// Build the configuration
    pub fn build(self) -> NtXentConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntxent_config_default() {
        let config = NtXentConfig::default();
        assert_eq!(config.temperature, 0.1);
        assert!(config.normalize_embeddings);
        assert!(config.symmetric);
        assert!(config.exclude_self);
        assert!(!config.use_temperature_annealing);
    }

    #[test]
    fn test_temperature_annealing() {
        let mut config = NtXentConfig {
            use_temperature_annealing: true,
            initial_temperature: 1.0,
            final_temperature: 0.1,
            total_steps: 100,
            current_step: 0,
            ..Default::default()
        };

        // At step 0, should be close to initial temperature
        assert!((config.current_temperature() - 1.0).abs() < 0.1);

        // At step 50 (middle), should be between initial and final
        config.update_step(50);
        let mid_temp = config.current_temperature();
        assert!(mid_temp < 1.0 && mid_temp > 0.1);

        // At final step, should be close to final temperature
        config.update_step(100);
        assert!((config.current_temperature() - 0.1).abs() < 0.1);
    }

    #[test]
    fn test_ntxent_config_builder() {
        let config = NtXentConfigBuilder::new()
            .temperature(0.2)
            .symmetric(false)
            .exclude_self(false)
            .log_base(2.0)
            .enable_temperature_annealing(0.5, 0.05, 1000)
            .build();

        assert_eq!(config.temperature, 0.2);
        assert!(!config.symmetric);
        assert!(!config.exclude_self);
        assert_eq!(config.log_base, 2.0);
        assert!(config.use_temperature_annealing);
        assert_eq!(config.initial_temperature, 0.5);
        assert_eq!(config.final_temperature, 0.05);
        assert_eq!(config.total_steps, 1000);
    }

    #[test]
    fn test_ntxent_loss_creation() {
        let config = NtXentConfig::default();
        let loss = NtXentLoss::new(config);
        assert_eq!(loss.name(), "NT-Xent");
    }

    #[tokio::test]
    async fn test_ntxent_loss_computation() -> Result<()> {
        let config = NtXentConfig::default();
        let loss_fn = NtXentLoss::new(config);

        // Create dummy embeddings
        let embeddings1 = mlx_rs::random::normal::<f32>(&[4, 128], None, None, None)?;
        let embeddings2 = mlx_rs::random::normal::<f32>(&[4, 128], None, None, None)?;

        let result = loss_fn.compute_loss(&embeddings1, &embeddings2, None)?;

        // Check that loss is computed
        assert!(result.scalar_loss() > 0.0);
        assert_eq!(result.metrics.positive_pairs, 8); // 4 pairs in each direction

        Ok(())
    }

    #[test]
    fn test_positive_mask_creation() -> Result<()> {
        let config = NtXentConfig::default();
        let loss_fn = NtXentLoss::new(config);

        let mask = loss_fn.create_positive_mask(2)?;
        assert_eq!(mask.shape(), vec![4, 4]);

        // Check that (0,2) and (2,0) are positive pairs
        assert_eq!(mask.index((0, 2)).item::<f32>(), 1.0);
        assert_eq!(mask.index((2, 0)).item::<f32>(), 1.0);

        // Check that (1,3) and (3,1) are positive pairs
        assert_eq!(mask.index((1, 3)).item::<f32>(), 1.0);
        assert_eq!(mask.index((3, 1)).item::<f32>(), 1.0);

        // Check that other pairs are not positive
        assert_eq!(mask.index((0, 1)).item::<f32>(), 0.0);
        assert_eq!(mask.index((0, 3)).item::<f32>(), 0.0);

        Ok(())
    }
}
