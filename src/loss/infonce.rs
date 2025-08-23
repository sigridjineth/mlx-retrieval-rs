//! InfoNCE (Information Noise Contrastive Estimation) Loss
//!
//! This module implements the InfoNCE loss function, commonly used in contrastive
//! learning for training embedding models. InfoNCE maximizes mutual information
//! between positive pairs while minimizing it for negative pairs.

use crate::error::Result;
use crate::loss::{utils, DistanceMetric, LossFunction, LossMetrics, LossResult};
use mlx_rs::{
    ops::indexing::{argmax, IndexOp},
    Array,
};
// Removed unused import std::ops::Neg
use serde::{Deserialize, Serialize};

/// Configuration for InfoNCE loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoNceConfig {
    /// Temperature parameter for scaling similarities
    pub temperature: f32,

    /// Whether to normalize embeddings before computing similarity
    pub normalize_embeddings: bool,

    /// Number of negative samples (if not using all in-batch negatives)
    pub num_negatives: Option<usize>,

    /// Whether to use symmetric loss (query->doc and doc->query)
    pub symmetric: bool,

    /// Distance metric for computing similarities
    pub distance_metric: DistanceMetric,

    /// Label smoothing parameter
    pub label_smoothing: f32,
}

impl Default for InfoNceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.05, // Match Python default
            normalize_embeddings: true,
            num_negatives: None, // Use all in-batch negatives
            symmetric: true,     // Changed to true to match Python bidirectional loss
            distance_metric: DistanceMetric::Cosine,
            label_smoothing: 0.0,
        }
    }
}

/// InfoNCE loss implementation
pub struct InfoNceLoss {
    config: InfoNceConfig,
}

impl InfoNceLoss {
    /// Create a new InfoNCE loss function
    pub fn new(config: InfoNceConfig) -> Self {
        Self { config }
    }

    /// Compute InfoNCE loss for a batch of query-document pairs
    pub fn compute_infonce_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
    ) -> Result<LossResult> {
        let batch_size = query_embeddings.shape()[0] as usize;

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

        // Scale by temperature
        let scaled_similarities = similarities.divide(Array::from(self.config.temperature))?;

        // Create positive pair labels (diagonal matrix)
        let labels = self.create_positive_labels(batch_size)?;

        // Compute InfoNCE loss
        let loss = self.compute_loss_from_similarities(&scaled_similarities, &labels)?;

        // Compute additional loss if symmetric
        let total_loss = if self.config.symmetric {
            let transposed_similarities = scaled_similarities.transpose(&[1, 0])?;
            let symmetric_loss =
                self.compute_loss_from_similarities(&transposed_similarities, &labels)?;
            loss.add(&symmetric_loss)?.divide(Array::from(2.0))?
        } else {
            loss
        };

        // Compute metrics
        let metrics = self.compute_infonce_metrics(&similarities, &labels)?;

        Ok(LossResult::new(total_loss, metrics))
    }

    /// Compute similarity matrix between queries and documents
    fn compute_similarity_matrix(&self, queries: &Array, docs: &Array) -> Result<Array> {
        match self.config.distance_metric {
            DistanceMetric::Cosine => queries
                .matmul(&docs.transpose(&[1, 0])?)
                .map_err(Into::into),
            DistanceMetric::DotProduct => queries
                .matmul(&docs.transpose(&[1, 0])?)
                .map_err(Into::into),
            DistanceMetric::Euclidean => self.compute_euclidean_similarity_matrix(queries, docs),
            DistanceMetric::Manhattan => self.compute_manhattan_similarity_matrix(queries, docs),
        }
    }

    /// Compute Euclidean distance-based similarity matrix
    fn compute_euclidean_similarity_matrix(&self, queries: &Array, docs: &Array) -> Result<Array> {
        let _batch_size = queries.shape()[0];
        let _doc_count = docs.shape()[0];

        // Expand dimensions for broadcasting
        let queries_expanded = queries.expand_dims(&[1])?; // (batch, 1, dim)
        let docs_expanded = docs.expand_dims(&[0])?; // (1, doc_count, dim)

        // Compute squared distances
        let diff = queries_expanded.subtract(&docs_expanded)?;
        let squared_distances = diff.square()?.sum(&[-1], false)?;

        // Convert distance to similarity (negative distance)
        Ok(&squared_distances * Array::from(-1.0))
    }

    /// Compute Manhattan distance-based similarity matrix
    fn compute_manhattan_similarity_matrix(&self, queries: &Array, docs: &Array) -> Result<Array> {
        let queries_expanded = queries.expand_dims(&[1])?;
        let docs_expanded = docs.expand_dims(&[0])?;

        let diff = queries_expanded.subtract(&docs_expanded)?;
        let manhattan_distances = diff.abs()?.sum(&[-1], false)?;

        // Convert distance to similarity (negative distance)
        Ok(&manhattan_distances * Array::from(-1.0))
    }

    /// Create positive pair labels (diagonal matrix)
    fn create_positive_labels(&self, batch_size: usize) -> Result<Array> {
        if self.config.label_smoothing > 0.0 {
            self.create_smoothed_labels(batch_size)
        } else {
            Array::eye::<f32>(batch_size as i32, None, None).map_err(Into::into)
        }
    }

    /// Create label-smoothed positive labels
    fn create_smoothed_labels(&self, batch_size: usize) -> Result<Array> {
        let epsilon = self.config.label_smoothing;
        let positive_prob = 1.0 - epsilon;
        let negative_prob = epsilon / (batch_size - 1) as f32;

        // Create base matrix filled with negative probability
        let base_labels = Array::full::<f32>(
            &[batch_size as i32, batch_size as i32],
            &Array::from(negative_prob),
        )?;

        // Create diagonal matrix with positive probability
        let eye_matrix = Array::eye::<f32>(batch_size as i32, None, None)?;
        let positive_diagonal = &eye_matrix * Array::from(positive_prob);
        let negative_diagonal = &eye_matrix * Array::from(negative_prob);

        // Combine: base + (positive - negative) on diagonal
        let diagonal_adjustment = positive_diagonal.subtract(&negative_diagonal)?;
        let labels = base_labels.add(&diagonal_adjustment)?;

        Ok(labels)
    }

    /// Compute cross-entropy loss from similarity matrix and labels
    fn compute_loss_from_similarities(
        &self,
        similarities: &Array,
        labels: &Array,
    ) -> Result<Array> {
        // Compute log softmax
        let log_softmax = self.log_softmax(similarities)?;

        // Compute cross-entropy loss
        let loss_per_sample = labels.multiply(&log_softmax)?.sum(&[-1], false)?;

        // Return negative mean loss
        let negated_loss = &loss_per_sample * Array::from(-1.0);
        negated_loss.mean(&[], false).map_err(Into::into)
    }

    /// Compute log softmax along the last dimension
    fn log_softmax(&self, logits: &Array) -> Result<Array> {
        let max_logits = logits.max(&[-1], true)?;
        let shifted_logits = logits.subtract(&max_logits)?;
        let sum_exp = shifted_logits.exp()?.sum(&[-1], true)?.log()?;
        shifted_logits.subtract(&sum_exp).map_err(Into::into)
    }

    /// Compute InfoNCE-specific metrics
    fn compute_infonce_metrics(
        &self,
        similarities: &Array,
        _labels: &Array,
    ) -> Result<LossMetrics> {
        let batch_size = similarities.shape()[0] as usize;

        // Compute accuracy (fraction of correct top-1 predictions)
        // Get predictions (argmax along last dimension)
        let predictions = argmax(similarities, -1, false)?;

        // Create ground truth (diagonal indices)
        let targets: Vec<i32> = (0..batch_size as i32).collect();
        let target_array = Array::from_slice(&targets, &[batch_size as i32]);

        // Calculate accuracy
        let correct = predictions.eq(&target_array)?;
        let accuracy_array = correct
            .sum(&[], false)?
            .divide(Array::from(batch_size as f32))?;
        let accuracy = accuracy_array.item::<f32>();

        // Compute average positive and negative similarities
        // Diagonal elements are positives, off-diagonal are negatives
        let eye_mask = Array::eye::<f32>(batch_size as i32, None, None)?;
        let positive_mask = eye_mask;
        let negative_mask = &Array::from(1.0) - &positive_mask;

        let positive_similarities =
            similarities.multiply(&positive_mask.as_dtype(similarities.dtype())?)?;
        let negative_similarities =
            similarities.multiply(&negative_mask.as_dtype(similarities.dtype())?)?;

        let avg_positive = positive_similarities.sum(&[], false)?.divide(
            &positive_mask
                .sum(&[], false)?
                .as_dtype(similarities.dtype())?,
        )?;
        let avg_negative = negative_similarities.sum(&[], false)?.divide(
            &negative_mask
                .sum(&[], false)?
                .as_dtype(similarities.dtype())?,
        )?;

        let avg_pos_sim = avg_positive.item::<f32>();
        let avg_neg_sim = avg_negative.item::<f32>();

        // Compute top-5 accuracy
        let top5_accuracy = self.compute_topk_accuracy(similarities, 5)?;

        Ok(LossMetrics {
            positive_pairs: batch_size,
            negative_pairs: batch_size * (batch_size - 1),
            avg_positive_similarity: avg_pos_sim,
            avg_negative_similarity: avg_neg_sim,
            accuracy,
            top_k_accuracy: Some(top5_accuracy),
        })
    }

    /// Compute top-k accuracy
    fn compute_topk_accuracy(&self, similarities: &Array, k: i32) -> Result<f32> {
        let batch_size = similarities.shape()[0];

        // Get top-k predictions using argsort (descending order)
        // argsort returns ascending order, so we need the last k elements
        let sorted_indices = mlx_rs::ops::argsort(similarities, -1)?;
        let start_idx = (similarities.shape()[1] - k).max(0);
        let topk_indices = sorted_indices.index((.., start_idx..));

        // Check if correct index (diagonal) is in top-k
        let mut correct_count = 0;
        for i in 0..batch_size {
            let target = i;
            let _topk_for_i = topk_indices.index((i,));

            // Get the top-k indices for this sample and check if target is in them
            // Since we can't use to_vec, we'll use a different approach
            let topk_for_sample = topk_indices.index((i,));

            // Check if any of the top-k indices equals the target
            // We'll compare element by element
            let mut is_correct = false;
            for j in 0..k {
                let idx_val = topk_for_sample.index((j,)).item::<i32>();
                if idx_val == target {
                    is_correct = true;
                    break;
                }
            }

            if is_correct {
                correct_count += 1;
            }
        }

        Ok(correct_count as f32 / batch_size as f32)
    }
}

impl LossFunction for InfoNceLoss {
    fn compute_loss(
        &self,
        query_embeddings: &Array,
        doc_embeddings: &Array,
        _labels: Option<&Array>,
    ) -> Result<LossResult> {
        self.compute_infonce_loss(query_embeddings, doc_embeddings)
    }

    fn name(&self) -> &'static str {
        "InfoNCE"
    }

    fn config(&self) -> &dyn std::any::Any {
        &self.config
    }
}

/// Builder for InfoNCE configuration
pub struct InfoNceConfigBuilder {
    config: InfoNceConfig,
}

impl Default for InfoNceConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InfoNceConfigBuilder {
    /// Create a new InfoNCE config builder
    pub fn new() -> Self {
        Self {
            config: InfoNceConfig::default(),
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

    /// Set number of negative samples
    pub fn num_negatives(mut self, num_negatives: usize) -> Self {
        self.config.num_negatives = Some(num_negatives);
        self
    }

    /// Set whether to use symmetric loss
    pub fn symmetric(mut self, symmetric: bool) -> Self {
        self.config.symmetric = symmetric;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Set label smoothing parameter
    pub fn label_smoothing(mut self, smoothing: f32) -> Self {
        self.config.label_smoothing = smoothing;
        self
    }

    /// Build the configuration
    pub fn build(self) -> InfoNceConfig {
        self.config
    }
}

/// Simple InfoNCE loss implementation that exactly matches the Python reference
/// This is used for golden test validation before integrating with the complex InfoNCE struct
pub fn simple_infonce_loss(
    query_embeddings: &Array,
    doc_embeddings: &Array,
    temperature: f32,
) -> Result<f32> {
    // 1. Compute similarity matrix: query @ doc.T
    let scores = query_embeddings.matmul(&doc_embeddings.transpose(&[1, 0])?)?;

    // 2. Scale by temperature: scores / temperature
    let scaled_scores = scores.divide(Array::from(temperature))?;

    // 3. Create labels as [0, 1, 2, 3] (diagonal elements are positive pairs)
    let batch_size = query_embeddings.shape()[0];
    let labels_vec: Vec<i32> = (0..batch_size).collect();
    let labels = Array::from_slice(&labels_vec, &[batch_size]);

    // 4. Compute cross-entropy loss using log_softmax approach
    let loss = compute_cross_entropy(&scaled_scores, &labels)?;

    // 5. Return scalar loss value
    Ok(loss.item::<f32>())
}

/// Simplified cross-entropy implementation
fn compute_cross_entropy(logits: &Array, labels: &Array) -> Result<Array> {
    let batch_size = logits.shape()[0];

    // Compute log_softmax
    let max_vals = logits.max(&[-1], true)?;
    let shifted = logits.subtract(&max_vals)?;
    let exp_vals = shifted.exp()?;
    let sum_exp = exp_vals.sum(&[-1], true)?;
    let log_softmax = shifted.subtract(&sum_exp.log()?)?;

    // Manual gathering for cross-entropy
    let mut total_loss = 0.0f32;

    for i in 0..batch_size {
        let label_idx = labels.index((i,)).item::<i32>() as usize;
        let log_prob = log_softmax.index((i, label_idx as i32)).item::<f32>();
        total_loss -= log_prob; // Negative log likelihood
    }

    // Return mean loss
    let mean_loss = total_loss / batch_size as f32;
    Ok(Array::from(mean_loss))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infonce_config_default() {
        let config = InfoNceConfig::default();
        assert_eq!(config.temperature, 0.07);
        assert!(config.normalize_embeddings);
        assert!(!config.symmetric);
        assert_eq!(config.label_smoothing, 0.0);
    }

    #[test]
    fn test_infonce_config_builder() {
        let config = InfoNceConfigBuilder::new()
            .temperature(0.1)
            .symmetric(true)
            .label_smoothing(0.1)
            .build();

        assert_eq!(config.temperature, 0.1);
        assert!(config.symmetric);
        assert_eq!(config.label_smoothing, 0.1);
    }

    #[test]
    fn test_infonce_loss_creation() {
        let config = InfoNceConfig::default();
        let loss = InfoNceLoss::new(config);
        assert_eq!(loss.name(), "InfoNCE");
    }

    #[tokio::test]
    async fn test_infonce_loss_computation() -> Result<()> {
        let config = InfoNceConfig::default();
        let loss_fn = InfoNceLoss::new(config);

        // Create dummy embeddings
        let query_embeddings = mlx_rs::random::normal::<f32>(&[4, 128], None, None, None)?;
        let doc_embeddings = mlx_rs::random::normal::<f32>(&[4, 128], None, None, None)?;

        let result = loss_fn.compute_loss(&query_embeddings, &doc_embeddings, None)?;

        // Check that loss is computed
        assert!(result.scalar_loss() > 0.0);
        assert_eq!(result.metrics.positive_pairs, 4);
        assert_eq!(result.metrics.negative_pairs, 12); // 4 * (4 - 1)

        Ok(())
    }

    #[test]
    fn test_positive_labels_creation() -> Result<()> {
        let config = InfoNceConfig::default();
        let loss_fn = InfoNceLoss::new(config);

        let labels = loss_fn.create_positive_labels(3)?;
        assert_eq!(labels.shape(), vec![3, 3]);

        // Check diagonal elements are 1
        assert_eq!(labels.index((0, 0)).item::<f32>(), 1.0);
        assert_eq!(labels.index((1, 1)).item::<f32>(), 1.0);
        assert_eq!(labels.index((2, 2)).item::<f32>(), 1.0);

        // Check off-diagonal elements are 0
        assert_eq!(labels.index((0, 1)).item::<f32>(), 0.0);
        assert_eq!(labels.index((1, 0)).item::<f32>(), 0.0);

        Ok(())
    }

    #[test]
    fn test_label_smoothing() -> Result<()> {
        let config = InfoNceConfigBuilder::new().label_smoothing(0.1).build();
        let loss_fn = InfoNceLoss::new(config);

        let labels = loss_fn.create_positive_labels(3)?;

        // With label smoothing, diagonal should be < 1.0
        let diagonal_val = labels.index((0, 0)).item::<f32>();
        assert!(diagonal_val < 1.0);
        assert!(diagonal_val > 0.8); // Should be close to 0.9 for epsilon=0.1

        // Off-diagonal should be > 0.0
        let off_diagonal_val = labels.index((0, 1)).item::<f32>();
        assert!(off_diagonal_val > 0.0);
        assert!(off_diagonal_val < 0.1); // Should be epsilon/(n-1)

        Ok(())
    }

    #[test]
    fn test_infonce_golden_test() -> Result<()> {
        use std::fs;

        // This test demonstrates the TDD approach for InfoNCE loss implementation
        // First, we write a failing test that loads golden data
        // Then we implement the minimal InfoNCE function to make it pass

        println!("🧪 Running InfoNCE Golden Test (TDD approach)");

        // Load golden test data
        let golden_data =
            fs::read_to_string("tests/infonce_loss.json").expect("Failed to read golden test file");
        let test_case: serde_json::Value =
            serde_json::from_str(&golden_data).expect("Failed to parse golden test JSON");

        // Extract input data from golden test
        let query_data = test_case["inputs"]["query_embeddings"]["data"]
            .as_array()
            .unwrap();
        let doc_data = test_case["inputs"]["doc_embeddings"]["data"]
            .as_array()
            .unwrap();
        let expected_loss = test_case["expected_output"]["loss"]["value"]
            .as_f64()
            .unwrap() as f32;
        let temperature = test_case["config"]["temperature"].as_f64().unwrap() as f32;

        println!("📊 Golden test expects:");
        println!("   Loss: {}", expected_loss);
        println!("   Temperature: {}", temperature);

        // Convert JSON data to flat vectors for MLX arrays
        let query_vec: Vec<f32> = query_data
            .iter()
            .flat_map(|batch| batch.as_array().unwrap())
            .map(|val| val.as_f64().unwrap() as f32)
            .collect();

        let doc_vec: Vec<f32> = doc_data
            .iter()
            .flat_map(|batch| batch.as_array().unwrap())
            .map(|val| val.as_f64().unwrap() as f32)
            .collect();

        // Create MLX arrays with correct shapes [4, 8]
        let query_embeddings = Array::from_slice(&query_vec, &[4, 8]);
        let doc_embeddings = Array::from_slice(&doc_vec, &[4, 8]);

        // Call our simple InfoNCE implementation
        // This follows the Python reference exactly:
        // scores = mx.matmul(query_embeddings, doc_embeddings.T) / self.temperature
        // labels = mx.arange(scores.shape[0])
        // return mx.mean(nn.losses.cross_entropy(scores, labels))
        let actual_loss = simple_infonce_loss(&query_embeddings, &doc_embeddings, temperature)?;

        // Validate results within tolerance
        let diff = (actual_loss - expected_loss).abs();
        let tolerance = 1e-3; // Slightly relaxed tolerance for numerical differences

        println!("📈 Results:");
        println!("   Expected: {}", expected_loss);
        println!("   Actual:   {}", actual_loss);
        println!("   Diff:     {}", diff);

        if diff < tolerance {
            println!("✅ InfoNCE golden test PASSED!");
        } else {
            println!("❌ InfoNCE golden test FAILED!");
        }

        assert!(
            diff < tolerance,
            "Loss difference {} exceeds tolerance {}, expected: {}, got: {}",
            diff,
            tolerance,
            expected_loss,
            actual_loss
        );

        Ok(())
    }
}
