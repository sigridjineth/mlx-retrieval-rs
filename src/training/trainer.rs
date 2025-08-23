//! Training loop and trainer implementation
//!
//! This module provides the main training infrastructure including
//! the trainer class that orchestrates the entire training process.

use crate::data::{Batch, BatchLoader};
use crate::error::{MlxRetrievalError, Result};
use crate::loss::LossFunction;
use crate::model::Model;
use crate::training::{
    GradientClipper, LRScheduler, Optimizer, TrainingCheckpoint, TrainingConfig, TrainingUtils,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,

    /// Current step within epoch
    pub step: usize,

    /// Global step across all epochs
    pub global_step: usize,

    /// Best validation metric seen so far
    pub best_metric: f32,

    /// Steps since best metric (for early stopping)
    pub steps_since_best: usize,

    /// Training start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Current learning rate
    pub learning_rate: f32,

    /// Whether training should stop
    pub should_stop: bool,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            step: 0,
            global_step: 0,
            best_metric: f32::INFINITY,
            steps_since_best: 0,
            start_time: chrono::Utc::now(),
            learning_rate: 0.0,
            should_stop: false,
        }
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Training metrics for a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss
    pub train_loss: f32,

    /// Learning rate
    pub learning_rate: f32,

    /// Gradient norm
    pub grad_norm: f32,

    /// Training time for this step (seconds)
    pub step_time: f64,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f32>,
}

impl TrainingMetrics {
    pub fn new(train_loss: f32, learning_rate: f32, grad_norm: f32, step_time: f64) -> Self {
        Self {
            train_loss,
            learning_rate,
            grad_norm,
            step_time,
            custom_metrics: HashMap::new(),
        }
    }

    pub fn add_metric(&mut self, name: String, value: f32) {
        self.custom_metrics.insert(name, value);
    }
}

/// Training metrics for a full epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number
    pub epoch: usize,

    /// Average training loss
    pub avg_train_loss: f32,

    /// Validation loss (if available)
    pub val_loss: Option<f32>,

    /// Validation accuracy (if available)
    pub val_accuracy: Option<f32>,

    /// Training time for the epoch
    pub epoch_time: f64,

    /// Additional metrics
    pub custom_metrics: HashMap<String, f32>,
}

/// Trainer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Base training configuration
    pub training_config: TrainingConfig,

    /// Whether to use mixed precision
    pub mixed_precision: bool,

    /// Whether to compile the model (if supported)
    pub compile_model: bool,

    /// Validation dataset size for progress reporting
    pub val_dataset_size: Option<usize>,

    /// Custom evaluation function name
    pub eval_function: Option<String>,

    /// Whether to save optimizer state in checkpoints
    pub save_optimizer_state: bool,

    /// Whether to resume from checkpoint
    pub resume_from_checkpoint: Option<String>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            training_config: TrainingConfig::default(),
            mixed_precision: false,
            compile_model: false,
            val_dataset_size: None,
            eval_function: None,
            save_optimizer_state: true,
            resume_from_checkpoint: None,
        }
    }
}

/// Main trainer class
pub struct Trainer {
    /// Trainer configuration
    config: TrainerConfig,

    /// Model being trained
    model: Box<dyn Model>,

    /// Loss function
    loss_fn: Box<dyn LossFunction>,

    /// Optimizer
    optimizer: Box<dyn Optimizer>,

    /// Learning rate scheduler
    lr_scheduler: Option<LRScheduler>,

    /// Gradient clipper
    grad_clipper: Option<GradientClipper>,

    /// Training state
    state: TrainingState,

    /// Metrics history
    metrics_history: Vec<EpochMetrics>,
}

impl Trainer {
    pub fn new(
        config: TrainerConfig,
        model: Box<dyn Model>,
        loss_fn: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let grad_clipper = if config.training_config.max_grad_norm > 0.0 {
            Some(GradientClipper::new(config.training_config.max_grad_norm))
        } else {
            None
        };

        Self {
            config,
            model,
            loss_fn,
            optimizer,
            lr_scheduler: None,
            grad_clipper,
            state: TrainingState::new(),
            metrics_history: Vec::new(),
        }
    }

    /// Set learning rate scheduler
    pub fn set_lr_scheduler(&mut self, scheduler: LRScheduler) {
        self.lr_scheduler = Some(scheduler);
    }

    /// Initialize training
    pub fn initialize(&mut self) -> Result<()> {
        // Set random seed if specified
        if let Some(seed) = self.config.training_config.seed {
            TrainingUtils::set_seed(seed)?;
        }

        // Set model to training mode
        self.model.train(true);

        // Initialize learning rate
        self.state.learning_rate = self.optimizer.get_lr();

        // Resume from checkpoint if specified
        if let Some(checkpoint_path) = &self.config.resume_from_checkpoint {
            let checkpoint_path_clone = checkpoint_path.clone();
            self.resume_from_checkpoint(&checkpoint_path_clone)?;
        }

        tracing::info!("Trainer initialized successfully");
        tracing::info!("Model parameters: {}", self.model.num_parameters());

        Ok(())
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, train_loader: &mut BatchLoader) -> Result<EpochMetrics> {
        let epoch_start = Instant::now();
        self.model.train(true);

        let mut epoch_loss = 0.0;
        let mut step_count = 0;
        let mut step_metrics = Vec::new();

        while let Some(batch) = train_loader.try_create_batch()? {
            let _step_start = Instant::now();

            // Training step
            let metrics = self.training_step(batch)?;
            step_metrics.push(metrics);

            epoch_loss += step_metrics.last().unwrap().train_loss;
            step_count += 1;
            self.state.step += 1;
            self.state.global_step += 1;

            // Update learning rate
            if let Some(ref mut scheduler) = self.lr_scheduler {
                self.state.learning_rate = scheduler.step();
                self.optimizer.set_lr(self.state.learning_rate);
            }

            // Logging
            if self.state.global_step % self.config.training_config.logging_steps == 0 {
                self.log_training_progress(&step_metrics)?;
            }

            // Evaluation
            if self.state.global_step % self.config.training_config.eval_steps == 0 {
                // Placeholder for evaluation
                tracing::info!("Evaluation step at global step {}", self.state.global_step);
            }

            // Checkpointing
            if self.state.global_step % self.config.training_config.save_steps == 0 {
                self.save_checkpoint()?;
            }

            // Early stopping check
            if self.should_stop_early() {
                self.state.should_stop = true;
                break;
            }

            // Max steps check
            if let Some(max_steps) = self.config.training_config.max_steps {
                if self.state.global_step >= max_steps {
                    self.state.should_stop = true;
                    break;
                }
            }
        }

        let epoch_time = epoch_start.elapsed().as_secs_f64();
        let avg_loss = if step_count > 0 {
            epoch_loss / step_count as f32
        } else {
            0.0
        };

        let epoch_metrics = EpochMetrics {
            epoch: self.state.epoch,
            avg_train_loss: avg_loss,
            val_loss: None,
            val_accuracy: None,
            epoch_time,
            custom_metrics: HashMap::new(),
        };

        self.metrics_history.push(epoch_metrics.clone());

        tracing::info!(
            "Epoch {} completed - Loss: {:.4}, Time: {}",
            self.state.epoch,
            avg_loss,
            TrainingUtils::format_training_time(epoch_time)
        );

        Ok(epoch_metrics)
    }

    /// Single training step
    fn training_step(&mut self, batch: Batch) -> Result<TrainingMetrics> {
        let step_start = Instant::now();

        // Forward pass (placeholder - actual implementation would depend on model)
        let query_embeddings =
            mlx_rs::random::normal::<f32>(&[batch.batch_size() as i32, 768], None, None, None)?;
        let doc_embeddings =
            mlx_rs::random::normal::<f32>(&[batch.batch_size() as i32, 768], None, None, None)?;

        // Compute loss
        let loss_result = self
            .loss_fn
            .compute_loss(&query_embeddings, &doc_embeddings, None)?;
        let loss_value = loss_result.scalar_loss();

        // Backward pass (placeholder - MLX would handle automatic differentiation)
        let gradients = vec![
            mlx_rs::random::normal::<f32>(&[768, 768], None, None, None)?,
            mlx_rs::random::normal::<f32>(&[768], None, None, None)?,
        ];

        // Gradient clipping
        let mut grad_norm = 0.0;
        if let Some(ref clipper) = self.grad_clipper {
            let mut grads_mut = gradients.clone();
            grad_norm = clipper.clip_grad_norm(&mut grads_mut)?;
        }

        // Optimizer step (placeholder - would use actual parameters)
        let mut params = vec![
            mlx_rs::random::normal::<f32>(&[768, 768], None, None, None)?,
            mlx_rs::random::normal::<f32>(&[768], None, None, None)?,
        ];
        self.optimizer.step(&mut params, &gradients)?;

        let step_time = step_start.elapsed().as_secs_f64();

        Ok(TrainingMetrics::new(
            loss_value,
            self.state.learning_rate,
            grad_norm,
            step_time,
        ))
    }

    /// Train for multiple epochs
    pub fn train(&mut self, train_loader: &mut BatchLoader) -> Result<()> {
        self.initialize()?;

        tracing::info!(
            "Starting training for {} epochs",
            self.config.training_config.num_epochs
        );

        for epoch in 0..self.config.training_config.num_epochs {
            if self.state.should_stop {
                break;
            }

            self.state.epoch = epoch;
            self.state.step = 0;

            let _epoch_metrics = self.train_epoch(train_loader)?;

            // Save checkpoint after each epoch
            if epoch % 5 == 0 || epoch == self.config.training_config.num_epochs - 1 {
                self.save_checkpoint()?;
            }
        }

        tracing::info!("Training completed!");
        Ok(())
    }

    /// Log training progress
    fn log_training_progress(&self, recent_metrics: &[TrainingMetrics]) -> Result<()> {
        if let Some(last_metric) = recent_metrics.last() {
            let steps_per_sec = 1.0 / last_metric.step_time;

            tracing::info!(
                "Step {}: Loss={:.4}, LR={:.2e}, GradNorm={:.4}, Steps/sec={:.2}",
                self.state.global_step,
                last_metric.train_loss,
                last_metric.learning_rate,
                last_metric.grad_norm,
                steps_per_sec
            );
        }

        Ok(())
    }

    /// Check if training should stop early
    fn should_stop_early(&self) -> bool {
        if let Some(patience) = self.config.training_config.early_stopping_patience {
            self.state.steps_since_best >= patience
        } else {
            false
        }
    }

    /// Save training checkpoint
    fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_dir = format!(
            "{}/checkpoint-{}",
            self.config.training_config.output_dir, self.state.global_step
        );

        // Create directory
        std::fs::create_dir_all(&checkpoint_dir).map_err(MlxRetrievalError::from)?;

        // Save model weights
        let model_path = format!("{checkpoint_dir}/model.safetensors");
        self.model.save_weights(&model_path)?;

        // Save training state
        let checkpoint = TrainingCheckpoint::new(
            self.state.epoch,
            self.state.global_step,
            self.state.best_metric,
            self.config.training_config.clone(),
        );

        let checkpoint_path = format!("{checkpoint_dir}/training_state.json");
        checkpoint.save(&checkpoint_path)?;

        tracing::info!("Saved checkpoint to {}", checkpoint_dir);
        Ok(())
    }

    /// Resume from checkpoint
    fn resume_from_checkpoint(&mut self, checkpoint_path: &str) -> Result<()> {
        // Load training state
        let state_path = format!("{checkpoint_path}/training_state.json");
        let checkpoint = TrainingCheckpoint::load(&state_path)?;

        self.state.epoch = checkpoint.epoch;
        self.state.global_step = checkpoint.step;
        self.state.best_metric = checkpoint.best_metric;

        // Load model weights
        let model_path = format!("{checkpoint_path}/model.safetensors");
        self.model.load_weights(&model_path)?;

        tracing::info!("Resumed training from checkpoint: {}", checkpoint_path);
        Ok(())
    }

    /// Get current training state
    pub fn get_state(&self) -> &TrainingState {
        &self.state
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> &[EpochMetrics] {
        &self.metrics_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::data::BatchConfig;
    use crate::loss::{InfoNceConfig, InfoNceLoss};
    use crate::model::{ModelConfig, ModelFactory};
    use crate::training::optimizer::{OptimizerConfig, OptimizerFactory};

    #[test]
    fn test_training_state_creation() {
        let state = TrainingState::new();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.global_step, 0);
        assert!(!state.should_stop);
    }

    #[test]
    fn test_training_metrics_creation() {
        let mut metrics = TrainingMetrics::new(0.5, 0.001, 1.2, 0.1);
        assert_eq!(metrics.train_loss, 0.5);
        assert_eq!(metrics.learning_rate, 0.001);
        assert_eq!(metrics.grad_norm, 1.2);
        assert_eq!(metrics.step_time, 0.1);

        metrics.add_metric("accuracy".to_string(), 0.85);
        assert_eq!(metrics.custom_metrics.get("accuracy"), Some(&0.85));
    }

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert!(!config.mixed_precision);
        assert!(!config.compile_model);
        assert!(config.save_optimizer_state);
        assert!(config.resume_from_checkpoint.is_none());
    }

    #[tokio::test]
    async fn test_trainer_creation() -> Result<()> {
        let trainer_config = TrainerConfig::default();
        let model_config = ModelConfig::default();
        let model = ModelFactory::create_model(model_config)?;
        let loss_fn = Box::new(InfoNceLoss::new(InfoNceConfig::default()));
        let optimizer_config = OptimizerConfig::default();
        let optimizer = OptimizerFactory::create(optimizer_config)?;

        let trainer = Trainer::new(trainer_config, model, loss_fn, optimizer);
        assert_eq!(trainer.state.epoch, 0);
        assert_eq!(trainer.state.global_step, 0);

        Ok(())
    }
}
