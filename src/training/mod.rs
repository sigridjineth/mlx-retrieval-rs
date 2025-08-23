//! Training utilities and optimization
//!
//! This module provides comprehensive training infrastructure including
//! optimizers, learning rate schedules, and training loops for embedding models.

pub mod optimizer;
pub mod trainer;

// Re-export common types
pub use optimizer::{
    AdamConfig, AdamWConfig, Optimizer, OptimizerConfig, OptimizerType, SGDConfig,
};
pub use trainer::{EpochMetrics, Trainer, TrainerConfig, TrainingMetrics, TrainingState};

use crate::error::Result;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Weight decay (L2 regularization)
    pub weight_decay: f32,

    /// Gradient clipping threshold
    pub max_grad_norm: f32,

    /// Warmup steps for learning rate
    pub warmup_steps: usize,

    /// Evaluation frequency (in steps)
    pub eval_steps: usize,

    /// Logging frequency (in steps)
    pub logging_steps: usize,

    /// Save model frequency (in steps)
    pub save_steps: usize,

    /// Maximum number of training steps
    pub max_steps: Option<usize>,

    /// Early stopping patience (in evaluation steps)
    pub early_stopping_patience: Option<usize>,

    /// Metric to use for early stopping
    pub early_stopping_metric: String,

    /// Whether higher is better for early stopping metric
    pub early_stopping_higher_is_better: bool,

    /// Output directory for checkpoints
    pub output_dir: String,

    /// Mixed precision training
    pub mixed_precision: bool,

    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            batch_size: 32,
            learning_rate: 2e-5,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            warmup_steps: 0,
            eval_steps: 500,
            logging_steps: 100,
            save_steps: 500,
            max_steps: None,
            early_stopping_patience: None,
            early_stopping_metric: "eval_loss".to_string(),
            early_stopping_higher_is_better: false,
            output_dir: "./output".to_string(),
            mixed_precision: false,
            gradient_accumulation_steps: 1,
            seed: None,
        }
    }
}

/// Learning rate schedule types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LRScheduleType {
    /// Constant learning rate
    Constant,
    /// Linear warmup followed by linear decay
    LinearWithWarmup,
    /// Cosine annealing with warmup
    CosineWithWarmup,
    /// Exponential decay
    ExponentialDecay,
    /// Step-wise decay
    StepLR,
    /// Polynomial decay
    PolynomialDecay,
}

impl std::fmt::Display for LRScheduleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LRScheduleType::Constant => write!(f, "constant"),
            LRScheduleType::LinearWithWarmup => write!(f, "linear_with_warmup"),
            LRScheduleType::CosineWithWarmup => write!(f, "cosine_with_warmup"),
            LRScheduleType::ExponentialDecay => write!(f, "exponential_decay"),
            LRScheduleType::StepLR => write!(f, "step_lr"),
            LRScheduleType::PolynomialDecay => write!(f, "polynomial_decay"),
        }
    }
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedulerConfig {
    /// Type of learning rate schedule
    pub schedule_type: LRScheduleType,

    /// Number of warmup steps
    pub warmup_steps: usize,

    /// Total number of training steps
    pub total_steps: usize,

    /// Minimum learning rate (for decay schedules)
    pub min_lr: f32,

    /// Decay rate (for exponential decay)
    pub decay_rate: f32,

    /// Step size (for step LR)
    pub step_size: usize,

    /// Gamma (for step LR)
    pub gamma: f32,

    /// Power (for polynomial decay)
    pub power: f32,
}

impl Default for LRSchedulerConfig {
    fn default() -> Self {
        Self {
            schedule_type: LRScheduleType::LinearWithWarmup,
            warmup_steps: 0,
            total_steps: 10000,
            min_lr: 0.0,
            decay_rate: 0.95,
            step_size: 1000,
            gamma: 0.1,
            power: 1.0,
        }
    }
}

/// Learning rate scheduler
pub struct LRScheduler {
    config: LRSchedulerConfig,
    base_lr: f32,
    current_step: usize,
}

impl LRScheduler {
    /// Create a new learning rate scheduler
    pub fn new(config: LRSchedulerConfig, base_lr: f32) -> Self {
        Self {
            config,
            base_lr,
            current_step: 0,
        }
    }

    /// Get the learning rate for the current step
    pub fn get_lr(&self) -> f32 {
        match self.config.schedule_type {
            LRScheduleType::Constant => self.base_lr,
            LRScheduleType::LinearWithWarmup => self.linear_with_warmup(),
            LRScheduleType::CosineWithWarmup => self.cosine_with_warmup(),
            LRScheduleType::ExponentialDecay => self.exponential_decay(),
            LRScheduleType::StepLR => self.step_lr(),
            LRScheduleType::PolynomialDecay => self.polynomial_decay(),
        }
    }

    /// Update the current step and return new learning rate
    pub fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    /// Reset the scheduler to step 0
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Set the current step
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Linear warmup followed by linear decay
    fn linear_with_warmup(&self) -> f32 {
        let step = self.current_step as f32;
        let warmup_steps = self.config.warmup_steps as f32;
        let total_steps = self.config.total_steps as f32;

        if step < warmup_steps {
            // Warmup phase
            self.base_lr * (step / warmup_steps)
        } else {
            // Linear decay phase
            let decay_steps = total_steps - warmup_steps;
            let decay_progress = (step - warmup_steps) / decay_steps;
            let decay_factor = 1.0 - decay_progress.min(1.0);

            (self.base_lr * decay_factor).max(self.config.min_lr)
        }
    }

    /// Cosine annealing with warmup
    fn cosine_with_warmup(&self) -> f32 {
        let step = self.current_step as f32;
        let warmup_steps = self.config.warmup_steps as f32;
        let total_steps = self.config.total_steps as f32;

        if step < warmup_steps {
            // Warmup phase
            self.base_lr * (step / warmup_steps)
        } else {
            // Cosine annealing phase
            let decay_steps = total_steps - warmup_steps;
            let decay_progress = ((step - warmup_steps) / decay_steps).min(1.0);
            let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * decay_progress).cos());

            let lr = self.config.min_lr + (self.base_lr - self.config.min_lr) * cosine_factor;
            lr.max(self.config.min_lr)
        }
    }

    /// Exponential decay
    fn exponential_decay(&self) -> f32 {
        let decay_factor = self.config.decay_rate.powf(self.current_step as f32);
        (self.base_lr * decay_factor).max(self.config.min_lr)
    }

    /// Step-wise learning rate decay
    fn step_lr(&self) -> f32 {
        let num_decays = self.current_step / self.config.step_size;
        let decay_factor = self.config.gamma.powf(num_decays as f32);
        (self.base_lr * decay_factor).max(self.config.min_lr)
    }

    /// Polynomial decay
    fn polynomial_decay(&self) -> f32 {
        let step = self.current_step.min(self.config.total_steps) as f32;
        let total_steps = self.config.total_steps as f32;

        let decay_factor = (1.0 - step / total_steps).powf(self.config.power);
        let lr = self.config.min_lr + (self.base_lr - self.config.min_lr) * decay_factor;
        lr.max(self.config.min_lr)
    }
}

/// Gradient clipping utilities
pub struct GradientClipper {
    max_norm: f32,
}

impl GradientClipper {
    /// Create a new gradient clipper
    pub fn new(max_norm: f32) -> Self {
        Self { max_norm }
    }

    /// Clip gradients by global norm
    pub fn clip_grad_norm(&self, gradients: &mut [Array]) -> Result<f32> {
        // Compute global norm
        let mut total_norm_squared = 0.0f32;

        for grad in gradients.iter() {
            let grad_norm_squared = grad.square()?.sum(None, false)?.item::<f32>();
            total_norm_squared += grad_norm_squared;
        }

        let global_norm = total_norm_squared.sqrt();

        // Clip if necessary
        if global_norm > self.max_norm {
            let clip_factor = self.max_norm / global_norm;

            for grad in gradients.iter_mut() {
                *grad = grad.multiply(Array::from(clip_factor))?;
            }
        }

        Ok(global_norm)
    }

    /// Clip gradients by value
    pub fn clip_grad_value(&self, gradients: &mut [Array], clip_value: f32) -> Result<()> {
        let min_val = Array::from(-clip_value);
        let max_val = Array::from(clip_value);

        for grad in gradients.iter_mut() {
            *grad = mlx_rs::ops::maximum(&mlx_rs::ops::minimum(&*grad, &max_val)?, &min_val)?;
        }

        Ok(())
    }
}

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current epoch
    pub epoch: usize,

    /// Current step
    pub step: usize,

    /// Best metric value seen so far
    pub best_metric: f32,

    /// Training metrics history
    pub metrics_history: Vec<EpochMetrics>,

    /// Random number generator state
    pub rng_state: Option<String>,

    /// Training configuration
    pub training_config: TrainingConfig,

    /// Model configuration path
    pub model_config_path: Option<String>,

    /// Timestamp of checkpoint creation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TrainingCheckpoint {
    /// Create a new training checkpoint
    pub fn new(
        epoch: usize,
        step: usize,
        best_metric: f32,
        training_config: TrainingConfig,
    ) -> Self {
        Self {
            epoch,
            step,
            best_metric,
            metrics_history: Vec::new(),
            rng_state: None,
            training_config,
            model_config_path: None,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add metrics to history
    pub fn add_metrics(&mut self, metrics: EpochMetrics) {
        self.metrics_history.push(metrics);
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json_str =
            serde_json::to_string_pretty(self).map_err(crate::error::MlxRetrievalError::from)?;

        std::fs::write(path, json_str).map_err(crate::error::MlxRetrievalError::from)?;

        tracing::info!("Saved training checkpoint to {}", path);
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &str) -> Result<Self> {
        let json_str =
            std::fs::read_to_string(path).map_err(crate::error::MlxRetrievalError::from)?;

        let checkpoint: Self =
            serde_json::from_str(&json_str).map_err(crate::error::MlxRetrievalError::from)?;

        tracing::info!("Loaded training checkpoint from {}", path);
        Ok(checkpoint)
    }
}

/// Training utilities
pub struct TrainingUtils;

impl TrainingUtils {
    /// Set random seeds for reproducibility
    pub fn set_seed(seed: u64) -> Result<()> {
        // Set MLX random seed
        mlx_rs::random::seed(seed)?;

        tracing::info!("Set random seed to {}", seed);
        Ok(())
    }

    /// Calculate model parameter count
    pub fn count_parameters(model: &dyn crate::model::Model) -> usize {
        model.num_parameters()
    }

    /// Calculate training steps per epoch
    pub fn steps_per_epoch(dataset_size: usize, batch_size: usize) -> usize {
        dataset_size.div_ceil(batch_size)
    }

    /// Calculate total training steps
    pub fn total_training_steps(
        dataset_size: usize,
        batch_size: usize,
        num_epochs: usize,
    ) -> usize {
        Self::steps_per_epoch(dataset_size, batch_size) * num_epochs
    }

    /// Format training time
    pub fn format_training_time(seconds: f64) -> String {
        if seconds < 60.0 {
            format!("{seconds:.1}s")
        } else if seconds < 3600.0 {
            let minutes = seconds / 60.0;
            format!("{minutes:.1}m")
        } else {
            let hours = seconds / 3600.0;
            format!("{hours:.1}h")
        }
    }

    /// Calculate memory usage (placeholder)
    pub fn get_memory_usage() -> Result<(usize, usize)> {
        // Placeholder for memory usage calculation
        // In practice, this would query MLX memory usage
        Ok((0, 0)) // (used, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.learning_rate, 2e-5);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_lr_scheduler_constant() {
        let config = LRSchedulerConfig {
            schedule_type: LRScheduleType::Constant,
            ..Default::default()
        };
        let scheduler = LRScheduler::new(config, 0.001);

        assert_eq!(scheduler.get_lr(), 0.001);
    }

    #[test]
    fn test_lr_scheduler_linear_warmup() {
        let config = LRSchedulerConfig {
            schedule_type: LRScheduleType::LinearWithWarmup,
            warmup_steps: 100,
            total_steps: 1000,
            min_lr: 0.0,
            ..Default::default()
        };
        let mut scheduler = LRScheduler::new(config, 0.001);

        // At step 0, should be 0
        assert_eq!(scheduler.get_lr(), 0.0);

        // At step 50 (half warmup), should be half learning rate
        scheduler.set_step(50);
        assert!((scheduler.get_lr() - 0.0005).abs() < 1e-6);

        // At step 100 (end of warmup), should be full learning rate
        scheduler.set_step(100);
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_step() {
        let mut scheduler = LRScheduler::new(LRSchedulerConfig::default(), 0.001);

        let lr1 = scheduler.step();
        let lr2 = scheduler.step();

        // Learning rates should be calculated for consecutive steps
        assert!(lr1 >= 0.0);
        assert!(lr2 >= 0.0);
    }

    #[test]
    fn test_gradient_clipper_creation() {
        let clipper = GradientClipper::new(1.0);
        assert_eq!(clipper.max_norm, 1.0);
    }

    #[test]
    fn test_training_utils_steps_calculation() {
        assert_eq!(TrainingUtils::steps_per_epoch(100, 32), 4); // 100/32 rounded up
        assert_eq!(TrainingUtils::steps_per_epoch(96, 32), 3); // Exact division
        assert_eq!(TrainingUtils::total_training_steps(100, 32, 3), 12); // 4 steps * 3 epochs
    }

    #[test]
    fn test_training_utils_time_formatting() {
        assert_eq!(TrainingUtils::format_training_time(30.5), "30.5s");
        assert_eq!(TrainingUtils::format_training_time(90.0), "1.5m");
        assert_eq!(TrainingUtils::format_training_time(3720.0), "1.0h");
    }

    #[test]
    fn test_training_checkpoint_creation() {
        let config = TrainingConfig::default();
        let checkpoint = TrainingCheckpoint::new(1, 100, 0.5, config);

        assert_eq!(checkpoint.epoch, 1);
        assert_eq!(checkpoint.step, 100);
        assert_eq!(checkpoint.best_metric, 0.5);
        assert!(checkpoint.metrics_history.is_empty());
    }

    #[test]
    fn test_lr_schedule_type_display() {
        assert_eq!(LRScheduleType::Constant.to_string(), "constant");
        assert_eq!(
            LRScheduleType::LinearWithWarmup.to_string(),
            "linear_with_warmup"
        );
        assert_eq!(
            LRScheduleType::CosineWithWarmup.to_string(),
            "cosine_with_warmup"
        );
    }
}
