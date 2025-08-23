//! Optimizers for training neural networks
//!
//! This module provides various optimization algorithms for training
//! embedding models with MLX.

use crate::error::Result;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adam optimizer
    Adam,
    /// AdamW optimizer (Adam with weight decay)
    AdamW,
    /// RMSprop optimizer
    RMSprop,
    /// AdaGrad optimizer
    AdaGrad,
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::SGD => write!(f, "sgd"),
            OptimizerType::Adam => write!(f, "adam"),
            OptimizerType::AdamW => write!(f, "adamw"),
            OptimizerType::RMSprop => write!(f, "rmsprop"),
            OptimizerType::AdaGrad => write!(f, "adagrad"),
        }
    }
}

/// Base optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Type of optimizer
    pub optimizer_type: OptimizerType,

    /// Learning rate
    pub learning_rate: f32,

    /// Weight decay (L2 regularization)
    pub weight_decay: f32,

    /// Optimizer-specific parameters
    pub params: OptimizerParams,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 2e-5,
            weight_decay: 0.01,
            params: OptimizerParams::AdamW(AdamWConfig::default()),
        }
    }
}

/// Optimizer-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizerParams {
    SGD(SGDConfig),
    Adam(AdamConfig),
    AdamW(AdamWConfig),
    RMSprop(RMSpropConfig),
    AdaGrad(AdaGradConfig),
}

/// SGD optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDConfig {
    /// Momentum factor
    pub momentum: f32,

    /// Nesterov momentum
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            momentum: 0.9,
            nesterov: false,
        }
    }
}

/// Adam optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamConfig {
    /// Beta1 parameter (momentum)
    pub beta1: f32,

    /// Beta2 parameter (RMSprop)
    pub beta2: f32,

    /// Epsilon for numerical stability
    pub eps: f32,

    /// AMSGrad variant
    pub amsgrad: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        }
    }
}

/// AdamW optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    /// Beta1 parameter (momentum)
    pub beta1: f32,

    /// Beta2 parameter (RMSprop)
    pub beta2: f32,

    /// Epsilon for numerical stability
    pub eps: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// RMSprop optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSpropConfig {
    /// Decay rate
    pub alpha: f32,

    /// Epsilon for numerical stability
    pub eps: f32,

    /// Momentum factor
    pub momentum: f32,

    /// Whether to center the gradient
    pub centered: bool,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            centered: false,
        }
    }
}

/// AdaGrad optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaGradConfig {
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl Default for AdaGradConfig {
    fn default() -> Self {
        Self { eps: 1e-8 }
    }
}

/// Optimizer state for a single parameter
#[derive(Debug, Clone)]
pub struct ParameterState {
    /// Momentum buffer (for SGD, Adam)
    pub momentum: Option<Array>,

    /// Squared gradient accumulator (for Adam, RMSprop, AdaGrad)
    pub exp_avg_sq: Option<Array>,

    /// Maximum squared gradient (for AMSGrad)
    pub max_exp_avg_sq: Option<Array>,

    /// Step count
    pub step: usize,
}

impl ParameterState {
    /// Create new parameter state
    pub fn new() -> Self {
        Self {
            momentum: None,
            exp_avg_sq: None,
            max_exp_avg_sq: None,
            step: 0,
        }
    }
}

impl Default for ParameterState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main optimizer trait
pub trait Optimizer {
    /// Update parameters given gradients
    fn step(&mut self, params: &mut [Array], grads: &[Array]) -> Result<()>;

    /// Zero gradients (not needed for MLX but kept for interface compatibility)
    fn zero_grad(&mut self) {}

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);

    /// Get optimizer configuration
    fn config(&self) -> &OptimizerConfig;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// SGD optimizer implementation
pub struct SGDOptimizer {
    config: OptimizerConfig,
    sgd_config: SGDConfig,
    state: HashMap<String, ParameterState>,
}

impl SGDOptimizer {
    pub fn new(config: OptimizerConfig, sgd_config: SGDConfig) -> Self {
        Self {
            config,
            sgd_config,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, params: &mut [Array], grads: &[Array]) -> Result<()> {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let param_key = format!("param_{i}");
            let state = self.state.entry(param_key).or_default();

            // Apply weight decay
            let mut grad = grad.clone();
            if self.config.weight_decay != 0.0 {
                grad = grad.add(&param.multiply(Array::from(self.config.weight_decay))?)?;
            }

            // Apply momentum
            if self.sgd_config.momentum != 0.0 {
                if state.momentum.is_none() {
                    state.momentum = Some(grad.clone());
                } else {
                    let momentum = state.momentum.as_ref().unwrap();
                    state.momentum = Some(
                        momentum
                            .multiply(Array::from(self.sgd_config.momentum))?
                            .add(&grad)?,
                    );
                }
                grad = state.momentum.as_ref().unwrap().clone();
            }

            // Update parameter
            *param = param.subtract(&grad.multiply(Array::from(self.config.learning_rate))?)?;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    fn reset(&mut self) {
        self.state.clear();
    }
}

/// AdamW optimizer implementation
pub struct AdamWOptimizer {
    config: OptimizerConfig,
    adam_config: AdamWConfig,
    state: HashMap<String, ParameterState>,
}

impl AdamWOptimizer {
    pub fn new(config: OptimizerConfig, adam_config: AdamWConfig) -> Self {
        Self {
            config,
            adam_config,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for AdamWOptimizer {
    fn step(&mut self, params: &mut [Array], grads: &[Array]) -> Result<()> {
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let param_key = format!("param_{i}");
            let state = self.state.entry(param_key).or_default();

            state.step += 1;

            // Initialize momentum and squared gradient accumulators
            if state.momentum.is_none() {
                state.momentum = Some(mlx_rs::ops::zeros_like(grad)?);
                state.exp_avg_sq = Some(mlx_rs::ops::zeros_like(grad)?);
            }

            let momentum = state.momentum.as_mut().unwrap();
            let exp_avg_sq = state.exp_avg_sq.as_mut().unwrap();

            // Update biased first moment estimate
            *momentum = momentum
                .multiply(Array::from(self.adam_config.beta1))?
                .add(&grad.multiply(Array::from(1.0 - self.adam_config.beta1))?)?;

            // Update biased second raw moment estimate
            *exp_avg_sq = exp_avg_sq
                .multiply(Array::from(self.adam_config.beta2))?
                .add(
                    &grad
                        .square()?
                        .multiply(Array::from(1.0 - self.adam_config.beta2))?,
                )?;

            // Compute bias-corrected estimates
            let bias_correction1 = 1.0 - self.adam_config.beta1.powi(state.step as i32);
            let bias_correction2 = 1.0 - self.adam_config.beta2.powi(state.step as i32);

            let corrected_momentum = momentum.divide(Array::from(bias_correction1))?;
            let corrected_exp_avg_sq = exp_avg_sq.divide(Array::from(bias_correction2))?;

            // Compute update
            let denominator = corrected_exp_avg_sq
                .sqrt()?
                .add(Array::from(self.adam_config.eps))?;
            let update = corrected_momentum.divide(&denominator)?;

            // Apply weight decay (AdamW style - decoupled)
            if self.config.weight_decay != 0.0 {
                *param = param.multiply(Array::from(
                    1.0 - self.config.learning_rate * self.config.weight_decay,
                ))?;
            }

            // Update parameter
            *param = param.subtract(&update.multiply(Array::from(self.config.learning_rate))?)?;
        }

        Ok(())
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    fn reset(&mut self) {
        self.state.clear();
    }
}

/// Factory for creating optimizers
pub struct OptimizerFactory;

impl OptimizerFactory {
    pub fn create(config: OptimizerConfig) -> Result<Box<dyn Optimizer>> {
        match config.optimizer_type {
            OptimizerType::SGD => {
                let sgd_config = match &config.params {
                    OptimizerParams::SGD(cfg) => cfg.clone(),
                    _ => SGDConfig::default(),
                };
                Ok(Box::new(SGDOptimizer::new(config, sgd_config)))
            }
            OptimizerType::AdamW => {
                let adam_config = match &config.params {
                    OptimizerParams::AdamW(cfg) => cfg.clone(),
                    _ => AdamWConfig::default(),
                };
                Ok(Box::new(AdamWOptimizer::new(config, adam_config)))
            }
            _ => Err(crate::error::MlxRetrievalError::model(format!(
                "Optimizer {:?} not implemented yet",
                config.optimizer_type
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.optimizer_type, OptimizerType::AdamW);
        assert_eq!(config.learning_rate, 2e-5);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_optimizer_type_display() {
        assert_eq!(OptimizerType::SGD.to_string(), "sgd");
        assert_eq!(OptimizerType::Adam.to_string(), "adam");
        assert_eq!(OptimizerType::AdamW.to_string(), "adamw");
    }

    #[test]
    fn test_parameter_state_creation() {
        let state = ParameterState::new();
        assert!(state.momentum.is_none());
        assert!(state.exp_avg_sq.is_none());
        assert_eq!(state.step, 0);
    }

    #[tokio::test]
    async fn test_sgd_optimizer_creation() -> Result<()> {
        let config = OptimizerConfig {
            optimizer_type: OptimizerType::SGD,
            learning_rate: 0.01,
            weight_decay: 0.001,
            params: OptimizerParams::SGD(SGDConfig::default()),
        };

        let optimizer = OptimizerFactory::create(config)?;
        assert_eq!(optimizer.get_lr(), 0.01);

        Ok(())
    }

    #[tokio::test]
    async fn test_adamw_optimizer_step() -> Result<()> {
        let config = OptimizerConfig::default();
        let mut optimizer = OptimizerFactory::create(config)?;

        let mut params = vec![Array::ones::<f32>(&[10, 10])?];
        let grads = vec![mlx_rs::random::normal::<f32>(&[10, 10], None, None, None)?];

        // Should not panic
        optimizer.step(&mut params, &grads)?;

        Ok(())
    }
}
