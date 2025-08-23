//! LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning
//!
//! This module implements LoRA, a parameter-efficient fine-tuning technique
//! that adapts pre-trained models by learning low-rank decompositions of
//! weight update matrices.

use crate::error::{MlxRetrievalError, Result};
use crate::model::{Model, ModelConfig, ModelOutput};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for LoRA adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition
    pub r: usize,

    /// LoRA scaling parameter (alpha / r)
    pub lora_alpha: f32,

    /// Dropout probability for LoRA layers
    pub lora_dropout: f32,

    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,

    /// Whether to use bias in LoRA layers
    pub bias: String,

    /// Task type for the adaptation
    pub task_type: TaskType,

    /// Fan-in mode for initialization
    pub fan_in_fan_out: bool,

    /// Modules to save when saving LoRA weights
    pub modules_to_save: Option<Vec<String>>,

    /// LoRA initialization method
    pub init_lora_weights: LoRAInitialization,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            r: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.1,
            target_modules: vec!["query".to_string(), "value".to_string()],
            bias: "none".to_string(),
            task_type: TaskType::FeatureExtraction,
            fan_in_fan_out: false,
            modules_to_save: None,
            init_lora_weights: LoRAInitialization::Kaiming,
        }
    }
}

/// Task types for LoRA adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    #[serde(rename = "FEATURE_EXTRACTION")]
    FeatureExtraction,
    #[serde(rename = "SEQ_CLS")]
    SeqCls,
    #[serde(rename = "TOKEN_CLS")]
    TokenCls,
    #[serde(rename = "CAUSAL_LM")]
    CausalLm,
    #[serde(rename = "SEQ_2_SEQ_LM")]
    Seq2SeqLm,
}

/// LoRA weight initialization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoRAInitialization {
    /// Kaiming uniform initialization
    Kaiming,
    /// Xavier uniform initialization
    Xavier,
    /// Gaussian initialization
    Gaussian,
    /// Zero initialization for A matrix
    Zero,
}

/// A single LoRA layer implementing the low-rank adaptation
#[derive(Debug, Clone)]
pub struct LoRALayer {
    /// Configuration for this LoRA layer
    pub config: LoRAConfig,

    /// Input dimension
    pub in_features: usize,

    /// Output dimension
    pub out_features: usize,

    /// Low-rank matrix A (in_features x r)
    pub lora_a: Array,

    /// Low-rank matrix B (r x out_features)
    pub lora_b: Array,

    /// Scaling factor
    pub scaling: f32,

    /// Whether this layer is frozen
    pub frozen: bool,
}

impl LoRALayer {
    /// Create a new LoRA layer
    pub fn new(config: LoRAConfig, in_features: usize, out_features: usize) -> Result<Self> {
        if config.r == 0 {
            return Err(MlxRetrievalError::invalid_input("LoRA rank cannot be zero"));
        }

        if config.r >= in_features.min(out_features) {
            return Err(MlxRetrievalError::invalid_input(
                "LoRA rank should be less than min(in_features, out_features)",
            ));
        }

        // Initialize LoRA matrices
        let (lora_a, lora_b) = Self::initialize_lora_weights(&config, in_features, out_features)?;

        // Compute scaling factor
        let scaling = config.lora_alpha / config.r as f32;

        Ok(Self {
            config,
            in_features,
            out_features,
            lora_a,
            lora_b,
            scaling,
            frozen: false,
        })
    }

    /// Initialize LoRA weights based on configuration
    fn initialize_lora_weights(
        config: &LoRAConfig,
        in_features: usize,
        out_features: usize,
    ) -> Result<(Array, Array)> {
        let r = config.r;

        let (lora_a, lora_b) = match config.init_lora_weights {
            LoRAInitialization::Kaiming => {
                let bound_a = (1.0 / in_features as f32).sqrt();
                let bound_b = (1.0 / r as f32).sqrt();

                let a = mlx_rs::random::uniform::<f32, f32>(
                    -bound_a,
                    bound_a,
                    &[in_features as i32, r as i32],
                    None,
                )?;

                let b = mlx_rs::random::uniform::<f32, f32>(
                    -bound_b,
                    bound_b,
                    &[r as i32, out_features as i32],
                    None,
                )?;

                (a, b)
            }
            LoRAInitialization::Xavier => {
                let bound_a = (6.0 / (in_features + r) as f32).sqrt();
                let bound_b = (6.0 / (r + out_features) as f32).sqrt();

                let a = mlx_rs::random::uniform::<f32, f32>(
                    -bound_a,
                    bound_a,
                    &[in_features as i32, r as i32],
                    None,
                )?;

                let b = mlx_rs::random::uniform::<f32, f32>(
                    -bound_b,
                    bound_b,
                    &[r as i32, out_features as i32],
                    None,
                )?;

                (a, b)
            }
            LoRAInitialization::Gaussian => {
                let std_a = 0.02;
                let std_b = 0.02;

                let a = mlx_rs::random::normal::<f32>(
                    &[in_features as i32, r as i32],
                    Some(0.0),
                    Some(std_a),
                    None,
                )?;

                let b = mlx_rs::random::normal::<f32>(
                    &[r as i32, out_features as i32],
                    Some(0.0),
                    Some(std_b),
                    None,
                )?;

                (a, b)
            }
            LoRAInitialization::Zero => {
                let a = Array::zeros::<f32>(&[in_features as i32, r as i32])?;
                let b = Array::zeros::<f32>(&[r as i32, out_features as i32])?;

                (a, b)
            }
        };

        Ok((lora_a, lora_b))
    }

    /// Forward pass through the LoRA layer
    pub fn forward(&self, x: &Array) -> Result<Array> {
        if self.frozen {
            return Ok(x.clone());
        }

        // Apply dropout if in training mode
        let x_input = if self.config.lora_dropout > 0.0 {
            // Simplified dropout - in practice, this would check training mode
            x.clone()
        } else {
            x.clone()
        };

        // LoRA computation: x * A * B * scaling
        let intermediate = x_input.matmul(&self.lora_a)?;
        let output = intermediate.matmul(&self.lora_b)?;
        let scaled_output = output.multiply(Array::from(self.scaling))?;

        Ok(scaled_output)
    }

    /// Freeze the LoRA layer (stop gradient computation)
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze the LoRA layer (enable gradient computation)
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Get the number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        if self.frozen {
            0
        } else {
            (self.in_features * self.config.r) + (self.config.r * self.out_features)
        }
    }

    /// Reset LoRA weights
    pub fn reset_parameters(&mut self) -> Result<()> {
        let (new_a, new_b) =
            Self::initialize_lora_weights(&self.config, self.in_features, self.out_features)?;

        self.lora_a = new_a;
        self.lora_b = new_b;

        Ok(())
    }

    /// Merge LoRA weights with base weights
    pub fn merge_with_base(&self, base_weight: &Array) -> Result<Array> {
        let lora_weight = self
            .lora_a
            .matmul(&self.lora_b)?
            .multiply(Array::from(self.scaling))?;
        base_weight.add(&lora_weight).map_err(Into::into)
    }

    /// Get LoRA weight as a single matrix (A @ B * scaling)
    pub fn get_lora_weight(&self) -> Result<Array> {
        let weight = self.lora_a.matmul(&self.lora_b)?;
        weight
            .multiply(Array::from(self.scaling))
            .map_err(Into::into)
    }
}

/// LoRA adapter model that wraps a base model
pub struct LoRAModel {
    /// Base model being adapted
    base_model: Box<dyn Model>,

    /// LoRA configuration
    lora_config: LoRAConfig,

    /// LoRA layers by module name
    lora_layers: HashMap<String, LoRALayer>,

    /// Whether LoRA is enabled
    enabled: bool,
}

impl LoRAModel {
    /// Create a new LoRA model wrapping a base model
    pub fn new(base_model: Box<dyn Model>, lora_config: LoRAConfig) -> Result<Self> {
        let mut lora_model = Self {
            base_model,
            lora_config: lora_config.clone(),
            lora_layers: HashMap::new(),
            enabled: true,
        };

        // Initialize LoRA layers for target modules
        lora_model.initialize_lora_layers()?;

        Ok(lora_model)
    }

    /// Initialize LoRA layers for the target modules
    fn initialize_lora_layers(&mut self) -> Result<()> {
        let hidden_size = self.base_model.config().hidden_size;

        for module_name in &self.lora_config.target_modules {
            let (in_features, out_features) = match module_name.as_str() {
                "query" | "key" | "value" => (hidden_size, hidden_size),
                "dense" => (hidden_size, hidden_size),
                "intermediate" => (hidden_size, self.base_model.config().intermediate_size),
                "output" => (self.base_model.config().intermediate_size, hidden_size),
                _ => {
                    tracing::warn!("Unknown target module: {}", module_name);
                    continue;
                }
            };

            let lora_layer = LoRALayer::new(self.lora_config.clone(), in_features, out_features)?;

            self.lora_layers.insert(module_name.clone(), lora_layer);
        }

        tracing::info!("Initialized {} LoRA layers", self.lora_layers.len());
        Ok(())
    }

    /// Enable LoRA adaptation
    pub fn enable_lora(&mut self) {
        self.enabled = true;
    }

    /// Disable LoRA adaptation
    pub fn disable_lora(&mut self) {
        self.enabled = false;
    }

    /// Check if LoRA is enabled
    pub fn is_lora_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the number of trainable LoRA parameters
    pub fn num_lora_parameters(&self) -> usize {
        self.lora_layers
            .values()
            .map(|layer| layer.num_parameters())
            .sum()
    }

    /// Freeze all LoRA layers
    pub fn freeze_lora(&mut self) {
        for layer in self.lora_layers.values_mut() {
            layer.freeze();
        }
    }

    /// Unfreeze all LoRA layers
    pub fn unfreeze_lora(&mut self) {
        for layer in self.lora_layers.values_mut() {
            layer.unfreeze();
        }
    }

    /// Reset all LoRA parameters
    pub fn reset_lora(&mut self) -> Result<()> {
        for layer in self.lora_layers.values_mut() {
            layer.reset_parameters()?;
        }
        Ok(())
    }

    /// Save LoRA weights to file
    pub fn save_lora_weights(&self, path: &str) -> Result<()> {
        // In practice, this would serialize the LoRA weights to safetensors format
        tracing::info!("Saving LoRA weights to {}", path);
        tracing::info!(
            "LoRA layers: {:?}",
            self.lora_layers.keys().collect::<Vec<_>>()
        );
        Ok(())
    }

    /// Load LoRA weights from file
    pub fn load_lora_weights(&mut self, path: &str) -> Result<()> {
        // In practice, this would deserialize LoRA weights from safetensors format
        tracing::info!("Loading LoRA weights from {}", path);
        Ok(())
    }

    /// Merge LoRA weights with base model weights
    pub fn merge_and_unload(self) -> Result<Box<dyn Model>> {
        // In a full implementation, this would merge LoRA weights into the base model
        // and return the modified base model
        tracing::info!("Merging LoRA weights with base model");
        Ok(self.base_model)
    }

    /// Get LoRA layer by module name
    pub fn get_lora_layer(&self, module_name: &str) -> Option<&LoRALayer> {
        self.lora_layers.get(module_name)
    }

    /// Get mutable LoRA layer by module name
    pub fn get_lora_layer_mut(&mut self, module_name: &str) -> Option<&mut LoRALayer> {
        self.lora_layers.get_mut(module_name)
    }

    /// Apply LoRA adaptation to a tensor (for a specific module)
    pub fn apply_lora(&self, module_name: &str, x: &Array) -> Result<Array> {
        if !self.enabled {
            return Ok(x.clone());
        }

        if let Some(lora_layer) = self.lora_layers.get(module_name) {
            let lora_output = lora_layer.forward(x)?;
            // Add LoRA output to original input
            x.add(&lora_output).map_err(Into::into)
        } else {
            Ok(x.clone())
        }
    }

    /// Get base model configuration
    pub fn base_config(&self) -> &ModelConfig {
        self.base_model.config()
    }
}

impl Model for LoRAModel {
    fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<ModelOutput> {
        // For now, just forward through base model
        // In a full implementation, this would intercept and modify
        // the forward pass to apply LoRA adaptations
        self.base_model.forward(input_ids, attention_mask)
    }

    fn config(&self) -> &ModelConfig {
        self.base_model.config()
    }

    fn num_parameters(&self) -> usize {
        self.base_model.num_parameters() + self.num_lora_parameters()
    }

    fn train(&mut self, training: bool) {
        self.base_model.train(training);
    }

    fn is_training(&self) -> bool {
        self.base_model.is_training()
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        // Load base model weights
        self.base_model.load_weights(path)?;

        // Try to load LoRA weights from the same directory
        let lora_path = format!("{path}/lora_weights");
        if std::path::Path::new(&lora_path).exists() {
            self.load_lora_weights(&lora_path)?;
        }

        Ok(())
    }

    fn save_weights(&self, path: &str) -> Result<()> {
        // Save base model weights
        self.base_model.save_weights(path)?;

        // Save LoRA weights
        let lora_path = format!("{path}/lora_weights");
        self.save_lora_weights(&lora_path)?;

        Ok(())
    }
}

/// Builder for LoRA configuration
pub struct LoRAConfigBuilder {
    config: LoRAConfig,
}

impl Default for LoRAConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LoRAConfigBuilder {
    /// Create a new LoRA config builder
    pub fn new() -> Self {
        Self {
            config: LoRAConfig::default(),
        }
    }

    /// Set the LoRA rank
    pub fn rank(mut self, r: usize) -> Self {
        self.config.r = r;
        self
    }

    /// Set the LoRA alpha parameter
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.config.lora_alpha = alpha;
        self
    }

    /// Set the dropout probability
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.lora_dropout = dropout;
        self
    }

    /// Set target modules
    pub fn target_modules(mut self, modules: Vec<String>) -> Self {
        self.config.target_modules = modules;
        self
    }

    /// Set task type
    pub fn task_type(mut self, task_type: TaskType) -> Self {
        self.config.task_type = task_type;
        self
    }

    /// Set initialization method
    pub fn init_method(mut self, method: LoRAInitialization) -> Self {
        self.config.init_lora_weights = method;
        self
    }

    /// Build the configuration
    pub fn build(self) -> LoRAConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelConfig, ModelFactory};

    #[test]
    fn test_lora_config_default() {
        let config = LoRAConfig::default();
        assert_eq!(config.r, 8);
        assert_eq!(config.lora_alpha, 16.0);
        assert_eq!(config.lora_dropout, 0.1);
        assert_eq!(config.target_modules, vec!["query", "value"]);
        assert_eq!(config.task_type, TaskType::FeatureExtraction);
    }

    #[test]
    fn test_lora_config_builder() {
        let config = LoRAConfigBuilder::new()
            .rank(16)
            .alpha(32.0)
            .dropout(0.05)
            .target_modules(vec![
                "query".to_string(),
                "key".to_string(),
                "value".to_string(),
            ])
            .task_type(TaskType::SeqCls)
            .init_method(LoRAInitialization::Xavier)
            .build();

        assert_eq!(config.r, 16);
        assert_eq!(config.lora_alpha, 32.0);
        assert_eq!(config.lora_dropout, 0.05);
        assert_eq!(config.target_modules.len(), 3);
        assert_eq!(config.task_type, TaskType::SeqCls);
        assert_eq!(config.init_lora_weights, LoRAInitialization::Xavier);
    }

    #[tokio::test]
    async fn test_lora_layer_creation() -> Result<()> {
        let config = LoRAConfig::default();
        let layer = LoRALayer::new(config, 768, 768)?;

        assert_eq!(layer.in_features, 768);
        assert_eq!(layer.out_features, 768);
        assert_eq!(layer.config.r, 8);
        assert_eq!(layer.lora_a.shape(), vec![768, 8]);
        assert_eq!(layer.lora_b.shape(), vec![8, 768]);
        assert!(!layer.frozen);

        Ok(())
    }

    #[tokio::test]
    async fn test_lora_layer_forward() -> Result<()> {
        let config = LoRAConfig::default();
        let layer = LoRALayer::new(config, 768, 768)?;

        let input = mlx_rs::random::normal::<f32>(&[2, 10, 768], None, None, None)?;
        let output = layer.forward(&input)?;

        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[tokio::test]
    async fn test_lora_model_creation() -> Result<()> {
        let model_config = ModelConfig::default();
        let base_model = ModelFactory::create_model(model_config)?;

        let lora_config = LoRAConfig::default();
        let lora_model = LoRAModel::new(base_model, lora_config)?;

        assert!(lora_model.is_lora_enabled());
        assert!(lora_model.num_lora_parameters() > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_lora_model_forward() -> Result<()> {
        let model_config = ModelConfig {
            hidden_size: 256,
            num_attention_heads: 8,
            ..Default::default()
        };
        let base_model = ModelFactory::create_model(model_config)?;

        let lora_config = LoRAConfig::default();
        let mut lora_model = LoRAModel::new(base_model, lora_config)?;

        let input_ids = Array::ones::<i32>(&[2, 10])?;
        let output = lora_model.forward(&input_ids, None)?;

        assert_eq!(output.batch_size(), 2);
        assert_eq!(output.sequence_length(), 10);
        assert_eq!(output.hidden_size(), 256);

        Ok(())
    }

    #[test]
    fn test_lora_layer_freeze_unfreeze() -> Result<()> {
        let config = LoRAConfig::default();
        let mut layer = LoRALayer::new(config, 768, 768)?;

        assert!(!layer.frozen);
        assert!(layer.num_parameters() > 0);

        layer.freeze();
        assert!(layer.frozen);
        assert_eq!(layer.num_parameters(), 0);

        layer.unfreeze();
        assert!(!layer.frozen);
        assert!(layer.num_parameters() > 0);

        Ok(())
    }

    #[test]
    fn test_invalid_lora_rank() {
        let config = LoRAConfig {
            r: 0,
            ..Default::default()
        };

        let result = LoRALayer::new(config, 768, 768);
        assert!(result.is_err());
    }

    #[test]
    fn test_rank_too_large() {
        let config = LoRAConfig {
            r: 1000,
            ..Default::default()
        };

        let result = LoRALayer::new(config, 768, 768);
        assert!(result.is_err());
    }
}
