//! MLX Retrieval RS - High-performance text retrieval and embedding training system
//!
//! This library provides a comprehensive framework for training and utilizing text embeddings
//! using Apple's MLX framework for optimal performance on Apple Silicon.
//!
//! ## Features
//!
//! - **High-Performance Embedding Training**: Leverage MLX for fast training on Apple Silicon
//! - **Multiple Data Sources**: Support for JSONL files, streaming data, and Elasticsearch
//! - **Flexible Loss Functions**: InfoNCE, hard negative mining, and NT-Xent implementations
//! - **LoRA Fine-tuning**: Efficient fine-tuning with Low-Rank Adaptation
//! - **Pooling Strategies**: Various pooling methods for token-to-sentence embeddings
//! - **CLI Interface**: Easy-to-use command line interface for training and inference
//!
//! ## Architecture
//!
//! The library follows a clean architecture with clear separation between:
//! - Data handling and preprocessing
//! - Model definitions and operations
//! - Training and optimization logic
//! - Loss function implementations
//! - Command line interface

pub mod error;

pub mod data;
pub mod embed;
pub mod loss;
pub mod model;
pub mod training;

pub mod cli;

// Re-export key types and functions for convenience
pub use error::{MlxRetrievalError, Result};

// Re-export MLX types for easier access
pub use mlx_rs::{Array, Dtype};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_exists() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_name_exists() {
        assert_eq!(NAME, "mlx-retrieval-rs");
    }
}
