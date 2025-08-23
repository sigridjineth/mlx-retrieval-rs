//! Error types for the MLX retrieval system
//!
//! This module provides a comprehensive error handling system using `eyre` for rich
//! error context and `thiserror` for structured error types.

/// Custom Result type for the MLX retrieval system
pub type Result<T> = std::result::Result<T, MlxRetrievalError>;

/// Main error type for the MLX retrieval system
#[derive(Debug, thiserror::Error)]
pub enum MlxRetrievalError {
    /// MLX framework errors
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// I/O errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Tokenization errors
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Data processing errors
    #[error("Data processing error: {0}")]
    DataProcessing(String),

    /// Model errors
    #[error("Model error: {0}")]
    Model(String),

    /// Training errors
    #[error("Training error: {0}")]
    Training(String),

    /// Loss function errors
    #[error("Loss function error: {0}")]
    Loss(String),

    /// Embedding extraction errors
    #[error("Embedding extraction error: {0}")]
    Embedding(String),

    /// Invalid input errors
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Feature not enabled
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: expected {expected:?}, got {got:?}")]
    DimensionMismatch { expected: Vec<i32>, got: Vec<i32> },

    /// Numerical computation error (NaN, Inf, etc.)
    #[error("Numerical error: {0}")]
    Numerical(String),

    /// File format error
    #[error("Invalid file format: {0}")]
    FileFormat(String),

    /// Safetensors error
    #[error("Safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// Async channel error
    #[error("Channel communication failed: {0}")]
    Channel(String),

    /// Generic error with context
    #[error("{context}: {source}")]
    Generic {
        context: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl MlxRetrievalError {
    /// Create a new tokenization error
    pub fn tokenization<S: Into<String>>(msg: S) -> Self {
        Self::Tokenization(msg.into())
    }

    /// Create a new configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }

    /// Create a new data processing error
    pub fn data_processing<S: Into<String>>(msg: S) -> Self {
        Self::DataProcessing(msg.into())
    }

    /// Create a new model error
    pub fn model<S: Into<String>>(msg: S) -> Self {
        Self::Model(msg.into())
    }

    /// Create a new training error
    pub fn training<S: Into<String>>(msg: S) -> Self {
        Self::Training(msg.into())
    }

    /// Create a new loss function error
    pub fn loss<S: Into<String>>(msg: S) -> Self {
        Self::Loss(msg.into())
    }

    /// Create a new embedding error
    pub fn embedding<S: Into<String>>(msg: S) -> Self {
        Self::Embedding(msg.into())
    }

    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a new feature not enabled error
    pub fn feature_not_enabled<S: Into<String>>(feature: S) -> Self {
        Self::FeatureNotEnabled(feature.into())
    }

    /// Create a generic error with context
    pub fn generic<E>(context: &str, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Generic {
            context: context.to_string(),
            source: Box::new(source),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: Vec<i32>, got: Vec<i32>) -> Self {
        Self::DimensionMismatch { expected, got }
    }

    /// Create a numerical error
    pub fn numerical<S: Into<String>>(msg: S) -> Self {
        Self::Numerical(msg.into())
    }

    /// Create a file format error
    pub fn file_format<S: Into<String>>(msg: S) -> Self {
        Self::FileFormat(msg.into())
    }

    /// Create a channel error
    pub fn channel<S: Into<String>>(msg: S) -> Self {
        Self::Channel(msg.into())
    }
}

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    /// Add static context to an error
    fn context(self, msg: &str) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| MlxRetrievalError::generic(&f(), e))
    }

    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| MlxRetrievalError::generic(msg, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = MlxRetrievalError::tokenization("test error");
        assert_eq!(err.to_string(), "Tokenization error: test error");
    }

    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));

        let err = result.context("Failed to read config file").unwrap_err();
        assert!(err.to_string().contains("Failed to read config file"));
    }
}
