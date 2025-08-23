//! Main entry point for the MLX retrieval CLI

use mlx_retrieval_rs::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing/logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("MLX-Retrieval-RS v{}", env!("CARGO_PKG_VERSION"));
    println!("High-performance retrieval model training in Rust");
    println!("=================================================\n");
    
    println!("Run examples with:");
    println!("  cargo run --example infonce_training");
    println!("  cargo run --example data_batching");
    println!("  cargo run --example embeddings_pooling");
    println!("  cargo run --example evaluation");
    println!("  cargo run --example full_training");
    
    Ok(())
}
