#!/bin/bash

# Test script for MLX-Retrieval-RS examples
# This script builds and tests all examples

set -e  # Exit on error

echo "Testing MLX-Retrieval-RS Examples"
echo "=================================="
echo

# Build all examples
echo "Building examples..."
echo "--------------------"

examples=("infonce_training" "data_batching" "embeddings_pooling" "evaluation" "full_training")

for example in "${examples[@]}"; do
    echo "Building example: $example"
    cargo build --example "$example" 2>&1 | tail -3
    if [ $? -eq 0 ]; then
        echo "✓ $example built successfully"
    else
        echo "✗ $example failed to build"
    fi
    echo
done

echo "All examples built!"
echo

# Run a simple test
echo "Running InfoNCE training example..."
echo "-----------------------------------"
cargo run --example infonce_training 2>&1 | head -20

echo
echo "Test complete!"