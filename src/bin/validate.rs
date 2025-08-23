#!/usr/bin/env cargo
//! Comprehensive Validation Binary for MLX Retrieval Golden Tests
//!
//! This binary loads all golden test JSON files, deserializes the test data using serde_json,
//! runs our implementations, and compares outputs with expected values using appropriate
//! floating point tolerance.
//!
//! Usage: cargo run --bin validate
//!
//! This validates:
//! - Mean pooling with mask (mean_pooling_with_mask.json)
//! - InfoNCE loss (infonce_loss.json)  
//! - Hard negative InfoNCE loss (hard_negative_infonce_loss.json)
//! - NT-Xent loss with percentage positives (ntxent_loss_perc_pos.json)
//! - NT-Xent loss with margin positives (ntxent_loss_margin_pos.json)

use eyre::{Context, Result};
use mlx_rs::Array;
use serde_json::Value;
use std::fs;
use std::path::Path;

// Import our implementations to test
use mlx_retrieval_rs::embed::pooling::mean_pooling;
use mlx_retrieval_rs::loss::infonce::simple_infonce_loss;

/// Floating point tolerance for comparisons
const DEFAULT_TOLERANCE: f32 = 1e-5;
const RELAXED_TOLERANCE: f32 = 1e-3; // For loss functions that may have more numerical differences

/// Test result for a single golden test
#[derive(Debug)]
struct TestResult {
    test_name: String,
    passed: bool,
    expected: String,
    actual: String,
    tolerance: f32,
    difference: f32,
    error: Option<String>,
}

impl TestResult {
    fn new_pass(
        test_name: String,
        expected: String,
        actual: String,
        tolerance: f32,
        difference: f32,
    ) -> Self {
        Self {
            test_name,
            passed: true,
            expected,
            actual,
            tolerance,
            difference,
            error: None,
        }
    }

    fn new_fail(
        test_name: String,
        expected: String,
        actual: String,
        tolerance: f32,
        difference: f32,
        error: Option<String>,
    ) -> Self {
        Self {
            test_name,
            passed: false,
            expected,
            actual,
            tolerance,
            difference,
            error,
        }
    }
}

/// Collection of all test results
#[derive(Debug)]
struct ValidationResults {
    results: Vec<TestResult>,
}

impl ValidationResults {
    fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    fn total_count(&self) -> usize {
        self.results.len()
    }

    fn print_summary(&self) {
        println!("\n🔍 VALIDATION SUMMARY");
        println!("=====================================");
        println!("Total Tests: {}", self.total_count());
        println!("✅ Passed: {}", self.passed_count());
        println!("❌ Failed: {}", self.failed_count());
        println!(
            "Success Rate: {:.1}%",
            (self.passed_count() as f32 / self.total_count() as f32) * 100.0
        );
        println!();

        // Print detailed results
        for result in &self.results {
            let status = if result.passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };
            println!("{} - {}", status, result.test_name);

            if result.passed {
                println!("   Expected: {}", result.expected);
                println!("   Actual:   {}", result.actual);
                println!(
                    "   Diff:     {:.6} (tolerance: {:.6})",
                    result.difference, result.tolerance
                );
            } else {
                println!("   Expected: {}", result.expected);
                println!("   Actual:   {}", result.actual);
                println!(
                    "   Diff:     {:.6} (tolerance: {:.6})",
                    result.difference, result.tolerance
                );
                if let Some(error) = &result.error {
                    println!("   Error:    {error}");
                }
            }
            println!();
        }
    }

    fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
}

/// Convert JSON array to Vec<f32> by flattening nested arrays
fn json_array_to_vec_f32(json_array: &Value) -> Result<Vec<f32>> {
    let mut result = Vec::new();

    fn flatten_recursive(value: &Value, result: &mut Vec<f32>) -> Result<()> {
        match value {
            Value::Array(arr) => {
                for item in arr {
                    flatten_recursive(item, result)?;
                }
            }
            Value::Number(num) => {
                let f_val = num
                    .as_f64()
                    .ok_or_else(|| eyre::eyre!("Failed to convert number to f64"))?
                    as f32;
                result.push(f_val);
            }
            _ => return Err(eyre::eyre!("Expected array or number, got {:?}", value)),
        }
        Ok(())
    }

    flatten_recursive(json_array, &mut result)?;
    Ok(result)
}

/// Extract shape from JSON test data  
fn extract_shape(test_data: &Value, path: &[&str]) -> Result<Vec<i32>> {
    let mut current = test_data;
    for key in path {
        current = current
            .get(key)
            .ok_or_else(|| eyre::eyre!("Missing key: {}", key))?;
    }

    let shape_array = current
        .as_array()
        .ok_or_else(|| eyre::eyre!("Shape is not an array"))?;

    let shape: Result<Vec<i32>> = shape_array
        .iter()
        .map(|v| {
            v.as_i64()
                .ok_or_else(|| eyre::eyre!("Shape element is not an integer"))
                .map(|i| i as i32)
        })
        .collect();

    shape
}

/// Validate mean pooling implementation
fn validate_mean_pooling() -> Result<TestResult> {
    println!("🧪 Testing Mean Pooling with Mask...");

    let test_file = "tests/mean_pooling_with_mask.json";
    if !Path::new(test_file).exists() {
        return Ok(TestResult::new_fail(
            "mean_pooling_with_mask".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
            DEFAULT_TOLERANCE,
            0.0,
            Some(format!("Test file not found: {test_file}")),
        ));
    }

    let golden_data = fs::read_to_string(test_file).context("Failed to read golden test file")?;
    let test_case: Value =
        serde_json::from_str(&golden_data).context("Failed to parse golden test JSON")?;

    // Extract input data
    let hidden_states_data = test_case["inputs"]["hidden_states"]["data"]
        .as_array()
        .ok_or_else(|| eyre::eyre!("Missing hidden_states data"))?;
    let attention_mask_data = test_case["inputs"]["attention_mask"]["data"]
        .as_array()
        .ok_or_else(|| eyre::eyre!("Missing attention_mask data"))?;
    let expected_data = test_case["expected_output"]["pooled_embedding"]["data"]
        .as_array()
        .ok_or_else(|| eyre::eyre!("Missing expected output data"))?;

    // Convert to flat vectors
    let hidden_states_vec = json_array_to_vec_f32(&Value::Array(hidden_states_data.clone()))?;
    let attention_mask_vec = json_array_to_vec_f32(&Value::Array(attention_mask_data.clone()))?;
    let expected_vec = json_array_to_vec_f32(&Value::Array(expected_data.clone()))?;

    // Extract shapes
    let hidden_states_shape = extract_shape(&test_case, &["inputs", "hidden_states", "shape"])?;
    let attention_mask_shape = extract_shape(&test_case, &["inputs", "attention_mask", "shape"])?;
    let expected_shape = extract_shape(
        &test_case,
        &["expected_output", "pooled_embedding", "shape"],
    )?;

    // Create MLX arrays
    let hidden_states = Array::from_slice(&hidden_states_vec, &hidden_states_shape);
    let attention_mask = Array::from_slice(&attention_mask_vec, &attention_mask_shape);
    let expected_output = Array::from_slice(&expected_vec, &expected_shape);

    // Call our implementation
    match mean_pooling(&hidden_states, &attention_mask) {
        Ok(result) => {
            // Check shapes match
            if result.shape() != expected_output.shape() {
                return Ok(TestResult::new_fail(
                    "mean_pooling_with_mask".to_string(),
                    format!("Shape: {:?}", expected_output.shape()),
                    format!("Shape: {:?}", result.shape()),
                    DEFAULT_TOLERANCE,
                    0.0,
                    Some("Shape mismatch".to_string()),
                ));
            }

            // Check values are close
            let diff = result.subtract(&expected_output)?;
            let abs_diff = diff.abs()?;
            let max_diff = abs_diff.max(&[], false)?;
            let max_diff_val = max_diff.item::<f32>();

            if max_diff_val < DEFAULT_TOLERANCE {
                Ok(TestResult::new_pass(
                    "mean_pooling_with_mask".to_string(),
                    format!(
                        "Max element: {:.6}",
                        expected_output.max(&[], false)?.item::<f32>()
                    ),
                    format!("Max element: {:.6}", result.max(&[], false)?.item::<f32>()),
                    DEFAULT_TOLERANCE,
                    max_diff_val,
                ))
            } else {
                Ok(TestResult::new_fail(
                    "mean_pooling_with_mask".to_string(),
                    format!(
                        "Max element: {:.6}",
                        expected_output.max(&[], false)?.item::<f32>()
                    ),
                    format!("Max element: {:.6}", result.max(&[], false)?.item::<f32>()),
                    DEFAULT_TOLERANCE,
                    max_diff_val,
                    Some(format!("Max difference {max_diff_val} exceeds tolerance")),
                ))
            }
        }
        Err(e) => Ok(TestResult::new_fail(
            "mean_pooling_with_mask".to_string(),
            "Success".to_string(),
            "Error".to_string(),
            DEFAULT_TOLERANCE,
            0.0,
            Some(format!("Implementation error: {e}")),
        )),
    }
}

/// Validate InfoNCE loss implementation
fn validate_infonce_loss() -> Result<TestResult> {
    println!("🧪 Testing InfoNCE Loss...");

    let test_file = "tests/infonce_loss.json";
    if !Path::new(test_file).exists() {
        return Ok(TestResult::new_fail(
            "infonce_loss".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
            RELAXED_TOLERANCE,
            0.0,
            Some(format!("Test file not found: {test_file}")),
        ));
    }

    let golden_data = fs::read_to_string(test_file).context("Failed to read golden test file")?;
    let test_case: Value =
        serde_json::from_str(&golden_data).context("Failed to parse golden test JSON")?;

    // Extract input data
    let query_data = test_case["inputs"]["query_embeddings"]["data"]
        .as_array()
        .ok_or_else(|| eyre::eyre!("Missing query_embeddings data"))?;
    let doc_data = test_case["inputs"]["doc_embeddings"]["data"]
        .as_array()
        .ok_or_else(|| eyre::eyre!("Missing doc_embeddings data"))?;

    let expected_loss = test_case["expected_output"]["loss"]["value"]
        .as_f64()
        .ok_or_else(|| eyre::eyre!("Missing expected loss value"))? as f32;
    let temperature = test_case["config"]["temperature"]
        .as_f64()
        .ok_or_else(|| eyre::eyre!("Missing temperature config"))? as f32;

    // Convert to flat vectors
    let query_vec = json_array_to_vec_f32(&Value::Array(query_data.clone()))?;
    let doc_vec = json_array_to_vec_f32(&Value::Array(doc_data.clone()))?;

    // Extract shapes
    let query_shape = extract_shape(&test_case, &["inputs", "query_embeddings", "shape"])?;
    let doc_shape = extract_shape(&test_case, &["inputs", "doc_embeddings", "shape"])?;

    // Create MLX arrays
    let query_embeddings = Array::from_slice(&query_vec, &query_shape);
    let doc_embeddings = Array::from_slice(&doc_vec, &doc_shape);

    // Call our implementation
    match simple_infonce_loss(&query_embeddings, &doc_embeddings, temperature) {
        Ok(actual_loss) => {
            let diff = (actual_loss - expected_loss).abs();

            if diff < RELAXED_TOLERANCE {
                Ok(TestResult::new_pass(
                    "infonce_loss".to_string(),
                    format!("{expected_loss:.6}"),
                    format!("{actual_loss:.6}"),
                    RELAXED_TOLERANCE,
                    diff,
                ))
            } else {
                Ok(TestResult::new_fail(
                    "infonce_loss".to_string(),
                    format!("{expected_loss:.6}"),
                    format!("{actual_loss:.6}"),
                    RELAXED_TOLERANCE,
                    diff,
                    Some(format!("Loss difference {diff} exceeds tolerance")),
                ))
            }
        }
        Err(e) => Ok(TestResult::new_fail(
            "infonce_loss".to_string(),
            format!("{expected_loss:.6}"),
            "Error".to_string(),
            RELAXED_TOLERANCE,
            0.0,
            Some(format!("Implementation error: {e}")),
        )),
    }
}

/// Placeholder for hard negative InfoNCE validation (not yet implemented)
fn validate_hard_negative_infonce() -> Result<TestResult> {
    println!("🧪 Testing Hard Negative InfoNCE Loss...");

    let test_file = "tests/hard_negative_infonce_loss.json";
    if !Path::new(test_file).exists() {
        return Ok(TestResult::new_fail(
            "hard_negative_infonce_loss".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
            RELAXED_TOLERANCE,
            0.0,
            Some(format!("Test file not found: {test_file}")),
        ));
    }

    // TODO: Implement hard negative InfoNCE validation once the function is ready
    Ok(TestResult::new_fail(
        "hard_negative_infonce_loss".to_string(),
        "Implemented".to_string(),
        "Not implemented yet".to_string(),
        RELAXED_TOLERANCE,
        0.0,
        Some("Hard negative InfoNCE implementation not yet complete".to_string()),
    ))
}

/// Placeholder for NT-Xent percentage positives validation (not yet implemented)
fn validate_ntxent_perc_pos() -> Result<TestResult> {
    println!("🧪 Testing NT-Xent Loss (Percentage Positives)...");

    let test_file = "tests/ntxent_loss_perc_pos.json";
    if !Path::new(test_file).exists() {
        return Ok(TestResult::new_fail(
            "ntxent_loss_perc_pos".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
            RELAXED_TOLERANCE,
            0.0,
            Some(format!("Test file not found: {test_file}")),
        ));
    }

    // TODO: Implement NT-Xent percentage positives validation once the function is ready
    Ok(TestResult::new_fail(
        "ntxent_loss_perc_pos".to_string(),
        "Implemented".to_string(),
        "Not implemented yet".to_string(),
        RELAXED_TOLERANCE,
        0.0,
        Some("NT-Xent percentage positives implementation not yet complete".to_string()),
    ))
}

/// Placeholder for NT-Xent margin positives validation (not yet implemented)
fn validate_ntxent_margin_pos() -> Result<TestResult> {
    println!("🧪 Testing NT-Xent Loss (Margin Positives)...");

    let test_file = "tests/ntxent_loss_margin_pos.json";
    if !Path::new(test_file).exists() {
        return Ok(TestResult::new_fail(
            "ntxent_loss_margin_pos".to_string(),
            "N/A".to_string(),
            "N/A".to_string(),
            RELAXED_TOLERANCE,
            0.0,
            Some(format!("Test file not found: {test_file}")),
        ));
    }

    // TODO: Implement NT-Xent margin positives validation once the function is ready
    Ok(TestResult::new_fail(
        "ntxent_loss_margin_pos".to_string(),
        "Implemented".to_string(),
        "Not implemented yet".to_string(),
        RELAXED_TOLERANCE,
        0.0,
        Some("NT-Xent margin positives implementation not yet complete".to_string()),
    ))
}

fn main() -> Result<()> {
    println!("🚀 MLX Retrieval Golden Test Validation");
    println!("========================================");

    let mut results = ValidationResults::new();

    // Run all validation tests
    results.add_result(validate_mean_pooling()?);
    results.add_result(validate_infonce_loss()?);
    results.add_result(validate_hard_negative_infonce()?);
    results.add_result(validate_ntxent_perc_pos()?);
    results.add_result(validate_ntxent_margin_pos()?);

    // Print summary
    results.print_summary();

    // Exit with error code if any tests failed
    if !results.all_passed() {
        println!("💥 Some tests failed!");
        std::process::exit(1);
    } else {
        println!("🎉 All tests passed!");
        std::process::exit(0);
    }
}
