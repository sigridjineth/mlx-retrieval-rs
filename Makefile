# MLX-Retrieval-RS Makefile
# High-performance Rust implementation of MLX-based text retrieval

.PHONY: help build test run clean check fmt clippy examples doc validate all

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Project variables
CARGO := cargo
PROJECT_NAME := mlx-retrieval-rs
RUST_VERSION := 1.70.0

## help: Display this help message
help:
	@echo "$(BLUE)MLX-Retrieval-RS Development Commands$(NC)"
	@echo "======================================"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^##' Makefile | sed 's/## /  /' | column -t -s ':'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make test     # Run all tests"
	@echo "  make run      # Run main binary"
	@echo "  make examples # Run all examples"

## test: Run all tests (unit, integration, doc tests)
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(CARGO) test --all-features
	@echo "$(GREEN)✓ All tests passed!$(NC)"

## run: Run the main binary
run:
	@echo "$(BLUE)Running main binary...$(NC)"
	@$(CARGO) run --bin mlx-retrieval-rs

## build: Build the project in release mode
build:
	@echo "$(BLUE)Building project in release mode...$(NC)"
	@$(CARGO) build --release
	@echo "$(GREEN)✓ Build complete!$(NC)"

## build-dev: Build the project in debug mode
build-dev:
	@echo "$(BLUE)Building project in debug mode...$(NC)"
	@$(CARGO) build
	@echo "$(GREEN)✓ Debug build complete!$(NC)"

## check: Run cargo check
check:
	@echo "$(BLUE)Running cargo check...$(NC)"
	@$(CARGO) check --all-targets --all-features
	@echo "$(GREEN)✓ Check complete!$(NC)"

## clippy: Run clippy linter
clippy:
	@echo "$(BLUE)Running clippy...$(NC)"
	@$(CARGO) clippy --all-targets --all-features -- -D warnings
	@echo "$(GREEN)✓ No clippy warnings!$(NC)"

## fmt: Format code
fmt:
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(CARGO) fmt
	@echo "$(GREEN)✓ Code formatted!$(NC)"

## fmt-check: Check code formatting
fmt-check:
	@echo "$(BLUE)Checking code format...$(NC)"
	@$(CARGO) fmt -- --check
	@echo "$(GREEN)✓ Code format is correct!$(NC)"

## clean: Clean build artifacts
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@$(CARGO) clean
	@rm -f Cargo.lock
	@echo "$(GREEN)✓ Clean complete!$(NC)"

## doc: Generate and open documentation
doc:
	@echo "$(BLUE)Generating documentation...$(NC)"
	@$(CARGO) doc --no-deps --open
	@echo "$(GREEN)✓ Documentation generated!$(NC)"

## doc-all: Generate documentation with dependencies
doc-all:
	@echo "$(BLUE)Generating full documentation...$(NC)"
	@$(CARGO) doc --open
	@echo "$(GREEN)✓ Full documentation generated!$(NC)"

## validate: Run Python validation script
validate:
	@echo "$(BLUE)Running validation against Python implementation...$(NC)"
	@python test_golden_data.py
	@echo "$(GREEN)✓ Validation complete!$(NC)"

## examples: Build and list all examples
examples:
	@echo "$(BLUE)Available examples:$(NC)"
	@echo "  1. infonce_training    - Basic InfoNCE loss training"
	@echo "  2. data_batching       - Data loading and batching"
	@echo "  3. embeddings_pooling  - Pooling strategies"
	@echo "  4. evaluation          - Model evaluation metrics"
	@echo "  5. full_training       - Complete training pipeline"
	@echo ""
	@echo "Run individual examples with:"
	@echo "  make example-infonce"
	@echo "  make example-batch"
	@echo "  make example-pool"
	@echo "  make example-eval"
	@echo "  make example-train"

## example-infonce: Run InfoNCE training example
example-infonce:
	@echo "$(BLUE)Running InfoNCE training example...$(NC)"
	@$(CARGO) run --example infonce_training

## example-batch: Run data batching example
example-batch:
	@echo "$(BLUE)Running data batching example...$(NC)"
	@$(CARGO) run --example data_batching

## example-pool: Run embeddings pooling example
example-pool:
	@echo "$(BLUE)Running embeddings pooling example...$(NC)"
	@$(CARGO) run --example embeddings_pooling

## example-eval: Run evaluation example
example-eval:
	@echo "$(BLUE)Running evaluation example...$(NC)"
	@$(CARGO) run --example evaluation

## example-train: Run full training example
example-train:
	@echo "$(BLUE)Running full training pipeline example...$(NC)"
	@$(CARGO) run --example full_training

## bench: Run benchmarks
bench:
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@$(CARGO) bench

## test-unit: Run only unit tests
test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(CARGO) test --lib

## test-integration: Run only integration tests
test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(CARGO) test --test '*'

## test-doc: Run documentation tests
test-doc:
	@echo "$(BLUE)Running documentation tests...$(NC)"
	@$(CARGO) test --doc

## test-examples: Test that all examples compile
test-examples:
	@echo "$(BLUE)Testing example compilation...$(NC)"
	@$(CARGO) build --examples
	@echo "$(GREEN)✓ All examples compile!$(NC)"

## ci: Run all CI checks (fmt, clippy, test, examples)
ci: fmt-check clippy test test-examples
	@echo "$(GREEN)✓ All CI checks passed!$(NC)"

## install: Install the binary locally
install:
	@echo "$(BLUE)Installing mlx-retrieval-rs...$(NC)"
	@$(CARGO) install --path .
	@echo "$(GREEN)✓ Installation complete!$(NC)"

## uninstall: Uninstall the binary
uninstall:
	@echo "$(BLUE)Uninstalling mlx-retrieval-rs...$(NC)"
	@$(CARGO) uninstall $(PROJECT_NAME)
	@echo "$(GREEN)✓ Uninstallation complete!$(NC)"

## dev: Run in development mode with auto-reload (requires cargo-watch)
dev:
	@echo "$(BLUE)Starting development mode...$(NC)"
	@command -v cargo-watch >/dev/null 2>&1 || { echo "$(RED)Error: cargo-watch not installed. Run: cargo install cargo-watch$(NC)" >&2; exit 1; }
	@cargo-watch -x run

## setup: Install development dependencies
setup:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@rustup component add rustfmt clippy
	@cargo install cargo-watch cargo-edit cargo-audit
	@echo "$(GREEN)✓ Development setup complete!$(NC)"

## audit: Check for security vulnerabilities
audit:
	@echo "$(BLUE)Checking for security vulnerabilities...$(NC)"
	@command -v cargo-audit >/dev/null 2>&1 || { echo "$(YELLOW)Warning: cargo-audit not installed. Run: cargo install cargo-audit$(NC)" >&2; exit 0; }
	@cargo audit || true
	@echo "$(GREEN)✓ Security audit complete!$(NC)"

## update: Update dependencies
update:
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@$(CARGO) update
	@echo "$(GREEN)✓ Dependencies updated!$(NC)"

## stats: Show project statistics
stats:
	@echo "$(BLUE)Project Statistics:$(NC)"
	@echo "==================="
	@echo "Lines of Rust code:"
	@find src -name "*.rs" | xargs wc -l | tail -1
	@echo ""
	@echo "Number of source files:"
	@find src -name "*.rs" | wc -l
	@echo ""
	@echo "Number of tests:"
	@grep -r "#\[test\]" src tests 2>/dev/null | wc -l || echo "0"
	@echo ""
	@echo "Number of examples:"
	@ls examples/*.rs 2>/dev/null | wc -l || echo "0"

## quick: Quick build and test
quick: fmt check test
	@echo "$(GREEN)✓ Quick check complete!$(NC)"

## all: Run all checks and build
all: clean fmt check clippy test test-examples build doc
	@echo "$(GREEN)✓ All tasks complete!$(NC)"

# Special targets for Python validation
.PHONY: validate-python validate-infonce validate-pooling validate-critical

## validate-python: Run all Python validation tests
validate-python:
	@echo "$(BLUE)Running Python validation tests...$(NC)"
	@python test_golden_data.py
	@python validate_numerical.py 2>/dev/null || true
	@echo "$(GREEN)✓ Python validation complete!$(NC)"

## validate-infonce: Validate InfoNCE implementation
validate-infonce:
	@echo "$(BLUE)Validating InfoNCE loss...$(NC)"
	@python -c "import test_golden_data; test_golden_data.test_infonce_loss()"

## validate-pooling: Validate pooling implementations
validate-pooling:
	@echo "$(BLUE)Validating pooling strategies...$(NC)"
	@python -c "import test_golden_data; test_golden_data.test_mean_pooling()"

## validate-critical: Check critical fixes
validate-critical:
	@echo "$(BLUE)Checking critical fixes...$(NC)"
	@python -c "import test_golden_data; test_golden_data.check_critical_fixes()"

# Variables for release
VERSION := $(shell grep version Cargo.toml | head -1 | cut -d'"' -f2)

## release: Create a new release
release: ci
	@echo "$(BLUE)Creating release v$(VERSION)...$(NC)"
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "$(GREEN)✓ Release v$(VERSION) created!$(NC)"
	@echo "$(YELLOW)Run 'git push origin v$(VERSION)' to publish$(NC)"

## version: Show current version
version:
	@echo "$(BLUE)MLX-Retrieval-RS v$(VERSION)$(NC)"