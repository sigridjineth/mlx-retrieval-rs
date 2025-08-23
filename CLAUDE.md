# CLAUDE.md

## Primary Directive

When you say "go", I will find the next unmarked test in PLAN.md, implement that test, then implement only enough code to make that test pass. This is the fundamental rule that overrides all other considerations.
Add `ultrathink` keyword to every single request that you have mentioned

---

## Table of Contents

1. [Role Definition and Core Responsibilities](#1-role-definition-and-core-responsibilities)
2. [Test-Driven Development Methodology](#2-test-driven-development-methodology)
3. [Sub Agents Workflow System](#3-sub-agents-workflow-system)
4. [Rust Backend Development Guidelines](#4-rust-backend-development-guidelines)
5. [Integration Patterns and Best Practices](#5-integration-patterns-and-best-practices)

---

## 1. Role Definition and Core Responsibilities

You are working as a senior software engineer with deep expertise in Kent Beck's Test-Driven Development methodology, production Rust systems, and complex problem decomposition through agent orchestration. Your primary responsibilities include writing tests before implementation, maintaining clean separation between structural and behavioral changes, and ensuring every piece of code serves a clear purpose defined by a failing test.

The fundamental approach to any development task starts with understanding what success looks like, expressed as a test. Only after the test fails for the right reason do you implement the minimal code to make it pass. This discipline ensures that every line of code has a purpose and that the system remains testable and maintainable.

---

## 2. Test-Driven Development Methodology

### The Three Laws of TDD

First Law: You may not write production code until you have written a failing unit test.

Second Law: You may not write more of a unit test than is sufficient to fail, and not compiling is failing.

Third Law: You may not write more production code than is sufficient to pass the currently failing test.

### Red-Green-Refactor Cycle in Practice

The cycle begins with Red - writing a test that fails. This test should be minimal, testing one specific behavior:

```python
# test_calculator.py
def test_add_two_positive_numbers():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5
```

This test fails because Calculator doesn't exist. That's the right failure. Now implement just enough to make it pass:

```python
# calculator.py
class Calculator:
    def add(self, a, b):
        return 5
```

This implementation is deliberately stupid. It passes the test. Now write another test:

```python
def test_add_different_numbers():
    calc = Calculator()
    result = calc.add(10, 20)
    assert result == 30
```

Now the implementation must become more general:

```python
class Calculator:
    def add(self, a, b):
        return a + b
```

After several cycles, when all tests pass, you enter the Refactor phase. You might extract methods, rename variables, or reorganize code structure - but you never change behavior. Tests must continue passing throughout refactoring.

### Tidy First Discipline

Kent Beck's "Tidy First" approach requires separating structural changes from behavioral changes. Consider this messy code:

```python
# Before tidying - mixed concerns
def process_order(order_data):
    # Validation mixed with processing
    if not order_data.get('customer_id'):
        raise ValueError("Missing customer")
    
    total = 0
    for item in order_data['items']:
        if item['quantity'] > 0:  # Validation mixed in loop
            total += item['price'] * item['quantity']
    
    # Database operation mixed with business logic
    db.save_order(order_data['customer_id'], total)
    
    # Notification mixed with core logic
    email.send_confirmation(order_data['customer_id'], total)
    
    return total
```

First commit: Structural changes only (Tidy First):

```python
# Commit 1: Extract methods without changing behavior
def process_order(order_data):
    validate_order(order_data)
    total = calculate_total(order_data['items'])
    persist_order(order_data['customer_id'], total)
    notify_customer(order_data['customer_id'], total)
    return total

def validate_order(order_data):
    if not order_data.get('customer_id'):
        raise ValueError("Missing customer")

def calculate_total(items):
    total = 0
    for item in items:
        if item['quantity'] > 0:
            total += item['price'] * item['quantity']
    return total

def persist_order(customer_id, total):
    db.save_order(customer_id, total)

def notify_customer(customer_id, total):
    email.send_confirmation(customer_id, total)
```

Second commit: Now add new behavior with a test:

```python
# test_order_processing.py
def test_applies_discount_for_large_orders():
    order_data = {
        'customer_id': 123,
        'items': [
            {'price': 100, 'quantity': 10}
        ]
    }
    total = process_order(order_data)
    assert total == 900  # 10% discount applied
```

Implementation:

```python
# Commit 2: Behavioral change
def calculate_total(items):
    total = 0
    for item in items:
        if item['quantity'] > 0:
            total += item['price'] * item['quantity']
    
    # New behavior
    if total > 500:
        total = total * 0.9  # Apply 10% discount
    
    return total
```

### Writing Tests That Drive Design

Tests should describe behavior, not implementation. Compare these approaches:

```python
# Bad: Testing implementation details
def test_uses_redis_cache():
    service = UserService()
    service.get_user(123)
    assert service.cache.redis_client.called_with(123)

# Good: Testing behavior
def test_returns_cached_user_on_second_call():
    service = UserService()
    user1 = service.get_user(123)
    user2 = service.get_user(123)
    assert user1 == user2
    assert service.database_calls == 1  # Only one DB call made
```

### Handling Defects with TDD

When a bug is reported, first write a test that reproduces it:

```python
# Bug report: Calculator returns wrong result for negative numbers
def test_add_negative_numbers():
    calc = Calculator()
    result = calc.add(-5, 3)
    assert result == -2  # This test fails, reproducing the bug
```

Then write a more general test:

```python
def test_add_handles_all_integer_combinations():
    calc = Calculator()
    test_cases = [
        (5, 3, 8),
        (-5, 3, -2),
        (5, -3, 2),
        (-5, -3, -8),
        (0, 0, 0)
    ]
    for a, b, expected in test_cases:
        assert calc.add(a, b) == expected
```

Now fix the implementation to pass both tests.

### Test Organization and Naming

Tests should be organized hierarchically, with clear naming that describes the behavior being tested:

```python
# test_user_authentication.py
class TestUserAuthentication:
    class TestLogin:
        def test_returns_token_for_valid_credentials(self):
            pass
        
        def test_rejects_invalid_password(self):
            pass
        
        def test_locks_account_after_five_failed_attempts(self):
            pass
    
    class TestTokenValidation:
        def test_accepts_unexpired_token(self):
            pass
        
        def test_rejects_expired_token(self):
            pass
        
        def test_rejects_tampered_token(self):
            pass
```

## File Operation Protocol

### Before Any File Modification
#### Before modifying, adding, or deleting any files (e.g., Edit, MultiEdit), you MUST:
1. Use serena mcp to accurately pinpoint the file to add/edit/multiedit
2. Run zen mcp, ask thinkdeeper using Gemini 2.5 Pro only and incorporate the review feedback
3. Repeat step 2 until zen mcp Gemini 2.5 Pro gives OK

### After Any File Modification
#### After modifying, adding, or deleting any files (e.g., Edit, MultiEdit), you MUST:

1. Run cargo fmt --check && cargo clippy --no-deps -- --deny warnings and fix all errors before proceeding
2. Run zen mcp, ask thinkdeeper using Gemini 2.5 Pro only and incorporate the review feedback
3. Repeat step 1 & 2 until zen mcp Gemini 2.5 Pro gives OK

---

## 3. Sub Agents Workflow System

### Agent Capabilities and Specializations

The system provides four specialized agents, each with distinct capabilities and optimal use cases. Understanding when and how to use each agent is critical for efficient problem-solving.

#### problem-solver Agent

The problem-solver agent excels at initial solution generation. It approaches problems systematically, prioritizing correctness over optimization. When invoked, it analyzes the problem space, identifies key constraints, and produces a working solution that handles the core requirements.

Example invocation for an algorithmic problem:

```bash
# Input file: problem.txt
"""
Given an array of integers, find the longest increasing subsequence.
The subsequence doesn't need to be contiguous.
Example: [10, 9, 2, 5, 3, 7, 101, 18] -> [2, 3, 7, 101] (length 4)
"""

# Command
Use problem-solver agent on problem.txt
```

The agent will produce:

```python
# output/solution_v1.py
def longest_increasing_subsequence(nums):
    """
    Find the longest increasing subsequence using dynamic programming.
    Time complexity: O(n^2)
    Space complexity: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = length of LIS ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Test cases to verify correctness
def test_solution():
    assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4
    assert longest_increasing_subsequence([7, 7, 7, 7, 7]) == 1
    assert longest_increasing_subsequence([]) == 0
    assert longest_increasing_subsequence([1]) == 1
    print("All tests passed!")

if __name__ == "__main__":
    test_solution()
```

#### solution-improver Agent

The solution-improver agent optimizes existing solutions. It identifies performance bottlenecks, reduces complexity, and enhances code quality while maintaining correctness.

```bash
# Command
Use solution-improver agent on output/solution_v1.py
```

The agent analyzes the O(n²) solution and produces:

```python
# output/solution_v2.py
import bisect

def longest_increasing_subsequence(nums):
    """
    Optimized solution using binary search.
    Time complexity: O(n log n)
    Space complexity: O(n)
    
    Improvements from v1:
    - Reduced time complexity from O(n^2) to O(n log n)
    - Uses binary search to maintain sorted subsequence
    - More efficient for large inputs
    """
    if not nums:
        return 0
    
    # tails[i] = smallest tail element for LIS of length i+1
    tails = []
    
    for num in nums:
        # Binary search for insertion position
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)

def longest_increasing_subsequence_with_sequence(nums):
    """
    Enhanced version that also returns the actual subsequence.
    """
    if not nums:
        return 0, []
    
    tails = []
    predecessors = []
    lis_indices = []
    
    for i, num in enumerate(nums):
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
            lis_indices.append(i)
        else:
            tails[pos] = num
            lis_indices[pos] = i
        
        if pos > 0:
            predecessors.append(lis_indices[pos - 1])
        else:
            predecessors.append(-1)
    
    # Reconstruct the sequence
    sequence = []
    current = lis_indices[-1]
    while current >= 0:
        sequence.append(nums[current])
        current = predecessors[current] if current < len(predecessors) else -1
    
    return len(tails), list(reversed(sequence))

# Enhanced test suite
def test_solution():
    # Original tests
    assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4
    
    # Performance test
    import time
    large_input = list(range(10000, 0, -1)) + list(range(10000))
    start = time.time()
    result = longest_increasing_subsequence(large_input)
    elapsed = time.time() - start
    assert result == 10000
    print(f"Large input processed in {elapsed:.3f} seconds")
    
    # Test with sequence reconstruction
    length, seq = longest_increasing_subsequence_with_sequence([10, 9, 2, 5, 3, 7, 101, 18])
    assert length == 4
    assert all(seq[i] < seq[i+1] for i in range(len(seq)-1))
    print("All tests passed!")
```

#### bug-report-reviewer Agent

The bug-report-reviewer agent performs systematic code analysis, identifying issues and creating actionable fix plans. It categorizes problems by severity and provides specific remediation steps.

```bash
# Command
Use bug-report-reviewer agent on output/solution_v2.py
```

The agent produces:

```markdown
# output/fix_plan.md

## Code Review Report for solution_v2.py

### Critical Issues
None identified.

### Major Issues

1. **Incorrect predecessor tracking in sequence reconstruction**
   - Location: Line 39-44 in `longest_increasing_subsequence_with_sequence`
   - Issue: Predecessors array doesn't correctly track the previous element in the LIS
   - Impact: The reconstructed sequence may be incorrect
   - Fix: Maintain a separate array to track actual predecessors for each position

### Minor Issues

1. **Missing input validation**
   - Location: Both functions
   - Issue: No validation for input type (assumes list of comparable elements)
   - Fix: Add type checking and handle edge cases like None input

2. **Incomplete documentation**
   - Location: Binary search logic
   - Issue: The binary search behavior for equal elements isn't documented
   - Fix: Clarify that bisect_left ensures stability for equal elements

### Recommendations

1. Add type hints for better code clarity:
   ```python
   from typing import List, Tuple
   
   def longest_increasing_subsequence(nums: List[int]) -> int:
   ```

2. Consider adding a parameter for custom comparison:
   ```python
   def longest_increasing_subsequence(nums, key=None, strict=True):
       # Allow for custom comparison and non-strict increasing
   ```

3. Add memory-efficient streaming version for very large inputs

### Test Coverage Gaps

- No tests for negative numbers
- No tests for floating-point numbers
- No tests for very large sequences (>100k elements)
- No tests for sequences with duplicate elements
```

#### solution-fixer Agent

The solution-fixer agent implements corrections based on fix plans, carefully applying each change and verifying the fixes don't introduce new issues.

```bash
# Command
Use solution-fixer agent on output/solution_v2.py with output/fix_plan.md
```

The agent produces:

```python
# output/solution_v3.py
import bisect
from typing import List, Tuple, Optional, Any

def longest_increasing_subsequence(nums: Optional[List[Any]]) -> int:
    """
    Find the length of the longest increasing subsequence.
    
    Time complexity: O(n log n)
    Space complexity: O(n)
    
    Args:
        nums: List of comparable elements (or None)
    
    Returns:
        Length of the longest increasing subsequence
    
    Raises:
        TypeError: If nums is not a list or contains non-comparable elements
    """
    if nums is None:
        raise TypeError("Input cannot be None")
    
    if not isinstance(nums, list):
        raise TypeError(f"Expected list, got {type(nums).__name__}")
    
    if not nums:
        return 0
    
    # Validate that elements are comparable
    try:
        # tails[i] = smallest tail element for LIS of length i+1
        tails = []
        
        for num in nums:
            # bisect_left returns the leftmost position where num should be inserted
            # to maintain sorted order. For equal elements, this maintains stability.
            pos = bisect.bisect_left(tails, num)
            
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    
    except TypeError as e:
        raise TypeError(f"Elements must be comparable: {e}")

def longest_increasing_subsequence_with_sequence(
    nums: Optional[List[Any]]
) -> Tuple[int, List[Any]]:
    """
    Find both the length and actual sequence of the LIS.
    
    Fixed version with correct predecessor tracking.
    
    Returns:
        Tuple of (length, sequence)
    """
    if nums is None:
        raise TypeError("Input cannot be None")
    
    if not isinstance(nums, list):
        raise TypeError(f"Expected list, got {type(nums).__name__}")
    
    if not nums:
        return 0, []
    
    n = len(nums)
    
    # dp[i] = length of LIS ending at index i
    dp = [1] * n
    
    # parent[i] = index of previous element in LIS ending at i
    parent = [-1] * n
    
    max_length = 1
    max_index = 0
    
    # Build the dp and parent arrays
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct the sequence
    sequence = []
    current = max_index
    
    while current != -1:
        sequence.append(nums[current])
        current = parent[current]
    
    sequence.reverse()
    
    return max_length, sequence

def test_solution():
    """Comprehensive test suite including fixes for identified gaps."""
    
    # Original tests
    assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4
    assert longest_increasing_subsequence([7, 7, 7, 7, 7]) == 1
    assert longest_increasing_subsequence([]) == 0
    assert longest_increasing_subsequence([1]) == 1
    
    # New tests for gaps identified in review
    
    # Test with negative numbers
    assert longest_increasing_subsequence([-10, -5, 0, 5, 10]) == 5
    assert longest_increasing_subsequence([-1, -2, -3]) == 1
    
    # Test with floating-point numbers
    assert longest_increasing_subsequence([1.5, 2.5, 1.0, 3.0]) == 3
    
    # Test with duplicates
    assert longest_increasing_subsequence([1, 2, 2, 3]) == 3
    
    # Test sequence reconstruction
    length, seq = longest_increasing_subsequence_with_sequence([10, 9, 2, 5, 3, 7, 101, 18])
    assert length == 4
    assert seq == [2, 3, 7, 101] or seq == [2, 3, 7, 18]  # Both are valid
    
    # Test error handling
    try:
        longest_increasing_subsequence(None)
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    try:
        longest_increasing_subsequence("not a list")
        assert False, "Should raise TypeError"
    except TypeError:
        pass
    
    print("All tests passed!")

if __name__ == "__main__":
    test_solution()
```

### Workflow Orchestration Patterns

#### Pattern 1: Complete Development Cycle

For new feature development, use all agents in sequence:

```bash
# Step 1: Initial solution
Use problem-solver agent on requirements.txt
# Creates: solution_v1.py

# Step 2: Optimization
Use solution-improver agent on output/solution_v1.py
# Creates: solution_v2.py

# Step 3: Review
Use bug-report-reviewer agent on output/solution_v2.py
# Creates: fix_plan_1.md

# Step 4: Fix issues
Use solution-fixer agent on output/solution_v2.py with output/fix_plan_1.md
# Creates: solution_v3.py

# Step 5: Final review
Use bug-report-reviewer agent on output/solution_v3.py
# Creates: fix_plan_2.md

# Step 6: Final fixes if needed
Use solution-fixer agent on output/solution_v3.py with output/fix_plan_2.md
# Creates: solution_final.py
```

#### Pattern 2: Debugging Existing Code

When working with existing buggy code:

```bash
# Step 1: Analyze existing code
Use bug-report-reviewer agent on legacy_code.py
# Creates: fix_plan.md

# Step 2: Apply fixes
Use solution-fixer agent on legacy_code.py with output/fix_plan.md
# Creates: legacy_code_fixed.py

# Step 3: Optimize if needed
Use solution-improver agent on output/legacy_code_fixed.py
# Creates: legacy_code_optimized.py
```

#### Pattern 3: Competitive Programming

For time-sensitive algorithmic problems:

```bash
# Step 1: Get working solution fast
Use problem-solver agent on problem.txt with focus on correctness
# Creates: solution_v1.py

# Step 2: Optimize for time limits
Use solution-improver agent on output/solution_v1.py with focus on complexity
# Creates: solution_v2.py

# Step 3: Quick validation
Use bug-report-reviewer agent on output/solution_v2.py with focus on edge cases
# Creates: validation_report.md
```

### Context Management Between Agents

Each agent operates with a fresh context, so information must be explicitly passed. Always include:

1. The original problem statement
2. Previous solution versions
3. Previous fix plans and reports
4. Any test results or error messages

Example of proper context passing:

```bash
# Bad: Missing context
Use solution-fixer agent on solution.py

# Good: Complete context
Use solution-fixer agent on output/solution_v2.py with:
- Original problem from problem.txt
- Previous version from output/solution_v1.py  
- Issues identified in output/fix_plan.md
- Test failures from output/test_results.txt
```

### Iteration Control and Termination

The workflow should terminate when:

1. All tests pass consistently
2. No major issues remain in bug reports
3. Performance meets requirements
4. Maximum iteration count (5) is reached

Track progress using metrics:

```python
# tracking.py
class WorkflowTracker:
    def __init__(self):
        self.iterations = []
    
    def record_iteration(self, version, test_pass_rate, issues_count, performance):
        self.iterations.append({
            'version': version,
            'test_pass_rate': test_pass_rate,
            'critical_issues': issues_count['critical'],
            'major_issues': issues_count['major'],
            'performance': performance
        })
    
    def should_continue(self):
        if len(self.iterations) >= 5:
            return False
        
        latest = self.iterations[-1]
        if latest['test_pass_rate'] == 1.0 and latest['critical_issues'] == 0:
            return False
        
        # Check if we're making progress
        if len(self.iterations) >= 2:
            previous = self.iterations[-2]
            if latest['test_pass_rate'] <= previous['test_pass_rate']:
                return False  # Not improving
        
        return True
```

---

## 4. Rust Backend Development Guidelines

# CLAUDE_RUST_GUIDELINE_EXPERT.md: Production-Grade Rust Backend Development Guide

> **Goal**: This guide helps internalize production Rust code patterns to consistently develop production-grade Rust services.

---

## 📋 Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Project Structure](#2-project-structure)
3. [Error Handling and Observability](#3-error-handling-and-observability)
4. [Configuration Management](#4-configuration-management)
5. [HTTP Layer and Routing](#5-http-layer-and-routing)
6. [API Documentation with OpenAPI](#6-api-documentation-with-openapi)
7. [Asynchronous Programming Patterns](#7-asynchronous-programming-patterns)
8. [External Process Management](#8-external-process-management)
9. [Object Storage Patterns](#9-object-storage-patterns)
10. [Security and Authentication](#10-security-and-authentication)
11. [Database Patterns](#11-database-patterns)
12. [Testing Strategies](#12-testing-strategies)
13. [Coding Conventions](#13-coding-conventions)
14. [Common Service Patterns](#14-common-service-patterns)
15. [Developer Checklist](#15-developer-checklist)

---

## 1. Core Philosophy

Production-level Rust code should follow these principles without exception:

| Principle | Why It Matters | Implementation in Code |
|-----------|----------------|------------------------|
| **Fail Fast and Clear** | Production debugging requires rich context | Always use `eyre::WrapErr` or `context()`, completely ban `unwrap()` except in tests |
| **Instrument Everything** | Distributed systems need end-to-end tracing | Use `#[tracing::instrument]` on every public function, `.in_current_span()` for spawned tasks |
| **Secure by Default** | Security cannot be an afterthought | All endpoints require authentication unless explicitly public, validate all inputs |
| **Data First** | Types drive system behavior and documentation | Define data models with `serde`, `utoipa::ToSchema`, use semantic types |
| **Loose Coupling** | Changes should have limited blast radius | Module boundaries, dependency injection, trait abstractions |
| **Async Native** | Modern services are I/O bound | Never block async runtime, use `tokio::spawn_blocking` for CPU work |
| **Observable by Design** | Can't fix what you can't measure | OpenTelemetry integration, structured logging, metrics for everything |

### Key Mental Models

**The Onion Architecture**: Your code should have clear layers - domain logic at the core, infrastructure at the edges. Dependencies point inward. The domain layer should never know about HTTP, databases, or external services.

**Error Context Accumulation**: Every error should add context as it bubbles up. By the time an error reaches the top level, it should tell a complete story of what went wrong, where, and why.

**Semantic Typing**: Don't use primitive types for domain concepts. A user ID is not a String, it's a `UserId`. An email is not a String, it's an `Email`. This prevents entire classes of bugs at compile time.

**Resource Lifecycle Management**: Every resource (connection, file handle, lock) should have a clear owner and cleanup strategy. Use RAII patterns consistently.

---

## 2. Project Structure

### Standard Service Layout

Every production Rust service should follow this structure to ensure consistency and discoverability:

```
crate/
├── src/
│   ├── main.rs                 # Entry point only - initialization and shutdown
│   ├── lib.rs                  # Public API when used as a library
│   ├── config.rs               # All configuration in one place
│   ├── error.rs                # Custom error types and conversions
│   ├── telemetry.rs            # Logging, tracing, metrics setup
│   │
│   ├── api/                    # HTTP layer (no business logic)
│   │   ├── mod.rs              # Router assembly
│   │   ├── middleware/         # Cross-cutting concerns
│   │   ├── extractors/         # Custom Axum extractors
│   │   ├── v1/                 # Version 1 endpoints
│   │   └── v2/                 # Version 2 endpoints
│   │
│   ├── domain/                 # Pure business logic
│   │   ├── mod.rs
│   │   ├── models/             # Domain entities
│   │   ├── services/           # Business operations
│   │   └── validators/         # Business rule validation
│   │
│   ├── infrastructure/         # External world interaction
│   │   ├── mod.rs
│   │   ├── database/           # Database repositories
│   │   ├── cache/              # Redis/in-memory caching
│   │   ├── messaging/          # Kafka/RabbitMQ/SQS
│   │   └── storage/            # S3/blob storage
│   │
│   └── utils/                  # Shared utilities
│       ├── mod.rs
│       ├── crypto.rs           # Hashing, encryption
│       └── validation.rs       # Common validators
│
├── migrations/                  # SQL migrations
├── tests/
│   ├── integration/            # Full API tests
│   └── fixtures/               # Test data
│
└── benches/                    # Performance benchmarks
```

### Module Organization Rules

1. **One concept per module**: Each module should have a single, clear responsibility
2. **Public API at module root**: Re-export public items in `mod.rs`
3. **Tests alongside code**: Unit tests in the same file, integration tests in `/tests`
4. **Maximum module size**: Split modules larger than 500 lines
5. **Hierarchical organization**: Submodules for related functionality

### Dependency Direction Rules

- **Domain → Nothing**: Domain layer has zero external dependencies
- **Infrastructure → Domain**: Infrastructure implements domain traits
- **API → Domain + Infrastructure**: API orchestrates but contains no logic
- **Never circular**: Use dependency injection to break cycles

---

## 3. Error Handling and Observability

### Error Handling Philosophy

Every error in production must answer three questions:
1. **What went wrong?** (The immediate error)
2. **Where did it go wrong?** (The context chain)
3. **Why did it go wrong?** (The root cause)

### Using eyre/anyhow Effectively

Choose `eyre` for applications (better error reports) or `anyhow` for libraries (smaller dependency).

**Rule 1: Always add context**
```rust
// ❌ Bad: No context
let user = fetch_user(id).await?;

// ✅ Good: Rich context
let user = fetch_user(id)
    .await
    .with_context(|| format!("Failed to fetch user with id={}", id))?;
```

**Rule 2: Use semantic error types at boundaries**
```rust
// ✅ Good: Semantic errors for API responses
pub enum ApiError {
    NotFound(String),
    Unauthorized,
    ValidationFailed(Vec<ValidationError>),
    Internal(eyre::Report),
}
```

**Rule 3: Wrap errors when crossing boundaries**
```rust
// ✅ Good: Transform infrastructure errors to domain errors
impl From<sqlx::Error> for DomainError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => DomainError::NotFound,
            _ => DomainError::Internal(err.into()),
        }
    }
}
```

### Observability with OpenTelemetry

Modern production services require three pillars of observability: logs, metrics, and traces. OpenTelemetry provides a vendor-agnostic way to collect all three.

**Setup Strategy**:
1. Initialize OpenTelemetry before anything else
2. Use environment variables for exporter configuration
3. Gracefully flush telemetry on shutdown
4. Use structured logging with trace correlation

**Tracing Best Practices**:

1. **Instrument at boundaries**: Every API endpoint, external call, and major operation
2. **Record key business events**: User actions, state transitions, errors
3. **Use semantic conventions**: Follow OpenTelemetry semantic conventions for span names and attributes
4. **Sampling strategy**: 100% for errors, 10% for normal traffic in production
5. **Span attributes**: Include user ID, request ID, and other correlation IDs

**Critical Spans to Always Include**:
- HTTP request/response cycles
- Database queries with SQL
- External API calls with URLs
- Message queue operations
- Cache hits/misses
- Background job execution

### Structured Logging Guidelines

**Log Levels and Their Meaning**:

| Level | Use Case | Example |
|-------|----------|---------|
| ERROR | Requires immediate attention | Database connection lost |
| WARN | Potential problem, but system continues | Rate limit approaching |
| INFO | Important business events | User registered, Order placed |
| DEBUG | Detailed execution flow | Entering function, Query results |
| TRACE | Very detailed debugging | Full request/response bodies |

**What to Log at Each Level**:

- **ERROR**: Unrecoverable errors, failed critical operations, security violations
- **WARN**: Retryable failures, deprecated API usage, performance degradation
- **INFO**: Application lifecycle, business transactions, external integrations
- **DEBUG**: Control flow, intermediate states, non-sensitive data
- **TRACE**: Full payloads, sensitive data (dev only), execution timings

**Logging Anti-patterns to Avoid**:
1. Logging sensitive data (passwords, tokens, PII)
2. Excessive logging in loops
3. Logging without context
4. Using println! or dbg! in production code
5. Inconsistent log formats

---

## 4. Configuration Management

### Configuration Philosophy

Configuration should be:
- **Typed**: Use structs, not hashmaps
- **Validated**: Fail fast on invalid configuration
- **Centralized**: One place for all configuration
- **Hierarchical**: Nested structs for organization
- **Documented**: Every field should have a purpose

### Configuration Sources (in priority order)

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration files** (.env for development)
4. **Default values** (lowest priority)

### LazyLock Pattern for Global Configuration

Use `LazyLock` for configuration that's initialized once and accessed everywhere:

```rust
pub static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    Config::from_env().expect("Failed to load configuration")
});
```

### Environment-Specific Configuration

**Development Environment**:
- Verbose logging
- Permissive CORS
- Exposed debugging endpoints
- Short timeouts for fast feedback

**Staging Environment**:
- Production-like configuration
- Real external services
- Moderate logging
- Performance monitoring enabled

**Production Environment**:
- Minimal logging (INFO level)
- Strict CORS policies
- Long cache TTLs
- Maximum security settings

### Configuration Validation Rules

1. **Required fields must not have defaults** - Force explicit configuration
2. **Validate ranges** - Ports, percentages, counts must be within valid ranges
3. **Validate relationships** - max_connections >= min_connections
4. **Validate formats** - URLs, emails, regexes must be valid
5. **Environment-specific validation** - Stricter rules for production

### Secret Management

**Never**:
- Commit secrets to version control
- Log secrets at any level
- Use default passwords in production
- Share secrets between environments

**Always**:
- Use environment variables for secrets
- Rotate secrets regularly
- Use different secrets per environment
- Validate secret strength at startup

---

## 5. HTTP Layer and Routing

### Router Organization Principles

The router is the entry point to your application. It should be:
- **Declarative**: Routes should clearly show API structure
- **Versioned**: Support multiple API versions simultaneously
- **Middleware-rich**: Cross-cutting concerns in middleware
- **Documented**: OpenAPI spec for every endpoint

### Routing Best Practices

**Route Naming Conventions**:
- Use lowercase with hyphens: `/user-profiles` not `/userProfiles`
- Use nouns for resources: `/users` not `/get-users`
- Use verbs for actions: `/users/{id}/activate`
- Version at the path level: `/api/v1/users`

**HTTP Method Semantics**:
| Method | Purpose | Idempotent | Safe | Request Body |
|--------|---------|------------|------|--------------|
| GET | Retrieve resource | Yes | Yes | No |
| POST | Create resource | No | No | Yes |
| PUT | Replace resource | Yes | No | Yes |
| PATCH | Update resource | No | No | Yes |
| DELETE | Remove resource | Yes | No | No |

**Status Code Guidelines**:
| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Successful GET, PUT, PATCH, DELETE |
| 201 | Created | Successful POST creating resource |
| 202 | Accepted | Request queued for processing |
| 204 | No Content | Successful DELETE, no response body |
| 400 | Bad Request | Invalid request format/parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but not authorized |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Resource state conflict |
| 422 | Unprocessable | Valid format but semantic errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Error | Server error (log details) |
| 503 | Service Unavailable | Temporary outage/maintenance |

### Middleware Stack Order

Middleware order matters. Apply in this sequence:
1. **Request ID injection** - For correlation
2. **Tracing** - Instrument everything
3. **Timeout** - Prevent hanging requests
4. **CORS** - Handle preflight requests
5. **Compression** - Reduce bandwidth
6. **Rate limiting** - Protect from abuse
7. **Authentication** - Identify user
8. **Authorization** - Check permissions
9. **Request validation** - Validate input
10. **Business logic** - Your handlers

### Request/Response Best Practices

**Request Validation**:
- Validate at the edge (in extractors)
- Use strong typing (parse, don't validate)
- Return all validation errors at once
- Provide clear error messages
- Validate business rules in domain layer

**Response Formatting**:
- Use consistent envelope format
- Include request ID in responses
- Provide pagination metadata
- Use ISO 8601 for timestamps
- Return created resource location

**Pagination Standards**:
- Default page size: 20-50 items
- Maximum page size: 100-500 items
- Use cursor pagination for large datasets
- Include total count when feasible
- Return next/previous links

### Static File Serving with rust-embed

For serving frontend assets, documentation, or other static files:

**When to Use**:
- Single binary deployment
- Embedded admin interfaces
- API documentation
- Static assets for SPAs

**Configuration Rules**:
- Set appropriate cache headers
- Use content-type detection
- Support gzip/brotli compression
- Handle 404s gracefully
- Implement etag support

---

## 6. API Documentation with OpenAPI

### Documentation as Code Philosophy

API documentation should be:
- **Generated from code** - Single source of truth
- **Always up-to-date** - Can't drift from implementation
- **Interactive** - Try it out functionality
- **Versioned** - Document all supported versions
- **Complete** - Every endpoint, every field

### Utoipa Integration Strategy

Use utoipa to generate OpenAPI specifications directly from Rust code:

**Documentation Levels**:
1. **Endpoint level** - Path, method, parameters
2. **Schema level** - Request/response bodies
3. **Field level** - Descriptions, constraints
4. **Example level** - Sample requests/responses

**What to Document**:
- Purpose of each endpoint
- Required vs optional parameters
- Default values and constraints
- Error responses and their meanings
- Rate limits and quotas
- Authentication requirements

### Schema Design Principles

**Request Schemas**:
- Use dedicated types for each endpoint
- Validate all constraints
- Provide sensible defaults
- Use semantic field names
- Document units (seconds, bytes, etc.)

**Response Schemas**:
- Use enums for closed sets
- Include metadata (timestamps, versions)
- Nest related data logically
- Use consistent field naming
- Provide expansion capabilities

**Error Schemas**:
- Consistent error format across API
- Include error codes for programmatic handling
- Provide human-readable messages
- Include field-level errors for validation
- Add debugging information in non-production

### Versioning Strategies

**When to Version**:
- Breaking changes to request format
- Removing response fields
- Changing field types
- Modifying authentication
- Altering rate limits

**How to Version**:
- Path versioning for major versions (/v1, /v2)
- Header versioning for minor versions
- Query parameter for experimental features
- Sunset headers for deprecation
- Migration guides for upgrades

---

## 7. Asynchronous Programming Patterns

### Async/Await Best Practices

**Fundamental Rules**:
1. Never block the async runtime
2. Use `tokio::spawn_blocking` for CPU-intensive work
3. Always propagate cancellation
4. Avoid holding locks across await points
5. Prefer channels over shared state

### Task Spawning Patterns

**When to Spawn Tasks**:
- Fire-and-forget operations
- Parallel independent work
- Background processing
- Periodic tasks
- Cleanup operations

**Task Lifecycle Management**:
- Always handle join errors
- Implement graceful shutdown
- Use abort handles for forced termination
- Monitor task panics
- Limit concurrent tasks

**Spawn Patterns**:

| Pattern | Use Case | Example |
|---------|----------|---------|
| `spawn` | Independent work | Background email sending |
| `spawn_blocking` | CPU-intensive work | Image processing, cryptography |
| `spawn_local` | !Send futures | JavaScript runtime interaction |
| `try_join!` | Parallel with all-success | Multiple API calls |
| `select!` | First completion wins | Timeout implementation |
| `join!` | Wait for all | Parallel initialization |

### Concurrency Control

**Limiting Concurrency**:
- Use `Semaphore` for resource pools
- Use `buffer_unordered` for stream processing
- Implement backpressure mechanisms
- Set connection pool limits
- Rate limit external calls

**Synchronization Primitives**:

| Primitive | Use Case | Notes |
|-----------|----------|-------|
| `Mutex` | Exclusive access | Avoid holding across awaits |
| `RwLock` | Multiple readers | Good for caches |
| `Semaphore` | Limit concurrency | Resource pooling |
| `Barrier` | Synchronization point | Batch processing |
| `Notify` | Task coordination | Wake one or all |
| `Channel` | Message passing | Preferred over shared state |

### Stream Processing Patterns

**Stream Operations**:
- Use `buffered` for parallel async operations
- Use `buffer_unordered` when order doesn't matter
- Implement chunking for batch processing
- Add timeout for slow consumers
- Handle backpressure properly

**Error Handling in Streams**:
- Use `try_stream!` for fallible streams
- Collect errors vs fail fast decision
- Implement retry logic with exponential backoff
- Log and skip bad items when appropriate
- Always set maximum retry limits

### Cancellation and Timeouts

**Timeout Strategy**:
- Set timeouts at multiple levels
- Use shorter timeouts in development
- Implement cascade timeout (total > sum of parts)
- Always handle timeout errors explicitly
- Log slow operations before timeout

**Cancellation Patterns**:
- Use `CancellationToken` for coordinated shutdown
- Implement graceful degradation
- Clean up resources on cancellation
- Propagate cancellation through call chain
- Test cancellation paths

---

## 8. External Process Management

### When to Use External Processes

**Appropriate Use Cases**:
- Invoking system utilities (pdf2text, imagemagick)
- Running Python/Node.js scripts
- Interfacing with legacy systems
- Sandboxing untrusted code
- GPU computation (CUDA, ML models)

**Avoid When**:
- Pure Rust solution exists
- Performance is critical
- Complex bidirectional communication needed
- Process overhead exceeds benefit

### Process Execution Safety Rules

1. **Never use shell expansion** - Avoid shell injection
2. **Validate all inputs** - Sanitize before passing to process
3. **Set resource limits** - CPU, memory, file descriptors
4. **Handle all exit codes** - Don't assume success
5. **Capture stderr** - Contains important error info
6. **Set timeouts** - Prevent hanging processes
7. **Clean up zombies** - Always wait on children

### Input/Output Handling

**Stdin Handling**:
- Use pipes for data transfer
- Close stdin after writing
- Handle SIGPIPE errors
- Set write timeouts
- Buffer large inputs appropriately

**Stdout/Stderr Handling**:
- Read both streams to prevent deadlock
- Use separate tasks for each stream
- Handle partial reads
- Set buffer size limits
- Parse output incrementally

### Process Lifecycle Management

**Startup**:
- Verify executable exists and is executable
- Set working directory explicitly
- Configure environment variables
- Set process group for cleanup
- Handle spawn failures gracefully

**Monitoring**:
- Track process state
- Monitor resource usage
- Implement heartbeat checks
- Log process events
- Collect performance metrics

**Shutdown**:
- Send SIGTERM first
- Wait with timeout
- Send SIGKILL if needed
- Clean up temporary files
- Log abnormal terminations

---

## 9. Object Storage Patterns

### Object Storage Best Practices

Modern applications often need to store files, images, or documents. Object storage (S3, GCS, Azure Blob) provides scalable, durable storage.

### Using object_store Crate

The `object_store` crate provides a unified interface across cloud providers:

**Advantages**:
- Vendor-agnostic API
- Async-first design
- Streaming support
- Automatic retries
- Presigned URLs

**Setup Principles**:
- Use environment variables for configuration
- Create singleton clients with LazyLock
- Set appropriate timeouts
- Configure retry policies
- Handle rate limits

### Upload Patterns

**Single File Upload**:
- Validate file size before upload
- Set content-type explicitly
- Use multipart for large files (>5MB)
- Generate unique keys (UUIDs)
- Handle duplicate uploads idempotently

**Batch Upload**:
- Limit concurrent uploads (50-100)
- Use semaphores for rate limiting
- Implement progress tracking
- Handle partial failures
- Provide resume capability

**Streaming Upload**:
- Use for large files or unknown size
- Implement chunking
- Handle network interruptions
- Validate checksums
- Clean up partial uploads

### Download Patterns

**Direct Download**:
- Stream to client without buffering
- Set appropriate cache headers
- Support range requests
- Validate etags
- Handle not found gracefully

**Presigned URLs**:
- Use for client-direct uploads/downloads
- Set short expiration times
- Include content-type restrictions
- Limit upload size
- Log URL generation

### Key Naming Strategies

**Best Practices**:
- Use forward slashes for hierarchy
- Include timestamp in key
- Add content hash for deduplication
- Avoid special characters
- Keep keys under 1024 bytes

**Example Patterns**:
```
uploads/{year}/{month}/{day}/{uuid}.{extension}
users/{user_id}/avatars/{timestamp}_{hash}.jpg
documents/{doc_type}/{doc_id}/v{version}/file.pdf
temp/{date}/{session_id}/{filename}
```

### Storage Lifecycle Management

**Retention Policies**:
- Set TTL for temporary files
- Archive old data to cheaper storage
- Delete orphaned uploads
- Implement soft delete
- Maintain audit trails

**Cost Optimization**:
- Use appropriate storage classes
- Enable compression where suitable
- Implement client-side caching
- Batch small files
- Monitor usage patterns

---

## 10. Security and Authentication

### Security First Mindset

Security must be built-in, not bolted-on. Every line of code should consider security implications.

### Authentication Strategies

**JWT (JSON Web Tokens)**:
- **Use when**: Stateless auth, microservices, mobile apps
- **Avoid when**: Need immediate revocation, sensitive operations
- **Best practices**: Short expiration, refresh tokens, RS256 for public APIs

**Session Tokens**:
- **Use when**: Traditional web apps, need revocation
- **Avoid when**: Stateless architecture, high scale
- **Best practices**: Secure cookies, CSRF protection, regular rotation

**API Keys**:
- **Use when**: Service-to-service, webhooks, public APIs
- **Avoid when**: User authentication, browser-based apps
- **Best practices**: Scope limitations, rate limiting, regular rotation

### Authorization Patterns

**Role-Based Access Control (RBAC)**:
- Define clear roles (admin, user, viewer)
- Assign permissions to roles
- Check at API boundary
- Audit role assignments
- Implement role hierarchy

**Attribute-Based Access Control (ABAC)**:
- Flexible policy rules
- Context-aware decisions
- Dynamic permissions
- Complex business rules
- Performance considerations

**Resource-Based Authorization**:
- Check ownership
- Implement scopes
- Validate relationships
- Cache decisions
- Audit access

### Input Validation and Sanitization

**Validation Layers**:
1. **Type level** - Use strong types
2. **Format level** - Regex, length, range
3. **Business level** - Domain rules
4. **Security level** - Injection prevention

**Common Vulnerabilities to Prevent**:
- SQL injection - Use parameterized queries
- XSS - Escape HTML output
- CSRF - Use tokens
- Path traversal - Validate file paths
- Command injection - Avoid shell commands
- XXE - Disable XML external entities

### Cryptography Guidelines

**Password Handling**:
- Use Argon2id for hashing
- Never store plain text
- Implement password policies
- Use secure random salts
- Support password managers

**Encryption**:
- Use AES-256-GCM for symmetric
- Use RSA-2048 minimum for asymmetric
- Generate secure random keys
- Rotate keys regularly
- Store keys separately

**Random Number Generation**:
- Use OS random for security
- Never use timestamp as seed
- Generate sufficient entropy
- Use appropriate distributions
- Validate randomness quality

### Rate Limiting and DDoS Protection

**Rate Limiting Strategies**:
- Per-user limits
- Per-IP limits
- Per-endpoint limits
- Global limits
- Adaptive limits

**Implementation Approaches**:
- Token bucket algorithm
- Sliding window
- Fixed window
- Leaky bucket
- Distributed rate limiting

---

## 11. Database Patterns

### Connection Pool Management

**Pool Configuration Guidelines**:
- Min connections = number of workers
- Max connections = min * 4
- Connection timeout = 30 seconds
- Idle timeout = 10 minutes
- Max lifetime = 30 minutes

**Health Checking**:
- Test connections before use
- Implement retry logic
- Monitor pool metrics
- Alert on exhaustion
- Gradual reconnection

### Query Patterns

**Query Optimization**:
- Use prepared statements
- Batch operations when possible
- Implement pagination
- Use appropriate indices
- Monitor slow queries

**Transaction Management**:
- Keep transactions short
- Use appropriate isolation levels
- Handle deadlocks gracefully
- Implement retry logic
- Monitor long-running transactions

### Migration Strategy

**Migration Best Practices**:
- Version control all migrations
- Make migrations idempotent
- Test rollback procedures
- Use transactions for DDL
- Separate schema and data migrations

**Safety Rules**:
- Never drop columns immediately
- Add defaults for new NOT NULL columns
- Create indices concurrently
- Test on production-like data
- Have rollback plan

### Caching Patterns

**Cache Levels**:
1. **Application memory** - Fastest, limited size
2. **Redis** - Shared, persistent option
3. **Database** - Materialized views
4. **CDN** - Static content

**Cache Strategies**:
- Cache-aside - Read through cache
- Write-through - Update cache on write
- Write-behind - Async cache update
- Refresh-ahead - Proactive refresh
- TTL-based - Time-based expiration

---

## 12. Testing Strategies

### Testing Pyramid

**Unit Tests (70%)**:
- Test individual functions
- Mock external dependencies
- Fast execution
- High coverage
- Run on every commit

**Integration Tests (20%)**:
- Test module interactions
- Use test databases
- Test API contracts
- Verify error handling
- Run before merge

**End-to-End Tests (10%)**:
- Test complete workflows
- Production-like environment
- Performance testing
- Security testing
- Run before release

### Test Organization

**Test File Structure**:
- Unit tests in same file as code
- Integration tests in tests/ directory
- Benchmarks in benches/ directory
- Fixtures in tests/fixtures/
- Helpers in tests/common/

**Test Naming Conventions**:
```rust
#[test]
fn test_module_function_scenario_expected_result() { }

// Examples:
fn test_user_service_create_with_valid_data_returns_user() { }
fn test_auth_middleware_missing_token_returns_401() { }
```

### Database Testing

**Test Database Strategies**:
- Use transactions with rollback
- Fresh database per test
- Fixtures for common data
- Test migrations separately
- Use in-memory when possible

**SQLx Test Macros**:
- Use `#[sqlx::test]` for async tests
- Automatic transaction rollback
- Connection pool provided
- Migration support
- Parallel test execution

### Mocking and Stubbing

**When to Mock**:
- External services
- Time-dependent code
- Random number generation
- File system operations
- Network calls

**Mocking Strategies**:
- Trait-based mocking
- Test doubles
- Fixture files
- Record/replay
- Service virtualization

---

## 13. Coding Conventions

### Naming Rules

| Item | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `user_service` |
| Types | PascalCase | `UserProfile` |
| Functions | snake_case | `calculate_total` |
| Variables | snake_case | `user_count` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_RETRIES` |
| Lifetimes | short lowercase | `'a`, `'b` |
| Type parameters | short uppercase | `T`, `K`, `V` |

### Import Organization

Order imports in this sequence:
1. Standard library imports
2. External crate imports
3. Internal crate imports
4. Super imports
5. Self imports

Group and alphabetize within each section.

### Error Handling Patterns

**Always prefer `?` over `unwrap()`**:
```rust
// ❌ Bad
let data = dangerous_operation().unwrap();

// ✅ Good
let data = dangerous_operation()
    .context("Failed to perform operation")?;
```

**Use `expect()` only in initialization**:
```rust
// ✅ Acceptable in main() or initialization
let config = Config::from_env()
    .expect("Failed to load config");
```

### Documentation Standards

**Module Documentation**:
- Purpose and responsibility
- Usage examples
- Important notes
- Link to related modules

**Function Documentation**:
- What it does
- Parameters explained
- Return value described
- Errors enumerated
- Examples provided
- Panics documented

**Type Documentation**:
- Purpose
- Invariants
- Usage patterns
- Field descriptions

### Code Organization

**Function Length**: Maximum 50 lines, prefer under 30
**Cyclomatic Complexity**: Maximum 10, prefer under 5
**Nesting Depth**: Maximum 4 levels
**Line Length**: Maximum 100 characters
**File Length**: Maximum 500 lines

### Performance Considerations

**Allocation Awareness**:
- Reuse buffers
- Use `Cow` for conditional ownership
- Prefer `&str` over `String` in APIs
- Use `SmallVec` for small collections
- Pool expensive resources

**Async Optimizations**:
- Batch operations
- Use channels over mutex
- Prefer `join!` over sequential awaits
- Implement debouncing
- Cache computed values

---

## 14. Common Service Patterns

### Health Check Pattern

Every service needs health checks for orchestration systems:

**Liveness Check** (/health/live):
- Simple response indicating service is running
- Should not check dependencies
- Used for restart decisions
- Fast response (< 100ms)

**Readiness Check** (/health/ready):
- Validates all dependencies
- Database connectivity
- Cache availability
- External service health
- Used for traffic routing

### Graceful Shutdown Pattern

Services must shut down cleanly:

1. Stop accepting new requests
2. Wait for in-flight requests (with timeout)
3. Close database connections
4. Flush telemetry
5. Clean up resources
6. Exit with appropriate code

### Circuit Breaker Pattern

Protect against cascading failures:

**States**:
- **Closed**: Normal operation
- **Open**: Failing, reject requests
- **Half-Open**: Testing recovery

**Configuration**:
- Failure threshold: 5 failures
- Success threshold: 3 successes
- Timeout: 60 seconds
- Reset timeout: 120 seconds

### Retry Pattern

Handle transient failures gracefully:

**Retry Strategy**:
- Maximum attempts: 3-5
- Exponential backoff
- Jitter to prevent thundering herd
- Circuit breaker integration
- Different strategies per error type

### Feature Flag Pattern

Deploy code without releasing features:

**Implementation**:
- Environment-based flags
- User-based flags
- Percentage rollout
- A/B testing support
- Kill switches

### Webhook Pattern

Reliable webhook delivery:

**Requirements**:
- Idempotency keys
- Exponential backoff retry
- Dead letter queue
- Signature verification
- Event ordering

### Background Job Pattern

Async processing requirements:

**Job Properties**:
- Unique job ID
- Idempotent execution
- Progress tracking
- Cancellation support
- Result storage

**Execution Strategy**:
- Priority queues
- Rate limiting
- Concurrent execution limits
- Failure handling
- Monitoring


### Project Architecture and Organization

Production Rust services require careful organization to maintain clarity as the codebase grows. The structure should reflect both the domain model and the technical architecture, making it easy for new developers to understand the system's design.

```rust
// Project structure for a production service
crate/
├── src/
│   ├── main.rs                 // Entry point and server initialization
│   ├── lib.rs                  // Public API when used as library
│   ├── config.rs               // Configuration management
│   ├── error.rs                // Error types and handling
│   ├── telemetry.rs            // Observability setup
│   │
│   ├── api/                    // HTTP API layer
│   │   ├── mod.rs
│   │   ├── router.rs           // Route definitions
│   │   ├── middleware.rs       // Custom middleware
│   │   ├── extractors.rs       // Custom extractors
│   │   ├── v1/
│   │   │   ├── mod.rs
│   │   │   ├── users.rs
│   │   │   └── orders.rs
│   │   └── v2/
│   │       ├── mod.rs
│   │       └── users.rs
│   │
│   ├── domain/                 // Business logic
│   │   ├── mod.rs
│   │   ├── user.rs
│   │   ├── order.rs
│   │   └── inventory.rs
│   │
│   ├── infrastructure/         // External services
│   │   ├── mod.rs
│   │   ├── database.rs
│   │   ├── cache.rs
│   │   ├── message_queue.rs
│   │   └── object_storage.rs
│   │
│   └── utils/                  // Shared utilities
│       ├── mod.rs
│       ├── crypto.rs
│       └── validation.rs
```

### Configuration Management with Type Safety

Configuration should be loaded once at startup, validated, and made globally accessible through a type-safe interface:

```rust
// src/config.rs
use std::sync::LazyLock;
use std::time::Duration;
use serde::Deserialize;
use eyre::{Context, Result};
use url::Url;

/// Application configuration loaded from environment variables
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    /// Application environment (local, staging, production)
    #[serde(default = "default_environment")]
    pub environment: Environment,
    
    /// Server configuration
    pub server: ServerConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Redis cache configuration
    pub cache: CacheConfig,
    
    /// Object storage configuration
    pub storage: StorageConfig,
    
    /// External service URLs
    pub services: ServiceUrls,
    
    /// Feature flags
    #[serde(default)]
    pub features: FeatureFlags,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Local,
    Staging,
    Production,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    
    #[serde(default = "default_port")]
    pub port: u16,
    
    #[serde(default = "default_workers")]
    pub workers: usize,
    
    #[serde(with = "humantime_serde", default = "default_shutdown_timeout")]
    pub shutdown_timeout: Duration,
    
    #[serde(default = "default_body_limit")]
    pub max_body_size: usize,
    
    pub jwt_secret: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    
    #[serde(default = "default_min_connections")]
    pub min_connections: u32,
    
    #[serde(with = "humantime_serde", default = "default_connection_timeout")]
    pub connection_timeout: Duration,
    
    #[serde(with = "humantime_serde", default = "default_idle_timeout")]
    pub idle_timeout: Duration,
    
    #[serde(default)]
    pub enable_logging: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CacheConfig {
    pub redis_url: String,
    
    #[serde(default = "default_cache_pool_size")]
    pub pool_size: u32,
    
    #[serde(with = "humantime_serde", default = "default_cache_ttl")]
    pub default_ttl: Duration,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StorageConfig {
    pub bucket: String,
    pub region: String,
    
    #[serde(default)]
    pub endpoint: Option<Url>,
    
    #[serde(default = "default_upload_concurrency")]
    pub upload_concurrency: usize,
    
    #[serde(default = "default_multipart_threshold")]
    pub multipart_threshold: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServiceUrls {
    pub auth_service: Url,
    pub payment_service: Url,
    pub notification_service: Url,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct FeatureFlags {
    #[serde(default)]
    pub enable_new_payment_flow: bool,
    
    #[serde(default)]
    pub enable_async_processing: bool,
    
    #[serde(default)]
    pub enable_rate_limiting: bool,
}

// Default value functions
fn default_environment() -> Environment { Environment::Local }
fn default_host() -> String { "0.0.0.0".to_string() }
fn default_port() -> u16 { 3000 }
fn default_workers() -> usize { num_cpus::get() }
fn default_shutdown_timeout() -> Duration { Duration::from_secs(30) }
fn default_body_limit() -> usize { 10 * 1024 * 1024 } // 10MB
fn default_max_connections() -> u32 { 100 }
fn default_min_connections() -> u32 { 10 }
fn default_connection_timeout() -> Duration { Duration::from_secs(30) }
fn default_idle_timeout() -> Duration { Duration::from_secs(600) }
fn default_cache_pool_size() -> u32 { 50 }
fn default_cache_ttl() -> Duration { Duration::from_secs(3600) }
fn default_upload_concurrency() -> usize { 100 }
fn default_multipart_threshold() -> usize { 5 * 1024 * 1024 } // 5MB

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok(); // Load .env file if present
        
        let config = envy::from_env::<Config>()
            .context("Failed to load configuration from environment")?;
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// Validate configuration values
    fn validate(&self) -> Result<()> {
        // Validate port range
        if self.server.port == 0 {
            eyre::bail!("Server port cannot be 0");
        }
        
        // Validate database connections
        if self.database.max_connections < self.database.min_connections {
            eyre::bail!("Max connections must be >= min connections");
        }
        
        // Validate environment-specific rules
        if self.environment == Environment::Production {
            if self.server.jwt_secret.len() < 32 {
                eyre::bail!("JWT secret must be at least 32 characters in production");
            }
            
            if self.database.enable_logging {
                eyre::bail!("Database logging should be disabled in production");
            }
        }
        
        Ok(())
    }
    
    /// Check if running in production
    pub fn is_production(&self) -> bool {
        self.environment == Environment::Production
    }
    
    /// Check if running in development
    pub fn is_development(&self) -> bool {
        self.environment == Environment::Local
    }
}

/// Global configuration instance
pub static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    Config::from_env().expect("Failed to initialize configuration")
});

/// Force configuration initialization (useful for testing fail-fast behavior)
pub fn init() {
    LazyLock::force(&CONFIG);
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_config_from_env() {
        // Set required environment variables
        env::set_var("SERVER_JWT_SECRET", "test_secret_key_for_testing_only");
        env::set_var("DATABASE_URL", "postgresql://localhost/test");
        env::set_var("CACHE_REDIS_URL", "redis://localhost");
        env::set_var("STORAGE_BUCKET", "test-bucket");
        env::set_var("STORAGE_REGION", "us-east-1");
        env::set_var("SERVICES_AUTH_SERVICE", "https://auth.example.com");
        env::set_var("SERVICES_PAYMENT_SERVICE", "https://payment.example.com");
        env::set_var("SERVICES_NOTIFICATION_SERVICE", "https://notify.example.com");
        
        let config = Config::from_env().unwrap();
        
        assert_eq!(config.environment, Environment::Local);
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.database.max_connections, 100);
    }
    
    #[test]
    fn test_validation_fails_for_invalid_config() {
        env::set_var("ENVIRONMENT", "production");
        env::set_var("SERVER_JWT_SECRET", "short"); // Too short for production
        
        let result = Config::from_env();
        assert!(result.is_err());
    }
}
```

### Error Handling with Context

Error handling in Rust should provide rich context for debugging production issues:

```rust
// src/error.rs
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::fmt;
use tracing::error;

/// Application error type that provides rich context
#[derive(Debug)]
pub struct AppError {
    /// The underlying error
    source: eyre::Report,
    
    /// HTTP status code to return
    status: StatusCode,
    
    /// User-facing error message
    message: String,
    
    /// Error code for client handling
    code: ErrorCode,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    // Authentication errors
    Unauthorized,
    InvalidCredentials,
    TokenExpired,
    
    // Validation errors
    ValidationFailed,
    InvalidInput,
    MissingField,
    
    // Resource errors
    NotFound,
    AlreadyExists,
    Conflict,
    
    // System errors
    InternalError,
    DatabaseError,
    ExternalServiceError,
    
    // Rate limiting
    TooManyRequests,
}

impl AppError {
    /// Create a new application error
    pub fn new(source: eyre::Report, status: StatusCode, code: ErrorCode) -> Self {
        let message = match code {
            ErrorCode::Unauthorized => "Authentication required",
            ErrorCode::InvalidCredentials => "Invalid username or password",
            ErrorCode::TokenExpired => "Authentication token has expired",
            ErrorCode::ValidationFailed => "Request validation failed",
            ErrorCode::InvalidInput => "Invalid input provided",
            ErrorCode::MissingField => "Required field missing",
            ErrorCode::NotFound => "Resource not found",
            ErrorCode::AlreadyExists => "Resource already exists",
            ErrorCode::Conflict => "Resource conflict",
            ErrorCode::InternalError => "Internal server error",
            ErrorCode::DatabaseError => "Database operation failed",
            ErrorCode::ExternalServiceError => "External service unavailable",
            ErrorCode::TooManyRequests => "Too many requests",
        }.to_string();
        
        Self {
            source,
            status,
            message,
            code,
        }
    }
    
    /// Create an unauthorized error
    pub fn unauthorized() -> Self {
        Self::new(
            eyre::eyre!("Unauthorized access attempt"),
            StatusCode::UNAUTHORIZED,
            ErrorCode::Unauthorized,
        )
    }
    
    /// Create a not found error
    pub fn not_found(resource: &str) -> Self {
        Self::new(
            eyre::eyre!("Resource not found: {}", resource),
            StatusCode::NOT_FOUND,
            ErrorCode::NotFound,
        )
    }
    
    /// Create a validation error
    pub fn validation<E: Into<eyre::Report>>(err: E) -> Self {
        Self::new(
            err.into(),
            StatusCode::BAD_REQUEST,
            ErrorCode::ValidationFailed,
        )
    }
    
    /// Create an internal error
    pub fn internal<E: Into<eyre::Report>>(err: E) -> Self {
        Self::new(
            err.into(),
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorCode::InternalError,
        )
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: ErrorCode,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<serde_json::Value>,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Log the full error chain for debugging
        error!(
            error = ?self.source,
            status = ?self.status,
            code = ?self.code,
            "Request failed"
        );
        
        let body = ErrorResponse {
            error: ErrorBody {
                code: self.code,
                message: self.message,
                details: None,
            },
        };
        
        (self.status, Json(body)).into_response()
    }
}

/// Extension trait for converting Results to AppError
pub trait AppErrorExt<T> {
    fn app_context(self, msg: &str) -> Result<T, AppError>;
    fn with_status(self, status: StatusCode) -> Result<T, AppError>;
}

impl<T, E> AppErrorExt<T> for Result<T, E>
where
    E: Into<eyre::Report>,
{
    fn app_context(self, msg: &str) -> Result<T, AppError> {
        self.map_err(|e| {
            AppError::internal(
                eyre::Report::from(e.into()).wrap_err(msg)
            )
        })
    }
    
    fn with_status(self, status: StatusCode) -> Result<T, AppError> {
        self.map_err(|e| {
            let code = match status {
                StatusCode::NOT_FOUND => ErrorCode::NotFound,
                StatusCode::BAD_REQUEST => ErrorCode::ValidationFailed,
                StatusCode::UNAUTHORIZED => ErrorCode::Unauthorized,
                StatusCode::TOO_MANY_REQUESTS => ErrorCode::TooManyRequests,
                _ => ErrorCode::InternalError,
            };
            
            AppError::new(e.into(), status, code)
        })
    }
}

// Usage in handlers
use axum::extract::{Path, Json as JsonExtract};
use uuid::Uuid;

async fn get_user(
    Path(user_id): Path<Uuid>,
) -> Result<JsonExtract<User>, AppError> {
    let user = fetch_user_from_db(user_id)
        .await
        .app_context("Failed to fetch user from database")?
        .ok_or_else(|| AppError::not_found("user"))?;
    
    Ok(JsonExtract(user))
}

async fn create_user(
    JsonExtract(input): JsonExtract<CreateUserInput>,
) -> Result<JsonExtract<User>, AppError> {
    // Validate input
    input.validate()
        .map_err(AppError::validation)?;
    
    // Check if user exists
    if user_exists(&input.email).await.app_context("Failed to check user existence")? {
        return Err(AppError::new(
            eyre::eyre!("User with email {} already exists", input.email),
            StatusCode::CONFLICT,
            ErrorCode::AlreadyExists,
        ));
    }
    
    // Create user
    let user = insert_user(input)
        .await
        .app_context("Failed to create user")?;
    
    Ok(JsonExtract(user))
}
```

### Telemetry and Observability

Production services need comprehensive observability. This includes structured logging, distributed tracing, and metrics:

```rust
// src/telemetry.rs
use opentelemetry::{global, KeyValue};
use opentelemetry::trace::{Tracer, TracerProvider};
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    trace::{self, RandomIdGenerator, Sampler},
    Resource,
};
use opentelemetry_otlp::WithExportConfig;
use tracing::{subscriber::set_global_default, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    EnvFilter,
    Registry,
};
use crate::config::CONFIG;

/// Initialize telemetry with OpenTelemetry and tracing
pub fn init_telemetry() -> eyre::Result<()> {
    // Set up OpenTelemetry
    global::set_text_map_propagator(TraceContextPropagator::new());
    
    let tracer = init_tracer()?;
    
    // Set up tracing subscriber
    let subscriber = get_subscriber(tracer);
    set_global_default(subscriber)?;
    
    Ok(())
}

fn init_tracer() -> eyre::Result<impl Tracer> {
    let resource = Resource::new(vec![
        KeyValue::new("service.name", env!("CARGO_PKG_NAME")),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        KeyValue::new("environment", format!("{:?}", CONFIG.environment)),
    ]);
    
    let sampler = if CONFIG.is_production() {
        Sampler::TraceIdRatioBased(0.1) // Sample 10% in production
    } else {
        Sampler::AlwaysOn // Sample everything in development
    };
    
    let trace_config = trace::config()
        .with_sampler(sampler)
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource);
    
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint("http://localhost:4317");
    
    let tracer_provider = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(trace_config)
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;
    
    Ok(tracer_provider.tracer("api"))
}

fn get_subscriber<T>(tracer: T) -> impl Subscriber + Send + Sync
where
    T: Tracer + Send + Sync + 'static,
{
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            if CONFIG.is_production() {
                "info,tower_http=debug".into()
            } else {
                "debug,tower_http=debug,sqlx=debug".into()
            }
        });
    
    let formatting_layer = if CONFIG.is_production() {
        // JSON formatting for production
        fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .with_target(true)
            .with_file(false)
            .with_line_number(false)
            .boxed()
    } else {
        // Pretty formatting for development
        fmt::layer()
            .pretty()
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .with_thread_ids(true)
            .boxed()
    };
    
    Registry::default()
        .with(env_filter)
        .with(formatting_layer)
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_error::ErrorLayer::default())
}

/// Shutdown telemetry providers
pub fn shutdown_telemetry() {
    global::shutdown_tracer_provider();
}

// Instrumentation macros for common operations
#[macro_export]
macro_rules! instrument_db {
    ($expr:expr) => {{
        let span = tracing::info_span!(
            "db.query",
            db.system = "postgresql",
            db.operation = stringify!($expr)
        );
        async move { $expr }
            .instrument(span)
            .await
    }};
}

#[macro_export]
macro_rules! instrument_http {
    ($client:expr, $method:expr, $url:expr) => {{
        let span = tracing::info_span!(
            "http.request",
            http.method = $method,
            http.url = $url,
            http.status_code = tracing::field::Empty,
        );
        
        async move {
            let result = $client
                .request($method, $url)
                .send()
                .await;
            
            if let Ok(ref response) = result {
                span.record("http.status_code", response.status().as_u16());
            }
            
            result
        }
        .instrument(span)
        .await
    }};
}
```

### HTTP API Layer with Axum

Building production APIs with Axum requires careful attention to routing, middleware, and documentation:

```rust
// src/api/router.rs
use axum::{
    Router,
    middleware,
    routing::{get, post},
    extract::DefaultBodyLimit,
};
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use std::time::Duration;
use crate::config::CONFIG;

/// Build the main application router
pub fn create_router(state: AppState) -> Router {
    let api_routes = Router::new()
        .nest("/v1", v1::router())
        .nest("/v2", v2::router())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));
    
    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .nest("/api", api_routes)
        .split_for_parts();
    
    router
        // Health check endpoints
        .route("/_health", get(health_check))
        .route("/_ready", get(readiness_check))
        
        // API documentation
        .merge(SwaggerUi::new("/docs").url("/openapi.json", api))
        
        // Global middleware stack
        .layer(
            ServiceBuilder::new()
                // Set request timeout
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                
                // Compress responses
                .layer(CompressionLayer::new())
                
                // Add CORS headers
                .layer(build_cors_layer())
                
                // Limit request body size
                .layer(RequestBodyLimitLayer::new(CONFIG.server.max_body_size))
                
                // Add tracing
                .layer(TraceLayer::new_for_http())
                
                // Rate limiting
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    rate_limit_middleware,
                ))
        )
        .with_state(state)
}

fn build_cors_layer() -> CorsLayer {
    if CONFIG.is_production() {
        CorsLayer::new()
            .allow_origin(["https://app.example.com".parse().unwrap()])
            .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
            .allow_headers([CONTENT_TYPE, AUTHORIZATION])
            .max_age(Duration::from_secs(3600))
    } else {
        CorsLayer::permissive()
    }
}

// src/api/middleware.rs
use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use axum_extra::headers::{Authorization, authorization::Bearer};
use jsonwebtoken::{decode, DecodingKey, Validation};

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    mut request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let token = auth.token();
    
    let claims = decode::<Claims>(
        token,
        &DecodingKey::from_secret(CONFIG.server.jwt_secret.as_bytes()),
        &Validation::default(),
    )
    .map_err(|_| AppError::unauthorized())?
    .claims;
    
    // Add user context to request extensions
    request.extensions_mut().insert(AuthContext {
        user_id: claims.sub,
        roles: claims.roles,
    });
    
    Ok(next.run(request).await)
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if !CONFIG.features.enable_rate_limiting {
        return Ok(next.run(request).await);
    }
    
    let key = extract_rate_limit_key(&request);
    
    let mut rate_limiter = state.rate_limiter.lock().await;
    
    if !rate_limiter.check_key(&key) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    
    Ok(next.run(request).await)
}

// src/api/v2/users.rs
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};
use uuid::Uuid;
use validator::Validate;

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateUserRequest {
    #[validate(email)]
    pub email: String,
    
    #[validate(length(min = 8, max = 128))]
    pub password: String,
    
    #[validate(length(min = 1, max = 100))]
    pub name: String,
    
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct UserResponse {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize, IntoParams)]
pub struct ListUsersQuery {
    /// Page number (1-indexed)
    #[param(minimum = 1)]
    pub page: Option<u32>,
    
    /// Items per page
    #[param(minimum = 1, maximum = 100)]
    pub limit: Option<u32>,
    
    /// Search query
    pub search: Option<String>,
    
    /// Sort field
    pub sort_by: Option<UserSortField>,
    
    /// Sort direction
    pub sort_dir: Option<SortDirection>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum UserSortField {
    Name,
    Email,
    CreatedAt,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Create a new user
#[utoipa::path(
    post,
    path = "/api/v2/users",
    tag = "users",
    request_body = CreateUserRequest,
    responses(
        (status = 201, description = "User created", body = UserResponse),
        (status = 400, description = "Invalid input", body = ErrorResponse),
        (status = 409, description = "User already exists", body = ErrorResponse),
    )
)]
#[tracing::instrument(skip(state, input))]
pub async fn create_user(
    State(state): State<AppState>,
    Json(input): Json<CreateUserRequest>,
) -> Result<(StatusCode, Json<UserResponse>), AppError> {
    // Validate input
    input.validate()
        .map_err(|e| AppError::validation(e))?;
    
    // Check if user exists
    let existing = sqlx::query!(
        "SELECT id FROM users WHERE email = $1",
        input.email
    )
    .fetch_optional(&state.db)
    .await
    .app_context("Failed to check existing user")?;
    
    if existing.is_some() {
        return Err(AppError::new(
            eyre::eyre!("User already exists"),
            StatusCode::CONFLICT,
            ErrorCode::AlreadyExists,
        ));
    }
    
    // Hash password
    let password_hash = argon2::hash_encoded(
        input.password.as_bytes(),
        &state.salt,
        &argon2::Config::default(),
    )
    .map_err(|e| AppError::internal(e))?;
    
    // Insert user
    let user = sqlx::query_as!(
        UserResponse,
        r#"
        INSERT INTO users (email, password_hash, name, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING id, email, name, created_at, updated_at
        "#,
        input.email,
        password_hash,
        input.name,
        input.metadata
    )
    .fetch_one(&state.db)
    .await
    .app_context("Failed to create user")?;
    
    Ok((StatusCode::CREATED, Json(user)))
}

/// Get a user by ID
#[utoipa::path(
    get,
    path = "/api/v2/users/{user_id}",
    tag = "users",
    params(
        ("user_id" = Uuid, Path, description = "User ID")
    ),
    responses(
        (status = 200, description = "User found", body = UserResponse),
        (status = 404, description = "User not found", body = ErrorResponse),
    )
)]
#[tracing::instrument(skip(state))]
pub async fn get_user(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> Result<Json<UserResponse>, AppError> {
    let user = sqlx::query_as!(
        UserResponse,
        r#"
        SELECT id, email, name, created_at, updated_at
        FROM users
        WHERE id = $1
        "#,
        user_id
    )
    .fetch_optional(&state.db)
    .await
    .app_context("Failed to fetch user")?
    .ok_or_else(|| AppError::not_found("user"))?;
    
    Ok(Json(user))
}

/// List users with pagination
#[utoipa::path(
    get,
    path = "/api/v2/users",
    tag = "users",
    params(ListUsersQuery),
    responses(
        (status = 200, description = "Users list", body = Vec<UserResponse>),
    )
)]
#[tracing::instrument(skip(state))]
pub async fn list_users(
    State(state): State<AppState>,
    Query(query): Query<ListUsersQuery>,
) -> Result<Json<Vec<UserResponse>>, AppError> {
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(20);
    let offset = (page - 1) * limit;
    
    let mut sql = QueryBuilder::new(
        "SELECT id, email, name, created_at, updated_at FROM users"
    );
    
    // Add search condition
    if let Some(search) = query.search {
        sql.push(" WHERE (email ILIKE ");
        sql.push_bind(format!("%{}%", search));
        sql.push(" OR name ILIKE ");
        sql.push_bind(format!("%{}%", search));
        sql.push(")");
    }
    
    // Add sorting
    let sort_field = query.sort_by.unwrap_or(UserSortField::CreatedAt);
    let sort_dir = query.sort_dir.unwrap_or(SortDirection::Desc);
    
    sql.push(" ORDER BY ");
    sql.push(match sort_field {
        UserSortField::Name => "name",
        UserSortField::Email => "email",
        UserSortField::CreatedAt => "created_at",
    });
    sql.push(match sort_dir {
        SortDirection::Asc => " ASC",
        SortDirection::Desc => " DESC",
    });
    
    // Add pagination
    sql.push(" LIMIT ");
    sql.push_bind(limit as i64);
    sql.push(" OFFSET ");
    sql.push_bind(offset as i64);
    
    let users = sql
        .build_query_as::<UserResponse>()
        .fetch_all(&state.db)
        .await
        .app_context("Failed to list users")?;
    
    Ok(Json(users))
}
```

### Asynchronous Patterns and Performance

Efficient async programming in Rust requires understanding when to use different concurrency patterns:

```rust
// src/infrastructure/object_storage.rs
use bytes::Bytes;
use futures_util::{stream, StreamExt, TryStreamExt};
use object_store::{
    aws::{AmazonS3, AmazonS3Builder},
    path::Path,
    ObjectStore,
    PutOptions,
    PutPayload,
};
use std::sync::LazyLock;
use tokio::sync::Semaphore;
use uuid::Uuid;

static S3: LazyLock<AmazonS3> = LazyLock::new(|| {
    AmazonS3Builder::from_env()
        .with_bucket_name(&CONFIG.storage.bucket)
        .with_region(&CONFIG.storage.region)
        .with_endpoint(CONFIG.storage.endpoint.as_ref().map(|u| u.to_string()))
        .build()
        .expect("Failed to build S3 client")
});

/// Service for managing object storage operations
pub struct StorageService {
    client: &'static AmazonS3,
    upload_semaphore: Semaphore,
}

impl StorageService {
    pub fn new() -> Self {
        Self {
            client: &*S3,
            upload_semaphore: Semaphore::new(CONFIG.storage.upload_concurrency),
        }
    }
    
    /// Upload a single file
    #[tracing::instrument(skip(self, data))]
    pub async fn upload_file(
        &self,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> eyre::Result<()> {
        let path = Path::from(key);
        
        let mut options = PutOptions::default();
        if let Some(ct) = content_type {
            options.attributes.insert(
                object_store::Attribute::ContentType,
                ct.into(),
            );
        }
        
        self.client
            .put_opts(&path, PutPayload::from(data), options)
            .await
            .wrap_err("Failed to upload file")?;
        
        tracing::info!(key, "File uploaded successfully");
        Ok(())
    }
    
    /// Upload multiple files concurrently with rate limiting
    #[tracing::instrument(skip(self, files))]
    pub async fn upload_batch(
        &self,
        files: Vec<(String, Bytes, Option<String>)>,
    ) -> eyre::Result<Vec<Result<String, String>>> {
        let total = files.len();
        
        let uploads = stream::iter(files)
            .map(|(key, data, content_type)| async move {
                // Acquire semaphore permit to limit concurrency
                let _permit = self.upload_semaphore.acquire().await
                    .map_err(|e| format!("Failed to acquire permit: {}", e))?;
                
                self.upload_file(&key, data, content_type.as_deref())
                    .await
                    .map(|_| key.clone())
                    .map_err(|e| format!("Failed to upload {}: {}", key, e))
            })
            .buffer_unordered(CONFIG.storage.upload_concurrency)
            .collect::<Vec<_>>()
            .await;
        
        let successful = uploads.iter().filter(|r| r.is_ok()).count();
        tracing::info!(
            successful,
            failed = total - successful,
            total,
            "Batch upload completed"
        );
        
        Ok(uploads)
    }
    
    /// Stream download for large files
    #[tracing::instrument(skip(self))]
    pub async fn download_stream(
        &self,
        key: &str,
    ) -> eyre::Result<impl Stream<Item = eyre::Result<Bytes>>> {
        let path = Path::from(key);
        
        let stream = self.client
            .get(&path)
            .await
            .wrap_err("Failed to initiate download")?
            .into_stream()
            .map_err(|e| eyre::eyre!("Stream error: {}", e));
        
        Ok(stream)
    }
    
    /// Multipart upload for large files
    #[tracing::instrument(skip(self, data))]
    pub async fn upload_large_file(
        &self,
        key: &str,
        data: impl Stream<Item = eyre::Result<Bytes>> + Send,
    ) -> eyre::Result<()> {
        let path = Path::from(key);
        
        // Check if file size exceeds multipart threshold
        let chunks: Vec<Bytes> = data.try_collect().await?;
        let total_size: usize = chunks.iter().map(|c| c.len()).sum();
        
        if total_size < CONFIG.storage.multipart_threshold {
            // Single upload for small files
            let combined = chunks.into_iter().flatten().collect();
            return self.upload_file(key, combined, None).await;
        }
        
        // Multipart upload for large files
        let upload = self.client
            .put_multipart(&path)
            .await
            .wrap_err("Failed to initiate multipart upload")?;
        
        let mut part_number = 1;
        for chunk in chunks {
            upload
                .put_part(part_number, chunk.into())
                .await
                .wrap_err(format!("Failed to upload part {}", part_number))?;
            part_number += 1;
        }
        
        upload
            .complete()
            .await
            .wrap_err("Failed to complete multipart upload")?;
        
        tracing::info!(
            key,
            parts = part_number - 1,
            size = total_size,
            "Large file uploaded successfully"
        );
        
        Ok(())
    }
}

// src/domain/processing.rs
use tokio::task;
use std::sync::Arc;

/// CPU-intensive processing that shouldn't block the async runtime
pub struct ProcessingService {
    thread_pool: Arc<rayon::ThreadPool>,
}

impl ProcessingService {
    pub fn new(num_threads: Option<usize>) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(num_cpus::get))
            .build()
            .expect("Failed to build thread pool");
        
        Self {
            thread_pool: Arc::new(thread_pool),
        }
    }
    
    /// Process data using a dedicated thread pool
    #[tracing::instrument(skip(self, data, processor))]
    pub async fn process<T, F, R>(
        &self,
        data: T,
        processor: F,
    ) -> eyre::Result<R>
    where
        T: Send + 'static,
        F: FnOnce(T) -> eyre::Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let pool = self.thread_pool.clone();
        
        // Spawn blocking task on dedicated thread pool
        let result = task::spawn_blocking(move || {
            pool.install(|| processor(data))
        })
        .await
        .wrap_err("Processing task failed")?;
        
        result
    }
    
    /// Process multiple items in parallel
    #[tracing::instrument(skip(self, items, processor))]
    pub async fn process_batch<T, F, R>(
        &self,
        items: Vec<T>,
        processor: F,
    ) -> eyre::Result<Vec<R>>
    where
        T: Send + 'static,
        F: Fn(T) -> eyre::Result<R> + Send + Sync + 'static,
        R: Send + 'static,
    {
        let processor = Arc::new(processor);
        let pool = self.thread_pool.clone();
        
        let result = task::spawn_blocking(move || {
            pool.install(|| {
                items
                    .into_par_iter()
                    .map(|item| processor(item))
                    .collect::<Vec<_>>()
            })
        })
        .await
        .wrap_err("Batch processing failed")?;
        
        result.into_iter().collect()
    }
}

// Example: PDF processing with external process
use tokio::process::Command;
use std::process::Stdio;

#[tracing::instrument(skip(pdf_data))]
pub async fn process_pdf_with_external_tool(
    pdf_data: Bytes,
) -> eyre::Result<Bytes> {
    let mut child = Command::new("pdftotext")
        .args(["-", "-"])  // Read from stdin, write to stdout
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .wrap_err("Failed to spawn pdftotext process")?;
    
    // Write input to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&pdf_data).await
            .wrap_err("Failed to write to process stdin")?;
    }
    
    // Read output
    let output = child.wait_with_output().await
        .wrap_err("Failed to wait for process")?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eyre::bail!("Process failed: {}", stderr);
    }
    
    Ok(Bytes::from(output.stdout))
}
```

### Database Patterns with SQLx

Database operations need careful handling for performance and correctness:

```rust
// src/infrastructure/database.rs
use sqlx::{
    postgres::{PgPool, PgPoolOptions, PgQueryResult},
    types::Json as SqlxJson,
    FromRow, QueryBuilder,
};
use std::time::Duration;
use uuid::Uuid;

/// Initialize database connection pool
pub async fn init_database() -> eyre::Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(CONFIG.database.max_connections)
        .min_connections(CONFIG.database.min_connections)
        .acquire_timeout(CONFIG.database.connection_timeout)
        .idle_timeout(Some(CONFIG.database.idle_timeout))
        .test_before_acquire(true)
        .connect(&CONFIG.database.url)
        .await
        .wrap_err("Failed to connect to database")?;
    
    // Run migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .wrap_err("Failed to run migrations")?;
    
    Ok(pool)
}

// Repository pattern for domain entities
#[derive(Clone)]
pub struct UserRepository {
    pool: PgPool,
}

impl UserRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    /// Get user by ID with caching
    #[tracing::instrument(skip(self))]
    pub async fn get_by_id(&self, id: Uuid) -> eyre::Result<Option<User>> {
        // Check cache first
        if let Some(cached) = cache_get(&format!("user:{}", id)).await? {
            return Ok(Some(cached));
        }
        
        // Query database
        let user = sqlx::query_as!(
            User,
            r#"
            SELECT 
                id,
                email,
                name,
                metadata as "metadata: SqlxJson<UserMetadata>",
                created_at,
                updated_at
            FROM users
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .wrap_err("Failed to fetch user")?;
        
        // Update cache
        if let Some(ref user) = user {
            cache_set(&format!("user:{}", id), user, Duration::from_secs(300)).await?;
        }
        
        Ok(user)
    }
    
    /// Bulk insert with conflict handling
    #[tracing::instrument(skip(self, users))]
    pub async fn bulk_insert(
        &self,
        users: Vec<CreateUserInput>,
    ) -> eyre::Result<Vec<Uuid>> {
        if users.is_empty() {
            return Ok(vec![]);
        }
        
        let mut query = QueryBuilder::new(
            r#"
            INSERT INTO users (email, name, password_hash, metadata)
            "#
        );
        
        query.push_values(users, |mut b, user| {
            b.push_bind(user.email)
                .push_bind(user.name)
                .push_bind(user.password_hash)
                .push_bind(SqlxJson(user.metadata));
        });
        
        query.push(
            r#"
            ON CONFLICT (email) DO NOTHING
            RETURNING id
            "#
        );
        
        let ids: Vec<(Uuid,)> = query
            .build_query_as()
            .fetch_all(&self.pool)
            .await
            .wrap_err("Failed to bulk insert users")?;
        
        Ok(ids.into_iter().map(|(id,)| id).collect())
    }
    
    /// Transaction example with rollback
    #[tracing::instrument(skip(self))]
    pub async fn transfer_ownership(
        &self,
        resource_id: Uuid,
        from_user: Uuid,
        to_user: Uuid,
    ) -> eyre::Result<()> {
        let mut tx = self.pool.begin().await
            .wrap_err("Failed to begin transaction")?;
        
        // Verify current owner
        let current_owner = sqlx::query_scalar!(
            "SELECT owner_id FROM resources WHERE id = $1 FOR UPDATE",
            resource_id
        )
        .fetch_optional(&mut *tx)
        .await
        .wrap_err("Failed to fetch resource")?
        .ok_or_else(|| eyre::eyre!("Resource not found"))?;
        
        if current_owner != from_user {
            return Err(eyre::eyre!("User is not the current owner"));
        }
        
        // Update ownership
        sqlx::query!(
            "UPDATE resources SET owner_id = $1 WHERE id = $2",
            to_user,
            resource_id
        )
        .execute(&mut *tx)
        .await
        .wrap_err("Failed to update ownership")?;
        
        // Log the transfer
        sqlx::query!(
            r#"
            INSERT INTO ownership_logs (resource_id, from_user, to_user, transferred_at)
            VALUES ($1, $2, $3, NOW())
            "#,
            resource_id,
            from_user,
            to_user
        )
        .execute(&mut *tx)
        .await
        .wrap_err("Failed to log transfer")?;
        
        // Commit transaction
        tx.commit().await
            .wrap_err("Failed to commit transaction")?;
        
        Ok(())
    }
    
    /// Optimized pagination with cursor
    #[tracing::instrument(skip(self))]
    pub async fn list_paginated(
        &self,
        cursor: Option<(chrono::DateTime<chrono::Utc>, Uuid)>,
        limit: i64,
    ) -> eyre::Result<(Vec<User>, Option<(chrono::DateTime<chrono::Utc>, Uuid)>)> {
        let users = if let Some((timestamp, id)) = cursor {
            sqlx::query_as!(
                User,
                r#"
                SELECT id, email, name, metadata, created_at, updated_at
                FROM users
                WHERE (created_at, id) > ($1, $2)
                ORDER BY created_at, id
                LIMIT $3
                "#,
                timestamp,
                id,
                limit + 1  // Fetch one extra to determine if there's a next page
            )
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query_as!(
                User,
                r#"
                SELECT id, email, name, metadata, created_at, updated_at
                FROM users
                ORDER BY created_at, id
                LIMIT $1
                "#,
                limit + 1
            )
            .fetch_all(&self.pool)
            .await?
        };
        
        let has_next = users.len() > limit as usize;
        let users = if has_next {
            users[..limit as usize].to_vec()
        } else {
            users
        };
        
        let next_cursor = if has_next {
            users.last().map(|u| (u.created_at, u.id))
        } else {
            None
        };
        
        Ok((users, next_cursor))
    }
}
```

---

## 5. Integration Patterns and Best Practices

### Testing Strategy

Comprehensive testing requires different approaches for different layers:

```rust
// tests/integration/api_tests.rs
use axum::http::StatusCode;
use sqlx::PgPool;
use tower::ServiceExt;

#[sqlx::test]
async fn test_user_creation_flow(pool: PgPool) {
    let app = create_test_app(pool);
    
    // Create user
    let create_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v2/users")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&json!({
                        "email": "test@example.com",
                        "password": "secure_password_123",
                        "name": "Test User"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(create_response.status(), StatusCode::CREATED);
    
    let body = hyper::body::to_bytes(create_response.into_body())
        .await
        .unwrap();
    let user: UserResponse = serde_json::from_slice(&body).unwrap();
    
    // Get created user
    let get_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v2/users/{}", user.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(get_response.status(), StatusCode::OK);
}

// tests/unit/domain_tests.rs
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_password_validation() {
        let valid_password = "SecurePass123!";
        let weak_password = "weak";
        
        assert!(validate_password(valid_password).is_ok());
        assert!(validate_password(weak_password).is_err());
    }
    
    #[tokio::test]
    async fn test_concurrent_processing() {
        let service = ProcessingService::new(Some(4));
        
        let items = vec![1, 2, 3, 4, 5];
        let results = service
            .process_batch(items, |n| Ok(n * n))
            .await
            .unwrap();
        
        assert_eq!(results, vec![1, 4, 9, 16, 25]);
    }
}

// Benchmarking
#[bench]
fn bench_json_serialization(b: &mut Bencher) {
    let user = UserResponse {
        id: Uuid::new_v4(),
        email: "test@example.com".to_string(),
        name: "Test User".to_string(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    
    b.iter(|| {
        serde_json::to_string(&user).unwrap()
    });
}
```

### Deployment Patterns

Production deployment requires careful consideration of health checks, graceful shutdown, and resource management:

```rust
// src/main.rs
use std::net::SocketAddr;
use tokio::signal;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize everything
    color_eyre::install()?;
    dotenvy::dotenv().ok();
    config::init();
    telemetry::init_telemetry()?;
    
    let db = database::init_database().await?;
    
    let state = AppState {
        db: db.clone(),
        storage: StorageService::new(),
        processing: ProcessingService::new(None),
        rate_limiter: Arc::new(Mutex::new(RateLimiter::new())),
    };
    
    let app = api::create_router(state);
    
    let addr = SocketAddr::from(([0, 0, 0, 0], CONFIG.server.port));
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    
    tracing::info!("Server listening on {}", addr);
    
    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    
    // Cleanup
    tracing::info!("Shutting down gracefully");
    telemetry::shutdown_telemetry();
    db.close().await;
    
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };
    
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

// Health check implementation
async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": Utc::now().to_rfc3339(),
    }))
}

async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    // Check database connectivity
    let db_healthy = sqlx::query("SELECT 1")
        .fetch_one(&state.db)
        .await
        .is_ok();
    
    // Check S3 connectivity
    let storage_healthy = state.storage
        .client
        .list(Some(&Path::from("health-check")))
        .await
        .is_ok();
    
    if db_healthy && storage_healthy {
        (StatusCode::OK, Json(json!({
            "status": "ready",
            "database": "connected",
            "storage": "connected",
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
            "status": "not_ready",
            "database": if db_healthy { "connected" } else { "disconnected" },
            "storage": if storage_healthy { "connected" } else { "disconnected" },
        })))
    }
}
```

### Performance Monitoring

Track key metrics to ensure system health:

```rust
// src/telemetry/metrics.rs
use prometheus::{
    register_histogram_vec, register_int_counter_vec,
    HistogramVec, IntCounterVec,
};
use std::sync::LazyLock;

static REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        &["method", "endpoint", "status"]
    )
    .unwrap()
});

static REQUEST_COUNT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "http_requests_total",
        "Total number of HTTP requests",
        &["method", "endpoint", "status"]
    )
    .unwrap()
});

static DB_QUERY_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "db_query_duration_seconds",
        "Database query duration in seconds",
        &["query_type", "table"]
    )
    .unwrap()
});

pub fn track_request(method: &str, endpoint: &str, status: u16, duration: f64) {
    REQUEST_DURATION
        .with_label_values(&[method, endpoint, &status.to_string()])
        .observe(duration);
    
    REQUEST_COUNT
        .with_label_values(&[method, endpoint, &status.to_string()])
        .inc();
}

pub fn track_db_query(query_type: &str, table: &str, duration: f64) {
    DB_QUERY_DURATION
        .with_label_values(&[query_type, table])
        .observe(duration);
}
```

## Conclusion

This guide represents years of production Rust experience distilled into actionable patterns. The key to building robust systems is not just knowing these patterns, but understanding when and why to apply them.

Remember: Perfect is the enemy of good. Start with correctness, optimize when measured, and always maintain code that your future self will thank you for writing.

The three pillars of production Rust are:
1. **Clear error handling** - Every error tells a story
2. **Type-driven design** - Make invalid states unrepresentable
3. **Observable behavior** - You can't fix what you can't see

Apply these principles consistently, and you'll build systems that are not just functional, but truly production-ready.
Please use [mlx-rs](https://github.com/oxideai/mlx-rs) for mlx bindings for Rust. you should read and consult with /Users/sigridjineth/RustroverProjects/mlx-retrieval-rs/MLX-RS.md.