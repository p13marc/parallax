# Parallax - Justfile
# Run `just --list` to see available recipes

# Default recipe: run tests
default: test

# Build the project
build:
    cargo build

# Build in release mode
build-release:
    cargo build --release

# Run all tests with nextest
test:
    cargo nextest run

# Run tests with verbose output
test-verbose:
    cargo nextest run --no-capture

# Run a specific test
test-one NAME:
    cargo nextest run {{NAME}}

# Run tests matching a pattern
test-filter PATTERN:
    cargo nextest run -E 'test({{PATTERN}})'

# Run tests with coverage (requires cargo-llvm-cov)
coverage:
    cargo llvm-cov nextest

# Run clippy lints
lint:
    cargo clippy -- -D warnings

# Run clippy with all features
lint-all:
    cargo clippy --all-features -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting without modifying
fmt-check:
    cargo fmt --check

# Run all checks (format, lint, test)
check: fmt-check lint test

# Run benchmarks
bench:
    cargo bench

# Run a specific benchmark
bench-one NAME:
    cargo bench --bench {{NAME}}

# Clean build artifacts
clean:
    cargo clean

# Generate documentation
doc:
    cargo doc --no-deps

# Open documentation in browser
doc-open:
    cargo doc --no-deps --open

# Watch for changes and run tests
watch:
    cargo watch -x 'nextest run'

# Watch for changes and run clippy
watch-lint:
    cargo watch -x 'clippy -- -D warnings'

# Install development dependencies
dev-deps:
    cargo install cargo-nextest cargo-watch cargo-llvm-cov

# Show project statistics
stats:
    @echo "Lines of code:"
    @tokei src/
    @echo ""
    @echo "Dependencies:"
    @cargo tree --depth 1

# Run memory pool benchmark
bench-pool:
    cargo bench --bench memory_pool

# Run throughput benchmark
bench-throughput:
    cargo bench --bench throughput
