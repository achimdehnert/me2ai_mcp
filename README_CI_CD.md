# ME2AI MCP CI/CD and Testing Guide

This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for the ME2AI MCP package.

## GitHub Actions Workflow

The ME2AI MCP package uses GitHub Actions for automated testing and code quality validation. The workflow is defined in `.github/workflows/python-tests.yml` and includes:

- Testing across multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Code formatting checks with Black and isort
- Linting with flake8
- Type checking with mypy
- Test coverage analysis with pytest-cov

## Running Tests Locally

You can run the same tests locally:

```bash
# Install test dependencies
pip install pytest pytest-cov flake8 black isort mypy

# Run unit tests
python -m pytest tests/unit/

# Run tests with coverage
python -m pytest --cov=me2ai_mcp tests/

# Run code formatting checks
black --check --line-length=100 me2ai_mcp tests examples
isort --check-only --profile black me2ai_mcp tests examples

# Run linting
flake8 me2ai_mcp tests examples --count --max-complexity=10 --max-line-length=100 --statistics

# Run type checking
mypy me2ai_mcp
```

## Test Structure

The testing framework follows ME2AI's standards with:

- **Unit Tests**: Located in `tests/unit/` - testing individual components in isolation
- **Integration Tests**: Located in `tests/integration/` - testing interactions between components
- **Performance Tests**: Located in `tests/performance/` - testing under load conditions

## Coverage Requirements

The ME2AI MCP package maintains a minimum 80% test coverage requirement, enforced by the CI/CD pipeline. Key components that must be thoroughly tested include:

- Authentication system
- Server base classes
- Utility functions
- Tool registration and execution

## Continuous Deployment

When all tests pass, the package can be published to PyPI with:

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload --username __token__ --password <PYPI_TOKEN> dist/*
```
