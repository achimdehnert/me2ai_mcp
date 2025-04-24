#!/usr/bin/env python3
"""
ME2AI MCP Test Runner

This script runs the test suite for the ME2AI MCP package.
"""

import os
import sys
import argparse
import pytest


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="ME2AI MCP Test Runner")
    
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Specific module to test (e.g., 'db', 'integrations')"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the test suite."""
    args = parse_arguments()
    
    # Prepare pytest arguments
    pytest_args = []
    
    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
    
    # Add specific module if provided
    if args.module:
        pytest_args.append(f"tests/{args.module}")
    else:
        pytest_args.append("tests")
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(["--cov=me2ai_mcp", "--cov-report=term"])
    
    # Add HTML report if requested
    if args.html_report:
        pytest_args.append("--html=test-report.html")
    
    # Run tests
    result = pytest.main(pytest_args)
    
    # Exit with pytest status
    sys.exit(result)


if __name__ == "__main__":
    main()
