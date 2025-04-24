#!/usr/bin/env python
"""
Vector Store Test Suite Runner

This script runs the complete test suite for the ME2AI MCP Vector Store service
and generates detailed coverage reports. Used for both CI and developer testing.
"""

import os
import sys
import subprocess
import datetime
import argparse


def main():
    """Run the Vector Store test suite with coverage reporting."""
    parser = argparse.ArgumentParser(description="Run Vector Store tests with coverage")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report for CI")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Determine which tests to run
    test_paths = ["tests/services/test_backend_operations.py", "tests/services/test_vectorstore_service.py"]
    if args.integration or args.all:
        test_paths.append("tests/services/test_vectorstore_integration.py")
    
    # Set up coverage parameters
    coverage_modules = [
        "me2ai_mcp.services.vectorstore_service",
        "me2ai_mcp.services.backend_operations"
    ]
    
    cov_params = []
    for module in coverage_modules:
        cov_params.append(f"--cov={module}")
    
    # Add coverage report formats
    cov_params.append("--cov-report=term-missing")
    if args.html:
        cov_params.append("--cov-report=html:coverage_reports/html")
    if args.xml:
        cov_params.append("--cov-report=xml:coverage_reports/coverage.xml")
    
    # Create reports directory if needed
    if args.html or args.xml:
        os.makedirs("coverage_reports/html", exist_ok=True)
    
    # Build the command
    cmd = [
        "python", "-m", "pytest"
    ] + test_paths + cov_params
    
    # Print information
    print(f"Running Vector Store tests at {datetime.datetime.now()}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Tests completed successfully with exit code {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
