#!/usr/bin/env python
"""
ME2AI MCP Test Runner

Executes all unit, integration, and performance tests for the ME2AI MCP package
and generates comprehensive test reports.

Usage:
    python run_mcp_tests.py [options]

Options:
    --unit-only           Run only unit tests
    --integration-only    Run only integration tests
    --performance-only    Run only performance tests
    --skip-performance    Skip performance tests
    --report-dir=<dir>    Directory for test reports (default: reports)
    --html                Generate HTML report
    --xml                 Generate XML report
    --json                Generate JSON report
    --cov                 Generate coverage report
    --debug               Show debug information
"""

import os
import sys
import argparse
import subprocess
import datetime
import shutil
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ME2AI MCP Test Runner")
    
    # Test selection options
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    
    # Report options
    parser.add_argument("--report-dir", default="reports", help="Directory for test reports")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--xml", action="store_true", help="Generate XML report")
    parser.add_argument("--json", action="store_true", help="Generate JSON report")
    parser.add_argument("--cov", action="store_true", help="Generate coverage report")
    
    # Other options
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    return parser.parse_args()


def run_tests(options):
    """Run the test suite with the specified options."""
    # Prepare report directory
    report_dir = Path(options.report_dir)
    report_dir.mkdir(exist_ok=True, parents=True)
    
    # Basic pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    cmd.append("-v")
    
    # Add test selection based on options
    if options.unit_only:
        cmd.append("-m unit")
    elif options.integration_only:
        cmd.append("-m integration")
    elif options.performance_only:
        cmd.append("-m performance")
    elif options.skip_performance:
        cmd.append("-m 'not performance'")
    
    # Add report options
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if options.html:
        html_report = report_dir / f"html_report_{timestamp}.html"
        cmd.extend(["--html", str(html_report), "--self-contained-html"])
    
    if options.xml:
        xml_report = report_dir / f"xml_report_{timestamp}.xml"
        cmd.extend(["--junitxml", str(xml_report)])
    
    if options.json:
        json_report = report_dir / f"json_report_{timestamp}.json"
        cmd.extend(["--json", str(json_report)])
    
    if options.cov:
        cmd.extend([
            "--cov=me2ai_mcp",
            "--cov-report", f"html:{report_dir}/coverage_{timestamp}",
            "--cov-report", "term",
            "--cov-report", f"xml:{report_dir}/coverage_{timestamp}.xml"
        ])
    
    # Target the tests directory
    cmd.append("tests/me2ai_mcp")
    
    # Show command if debug is enabled
    if options.debug:
        print(f"Running command: {' '.join(cmd)}")
    
    # Run the tests
    result = subprocess.run(cmd)
    
    return result.returncode


def generate_summary(options):
    """Generate a summary of the test results."""
    report_dir = Path(options.report_dir)
    
    # Find the most recent reports
    html_reports = list(report_dir.glob("html_report_*.html"))
    xml_reports = list(report_dir.glob("xml_report_*.xml"))
    json_reports = list(report_dir.glob("json_report_*.json"))
    coverage_reports = list(report_dir.glob("coverage_*"))
    
    latest_html = max(html_reports, key=lambda p: p.stat().st_mtime) if html_reports else None
    latest_xml = max(xml_reports, key=lambda p: p.stat().st_mtime) if xml_reports else None
    latest_json = max(json_reports, key=lambda p: p.stat().st_mtime) if json_reports else None
    latest_coverage = max(coverage_reports, key=lambda p: p.stat().st_mtime) if coverage_reports else None
    
    # Create summary file
    summary_path = report_dir / "summary.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(summary_path, "w") as summary:
        summary.write(f"ME2AI MCP Test Summary\n")
        summary.write(f"=====================\n\n")
        summary.write(f"Generated: {timestamp}\n\n")
        
        summary.write(f"Test Reports:\n")
        if latest_html:
            summary.write(f"- HTML Report: {latest_html.name}\n")
        if latest_xml:
            summary.write(f"- XML Report: {latest_xml.name}\n")
        if latest_json:
            summary.write(f"- JSON Report: {latest_json.name}\n")
        if latest_coverage:
            summary.write(f"- Coverage Report: {latest_coverage.name}\n")
            
        summary.write(f"\nOptions:\n")
        for option, value in vars(options).items():
            summary.write(f"- {option}: {value}\n")
    
    print(f"\nSummary written to {summary_path}")
    print(f"\nTest reports can be found in {report_dir.absolute()}")


def install_test_dependencies():
    """Install dependencies required for testing."""
    dependencies = [
        "pytest",
        "pytest-asyncio",
        "pytest-html",
        "pytest-cov",
        "pytest-json",
        "psutil",  # For memory usage tracking in performance tests
    ]
    
    print("Installing test dependencies...")
    cmd = [sys.executable, "-m", "pip", "install"] + dependencies
    subprocess.run(cmd)
    print("Dependencies installed.")


def main():
    """Main function."""
    options = parse_args()
    
    # Create reports directory
    reports_dir = Path(options.report_dir)
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    # Install dependencies if needed
    if options.debug:
        install_test_dependencies()
    
    # Run tests
    print(f"\nRunning ME2AI MCP tests...")
    returncode = run_tests(options)
    
    # Generate summary
    generate_summary(options)
    
    # Exit with the same code as pytest
    sys.exit(returncode)


if __name__ == "__main__":
    main()
