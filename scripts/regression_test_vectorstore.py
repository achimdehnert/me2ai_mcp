#!/usr/bin/env python
"""
Vector Store Regression Test Suite

This script runs comprehensive regression tests for the ME2AI MCP Vector Store
service across all supported backends. It's designed to be run on a scheduled
basis (e.g., nightly) to catch regressions.

Features:
1. Tests all backends: ChromaDB, FAISS, Qdrant, Pinecone
2. Tests all basic operations: collection management, upsert, query, delete
3. Tests integration with Knowledge Assistant
4. Generates HTML and XML test reports
5. Logs results and sends notifications on failure

Usage:
    python regression_test_vectorstore.py [--email] [--slack]
"""

import os
import sys
import datetime
import argparse
import subprocess
import smtplib
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("vectorstore_regression.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("vectorstore-regression")

# Constants
REPORT_DIR = Path("regression_reports")
TEST_MODULES = [
    "tests.services.test_backend_operations",
    "tests.services.test_vectorstore_service",
    "tests.services.test_vectorstore_integration",
    "tests.integration.test_knowledge_assistant_integration"
]


def setup_environment() -> bool:
    """Set up the test environment."""
    try:
        # Create reports directory
        os.makedirs(REPORT_DIR, exist_ok=True)
        os.makedirs(REPORT_DIR / "html", exist_ok=True)
        
        # Set up test environment variables
        os.environ["DATA_DIR"] = str(Path("test_data").absolute())
        os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
        
        # Test environment variables are properly set
        logger.info(f"Test data directory: {os.environ['DATA_DIR']}")
        return True
    except Exception as e:
        logger.error(f"Failed to set up environment: {e}")
        return False


def run_tests() -> Dict[str, Any]:
    """Run all regression tests and return results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "success": True,
        "modules": {}
    }
    
    # Run tests for each module
    for module in TEST_MODULES:
        module_name = module.split(".")[-1]
        logger.info(f"Running tests for {module_name}")
        
        # Build command
        cmd = [
            "python", "-m", "pytest", module,
            f"--html={REPORT_DIR}/html/{module_name}_{timestamp}.html",
            f"--junitxml={REPORT_DIR}/{module_name}_{timestamp}.xml",
            "--cov=me2ai_mcp.services.vectorstore_service",
            "--cov=me2ai_mcp.services.backend_operations",
            "--cov-report=html:coverage_reports/html",
            "--cov-report=xml:coverage_reports/coverage.xml"
        ]
        
        # Run the tests
        try:
            result = subprocess.run(
                cmd, 
                check=False,
                capture_output=True,
                text=True
            )
            
            module_result = {
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "report_html": f"{REPORT_DIR}/html/{module_name}_{timestamp}.html",
                "report_xml": f"{REPORT_DIR}/{module_name}_{timestamp}.xml"
            }
            
            results["modules"][module_name] = module_result
            if not module_result["success"]:
                results["success"] = False
                logger.error(f"Tests failed for {module_name}")
                logger.error(f"Error: {result.stderr}")
            else:
                logger.info(f"Tests passed for {module_name}")
            
        except Exception as e:
            logger.error(f"Error running tests for {module_name}: {e}")
            results["modules"][module_name] = {
                "exit_code": -1,
                "success": False,
                "output": "",
                "error": str(e),
                "report_html": "",
                "report_xml": ""
            }
            results["success"] = False
    
    # Write results to JSON
    with open(f"{REPORT_DIR}/results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def send_email_notification(results: Dict[str, Any], email_to: str) -> bool:
    """Send email notification with test results."""
    try:
        # Get email settings from environment
        email_from = os.environ.get("REGRESSION_EMAIL_FROM", "no-reply@me2ai.com")
        smtp_server = os.environ.get("REGRESSION_SMTP_SERVER", "localhost")
        smtp_port = int(os.environ.get("REGRESSION_SMTP_PORT", "25"))
        
        # Create message
        msg = MIMEMultipart()
        msg["Subject"] = f"VectorStore Regression Test Results: {'SUCCESS' if results['success'] else 'FAILURE'}"
        msg["From"] = email_from
        msg["To"] = email_to
        
        # Build HTML content
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>VectorStore Regression Test Results</h1>
            <p>Test run completed at: {results['timestamp']}</p>
            <p>Overall result: <span class="{'success' if results['success'] else 'failure'}">{
                'SUCCESS' if results['success'] else 'FAILURE'}</span></p>
            
            <h2>Module Results:</h2>
            <table>
                <tr>
                    <th>Module</th>
                    <th>Result</th>
                    <th>Exit Code</th>
                </tr>
        """
        
        for module, result in results["modules"].items():
            html += f"""
                <tr>
                    <td>{module}</td>
                    <td class="{'success' if result['success'] else 'failure'}">{
                        'PASS' if result['success'] else 'FAIL'}</td>
                    <td>{result['exit_code']}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <p>See attached logs for details.</p>
        </body>
        </html>
        """
        
        # Attach HTML part
        msg.attach(MIMEText(html, "html"))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.send_message(msg)
        
        logger.info(f"Email notification sent to {email_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False


def send_slack_notification(results: Dict[str, Any]) -> bool:
    """Send Slack notification with test results."""
    try:
        # Get Slack webhook URL from environment
        webhook_url = os.environ.get("REGRESSION_SLACK_WEBHOOK")
        if not webhook_url:
            logger.error("REGRESSION_SLACK_WEBHOOK environment variable not set")
            return False
        
        # Build message
        message = {
            "text": f"VectorStore Regression Test Results: {'SUCCESS' if results['success'] else 'FAILURE'}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "VectorStore Regression Test Results"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Test run completed at: {results['timestamp']}\nOverall result: *{'SUCCESS' if results['success'] else 'FAILURE'}*"
                    }
                },
                {
                    "type": "divider"
                }
            ]
        }
        
        # Add module results
        for module, result in results["modules"].items():
            message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{module}*: {'✅ PASS' if result['success'] else '❌ FAIL'} (Exit code: {result['exit_code']})"
                }
            })
        
        # Send to Slack
        import requests
        response = requests.post(
            webhook_url,
            json=message,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info("Slack notification sent successfully")
            return True
        else:
            logger.error(f"Failed to send Slack notification: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
        return False


def main() -> int:
    """Main entry point for regression test suite."""
    parser = argparse.ArgumentParser(description="Run Vector Store regression tests")
    parser.add_argument("--email", help="Email address to send notifications to")
    parser.add_argument("--slack", action="store_true", help="Send Slack notifications")
    args = parser.parse_args()
    
    # Log start
    logger.info("Starting Vector Store regression test suite")
    
    # Setup environment
    if not setup_environment():
        logger.error("Failed to set up environment")
        return 1
    
    # Run tests
    results = run_tests()
    
    # Send notifications
    if args.email:
        send_email_notification(results, args.email)
    
    if args.slack:
        send_slack_notification(results)
    
    # Log completion
    logger.info(f"Vector Store regression test suite completed with overall result: {'SUCCESS' if results['success'] else 'FAILURE'}")
    
    # Return exit code based on results
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
