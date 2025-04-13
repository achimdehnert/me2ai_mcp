# ME2AI MCP Robot Framework Test Suite

This directory contains Robot Framework test suites for the ME2AI MCP package, providing comprehensive BDD-style testing for API and UI functionality.

## Test Structure

The test suite follows the structure defined in the ME2AI development guidelines:

```
tests/robot/
├── resources/              # Shared resources
│   ├── common.robot        # Common setup and teardown
│   ├── variables.robot     # Global variables
│   └── keywords/           # Reusable keywords
│       ├── ui_keywords.robot
│       └── api_keywords.robot
├── libraries/              # Custom test libraries
│   └── mcp_ui_library.py   # UI testing utilities
├── tests/                  # Test suites
│   ├── api/                # API tests
│   │   ├── me2ai_mcp_base_tests.robot
│   │   ├── me2ai_mcp_web_tools_tests.robot
│   │   └── me2ai_mcp_github_tools_tests.robot
│   └── ui/                 # UI tests with Selenium
│       └── me2ai_mcp_ui_tests.robot
└── README.md               # This file
```

## Running Tests

### Prerequisites

1. Install Robot Framework and required libraries:

```bash
pip install robotframework robotframework-seleniumlibrary robotframework-requests webdrivermanager
```

2. Download and set up Chrome/Firefox WebDriver:

```bash
webdrivermanager chrome
# or
webdrivermanager firefox
```

3. Make sure an MCP server is running for tests:

```bash
python examples/custom_mcp_server.py
```

### Running All Tests

```bash
robot -d reports tests/robot
```

### Running Specific Test Suites

```bash
# Run only API tests
robot -d reports tests/robot/tests/api

# Run only UI tests
robot -d reports tests/robot/tests/ui

# Run a specific test file
robot -d reports tests/robot/tests/api/me2ai_mcp_web_tools_tests.robot
```

### Running Tests by Tag

```bash
# Run all critical tests
robot -d reports -i critical tests/robot

# Run all positive web tests
robot -d reports -i web -i positive tests/robot
```

## Test Tags

Tests are tagged according to the ME2AI development guidelines:

### Required Tags

- **Feature area**: `web`, `github`, `auth`, etc.
- **Test type**: `smoke`, `regression`, `integration`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Status**: `stable`, `unstable`, `experimental`

### Optional Tags

- **Environment**: `dev`, `staging`, `prod`
- **Component**: `ui`, `api`
- **Positive/Negative**: `positive`, `negative`

## Writing New Tests

New tests should follow the Gherkin-style BDD format:

```robotframework
*** Test Cases ***
Scenario: User performs specific action
    [Documentation]    Test description
    [Tags]    feature    priority    status
    Given prerequisite condition
    When user performs action
    Then expected result should occur
    And another verification should pass
```

## Test Reports

Test reports are generated in the `reports` directory and include:
- Log file with detailed execution information
- Report file with test statistics and results
- Output.xml for programmatic processing

## Continuous Integration

These tests are designed to be run in CI environments. Set up the following environment variables:

- `GITHUB_TOKEN`: Valid GitHub API token for GitHub tool tests
- `EXAMPLE_API_KEY`: API key for authentication tests
