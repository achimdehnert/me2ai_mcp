# ME2AI MCP Robot Framework Test Suite

This directory contains Robot Framework test suites for the ME2AI MCP package, providing comprehensive BDD-style testing for API, UI functionality, and integrations with applications like the ME2AI Knowledge Assistant.

## Test Structure

The test suite follows a modular structure defined in the ME2AI development guidelines:

```text
tests/robot/
├── resources/                 # Shared resources
│   ├── common.robot           # Common setup and teardown
│   ├── variables.robot        # Global variables
│   └── keywords/              # Reusable keywords
│       ├── ui_keywords.robot  # UI test keywords
│       ├── api_keywords.robot # API test keywords
│       ├── vectorstore_keywords.robot  # Vector store keywords
│       ├── openai/            # OpenAI client keywords
│       │   └── openai_keywords.robot
│       └── embedding/         # Embedding test keywords
│           └── embedding_keywords.robot
├── libraries/                 # Custom test libraries
│   ├── OpenAIClientTestLibrary.py   # OpenAI client testing
│   ├── EmbeddingTestLibrary.py      # Embedding testing
│   └── VectorStoreTestLibrary.py    # Vector store testing
├── tests/                     # Test suites
│   ├── api/                   # API tests
│   │   ├── me2ai_mcp_base_tests.robot
│   │   ├── me2ai_mcp_web_tools_tests.robot
│   │   ├── openai_integration_tests.robot  # OpenAI client tests
│   │   └── vectorstore_adapter_tests.robot
│   ├── embedding/             # Embedding tests
│   │   └── embedding_service_tests.robot
│   └── ui/                    # UI tests with Selenium
│       └── me2ai_mcp_ui_tests.robot
└── README.md                  # This file
```

## Running Tests

### Prerequisites

The easiest way to set up the testing environment is to use the provided setup script:

```bash
python setup_robot_environment.py
```

This will:

1. Create a dedicated virtual environment for Robot Framework tests (`.venv-robot`)
2. Install all required packages
3. Configure VS Code settings for Robot Framework

If you prefer manual setup:

1. Install Robot Framework and required libraries:

   ```bash
   pip install robotframework>=7.0.0 robotframework-seleniumlibrary>=6.1.0 robotframework-requests>=0.9.5 robotframework-pythonlibcore>=4.1.2 webdrivermanager>=0.10.0
   ```

2. Download and set up Chrome/Firefox WebDriver:

   ```bash
   webdrivermanager chrome
   # or
   webdrivermanager firefox
   ```

3. Set up environment variables in a `.env` file:

   ```text
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Make sure an MCP server is running for tests:

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

## Test Categories

- **API Tests**: Verify the API functionality of ME2AI MCP components
- **Embedding Tests**: Verify the embedding capabilities and services
- **OpenAI Client Tests**: Verify the OpenAI client integration
- **Vector Store Tests**: Verify vector store adapters functionality
- **UI Tests**: Verify browser-based UI functionality with Selenium

## Test Tags

Tests are tagged for selective execution:

- `smoke`: Basic functionality tests
- `integration`: Integration tests
- `regression`: Regression tests
- `tools`: Tool-specific tests
- `ui`: UI tests
- `api`: API tests
- `openai`: OpenAI client tests
- `embedding`: Embedding-related tests
- `vectorstore`: Vector store tests
- `knowledge-assistant`: Tests specifically for ME2AI Knowledge Assistant compatibility

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

## Adding New Tests

Follow these guidelines when adding new tests:

1. Use BDD-style structure (Given, When, Then)
2. Keep test cases focused on a single functionality
3. Use shared keywords for common operations
4. Add proper documentation and tags
5. Follow the naming conventions:
   - Test cases: `Test Should [Expected Behavior] When [Condition]`
   - Keywords: `[Action] [Subject] [Details]`
   - Variables: Scalar: `${VARIABLE_NAME}`, List: `@{LIST_NAME}`, Dict: `&{DICT_NAME}`

## Test Results

Test results are generated in HTML and XML formats in the `results` directory:

- `report.html`: High-level test report
- `log.html`: Detailed test log
- `output.xml`: Machine-readable test results

## Running Specific Tests

Use tags to run specific test categories:

```bash
# Run only OpenAI integration tests
robot --outputdir results --include openai tests/robot/

# Run knowledge assistant integration tests
robot --outputdir results --include knowledge-assistant tests/robot/

# Run basic smoke tests for Vector Store adapters
robot --outputdir results --include vectorstore --include smoke tests/robot/
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines using:

```yaml
- name: Setup Robot Framework Environment
  run: python setup_robot_environment.py

- name: Run Knowledge Assistant Compatibility Tests
  run: |
    source .venv-robot/bin/activate  # On Windows: .venv-robot\Scripts\activate
    robot --outputdir results --include knowledge-assistant tests/robot/
- `EXAMPLE_API_KEY`: API key for authentication tests
