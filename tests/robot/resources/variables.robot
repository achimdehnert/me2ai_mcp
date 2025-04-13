*** Settings ***
Documentation     Common variables for ME2AI MCP package testing

*** Variables ***
# Server configuration
${SERVER_HOST}    localhost
${SERVER_PORT}    8080

# Test data paths
${TEST_DATA_DIR}    ${CURDIR}/../../data

# Authentication variables
${GITHUB_TOKEN}    %{GITHUB_TOKEN}
${API_KEY}         %{EXAMPLE_API_KEY}

# Test configuration
${MEDIUM_TIMEOUT}     10s
${LONG_TIMEOUT}       30s
${SHORT_TIMEOUT}      5s
