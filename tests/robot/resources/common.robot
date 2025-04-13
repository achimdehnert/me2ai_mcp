*** Settings ***
Documentation     Common resources and keywords for ME2AI MCP package testing
Library           SeleniumLibrary
Library           RequestsLibrary
Library           OperatingSystem
Library           String
Library           Collections
Library           DateTime
Resource          variables.robot
Resource          keywords/ui_keywords.robot
Resource          keywords/api_keywords.robot

*** Variables ***
${BROWSER}        chrome
${HEADLESS}       ${TRUE}
${TIMEOUT}        10s
${SERVER_URL}     http://localhost:8080

*** Keywords ***
Setup Environment
    [Documentation]    Set up the test environment
    Set Log Level      DEBUG
    
Teardown Environment
    [Documentation]    Clean up the test environment
    Run Keyword If Test Failed    Capture Page Screenshot
    Close All Browsers

Setup API Test
    [Documentation]    Set up environment for API tests
    Setup Environment
    Create Session    mcp_api    ${SERVER_URL}    verify=${FALSE}

Teardown API Test
    [Documentation]    Clean up after API tests
    Delete All Sessions
    Teardown Environment

Setup UI Test
    [Documentation]    Set up environment for UI tests
    Setup Environment
    Open Browser    ${SERVER_URL}    ${BROWSER}    options=add_argument("--headless")
    Set Window Size    1920    1080

Teardown UI Test
    [Documentation]    Clean up after UI tests
    Close All Browsers
    Teardown Environment

Take Screenshot On Failure
    [Documentation]    Capture screenshot when a test fails
    Run Keyword If Test Failed    Capture Page Screenshot    filename=failure-{index}.png
