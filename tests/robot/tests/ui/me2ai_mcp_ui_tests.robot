*** Settings ***
Documentation    Test suite for ME2AI MCP UI functionality
Resource         ../../resources/common.robot
Library          ../../libraries/mcp_ui_library.py
Test Setup      Setup UI Test
Test Teardown   Teardown UI Test
Force Tags      ui    regression
Default Tags    smoke    critical

*** Test Cases ***
Scenario: User navigates to MCP server UI successfully
    [Documentation]    Test basic navigation to MCP server UI
    [Tags]    navigation    stable
    Given MCP server is running
    When User navigates to MCP UI
    Then Page title should contain "MCP Server"
    And Tool selector should be visible

Scenario: User retrieves server information via UI
    [Documentation]    Test server_info tool via UI
    [Tags]    server_info    stable
    Given User is on MCP UI page
    When User selects "server_info" tool
    And User executes the tool
    Then Tool execution should succeed
    And Result should contain server name
    And Result should contain server version

Scenario: User fetches webpage content via UI
    [Documentation]    Test web fetch tool via UI
    [Tags]    web    fetch    high
    Given User is on MCP UI page
    When User selects "fetch_webpage" tool
    And User inputs URL "https://example.com"
    And User executes the tool
    Then Tool execution should succeed
    And Result should contain webpage title
    And Result should contain webpage content

Scenario: User receives error when invalid URL provided
    [Documentation]    Test error handling for invalid URL
    [Tags]    web    error_handling    high
    Given User is on MCP UI page
    When User selects "fetch_webpage" tool
    And User inputs URL "invalid-url"
    And User executes the tool
    Then Tool execution should fail
    And Error message should mention "URL"

Scenario: User uses GitHub repository search via UI
    [Documentation]    Test GitHub repository search via UI
    [Tags]    github    repository    medium
    Given User is on MCP UI page
    And GitHub token is configured
    When User selects "github_repository" tool
    And User inputs the following parameters:
    ...    operation=search
    ...    query=me2ai
    And User executes the tool
    Then Tool execution should succeed
    And Result should contain repositories
    And Each repository should have name and stars

*** Keywords ***
MCP server is running
    [Documentation]    Verify MCP server is running
    ${status}    ${message}=    Run Keyword And Ignore Error    
    ...    GET    ${SERVER_URL}/ping    expected_status=200
    Run Keyword If    '${status}' == 'FAIL'    Fail    MCP server is not running

User navigates to MCP UI
    [Documentation]    Navigate to MCP server UI
    Navigate To MCP UI    ${SERVER_URL}

Page title should contain "${text}"
    [Documentation]    Verify page title
    Title Should Contain    ${text}

Tool selector should be visible
    [Documentation]    Verify tool selector is visible
    Element Should Be Visible    id:tool-selector

User is on MCP UI page
    [Documentation]    Ensure user is on MCP UI page
    Navigate To MCP UI    ${SERVER_URL}

User selects "${tool_name}" tool
    [Documentation]    Select a tool from dropdown
    Select MCP Tool    ${tool_name}

User executes the tool
    [Documentation]    Execute the selected tool
    Execute Tool
    Wait For Tool Execution

Tool execution should succeed
    [Documentation]    Verify successful execution
    Verify Success Response

Tool execution should fail
    [Documentation]    Verify failed execution
    Verify Error Response

Result should contain server name
    [Documentation]    Verify server name in result
    Verify Tool Result Key    server
    Element Should Contain    id:result-container    "server_name":

Result should contain server version
    [Documentation]    Verify server version in result
    Element Should Contain    id:result-container    "version":

User inputs URL "${url}"
    [Documentation]    Input URL parameter
    Input Tool Parameter    url    ${url}

Result should contain webpage title
    [Documentation]    Verify webpage title in result
    Verify Tool Result Key    title

Result should contain webpage content
    [Documentation]    Verify webpage content in result
    Verify Tool Result Key    content

Error message should mention "${text}"
    [Documentation]    Verify error message contains text
    Verify Error Response    ${text}

GitHub token is configured
    [Documentation]    Verify GitHub token is configured
    ${token}=    Get Environment Variable    GITHUB_TOKEN    default=${EMPTY}
    Run Keyword If    '${token}' == '${EMPTY}'    Skip    GitHub token not configured

User inputs the following parameters:
    [Documentation]    Input multiple parameters
    [Arguments]    @{params}
    ${param_dict}=    Create Dictionary
    FOR    ${param}    IN    @{params}
        ${name}    ${value}=    Split String    ${param}    =
        Set To Dictionary    ${param_dict}    ${name}    ${value}
    END
    Input Tool Parameters    ${param_dict}

Result should contain repositories
    [Documentation]    Verify repositories in result
    Verify Tool Result Key    repositories

Each repository should have name and stars
    [Documentation]    Verify repository structure
    ${result}=    Get Tool Result
    FOR    ${repo}    IN    @{result["repositories"]}
        Dictionary Should Contain Key    ${repo}    name
        Dictionary Should Contain Key    ${repo}    stars
    END
