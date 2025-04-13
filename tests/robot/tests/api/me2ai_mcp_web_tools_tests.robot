*** Settings ***
Documentation    Test suite for ME2AI MCP web tools
Resource         ../../resources/common.robot
Test Setup      Setup API Test
Test Teardown   Teardown API Test
Force Tags      api    web_tools    regression
Default Tags    smoke    high

*** Test Cases ***
Scenario: User fetches valid webpage content
    [Documentation]    Test web fetch tool with valid URL
    [Tags]    web    fetch    positive
    Given Web fetch tool is available
    When User fetches content from "https://example.com"
    Then Response should be successful
    And Response should contain webpage title
    And Response should contain webpage content

Scenario: User fetches content from invalid URL
    [Documentation]    Test web fetch tool with invalid URL
    [Tags]    web    fetch    negative
    Given Web fetch tool is available
    When User fetches content from "invalid-url"
    Then Response should contain error
    And Error should indicate invalid URL

Scenario: User extracts elements using HTML parser
    [Documentation]    Test HTML parser tool
    [Tags]    web    parser    positive
    Given HTML content is available
    When User extracts elements with selector "h1"
    Then Response should be successful
    And Response should contain extracted elements

Scenario: User processes URL with URL utils tool
    [Documentation]    Test URL utilities tool
    [Tags]    web    url    positive
    Given URL utils tool is available
    When User parses URL "https://example.com/path?query=value"
    Then Response should be successful
    And Response should contain parsed URL components
    And Query parameters should be extracted correctly

*** Keywords ***
Web fetch tool is available
    [Documentation]    Verify web fetch tool is available
    ${response}=    Call MCP Server Tool    mcp_api    server_info
    ${json}=    Verify Success Response    ${response}
    List Should Contain Value    ${json["tools"]}    fetch_webpage

User fetches content from "${url}"
    [Documentation]    Call web fetch tool with URL
    ${params}=    Create Dictionary    url=${url}
    ${response}=    Call MCP Server Tool    mcp_api    fetch_webpage    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${url}

Response should be successful
    [Documentation]    Verify successful response
    ${json}=    Verify Success Response    ${response}
    Set Test Variable    ${json}

Response should contain webpage title
    [Documentation]    Verify response contains title
    Dictionary Should Contain Key    ${json}    title
    Should Not Be Empty    ${json["title"]}

Response should contain webpage content
    [Documentation]    Verify response contains content
    Dictionary Should Contain Key    ${json}    content
    Should Not Be Empty    ${json["content"]}

Response should contain error
    [Documentation]    Verify response contains error
    ${json}=    Verify Error Response    ${response}
    Set Test Variable    ${json}

Error should indicate invalid URL
    [Documentation]    Verify error message about invalid URL
    Should Contain    ${json["error"]}    URL

HTML content is available
    [Documentation]    Setup HTML content for testing
    ${html}=    Set Variable    <html><body><h1>Test Header</h1><p>Test paragraph</p></body></html>
    Set Test Variable    ${html}

User extracts elements with selector "${selector}"
    [Documentation]    Call HTML parser with selector
    ${params}=    Create Dictionary    html=${html}    selectors=${{"test_elements": {"selector": "${selector}", "multiple": true}}}
    ${response}=    Call MCP Server Tool    mcp_api    parse_html    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${selector}

Response should contain extracted elements
    [Documentation]    Verify extracted elements
    Dictionary Should Contain Key    ${json}    extracted
    Dictionary Should Contain Key    ${json["extracted"]}    test_elements

URL utils tool is available
    [Documentation]    Verify URL utils tool is available
    ${response}=    Call MCP Server Tool    mcp_api    server_info
    ${json}=    Verify Success Response    ${response}
    List Should Contain Value    ${json["tools"]}    url_utils

User parses URL "${url}"
    [Documentation]    Parse URL with URL utils tool
    ${params}=    Create Dictionary    operation=parse    url=${url}
    ${response}=    Call MCP Server Tool    mcp_api    url_utils    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${url}

Response should contain parsed URL components
    [Documentation]    Verify parsed URL components
    Dictionary Should Contain Key    ${json}    parsed
    Dictionary Should Contain Key    ${json["parsed"]}    scheme
    Dictionary Should Contain Key    ${json["parsed"]}    netloc
    Dictionary Should Contain Key    ${json["parsed"]}    path

Query parameters should be extracted correctly
    [Documentation]    Verify query parameters
    Dictionary Should Contain Key    ${json}    query_params
    Dictionary Should Contain Key    ${json["query_params"]}    query
    Should Be Equal    ${json["query_params"]["query"]}    value
