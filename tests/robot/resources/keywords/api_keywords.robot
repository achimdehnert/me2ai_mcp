*** Settings ***
Documentation     API test keywords for ME2AI MCP package testing
Library           RequestsLibrary
Library           Collections
Library           String
Library           JSONLibrary

*** Keywords ***
Call MCP Server Tool
    [Documentation]    Call an MCP server tool and verify response format
    [Arguments]    ${session}    ${tool_name}    ${params}=${None}
    ${payload}=    Create Dictionary    name=${tool_name}
    Run Keyword If    $params is not $None    Set To Dictionary    ${payload}    params=${params}
    ${headers}=    Create Dictionary    Content-Type=application/json
    ${response}=    POST On Session    ${session}    /mcp    json=${payload}    headers=${headers}
    [Return]    ${response}

Verify Success Response
    [Documentation]    Verify that the MCP response indicates success
    [Arguments]    ${response}
    Should Be Equal As Numbers    ${response.status_code}    200
    ${json}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${json}    success
    Should Be Equal    ${json["success"]}    ${True}
    [Return]    ${json}

Verify Error Response
    [Documentation]    Verify that the MCP response indicates an error
    [Arguments]    ${response}    ${expected_error}=${None}
    Should Be Equal As Numbers    ${response.status_code}    200
    ${json}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${json}    success
    Should Be Equal    ${json["success"]}    ${False}
    Dictionary Should Contain Key    ${json}    error
    Run Keyword If    $expected_error is not $None
    ...    Should Contain    ${json["error"]}    ${expected_error}
    [Return]    ${json}

Wait For Server To Start
    [Documentation]    Wait for the MCP server to become available
    [Arguments]    ${timeout}=${MEDIUM_TIMEOUT}
    Wait Until Keyword Succeeds    ${timeout}    1s    Check Server Status

Check Server Status
    [Documentation]    Check if the MCP server is running
    ${response}=    GET    ${SERVER_URL}/ping    expected_status=200
    Should Be Equal    ${response.text}    pong
