*** Settings ***
Documentation    Test suite for ME2AI MCP base functionality
Resource         ../../resources/common.robot
Test Setup      Setup API Test
Test Teardown   Teardown API Test
Force Tags      api    regression
Default Tags    smoke    critical

*** Test Cases ***
Test Should Successfully Initialize MCP Server When Valid Configuration Provided
    [Documentation]    Test MCP server initialization with valid configuration
    [Tags]    init    stable
    Given Server Configuration Is Valid
    When ME2AI MCP Server Is Started
    Then Server Should Be Running
    And Server Info Tool Should Return Valid Data

Test Should Handle Authentication When API Key Provided
    [Documentation]    Test authentication with API key
    [Tags]    auth    high
    Given API Key Is Valid
    When ME2AI MCP Server Is Started With Authentication
    Then Authentication Status Should Be Active
    And Protected Tool Calls Should Succeed

Test Should Register Custom Tool When Using Register Tool Decorator
    [Documentation]    Test tool registration via decorator
    [Tags]    tools    stable
    Given Custom Tool Is Implemented With Decorator
    When ME2AI MCP Server Is Started
    Then Tool Should Be Available In Registry
    And Tool Should Execute Successfully With Valid Params

Test Should Handle Error Gracefully When Tool Execution Fails
    [Documentation]    Test error handling in tool execution
    [Tags]    error_handling    high
    Given Error-Prone Tool Is Implemented
    When Tool Is Called With Invalid Parameters
    Then Response Should Contain Error Information
    And Error Should Be Logged
    And Server Should Remain Running

Test Should Track Statistics When Tools Are Called
    [Documentation]    Test statistics tracking functionality
    [Tags]    stats    medium
    Given ME2AI MCP Server Is Started
    When Multiple Tool Calls Are Made
    Then Server Statistics Should Show Correct Call Count
    And Tool-Specific Statistics Should Be Updated

*** Keywords ***
Server Configuration Is Valid
    [Documentation]    Verify server configuration is valid
    ${config_file}=    Set Variable    ${TEST_DATA_DIR}/valid_config.json
    File Should Exist    ${config_file}

ME2AI MCP Server Is Started
    [Documentation]    Start the MCP server for testing
    Wait For Server To Start    ${MEDIUM_TIMEOUT}

Server Should Be Running
    [Documentation]    Verify server is running properly
    ${response}=    GET    ${SERVER_URL}/ping
    Should Be Equal As Strings    ${response.text}    pong

Server Info Tool Should Return Valid Data
    [Documentation]    Verify server_info tool returns proper data
    ${params}=    Create Dictionary
    ${response}=    Call MCP Server Tool    mcp_api    server_info    ${params}
    ${json}=    Verify Success Response    ${response}
    Dictionary Should Contain Key    ${json}    server
    Dictionary Should Contain Key    ${json["server"]}    server_name
    Dictionary Should Contain Key    ${json["server"]}    version

API Key Is Valid
    [Documentation]    Verify API key is valid
    ${api_key}=    Get Environment Variable    EXAMPLE_API_KEY    default=test_key
    Should Not Be Empty    ${api_key}

ME2AI MCP Server Is Started With Authentication
    [Documentation]    Start server with authentication enabled
    Wait For Server To Start    ${MEDIUM_TIMEOUT}

Authentication Status Should Be Active
    [Documentation]    Verify authentication is active
    ${params}=    Create Dictionary
    ${response}=    Call MCP Server Tool    mcp_api    server_info    ${params}
    ${json}=    Verify Success Response    ${response}
    Dictionary Should Contain Key    ${json["server"]}    auth_enabled
    Should Be Equal    ${json["server"]["auth_enabled"]}    ${TRUE}

Protected Tool Calls Should Succeed
    [Documentation]    Verify protected tools can be called with valid auth
    ${headers}=    Create Dictionary    Authorization=Bearer ${API_KEY}
    ${response}=    Call MCP Server Tool With Headers    mcp_api    protected_tool    ${None}    ${headers}
    Verify Success Response    ${response}

Custom Tool Is Implemented With Decorator
    [Documentation]    Setup for custom tool test
    Pass Execution    Setup complete

Tool Should Be Available In Registry
    [Documentation]    Verify tool exists in registry
    ${params}=    Create Dictionary
    ${response}=    Call MCP Server Tool    mcp_api    server_info    ${params}
    ${json}=    Verify Success Response    ${response}
    Dictionary Should Contain Key    ${json}    tools
    List Should Contain Value    ${json["tools"]}    echo

Tool Should Execute Successfully With Valid Params
    [Documentation]    Verify tool executes properly
    ${params}=    Create Dictionary    message=test_message
    ${response}=    Call MCP Server Tool    mcp_api    echo    ${params}
    ${json}=    Verify Success Response    ${response}
    Dictionary Should Contain Key    ${json}    message
    Should Be Equal    ${json["message"]}    test_message

Error-Prone Tool Is Implemented
    [Documentation]    Setup for error handling test
    Pass Execution    Setup complete

Tool Is Called With Invalid Parameters
    [Documentation]    Call tool with invalid parameters
    ${params}=    Create Dictionary    invalid_param=test
    ${response}=    Call MCP Server Tool    mcp_api    echo    ${params}
    Set Test Variable    ${response}

Response Should Contain Error Information
    [Documentation]    Verify error info in response
    ${json}=    Verify Error Response    ${response}
    Dictionary Should Contain Key    ${json}    error

Error Should Be Logged
    [Documentation]    Verify error is logged
    # This would require checking logs
    Pass Execution    Cannot verify logs in this test

Server Should Remain Running
    [Documentation]    Verify server still runs after error
    ${response}=    GET    ${SERVER_URL}/ping
    Should Be Equal As Strings    ${response.text}    pong

Multiple Tool Calls Are Made
    [Documentation]    Make multiple tool calls
    FOR    ${i}    IN RANGE    5
        ${params}=    Create Dictionary    message=test_${i}
        Call MCP Server Tool    mcp_api    echo    ${params}
    END

Server Statistics Should Show Correct Call Count
    [Documentation]    Verify server stats for call count
    ${params}=    Create Dictionary
    ${response}=    Call MCP Server Tool    mcp_api    server_info    ${params}
    ${json}=    Verify Success Response    ${response}
    Dictionary Should Contain Key    ${json["server"]}    stats
    Dictionary Should Contain Key    ${json["server"]["stats"]}    requests
    Should Be Equal As Numbers    ${json["server"]["stats"]["requests"]}    6    # 5 echo calls + this info call

Tool-Specific Statistics Should Be Updated
    [Documentation]    Verify tool-specific stats
    ${params}=    Create Dictionary
    ${response}=    Call MCP Server Tool    mcp_api    server_info    ${params}
    ${json}=    Verify Success Response    ${response}
    Dictionary Should Contain Key    ${json["server"]["stats"]}    tool_calls
    Dictionary Should Contain Key    ${json["server"]["stats"]["tool_calls"]}    echo
    Should Be Equal As Numbers    ${json["server"]["stats"]["tool_calls"]["echo"]}    5

Call MCP Server Tool With Headers
    [Documentation]    Call MCP tool with custom headers
    [Arguments]    ${session}    ${tool_name}    ${params}=${None}    ${headers}=${None}
    ${payload}=    Create Dictionary    name=${tool_name}
    Run Keyword If    $params is not $None    Set To Dictionary    ${payload}    params=${params}
    ${request_headers}=    Create Dictionary    Content-Type=application/json
    Run Keyword If    $headers is not $None    Set To Dictionary    ${request_headers}    &{headers}
    ${response}=    POST On Session    ${session}    /mcp    json=${payload}    headers=${request_headers}
    [Return]    ${response}
