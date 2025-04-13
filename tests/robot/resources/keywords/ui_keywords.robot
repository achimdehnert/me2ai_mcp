*** Settings ***
Documentation     UI test keywords for ME2AI MCP package testing
Library           SeleniumLibrary
Library           String
Library           Collections

*** Keywords ***
Navigate To Server UI
    [Documentation]    Navigate to the MCP server web UI
    Go To    ${SERVER_URL}
    Wait Until Page Contains    MCP Server    timeout=${MEDIUM_TIMEOUT}
    
Wait For Page To Load
    [Documentation]    Wait for the page to load completely
    Wait For Condition    return document.readyState == "complete"    ${MEDIUM_TIMEOUT}

Input Tool Parameters
    [Documentation]    Enter parameters for an MCP tool
    [Arguments]    ${tool_name}    ${params_dict}
    Select Tool    ${tool_name}
    FOR    ${key}    ${value}    IN    &{params_dict}
        Input Text    //label[contains(text(), "${key}")]/following-sibling::input    ${value}
    END

Select Tool
    [Documentation]    Select a tool from the UI dropdown
    [Arguments]    ${tool_name}
    Click Element    id=tool-selector
    Wait Until Element Is Visible    xpath=//option[text()="${tool_name}"]    ${SHORT_TIMEOUT}
    Click Element    xpath=//option[text()="${tool_name}"]

Execute Tool
    [Documentation]    Click the execute button to run a tool
    Click Button    id=execute-button
    Wait For Tool Execution

Wait For Tool Execution
    [Documentation]    Wait for the tool execution to complete
    Wait Until Element Is Visible    id=result-container    ${LONG_TIMEOUT}
    Wait Until Page Does Not Contain Element    id=loading-indicator    ${SHORT_TIMEOUT}

Verify Tool Result Contains
    [Documentation]    Verify that the tool result contains expected text
    [Arguments]    ${expected_text}
    Element Should Contain    id=result-container    ${expected_text}

Verify Tool Result JSON Key
    [Documentation]    Verify a specific key in the JSON result
    [Arguments]    ${key}    ${expected_value}
    ${result_text}=    Get Text    id=result-container
    Should Contain    ${result_text}    "${key}": "${expected_value}"
