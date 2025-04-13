*** Settings ***
Documentation    Test suite for ME2AI MCP GitHub tools
Resource         ../../resources/common.robot
Test Setup      Setup API Test
Test Teardown   Teardown API Test
Force Tags      api    github_tools    regression
Default Tags    smoke    high

*** Test Cases ***
Scenario: User searches GitHub repositories successfully
    [Documentation]    Test GitHub repository search functionality
    [Tags]    github    repository    search    positive
    Given GitHub repository tool is available
    When User searches repositories with query "me2ai"
    Then Response should be successful
    And Response should contain repository results
    And Search metadata should be correct

Scenario: User gets repository details when valid repo name provided
    [Documentation]    Test GitHub repository details functionality
    [Tags]    github    repository    details    positive
    Given GitHub repository tool is available
    When User gets details for repository "achimdehnert/me2ai"
    Then Response should be successful
    And Response should contain repository information
    And Repository metadata should be included

Scenario: User lists repository contents when valid path provided
    [Documentation]    Test GitHub repository contents listing
    [Tags]    github    repository    contents    positive
    Given GitHub repository tool is available
    When User lists contents for repository "achimdehnert/me2ai" with path ""
    Then Response should be successful
    And Response should contain directory listing
    And Content items should be properly categorized

Scenario: User searches code with specific language filter
    [Documentation]    Test GitHub code search with language filter
    [Tags]    github    code    search    positive
    Given GitHub code tool is available
    When User searches code with query "function" and language "python"
    Then Response should be successful
    And Response should contain code search results
    And Results should include matching code snippets

Scenario: User gets file content when valid file path provided
    [Documentation]    Test GitHub file content retrieval
    [Tags]    github    code    file    positive
    Given GitHub code tool is available
    When User gets file content from repository "achimdehnert/me2ai" with path "README.md"
    Then Response should be successful
    And Response should contain file content
    And File metadata should be included

Scenario: User lists issues for repository successfully
    [Documentation]    Test GitHub issues listing functionality
    [Tags]    github    issues    list    positive
    Given GitHub issues tool is available
    When User lists issues for repository "achimdehnert/me2ai" with state "open"
    Then Response should be successful
    And Response should contain issue listing
    And Issue items should have required properties

Scenario: User handles authentication error when no token provided
    [Documentation]    Test GitHub authentication error handling
    [Tags]    github    auth    negative
    Given GitHub authentication is disabled
    When User attempts authenticated GitHub operation
    Then Response should contain authentication error
    And Error should suggest configuring GitHub token

*** Keywords ***
GitHub repository tool is available
    [Documentation]    Verify GitHub repository tool is available
    ${response}=    Call MCP Server Tool    mcp_api    server_info
    ${json}=    Verify Success Response    ${response}
    List Should Contain Value    ${json["tools"]}    github_repository

User searches repositories with query "${query}"
    [Documentation]    Call GitHub repository search
    ${params}=    Create Dictionary    operation=search    query=${query}
    ${response}=    Call MCP Server Tool    mcp_api    github_repository    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${query}

Response should be successful
    [Documentation]    Verify successful response
    ${json}=    Verify Success Response    ${response}
    Set Test Variable    ${json}

Response should contain repository results
    [Documentation]    Verify repository results
    Dictionary Should Contain Key    ${json}    repositories
    Should Not Be Empty    ${json["repositories"]}

Search metadata should be correct
    [Documentation]    Verify search metadata
    Dictionary Should Contain Key    ${json}    query
    Should Be Equal    ${json["query"]}    ${query}
    Dictionary Should Contain Key    ${json}    total_count
    Dictionary Should Contain Key    ${json}    count

User gets details for repository "${repo_name}"
    [Documentation]    Get repository details
    ${params}=    Create Dictionary    operation=get_details    repo_name=${repo_name}
    ${response}=    Call MCP Server Tool    mcp_api    github_repository    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${repo_name}

Response should contain repository information
    [Documentation]    Verify repository information
    Dictionary Should Contain Key    ${json}    repository
    Dictionary Should Contain Key    ${json["repository"]}    full_name
    Should Be Equal    ${json["repository"]["full_name"]}    ${repo_name}

Repository metadata should be included
    [Documentation]    Verify repository metadata
    Dictionary Should Contain Key    ${json["repository"]}    description
    Dictionary Should Contain Key    ${json["repository"]}    stars
    Dictionary Should Contain Key    ${json["repository"]}    forks
    Dictionary Should Contain Key    ${json["repository"]}    languages

User lists contents for repository "${repo_name}" with path "${path}"
    [Documentation]    List repository contents
    ${params}=    Create Dictionary    operation=list_contents    repo_name=${repo_name}    path=${path}
    ${response}=    Call MCP Server Tool    mcp_api    github_repository    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${repo_name}
    Set Test Variable    ${path}

Response should contain directory listing
    [Documentation]    Verify directory listing
    Dictionary Should Contain Key    ${json}    contents
    Should Not Be Empty    ${json["contents"]}

Content items should be properly categorized
    [Documentation]    Verify content categorization
    FOR    ${item}    IN    @{json["contents"]}
        Dictionary Should Contain Key    ${item}    type
        Dictionary Should Contain Key    ${item}    name
        Dictionary Should Contain Key    ${item}    path
        Run Keyword If    "${item["type"]}" == "file"    Dictionary Should Contain Key    ${item}    size
    END

GitHub code tool is available
    [Documentation]    Verify GitHub code tool is available
    ${response}=    Call MCP Server Tool    mcp_api    server_info
    ${json}=    Verify Success Response    ${response}
    List Should Contain Value    ${json["tools"]}    github_code

User searches code with query "${query}" and language "${language}"
    [Documentation]    Search code with language filter
    ${params}=    Create Dictionary    operation=search    query=${query}    language=${language}
    ${response}=    Call MCP Server Tool    mcp_api    github_code    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${query}
    Set Test Variable    ${language}

Response should contain code search results
    [Documentation]    Verify code search results
    Dictionary Should Contain Key    ${json}    results
    Dictionary Should Contain Key    ${json}    language
    Should Be Equal    ${json["language"]}    ${language}

Results should include matching code snippets
    [Documentation]    Verify code snippets in results
    FOR    ${item}    IN    @{json["results"]}
        Dictionary Should Contain Key    ${item}    name
        Dictionary Should Contain Key    ${item}    path
        Dictionary Should Contain Key    ${item}    html_url
        Dictionary Should Contain Key    ${item}    repository
    END

User gets file content from repository "${repo_name}" with path "${file_path}"
    [Documentation]    Get file content
    ${params}=    Create Dictionary    operation=get_file    repo_name=${repo_name}    file_path=${file_path}
    ${response}=    Call MCP Server Tool    mcp_api    github_code    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${repo_name}
    Set Test Variable    ${file_path}

Response should contain file content
    [Documentation]    Verify file content
    Dictionary Should Contain Key    ${json}    content
    Should Not Be Empty    ${json["content"]}

File metadata should be included
    [Documentation]    Verify file metadata
    Dictionary Should Contain Key    ${json}    name
    Dictionary Should Contain Key    ${json}    size
    Dictionary Should Contain Key    ${json}    html_url
    Dictionary Should Contain Key    ${json}    download_url

GitHub issues tool is available
    [Documentation]    Verify GitHub issues tool is available
    ${response}=    Call MCP Server Tool    mcp_api    server_info
    ${json}=    Verify Success Response    ${response}
    List Should Contain Value    ${json["tools"]}    github_issues

User lists issues for repository "${repo_name}" with state "${state}"
    [Documentation]    List repository issues
    ${params}=    Create Dictionary    operation=list    repo_name=${repo_name}    state=${state}
    ${response}=    Call MCP Server Tool    mcp_api    github_issues    ${params}
    Set Test Variable    ${response}
    Set Test Variable    ${repo_name}
    Set Test Variable    ${state}

Response should contain issue listing
    [Documentation]    Verify issue listing
    Dictionary Should Contain Key    ${json}    issues
    Dictionary Should Contain Key    ${json}    state
    Should Be Equal    ${json["state"]}    ${state}

Issue items should have required properties
    [Documentation]    Verify issue properties
    FOR    ${item}    IN    @{json["issues"]}
        Dictionary Should Contain Key    ${item}    number
        Dictionary Should Contain Key    ${item}    title
        Dictionary Should Contain Key    ${item}    state
        Dictionary Should Contain Key    ${item}    url
        Dictionary Should Contain Key    ${item}    user
    END

GitHub authentication is disabled
    [Documentation]    Set up environment without GitHub token
    Set Environment Variable    GITHUB_TOKEN    ${EMPTY}
    Set Environment Variable    GITHUB_API_KEY    ${EMPTY}

User attempts authenticated GitHub operation
    [Documentation]    Attempt operation requiring authentication
    ${params}=    Create Dictionary    operation=get_details    repo_name=private/repo
    ${response}=    Call MCP Server Tool    mcp_api    github_repository    ${params}
    Set Test Variable    ${response}

Response should contain authentication error
    [Documentation]    Verify authentication error
    ${json}=    Verify Error Response    ${response}
    Set Test Variable    ${json}

Error should suggest configuring GitHub token
    [Documentation]    Verify error message contains token suggestion
    Should Contain    ${json["error"]}    token
    Should Contain    ${json["error"]}    authentication
