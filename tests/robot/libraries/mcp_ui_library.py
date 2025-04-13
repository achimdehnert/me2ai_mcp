"""
Custom MCP UI testing library for Robot Framework.

This library provides specialized keywords for testing MCP servers' web UI
using Selenium WebDriver through Robot Framework.
"""
from typing import Dict, Any, Optional, Union, List
import json
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from robot.api.deco import keyword
from robot.libraries.BuiltIn import BuiltIn


class MCPUILibrary:
    """Custom library for testing MCP server web interfaces."""
    
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    
    def __init__(self):
        """Initialize the library."""
        self.selenium_lib = None
        self.builtin = BuiltIn()
    
    def _get_selenium_library(self):
        """Get reference to the SeleniumLibrary instance."""
        if self.selenium_lib is None:
            self.selenium_lib = self.builtin.get_library_instance('SeleniumLibrary')
        return self.selenium_lib
    
    @keyword
    def navigate_to_mcp_ui(self, url: str):
        """Navigate to the MCP server UI.
        
        Args:
            url: URL of the MCP server UI
        """
        selenium = self._get_selenium_library()
        selenium.go_to(url)
        selenium.wait_until_page_contains_element("//h1[contains(text(), 'MCP')]", timeout="10s")
    
    @keyword
    def select_mcp_tool(self, tool_name: str):
        """Select an MCP tool from the UI dropdown.
        
        Args:
            tool_name: Name of the tool to select
        """
        selenium = self._get_selenium_library()
        # Find and click the tool selector
        selenium.click_element("id:tool-selector")
        # Wait for dropdown to appear and select the tool
        selenium.wait_until_element_is_visible(f"//option[text()='{tool_name}']", timeout="5s")
        selenium.click_element(f"//option[text()='{tool_name}']")
    
    @keyword
    def input_tool_parameter(self, param_name: str, param_value: str):
        """Input a parameter value for the selected tool.
        
        Args:
            param_name: Name of the parameter
            param_value: Value to input
        """
        selenium = self._get_selenium_library()
        # Find the parameter input field by its label
        selector = f"//label[contains(text(), '{param_name}')]/following-sibling::input"
        selenium.wait_until_element_is_visible(selector, timeout="5s")
        selenium.input_text(selector, param_value)
    
    @keyword
    def input_tool_parameters(self, parameters: Dict[str, str]):
        """Input multiple parameters for the selected tool.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        for name, value in parameters.items():
            self.input_tool_parameter(name, value)
    
    @keyword
    def execute_tool(self):
        """Click the execute button to run the tool."""
        selenium = self._get_selenium_library()
        selenium.click_button("id:execute-button")
        # Wait for execution to start
        selenium.wait_until_element_is_visible("id:loading-indicator", timeout="5s")
    
    @keyword
    def wait_for_tool_execution(self, timeout: str = "30s"):
        """Wait for the tool execution to complete.
        
        Args:
            timeout: Maximum time to wait for execution to complete
        """
        selenium = self._get_selenium_library()
        # Wait for loading indicator to disappear
        selenium.wait_until_element_is_not_visible("id:loading-indicator", timeout=timeout)
        # Wait for result container to appear
        selenium.wait_until_element_is_visible("id:result-container", timeout="5s")
    
    @keyword
    def get_tool_result(self) -> Dict[str, Any]:
        """Get the result of the tool execution as a dictionary.
        
        Returns:
            Dictionary containing the tool execution result
        """
        selenium = self._get_selenium_library()
        # Get the result text from the container
        result_text = selenium.get_text("id:result-container")
        # Parse JSON result
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON result", "raw_result": result_text}
    
    @keyword
    def verify_tool_result_contains(self, expected_content: str):
        """Verify that the tool result contains the expected text.
        
        Args:
            expected_content: Text that should be in the result
        """
        selenium = self._get_selenium_library()
        selenium.element_should_contain("id:result-container", expected_content)
    
    @keyword
    def verify_tool_result_key(self, key: str, expected_value: Optional[str] = None):
        """Verify a specific key (and optionally its value) in the tool result.
        
        Args:
            key: JSON key to verify
            expected_value: Expected value for the key (optional)
        """
        result = self.get_tool_result()
        # Check if key exists
        assert key in result, f"Key '{key}' not found in tool result"
        # Check value if provided
        if expected_value is not None:
            assert str(result[key]) == expected_value, f"Expected '{key}' to be '{expected_value}', got '{result[key]}'"
    
    @keyword
    def verify_success_response(self):
        """Verify that the tool result indicates success."""
        result = self.get_tool_result()
        assert "success" in result, "Success field not found in result"
        assert result["success"] is True, "Tool execution did not succeed"
    
    @keyword
    def verify_error_response(self, expected_error: Optional[str] = None):
        """Verify that the tool result indicates an error.
        
        Args:
            expected_error: Expected error message (optional)
        """
        result = self.get_tool_result()
        # Check success field
        assert "success" in result, "Success field not found in result"
        assert result["success"] is False, "Tool execution unexpectedly succeeded"
        # Check error field
        assert "error" in result, "Error field not found in result"
        # Check error message if provided
        if expected_error is not None:
            assert expected_error in result["error"], f"Expected error message containing '{expected_error}', got '{result['error']}'"
    
    @keyword
    def complete_tool_execution(self, tool_name: str, parameters: Dict[str, str]):
        """Complete a full tool execution cycle.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameter names and values
        """
        self.select_mcp_tool(tool_name)
        self.input_tool_parameters(parameters)
        self.execute_tool()
        self.wait_for_tool_execution()
        return self.get_tool_result()
