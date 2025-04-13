"""
Tests for the ME2AI MCP utility functions.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from me2ai_mcp.utils import (
    sanitize_input, 
    format_response, 
    load_config, 
    extract_text,
    summarize_text
)


class TestSanitizeInput:
    """Tests for the sanitize_input function."""

    def test_should_return_original_string_when_no_dangerous_content(self):
        """Test sanitization with safe input."""
        safe_input = "This is a safe string"
        result = sanitize_input(safe_input)
        assert result == safe_input

    def test_should_remove_script_tags_when_present_in_input(self):
        """Test sanitization with script tags."""
        unsafe_input = "Text with <script>alert('XSS')</script> script tags"
        result = sanitize_input(unsafe_input)
        assert "<script>" not in result
        assert "alert('XSS')" not in result
        assert "Text with  script tags" in result

    def test_should_handle_empty_input_when_provided(self):
        """Test sanitization with empty input."""
        result = sanitize_input("")
        assert result == ""
        
        result = sanitize_input(None)
        assert result == ""

    def test_should_handle_non_string_input_when_provided(self):
        """Test sanitization with non-string input."""
        result = sanitize_input(123)
        assert result == "123"
        
        result = sanitize_input({"key": "value"})
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    def test_should_allow_safe_html_when_specified(self):
        """Test sanitization with allowed HTML."""
        input_with_safe_html = "Text with <b>bold</b> and <i>italic</i> formatting"
        result = sanitize_input(input_with_safe_html, allow_safe_html=True)
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result
        
        # Test with disallowed HTML
        result = sanitize_input(input_with_safe_html, allow_safe_html=False)
        assert "<b>" not in result
        assert "<i>" not in result


class TestFormatResponse:
    """Tests for the format_response function."""

    def test_should_format_successful_response_when_data_provided(self):
        """Test formatting a successful response."""
        data = {"result": "test_value"}
        result = format_response(data)
        
        assert result["success"] is True
        assert result["result"] == "test_value"

    def test_should_format_error_response_when_error_provided(self):
        """Test formatting an error response."""
        error = "Test error message"
        result = format_response(error=error)
        
        assert result["success"] is False
        assert result["error"] == error

    def test_should_format_response_with_metadata_when_provided(self):
        """Test formatting a response with metadata."""
        data = {"result": "test_value"}
        metadata = {"execution_time": 0.5, "source": "test"}
        
        result = format_response(data, metadata=metadata)
        
        assert result["success"] is True
        assert result["result"] == "test_value"
        assert result["metadata"]["execution_time"] == 0.5
        assert result["metadata"]["source"] == "test"

    def test_should_prioritize_error_when_both_data_and_error_provided(self):
        """Test formatting when both data and error are provided."""
        data = {"result": "test_value"}
        error = "Test error message"
        
        result = format_response(data, error=error)
        
        assert result["success"] is False
        assert result["error"] == error
        assert "result" not in result


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_should_load_config_when_valid_json_file_exists(self):
        """Test loading a valid config file."""
        config_data = {
            "server_name": "test-server",
            "description": "Test server",
            "version": "1.0.0"
        }
        
        # Mock open to return a file-like object with our test config
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("pathlib.Path.exists", return_value=True):
                result = load_config("test_config.json")
                
                assert result["server_name"] == "test-server"
                assert result["description"] == "Test server"
                assert result["version"] == "1.0.0"

    def test_should_return_empty_dict_when_file_not_found(self):
        """Test loading a non-existent config file."""
        with patch("pathlib.Path.exists", return_value=False):
            result = load_config("nonexistent_config.json")
            assert result == {}

    def test_should_return_empty_dict_when_file_has_invalid_json(self):
        """Test loading a config file with invalid JSON."""
        # Mock open to return a file with invalid JSON
        with patch("builtins.open", mock_open(read_data="invalid json content")):
            with patch("pathlib.Path.exists", return_value=True):
                result = load_config("invalid_config.json")
                assert result == {}

    def test_should_merge_default_config_when_provided(self):
        """Test merging with default config."""
        config_data = {
            "server_name": "test-server",
            "version": "1.0.0"
        }
        
        default_config = {
            "server_name": "default-server",
            "description": "Default description",
            "debug": True
        }
        
        # Mock open to return a file-like object with our test config
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("pathlib.Path.exists", return_value=True):
                result = load_config("test_config.json", default_config)
                
                # Config from file should override defaults
                assert result["server_name"] == "test-server"
                # Missing values should be taken from defaults
                assert result["description"] == "Default description"
                assert result["debug"] is True
                # Values in config but not in defaults should remain
                assert result["version"] == "1.0.0"


class TestExtractText:
    """Tests for the extract_text function."""

    def test_should_extract_plain_text_when_html_provided(self):
        """Test extracting text from HTML."""
        html = "<html><body><h1>Title</h1><p>Paragraph text</p></body></html>"
        
        try:
            import bs4
            # BeautifulSoup is available
            result = extract_text(html)
            assert "Title" in result
            assert "Paragraph text" in result
            assert "<html>" not in result
            assert "<h1>" not in result
        except ImportError:
            # Skip test if BeautifulSoup not available
            pytest.skip("BeautifulSoup not available")

    def test_should_return_original_text_when_not_html(self):
        """Test extracting text from non-HTML content."""
        text = "This is plain text"
        result = extract_text(text)
        assert result == text

    def test_should_handle_empty_input_when_provided(self):
        """Test extracting text from empty input."""
        result = extract_text("")
        assert result == ""
        
        result = extract_text(None)
        assert result == ""


class TestSummarizeText:
    """Tests for the summarize_text function."""

    def test_should_truncate_to_max_length_when_text_too_long(self):
        """Test summarizing by truncation."""
        text = "This is a very long text that needs to be summarized"
        result = summarize_text(text, max_length=20)
        assert len(result) <= 24  # 20 + length of "..."
        assert result.endswith("...")
        assert result.startswith("This is a very")

    def test_should_return_original_text_when_shorter_than_max_length(self):
        """Test summarizing text shorter than max_length."""
        text = "Short text"
        result = summarize_text(text, max_length=20)
        assert result == text

    def test_should_handle_empty_input_when_provided(self):
        """Test summarizing empty input."""
        result = summarize_text("")
        assert result == ""
        
        result = summarize_text(None)
        assert result == ""
