"""
Unit tests for the ME2AI MCP utils module.

These tests verify the functionality of utility functions for
text processing, response formatting, and data handling.
"""
import pytest
from typing import Dict, Any, Optional
import json
import re

from me2ai_mcp.utils import (
    sanitize_input,
    format_response,
    extract_text,
    summarize_text,
    wrap_text_block
)


class TestSanitizeInput:
    """Tests for the sanitize_input function."""

    def test_should_handle_normal_text_input(self) -> None:
        """Test sanitizing normal text input."""
        # Arrange
        text = "Hello, world!"
        
        # Act
        result = sanitize_input(text)
        
        # Assert
        assert result == text
        assert isinstance(result, str)

    def test_should_truncate_long_input(self) -> None:
        """Test truncating input exceeding max length."""
        # Arrange
        text = "x" * 12000
        max_length = 5000
        
        # Act
        result = sanitize_input(text, max_length)
        
        # Assert
        assert len(result) == max_length
        assert result == "x" * max_length

    def test_should_convert_non_string_input(self) -> None:
        """Test converting non-string input to string."""
        # Arrange
        number_input = 12345
        
        # Act
        result = sanitize_input(number_input)  # type: ignore
        
        # Assert
        assert result == "12345"
        assert isinstance(result, str)

    def test_should_remove_control_characters(self) -> None:
        """Test removing control characters from input."""
        # Arrange
        text = "Hello\x00World\x01\x02\x03"
        
        # Act
        result = sanitize_input(text)
        
        # Assert
        assert result == "HelloWorld"
        assert "\x00" not in result
        assert "\x01" not in result


class TestFormatResponse:
    """Tests for the format_response function."""

    def test_should_format_successful_text_response(self) -> None:
        """Test formatting a successful text response."""
        # Arrange
        data = "Hello, world!"
        
        # Act
        result = format_response(data, format_type="text")
        
        # Assert
        assert result["success"] is True
        assert result["data"] == data
        assert result["format"] == "text"
        assert "error" not in result

    def test_should_format_error_response(self) -> None:
        """Test formatting an error response."""
        # Arrange
        error_message = "Something went wrong"
        
        # Act
        result = format_response(None, success=False, error=error_message)
        
        # Assert
        assert result["success"] is False
        assert result["error"] == error_message
        assert result.get("data") is None

    def test_should_auto_detect_json_format(self) -> None:
        """Test auto-detecting JSON format from dictionary data."""
        # Arrange
        data = {"name": "Test", "value": 123}
        
        # Act
        result = format_response(data)  # Using auto format
        
        # Assert
        assert result["success"] is True
        assert result["data"] == data
        assert result["format"] == "json"

    def test_should_auto_detect_text_format(self) -> None:
        """Test auto-detecting text format from string data."""
        # Arrange
        data = "Plain text data"
        
        # Act
        result = format_response(data)  # Using auto format
        
        # Assert
        assert result["success"] is True
        assert result["data"] == data
        assert result["format"] == "text"


class TestExtractText:
    """Tests for the extract_text function."""

    def test_should_extract_text_from_html(self) -> None:
        """Test extracting text from HTML content."""
        # Arrange
        html = "<html><body><h1>Title</h1><p>Paragraph text</p></body></html>"
        
        # Act
        result = extract_text(html)
        
        # Assert
        assert "Title" in result
        assert "Paragraph text" in result
        assert "<html>" not in result
        assert "<body>" not in result

    def test_should_respect_max_length(self) -> None:
        """Test respecting max length when extracting text."""
        # Arrange
        html = "<html><body>" + "<p>Test</p>" * 1000 + "</body></html>"
        max_length = 100
        
        # Act
        result = extract_text(html, max_length=max_length)
        
        # Assert
        assert len(result) <= max_length

    def test_should_handle_empty_html(self) -> None:
        """Test handling empty HTML content."""
        # Arrange
        html = ""
        
        # Act
        result = extract_text(html)
        
        # Assert
        assert result == ""

    def test_should_exclude_headings_when_specified(self) -> None:
        """Test excluding headings when specified."""
        # Arrange
        html = "<html><body><h1>Title</h1><p>Paragraph text</p></body></html>"
        
        # Act
        result = extract_text(html, include_headings=False)
        
        # Assert
        assert "Title" not in result
        assert "Paragraph text" in result


class TestSummarizeText:
    """Tests for the summarize_text function."""

    def test_should_summarize_text_by_truncation(self) -> None:
        """Test summarizing text by truncation."""
        # Arrange
        text = "This is a very long text. " * 100
        max_length = 50
        
        # Act
        result = summarize_text(text, max_length=max_length, preserve_sentences=False)
        
        # Assert
        assert len(result) <= max_length
        assert result.startswith("This is")

    def test_should_preserve_complete_sentences(self) -> None:
        """Test preserving complete sentences when summarizing."""
        # Arrange
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        max_length = 30
        
        # Act
        result = summarize_text(text, max_length=max_length)
        
        # Assert
        assert len(result) <= max_length
        assert result.endswith(".")
        assert "sentence." in result
        # Check it doesn't cut in the middle of a sentence
        assert not result.endswith("sentence")

    def test_should_handle_short_text(self) -> None:
        """Test handling text shorter than max length."""
        # Arrange
        text = "Short text."
        max_length = 100
        
        # Act
        result = summarize_text(text, max_length=max_length)
        
        # Assert
        assert result == text


class TestWrapTextBlock:
    """Tests for the wrap_text_block function."""

    def test_should_wrap_text_to_specified_width(self) -> None:
        """Test wrapping text to specified width."""
        # Arrange
        text = "This is a long line of text that needs to be wrapped to a shorter width."
        width = 20
        
        # Act
        result = wrap_text_block(text, width=width)
        
        # Assert
        lines = result.split("\n")
        for line in lines:
            assert len(line) <= width

    def test_should_add_prefix_to_each_line(self) -> None:
        """Test adding prefix to each wrapped line."""
        # Arrange
        text = "This is a text block that will be wrapped with a prefix."
        prefix = ">> "
        width = 20
        
        # Act
        result = wrap_text_block(text, width=width, prefix=prefix)
        
        # Assert
        lines = result.split("\n")
        for line in lines:
            assert line.startswith(prefix)
            assert len(line) <= width + len(prefix)

    def test_should_handle_empty_text(self) -> None:
        """Test handling empty text."""
        # Arrange
        text = ""
        
        # Act
        result = wrap_text_block(text)
        
        # Assert
        assert result == ""


if __name__ == "__main__":
    pytest.main()
