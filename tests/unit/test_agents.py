"""
Unit tests for ME2AI MCP agents module.

Diese Tests prüfen die Funktionalität der Agent-Abstraktionen
mit 100% Coverage für das agents.py Modul.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Set

from me2ai_mcp.agents import (
    ToolCategory, BaseAgent, RoutingAgent, 
    SpecializedAgent, DEFAULT_CATEGORIES
)
from me2ai_mcp.base import ME2AIMCPServer


class TestToolCategory:
    """Tests für die ToolCategory-Klasse."""
    
    def test_should_initialize_with_correct_attributes(self) -> None:
        """Test, dass ToolCategory-Objekte korrekt initialisiert werden."""
        # Arrange
        name = "test_category"
        description = "Test category description"
        keywords = {"key1", "key2", "KEY3"}
        
        # Act
        category = ToolCategory(name, description, keywords)
        
        # Assert
        assert category.name == name
        assert category.description == description
        assert "key1" in category.keywords
        assert "key2" in category.keywords
        assert "key3" in category.keywords  # Sollte kleingeschrieben sein
        assert len(category.keywords) == 3
    
    def test_should_convert_keywords_to_lowercase(self) -> None:
        """Test, dass Keywords in Kleinbuchstaben konvertiert werden."""
        # Arrange & Act
        category = ToolCategory(
            "test", "description", {"TEST", "Test", "test"}
        )
        
        # Assert
        assert len(category.keywords) == 1  # Duplikate werden entfernt
        assert "test" in category.keywords
        assert "TEST" not in category.keywords
    
    def test_should_match_when_query_contains_keyword(self) -> None:
        """Test, dass matches() true zurückgibt, wenn die Anfrage ein Keyword enthält."""
        # Arrange
        category = ToolCategory(
            "test", "description", {"python", "code", "test"}
        )
        
        # Act & Assert
        assert category.matches("Dies ist ein Python Test") is True
        assert category.matches("Code beispiel") is True
        assert category.matches("TEST in großbuchstaben") is True
        assert category.matches("Kein passender Begriff") is False
    
    def test_should_handle_empty_keywords(self) -> None:
        """Test, dass matches() mit leerer Keyword-Liste umgehen kann."""
        # Arrange
        category = ToolCategory("empty", "No keywords", set())
        
        # Act & Assert
        assert category.matches("Test query") is False


class TestBaseAgent:
    """Tests für die BaseAgent-Klasse."""
    
    def test_should_initialize_with_correct_attributes(self) -> None:
        """Test, dass BaseAgent-Objekte korrekt initialisiert werden."""
        # Arrange
        agent_id = "test_agent"
        name = "Test Agent"
        description = "Test agent description"
        server_mock = MagicMock(spec=ME2AIMCPServer)
        
        # Act
        # Da BaseAgent abstrakt ist, erstellen wir eine konkrete Unterklasse für Tests
        class ConcreteAgent(BaseAgent):
            def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
                return {"processed": request}
        
        agent = ConcreteAgent(agent_id, name, description, server_mock)
        
        # Assert
        assert agent.agent_id == agent_id
        assert agent.name == name
        assert agent.description == description
        assert agent.server == server_mock
        assert agent.request_count == 0
        assert agent.error_count == 0
    
    def test_should_connect_to_server(self) -> None:
        """Test, dass connect_to_server() den Server korrekt setzt."""
        # Arrange
        class ConcreteAgent(BaseAgent):
            def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
                return {"processed": request}
        
        agent = ConcreteAgent("test", "Test", "Test description")
        server_mock = MagicMock(spec=ME2AIMCPServer)
        server_mock.server_name = "test_server"
        
        # Act
        agent.connect_to_server(server_mock)
        
        # Assert
        assert agent.server == server_mock
    
    def test_should_get_available_tools(self) -> None:
        """Test, dass _get_available_tools() die verfügbaren Tools zurückgibt."""
        # Arrange
        class ConcreteAgent(BaseAgent):
            def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
                return {"processed": request}
        
        agent = ConcreteAgent("test", "Test", "Test description")
        server_mock = MagicMock(spec=ME2AIMCPServer)
        
        # Tool-Methoden hinzufügen
        server_mock.tool1 = lambda x: x
        server_mock.tool2 = lambda x: x
        server_mock._internal = lambda x: x  # Sollte ignoriert werden
        
        agent.server = server_mock
        
        # Act
        tools = agent._get_available_tools()
        
        # Assert
        assert "tool1" in tools
        assert "tool2" in tools
        assert "_internal" not in tools
        assert callable(tools["tool1"])
    
    def test_should_handle_missing_server(self) -> None:
        """Test, dass _get_available_tools() ein leeres Dict zurückgibt, wenn kein Server verbunden ist."""
        # Arrange
        class ConcreteAgent(BaseAgent):
            def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
                return {"processed": request}
        
        agent = ConcreteAgent("test", "Test", "Test description")
        agent.server = None
        
        # Act
        tools = agent._get_available_tools()
        
        # Assert
        assert isinstance(tools, dict)
        assert len(tools) == 0
    
    def test_should_log_request(self) -> None:
        """Test, dass _log_request() die Anfragenzahl erhöht."""
        # Arrange
        class ConcreteAgent(BaseAgent):
            def process_request(self, request: str, **kwargs) -> Dict[str, Any]:
                self._log_request(request)
                return {"processed": request}
        
        agent = ConcreteAgent("test", "Test", "Test description")
        
        # Act
        agent.process_request("Test request")
        agent.process_request("Another test request")
        
        # Assert
        assert agent.request_count == 2


class TestRoutingAgent:
    """Tests für die RoutingAgent-Klasse."""
    
    def test_should_initialize_with_categories(self) -> None:
        """Test, dass RoutingAgent mit Kategorien initialisiert werden kann."""
        # Arrange
        categories = [
            ToolCategory("cat1", "Category 1", {"key1"}),
            ToolCategory("cat2", "Category 2", {"key2"})
        ]
        
        # Act
        agent = RoutingAgent("test", "Test", categories=categories)
        
        # Assert
        assert len(agent.categories) == 2
        assert agent.categories[0].name == "cat1"
        assert agent.categories[1].name == "cat2"
        assert len(agent.tool_registry) == 0
    
    def test_should_register_tool_categories(self) -> None:
        """Test, dass register_tool_categories() Tools korrekt kategorisiert."""
        # Arrange
        categories = [
            ToolCategory("text", "Text processing", {"text", "process"}),
            ToolCategory("data", "Data tools", {"data", "store"})
        ]
        
        server_mock = MagicMock(spec=ME2AIMCPServer)
        
        # Tool-Methoden mit Docstrings hinzufügen
        server_mock.process_text = MagicMock(__doc__="Process text data")
        server_mock.store_data = MagicMock(__doc__="Store data in the database")
        server_mock.list_tools = MagicMock()  # Sollte ignoriert werden
        
        agent = RoutingAgent("test", "Test", server=server_mock, categories=categories)
        
        # Act
        agent.register_tool_categories()
        
        # Assert
        assert "process_text" in agent.tool_registry
        assert "store_data" in agent.tool_registry
        assert "list_tools" not in agent.tool_registry
        
        # Überprüfen, dass Tool den richtigen Kategorien zugeordnet ist
        text_tool, text_categories = agent.tool_registry["process_text"]
        assert text_categories[0].name == "text"
    
    def test_should_handle_missing_server_in_register(self) -> None:
        """Test, dass register_tool_categories() ohne Server nichts macht."""
        # Arrange
        agent = RoutingAgent("test", "Test")
        agent.server = None
        
        # Act
        agent.register_tool_categories()
        
        # Assert
        assert len(agent.tool_registry) == 0
    
    def test_should_process_request_with_matching_tools(self) -> None:
        """Test, dass process_request() passende Tools findet und ausführt."""
        # Arrange
        categories = [
            ToolCategory("text", "Text processing", {"text", "process"})
        ]
        
        server_mock = MagicMock(spec=ME2AIMCPServer)
        tool_mock = MagicMock(return_value={"result": "processed"})
        tool_mock.__doc__ = "Process text data"
        server_mock.process_text = tool_mock
        
        agent = RoutingAgent("test", "Test", server=server_mock, categories=categories)
        
        # Mock für _find_matching_tools
        agent._find_matching_tools = MagicMock(
            return_value={"process_text": tool_mock}
        )
        
        # Act
        result = agent.process_request("Process this text")
        
        # Assert
        assert result["success"] is True
        assert "results" in result["data"]
        assert "process_text" in result["data"]["results"]
        assert result["data"]["tools_used"] == ["process_text"]
        tool_mock.assert_called_once()
    
    def test_should_handle_errors_in_process_request(self) -> None:
        """Test, dass process_request() Fehler in Tools abfängt."""
        # Arrange
        categories = [
            ToolCategory("text", "Text processing", {"text", "process"})
        ]
        
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = RoutingAgent("test", "Test", server=server_mock, categories=categories)
        
        # Mock für _find_matching_tools mit einem fehlerhaften Tool
        error_tool = MagicMock(side_effect=ValueError("Test error"))
        agent._find_matching_tools = MagicMock(
            return_value={"error_tool": error_tool}
        )
        
        # Act
        result = agent.process_request("Test request")
        
        # Assert
        assert "errors" in result["data"]
        assert "error_tool: Test error" in result["data"]["errors"]
        assert agent.error_count == 1
    
    def test_should_find_matching_tools(self) -> None:
        """Test, dass _find_matching_tools() passende Tools findet."""
        # Arrange
        categories = [
            ToolCategory("text", "Text processing", {"text", "process"}),
            ToolCategory("data", "Data tools", {"data", "store"})
        ]
        
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = RoutingAgent("test", "Test", server=server_mock, categories=categories)
        
        # Tool-Registry manuell setzen
        text_tool = MagicMock()
        data_tool = MagicMock()
        agent.tool_registry = {
            "process_text": (text_tool, [categories[0]]),
            "store_data": (data_tool, [categories[1]])
        }
        
        # Act - nach Text-Tools suchen
        text_tools = agent._find_matching_tools("process text example")
        
        # Assert
        assert "process_text" in text_tools
        assert "store_data" not in text_tools
    
    def test_should_register_tools_in_find_matching_if_empty(self) -> None:
        """Test, dass _find_matching_tools() register_tool_categories() aufruft, wenn registry leer ist."""
        # Arrange
        categories = [
            ToolCategory("text", "Text processing", {"text", "process"})
        ]
        
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = RoutingAgent("test", "Test", server=server_mock, categories=categories)
        
        # Mock für register_tool_categories
        agent.register_tool_categories = MagicMock()
        
        # Act
        agent._find_matching_tools("test request")
        
        # Assert
        agent.register_tool_categories.assert_called_once()


class TestSpecializedAgent:
    """Tests für die SpecializedAgent-Klasse."""
    
    def test_should_initialize_with_tool_names(self) -> None:
        """Test, dass SpecializedAgent mit Tool-Namen initialisiert werden kann."""
        # Arrange
        tool_names = ["tool1", "tool2"]
        
        # Act
        agent = SpecializedAgent("test", "Test", tool_names=tool_names)
        
        # Assert
        assert agent.tool_names == tool_names
        assert len(agent.tools) == 0
    
    def test_should_connect_to_server_and_load_tools(self) -> None:
        """Test, dass connect_to_server() auch die Tools lädt."""
        # Arrange
        tool_names = ["tool1", "tool2"]
        server_mock = MagicMock(spec=ME2AIMCPServer)
        server_mock.tool1 = MagicMock()
        server_mock.tool2 = MagicMock()
        
        agent = SpecializedAgent("test", "Test", tool_names=tool_names)
        
        # Mock für _load_tools
        original_load_tools = agent._load_tools
        agent._load_tools = MagicMock()
        
        # Act
        agent.connect_to_server(server_mock)
        
        # Assert
        assert agent.server == server_mock
        agent._load_tools.assert_called_once()
        
        # Original-Methode wiederherstellen und testen
        agent._load_tools = original_load_tools
        agent._load_tools()
        
        assert "tool1" in agent.tools
        assert "tool2" in agent.tools
    
    def test_should_load_only_specified_tools(self) -> None:
        """Test, dass _load_tools() nur die angegebenen Tools lädt."""
        # Arrange
        tool_names = ["tool1", "missing_tool"]
        server_mock = MagicMock(spec=ME2AIMCPServer)
        server_mock.tool1 = MagicMock()
        # missing_tool existiert nicht
        
        agent = SpecializedAgent("test", "Test", tool_names=tool_names)
        agent.server = server_mock
        
        # Act
        agent._load_tools()
        
        # Assert
        assert "tool1" in agent.tools
        assert "missing_tool" not in agent.tools
        assert len(agent.tools) == 1
    
    def test_should_handle_missing_server_in_load_tools(self) -> None:
        """Test, dass _load_tools() nichts tut, wenn kein Server verbunden ist."""
        # Arrange
        agent = SpecializedAgent("test", "Test", tool_names=["tool1"])
        agent.server = None
        
        # Act
        agent._load_tools()
        
        # Assert
        assert len(agent.tools) == 0
    
    def test_should_process_request_with_loaded_tools(self) -> None:
        """Test, dass process_request() die geladenen Tools verwendet."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = SpecializedAgent(
            "test", "Test",
            description="Test domain",
            server=server_mock, 
            tool_names=["tool1", "tool2"]
        )
        
        # Tools manuell setzen
        tool1_mock = MagicMock(return_value={"result1": "value1"})
        tool2_mock = MagicMock(return_value={"result2": "value2"})
        agent.tools = {
            "tool1": tool1_mock,
            "tool2": tool2_mock
        }
        
        # Act
        result = agent.process_request("Test request")
        
        # Assert
        assert result["success"] is True
        assert result["data"]["agent"] == "Test"
        assert result["data"]["domain"] == "Test domain"
        assert "tool1" in result["data"]["results"]
        assert "tool2" in result["data"]["results"]
        tool1_mock.assert_called_once()
        tool2_mock.assert_called_once()
    
    def test_should_load_tools_if_empty_during_process_request(self) -> None:
        """Test, dass process_request() _load_tools() aufruft, wenn tools leer ist."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = SpecializedAgent(
            "test", "Test", server=server_mock, tool_names=["tool1"]
        )
        
        # Mock für _load_tools
        agent._load_tools = MagicMock()
        
        # Act
        agent.process_request("Test request")
        
        # Assert
        agent._load_tools.assert_called_once()
    
    def test_should_handle_errors_in_tool_execution(self) -> None:
        """Test, dass process_request() Fehler in Tools abfängt."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = SpecializedAgent(
            "test", "Test", server=server_mock, tool_names=["tool1", "tool2"]
        )
        
        # Tools manuell setzen, eines wirft einen Fehler
        tool1_mock = MagicMock(return_value={"result1": "value1"})
        tool2_mock = MagicMock(side_effect=ValueError("Test error"))
        agent.tools = {
            "tool1": tool1_mock,
            "tool2": tool2_mock
        }
        
        # Mock für logger
        agent.logger = MagicMock()
        
        # Act
        result = agent.process_request("Test request")
        
        # Assert
        assert "tool1" in result["data"]["results"]
        assert "tool2" not in result["data"]["results"]
        agent.logger.error.assert_called_once()
    
    def test_should_handle_missing_server_in_process_request(self) -> None:
        """Test, dass process_request() einen Fehler zurückgibt, wenn kein Server verbunden ist."""
        # Arrange
        agent = SpecializedAgent("test", "Test", tool_names=["tool1"])
        agent.server = None
        
        # Act
        result = agent.process_request("Test request")
        
        # Assert
        assert result["success"] is False
        assert "error" in result
    
    def test_should_handle_no_tools_in_process_request(self) -> None:
        """Test, dass process_request() einen Fehler zurückgibt, wenn keine Tools verfügbar sind."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        agent = SpecializedAgent("test", "Test", server=server_mock)
        
        # Mock für _load_tools, die keine Tools lädt
        agent._load_tools = MagicMock()
        
        # Act
        result = agent.process_request("Test request")
        
        # Assert
        assert result["success"] is False
        assert "No tools available" in result["error"]


class TestDefaultCategories:
    """Tests für die DEFAULT_CATEGORIES-Konstante."""
    
    def test_should_have_expected_default_categories(self) -> None:
        """Test, dass DEFAULT_CATEGORIES die erwarteten Kategorien enthält."""
        # Act & Assert
        assert len(DEFAULT_CATEGORIES) >= 4  # Mindestens 4 Kategorien
        
        # Kategorien nach Namen finden
        category_names = [cat.name for cat in DEFAULT_CATEGORIES]
        assert "text_processing" in category_names
        assert "data_retrieval" in category_names
        assert "github" in category_names
        assert "system" in category_names
