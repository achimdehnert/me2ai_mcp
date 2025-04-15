"""
Integration tests for ME2AI MCP agent routing functionality.

Diese Tests prüfen die Zusammenarbeit der Routing-Komponenten
unter realen Bedingungen mit vollständiger Integration.
"""
import pytest
from unittest.mock import MagicMock, patch

from me2ai_mcp.base import ME2AIMCPServer, BaseTool
from me2ai_mcp.agents import (
    BaseAgent, RoutingAgent, SpecializedAgent, 
    ToolCategory, DEFAULT_CATEGORIES
)
from me2ai_mcp.routing import (
    RoutingRule, AgentRegistry, MCPRouter, create_default_rules
)


class TestTool(BaseTool):
    """Test-Tool für Integrationstests."""
    
    def __init__(self, name: str, category: str):
        """Initialisiert ein Test-Tool mit Namen und Kategorie."""
        self.name = name
        self.category = category
        self.call_count = 0
    
    def __call__(self, request: str) -> dict:
        """Ruft das Tool auf und gibt ein Ergebnis zurück."""
        self.call_count += 1
        return {
            "tool": self.name,
            "category": self.category,
            "request": request,
            "call_count": self.call_count
        }


class TestAgentRoutingIntegration:
    """Integrationstest für Agent-Routing-Funktionalität."""
    
    @pytest.fixture
    def setup_server_with_tools(self):
        """Fixture zum Erstellen eines Servers mit Test-Tools."""
        server = ME2AIMCPServer("test_server")
        
        # Text-Tools hinzufügen
        server.process_text = TestTool("process_text", "text")
        server.summarize_text = TestTool("summarize_text", "text")
        
        # Daten-Tools hinzufügen
        server.store_data = TestTool("store_data", "data")
        server.retrieve_data = TestTool("retrieve_data", "data")
        
        # GitHub-Tools hinzufügen
        server.search_repos = TestTool("search_repos", "github")
        server.list_issues = TestTool("list_issues", "github")
        
        # System-Tools hinzufügen
        server.get_system_info = TestTool("get_system_info", "system")
        
        return server
    
    def test_should_route_requests_to_correct_specialized_agents(self, setup_server_with_tools):
        """Test, dass Anfragen an die richtigen spezialisierten Agenten weitergeleitet werden."""
        # Arrange
        server = setup_server_with_tools
        router = MCPRouter(server)
        
        # Spezialisierte Agenten erstellen und registrieren
        text_agent = SpecializedAgent(
            "text_agent", "Text Agent",
            "Specializes in text processing",
            tool_names=["process_text", "summarize_text"]
        )
        
        data_agent = SpecializedAgent(
            "data_agent", "Data Agent",
            "Specializes in data operations",
            tool_names=["store_data", "retrieve_data"]
        )
        
        github_agent = SpecializedAgent(
            "github_agent", "GitHub Agent",
            "Specializes in GitHub operations",
            tool_names=["search_repos", "list_issues"]
        )
        
        router.register_agent(text_agent)
        router.register_agent(data_agent)
        router.register_agent(github_agent, make_default=True)
        
        # Routing-Regeln hinzufügen
        router.add_routing_rule(RoutingRule(
            r"text|process|summarize", "text_agent", 100,
            "Route text-related requests to the Text Agent"
        ))
        
        router.add_routing_rule(RoutingRule(
            r"data|store|retrieve", "data_agent", 90,
            "Route data-related requests to the Data Agent"
        ))
        
        router.add_routing_rule(RoutingRule(
            r"github|repo|issue", "github_agent", 80,
            "Route GitHub-related requests to the GitHub Agent"
        ))
        
        # Act - Verschiedene Anfragen verarbeiten
        text_result = router.process_request("Process this text for me")
        data_result = router.process_request("Store some data in the database")
        github_result = router.process_request("List all issues in my repository")
        
        # Assert - Überprüfen, ob die richtigen Agenten verwendet wurden
        assert text_result["_routing"]["agent_id"] == "text_agent"
        assert data_result["_routing"]["agent_id"] == "data_agent"
        assert github_result["_routing"]["agent_id"] == "github_agent"
        
        # Überprüfen, ob die richtigen Tools aufgerufen wurden
        assert "process_text" in text_result["results"]
        assert "store_data" in data_result["results"]
        assert "list_issues" in github_result["results"]
        
        # Routing-Statistiken überprüfen
        stats = router.get_routing_stats()
        assert stats["total_requests"] == 3
        assert stats["agent_distribution"]["text_agent"] == 1
        assert stats["agent_distribution"]["data_agent"] == 1
        assert stats["agent_distribution"]["github_agent"] == 1
    
    def test_should_route_to_default_agent_when_no_rules_match(self, setup_server_with_tools):
        """Test, dass Anfragen zum Default-Agenten weitergeleitet werden, wenn keine Regel passt."""
        # Arrange
        server = setup_server_with_tools
        router = MCPRouter(server)
        
        # Agenten erstellen und registrieren
        default_agent = SpecializedAgent(
            "default_agent", "Default Agent",
            tool_names=["get_system_info"]
        )
        
        other_agent = SpecializedAgent(
            "other_agent", "Other Agent",
            tool_names=["process_text"]
        )
        
        router.register_agent(other_agent)
        router.register_agent(default_agent, make_default=True)
        
        # Regel hinzufügen, die nicht passen wird
        router.add_routing_rule(RoutingRule(
            r"specific_pattern_that_wont_match", "other_agent", 100
        ))
        
        # Act
        result = router.process_request("This is a generic request")
        
        # Assert
        assert result["_routing"]["agent_id"] == "default_agent"
        assert "get_system_info" in result["results"]
    
    def test_should_handle_routing_agent_with_automatic_tool_discovery(self, setup_server_with_tools):
        """Test, dass ein RoutingAgent Tools automatisch anhand der Anfrage finden kann."""
        # Arrange
        server = setup_server_with_tools
        router = MCPRouter(server)
        
        # Routing-Agent mit Standard-Kategorien erstellen
        routing_agent = RoutingAgent(
            "auto_agent", "Auto-Discovery Agent",
            "Automatically discovers appropriate tools",
            categories=DEFAULT_CATEGORIES
        )
        
        router.register_agent(routing_agent)
        
        # Act - Verschiedene Anfragen verarbeiten
        text_result = router.process_request("Process this text for me")
        github_result = router.process_request("Search for repositories")
        
        # Assert
        assert text_result["_routing"]["agent_id"] == "auto_agent"
        assert github_result["_routing"]["agent_id"] == "auto_agent"
        
        # Text-Anfrage sollte Text-Tools finden
        assert "results" in text_result["data"]
        assert "process_text" in text_result["data"]["results"]
        
        # GitHub-Anfrage sollte GitHub-Tools finden
        assert "results" in github_result["data"]
        assert "search_repos" in github_result["data"]["results"]
    
    def test_should_integrate_with_full_routing_stack(self, setup_server_with_tools):
        """Test der vollständigen Routing-Stack-Integration."""
        # Arrange
        server = setup_server_with_tools
        
        # Vollständige Routing-Konfiguration erstellen
        router = MCPRouter(server)
        
        # Spezialisierte Agenten registrieren
        for agent_type in ["text", "data", "github", "system"]:
            # Tool-Namen für jeden Agententyp finden
            tool_names = []
            for attr_name in dir(server):
                if not attr_name.startswith("_") and callable(getattr(server, attr_name)):
                    tool = getattr(server, attr_name)
                    if hasattr(tool, "category") and tool.category == agent_type:
                        tool_names.append(attr_name)
            
            # Spezialisierten Agenten erstellen und registrieren
            agent = SpecializedAgent(
                f"{agent_type}_agent",
                f"{agent_type.capitalize()} Agent",
                f"Specializes in {agent_type} operations",
                tool_names=tool_names
            )
            router.register_agent(agent)
        
        # Standard-Routing-Regeln hinzufügen
        for rule in create_default_rules():
            router.add_routing_rule(rule)
        
        # Act - Verschiedene Anfragen verarbeiten
        results = {}
        requests = {
            "text": "Process and summarize this text",
            "data": "Store and retrieve this data",
            "github": "Search for GitHub repositories and list issues",
            "system": "Get system information"
        }
        
        for req_type, request in requests.items():
            results[req_type] = router.process_request(request)
        
        # Assert - Überprüfen, ob alle Anfragen korrekt weitergeleitet wurden
        assert results["text"]["_routing"]["agent_id"] == "text_agent"
        assert results["data"]["_routing"]["agent_id"] == "data_agent"
        assert results["github"]["_routing"]["agent_id"] == "github_agent"
        assert results["system"]["_routing"]["agent_id"] == "system_agent"
        
        # Agent-Statistiken überprüfen
        stats = router.get_agent_stats()
        for agent_id in ["text_agent", "data_agent", "github_agent", "system_agent"]:
            assert agent_id in stats
            assert stats[agent_id]["request_count"] == 1
        
        # Routing-Statistiken überprüfen
        routing_stats = router.get_routing_stats()
        assert routing_stats["total_requests"] == 4
        assert len(routing_stats["agent_distribution"]) == 4
    
    def test_should_handle_errors_gracefully(self, setup_server_with_tools):
        """Test, dass Fehler in Tools oder Agenten ordnungsgemäß behandelt werden."""
        # Arrange
        server = setup_server_with_tools
        
        # Fehlerhafte Tool-Methode hinzufügen
        def error_tool(request):
            raise ValueError("Simulated tool error")
        
        server.error_tool = error_tool
        
        # Router mit fehlerhaftem Agent erstellen
        router = MCPRouter(server)
        
        # Agent mit fehlerhaftem Tool registrieren
        error_agent = SpecializedAgent(
            "error_agent", "Error Agent",
            tool_names=["error_tool"]
        )
        router.register_agent(error_agent)
        
        # Routing-Regel hinzufügen
        router.add_routing_rule(RoutingRule(
            r"error|fail", "error_agent", 100
        ))
        
        # Act
        result = router.process_request("This should trigger an error")
        
        # Assert
        assert result["_routing"]["agent_id"] == "error_agent"
        assert result["success"] is True  # Gesamtanfrage sollte erfolgreich sein
        assert "errors" in result["data"]  # Aber mit Fehlern im Ergebnis
