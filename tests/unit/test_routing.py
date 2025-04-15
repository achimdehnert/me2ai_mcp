"""
Unit tests for ME2AI MCP routing module.

Diese Tests prüfen die Funktionalität der Routing-Schicht
mit 100% Coverage für das routing.py Modul.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any, Tuple

from me2ai_mcp.routing import (
    RoutingRule, AgentRegistry, MCPRouter, create_default_rules
)
from me2ai_mcp.agents import BaseAgent, ToolCategory
from me2ai_mcp.base import ME2AIMCPServer


class TestRoutingRule:
    """Tests für die RoutingRule-Klasse."""
    
    def test_should_initialize_with_correct_attributes(self) -> None:
        """Test, dass RoutingRule-Objekte korrekt initialisiert werden."""
        # Arrange
        pattern = r"test\s+pattern"
        agent_id = "test_agent"
        priority = 100
        description = "Test rule description"
        
        # Act
        rule = RoutingRule(pattern, agent_id, priority, description)
        
        # Assert
        assert rule.pattern == pattern
        assert rule.agent_id == agent_id
        assert rule.priority == priority
        assert rule.description == description
        assert hasattr(rule, "compiled_pattern")
    
    def test_should_match_when_pattern_matches_request(self) -> None:
        """Test, dass matches() true zurückgibt, wenn das Pattern übereinstimmt."""
        # Arrange
        rule = RoutingRule(r"github|repo", "github_agent", 100)
        
        # Act & Assert
        assert rule.matches("Check my github repository") is True
        assert rule.matches("Create a new repo") is True
        assert rule.matches("This contains GITHUB keyword") is True  # Case-insensitive
        assert rule.matches("No matching pattern here") is False
    
    def test_should_handle_complex_regex_patterns(self) -> None:
        """Test, dass matches() mit komplexen Regex-Patterns umgehen kann."""
        # Arrange
        rule = RoutingRule(r"^(text|process)\s+\w+", "text_agent", 100)
        
        # Act & Assert
        assert rule.matches("text processing example") is True
        assert rule.matches("process data") is True
        assert rule.matches("not at the beginning text") is False


class TestAgentRegistry:
    """Tests für die AgentRegistry-Klasse."""
    
    def test_should_initialize_empty(self) -> None:
        """Test, dass AgentRegistry leer initialisiert wird."""
        # Act
        registry = AgentRegistry()
        
        # Assert
        assert len(registry.agents) == 0
        assert len(registry.routing_rules) == 0
        assert registry.default_agent_id is None
    
    def test_should_register_agent(self) -> None:
        """Test, dass register_agent() einen Agenten korrekt registriert."""
        # Arrange
        registry = AgentRegistry()
        agent_mock = MagicMock(spec=BaseAgent)
        agent_mock.agent_id = "test_agent"
        agent_mock.name = "Test Agent"
        
        # Act
        registry.register_agent(agent_mock)
        
        # Assert
        assert "test_agent" in registry.agents
        assert registry.agents["test_agent"] == agent_mock
    
    def test_should_set_default_agent(self) -> None:
        """Test, dass register_agent() den Default-Agenten korrekt setzt."""
        # Arrange
        registry = AgentRegistry()
        agent1 = MagicMock(spec=BaseAgent)
        agent1.agent_id = "agent1"
        agent2 = MagicMock(spec=BaseAgent)
        agent2.agent_id = "agent2"
        
        # Act - Erst ohne make_default, dann mit make_default
        registry.register_agent(agent1)
        registry.register_agent(agent2, make_default=True)
        
        # Assert
        assert registry.default_agent_id == "agent2"
    
    def test_should_set_first_agent_as_default_if_none_specified(self) -> None:
        """Test, dass der erste Agent als Default gesetzt wird, wenn keiner angegeben ist."""
        # Arrange
        registry = AgentRegistry()
        agent = MagicMock(spec=BaseAgent)
        agent.agent_id = "first_agent"
        
        # Act
        registry.register_agent(agent)
        
        # Assert
        assert registry.default_agent_id == "first_agent"
    
    def test_should_add_routing_rule(self) -> None:
        """Test, dass add_routing_rule() eine Regel korrekt hinzufügt."""
        # Arrange
        registry = AgentRegistry()
        rule = RoutingRule("pattern", "agent_id", 100, "description")
        
        # Act
        registry.add_routing_rule(rule)
        
        # Assert
        assert len(registry.routing_rules) == 1
        assert registry.routing_rules[0] == rule
    
    def test_should_sort_rules_by_priority(self) -> None:
        """Test, dass add_routing_rule() die Regeln nach Priorität sortiert."""
        # Arrange
        registry = AgentRegistry()
        rule1 = RoutingRule("pattern1", "agent1", 100)
        rule2 = RoutingRule("pattern2", "agent2", 200)
        rule3 = RoutingRule("pattern3", "agent3", 50)
        
        # Act - Regeln in einer anderen Reihenfolge als ihre Priorität hinzufügen
        registry.add_routing_rule(rule1)
        registry.add_routing_rule(rule2)
        registry.add_routing_rule(rule3)
        
        # Assert - Regeln sollten nach absteigender Priorität sortiert sein
        assert registry.routing_rules[0] == rule2  # Priorität 200
        assert registry.routing_rules[1] == rule1  # Priorität 100
        assert registry.routing_rules[2] == rule3  # Priorität 50
    
    def test_should_route_to_matching_rule(self) -> None:
        """Test, dass route_request() den Agenten für die passende Regel zurückgibt."""
        # Arrange
        registry = AgentRegistry()
        
        # Agenten einrichten
        agent1 = MagicMock(spec=BaseAgent)
        agent1.agent_id = "agent1"
        agent2 = MagicMock(spec=BaseAgent)
        agent2.agent_id = "agent2"
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        # Regeln einrichten
        rule1 = RoutingRule("pattern1", "agent1", 100)
        rule2 = RoutingRule("github|repo", "agent2", 200)
        registry.add_routing_rule(rule1)
        registry.add_routing_rule(rule2)
        
        # Act
        matched_agent, sanitized_request = registry.route_request("Check my github repo")
        
        # Assert
        assert matched_agent == agent2
    
    def test_should_route_to_default_agent_if_no_match(self) -> None:
        """Test, dass route_request() zum Default-Agenten weiterleitet, wenn keine Regel passt."""
        # Arrange
        registry = AgentRegistry()
        
        # Agenten einrichten
        agent1 = MagicMock(spec=BaseAgent)
        agent1.agent_id = "agent1"
        agent2 = MagicMock(spec=BaseAgent)
        agent2.agent_id = "agent2"
        registry.register_agent(agent1)
        registry.register_agent(agent2, make_default=True)
        
        # Regeln einrichten, die nicht passen werden
        rule = RoutingRule("pattern_that_wont_match", "agent1", 100)
        registry.add_routing_rule(rule)
        
        # Act
        matched_agent, sanitized_request = registry.route_request("This won't match any rule")
        
        # Assert
        assert matched_agent == agent2  # Default-Agent
    
    def test_should_fall_back_to_first_agent_if_no_default(self) -> None:
        """Test, dass route_request() zum ersten Agenten weiterleitet, wenn kein Default gesetzt ist."""
        # Arrange
        registry = AgentRegistry()
        
        # Default-Agent löschen
        registry.default_agent_id = None
        
        # Agenten einrichten
        agent1 = MagicMock(spec=BaseAgent)
        agent1.agent_id = "agent1"
        agent2 = MagicMock(spec=BaseAgent)
        agent2.agent_id = "agent2"
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        # Regeln einrichten, die nicht passen werden
        rule = RoutingRule("pattern_that_wont_match", "agent_that_doesnt_exist", 100)
        registry.add_routing_rule(rule)
        
        # Act
        matched_agent, sanitized_request = registry.route_request("This won't match any rule")
        
        # Assert
        assert matched_agent == agent1  # Erster registrierter Agent
    
    def test_should_raise_error_if_no_agents_registered(self) -> None:
        """Test, dass route_request() einen Fehler wirft, wenn keine Agenten registriert sind."""
        # Arrange
        registry = AgentRegistry()
        
        # Act & Assert
        with pytest.raises(ValueError):
            registry.route_request("Test request")
    
    def test_should_sanitize_request(self) -> None:
        """Test, dass route_request() die Anfrage bereinigt."""
        # Arrange
        registry = AgentRegistry()
        
        # Agent einrichten
        agent = MagicMock(spec=BaseAgent)
        agent.agent_id = "agent"
        registry.register_agent(agent)
        
        # Act
        # Eine Anfrage mit unerwünschten Zeichen, die bereinigt werden sollten
        matched_agent, sanitized_request = registry.route_request("Test\x00request\x01")
        
        # Assert
        assert sanitized_request == "Testrequest"  # Bereinigte Version


class TestMCPRouter:
    """Tests für die MCPRouter-Klasse."""
    
    def test_should_initialize_with_server(self) -> None:
        """Test, dass MCPRouter mit einem Server initialisiert werden kann."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        
        # Act
        router = MCPRouter(server_mock)
        
        # Assert
        assert router.server == server_mock
        assert isinstance(router.registry, AgentRegistry)
        assert len(router.request_history) == 0
    
    def test_should_use_provided_registry(self) -> None:
        """Test, dass MCPRouter die bereitgestellte Registry verwendet."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        registry_mock = MagicMock(spec=AgentRegistry)
        
        # Act
        router = MCPRouter(server_mock, registry_mock)
        
        # Assert
        assert router.registry == registry_mock
    
    def test_should_register_agent_and_connect_to_server(self) -> None:
        """Test, dass register_agent() den Agenten mit dem Server verbindet und registriert."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        registry_mock = MagicMock(spec=AgentRegistry)
        router = MCPRouter(server_mock, registry_mock)
        
        agent_mock = MagicMock(spec=BaseAgent)
        
        # Act
        router.register_agent(agent_mock, make_default=True)
        
        # Assert
        agent_mock.connect_to_server.assert_called_once_with(server_mock)
        registry_mock.register_agent.assert_called_once_with(agent_mock, make_default=True)
    
    def test_should_add_routing_rule(self) -> None:
        """Test, dass add_routing_rule() die Regel zur Registry hinzufügt."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        registry_mock = MagicMock(spec=AgentRegistry)
        router = MCPRouter(server_mock, registry_mock)
        
        rule_mock = MagicMock(spec=RoutingRule)
        
        # Act
        router.add_routing_rule(rule_mock)
        
        # Assert
        registry_mock.add_routing_rule.assert_called_once_with(rule_mock)
    
    def test_should_process_request_and_record_history(self) -> None:
        """Test, dass process_request() die Anfrage weiterleitet und die Historie aktualisiert."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        registry_mock = MagicMock(spec=AgentRegistry)
        router = MCPRouter(server_mock, registry_mock)
        
        # Mock-Antworten einrichten
        agent_mock = MagicMock(spec=BaseAgent)
        agent_mock.agent_id = "test_agent"
        agent_mock.name = "Test Agent"
        agent_mock.process_request.return_value = {"result": "success"}
        
        registry_mock.route_request.return_value = (agent_mock, "sanitized request")
        
        # Act
        result = router.process_request("Test request")
        
        # Assert
        registry_mock.route_request.assert_called_once_with("Test request")
        agent_mock.process_request.assert_called_once()
        assert result == {"result": "success", "_routing": {"agent_id": "test_agent", "agent_name": "Test Agent"}}
        assert len(router.request_history) == 1
        assert router.request_history[0]["request"] == "Test request"
        assert router.request_history[0]["agent_id"] == "test_agent"
    
    def test_should_handle_error_in_process_request(self) -> None:
        """Test, dass process_request() Fehler abfängt."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        registry_mock = MagicMock(spec=AgentRegistry)
        router = MCPRouter(server_mock, registry_mock)
        
        # Eine Exception simulieren
        registry_mock.route_request.side_effect = ValueError("Test error")
        
        # Act
        result = router.process_request("Test request")
        
        # Assert
        assert result["success"] is False
        assert "Error processing request" in result["error"]
    
    def test_should_get_agent_stats(self) -> None:
        """Test, dass get_agent_stats() die korrekten Statistiken zurückgibt."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        router = MCPRouter(server_mock)
        
        # Agenten mit Statistiken einrichten
        agent1 = MagicMock(spec=BaseAgent)
        agent1.agent_id = "agent1"
        agent1.name = "Agent 1"
        agent1.description = "Test agent 1"
        agent1.request_count = 5
        agent1.error_count = 1
        
        agent2 = MagicMock(spec=BaseAgent)
        agent2.agent_id = "agent2"
        agent2.name = "Agent 2"
        agent2.description = "Test agent 2"
        agent2.request_count = 3
        agent2.error_count = 0
        
        # Agenten in die Registry einfügen
        router.registry.agents = {
            "agent1": agent1,
            "agent2": agent2
        }
        
        # Act
        stats = router.get_agent_stats()
        
        # Assert
        assert "agent1" in stats
        assert "agent2" in stats
        assert stats["agent1"]["name"] == "Agent 1"
        assert stats["agent1"]["request_count"] == 5
        assert stats["agent1"]["error_count"] == 1
        assert stats["agent2"]["request_count"] == 3
    
    def test_should_get_routing_stats(self) -> None:
        """Test, dass get_routing_stats() die korrekten Statistiken zurückgibt."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        router = MCPRouter(server_mock)
        
        # Request-Historie einrichten
        router.request_history = [
            {"request": "req1", "agent_id": "agent1", "response": {}},
            {"request": "req2", "agent_id": "agent2", "response": {}},
            {"request": "req3", "agent_id": "agent1", "response": {}}
        ]
        
        # Act
        stats = router.get_routing_stats()
        
        # Assert
        assert stats["total_requests"] == 3
        assert stats["agent_distribution"]["agent1"] == 2
        assert stats["agent_distribution"]["agent2"] == 1
    
    def test_should_handle_empty_history_in_routing_stats(self) -> None:
        """Test, dass get_routing_stats() mit leerer Historie umgehen kann."""
        # Arrange
        server_mock = MagicMock(spec=ME2AIMCPServer)
        router = MCPRouter(server_mock)
        router.request_history = []
        
        # Act
        stats = router.get_routing_stats()
        
        # Assert
        assert stats["total_requests"] == 0
        assert "agent_distribution" not in stats


class TestCreateDefaultRules:
    """Tests für die create_default_rules-Funktion."""
    
    def test_should_create_default_routing_rules(self) -> None:
        """Test, dass create_default_rules() die erwarteten Standard-Regeln erstellt."""
        # Act
        rules = create_default_rules()
        
        # Assert
        assert len(rules) >= 4  # Mindestens 4 Standardregeln
        
        # Regeln nach agent_id sortieren
        rule_ids = [rule.agent_id for rule in rules]
        assert "github_agent" in rule_ids
        assert "text_agent" in rule_ids
        assert "data_agent" in rule_ids
        assert "system_agent" in rule_ids
        
        # Prioritäten prüfen
        for rule in rules:
            if rule.agent_id == "github_agent":
                assert rule.priority == 100
            elif rule.agent_id == "text_agent":
                assert rule.priority == 90
