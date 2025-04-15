#!/usr/bin/env python
"""
Agent Routing Example for ME2AI MCP

Dieses Beispiel demonstriert die vollständige Agent-Tool-Verbindungsschicht mit:
- Spezialisierten Agenten für verschiedene Aufgaben
- Automatisches Routing von Anfragen zu passenden Agenten
- Integration von Agenten mit MCP-Tools

Ausführen mit:
    python agent_routing_example.py
"""
from typing import Dict, List, Any, Optional
import os
import json
import logging
from datetime import datetime

from me2ai_mcp.base import ME2AIMCPServer
from me2ai_mcp.agents import (
    BaseAgent, RoutingAgent, SpecializedAgent, 
    ToolCategory, DEFAULT_CATEGORIES
)
from me2ai_mcp.routing import (
    RoutingRule, AgentRegistry, MCPRouter, 
    create_default_rules
)
from me2ai_mcp.auth import AuthManager


# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_routing.log")
    ]
)
logger = logging.getLogger("agent-routing-example")


class AgentRoutingServer(ME2AIMCPServer):
    """Server mit erweiterter Tool-Unterstützung für Agenten-Routing."""
    
    def __init__(
        self, 
        server_name: str = "agent_routing_server",
        description: str = "ME2AI MCP Server mit Agent-Routing-Unterstützung",
        version: str = "0.0.6"
    ) -> None:
        """Server initialisieren."""
        super().__init__(server_name, description, version)
        self.logger = logging.getLogger(f"me2ai-{server_name}")
        self.start_time = datetime.now()
        
        # Tools registrieren
        self._register_text_tools()
        self._register_data_tools()
        self._register_github_tools()
        self._register_system_tools()
    
    def _register_text_tools(self) -> None:
        """Text-Verarbeitungs-Tools registrieren."""
        
        @self.register_tool
        def analyze_text(text: str) -> Dict[str, Any]:
            """
            Text analysieren und Statistiken zurückgeben.
            
            Args:
                text: Der zu analysierende Text
                
            Returns:
                Textstatistiken
            """
            word_count = len(text.split())
            char_count = len(text)
            line_count = len(text.splitlines())
            
            return {
                "word_count": word_count,
                "character_count": char_count,
                "line_count": line_count,
                "average_word_length": char_count / max(word_count, 1)
            }
        
        @self.register_tool
        def transform_text(text: str, operation: str = "uppercase") -> Dict[str, Any]:
            """
            Text mit verschiedenen Operationen transformieren.
            
            Args:
                text: Der zu transformierende Text
                operation: Die durchzuführende Operation (uppercase, lowercase, reverse)
                
            Returns:
                Transformierter Text
            """
            if operation == "uppercase":
                result = text.upper()
            elif operation == "lowercase":
                result = text.lower()
            elif operation == "reverse":
                result = text[::-1]
            else:
                return {"error": f"Unbekannte Operation: {operation}"}
            
            return {
                "original": text,
                "operation": operation,
                "result": result
            }
        
        @self.register_tool
        def summarize_text(text: str, max_length: int = 100) -> Dict[str, Any]:
            """
            Text zusammenfassen.
            
            Args:
                text: Der Text, der zusammengefasst werden soll
                max_length: Maximale Länge der Zusammenfassung
                
            Returns:
                Zusammengefasster Text
            """
            words = text.split()
            if len(words) <= max_length // 5:  # Annahme: Durchschnittliches Wort ist 5 Zeichen lang
                return {
                    "original_length": len(text),
                    "summary": text,
                    "summarized": False
                }
            
            # Einfache Zusammenfassung: Ersten Teil behalten
            summary = " ".join(words[:max(1, max_length // 5)]) + "..."
            
            return {
                "original_length": len(text),
                "summary_length": len(summary),
                "summary": summary,
                "summarized": True,
                "reduction_percent": round((1 - len(summary) / len(text)) * 100)
            }
    
    def _register_data_tools(self) -> None:
        """Daten-Verarbeitungs-Tools registrieren."""
        
        self._data_store = {}
        
        @self.register_tool
        def store_data(key: str, value: Any) -> Dict[str, str]:
            """
            Daten im Server speichern.
            
            Args:
                key: Schlüssel für die Daten
                value: Der zu speichernde Wert
                
            Returns:
                Bestätigungsnachricht
            """
            self._data_store[key] = value
            return {
                "status": "success",
                "message": f"Daten mit Schlüssel '{key}' gespeichert"
            }
        
        @self.register_tool
        def retrieve_data(key: str) -> Dict[str, Any]:
            """
            Gespeicherte Daten abrufen.
            
            Args:
                key: Schlüssel der abzurufenden Daten
                
            Returns:
                Gespeicherte Daten oder Fehlermeldung
            """
            if key in self._data_store:
                return {
                    "status": "success",
                    "key": key,
                    "data": self._data_store[key]
                }
            else:
                return {
                    "status": "error",
                    "message": f"Keine Daten mit Schlüssel '{key}' gefunden"
                }
        
        @self.register_tool
        def list_stored_data() -> Dict[str, Any]:
            """
            Alle gespeicherten Datenschlüssel auflisten.
            
            Returns:
                Liste der Datenschlüssel
            """
            return {
                "count": len(self._data_store),
                "keys": list(self._data_store.keys())
            }
    
    def _register_github_tools(self) -> None:
        """GitHub-Integration-Tools registrieren."""
        
        @self.register_tool
        def search_github_repositories(query: str, limit: int = 5) -> Dict[str, Any]:
            """
            GitHub-Repositories nach einem Suchbegriff durchsuchen (Simulation).
            
            Args:
                query: Suchbegriff
                limit: Maximale Anzahl von Ergebnissen
                
            Returns:
                Simulierte Suchergebnisse
            """
            # Simulierte Ergebnisse - in einer echten Implementierung würde hier
            # die GitHub API aufgerufen werden
            simulated_results = [
                {
                    "name": f"repo-{i}-{query}",
                    "description": f"Ein Repository mit {query} Funktionalität",
                    "stars": 100 - i * 10,
                    "owner": f"user-{i}",
                    "url": f"https://github.com/user-{i}/repo-{i}-{query}"
                }
                for i in range(min(limit, 5))
            ]
            
            return {
                "query": query,
                "count": len(simulated_results),
                "results": simulated_results
            }
        
        @self.register_tool
        def get_github_repository_info(repo_name: str, owner: str) -> Dict[str, Any]:
            """
            Informationen zu einem bestimmten GitHub-Repository abrufen (Simulation).
            
            Args:
                repo_name: Name des Repositories
                owner: Eigentümer des Repositories
                
            Returns:
                Simulierte Repository-Informationen
            """
            # Simulierte Informationen
            return {
                "name": repo_name,
                "owner": owner,
                "description": f"Beschreibung von {owner}/{repo_name}",
                "stars": 42,
                "forks": 13,
                "language": "Python",
                "created_at": "2023-01-01T00:00:00Z",
                "url": f"https://github.com/{owner}/{repo_name}"
            }
    
    def _register_system_tools(self) -> None:
        """System-Tools registrieren."""
        
        @self.register_tool
        def get_server_status() -> Dict[str, Any]:
            """
            Server-Statusdaten abrufen.
            
            Returns:
                Server-Statusinformationen
            """
            uptime = datetime.now() - self.start_time
            tool_count = len(self.list_tools())
            
            return {
                "server_name": self.server_name,
                "version": self.version,
                "uptime_seconds": int(uptime.total_seconds()),
                "registered_tools": tool_count,
                "status": "running"
            }
        
        @self.register_tool
        def get_tool_list() -> Dict[str, Any]:
            """
            Liste aller verfügbaren Tools abrufen.
            
            Returns:
                Liste der registrierten Tools
            """
            tools = self.list_tools()
            
            categorized_tools = {}
            for tool in tools:
                # Einfache Kategorisierung basierend auf Präfixen
                if tool.startswith("analyze") or tool.startswith("transform") or tool.startswith("summarize"):
                    category = "text"
                elif tool.startswith("store") or tool.startswith("retrieve") or tool.startswith("list"):
                    category = "data"
                elif tool.startswith("github") or tool.startswith("search_github"):
                    category = "github"
                elif tool.startswith("get_server") or tool.startswith("get_tool"):
                    category = "system"
                else:
                    category = "other"
                
                if category not in categorized_tools:
                    categorized_tools[category] = []
                categorized_tools[category].append(tool)
            
            return {
                "total": len(tools),
                "by_category": categorized_tools
            }


def create_specialized_agents(server: ME2AIMCPServer) -> Dict[str, BaseAgent]:
    """
    Spezialisierte Agenten für verschiedene Aufgabenbereiche erstellen.
    
    Args:
        server: Der MCP-Server mit registrierten Tools
        
    Returns:
        Dictionary mit erstellten Agenten
    """
    # Text-Agent
    text_agent = SpecializedAgent(
        agent_id="text_agent",
        name="Text-Verarbeitungs-Agent",
        description="Spezialisiert auf Textanalyse und -transformation",
        server=server,
        tool_names=["analyze_text", "transform_text", "summarize_text"]
    )
    
    # Daten-Agent
    data_agent = SpecializedAgent(
        agent_id="data_agent",
        name="Daten-Management-Agent",
        description="Spezialisiert auf Datenspeicherung und -abruf",
        server=server,
        tool_names=["store_data", "retrieve_data", "list_stored_data"]
    )
    
    # GitHub-Agent
    github_agent = SpecializedAgent(
        agent_id="github_agent",
        name="GitHub-Integrations-Agent",
        description="Spezialisiert auf GitHub-Interaktionen",
        server=server,
        tool_names=["search_github_repositories", "get_github_repository_info"]
    )
    
    # System-Agent
    system_agent = SpecializedAgent(
        agent_id="system_agent",
        name="System-Management-Agent",
        description="Spezialisiert auf Serververwaltung und Monitoring",
        server=server,
        tool_names=["get_server_status", "get_tool_list"]
    )
    
    # Allgemeiner Routing-Agent als Fallback
    routing_agent = RoutingAgent(
        agent_id="general_agent",
        name="Allgemeiner Routing-Agent",
        description="Kann verschiedene Anfragen an passende Tools weiterleiten",
        server=server,
        categories=DEFAULT_CATEGORIES
    )
    
    return {
        "text_agent": text_agent,
        "data_agent": data_agent,
        "github_agent": github_agent,
        "system_agent": system_agent,
        "general_agent": routing_agent
    }


def main():
    """Hauptfunktion zum Ausführen des Agent-Routing-Beispiels."""
    logger.info("Agent-Routing-Beispiel wird gestartet")
    
    # Server erstellen
    server = AgentRoutingServer()
    logger.info(f"Server erstellt: {server.server_name} (v{server.version})")
    
    # Agenten erstellen
    agents = create_specialized_agents(server)
    logger.info(f"{len(agents)} Agenten erstellt")
    
    # Router erstellen
    router = MCPRouter(server)
    
    # Agenten registrieren
    for agent_id, agent in agents.items():
        make_default = (agent_id == "general_agent")
        router.register_agent(agent, make_default=make_default)
    
    # Routing-Regeln hinzufügen
    for rule in create_default_rules():
        router.add_routing_rule(rule)
    
    logger.info("Agent-Router konfiguriert und bereit")
    
    # Beispielanfragen verarbeiten
    example_requests = [
        "Bitte analysiere folgenden Text: Dies ist ein Beispieltext zur Demonstration.",
        "Speichere diese Daten unter dem Schlüssel 'test': {'name': 'Beispiel', 'wert': 42}",
        "Suche nach GitHub-Repositories mit dem Stichwort 'python'",
        "Wie ist der aktuelle Server-Status?",
        "Wandle diesen Text in Großbuchstaben um: Hallo Welt!"
    ]
    
    logger.info(f"Verarbeite {len(example_requests)} Beispielanfragen")
    
    for i, request in enumerate(example_requests):
        logger.info(f"Anfrage {i+1}: {request}")
        response = router.process_request(request)
        
        # Ergebnis anzeigen
        print(f"\n--- Anfrage {i+1} ---")
        print(f"Anfrage: {request}")
        print(f"Verarbeitet von: {response.get('_routing', {}).get('agent_name', 'Unbekannt')}")
        print("Ergebnis:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    
    # Statistiken anzeigen
    print("\n--- Agent-Statistiken ---")
    stats = router.get_agent_stats()
    for agent_id, agent_stats in stats.items():
        print(f"{agent_stats['name']}: {agent_stats['request_count']} Anfragen, {agent_stats['error_count']} Fehler")
    
    print("\n--- Routing-Statistiken ---")
    routing_stats = router.get_routing_stats()
    print(f"Gesamtanfragen: {routing_stats['total_requests']}")
    print("Agent-Verteilung:")
    for agent_id, count in routing_stats.get('agent_distribution', {}).items():
        agent_name = stats.get(agent_id, {}).get('name', agent_id)
        print(f"  {agent_name}: {count} Anfragen")
    
    logger.info("Agent-Routing-Beispiel abgeschlossen")
    print("\nDieses Beispiel zeigt, wie Agenten und Tools über die ME2AI MCP Verbindungsschicht kommunizieren.")


if __name__ == "__main__":
    main()
