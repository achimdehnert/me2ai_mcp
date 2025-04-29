# ME2AI MCP - Self-Contained Model Context Protocol Framework

ME2AI MCP is a self-contained Model Context Protocol framework that provides standardized patterns, tools, and utilities for building robust AI applications. Version 0.2.0 introduces a highly modularized architecture with specialized handlers for vector stores and embeddings, comprehensive Robot Framework test integration, and enhanced provider-agnostic design for easier integration with applications like the ME2AI Knowledge Assistant.

## Overview

ME2AI MCP ist ein selbstständiges Framework, das eine vollständige, unabhängige Implementierung des Model Context Protocol bietet. Die Architektur folgt einer sauberen, modularen Struktur mit klarer Trennung der Verantwortlichkeiten:

* **Core**: Zentrale Abstraktionen, Protokolle und Basisklassen
* **Clients**: Implementierungen für verschiedene MCP-Server (OpenAI, Anthropic, Groq, ME2AI, etc.)
* **Adapters**: Austauschbare Komponenten für Vector Stores, Embedding-Provider und externe Integrationen
* **Services**: Höherwertige Dienste mit spezialisierten Handlern für Vector-Store-Management und Embedding
* **Handlers**: Spezialisierte Handler für verschiedene Backends (ChromaDB, FAISS, Qdrant, OpenAI, etc.)
* **Utils**: Hilfsfunktionen für Text, Netzwerk, Konfiguration und mehr

Die Version 0.2.0 führt eine vollständig modularisierte Handler-Architektur ein und verbessert die Testbarkeit durch umfassende Robot Framework Integration.

## Features

### Core Architecture (v0.2.0)

* **Modularisierte Handler-Architektur**: Spezialisierte Handler für verschiedene Backends und Services
* **Protokoll- und Interface-basiertes Design**: Sauber definierte Protokolle für alle Komponenten
* **Provider-agnostische Strategien**: Austauschbare Strategien für Embedding und Vector Stores
* **Robuste Fehlerbehandlung**: Einheitliche Fehlerklassenhierarchie und verbesserte Logging-Struktur
* **Pydantic-basierte Typsicherheit**: Validierung und Typsicherheit für alle Datenstrukturen
* **Umgebungsvariablen-Unterstützung**: Konfiguration über Umgebungsvariablen oder Parameter
* **Comprehensive Testing**: Robot Framework Testsuite für alle Komponenten

### Universal Client System

* **Einheitliche Client-Schnittstelle**: Generischer Zugriff auf verschiedene MCP-Server
* **Client-Registrierung**: Dynamische Registrierung und Discovery von Client-Implementierungen
* **Multi-Provider-Support**: Unterstützung für OpenAI, Anthropic, Groq, ME2AI, lokale Server
* **Automatische Discovery**: Erkennung verfügbarer MCP-Server in der Umgebung
* **Kontextmanager-Unterstützung**: Ressourcenverwaltung mit `async with`-Syntax

### Vector Store System

* **Spezialisierte Handler-Struktur**: Dedizierte Handler für ChromaDB, FAISS, Qdrant, Pinecone
* **Einheitliche Fehlerbehandlung**: Konsistente Fehlerklassen über alle Backends hinweg
* **Verbesserte Metadaten-Filterung**: Erweiterte Filtermöglichkeiten für präzise Abfragen
* **Chunking & Verarbeitung**: Integrierte Text-Extraktion und Dokumentenverarbeitung
* **Robot Framework Tests**: Umfassende Testsuite für alle Vector Store Backends

### Embedding System

* **EmbeddingStrategy Protocol**: Klare Schnittstelle für alle Embedding-Provider
* **Factory-basierte Instanziierung**: Flexible `create_embedding_strategy`-Factory
* **Spezialisierte Handler**: Dedizierte Handler für OpenAI, Sentence Transformers, Cohere, HuggingFace
* **Lokale Modelle**: Verbesserte Unterstützung für SentenceTransformers mit Model-Caching
* **Cloud-Provider**: Integration mit OpenAI, Cohere und anderen Cloud-Embedding-Diensten
* **Robot Framework Tests**: Umfassende Testsuite für alle Embedding-Provider

### Tool Routing & Orchestration

* **Tool Router Service**: Intelligentes Routing von Tool-Aufrufen an verschiedene Provider
* **Provider-Priorisierung**: Konfigurierbare Präferenzen für verschiedene Tool-Typen
* **Parallele Ausführung**: Asynchrone Ausführung von Tools über mehrere Provider
* **Fehlertoleranz**: Automatische Wiederholungsversuche und Failover-Mechanismen

### Database Integrations (v0.1.1)
- **Flexible Database Credentials**: Support for environment variables, JSON files, and multiple connection specifications
- **PostgreSQL Integration**: Comprehensive PostgreSQL support with connection pooling and schema validation
- **MySQL Integration**: Robust MySQL tools with connection pooling and enhanced error handling
- **LangChain Compatibility**: Ready-to-use LangChain tool adapters for all database components
- **Comprehensive Test Suite**: Test coverage across database integrations and framework components

## Installation

```bash
# Install from PyPI (recommended)
pip install me2ai_mcp

# Install specific version
pip install me2ai_mcp==0.1.1

# Install from GitHub
pip install git+https://github.com/achimdehnert/me2ai_mcp.git

# Install with all optional dependencies
pip install me2ai_mcp[all]

# Install with specific feature set
pip install me2ai_mcp[web]

# Install with database support
pip install me2ai_mcp[db]

# Install PostgreSQL support only
pip install me2ai_mcp[postgres]

# Install MySQL support only
pip install me2ai_mcp[mysql]

# Install embedding support
pip install me2ai_mcp[embeddings]

# Install LLM support
pip install me2ai_mcp[llm]

# Install Robot Framework testing support
pip install me2ai_mcp[robotframework]
```

Alternatively, install directly from GitHub:

```bash
# Install from GitHub
pip install git+https://github.com/achimdehnert/me2ai_mcp.git
```

See [INSTALLATION.md](INSTALLATION.md) for detailed installation options.

## Agent-Tool Routing Layer

The Agent-Tool Routing Layer introduced in v0.0.8 enables intelligent routing of requests to specialized agents based on request patterns and tool categories:

### Main Components

- **BaseAgent**: Base class for all agent implementations
- **RoutingAgent**: Agent with dynamic tool selection based on requests
- **SpecializedAgent**: Domain-specific agent for dedicated toolsets
- **MCPRouter**: Central component for request routing and agent management

### Example: Agent-Routing System

```python
from me2ai_mcp import ME2AIMCPServer, SpecializedAgent, MCPRouter, RoutingRule

# Create server with tools
server = ME2AIMCPServer("routing_example")

@server.register_tool
def process_text(text):
    """Process text."""
    return {"processed": text.upper()}

@server.register_tool
def store_data(data):
    """Store data."""
    return {"stored": True, "data": data}

# Create router
router = MCPRouter(server)

# Register specialized agents
text_agent = SpecializedAgent(
    "text_agent", "Text Agent", tool_names=["process_text"]
)
data_agent = SpecializedAgent(
    "data_agent", "Data Agent", tool_names=["store_data"]
)

router.register_agent(text_agent)
router.register_agent(data_agent)

# Add routing rules
router.add_routing_rule(RoutingRule("text|process", "text_agent", 100))
router.add_routing_rule(RoutingRule("data|store", "data_agent", 90))

# Process requests
result_text = router.process_request("Process this text")
result_data = router.process_request("Store some data")
```

For more details and examples, see the examples in the `examples/` directory.

## Quick Start

```python
from me2ai_mcp.base import ME2AIMCPServer

# Create a new MCP server
server = ME2AIMCPServer(
    server_name="my_server",
    description="Example ME2AI MCP Server",
    version="1.0.0"
)

# Register a tool
@server.register_tool
def process_data(input_text: str):
    """Process input data and return results."""
    return {
        "processed": input_text.upper(),
        "length": len(input_text)
    }

# Execute the tool
result = server.execute_tool("process_data", {"input_text": "hello world"})
print(result)  # {'processed': 'HELLO WORLD', 'length': 11}
```

## Testing with Robot Framework

ME2AI MCP now includes comprehensive testing with Robot Framework:

```bash
# Setup the Robot Framework environment
python setup_robot_environment.py

# Run all tests
robot --outputdir results tests/robot/

# Run specific test categories
robot --outputdir results --include knowledge-assistant tests/robot/
robot --outputdir results --include embedding tests/robot/
robot --outputdir results --include openai tests/robot/
```

For more details on the test structure and how to run tests, see the [Robot Framework README](tests/robot/README.md).

## Integration Guides

Detailed integration guides for common use cases:

- [Knowledge Assistant Integration Guide](docs/integration_guides/knowledge_assistant_integration.md): How to integrate ME2AI MCP with Knowledge Assistant applications

## Core Components

### Base Server

The `ME2AIMCPServer` class provides the foundation for all ME2AI MCP servers with:

- Automatic tool registration and discovery
- Consistent error handling and logging
- Tool execution statistics tracking
- Standard response formatting

### Authentication

The `AuthManager` class provides standardized authentication handling:

- Environment variable based token management
- Support for multiple token sources
- Token validation and verification

### Tools

Pre-built tools for common operations:

- Web content fetching and HTML parsing
- File system operations
- GitHub repository operations

## Examples

See the `examples/` directory for detailed implementation examples:

- `basic_server.py` - Simple MCP server implementation
- `github_mcp_server.py` - GitHub integration example

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=me2ai_mcp
```

## Development

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Follow ME2AI coding standards for contributions

## Documentation

For detailed documentation on components and usage, see the docstrings in the code.

## License

MIT License

## Contact

- Project maintained by ME2AI Team
- Email: info@me2ai.dev
- GitHub: https://github.com/achimdehnert/me2ai_mcp

### LLM Providers
- **OpenAI**: Utilizing GPT models for advanced language understanding
- **Groq**: High-performance inference with Mixtral-8x7b model
- **Anthropic**: Claude models for enhanced reasoning capabilities

### Expert Agents

#### German Professor
- Language learning and cultural guidance
- Tools:
  - German dictionary lookup
  - Grammar checking
  - Text translation
  - Cultural research

#### Dating Expert
- Relationship advice and interpersonal skills
- Tools:
  - Dating profile analysis
  - Conversation pattern evaluation
  - Relationship research

#### SEO Expert
- Search engine optimization strategies
- Tools:
  - Website SEO analysis
  - Keyword research
  - Competition analysis

#### Researcher
- Academic research and analysis
- Tools:
  - Comprehensive search (Web, Wikipedia, Academic papers)
  - Data analysis with Python
  - Citation generation
  - Research summarization

#### Life Coach
- Personal development and goal setting
- Tools:
  - Goal setting templates
  - Mindfulness exercises
  - Progress tracking

#### Moderator
- Conversation management and guidance
- Tools:
  - Conversation flow management
  - Topic suggestion
  - Conflict resolution

### System Features
- **Enhanced Memory Management**: Persistent conversation history for contextual responses
- **Interactive CLI**: User-friendly command-line interface
- **Flexible Architecture**: Easily extensible for new agents and tools
- **Automatic Routing**: Smart query routing to the most appropriate expert
- **Comprehensive Testing**: Unit, integration, performance, and load tests

## Installation

### Basic Installation
```bash
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## Environment Setup

Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

### Starting the CLI
```bash
python -m me2ai
```

### Available Commands
- `talk <message>`: Send a message to the current agent
- `switch <agent>`: Switch to a different agent
- `auto <message>`: Let the router automatically select the best expert
- `list`: Show available agents
- `clear`: Clear conversation history
- `help`: Show help message
- `quit`: Exit the program

### Example Interactions

#### Automatic Expert Selection
```
You> auto How do I optimize my website for search engines?
Routing your question...
Selected expert: SEO Expert
Reason: Query relates to website optimization

SEO Expert: Let me analyze your website's SEO factors...
[Uses SEO analysis tools to provide recommendations]
```

#### Research Query
```
You> auto What are the latest developments in quantum computing?
Routing your question...
Selected expert: Researcher
Reason: Query requires academic research

Researcher: Let me research this comprehensively...
[Uses multiple research tools to provide cited findings]
```

## Development

### Project Structure
```
me2ai/
├── agents/             # Agent implementations
│   ├── base.py        # Base agent interface
│   ├── coaching_agents.py
│   ├── expert_agents.py
│   ├── routing_agent.py
│   └── factory.py
├── tools/             # Specialized tools
│   ├── web_tools.py
│   ├── language_tools.py
│   ├── dating_tools.py
│   └── research_tools.py
├── llms/              # LLM providers
│   ├── base.py
│   ├── openai_provider.py
│   ├── groq_provider.py
│   └── anthropic_provider.py
├── tests/             # Test suite
└── cli.py            # CLI implementation
```

### Adding New Tools
1. Create a new tool class implementing the Tool protocol
2. Add tool-specific dependencies to pyproject.toml
3. Update relevant agent to use the new tool

### Adding New Agents
1. Create a new agent class inheriting from BaseAgent
2. Add agent-specific tools and system prompt
3. Update factory.py to support the new agent
4. Update router agent to recognize the new expertise

### Running Tests

#### Basic Test Suite
```bash
pytest tests/ -v
```

#### With Coverage Report
```bash
pytest tests/ -v --cov=. --cov-report=html
```

#### Specific Test Categories
```bash
# Run only integration tests
pytest tests/ -m integration

# Run only performance tests
pytest tests/ -m performance

# Run only load tests
pytest tests/ -m load

# Run only slow tests
pytest tests/ -m slow
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

### Code Quality Standards
- Use type hints
- Follow Google style docstrings
- Maintain test coverage
- Keep code modular and extensible

## Current Development Status

#### Test Status (as of 2024-12-22)
- ✅ CLI Tests: All 19 tests passing
- ❌ Load Tests: 4 tests failing
  - `test_moderate_load`
  - `test_heavy_load`
  - `test_mixed_agent_load`
  - `test_memory_load`

#### Recent Changes
1. **Agent Initialization**
   - Fixed agent initialization to properly handle system prompts and roles
   - Implemented memory management in expert agents
   - Added proper async support for agent responses

2. **Expert Agent Updates**
   - German Professor: Enhanced language learning capabilities
   - Dating Expert: Improved relationship advice system
   - SEO Expert: Added technical SEO analysis tools

3. **Code Quality**
   - Current test coverage: 37%
   - Improved error handling in agent factory
   - Enhanced async/await patterns in CLI

#### Known Issues
1. **Load Tests**
   - Memory initialization issues in load tests
   - Need to improve async handling in high-load scenarios

2. **Performance**
   - Some response delays under heavy load
   - Memory usage optimization needed

#### Next Steps
1. Fix load test failures
2. Improve test coverage
3. Optimize memory management
4. Enhance error handling

## License

MIT License - See LICENSE file for details
