# ME2AI API Documentation

## Core Components

### BaseAgent

Base class for all agents in the system.

```python
class BaseAgent:
    def __init__(
        self,
        role: str,
        system_prompt: str,
        llm_provider: LLMProvider,
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[Tool]] = None
    )
```

#### Methods
- `respond(message: str) -> str`: Generate response to user message
- `use_tool(tool_name: str, **kwargs: Any) -> str`: Use a specific tool
- `get_available_tools() -> List[str]`: Get list of available tools

### Tool Protocol

Interface for all tools in the system.

```python
class Tool(Protocol):
    name: str
    description: str
    
    def run(self, **kwargs: Any) -> str:
        """Run the tool with given arguments."""
```

## Tools

### Web Tools

#### WebSearchTool
- **Purpose**: Search the web using DuckDuckGo
- **Methods**: `run(query: str, max_results: int = 3) -> str`

#### TranslationTool
- **Purpose**: Translate text between languages
- **Methods**: `run(text: str, source_lang: str = "auto", target_lang: str = "en") -> str`

#### SEOAnalysisTool
- **Purpose**: Analyze websites for SEO factors
- **Methods**: `run(url: str) -> str`

### Language Tools

#### GermanDictionaryTool
- **Purpose**: Look up German words and meanings
- **Methods**: `run(word: str) -> str`

#### GrammarCheckerTool
- **Purpose**: Check German grammar
- **Methods**: `run(text: str) -> str`

### Dating Tools

#### ProfileAnalyzerTool
- **Purpose**: Analyze dating profiles
- **Methods**: `run(profile_text: str) -> str`

#### ConversationAnalyzerTool
- **Purpose**: Analyze conversation patterns
- **Methods**: `run(conversation: str) -> str`

### Research Tools

#### ComprehensiveSearchTool
- **Purpose**: Search across multiple sources
- **Methods**: `run(query: str, max_results: int = 3) -> str`
- **Sources**: Web, Wikipedia, Academic papers

#### DataAnalysisTool
- **Purpose**: Execute Python code for data analysis
- **Methods**: `run(code: str) -> str`

#### CitationTool
- **Purpose**: Generate academic citations
- **Methods**: `run(metadata: Dict[str, str], style: str = "apa") -> str`

#### ResearchSummaryTool
- **Purpose**: Summarize research content
- **Methods**: `run(content: str, max_length: int = 500) -> str`

## Expert Agents

### GermanProfessor
- **Role**: German language and culture expert
- **Tools**: GermanDictionaryTool, GrammarCheckerTool, TranslationTool, WebSearchTool

### DatingExpert
- **Role**: Dating and relationship advisor
- **Tools**: ProfileAnalyzerTool, ConversationAnalyzerTool, WebSearchTool

### SEOExpert
- **Role**: Search engine optimization specialist
- **Tools**: SEOAnalysisTool, WebSearchTool

### Researcher
- **Role**: Academic research and analysis expert
- **Tools**: ComprehensiveSearchTool, DataAnalysisTool, CitationTool, ResearchSummaryTool

## LLM Providers

### OpenAIProvider
- **Models**: GPT-4, GPT-3.5-turbo
- **Config**: Uses OPENAI_API_KEY

### GroqProvider
- **Models**: Mixtral-8x7b
- **Config**: Uses GROQ_API_KEY

### AnthropicProvider
- **Models**: Claude
- **Config**: Uses ANTHROPIC_API_KEY

## CLI Interface

### Commands
- `talk <message>`: Send message to current agent
- `switch <agent>`: Switch to different agent
- `auto <message>`: Auto-select best expert
- `list`: Show available agents
- `clear`: Clear conversation history
- `help`: Show help message
- `quit`: Exit program

### Example Usage

```python
from me2ai.cli import AgentCLI

# Start CLI
cli = AgentCLI()
cli.cmdloop()

# Manual agent usage
from me2ai.agents.factory import create_expert_agent
from me2ai.llms.openai_provider import OpenAIProvider

# Create an agent
researcher = create_expert_agent(
    role="researcher",
    llm_provider=OpenAIProvider(),
    memory=None
)

# Use the agent
response = researcher.respond("What are the latest developments in quantum computing?")
```

## Adding New Components

### Creating New Tools
```python
from me2ai.tools import Tool

class MyNewTool:
    name = "my_tool"
    description = "Description of what the tool does"
    
    def run(self, **kwargs: Any) -> str:
        # Tool implementation
        pass
```

### Creating New Agents
```python
from me2ai.agents.base import BaseAgent

class MyNewAgent(BaseAgent):
    def __init__(self, llm_provider, memory=None):
        super().__init__(
            role="My Agent",
            system_prompt="Agent behavior description",
            llm_provider=llm_provider,
            memory=memory,
            tools=[MyNewTool()]
        )
```

## Error Handling

The system includes comprehensive error handling:
- Tool execution errors return error messages
- Invalid commands show usage help
- LLM provider errors are caught and reported
- Memory errors maintain system stability

## Testing

### Test Categories
- Unit tests for individual components
- Integration tests for component interaction
- Performance tests for response times
- Load tests for system stability

### Running Tests
```bash
# All tests
pytest tests/

# Specific categories
pytest tests/ -m integration
pytest tests/ -m performance
pytest tests/ -m load
```
