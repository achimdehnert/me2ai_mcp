# ME2AI MCP Installation Guide

The ME2AI MCP package extends the official MCP package with enhanced functionality for building robust MCP servers.

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
# Install latest version
pip install me2ai_mcp

# Install specific version
pip install me2ai_mcp==0.0.5

# Install with extras
pip install me2ai_mcp[web,github]
```

### Method 2: Install from GitHub

For contributing to the ME2AI MCP package:

```bash
# Clone the repository
git clone https://github.com/achimdehnert/me2ai.git
cd me2ai/me2ai_mcp

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
```

## Verification

Verify your installation with:

```python
import me2ai_mcp
print(f"ME2AI MCP version: {me2ai_mcp.__version__}")
```

## Requirements

- Python 3.8+
- mcp>=1.6.0
- requests>=2.31.0
- python-dotenv>=1.0.0

## Optional Dependencies

- **Web**: beautifulsoup4>=4.12.0
- **GitHub**: PyGithub>=2.1.0
- **Testing**: pytest and related packages
- **Robot**: robotframework and related packages
