# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-04-24

### Added

- Scalable Vector Store microservice with pluggable backend architecture
- Support for multiple vector store backends: ChromaDB, FAISS, Qdrant, and Pinecone
- Flexible embedding model integration: Sentence Transformers, OpenAI, Cohere, HuggingFace
- Comprehensive REST API for collection management and document operations
- File upload and embedding functionality with content safety limits
- Advanced filtering capabilities for both metadata and document content
- Comprehensive test suite with 95%+ code coverage across all backends
- Knowledge Assistant integration with end-to-end testing
- Detailed documentation with backend comparisons and integration patterns
- Automated test reporting and regression testing infrastructure

### Changed

- Enhanced service initialization with dynamic dependency detection
- Improved error handling for missing optional dependencies
- Updated documentation structure with dedicated service guides

## [0.1.1] - 2025-04-22

### Added

- Flexible database credential management system with multiple source support
- Comprehensive PostgreSQL integration with connection pooling and schema validation
- Robust MySQL integration with enhanced error handling and connection pooling
- LangChain compatibility layer for seamless tool integration with agent frameworks
- Extensive test suite for database components and LangChain integration
- Standalone PostgreSQL and MySQL MCP servers with comprehensive documentation
- Database-specific command-line tools with configuration options

### Changed

- Enhanced package structure with clearer module organization
- Updated dependency management with optional database components
- Improved documentation with comprehensive database integration guide
- Optimized connection handling for better performance and reliability

### Fixed

- Schema validation in database connections
- Error handling in MCP tool execution
- LangChain integration parameter parsing

## [0.0.6] - 2025-04-13

### Added

- Groq LLM provider support with Mixtral-8x7b model
- Comprehensive test suite for CLI and load testing
- Memory management system for conversation history
- SEO expert agent with technical analysis tools
- German professor agent with language learning tools
- Dating expert agent with relationship advice tools

### Changed

- Improved agent initialization to properly handle system prompts and roles
- Enhanced async/await patterns in CLI for better performance
- Updated expert agents to use proper memory management
- Consistent package naming across all platforms (PyPI, imports, GitHub)

### Fixed

- Package configuration with proper README path
- Standardized version numbering
- Refactored agent factory for better error handling
- Agent initialization issues with system prompts and roles
- Memory initialization in expert agents
- Async response handling in CLI
- Import paths for agent modules

### Known Issues
- Load tests failing due to memory initialization and async handling
- Performance degradation under heavy load
- Memory usage needs optimization

## [0.1.0] - 2024-12-22

### Added
- Initial release with basic agent framework
- CLI interface for agent interaction
- Support for multiple LLM providers
- Basic expert agents implementation
- Tool system for enhanced agent capabilities
