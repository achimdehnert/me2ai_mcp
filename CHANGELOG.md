# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Refactored agent factory for better error handling

### Fixed
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
