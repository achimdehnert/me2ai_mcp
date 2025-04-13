"""
LLM Provider Implementations for ME2AI

This package contains implementations for various Large Language Model (LLM) providers.
Each provider implements a common interface for generating responses while handling
provider-specific configurations and API interactions.

Available Providers:
- OpenAI: Uses GPT models through the OpenAI API
- Groq: Uses Mixtral-8x7b model through the Groq API
- Anthropic: Uses Claude models through the Anthropic API

Configuration:
Each provider requires an API key to be set in the environment variables:
- OPENAI_API_KEY for OpenAI
- GROQ_API_KEY for Groq
- ANTHROPIC_API_KEY for Anthropic

The providers handle:
- API authentication
- Request formatting
- Response parsing
- Error handling
- Rate limiting (where applicable)
"""
