"""
Setup script for the ME2AI MCP package.
"""
from setuptools import setup, find_packages
import os
import sys

# Explicitly hardcode the version
VERSION = "0.1.2"

setup(
    name="me2ai_mcp",
    version=VERSION,
    description="Enhanced Model Context Protocol framework for ME2AI agents and services",
    author="ME2AI Team",
    author_email="info@me2ai.dev",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/achimdehnert/me2ai",
    packages=find_packages(),
    install_requires=[
        "mcp>=1.6.0",
        "requests>=2.25.0",
        "python-dotenv>=0.15.0",
        "pydantic>=1.9.0",
    ],
    extras_require={
        "web": ["beautifulsoup4>=4.12.0"],
        "github": ["PyGithub>=2.1.0"],
        "postgres": ["psycopg2-binary>=2.9.0"],
        "mysql": ["mysql-connector-python>=8.0.0"],
        "db": [
            "psycopg2-binary>=2.9.0",
            "mysql-connector-python>=8.0.0",
        ],
        "langchain": ["langchain>=0.0.267"],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "pytest-html>=4.0.0",
            "pytest-json>=0.4.0",
            "psutil>=5.9.0",
        ],
        "robot": [
            "robotframework>=6.0.0",
            "robotframework-seleniumlibrary>=6.0.0",
        ],
        "all": [
            "beautifulsoup4>=4.12.0",
            "PyGithub>=2.1.0",
            "psycopg2-binary>=2.9.0",
            "mysql-connector-python>=8.0.0",
            "langchain>=0.0.267",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "pytest-html>=4.0.0",
            "pytest-json>=0.4.0",
            "psutil>=5.9.0",
            "robotframework>=6.0.0",
            "robotframework-seleniumlibrary>=6.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["mcp", "model", "context", "protocol", "ai", "agent", "database", "postgresql", "mysql", "langchain"],
)
