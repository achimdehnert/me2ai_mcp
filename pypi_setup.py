"""
Setup script for the ME2AI-MCP package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="me2ai-mcp",
    version="0.1.0",
    author="ME2AI Team",
    author_email="your-email@example.com",
    description="Enhanced Model Context Protocol framework for ME2AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/achimdehnert/me2ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mcp>=1.6.0",
        "requests>=2.25.0",
        "python-dotenv>=0.15.0",
    ],
    extras_require={
        "web": ["beautifulsoup4>=4.9.3", "bleach>=3.3.0"],
        "github": ["requests>=2.25.0"],
        "all": ["beautifulsoup4>=4.9.3", "bleach>=3.3.0", "requests>=2.25.0"],
    },
    keywords="mcp, ai, model context protocol, agent, tools",
)
