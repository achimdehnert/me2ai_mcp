#!/usr/bin/env python
"""
ME2AI MCP Package Installation Script

This script simplifies the installation process for the ME2AI MCP package,
allowing users to choose installation options and handling dependencies.
"""
import os
import sys
import subprocess
import argparse
from typing import List, Dict, Any, Optional


def check_python_version() -> bool:
    """Check if the Python version is compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        print(f"Error: Python {min_version[0]}.{min_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    return True


def run_command(command: List[str]) -> bool:
    """Run a command and return if it was successful."""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error output: {e.stderr}")
        return False


def install_package(extras: Optional[List[str]] = None, dev: bool = False, 
                   editable: bool = False) -> bool:
    """Install the ME2AI MCP package with specified options."""
    base_command = [sys.executable, "-m", "pip", "install"]
    
    # Add editable flag if specified
    if editable:
        base_command.append("-e")
    
    # Determine package specification
    if extras:
        extras_str = ",".join(extras)
        package_spec = f".[{extras_str}]"
    else:
        package_spec = "."
    
    base_command.append(package_spec)
    
    # Add dev dependencies if specified
    if dev:
        dev_command = base_command + ["-r", "requirements-dev.txt"]
        if not run_command(dev_command):
            return False
    
    # Install the package
    return run_command(base_command)


def setup_virtual_env() -> bool:
    """Set up a virtual environment."""
    venv_dir = "venv"
    
    # Check if venv directory already exists
    if os.path.exists(venv_dir):
        print(f"Virtual environment directory '{venv_dir}' already exists.")
        response = input("Do you want to create a new one? (y/n): ").lower()
        if response != 'y':
            return True
        
        # Remove existing venv directory
        if os.name == 'nt':  # Windows
            run_command(["rmdir", "/s", "/q", venv_dir])
        else:  # Unix/Linux/MacOS
            run_command(["rm", "-rf", venv_dir])
    
    # Create virtual environment
    print(f"Creating virtual environment in '{venv_dir}'...")
    if not run_command([sys.executable, "-m", "venv", venv_dir]):
        return False
    
    # Activate virtual environment instructions
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_dir, "Scripts", "activate")
        print(f"\nTo activate the virtual environment, run:\n{activate_script}")
    else:  # Unix/Linux/MacOS
        activate_script = os.path.join(venv_dir, "bin", "activate")
        print(f"\nTo activate the virtual environment, run:\nsource {activate_script}")
    
    return True


def create_requirements_dev() -> None:
    """Create requirements-dev.txt if it doesn't exist."""
    req_dev_file = "requirements-dev.txt"
    
    if not os.path.exists(req_dev_file):
        with open(req_dev_file, "w") as f:
            f.write("# Development dependencies\n")
            f.write("pytest>=7.0.0\n")
            f.write("pytest-asyncio>=0.20.0\n")
            f.write("mypy>=0.990\n")
            f.write("black>=23.0.0\n")
            f.write("isort>=5.12.0\n")
            f.write("robotframework>=6.0.0\n")
            f.write("robotframework-seleniumlibrary>=6.0.0\n")
            f.write("robotframework-requests>=0.9.0\n")
            f.write("webdrivermanager>=0.10.0\n")
            f.write("flake8>=6.0.0\n")
            f.write("twine>=4.0.0\n")
            f.write("build>=0.10.0\n")
        
        print(f"Created {req_dev_file} with development dependencies")


def main() -> int:
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install ME2AI MCP package")
    parser.add_argument("--all", action="store_true", help="Install with all extras")
    parser.add_argument("--web", action="store_true", help="Install with web extras")
    parser.add_argument("--github", action="store_true", help="Install with GitHub extras")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--venv", action="store_true", help="Set up a virtual environment")
    parser.add_argument("--editable", "-e", action="store_true", help="Install in development/editable mode")
    
    args = parser.parse_args()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Set up virtual environment if requested
    if args.venv and not setup_virtual_env():
        return 1
    
    # Prepare development requirements if needed
    if args.dev:
        create_requirements_dev()
    
    # Determine extras to install
    extras = []
    if args.all:
        extras.append("all")
    else:
        if args.web:
            extras.append("web")
        if args.github:
            extras.append("github")
    
    # Install the package
    if install_package(extras, args.dev, args.editable):
        print("\n✅ ME2AI MCP package installed successfully!")
        
        # Print next steps
        print("\nNext steps:")
        print("1. Try running an example: python examples/custom_mcp_server.py")
        print("2. Run the tests: pytest tests/")
        if args.dev:
            print("3. Run Robot Framework tests: robot -d reports tests/robot/")
        
        return 0
    else:
        print("\n❌ Installation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
