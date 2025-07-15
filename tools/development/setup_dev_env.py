#!/usr/bin/env python3
"""
Development environment setup script for Traffic Monitor.

This script helps set up a complete development environment including:
- Python dependencies
- Pre-commit hooks
- Development configurations
- Test data setup
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result


def main():
    """Set up the development environment."""
    print("Setting up Traffic Monitor development environment...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found. Please run from project root.")
        sys.exit(1)
    
    # Install dependencies
    print("\n1. Installing Python dependencies...")
    run_command("uv sync")
    
    # Install pre-commit hooks
    print("\n2. Setting up pre-commit hooks...")
    result = run_command("uv run pre-commit install", check=False)
    if result.returncode != 0:
        print("Warning: Could not install pre-commit hooks. Install pre-commit manually if needed.")
    
    # Create necessary directories
    print("\n3. Creating necessary directories...")
    directories = [
        "data/models",
        "data/videos/input",
        "data/videos/output", 
        "data/db",
        "logs",
        "data/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    # Run initial tests
    print("\n4. Running initial tests...")
    result = run_command("uv run pytest tests/unit/ -x", check=False)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. This is normal for initial setup.")
    
    # Check code quality
    print("\n5. Running code quality checks...")
    result = run_command("uv run ruff check src/ --fix", check=False)
    if result.returncode == 0:
        print("✅ Code quality checks passed!")
    else:
        print("⚠️  Some code quality issues found. Run 'make lint' to see details.")
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Download model files (see data/models/README.md)")
    print("2. Add test videos to data/videos/input/")
    print("3. Run 'make test' to verify everything works")
    print("4. Run 'make help' to see available commands")


if __name__ == "__main__":
    main()