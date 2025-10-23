#!/usr/bin/env python3
"""
MCPulse Installation Verification Script

This script checks that MCPulse is properly installed and configured.
Run this after installation to verify everything is ready.
"""

import sys
import os
from pathlib import Path
import importlib.util


def print_status(message, status):
    """Print a status message with emoji."""
    if status == "ok":
        print(f"✅ {message}")
    elif status == "warn":
        print(f"⚠️  {message}")
    elif status == "error":
        print(f"❌ {message}")
    else:
        print(f"ℹ️  {message}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "ok")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} (3.8+ required)", "error")
        return False


def check_file_exists(filepath, required=True):
    """Check if a file exists."""
    if Path(filepath).exists():
        print_status(f"{filepath} exists", "ok")
        return True
    else:
        status = "error" if required else "warn"
        print_status(f"{filepath} missing", status)
        return not required


def check_directory_exists(dirpath):
    """Check if a directory exists."""
    if Path(dirpath).is_dir():
        print_status(f"{dirpath}/ exists", "ok")
        return True
    else:
        print_status(f"{dirpath}/ missing", "error")
        return False


def check_package_installed(package_name):
    """Check if a Python package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print_status(f"{package_name} installed", "ok")
        return True
    else:
        print_status(f"{package_name} not installed", "error")
        return False


def check_env_file():
    """Check .env file and configuration."""
    if not Path(".env").exists():
        print_status(".env file missing (using .env.example as template)", "warn")
        return False
    
    with open(".env", "r") as f:
        content = f.read()
    
    has_openai = "OPENAI_API_KEY" in content and "your_" not in content
    has_anthropic = "ANTHROPIC_API_KEY" in content and "your_" not in content
    
    if has_openai or has_anthropic:
        print_status(".env file configured with API key", "ok")
        return True
    else:
        print_status(".env file exists but no API key configured", "warn")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("🔌 MCPulse Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("📍 Python Version:")
    if not check_python_version():
        all_ok = False
    print()
    
    # Check required files
    print("📍 Required Files:")
    required_files = [
        "main.py",
        "requirements.txt",
        "README.md",
    ]
    for file in required_files:
        if not check_file_exists(file, required=True):
            all_ok = False
    print()
    
    # Check optional files
    print("📍 Optional Files:")
    optional_files = [
        ".env",
        "config/mcp_servers.json",
    ]
    for file in optional_files:
        check_file_exists(file, required=False)
    print()
    
    # Check directories
    print("📍 Required Directories:")
    required_dirs = [
        "src",
        "src/client",
        "src/database",
        "src/config",
        "src/ui",
        "config",
        "examples",
    ]
    for dir in required_dirs:
        if not check_directory_exists(dir):
            all_ok = False
    print()
    
    # Check Python packages
    print("📍 Python Packages:")
    required_packages = [
        "gradio",
        "mcp",
        "pymongo",
        "motor",
        "pydantic",
        "pydantic_settings",
        "dotenv",
    ]
    
    packages_ok = True
    for package in required_packages:
        if not check_package_installed(package):
            packages_ok = False
            all_ok = False
    
    if not packages_ok:
        print()
        print_status("Some packages are missing. Run: pip install -r requirements.txt", "warn")
    print()
    
    # Check LLM packages
    print("📍 LLM Provider Packages:")
    has_openai = check_package_installed("openai")
    has_anthropic = check_package_installed("anthropic")
    
    if not has_openai and not has_anthropic:
        print_status("No LLM provider installed. Install openai or anthropic.", "warn")
    print()
    
    # Check environment configuration
    print("📍 Environment Configuration:")
    env_ok = check_env_file()
    if not env_ok:
        print_status("Create .env from .env.example and add your API keys", "warn")
    print()
    
    # Final summary
    print("=" * 60)
    if all_ok and packages_ok and (has_openai or has_anthropic):
        print("✅ All checks passed! MCPulse is ready to run.")
        print()
        print("To start the application:")
        print("  python main.py")
        print()
        print("The application will be available at:")
        print("  http://localhost:7860")
    elif not all_ok:
        print("❌ Some required components are missing.")
        print()
        print("Please run the setup script:")
        print("  ./setup.sh")
    elif not packages_ok:
        print("⚠️  Some packages are not installed.")
        print()
        print("Install dependencies:")
        print("  pip install -r requirements.txt")
    elif not (has_openai or has_anthropic):
        print("⚠️  No LLM provider configured.")
        print()
        print("Install at least one LLM provider:")
        print("  pip install openai")
        print("  # or")
        print("  pip install anthropic")
        print()
        print("Then add your API key to .env file")
    else:
        print("⚠️  Configuration incomplete.")
        print()
        print("Please check the warnings above and:")
        print("  1. Create .env file from .env.example")
        print("  2. Add your API keys to .env")
    
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
