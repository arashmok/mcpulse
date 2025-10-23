# Contributing to MCPulse

Thank you for your interest in contributing to MCPulse! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate openly

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant logs or screenshots

### Suggesting Features

1. **Open an issue** with the "enhancement" label
2. Describe:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative solutions considered
   - Why this would benefit other users

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the code style (see below)
   - Add tests if applicable
   - Update documentation
4. **Commit your changes**:
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/demos if UI changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/mcpulse.git
cd mcpulse

# Run setup
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Create .env with test credentials
cp .env.example .env
# Edit .env with your test API keys
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Double quotes for strings
- **Imports**: Organized and sorted
- **Type hints**: Use them for function parameters and returns

### Formatting

```bash
# Format code with black
black src/ --line-length 100

# Check style with flake8
flake8 src/ --max-line-length 100

# Type checking with mypy
mypy src/
```

### Code Organization

```python
# Good: Clear, documented, typed
async def connect_server(
    self,
    name: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Connect to an MCP server.
    
    Args:
        name: Server name
        timeout: Connection timeout in seconds
    
    Returns:
        Status dictionary with connection info
    """
    # Implementation
    pass

# Bad: No types, no docs
async def connect_server(name, timeout=30):
    pass
```

## Testing

### Writing Tests

```python
# tests/test_session_manager.py
import pytest
from src.client import SessionManager

@pytest.mark.asyncio
async def test_add_server():
    """Test adding a new MCP server."""
    sm = SessionManager()
    result = sm.add_server("test", "http://localhost:8000/sse")
    
    assert result["success"] is True
    assert "test" in sm.servers

@pytest.mark.asyncio
async def test_connect_server():
    """Test connecting to an MCP server."""
    sm = SessionManager()
    sm.add_server("test", "http://localhost:8000/sse")
    
    result = await sm.connect_server("test")
    
    assert result["success"] is True
    assert "test" in sm.clients
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_session_manager.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

## Project Structure

When adding new features, maintain the existing structure:

```
src/
├── client/          # MCP client code
│   ├── mcp_client.py
│   └── session_manager.py
├── database/        # Database handlers
│   └── mongo_handler.py
├── ui/             # Gradio interface
│   └── app.py
└── config/         # Configuration
    └── settings.py
```

### Adding a New Feature

Example: Adding PostgreSQL support

1. **Create new module**:
   ```
   src/database/postgres_handler.py
   ```

2. **Follow existing patterns**:
   ```python
   class PostgresHandler:
       """Similar interface to MongoHandler"""
       
       async def connect(self) -> bool:
           pass
       
       async def save_message(self, ...) -> bool:
           pass
   ```

3. **Update UI**:
   - Add PostgreSQL configuration options
   - Add database selection dropdown

4. **Update documentation**:
   - Add to README.md
   - Update ARCHITECTURE.md
   - Add example to examples/

5. **Add tests**:
   ```python
   # tests/test_postgres_handler.py
   ```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Explain what the function does,
    when to use it, any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When and why this exception is raised
        ConnectionError: When and why this exception is raised
    
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

### Updating Documentation

When making changes, update:

- **README.md**: For user-facing changes
- **ARCHITECTURE.md**: For technical/structural changes
- **QUICKSTART.md**: If setup process changes
- **Code comments**: For complex logic

## Commit Messages

### Format

```
Type: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changed and why, not how (code shows how).

Fixes #123
```

### Types

- **Add**: New feature or functionality
- **Fix**: Bug fix
- **Update**: Existing feature improvement
- **Refactor**: Code restructuring without behavior change
- **Docs**: Documentation changes
- **Test**: Adding or updating tests
- **Style**: Code style changes (formatting, etc.)
- **Chore**: Maintenance tasks

### Examples

```
Add: PostgreSQL support for chat history

Implements PostgresHandler with same interface as MongoHandler.
Users can now choose between MongoDB and PostgreSQL for storage.

Fixes #45
```

```
Fix: MCP connection timeout handling

Connections now properly timeout after 30 seconds instead of
hanging indefinitely. Added retry logic with exponential backoff.

Fixes #67
```

## Pull Request Process

1. **Update documentation** for any user-facing changes
2. **Add tests** for new functionality
3. **Ensure all tests pass**: `pytest`
4. **Format code**: `black src/`
5. **Check for issues**: `flake8 src/`
6. **Update CHANGELOG** (if exists)
7. **Request review** from maintainers

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
- [ ] Backwards compatible (or breaking changes documented)

## Release Process

(For maintainers)

1. Update version in `src/__init__.py`
2. Update CHANGELOG.md
3. Create release branch: `release/v1.x.x`
4. Test thoroughly
5. Merge to main
6. Tag release: `git tag v1.x.x`
7. Push tags: `git push --tags`
8. Create GitHub release with notes

## Getting Help

- **Questions**: Open a GitHub issue with "question" label
- **Discussion**: Use GitHub Discussions
- **Chat**: Join our community (if available)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MCPulse! 🙏
