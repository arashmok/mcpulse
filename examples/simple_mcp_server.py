#!/usr/bin/env python3
"""
Simple example MCP server for testing MCPulse.

This server provides basic calculator and utility tools via SSE transport.
Run this server and connect to it from MCPulse to test the integration.

Usage:
    python simple_mcp_server.py
    
Then in MCPulse, add:
    Name: Test Server
    URL: http://localhost:8000/sse
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: fastmcp not installed")
    print("Install with: pip install mcp[cli]")
    exit(1)

# Create MCP server instance
mcp = FastMCP("Test Calculator Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract b from a.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Difference (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide a by b.
    
    Args:
        a: Numerator
        b: Denominator
    
    Returns:
        Quotient (a / b)
    
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def get_current_time() -> str:
    """
    Get the current time.
    
    Returns:
        Current time as ISO format string
    """
    return datetime.now().isoformat()


@mcp.tool()
def reverse_string(text: str) -> str:
    """
    Reverse a string.
    
    Args:
        text: String to reverse
    
    Returns:
        Reversed string
    """
    return text[::-1]


@mcp.tool()
def count_words(text: str) -> Dict[str, Any]:
    """
    Count words in a text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with word count statistics
    """
    words = text.split()
    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(" ", "")),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }


@mcp.resource("greeting://hello")
def get_greeting() -> str:
    """Get a friendly greeting."""
    return "Hello from the MCP Test Server! 👋"


@mcp.resource("info://server")
def get_server_info() -> str:
    """Get information about this server."""
    return json.dumps({
        "name": "Test Calculator Server",
        "version": "1.0.0",
        "description": "A simple MCP server for testing",
        "tools": [
            "add", "subtract", "multiply", "divide",
            "get_current_time", "reverse_string", "count_words"
        ],
        "uptime": "Since server start"
    }, indent=2)


def main():
    """Run the MCP server."""
    print("=" * 60)
    print("🧮 MCP Test Calculator Server")
    print("=" * 60)
    print()
    print("This server provides the following tools:")
    print("  • add(a, b) - Add two numbers")
    print("  • subtract(a, b) - Subtract b from a")
    print("  • multiply(a, b) - Multiply two numbers")
    print("  • divide(a, b) - Divide a by b")
    print("  • get_current_time() - Get current time")
    print("  • reverse_string(text) - Reverse a string")
    print("  • count_words(text) - Count words in text")
    print()
    print("Resources:")
    print("  • greeting://hello - Get a greeting")
    print("  • info://server - Get server information")
    print()
    print("Server starting on http://localhost:8000")
    print("SSE endpoint: http://localhost:8000/sse")
    print()
    print("Add this server to MCPulse:")
    print("  Name: Test Server")
    print("  URL: http://localhost:8000/sse")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run the server with SSE transport
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
