"""MCPulse - MCP SSE Client with Gradio Interface."""

__version__ = "0.1.0"
__author__ = "MCPulse Team"
__description__ = "A ChatGPT-like web application that acts as an MCP SSE client"

from .client import MCPClient, SessionManager
from .database import MongoHandler
from .config import settings
from .ui import create_app, MCPulseApp

__all__ = [
    "MCPClient",
    "SessionManager",
    "MongoHandler",
    "settings",
    "create_app",
    "MCPulseApp",
]
