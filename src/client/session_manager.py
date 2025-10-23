"""Session manager for handling multiple MCP connections."""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from .mcp_client import MCPClient

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages multiple MCP server connections and configuration."""
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        """
        Initialize session manager.
        
        Args:
            config_path: Path to MCP servers configuration file
        """
        self.config_path = Path(config_path)
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.clients: Dict[str, MCPClient] = {}
        self._load_config()
    
    def _load_config(self):
        """Load server configurations from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.servers = {
                        server['name']: server
                        for server in data.get('servers', [])
                    }
                logger.info(f"Loaded {len(self.servers)} server configurations")
            else:
                logger.info("No configuration file found, starting with empty config")
                self.servers = {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.servers = {}
    
    def _save_config(self):
        """Save server configurations to file."""
        try:
            # Create config directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "servers": list(self.servers.values())
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.servers)} server configurations")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def add_server(self, name: str, url: str, description: str = "", enabled: bool = True) -> Dict[str, Any]:
        """
        Add a new MCP server configuration.
        
        Args:
            name: Server name (unique identifier)
            url: Server SSE endpoint URL
            description: Server description
            enabled: Whether the server is enabled
        
        Returns:
            Status dictionary
        """
        if name in self.servers:
            return {"success": False, "error": f"Server '{name}' already exists"}
        
        self.servers[name] = {
            "name": name,
            "url": url,
            "description": description,
            "enabled": enabled
        }
        
        self._save_config()
        
        return {
            "success": True,
            "message": f"Server '{name}' added successfully"
        }
    
    def remove_server(self, name: str) -> Dict[str, Any]:
        """
        Remove a server configuration.
        
        Args:
            name: Server name
        
        Returns:
            Status dictionary
        """
        if name not in self.servers:
            return {"success": False, "error": f"Server '{name}' not found"}
        
        # Disconnect if connected
        if name in self.clients:
            import asyncio
            asyncio.create_task(self.disconnect_server(name))
        
        del self.servers[name]
        self._save_config()
        
        return {
            "success": True,
            "message": f"Server '{name}' removed successfully"
        }
    
    def update_server(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Update server configuration.
        
        Args:
            name: Server name
            **kwargs: Fields to update
        
        Returns:
            Status dictionary
        """
        if name not in self.servers:
            return {"success": False, "error": f"Server '{name}' not found"}
        
        # Update fields
        for key, value in kwargs.items():
            if key in ['url', 'description', 'enabled']:
                self.servers[name][key] = value
        
        self._save_config()
        
        return {
            "success": True,
            "message": f"Server '{name}' updated successfully"
        }
    
    def get_servers(self) -> List[Dict[str, Any]]:
        """
        Get list of all configured servers.
        
        Returns:
            List of server configurations
        """
        return list(self.servers.values())
    
    def get_enabled_servers(self) -> List[Dict[str, Any]]:
        """
        Get list of enabled servers.
        
        Returns:
            List of enabled server configurations
        """
        return [s for s in self.servers.values() if s.get('enabled', True)]
    
    async def connect_server(self, name: str) -> Dict[str, Any]:
        """
        Connect to an MCP server.
        
        Args:
            name: Server name
        
        Returns:
            Status dictionary with connection info
        """
        if name not in self.servers:
            return {"success": False, "error": f"Server '{name}' not found"}
        
        if name in self.clients and self.clients[name].connected:
            return {
                "success": True,
                "message": f"Already connected to '{name}'",
                "info": self.clients[name].get_info()
            }
        
        server_config = self.servers[name]
        client = MCPClient(
            name=name,
            url=server_config['url'],
            description=server_config.get('description', '')
        )
        
        success = await client.connect()
        
        if success:
            self.clients[name] = client
            return {
                "success": True,
                "message": f"Connected to '{name}'",
                "info": client.get_info()
            }
        else:
            return {
                "success": False,
                "error": f"Failed to connect to '{name}'"
            }
    
    async def disconnect_server(self, name: str) -> Dict[str, Any]:
        """
        Disconnect from an MCP server.
        
        Args:
            name: Server name
        
        Returns:
            Status dictionary
        """
        if name not in self.clients:
            return {"success": False, "error": f"Not connected to '{name}'"}
        
        await self.clients[name].disconnect()
        del self.clients[name]
        
        return {
            "success": True,
            "message": f"Disconnected from '{name}'"
        }
    
    async def connect_selected_servers(self, server_names: List[str]) -> Dict[str, Any]:
        """
        Connect to multiple servers.
        
        Args:
            server_names: List of server names to connect
        
        Returns:
            Status dictionary with results for each server
        """
        results = {}
        
        for name in server_names:
            results[name] = await self.connect_server(name)
        
        return {
            "success": True,
            "results": results
        }
    
    def get_connected_clients(self) -> Dict[str, MCPClient]:
        """
        Get all connected clients.
        
        Returns:
            Dictionary of connected clients
        """
        return {
            name: client
            for name, client in self.clients.items()
            if client.connected
        }
    
    def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available tools from connected servers.
        
        Returns:
            Dictionary mapping server names to their tools
        """
        return {
            name: client.tools
            for name, client in self.clients.items()
            if client.connected
        }
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on a specific server.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        if server_name not in self.clients:
            return {"success": False, "error": f"Not connected to '{server_name}'"}
        
        return await self.clients[server_name].call_tool(tool_name, arguments)
    
    async def cleanup(self):
        """Disconnect all clients and cleanup."""
        for name in list(self.clients.keys()):
            await self.disconnect_server(name)
        
        logger.info("Session manager cleanup complete")
