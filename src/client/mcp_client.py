"""MCP SSE Client implementation."""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP SSE Client for connecting to MCP servers."""
    
    def __init__(self, name: str, url: str, description: str = ""):
        """
        Initialize MCP client.
        
        Args:
            name: Server name
            url: Server SSE endpoint URL
            description: Server description
        """
        self.name = name
        self.url = url
        self.description = description
        self.session: Optional[ClientSession] = None
        self.connected = False
        self.tools: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        self.prompts: List[Dict[str, Any]] = []
        self._read_stream = None
        self._write_stream = None
        self._sse_context = None
    
    async def connect(self) -> bool:
        """
        Connect to the MCP server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MCP server: {self.name} at {self.url}")
            
            # Create SSE client connection
            self._sse_context = sse_client(self.url)
            self._read_stream, self._write_stream = await self._sse_context.__aenter__()
            
            # Create client session
            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            
            # Initialize the session
            await self.session.initialize()
            
            # Discover available capabilities
            await self._discover_capabilities()
            
            self.connected = True
            logger.info(f"Successfully connected to {self.name}")
            logger.info(f"Available tools: {[t['name'] for t in self.tools]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self._sse_context:
                await self._sse_context.__aexit__(None, None, None)
            
            self.connected = False
            logger.info(f"Disconnected from {self.name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.name}: {e}")
    
    async def _discover_capabilities(self):
        """Discover tools, resources, and prompts from the server."""
        try:
            # List available tools
            if self.session:
                tools_response = await self.session.list_tools()
                self.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema
                    }
                    for tool in tools_response.tools
                ]
                
                # Try to list resources (optional)
                try:
                    resources_response = await self.session.list_resources()
                    self.resources = [
                        {
                            "uri": resource.uri,
                            "name": resource.name or "",
                            "description": resource.description or "",
                            "mime_type": resource.mimeType or ""
                        }
                        for resource in resources_response.resources
                    ]
                except Exception as e:
                    logger.debug(f"No resources available or error listing: {e}")
                
                # Try to list prompts (optional)
                try:
                    prompts_response = await self.session.list_prompts()
                    self.prompts = [
                        {
                            "name": prompt.name,
                            "description": prompt.description or "",
                            "arguments": prompt.arguments or []
                        }
                        for prompt in prompts_response.prompts
                    ]
                except Exception as e:
                    logger.debug(f"No prompts available or error listing: {e}")
                
        except Exception as e:
            logger.error(f"Error discovering capabilities: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
        
        Returns:
            Tool result as dictionary
        """
        if not self.connected or not self.session:
            return {"error": "Not connected to server"}
        
        try:
            logger.info(f"Calling tool {tool_name} on {self.name}")
            result = await self.session.call_tool(tool_name, arguments)
            
            return {
                "success": True,
                "content": result.content,
                "isError": result.isError if hasattr(result, 'isError') else False
            }
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: Resource URI
        
        Returns:
            Resource content as dictionary
        """
        if not self.connected or not self.session:
            return {"error": "Not connected to server"}
        
        try:
            logger.info(f"Reading resource {uri} from {self.name}")
            result = await self.session.read_resource(uri)
            
            return {
                "success": True,
                "contents": result.contents
            }
            
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a prompt from the MCP server.
        
        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments (optional)
        
        Returns:
            Prompt content as dictionary
        """
        if not self.connected or not self.session:
            return {"error": "Not connected to server"}
        
        try:
            logger.info(f"Getting prompt {prompt_name} from {self.name}")
            result = await self.session.get_prompt(prompt_name, arguments or {})
            
            return {
                "success": True,
                "messages": result.messages
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get server information.
        
        Returns:
            Server information dictionary
        """
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "connected": self.connected,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts
        }
