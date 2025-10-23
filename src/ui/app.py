"""Gradio web interface for MCPulse."""

import gradio as gr
import asyncio
import logging
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from ..client import SessionManager
from ..database import MongoHandler
from ..config import settings

logger = logging.getLogger(__name__)

# Path for storing API keys
API_KEYS_FILE = Path.home() / ".mcpulse" / "api_keys.json"


def save_api_keys_to_file(api_keys: Dict[str, str]):
    """Save API keys to a local file."""
    try:
        API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(api_keys, f)
        logger.info(f"API keys saved to {API_KEYS_FILE}")
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")


def load_api_keys_from_file() -> Dict[str, str]:
    """Load API keys from local file."""
    try:
        if API_KEYS_FILE.exists():
            with open(API_KEYS_FILE, 'r') as f:
                keys = json.load(f)
            logger.info(f"API keys loaded from {API_KEYS_FILE}")
            return keys
        return {}
    except Exception as e:
        logger.error(f"Error loading API keys: {e}")
        return {}


def serialize_mcp_content(content: Any) -> str:
    """
    Convert MCP content (TextContent, ImageContent, etc.) to JSON-serializable string.
    
    Args:
        content: MCP content object or list of content objects
    
    Returns:
        JSON-serializable string representation
    """
    try:
        if isinstance(content, list):
            # Handle list of content objects
            parts = []
            for item in content:
                if hasattr(item, 'text'):
                    # TextContent object
                    parts.append(item.text)
                elif hasattr(item, 'data'):
                    # ImageContent or other binary content
                    parts.append(f"[Binary content: {item.type if hasattr(item, 'type') else 'unknown'}]")
                elif isinstance(item, dict):
                    parts.append(json.dumps(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        elif hasattr(content, 'text'):
            # Single TextContent object
            return content.text
        elif isinstance(content, dict):
            return json.dumps(content)
        else:
            return str(content)
    except Exception as e:
        logger.error(f"Error serializing MCP content: {e}")
        return str(content)


class MCPulseApp:
    """Main application class for MCPulse Gradio interface."""
    
    def __init__(self):
        """Initialize the application."""
        self.session_manager = SessionManager()
        self.mongo_handler: Optional[MongoHandler] = None
        self.current_session_id = str(uuid.uuid4())
        self.use_mongodb = False
        
        # Load API keys from file first, then override with env vars if present and valid
        saved_keys = load_api_keys_from_file()
        
        # Helper to check if key is valid (not None, not empty, not placeholder)
        def is_valid_key(key):
            return key and not key.startswith("your_") and key.strip() != ""
        
        # LLM configuration (stored in memory, can be updated via GUI)
        self.api_keys = {
            "openai": saved_keys.get("openai", "") if saved_keys.get("openai") else (settings.openai_api_key if is_valid_key(settings.openai_api_key) else ""),
            "anthropic": saved_keys.get("anthropic", "") if saved_keys.get("anthropic") else (settings.anthropic_api_key if is_valid_key(settings.anthropic_api_key) else ""),
            "openrouter": saved_keys.get("openrouter", "") if saved_keys.get("openrouter") else (settings.openrouter_api_key if is_valid_key(settings.openrouter_api_key) else "")
        }
        self.current_provider = settings.default_provider
        self.current_model = settings.default_model
        
        # LLM client (will be initialized based on available API keys)
        self.llm_client = None
        self._initialize_llm()
    
    def _initialize_llm(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize LLM client based on available API keys."""
        try:
            # Use provided values or fall back to stored configuration
            if provider:
                self.current_provider = provider
            if api_key:
                self.api_keys[self.current_provider] = api_key
            
            # Get the API key for current provider
            key = self.api_keys.get(self.current_provider)
            
            if self.current_provider == "openai" and key and not key.startswith("your_"):
                from openai import AsyncOpenAI
                self.llm_client = AsyncOpenAI(api_key=key)
                self.llm_provider = "openai"
                logger.info("Initialized OpenAI client")
                return True
            elif self.current_provider == "anthropic" and key and not key.startswith("your_"):
                from anthropic import AsyncAnthropic
                self.llm_client = AsyncAnthropic(api_key=key)
                self.llm_provider = "anthropic"
                logger.info("Initialized Anthropic client")
                return True
            elif self.current_provider == "openrouter" and key and not key.startswith("your_"):
                from openai import AsyncOpenAI
                self.llm_client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=key
                )
                self.llm_provider = "openrouter"
                logger.info("Initialized OpenRouter client")
                return True
            else:
                logger.warning(f"No valid API key configured for {self.current_provider}")
                self.llm_provider = None
                return False
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            self.llm_provider = None
            return False
    
    async def setup_mongodb(self, uri: str, database: str, collection: str) -> str:
        """
        Setup MongoDB connection and collection.
        
        Args:
            uri: MongoDB connection URI
            database: Database name
            collection: Collection name
        
        Returns:
            Status message
        """
        try:
            self.mongo_handler = MongoHandler(uri, database, collection)
            connected = await self.mongo_handler.connect()
            
            if not connected:
                return "❌ Failed to connect to MongoDB. Check your connection URI."
            
            result = await self.mongo_handler.setup_collection()
            
            if result["success"]:
                self.use_mongodb = True
                return f"✅ {result['message']}"
            else:
                return f"❌ Error: {result['error']}"
                
        except Exception as e:
            logger.error(f"MongoDB setup error: {e}")
            return f"❌ Error setting up MongoDB: {str(e)}"
    
    def update_llm_config(self, provider: str, api_key: str, model: str) -> str:
        """
        Update LLM configuration.
        
        Args:
            provider: LLM provider (openai, anthropic, openrouter)
            api_key: API key for the provider
            model: Model name to use
        
        Returns:
            Status message
        """
        try:
            logger.info(f"update_llm_config called - provider: {provider}, api_key present: {bool(api_key)}, model: {model}")
            
            if not api_key or api_key.startswith("your_") or api_key.strip() == "":
                return "❌ Please enter a valid API key"
            
            # Update model
            if model and model.strip():
                self.current_model = model.strip()
            
            # Initialize with new configuration
            success = self._initialize_llm(provider=provider, api_key=api_key)
            
            logger.info(f"LLM initialization result - success: {success}, llm_provider: {self.llm_provider}")
            
            if success:
                return f"✅ Successfully configured {provider.upper()} with model {self.current_model}"
            else:
                return f"❌ Failed to initialize {provider.upper()} client. Check your API key."
                
        except Exception as e:
            logger.error(f"Error updating LLM config: {e}")
            return f"❌ Error: {str(e)}"
    
    def get_current_llm_config(self) -> Dict[str, str]:
        """Get current LLM configuration."""
        logger.info(f"Getting LLM config - llm_client: {self.llm_client is not None}, api_keys: {[(k, 'SET' if v else 'NONE') for k, v in self.api_keys.items()]}")
        
        if self.llm_client:
            status = f"✅ {self.current_provider.upper()}: {self.current_model}"
        else:
            configured_providers = [k.upper() for k, v in self.api_keys.items() if v and not v.startswith("your_")]
            logger.info(f"Configured providers: {configured_providers}")
            if configured_providers:
                status = f"⚠️ API keys configured for: {', '.join(configured_providers)}\n💡 Select provider and model above"
            else:
                status = "⚠️ No API keys configured\n💡 Add them in Configuration tab"
        
        return {
            "provider": self.current_provider,
            "model": self.current_model,
            "status": status
        }
    
    async def fetch_available_models(self, provider: str, api_key: str) -> Tuple[gr.update, str]:
        """
        Fetch available models from the provider.
        
        Args:
            provider: LLM provider (openai, anthropic, openrouter)
            api_key: API key for the provider
        
        Returns:
            Tuple of (Dropdown update with models, status message)
        """
        try:
            if not api_key or api_key.startswith("your_") or api_key.strip() == "":
                return gr.update(choices=[], value=None), "⚠️ Please enter a valid API key first"
            
            models = []
            
            if provider == "openai":
                try:
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(api_key=api_key)
                    response = await client.models.list()
                    # Filter for chat models
                    models = [
                        model.id for model in response.data 
                        if any(x in model.id for x in ['gpt-4', 'gpt-3.5', 'gpt'])
                    ]
                    models.sort(reverse=True)  # Newest first
                except Exception as e:
                    logger.error(f"Error fetching OpenAI models: {e}")
                    return gr.update(choices=[], value=None), f"❌ Error: {str(e)}"
            
            elif provider == "anthropic":
                # Anthropic doesn't have a list endpoint, use known models
                models = [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "claude-2.1",
                    "claude-2.0",
                ]
            
            elif provider == "openrouter":
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            "https://openrouter.ai/api/v1/models",
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10.0
                        )
                        if response.status_code == 200:
                            data = response.json()
                            models = [model["id"] for model in data.get("data", [])]
                            # Sort popular models first
                            models.sort()
                        else:
                            return gr.update(choices=[], value=None), f"❌ Error: API returned status {response.status_code}"
                except Exception as e:
                    logger.error(f"Error fetching OpenRouter models: {e}")
                    return gr.update(choices=[], value=None), f"❌ Error: {str(e)}"
            
            if models:
                return gr.update(choices=models, value=models[0]), f"✅ Found {len(models)} models"
            else:
                return gr.update(choices=[], value=None), "⚠️ No models found"
                
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return gr.update(choices=[], value=None), f"❌ Error: {str(e)}"
    
    def save_api_keys(self, openai_key: str, anthropic_key: str, openrouter_key: str) -> str:
        """Save all API keys at once."""
        try:
            updated = []
            
            if openai_key and openai_key.strip() and not openai_key.startswith("your_"):
                self.api_keys["openai"] = openai_key.strip()
                updated.append("OpenAI")
            
            if anthropic_key and anthropic_key.strip() and not anthropic_key.startswith("your_"):
                self.api_keys["anthropic"] = anthropic_key.strip()
                updated.append("Anthropic")
            
            if openrouter_key and openrouter_key.strip() and not openrouter_key.startswith("your_"):
                self.api_keys["openrouter"] = openrouter_key.strip()
                updated.append("OpenRouter")
            
            if updated:
                # Save only valid keys to file for persistence (filter out empty ones)
                keys_to_save = {k: v for k, v in self.api_keys.items() if v and not v.startswith("your_")}
                save_api_keys_to_file(keys_to_save)
                
                configured = [k.upper() for k, v in self.api_keys.items() if v and not v.startswith("your_")]
                
                # If current provider has a key now and no LLM client, try to initialize
                current_key = self.api_keys.get(self.current_provider, "")
                if current_key and not current_key.startswith("your_") and not self.llm_client:
                    self._initialize_llm(provider=self.current_provider, api_key=current_key)
                
                return f"✅ Saved: {', '.join(updated)}\n🔑 Ready to use: {', '.join(configured)}\n💾 Keys saved to: {API_KEYS_FILE}\n💡 Go to Chat tab to select model!"
            else:
                return "⚠️ No valid API keys provided"
        
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
            return f"❌ Error: {str(e)}"
    
    def add_server(self, name: str, url: str, description: str) -> Tuple[str, gr.update]:
        """Add a new MCP server."""
        if not name or not url:
            return "❌ Name and URL are required", gr.update()
        
        result = self.session_manager.add_server(name, url, description)
        
        if result["success"]:
            servers = self.session_manager.get_servers()
            return f"✅ {result['message']}", gr.update(choices=[s['name'] for s in servers])
        else:
            return f"❌ {result['error']}", gr.update()
    
    def remove_server(self, server_name: str) -> Tuple[str, gr.update, gr.update]:
        """Remove an MCP server."""
        if not server_name:
            return "❌ Please select a server to remove", gr.update(), gr.update()
        
        result = self.session_manager.remove_server(server_name)
        
        if result["success"]:
            servers = self.session_manager.get_servers()
            server_choices = [s['name'] for s in servers]
            return (
                f"✅ {result['message']}", 
                gr.update(choices=server_choices),
                gr.update(choices=server_choices)
            )
        else:
            return f"❌ {result['error']}", gr.update(), gr.update()
    
    def get_server_list(self) -> List[str]:
        """Get list of configured server names."""
        return [s['name'] for s in self.session_manager.get_servers()]
    
    async def connect_servers(self, selected_servers: List[str]) -> str:
        """Connect to selected MCP servers."""
        if not selected_servers:
            return "❌ Please select at least one server"
        
        results = await self.session_manager.connect_selected_servers(selected_servers)
        
        messages = []
        for server_name, result in results["results"].items():
            if result["success"]:
                info = result.get("info", {})
                tools = info.get("tools", [])
                messages.append(f"✅ {server_name}: Connected ({len(tools)} tools available)")
            else:
                messages.append(f"❌ {server_name}: {result.get('error', 'Failed')}")
        
        return "\n".join(messages)
    
    async def chat(
        self,
        message: str,
        history: List[List[str]],
        selected_servers: List[str],
        system_prompt: str
    ):
        """
        Process chat message with MCP tool integration.
        
        Args:
            message: User message
            history: Chat history
            selected_servers: Selected MCP servers for this chat
            system_prompt: System prompt
        
        Yields:
            Updated history and empty string for input box
        """
        if not message.strip():
            yield history, ""
            return
        
        # Add user message to history and immediately show it
        history.append([message, None])
        yield history, ""
        
        # Show thinking indicator with animated dots
        history[-1][1] = '<div class="typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>'
        yield history, ""
        
        try:
            # Save user message to MongoDB if enabled
            if self.use_mongodb and self.mongo_handler:
                await self.mongo_handler.save_message(
                    session_id=self.current_session_id,
                    role="user",
                    content=message
                )
            
            # Get available tools from selected servers
            connected_clients = self.session_manager.get_connected_clients()
            available_tools = []
            tool_map = {}  # Map tool names to (server_name, tool_info)
            
            for server_name in selected_servers:
                if server_name in connected_clients:
                    client = connected_clients[server_name]
                    for tool in client.tools:
                        tool_name = f"{server_name}_{tool['name']}"
                        available_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": tool.get('description', ''),
                                "parameters": tool.get('input_schema', {})
                            }
                        })
                        tool_map[tool_name] = (server_name, tool['name'])
            
            # Build messages for LLM
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for user_msg, assistant_msg in history[:-1]:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            messages.append({"role": "user", "content": message})
            
            # Call LLM
            if not self.llm_client:
                response_text = "⚠️ No LLM API key configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
            else:
                response_text = await self._call_llm_with_tools(
                    messages, available_tools, tool_map
                )
            
            # Update history with assistant response
            history[-1][1] = response_text
            
            # Save assistant message to MongoDB if enabled
            if self.use_mongodb and self.mongo_handler:
                await self.mongo_handler.save_message(
                    session_id=self.current_session_id,
                    role="assistant",
                    content=response_text
                )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            history[-1][1] = f"❌ Error: {str(e)}"
        
        yield history, ""
    
    async def _call_llm_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_map: Dict[str, Tuple[str, str]]
    ) -> str:
        """
        Call LLM with tool support.
        
        Args:
            messages: Conversation messages
            tools: Available tools
            tool_map: Mapping of tool names to server and tool info
        
        Returns:
            Assistant response text
        """
        max_iterations = 5
        iteration = 0
        
        logger.info(f"Chat request - llm_provider: {self.llm_provider}, current_provider: {self.current_provider}, current_model: {self.current_model}")
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                if self.llm_provider in ["openai", "openrouter"]:
                    # OpenRouter uses OpenAI-compatible API
                    response = await self._call_openai(messages, tools)
                elif self.llm_provider == "anthropic":
                    response = await self._call_anthropic(messages, tools)
                else:
                    # No LLM provider is initialized
                    logger.warning(f"No LLM provider initialized. llm_provider={self.llm_provider}, llm_client={self.llm_client}")
                    configured_providers = [k.upper() for k, v in self.api_keys.items() if v and not v.startswith("your_")]
                    if configured_providers:
                        return f"⚠️ Please select a provider and model in the sidebar.\n\n📋 Available: {', '.join(configured_providers)}"
                    else:
                        return "⚠️ No API keys configured. Please:\n1. Go to Configuration tab\n2. Enter your API key\n3. Return to Chat tab\n4. Select provider and model"
                
                # Check if tool calls are needed
                if not response.get("tool_calls"):
                    return response.get("content", "")
                
                # Execute tool calls
                tool_results = []
                for tool_call in response["tool_calls"]:
                    # Extract tool info - handle both formats
                    if "function" in tool_call:
                        # OpenAI format with nested function
                        tool_name = tool_call["function"]["name"]
                        tool_args_str = tool_call["function"]["arguments"]
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                        tool_call_id = tool_call.get("id", tool_name)
                    else:
                        # Simplified format
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]
                        tool_call_id = tool_call.get("id", tool_name)
                    
                    if tool_name in tool_map:
                        server_name, actual_tool_name = tool_map[tool_name]
                        result = await self.session_manager.call_tool(
                            server_name, actual_tool_name, tool_args
                        )
                        
                        # Extract content from MCP result and serialize
                        if isinstance(result, dict) and "content" in result:
                            result_content = serialize_mcp_content(result["content"])
                        else:
                            result_content = serialize_mcp_content(result)
                        
                        tool_results.append({
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "result": result_content
                        })
                
                # Add tool results to messages and continue
                messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": response["tool_calls"]
                })
                
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["result"]  # Already serialized as string
                    })
                
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return f"❌ Error calling LLM: {str(e)}"
        
        return "⚠️ Max iterations reached"
    
    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        kwargs = {
            "model": settings.default_model,
            "messages": messages
        }
        
        if tools:
            kwargs["tools"] = tools
        
        response = await self.llm_client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        
        result = {
            "content": message.content or ""
        }
        
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments  # Keep as string
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result
    
    async def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        # Extract system message
        system_msg = ""
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered_messages.append(msg)
        
        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "messages": filtered_messages
        }
        
        if system_msg:
            kwargs["system"] = system_msg
        
        if tools:
            kwargs["tools"] = [t["function"] for t in tools]
        
        response = await self.llm_client.messages.create(**kwargs)
        
        result = {
            "content": ""
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                if "tool_calls" not in result:
                    result["tool_calls"] = []
                result["tool_calls"].append({
                    "name": block.name,
                    "arguments": block.input
                })
        
        return result
    
    def new_session(self) -> Tuple[List, str]:
        """Start a new chat session."""
        self.current_session_id = str(uuid.uuid4())
        return [], f"New session started: {self.current_session_id[:8]}"
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface."""
        
        # Custom CSS for thinking animation and password fields
        custom_css = """
        @keyframes bounce {
            0%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-10px);
            }
        }
        
        .typing-indicator {
            display: inline-flex;
            align-items: flex-end;
            gap: 6px;
            height: 20px;
        }
        
        .typing-indicator .dot {
            width: 8px;
            height: 8px;
            background-color: #666;
            border-radius: 50%;
            display: inline-block;
            animation: bounce 1.2s infinite ease-in-out;
        }
        
        .typing-indicator .dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator .dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        /* Password field with inline eye icon */
        .password-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .password-wrapper input {
            padding-right: 45px !important;
        }
        
        .eye-toggle {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
            color: #888;
            font-size: 18px;
            z-index: 100;
            user-select: none;
            transition: color 0.2s;
        }
        
        .eye-toggle:hover {
            color: #333;
        }
        """
        
        with gr.Blocks(title="MCPulse - MCP SSE Client", theme=gr.themes.Soft(), css=custom_css, head="""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        .eye-icon-btn {
            position: absolute !important;
            right: 10px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            cursor: pointer !important;
            font-size: 18px !important;
            z-index: 9999 !important;
            background: transparent !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
            color: #666 !important;
            transition: color 0.2s !important;
        }
        .eye-icon-btn:hover {
            color: #333 !important;
        }
        </style>
        """) as app:
            gr.Markdown("# 🔌 MCPulse - MCP SSE Client")
            gr.Markdown("Connect to MCP servers and chat with AI assistants that can use their tools.")
            
            with gr.Tabs():
                # Chat Tab
                with gr.Tab("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=500,
                                show_copy_button=True
                            )
                            
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    label="Message",
                                    placeholder="Type your message here...",
                                    scale=4
                                )
                                send_btn = gr.Button("Send", scale=1, variant="primary")
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")
                                new_session_btn = gr.Button("New Session")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 🤖 AI Model Selection")
                            
                            chat_llm_provider = gr.Dropdown(
                                choices=["openai", "anthropic", "openrouter"],
                                label="Provider",
                                value=self.current_provider,
                                info="Select your AI provider"
                            )
                            
                            with gr.Row():
                                refresh_models_btn = gr.Button("🔄", scale=1, size="sm")
                                chat_llm_model = gr.Dropdown(
                                    choices=[self.current_model] if self.current_model else [],
                                    label="Model",
                                    value=self.current_model,
                                    allow_custom_value=True,
                                    scale=4,
                                    info="Select or type model name"
                                )
                            
                            llm_info_status = gr.Textbox(
                                label="",
                                value="Loading status...",
                                interactive=False,
                                show_label=False,
                                lines=2
                            )
                            
                            gr.Markdown("---")
                            gr.Markdown("### 🔌 Server Selection")
                            server_selector = gr.CheckboxGroup(
                                choices=self.get_server_list(),
                                label="Active Servers",
                                info="Select servers to use in this chat"
                            )
                            
                            connect_btn = gr.Button("Connect Selected", variant="secondary")
                            connection_status = gr.Textbox(
                                label="Connection Status",
                                lines=4,
                                interactive=False
                            )
                            
                            gr.Markdown("---")
                            system_prompt = gr.Textbox(
                                label="System Prompt",
                                placeholder="You are a helpful assistant...",
                                lines=3,
                                value="You are a helpful AI assistant with access to various tools through MCP servers. Use the available tools when appropriate to help the user."
                            )
                            
                            session_info = gr.Textbox(
                                label="Session Info",
                                value=f"Session: {self.current_session_id[:8]}",
                                interactive=False
                            )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration"):
                    gr.Markdown("## 🔑 API Keys Configuration")
                    gr.Markdown("Enter your API keys for each provider. Keys are stored securely and changes take effect immediately.")
                    
                    with gr.Row():
                        with gr.Column():
                            config_openai_key = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="sk-...",
                                value=self.api_keys.get("openai", ""),
                                type="password",
                                elem_id="openai-api-key",
                                info="Get your key at: https://platform.openai.com/api-keys"
                            )
                            
                            config_anthropic_key = gr.Textbox(
                                label="Anthropic API Key",
                                placeholder="sk-ant-...",
                                value=self.api_keys.get("anthropic", ""),
                                type="password",
                                elem_id="anthropic-api-key",
                                info="Get your key at: https://console.anthropic.com/settings/keys"
                            )
                            
                            config_openrouter_key = gr.Textbox(
                                label="OpenRouter API Key",
                                placeholder="sk-or-...",
                                value=self.api_keys.get("openrouter", ""),
                                type="password",
                                elem_id="openrouter-api-key",
                                info="Get your key at: https://openrouter.ai/keys (Access 100+ models)"
                            )
                            
                            save_keys_btn = gr.Button("💾 Save API Keys", variant="primary")
                            keys_status = gr.Textbox(
                                label="Status",
                                value=f"Current providers configured: {', '.join([k for k, v in self.api_keys.items() if v])}",
                                interactive=False
                            )
                    
                    gr.Markdown("---")
                    gr.Markdown("## MCP Server Management")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Add New Server")
                            new_server_name = gr.Textbox(label="Server Name")
                            new_server_url = gr.Textbox(
                                label="Server URL",
                                placeholder="http://localhost:8000/sse"
                            )
                            new_server_desc = gr.Textbox(
                                label="Description",
                                placeholder="Optional description"
                            )
                            add_server_btn = gr.Button("Add Server", variant="primary")
                        
                        with gr.Column():
                            gr.Markdown("### Remove Server")
                            remove_server_selector = gr.Dropdown(
                                choices=self.get_server_list(),
                                label="Select Server to Remove"
                            )
                            remove_server_btn = gr.Button("Remove Server", variant="stop")
                    
                    # Single status box for both Add and Remove operations
                    server_management_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("## MongoDB Configuration")
                    
                    with gr.Row():
                        mongodb_uri = gr.Textbox(
                            label="MongoDB URI",
                            value=settings.mongodb_uri,
                            placeholder="mongodb://localhost:27017"
                        )
                        mongodb_db = gr.Textbox(
                            label="Database",
                            value=settings.mongodb_database
                        )
                        mongodb_collection = gr.Textbox(
                            label="Collection",
                            value=settings.mongodb_collection
                        )
                    
                    setup_mongo_btn = gr.Button("Setup MongoDB Collection", variant="secondary")
                    mongo_status = gr.Textbox(label="MongoDB Status", interactive=False)
                    
                    use_mongo_checkbox = gr.Checkbox(
                        label="Enable MongoDB for chat history",
                        value=False
                    )
            
            # Event handlers
            async def send_message(msg, history, servers, sys_prompt):
                async for updated_history, cleared_input in self.chat(msg, history, servers, sys_prompt):
                    yield updated_history, cleared_input
            
            async def connect_servers_handler(servers):
                return await self.connect_servers(servers)
            
            async def setup_mongo_handler(uri, db, collection):
                return await self.setup_mongodb(uri, db, collection)
            
            def toggle_mongo(enabled):
                self.use_mongodb = enabled
                return f"MongoDB: {'Enabled' if enabled else 'Disabled'}"
            
            def save_keys_handler(openai_key, anthropic_key, openrouter_key):
                return self.save_api_keys(openai_key, anthropic_key, openrouter_key)
            
            async def fetch_models_handler(provider):
                # Use the stored API key for the selected provider
                api_key = self.api_keys.get(provider, "")
                if not api_key:
                    return gr.update(choices=[], value=None), f"⚠️ Please configure {provider} API key in Configuration tab first"
                return await self.fetch_available_models(provider, api_key)
            
            def switch_provider_handler(provider, model):
                # When provider changes, update LLM and return status
                api_key = self.api_keys.get(provider, "")
                if not api_key:
                    return gr.update(value=model), f"⚠️ No API key configured for {provider}. Please add it in Configuration tab."
                
                # Update the LLM client
                status = self.update_llm_config(provider, api_key, model)
                return gr.update(value=model), status
            
            # Wire up events
            
            # Chat tab - Model selection
            refresh_models_btn.click(
                fetch_models_handler,
                inputs=[chat_llm_provider],
                outputs=[chat_llm_model, llm_info_status]
            )
            
            chat_llm_provider.change(
                switch_provider_handler,
                inputs=[chat_llm_provider, chat_llm_model],
                outputs=[chat_llm_model, llm_info_status]
            )
            
            chat_llm_model.change(
                lambda provider, model: self.update_llm_config(
                    provider, 
                    self.api_keys.get(provider, ""), 
                    model
                ),
                inputs=[chat_llm_provider, chat_llm_model],
                outputs=[llm_info_status]
            )
            
            # Configuration tab - API Keys
            
            save_keys_btn.click(
                save_keys_handler,
                inputs=[config_openai_key, config_anthropic_key, config_openrouter_key],
                outputs=[keys_status]
            ).then(
                lambda: self.get_current_llm_config()["status"],
                outputs=[llm_info_status]
            )
            
            # Chat events
            send_btn.click(
                send_message,
                inputs=[msg_input, chatbot, server_selector, system_prompt],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                send_message,
                inputs=[msg_input, chatbot, server_selector, system_prompt],
                outputs=[chatbot, msg_input]
            )
            
            clear_btn.click(lambda: [], outputs=[chatbot])
            
            new_session_btn.click(
                self.new_session,
                outputs=[chatbot, session_info]
            )
            
            connect_btn.click(
                connect_servers_handler,
                inputs=[server_selector],
                outputs=[connection_status]
            )
            
            add_server_btn.click(
                self.add_server,
                inputs=[new_server_name, new_server_url, new_server_desc],
                outputs=[server_management_status, server_selector]
            ).then(
                lambda: self.get_server_list(),
                outputs=[remove_server_selector]
            )
            
            remove_server_btn.click(
                self.remove_server,
                inputs=[remove_server_selector],
                outputs=[server_management_status, server_selector, remove_server_selector]
            )
            
            setup_mongo_btn.click(
                setup_mongo_handler,
                inputs=[mongodb_uri, mongodb_db, mongodb_collection],
                outputs=[mongo_status]
            )
            
            use_mongo_checkbox.change(
                toggle_mongo,
                inputs=[use_mongo_checkbox],
                outputs=[mongo_status]
            )
            
            # Update status on page load
            app.load(
                lambda: self.get_current_llm_config()["status"],
                outputs=[llm_info_status]
            )
            
            # Load saved API keys on page load
            app.load(
                lambda: [
                    self.api_keys.get("openai", ""),
                    self.api_keys.get("anthropic", ""),
                    self.api_keys.get("openrouter", ""),
                    f"Current providers configured: {', '.join([k.upper() for k, v in self.api_keys.items() if v and not v.startswith('your_')])}"
                ],
                outputs=[config_openai_key, config_anthropic_key, config_openrouter_key, keys_status]
            )
            
            # Add eye icons using JavaScript on app load
            app.load(
                None,
                None,
                None,
                js="""
                () => {
                    function addEyeIcons() {
                        const pwFields = document.querySelectorAll('input[type="password"]');
                        
                        pwFields.forEach(function(input) {
                            if (input.dataset.hasEye) return;
                            
                            const ph = input.getAttribute('placeholder') || '';
                            if (!ph.startsWith('sk')) return;
                            
                            input.dataset.hasEye = 'true';
                            input.style.paddingRight = '40px';
                            
                            const eye = document.createElement('i');
                            eye.className = 'fas fa-eye eye-icon-btn';
                            eye.onclick = function() {
                                if (input.type === 'password') {
                                    input.type = 'text';
                                    eye.className = 'fas fa-eye-slash eye-icon-btn';
                                } else {
                                    input.type = 'password';
                                    eye.className = 'fas fa-eye eye-icon-btn';
                                }
                            };
                            
                            input.parentElement.style.position = 'relative';
                            input.parentElement.appendChild(eye);
                        });
                    }
                    
                    // Run after delays to handle async rendering
                    setTimeout(addEyeIcons, 500);
                    setTimeout(addEyeIcons, 1000);
                    setTimeout(addEyeIcons, 2000);
                    
                    // Listen for config tab clicks
                    document.addEventListener('click', function(e) {
                        if (e.target.textContent && e.target.textContent.includes('Configuration')) {
                            setTimeout(addEyeIcons, 300);
                        }
                    });
                }
                """
            )
        
        return app
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.session_manager.cleanup()
        if self.mongo_handler:
            await self.mongo_handler.disconnect()


def create_app() -> gr.Blocks:
    """Create and return the Gradio app."""
    app_instance = MCPulseApp()
    return app_instance.create_interface()
