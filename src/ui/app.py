"""Gradio web interface for MCPulse."""

import gradio as gr
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from ..client import SessionManager
from ..database import MongoHandler
from ..config import settings

logger = logging.getLogger(__name__)


class MCPulseApp:
    """Main application class for MCPulse Gradio interface."""
    
    def __init__(self):
        """Initialize the application."""
        self.session_manager = SessionManager()
        self.mongo_handler: Optional[MongoHandler] = None
        self.current_session_id = str(uuid.uuid4())
        self.use_mongodb = False
        
        # LLM configuration (stored in memory, can be updated via GUI)
        self.api_keys = {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "openrouter": settings.openrouter_api_key
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
            if not api_key or api_key.startswith("your_") or api_key.strip() == "":
                return "❌ Please enter a valid API key"
            
            # Update model
            if model and model.strip():
                self.current_model = model.strip()
            
            # Initialize with new configuration
            success = self._initialize_llm(provider=provider, api_key=api_key)
            
            if success:
                return f"✅ Successfully configured {provider.upper()} with model {self.current_model}"
            else:
                return f"❌ Failed to initialize {provider.upper()} client. Check your API key."
                
        except Exception as e:
            logger.error(f"Error updating LLM config: {e}")
            return f"❌ Error: {str(e)}"
    
    def get_current_llm_config(self) -> Dict[str, str]:
        """Get current LLM configuration."""
        return {
            "provider": self.current_provider,
            "model": self.current_model,
            "status": "✅ Connected" if self.llm_client else "❌ Not configured"
        }
    
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
    
    def remove_server(self, server_name: str) -> Tuple[str, gr.update]:
        """Remove an MCP server."""
        if not server_name:
            return "❌ Please select a server to remove", gr.update()
        
        result = self.session_manager.remove_server(server_name)
        
        if result["success"]:
            servers = self.session_manager.get_servers()
            return f"✅ {result['message']}", gr.update(choices=[s['name'] for s in servers])
        else:
            return f"❌ {result['error']}", gr.update()
    
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
    ) -> Tuple[List[List[str]], str]:
        """
        Process chat message with MCP tool integration.
        
        Args:
            message: User message
            history: Chat history
            selected_servers: Selected MCP servers for this chat
            system_prompt: System prompt
        
        Returns:
            Updated history and empty string for input box
        """
        if not message.strip():
            return history, ""
        
        # Add user message to history
        history.append([message, None])
        
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
        
        return history, ""
    
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
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                if self.llm_provider == "openai":
                    response = await self._call_openai(messages, tools)
                elif self.llm_provider == "anthropic":
                    response = await self._call_anthropic(messages, tools)
                else:
                    return "⚠️ No LLM provider configured"
                
                # Check if tool calls are needed
                if not response.get("tool_calls"):
                    return response.get("content", "")
                
                # Execute tool calls
                tool_results = []
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    if tool_name in tool_map:
                        server_name, actual_tool_name = tool_map[tool_name]
                        result = await self.session_manager.call_tool(
                            server_name, actual_tool_name, tool_args
                        )
                        tool_results.append({
                            "tool_name": tool_name,
                            "result": result
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
                        "tool_call_id": tr["tool_name"],
                        "content": json.dumps(tr["result"])
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
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
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
        
        with gr.Blocks(title="MCPulse - MCP SSE Client", theme=gr.themes.Soft()) as app:
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
                            gr.Markdown("### Server Selection")
                            server_selector = gr.CheckboxGroup(
                                choices=self.get_server_list(),
                                label="Active Servers",
                                info="Select servers to use in this chat"
                            )
                            
                            connect_btn = gr.Button("Connect Selected", variant="secondary")
                            connection_status = gr.Textbox(
                                label="Connection Status",
                                lines=5,
                                interactive=False
                            )
                            
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
                    gr.Markdown("## LLM Provider Configuration")
                    gr.Markdown("Configure your AI provider and API key here. Changes take effect immediately.")
                    
                    with gr.Row():
                        with gr.Column():
                            llm_provider = gr.Dropdown(
                                choices=["openai", "anthropic", "openrouter"],
                                label="LLM Provider",
                                value=self.current_provider,
                                info="Select your AI provider"
                            )
                            llm_api_key = gr.Textbox(
                                label="API Key",
                                placeholder="Enter your API key here",
                                type="password",
                                info="Your API key (kept secure, not logged)"
                            )
                            llm_model = gr.Textbox(
                                label="Model Name",
                                value=self.current_model,
                                placeholder="e.g., gpt-4-turbo-preview",
                                info="Model to use for chat completions"
                            )
                            
                            with gr.Accordion("📝 Provider Information & Model Examples", open=False):
                                gr.Markdown("""
                                **OpenAI:**
                                - Get API key: https://platform.openai.com/api-keys
                                - Models: `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`
                                
                                **Anthropic:**
                                - Get API key: https://console.anthropic.com/settings/keys
                                - Models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
                                
                                **OpenRouter:**
                                - Get API key: https://openrouter.ai/keys
                                - Models: `openai/gpt-4-turbo`, `anthropic/claude-3.5-sonnet`, `google/gemini-pro`
                                - Supports 100+ models from multiple providers
                                """)
                            
                            update_llm_btn = gr.Button("💾 Update LLM Configuration", variant="primary")
                            llm_status = gr.Textbox(
                                label="Status",
                                value=self.get_current_llm_config()["status"],
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
                            add_server_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("### Remove Server")
                            remove_server_selector = gr.Dropdown(
                                choices=self.get_server_list(),
                                label="Select Server to Remove"
                            )
                            remove_server_btn = gr.Button("Remove Server", variant="stop")
                            remove_server_status = gr.Textbox(label="Status", interactive=False)
                    
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
                return await self.chat(msg, history, servers, sys_prompt)
            
            async def connect_servers_handler(servers):
                return await self.connect_servers(servers)
            
            async def setup_mongo_handler(uri, db, collection):
                return await self.setup_mongodb(uri, db, collection)
            
            def toggle_mongo(enabled):
                self.use_mongodb = enabled
                return f"MongoDB: {'Enabled' if enabled else 'Disabled'}"
            
            def update_llm_handler(provider, api_key, model):
                return self.update_llm_config(provider, api_key, model)
            
            # Wire up events
            
            # LLM Configuration
            update_llm_btn.click(
                update_llm_handler,
                inputs=[llm_provider, llm_api_key, llm_model],
                outputs=[llm_status]
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
                outputs=[add_server_status, server_selector]
            ).then(
                lambda: self.get_server_list(),
                outputs=[remove_server_selector]
            )
            
            remove_server_btn.click(
                self.remove_server,
                inputs=[remove_server_selector],
                outputs=[remove_server_status, server_selector]
            ).then(
                lambda: self.get_server_list(),
                outputs=[remove_server_selector]
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
