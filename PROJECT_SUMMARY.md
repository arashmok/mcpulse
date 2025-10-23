# MCPulse - Project Summary

## 📋 Overview

**MCPulse** is a production-ready web application that provides a ChatGPT-like interface for interacting with MCP (Model Context Protocol) servers. Built with Gradio, it allows users to connect to multiple MCP servers simultaneously and leverage their tools through natural language conversations with AI assistants.

## 🎯 Key Features

✅ **Multi-Server Support** - Connect to multiple MCP servers at once  
✅ **SSE Transport** - Real-time communication using Server-Sent Events  
✅ **ChatGPT-like UI** - Clean, intuitive Gradio interface  
✅ **Tool Integration** - Automatic tool discovery and execution  
✅ **MongoDB Support** - Optional chat history persistence  
✅ **LLM Flexibility** - Supports OpenAI and Anthropic APIs  
✅ **Easy Configuration** - Simple UI for managing servers  
✅ **Session Management** - Multiple chat sessions with isolation  

## 📁 Project Structure

```
mcpulse/
├── 📄 Configuration Files
│   ├── .env.example              # Environment variables template
│   ├── .gitignore               # Git ignore rules
│   ├── requirements.txt         # Python dependencies
│   ├── setup.sh                 # Linux/macOS setup script
│   ├── setup.bat                # Windows setup script
│   └── LICENSE                  # MIT License
│
├── 📚 Documentation
│   ├── README.md                # Main documentation
│   ├── QUICKSTART.md            # Quick start guide
│   ├── ARCHITECTURE.md          # Technical architecture
│   ├── TESTING.md               # Testing guide
│   └── CONTRIBUTING.md          # Contribution guidelines
│
├── ⚙️ config/
│   └── mcp_servers.json         # MCP server configurations
│
├── 📝 examples/
│   ├── README.md                # Examples documentation
│   └── simple_mcp_server.py     # Test MCP server
│
├── 🚀 main.py                   # Application entry point
│
└── 💻 src/                      # Source code
    ├── __init__.py              # Package initialization
    │
    ├── 🔌 client/               # MCP Client Implementation
    │   ├── __init__.py
    │   ├── mcp_client.py        # Individual server client
    │   └── session_manager.py   # Multi-server management
    │
    ├── 💾 database/             # Database Integration
    │   ├── __init__.py
    │   └── mongo_handler.py     # MongoDB operations
    │
    ├── ⚙️ config/               # Configuration Management
    │   ├── __init__.py
    │   └── settings.py          # Settings & environment vars
    │
    └── 🎨 ui/                   # User Interface
        ├── __init__.py
        └── app.py               # Gradio application
```

## 🔧 Core Components

### 1. MCP Client (`src/client/`)

**Purpose**: Handles connections to MCP servers using SSE transport

**Files**:
- `mcp_client.py`: Implements connection to individual MCP server
  - SSE connection management
  - Tool/resource/prompt discovery
  - Tool execution
  - Error handling

- `session_manager.py`: Manages multiple server connections
  - Server configuration (add/remove/update)
  - Connection lifecycle
  - Configuration persistence
  - Tool aggregation

**Key Features**:
- Async/await for non-blocking operations
- Automatic reconnection
- Tool discovery and schema extraction
- Support for resources and prompts

### 2. Database Handler (`src/database/`)

**Purpose**: Optional MongoDB integration for chat history

**Files**:
- `mongo_handler.py`: Async MongoDB operations
  - Connection management
  - Collection setup with schema validation
  - Message persistence
  - Session history retrieval
  - Query operations

**Schema**:
```json
{
  "session_id": "string",
  "timestamp": "datetime",
  "role": "user|assistant|system",
  "content": "string",
  "tools_used": ["array"],
  "metadata": {}
}
```

### 3. Configuration (`src/config/`)

**Purpose**: Application settings and environment management

**Files**:
- `settings.py`: Pydantic-based settings
  - Environment variable loading
  - Type validation
  - Default values
  - API key management

### 4. User Interface (`src/ui/`)

**Purpose**: Gradio web interface

**Files**:
- `app.py`: Complete Gradio application
  - Chat interface with history
  - Server selection and management
  - Configuration UI
  - MongoDB setup interface
  - LLM integration with tool calling

**Features**:
- Real-time chat with AI
- Server connection status
- Tool usage transparency
- Session management
- MongoDB integration toggle

## 🔄 Data Flow

### Chat Flow
```
User Input → Gradio UI → MCPulseApp.chat()
                            ↓
                    Get connected servers
                            ↓
                    Aggregate available tools
                            ↓
                    Build LLM messages with tools
                            ↓
                    Call LLM (OpenAI/Anthropic)
                            ↓
                    LLM requests tool execution
                            ↓
                    Execute via MCP Client → Server
                            ↓
                    Return results to LLM
                            ↓
                    Generate final response
                            ↓
                    Save to MongoDB (optional)
                            ↓
                    Display in UI
```

### Connection Flow
```
User adds server → SessionManager.add_server()
                        ↓
                Save to mcp_servers.json
                        ↓
User selects server → SessionManager.connect_server()
                        ↓
                MCPClient.connect()
                        ↓
                Establish SSE connection
                        ↓
                Send initialize JSON-RPC
                        ↓
                Discover capabilities
                        ↓
                Tools available for chat
```

## 🛠️ Technology Stack

### Core
- **Python 3.8+**: Programming language
- **Gradio 4.0+**: Web UI framework
- **MCP Python SDK 1.2+**: Model Context Protocol client

### LLM Providers
- **OpenAI**: GPT-4 Turbo and other models
- **Anthropic**: Claude 3.5 Sonnet and other models

### Database (Optional)
- **MongoDB**: Chat history persistence
- **Motor**: Async MongoDB driver

### Utilities
- **Pydantic**: Settings and validation
- **python-dotenv**: Environment variables
- **httpx/aiohttp**: HTTP clients

## 📦 Installation

### Quick Start
```bash
# Clone repository
git clone <repo-url>
cd mcpulse

# Run setup script
./setup.sh

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run application
source venv/bin/activate
python main.py
```

### Manual Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## 🚀 Usage

### Starting the Application
```bash
python main.py
# Opens at http://localhost:7860
```

### Connecting to MCP Servers

1. **Add Server** (Configuration tab):
   - Name: My Server
   - URL: http://localhost:8000/sse
   - Description: (optional)

2. **Connect** (Chat tab):
   - Select server from checkboxes
   - Click "Connect Selected"

3. **Chat**:
   - Type message
   - AI automatically uses tools when needed

### Example with Test Server

Terminal 1:
```bash
python examples/simple_mcp_server.py
```

Terminal 2:
```bash
python main.py
```

In MCPulse UI:
1. Add server: http://localhost:8000/sse
2. Connect to "Test Server"
3. Ask: "What is 42 plus 58?"
4. AI uses calculator tool automatically

## 🧪 Testing

### Manual Testing
```bash
# Start application
python main.py

# In browser: http://localhost:7860
# Follow testing checklist in TESTING.md
```

### Example Server Testing
```bash
# Terminal 1: Run test server
python examples/simple_mcp_server.py

# Terminal 2: Run application
python main.py

# Connect and test tools
```

### Automated Testing (Future)
```bash
pytest tests/
```

## 📝 Configuration

### Environment Variables (.env)
```env
# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# MongoDB (optional)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=mcpulse
MONGODB_COLLECTION=chat_history

# Application
DEFAULT_MODEL=gpt-4-turbo-preview
LOG_LEVEL=INFO
GRADIO_SERVER_PORT=7860
```

### MCP Servers (config/mcp_servers.json)
```json
{
  "servers": [
    {
      "name": "server-name",
      "url": "http://localhost:8000/sse",
      "description": "Server description",
      "enabled": true
    }
  ]
}
```

## 🔐 Security

- ✅ API keys in .env (not committed)
- ✅ No hardcoded credentials
- ✅ Input validation on all user inputs
- ✅ MongoDB authentication support
- ⚠️ MCP connections currently without auth (add if needed)
- ⚠️ Run in trusted network or add HTTPS

## 🎯 Use Cases

1. **Development Tool**: Connect to MCP development servers for testing
2. **API Gateway**: Unified interface for multiple tool providers
3. **Research**: Experiment with tool-augmented LLM interactions
4. **Automation**: Chat-based interface for automation tools
5. **Integration Hub**: Connect various services via MCP protocol

## 🚧 Future Enhancements

- [ ] Streaming LLM responses
- [ ] Multi-user support with authentication
- [ ] Custom prompt templates library
- [ ] Rich media support (images, files)
- [ ] PostgreSQL support
- [ ] WebSocket alternative to SSE
- [ ] Mobile-responsive UI improvements
- [ ] Plugin system for extensions
- [ ] Usage analytics and metrics
- [ ] Conversation export/import

## 📖 Documentation

- **README.md**: Main documentation and overview
- **QUICKSTART.md**: Installation and first steps
- **ARCHITECTURE.md**: Technical deep dive
- **TESTING.md**: Comprehensive testing guide
- **CONTRIBUTING.md**: How to contribute
- **examples/README.md**: Example MCP servers

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Development setup
- Testing requirements
- Pull request process

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Gradio](https://www.gradio.app/)
- [OpenAI](https://openai.com/)
- [Anthropic](https://www.anthropic.com/)
- [MongoDB](https://www.mongodb.com/)

## 📞 Support

- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Documentation**: See docs/ directory

---

**Status**: ✅ Production Ready  
**Version**: 0.1.0  
**Last Updated**: 2025-10-23  

Built with ❤️ for the MCP community
