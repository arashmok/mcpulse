# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) MongoDB for chat history persistence

## Installation

### Linux/macOS

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Edit .env file with your API keys
nano .env
```

### Windows

```cmd
# Run the setup script
setup.bat

# Activate virtual environment
venv\Scripts\activate.bat

# Edit .env file with your API keys
notepad .env
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your settings
```

## Configuration

### 1. API Keys

Edit `.env` and add at least one LLM API key:

```env
# Choose one or both
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. MCP Servers

Add MCP servers through the UI or edit `config/mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "my-server",
      "url": "http://localhost:8000/sse",
      "description": "My MCP Server",
      "enabled": true
    }
  ]
}
```

### 3. MongoDB (Optional)

If you want to persist chat history:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=mcpulse
MONGODB_COLLECTION=chat_history
```

Then use the "Setup MongoDB Collection" button in the Configuration tab.

## Running the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
python main.py
```

The application will start at: **http://localhost:7860**

## First Steps

1. **Configure MCP Servers**
   - Go to the "Configuration" tab
   - Add your MCP server(s) with their SSE endpoints
   - Example: `http://localhost:8000/sse`

2. **Connect to Servers**
   - Go to the "Chat" tab
   - Select servers from the "Active Servers" checkbox list
   - Click "Connect Selected"
   - Wait for connection confirmation

3. **Start Chatting**
   - Type your message in the input box
   - The AI will automatically use tools from connected servers
   - Tool usage is transparent and shown in responses

## Creating a Test MCP Server

See `examples/simple_mcp_server.py` for a basic MCP server implementation you can use for testing.

```bash
# In a separate terminal
cd examples
python simple_mcp_server.py
```

Then add this server in MCPulse:
- Name: Test Server
- URL: http://localhost:8000/sse
- Description: Simple test server

## Troubleshooting

### "No LLM API key configured"
- Add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file
- Restart the application

### "Failed to connect to MCP server"
- Verify the server URL is correct
- Check that the MCP server is running
- Ensure the URL includes the `/sse` endpoint

### "MongoDB connection failed"
- Verify MongoDB is running
- Check the connection URI in .env
- Ensure you have proper permissions

### Import errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Advanced Usage

### System Prompts

Customize the AI behavior using system prompts in the chat interface. Example:

```
You are a software development assistant. When users ask about code,
always provide well-documented examples and explain your reasoning.
```

### Multiple Servers

You can connect to multiple MCP servers simultaneously. Each server's tools will be available with a prefix: `servername_toolname`

### Session Management

- Click "New Session" to start fresh (previous chat stays in UI until cleared)
- If MongoDB is enabled, all sessions are stored and can be retrieved later

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Check out [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Explore [examples/](examples/) for MCP server examples
