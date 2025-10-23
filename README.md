# MCPulse - MCP SSE Client with Gradio Interface

A ChatGPT-like web application built with Gradio that acts as an MCP (Model Context Protocol) SSE client. Connect to multiple MCP servers and leverage their tools in your conversations.

## Features

- 🔌 **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- 💬 **ChatGPT-like Interface**: Clean, intuitive chat interface powered by Gradio
- 🛠️ **Tool Integration**: Automatically discover and use tools from connected MCP servers
- 💾 **Optional MongoDB Storage**: Store chat history in MongoDB for persistence
- ⚙️ **Easy Configuration**: Simple UI for managing MCP server connections
- 🔄 **Real-time Updates**: SSE transport for responsive server communication

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mcpulse
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Configuration

### MCP Servers

MCP servers can be configured through the UI or by editing `config/mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "My MCP Server",
      "url": "http://localhost:8000/sse",
      "description": "Local MCP server",
      "enabled": true
    }
  ]
}
```

### MongoDB (Optional)

To enable chat history storage:

1. Set MongoDB connection details in `.env`
2. Use the "Setup MongoDB" button in the UI to create the collection
3. Chat history will automatically be saved and can be loaded later

**MongoDB Schema:**
```json
{
  "_id": "ObjectId",
  "session_id": "string",
  "timestamp": "datetime",
  "role": "user|assistant|system",
  "content": "string",
  "tools_used": ["array"],
  "metadata": {}
}
```

## Usage

### Starting the Application

```bash
python main.py
```

The Gradio interface will launch at `http://localhost:7860`

### Using the Interface

1. **Configure MCP Servers:**
   - Go to the "Configuration" tab
   - Add your MCP server URLs
   - Enable/disable servers as needed

2. **Start Chatting:**
   - Select which MCP servers to use in the chat
   - Type your message and click Send
   - The assistant will automatically use available tools when needed

3. **MongoDB Setup (Optional):**
   - Configure MongoDB connection in `.env`
   - Click "Setup MongoDB Collection" in the Configuration tab
   - Enable "Use MongoDB" to persist chat history

## Project Structure

```
mcpulse/
├── src/
│   ├── client/          # MCP client implementation
│   │   ├── __init__.py
│   │   ├── mcp_client.py
│   │   └── session_manager.py
│   ├── database/        # MongoDB integration
│   │   ├── __init__.py
│   │   └── mongo_handler.py
│   ├── ui/             # Gradio interface
│   │   ├── __init__.py
│   │   └── app.py
│   └── config/         # Configuration management
│       ├── __init__.py
│       └── settings.py
├── config/             # Configuration files
│   └── mcp_servers.json
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
├── .env.example       # Example environment variables
└── README.md         # This file
```

## Development

### Adding New Features

The project is organized into modular components:

- **MCP Client** (`src/client/`): Handles connections to MCP servers
- **Database** (`src/database/`): MongoDB integration for chat history
- **UI** (`src/ui/`): Gradio interface components
- **Config** (`src/config/`): Application settings and configuration

### Running Tests

```bash
# TODO: Add test suite
pytest tests/
```

## Troubleshooting

### MCP Server Connection Issues

- Verify the server URL is correct and includes `/sse` endpoint
- Check that the MCP server is running and accessible
- Review server logs for connection errors

### MongoDB Connection Issues

- Ensure MongoDB is running
- Verify connection string in `.env`
- Check database/collection permissions

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Gradio](https://www.gradio.app/)
- [MongoDB](https://www.mongodb.com/)
