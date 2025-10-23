# MCP Server Examples

This directory contains example MCP servers for testing MCPulse.

## Simple Calculator Server

A basic MCP server that provides calculator and utility tools.

### Running the Server

```bash
# Install fastmcp if not already installed
pip install "mcp[cli]"

# Run the server
python simple_mcp_server.py
```

The server will start on `http://localhost:8000` with SSE endpoint at `/sse`.

### Available Tools

- **add(a, b)** - Add two numbers
- **subtract(a, b)** - Subtract b from a
- **multiply(a, b)** - Multiply two numbers
- **divide(a, b)** - Divide a by b
- **get_current_time()** - Get current time
- **reverse_string(text)** - Reverse a string
- **count_words(text)** - Count words and characters in text

### Available Resources

- **greeting://hello** - Get a friendly greeting
- **info://server** - Get server information

### Connecting from MCPulse

1. Start the example server
2. Open MCPulse web interface
3. Go to Configuration tab
4. Add new server:
   - **Name**: Test Server
   - **URL**: http://localhost:8000/sse
   - **Description**: Simple calculator server for testing
5. Go to Chat tab
6. Select "Test Server" from Active Servers
7. Click "Connect Selected"
8. Try asking: "What's 42 plus 58?" or "Reverse the string 'hello world'"

## Creating Your Own MCP Server

### Using FastMCP (Recommended)

```python
from mcp.server.fastmcp import FastMCP

# Create server instance
mcp = FastMCP("My Custom Server")

# Define a tool
@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description."""
    return f"Processed: {param}"

# Define a resource
@mcp.resource("my://resource")
def my_resource() -> str:
    """Resource description."""
    return "Resource content"

# Run with SSE transport
if __name__ == "__main__":
    mcp.run(transport="sse")
```

### Using Python SDK Directly

```python
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

app = Server("my-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="My tool description",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "my_tool":
        result = f"Processed: {arguments['param']}"
        return [TextContent(type="text", text=result)]

# Run with SSE
async def main():
    async with SseServerTransport("/sse") as transport:
        await app.run(transport)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Testing Your Server

### Manual Testing with curl

```bash
# Check if server is running
curl http://localhost:8000/health

# Open SSE connection (in one terminal)
curl -N -H "Accept: text/event-stream" http://localhost:8000/sse

# Send a message (in another terminal)
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
  }'
```

### Testing with MCPulse

The easiest way to test your MCP server is to connect it to MCPulse and try using its tools through natural language chat.

## More Examples

Check out the official MCP documentation for more examples:
- https://modelcontextprotocol.io/docs/develop/build-server
- https://github.com/modelcontextprotocol/python-sdk/tree/main/examples

## Common Issues

### Port Already in Use

If port 8000 is already in use, you can change it:

```python
# In your server script
mcp.run(transport="sse", port=8001)
```

Then update the URL in MCPulse to `http://localhost:8001/sse`.

### FastMCP Not Found

Install the MCP SDK with CLI support:

```bash
pip install "mcp[cli]"
```

### Connection Refused

- Make sure the server is running
- Check firewall settings
- Verify the URL includes the `/sse` endpoint
- Check server logs for errors
