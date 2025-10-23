# Architecture Overview

## System Design

MCPulse is designed as a modular application with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                     │
│                    (User Interaction Layer)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─────────────────┐
                       │                 │
┌──────────────────────▼─────┐  ┌────────▼──────────────────┐
│   Session Manager           │  │   MongoDB Handler         │
│   (MCP Connection Mgmt)     │  │   (Chat Persistence)      │
└──────────────────────┬──────┘  └───────────────────────────┘
                       │
                       ├─────────────────┐
                       │                 │
┌──────────────────────▼─────┐  ┌────────▼──────────────────┐
│   MCP Client 1              │  │   MCP Client 2            │
│   (Server Connection)       │  │   (Server Connection)     │
└──────────────────────┬──────┘  └────────┬──────────────────┘
                       │                  │
                ┌──────▼──────────────────▼──────┐
                │        SSE Transport            │
                │     (JSON-RPC over SSE)         │
                └──────┬──────────────────┬───────┘
                       │                  │
                ┌──────▼─────┐     ┌──────▼─────┐
                │ MCP Server 1│     │ MCP Server 2│
                └─────────────┘     └─────────────┘
```

## Component Breakdown

### 1. UI Layer (`src/ui/`)

**Purpose**: Gradio-based web interface for user interaction

**Key Files**:
- `app.py`: Main Gradio application with chat and configuration interfaces

**Responsibilities**:
- Render chat interface
- Manage server selection UI
- Handle user input/output
- Display connection status
- Configuration management UI

### 2. Client Layer (`src/client/`)

**Purpose**: MCP protocol implementation and connection management

**Key Files**:
- `mcp_client.py`: Individual MCP server client using SSE transport
- `session_manager.py`: Manages multiple MCP connections and configuration

**Responsibilities**:
- Establish SSE connections to MCP servers
- Implement JSON-RPC 2.0 protocol
- Discover and manage tools/resources/prompts
- Execute tool calls
- Handle reconnection logic

**Key Classes**:

```python
class MCPClient:
    """Represents a connection to a single MCP server"""
    - connect() / disconnect()
    - call_tool()
    - read_resource()
    - get_prompt()
    
class SessionManager:
    """Manages multiple MCP client instances"""
    - add_server() / remove_server()
    - connect_server() / disconnect_server()
    - get_all_tools()
    - Configuration persistence
```

### 3. Database Layer (`src/database/`)

**Purpose**: Optional MongoDB integration for chat history

**Key Files**:
- `mongo_handler.py`: Async MongoDB operations

**Responsibilities**:
- Connect to MongoDB
- Create collections with proper schema
- Save/retrieve chat messages
- Session management
- Query chat history

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

### 4. Configuration Layer (`src/config/`)

**Purpose**: Application settings and environment management

**Key Files**:
- `settings.py`: Pydantic settings with environment variable loading

**Responsibilities**:
- Load .env variables
- Provide typed configuration
- Default values
- Validation

## Data Flow

### Chat Message Flow

```
1. User types message in Gradio
   ↓
2. UI calls chat() method
   ↓
3. Message saved to MongoDB (if enabled)
   ↓
4. Session Manager retrieves tools from connected servers
   ↓
5. Message + tools sent to LLM (OpenAI/Anthropic)
   ↓
6. LLM may request tool execution
   ↓
7. Tools executed via MCP Client → Server
   ↓
8. Results returned to LLM
   ↓
9. Final response generated
   ↓
10. Response saved to MongoDB and displayed
```

### MCP Connection Flow

```
1. User adds server in Configuration tab
   ↓
2. Configuration saved to mcp_servers.json
   ↓
3. User selects server in Chat tab
   ↓
4. SessionManager.connect_server() called
   ↓
5. MCPClient initialized with server URL
   ↓
6. SSE connection established
   ↓
7. JSON-RPC initialize() sent
   ↓
8. Server capabilities discovered (tools/resources/prompts)
   ↓
9. Connection marked as active
   ↓
10. Tools available for use in chat
```

## Protocol Details

### MCP SSE Transport

MCPulse implements the MCP SSE (Server-Sent Events) transport:

- **Client → Server**: HTTP POST to `/message` endpoint with JSON-RPC payloads
- **Server → Client**: SSE stream on `/sse` endpoint for notifications

**JSON-RPC Message Format**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {}
  }
}
```

### Tool Execution

When the LLM wants to use a tool:

1. Tool call is intercepted by MCPulse
2. Tool name is mapped to server (format: `servername_toolname`)
3. JSON-RPC request sent to appropriate MCP server
4. Result returned to LLM for processing
5. LLM incorporates result in final response

## Extensibility Points

### Adding New LLM Providers

1. Add provider-specific client initialization in `MCPulseApp._initialize_llm()`
2. Implement provider-specific call method (e.g., `_call_openai()`)
3. Update requirements.txt with provider SDK

### Adding New Storage Backends

1. Create new handler in `src/database/` (e.g., `postgres_handler.py`)
2. Implement same interface as `MongoHandler`
3. Update UI to allow backend selection

### Custom MCP Transports

1. MCP SDK supports stdio and SSE
2. For custom transport, extend `ClientSession` with new stream classes
3. Update `MCPClient` to support transport selection

## Security Considerations

1. **API Keys**: Stored in .env (not committed to git)
2. **MCP Connections**: Currently no authentication - add auth headers if needed
3. **MongoDB**: Use connection strings with authentication
4. **Input Validation**: All user inputs validated before processing
5. **Tool Execution**: Tools run with server permissions - ensure MCP servers are trusted

## Performance Optimization

1. **Async Operations**: All I/O operations are async (MongoDB, MCP, LLM)
2. **Connection Pooling**: MongoDB uses Motor for async connection pooling
3. **Lazy Loading**: MCP connections only established when selected
4. **Streaming**: SSE provides efficient real-time updates

## Error Handling

- **Connection Failures**: Graceful degradation, user notified
- **Tool Errors**: Captured and reported to LLM for handling
- **MongoDB Errors**: Optional feature, app continues without it
- **LLM Errors**: Displayed to user with retry option

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (MCP servers, MongoDB, LLM APIs)

### Integration Tests
- Test component interactions
- Use local test MCP server
- Test MongoDB operations with test database

### E2E Tests
- Test complete user workflows
- Automated browser testing with Gradio interface

## Future Enhancements

1. **Multi-user Support**: Session isolation, user authentication
2. **Streaming Responses**: Stream LLM responses token-by-token
3. **Rich Media**: Support for images, files in chat
4. **Prompt Library**: Pre-built prompts for common tasks
5. **Analytics**: Usage statistics, tool call metrics
6. **Plugin System**: Allow custom extensions
7. **Mobile App**: React Native or Flutter client
