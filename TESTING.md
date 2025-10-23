# Testing Guide

This guide helps you test MCPulse and verify everything is working correctly.

## Quick Test Checklist

- [ ] Installation completed successfully
- [ ] Environment variables configured
- [ ] Application starts without errors
- [ ] UI is accessible in browser
- [ ] Can add MCP server configuration
- [ ] Can connect to MCP server
- [ ] Can send chat messages
- [ ] Tools are discovered and usable
- [ ] MongoDB setup works (if using)
- [ ] Chat history persists (if MongoDB enabled)

## Step-by-Step Testing

### 1. Installation Test

```bash
# Run setup script
./setup.sh

# Verify virtual environment
source venv/bin/activate
python --version  # Should be 3.8+
pip list | grep gradio  # Should show gradio installed
```

**Expected**: No errors, all dependencies installed

### 2. Configuration Test

```bash
# Check .env file exists
cat .env

# Verify at least one API key is set
grep -E "OPENAI_API_KEY|ANTHROPIC_API_KEY" .env
```

**Expected**: .env file exists with at least one API key configured

### 3. Start Application Test

```bash
python main.py
```

**Expected Output**:
```
INFO - Starting MCPulse application
INFO - Gradio server will run on 0.0.0.0:7860
INFO - Initialized OpenAI client (or Anthropic)
Running on local URL:  http://127.0.0.1:7860
```

**What to Check**:
- No Python errors
- Server starts successfully
- URL is displayed
- Can access UI in browser at http://localhost:7860

### 4. UI Test

Open http://localhost:7860 in your browser.

**Check**:
- [ ] Page loads correctly
- [ ] "Chat" and "Configuration" tabs visible
- [ ] Chat interface shows empty conversation
- [ ] Server selection area visible
- [ ] System prompt input visible

### 5. Test MCP Server (Example Server)

In a **separate terminal**:

```bash
# Activate virtual environment
source venv/bin/activate

# Install fastmcp if needed
pip install "mcp[cli]"

# Run example server
python examples/simple_mcp_server.py
```

**Expected Output**:
```
🧮 MCP Test Calculator Server
...
Server starting on http://localhost:8000
SSE endpoint: http://localhost:8000/sse
```

### 6. Add Server in MCPulse

In the MCPulse UI:

1. Go to **Configuration** tab
2. Fill in:
   - **Server Name**: Test Server
   - **Server URL**: http://localhost:8000/sse
   - **Description**: Test calculator server
3. Click **Add Server**

**Expected**: ✅ Server 'Test Server' added successfully

### 7. Connect to Server

In the **Chat** tab:

1. Check **Test Server** in "Active Servers"
2. Click **Connect Selected**

**Expected**: 
```
✅ Test Server: Connected (7 tools available)
```

### 8. Test Chat Without Tools

Type in chat: `Hello! Can you introduce yourself?`

**Expected**: 
- AI responds with a greeting
- No tool usage needed
- Response appears in chat

### 9. Test Chat With Tools

Type in chat: `What is 42 plus 58?`

**Expected**:
- AI recognizes it needs to use the `add` tool
- Tool is called automatically
- Response includes the correct answer: 100

### 10. Test Multiple Tool Calls

Type: `What's 100 divided by 5, and then multiply the result by 3?`

**Expected**:
- Multiple tools called (`divide`, then `multiply`)
- Correct final answer: 60

### 11. Test String Manipulation

Type: `Can you reverse the string "MCPulse" for me?`

**Expected**:
- `reverse_string` tool called
- Response includes: "esluPCM"

### 12. Test Word Counting

Type: `Count the words in this sentence: "The quick brown fox jumps over the lazy dog"`

**Expected**:
- `count_words` tool called
- Response shows: 9 words

### 13. Test MongoDB Setup (Optional)

If you have MongoDB installed:

1. Start MongoDB:
   ```bash
   # On Linux
   sudo systemctl start mongod
   
   # On macOS
   brew services start mongodb-community
   
   # Or use Docker
   docker run -d -p 27017:27017 mongo:latest
   ```

2. In **Configuration** tab:
   - Verify MongoDB settings:
     - URI: mongodb://localhost:27017
     - Database: mcpulse
     - Collection: chat_history
   - Click **Setup MongoDB Collection**

**Expected**: ✅ Collection 'chat_history' created successfully...

3. Enable MongoDB:
   - Check **Enable MongoDB for chat history**

**Expected**: MongoDB: Enabled

4. Send a few chat messages
5. Check MongoDB:
   ```bash
   mongosh
   use mcpulse
   db.chat_history.find().pretty()
   ```

**Expected**: Your chat messages are stored in MongoDB

### 14. Test New Session

1. Click **New Session**

**Expected**:
- Chat clears
- New session ID shown
- If MongoDB enabled, new session ID used for storage

### 15. Test Server Management

**Add Second Server** (if you have one):
1. Go to Configuration
2. Add another server with different name/URL
3. Verify it appears in the server list

**Remove Server**:
1. Select server from dropdown in "Remove Server" section
2. Click Remove Server
3. Verify it's removed from all lists

## Common Test Failures

### Application Won't Start

**Error**: `ModuleNotFoundError: No module named 'mcp'`

**Fix**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "No LLM API key configured"

**Error**: Warning about missing API key

**Fix**:
```bash
# Edit .env file
nano .env

# Add your key
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Restart application
```

### Cannot Connect to MCP Server

**Error**: ❌ Test Server: Failed to connect

**Check**:
1. Is example server running? Check the terminal
2. Is URL correct? Should be `http://localhost:8000/sse`
3. Is port 8000 available?
   ```bash
   lsof -i :8000
   ```

### Tools Not Working

**Error**: AI doesn't use tools or says tools not available

**Check**:
1. Server is connected (green checkmark)
2. Tools are listed in connection status
3. Try asking more explicitly: "Use the add tool to calculate 5 + 3"

### MongoDB Connection Failed

**Error**: ❌ Failed to connect to MongoDB

**Fix**:
```bash
# Check if MongoDB is running
systemctl status mongod
# or
brew services list | grep mongodb
# or
docker ps | grep mongo

# Start if not running
sudo systemctl start mongod
# or
brew services start mongodb-community
```

## Performance Testing

### Response Time

Normal response times:
- Server connection: 1-3 seconds
- Chat without tools: 2-5 seconds
- Chat with 1 tool: 3-7 seconds
- Chat with multiple tools: 5-15 seconds

### Concurrent Connections

Test multiple servers:
1. Run 2-3 example servers on different ports
2. Connect to all simultaneously
3. Verify all tools are available
4. Test tools from different servers

## Integration Testing

### Test Full Workflow

```python
# test_workflow.py
import asyncio
from src.client import SessionManager
from src.database import MongoHandler

async def test_full_workflow():
    # Initialize
    sm = SessionManager()
    
    # Add server
    result = sm.add_server("test", "http://localhost:8000/sse", "Test")
    assert result["success"]
    
    # Connect
    result = await sm.connect_server("test")
    assert result["success"]
    
    # Call tool
    result = await sm.call_tool("test", "add", {"a": 5, "b": 3})
    assert result["success"]
    assert result["content"] == 8
    
    # Cleanup
    await sm.cleanup()
    print("✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_full_workflow())
```

Run: `python test_workflow.py`

## Automated Testing

### Unit Tests (Future)

```bash
# Install pytest
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Load Testing (Future)

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py
```

## Reporting Issues

When reporting issues, include:

1. **Environment**:
   - OS and version
   - Python version
   - Package versions: `pip list`

2. **Steps to Reproduce**:
   - Exact commands run
   - Configuration used
   - Expected vs actual behavior

3. **Logs**:
   - Terminal output
   - mcpulse.log file contents
   - Browser console errors (F12)

4. **Screenshots**: If UI-related

## Success Criteria

MCPulse is working correctly when:

✅ Application starts without errors  
✅ UI is responsive and functional  
✅ Can configure and connect to MCP servers  
✅ AI uses tools appropriately  
✅ Tool results are incorporated in responses  
✅ Multiple servers can be used simultaneously  
✅ MongoDB integration works (if enabled)  
✅ Sessions can be started/cleared  
✅ No data loss or corruption  

## Next Steps

After successful testing:

1. Deploy to production environment
2. Connect to real MCP servers
3. Customize system prompts
4. Set up MongoDB for production
5. Monitor logs and performance
6. Collect user feedback
