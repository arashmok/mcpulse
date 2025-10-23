# 🚀 Getting Started with MCPulse

Welcome to MCPulse! This guide will help you get up and running in minutes.

## 📋 What You'll Need

- **Python 3.8+** installed on your system
- **pip** (Python package manager)
- An **API key** from OpenAI or Anthropic
- (Optional) **MongoDB** for chat history persistence
- (Optional) An **MCP server** to connect to (we provide a test server)

## 🎯 Quick Start (5 minutes)

### Step 1: Get the Code

```bash
git clone <your-repo-url>
cd mcpulse
```

### Step 2: Run Setup

**On Linux/macOS:**
```bash
./setup.sh
```

**On Windows:**
```cmd
setup.bat
```

**Or manually:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit it with your favorite editor
nano .env  # or vim, code, notepad, etc.
```

Add your API key:
```env
# Choose ONE of these (or both)
OPENAI_API_KEY=sk-proj-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 4: Verify Installation

```bash
python verify_installation.py
```

You should see ✅ for all checks!

### Step 5: Start the Application

```bash
python main.py
```

You'll see:
```
INFO - Starting MCPulse application
INFO - Gradio server will run on 0.0.0.0:7860
Running on local URL:  http://127.0.0.1:7860
```

Open your browser to: **http://localhost:7860** 🎉

## 🧪 Try It Out (with Test Server)

Let's test MCPulse with our example calculator server:

### Terminal 1: Start Test MCP Server

```bash
# In the mcpulse directory
source venv/bin/activate  # Windows: venv\Scripts\activate
python examples/simple_mcp_server.py
```

You should see:
```
🧮 MCP Test Calculator Server
...
Server starting on http://localhost:8000
SSE endpoint: http://localhost:8000/sse
```

### Terminal 2: Start MCPulse

```bash
# In a NEW terminal, in the mcpulse directory
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py
```

### In Your Browser (http://localhost:7860)

1. **Go to the "Configuration" tab**

2. **Add the test server:**
   - Server Name: `Test Server`
   - Server URL: `http://localhost:8000/sse`
   - Description: `Calculator and utilities`
   - Click **"Add Server"**
   
   You should see: ✅ Server 'Test Server' added successfully

3. **Go back to the "Chat" tab**

4. **Connect to the server:**
   - Check the box next to **"Test Server"** in "Active Servers"
   - Click **"Connect Selected"**
   
   You should see: ✅ Test Server: Connected (7 tools available)

5. **Start chatting!** Try these:

   ```
   What is 42 plus 58?
   ```
   
   The AI will use the `add` tool and respond: "The result is 100"
   
   ```
   Can you reverse the string "Hello World"?
   ```
   
   The AI will use the `reverse_string` tool: "dlroW olleH"
   
   ```
   Count the words in: The quick brown fox jumps
   ```
   
   The AI will use `count_words` and tell you there are 5 words

## 🎓 Understanding How It Works

When you ask "What is 42 plus 58?":

1. **Your message** goes to MCPulse
2. **MCPulse** sends it to the LLM (OpenAI/Anthropic) along with available tools
3. **The LLM** recognizes it needs to use the `add` tool
4. **MCPulse** calls the tool on the MCP server: `add(42, 58)`
5. **The server** responds with `100`
6. **The LLM** generates a natural response: "The result is 100"
7. **You see** the final answer in the chat

All of this happens automatically! 🪄

## 🔧 Next Steps

### Connect to Real MCP Servers

Replace the test server with actual MCP servers:

1. Find or build an MCP server (see [MCP Documentation](https://modelcontextprotocol.io/))
2. Add it in the Configuration tab
3. Connect and start using its tools

### Enable MongoDB for Chat History

If you want to keep your chat history:

1. **Install MongoDB:**
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install -y mongodb
   sudo systemctl start mongodb
   ```
   
   **macOS:**
   ```bash
   brew tap mongodb/brew
   brew install mongodb-community
   brew services start mongodb-community
   ```
   
   **Windows or Docker:**
   ```bash
   docker run -d -p 27017:27017 mongo:latest
   ```

2. **Configure in .env:**
   ```env
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DATABASE=mcpulse
   MONGODB_COLLECTION=chat_history
   ```

3. **Setup in MCPulse:**
   - Go to Configuration tab
   - Click **"Setup MongoDB Collection"**
   - Check **"Enable MongoDB for chat history"**
   
   Now your chats are saved! 💾

### Customize System Prompt

In the Chat tab, edit the "System Prompt" to change AI behavior:

```
You are a helpful coding assistant specialized in Python.
When providing code, always include comments and explanations.
```

### Add Multiple Servers

You can connect to multiple MCP servers at once:

1. Add Server 1: http://localhost:8000/sse
2. Add Server 2: http://localhost:8001/sse
3. Select both in the chat
4. All tools from both servers are available!

## 📚 Learn More

- **[README.md](README.md)** - Complete documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - How MCPulse works internally
- **[TESTING.md](TESTING.md)** - Full testing guide
- **[examples/README.md](examples/README.md)** - Build your own MCP server

## 🆘 Troubleshooting

### "No LLM API key configured"

**Problem**: You see this warning in the terminal

**Solution**:
```bash
# Edit .env file
nano .env

# Add your API key
OPENAI_API_KEY=sk-proj-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Restart application
```

### Cannot connect to MCP server

**Problem**: Connection fails with error

**Check**:
1. Is the server running? (check other terminal)
2. Is the URL correct? (must include `/sse`)
3. Is the port available?
   ```bash
   lsof -i :8000  # Check if port is in use
   ```

### Port 7860 already in use

**Problem**: Gradio can't start

**Solution**: Change port in .env
```env
GRADIO_SERVER_PORT=7861
```

Or stop the other application using port 7860.

### Import errors

**Problem**: `ModuleNotFoundError: No module named 'gradio'`

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Windows: venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### MongoDB connection failed

**Problem**: Can't connect to MongoDB

**Check**:
```bash
# Is MongoDB running?
sudo systemctl status mongodb  # Linux
brew services list | grep mongo  # macOS
docker ps | grep mongo  # Docker

# If not running, start it
sudo systemctl start mongodb  # Linux
brew services start mongodb-community  # macOS
```

## 💡 Tips & Tricks

1. **Multiple chat sessions**: Click "New Session" to start fresh without losing current chat

2. **Clear chat**: Use "Clear Chat" button to clear just the visible messages

3. **Tool transparency**: The AI will often explain which tools it's using

4. **System prompts**: Experiment with different system prompts to change behavior

5. **Server management**: You can disable servers without deleting them

6. **Logs**: Check `mcpulse.log` file for detailed debugging information

## 🎉 You're Ready!

You now have:
- ✅ MCPulse running
- ✅ Connected to an MCP server
- ✅ Successfully used AI tools
- ✅ Understanding of the workflow

**What's next?**
- Build your own MCP server
- Connect to real-world tools and APIs
- Integrate with your existing systems
- Share your experience with the community

Happy building! 🚀

---

**Need help?** 
- 📖 Check the [documentation](README.md)
- 🐛 [Report issues](https://github.com/your-repo/issues)
- 💬 Join the discussion

**Want to contribute?**
- 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md)
