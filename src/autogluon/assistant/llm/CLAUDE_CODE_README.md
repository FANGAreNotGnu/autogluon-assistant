# Claude Code Agent Integration

This integration provides access to the full Claude Code agentic system, leveraging its built-in capabilities instead of reimplementing them. Claude Code is an advanced AI agent with extensive tool support and problem-solving capabilities.

## What is Claude Code?

Claude Code is Anthropic's official CLI and agent system that provides:
- **Web Search** - Real-time internet search capabilities
- **Code Execution** - Safe execution of Bash commands, Python scripts, etc.
- **File Operations** - Read, Write, Edit, Glob, Grep operations
- **Task Spawning** - Parallel execution of sub-tasks
- **MCP Integration** - Connect to Model Context Protocol servers
- **Custom Tools & Skills** - Extensible tool system

## Architecture

This implementation uses the **Claude Agent SDK** to programmatically invoke Claude Code:

```
Your Application
       ↓
ClaudeCodeAgent (this module)
       ↓
Claude Agent SDK / CLI
       ↓
Claude Model (via Anthropic API or Bedrock)
       ↓
Built-in Tools (web search, bash, file ops, etc.)
```

## Prerequisites

### Install Claude Code and Claude Agent SDK (Required)

**Step 1: Install Claude Code CLI**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Step 2: Install Claude Agent SDK**
```bash
pip install claude-agent-sdk
```

**Note**: Both Claude Code CLI and the Claude Agent SDK are required for the claude-code provider.

## Configuration

### Using Anthropic API

```yaml
provider: claude-code
backend: anthropic  # Uses Anthropic API (default)
model: claude-3-5-sonnet-20241022
max_tokens: 4096
temperature: 0.7
verbose: false
```

### Using AWS Bedrock

```yaml
provider: claude-code
backend: bedrock  # Uses AWS Bedrock
model: claude-3-5-sonnet-20241022
max_tokens: 4096
temperature: 0.7
verbose: false
aws_region: us-west-2  # Optional, defaults to AWS_DEFAULT_REGION
```

### Advanced Options

```yaml
provider: claude-code
backend: anthropic
model: claude-3-5-sonnet-20241022
max_tokens: 4096
temperature: 0.7
verbose: true
working_directory: /path/to/project  # Working dir for file operations
enabled_tools:  # Optional: specify which tools to enable
  - web_search
  - bash
  - read_file
  - write_file
```

## Environment Variables

### For Anthropic Backend

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### For Bedrock Backend

```bash
export AWS_DEFAULT_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

## Usage Examples

### Basic Usage

```python
from omegaconf import DictConfig
from autogluon.assistant.llm import ChatLLMFactory

# Configure Claude Code
config = DictConfig({
    "provider": "claude-code",
    "backend": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "temperature": 0.7,
    "verbose": True
})

# Create agent
agent = ChatLLMFactory.get_chat_model(config, session_name="my_session")

# Use the agent - Claude Code will autonomously use tools as needed
response = agent.assistant_chat(
    "Search for the latest Python 3.12 features and create a summary file"
)
print(response)
```

### Research Tasks (with Web Search)

```python
config = DictConfig({
    "provider": "claude-code",
    "backend": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "temperature": 0.3,  # Lower temp for factual tasks
})

agent = ChatLLMFactory.get_chat_model(config, session_name="research")

# Claude Code will automatically search the web
response = agent.assistant_chat(
    "What are the latest developments in quantum computing? "
    "Find recent papers and summarize the key breakthroughs."
)
```

### Code Analysis and Execution

```python
config = DictConfig({
    "provider": "claude-code",
    "backend": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 8192,
    "working_directory": "/path/to/my/project"
})

agent = ChatLLMFactory.get_chat_model(config, session_name="dev")

# Claude Code will read files, analyze code, and execute tests
response = agent.assistant_chat(
    "Analyze the codebase in this directory, find potential bugs, "
    "and run the test suite to verify correctness."
)
```

### Data Processing

```python
# Claude Code can execute Python for data analysis
response = agent.assistant_chat(
    "Read the CSV file at data/sales.csv, calculate monthly totals, "
    "create a visualization, and save the results."
)
```

### Multi-Step Tasks

```python
# Claude Code excels at complex multi-step tasks
response = agent.assistant_chat(
    """
    1. Search for the current weather API best practices
    2. Create a Python script to fetch weather data
    3. Test the script with a few cities
    4. Save the results to a JSON file
    """
)
```

## Features and Capabilities

### Built-in Tools (Automatically Used)

Claude Code has access to these tools and will use them autonomously:

1. **Web Search** - Search the internet using DuckDuckGo or other engines
2. **Bash** - Execute shell commands safely
3. **Read** - Read file contents
4. **Write** - Create or overwrite files
5. **Edit** - Make surgical edits to files
6. **Glob** - Find files by pattern
7. **Grep** - Search file contents
8. **LSP** - Language server protocol for code intelligence
9. **Task** - Spawn sub-agents for complex tasks

### Intelligent Tool Usage

Claude Code determines which tools to use based on the task:

```python
# This query will trigger web search
agent.assistant_chat("What's the current price of Bitcoin?")

# This will trigger file operations
agent.assistant_chat("Find all TODO comments in Python files")

# This will trigger code execution
agent.assistant_chat("Calculate the factorial of 100")

# This will trigger multiple tools in sequence
agent.assistant_chat("Search for best sorting algorithms, implement them, and benchmark")
```

### Token Tracking

Token usage is automatically tracked:

```python
agent = ChatLLMFactory.get_chat_model(config, session_name="my_session")

# Make some requests
agent.assistant_chat("Hello!")
agent.assistant_chat("Search for Python news")

# Check usage
usage = agent.describe()
print(f"Tokens used: {usage['conversation_tokens']}")

# Get global usage across all sessions
total_usage = ChatLLMFactory.get_total_token_usage()
print(total_usage)
```

## Comparison with Standard Providers

| Feature | Standard (anthropic/bedrock) | Claude Code |
|---------|------------------------------|-------------|
| Basic Chat | ✅ | ✅ |
| Web Search | ❌ | ✅ (Built-in) |
| Code Execution | ❌ | ✅ (Built-in) |
| File Operations | ❌ | ✅ (Built-in) |
| Multi-Step Reasoning | ✅ | ✅✅ (Enhanced) |
| Task Spawning | ❌ | ✅ |
| Autonomous Tool Use | ❌ | ✅ |
| Token Tracking | ✅ | ✅ |
| Multi-turn | ✅ | ✅ |

## Implementation

This integration uses the **Claude Agent SDK** exclusively:

- Requires: `pip install claude-sdk`
- Direct Python integration
- Full control over configuration
- Better performance and reliability

**Note**: CLI-based fallback has been removed. The SDK is now required.

## Advanced Configuration

### Custom Working Directory

```python
config = DictConfig({
    "provider": "claude-code",
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "working_directory": "/path/to/project",  # All file operations relative to this
})
```

### Verbose Logging

```python
config = DictConfig({
    "provider": "claude-code",
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "verbose": True,  # See tool calls and agent reasoning
})
```

### Using Bedrock

```python
config = DictConfig({
    "provider": "claude-code",
    "backend": "bedrock",  # Use AWS Bedrock
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "max_tokens": 4096,
    "aws_region": "us-west-2",
})
```

## Best Practices

1. **Be Specific**: Give clear, detailed instructions for best results
   ```python
   # Good
   agent.assistant_chat("Search for Python 3.12 release notes and list the top 5 new features")

   # Less specific
   agent.assistant_chat("Tell me about Python 3.12")
   ```

2. **Leverage Autonomy**: Let Claude Code decide which tools to use
   ```python
   # Claude Code will figure out it needs web search + file operations
   agent.assistant_chat(
       "Research best practices for async Python and create a reference guide"
   )
   ```

3. **Set Working Directory**: For file-heavy tasks, set the working directory
   ```python
   config.working_directory = "/path/to/project"
   ```

4. **Use Appropriate Models**:
   - `claude-3-5-sonnet-20241022`: Best for complex reasoning and tool use
   - `claude-3-5-haiku-20241022`: Faster, cheaper for simpler tasks
   - `claude-3-opus-20240229`: Most capable for very complex tasks

5. **Monitor Token Usage**: Claude Code can use significant tokens with tool calls
   ```python
   usage = agent.describe()
   print(f"Tokens: {usage['conversation_tokens']}")
   ```

## Troubleshooting

### SDK Not Found

```bash
# Step 1: Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# Step 2: Install Claude Agent SDK
pip install claude-agent-sdk
```

If you get an ImportError, both Claude Code and the Claude Agent SDK must be installed. There is no fallback mode.

### API Key Issues

```bash
# Check environment variables
echo $ANTHROPIC_API_KEY  # For Anthropic backend
echo $AWS_DEFAULT_REGION  # For Bedrock backend
```

### Permission Errors

Claude Code respects `.claude/settings.json` permissions. Check your Claude Code settings:

```bash
claude config show
```

### Agent Initialization Errors

If you encounter errors during agent initialization, ensure:
- Claude Code CLI is installed: `curl -fsSL https://claude.ai/install.sh | bash`
- Claude Agent SDK is installed: `pip install claude-agent-sdk`
- API credentials are set correctly (ANTHROPIC_API_KEY or AWS credentials)
- The model ID is valid for your chosen backend

## Security Considerations

1. **Code Execution**: Claude Code can execute arbitrary code via Bash tool
   - Only use in trusted environments
   - Review Claude Code's permission system
   - Consider restricting working directory

2. **File Access**: Claude Code can read/write files in working directory
   - Set appropriate `working_directory`
   - Monitor file operations in verbose mode

3. **Web Access**: Claude Code can search the web and fetch URLs
   - Results from external sources should be validated
   - Be aware of data privacy implications

4. **API Keys**: Protect your API keys
   - Use environment variables
   - Never commit keys to version control
   - Rotate keys periodically

## Performance Tips

1. **Use Haiku for Simple Tasks**: Much faster and cheaper
   ```python
   config.model = "claude-3-5-haiku-20241022"
   ```

2. **Adjust max_tokens**: Lower for shorter responses
   ```python
   config.max_tokens = 2048  # Faster, cheaper
   ```

3. **Install Required Dependencies**: Both Claude Code and Agent SDK are required
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash  # Claude Code CLI
   pip install claude-agent-sdk                    # Claude Agent SDK
   ```

4. **Batch Requests**: For multiple independent tasks, make separate calls rather than one complex prompt

## Examples Repository

See `/fsx/mlzero-dev/autogluon-assistant/src/autogluon/assistant/llm/example_claude_code_config.yaml` for configuration examples.

## Support and Documentation

- Claude Code Documentation: https://code.anthropic.com/docs
- Claude Agent SDK: https://platform.claude.com/docs/agent-sdk
- Anthropic API Docs: https://docs.anthropic.com
- AWS Bedrock Docs: https://docs.aws.amazon.com/bedrock/

## Migration from Standard Providers

Migrating from `anthropic` or `bedrock` to `claude-code` is simple:

```python
# Before
config = DictConfig({
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
})

# After
config = DictConfig({
    "provider": "claude-code",  # Just change this!
    "backend": "anthropic",     # Add this
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
})
```

All existing code using `assistant_chat()` will work the same, but now with enhanced capabilities!
