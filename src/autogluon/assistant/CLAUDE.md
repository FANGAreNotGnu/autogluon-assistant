# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoGluon Assistant (MLZero) is a multi-agent system that automates end-to-end multimodal machine learning workflows. It uses Monte Carlo Tree Search (MCTS) to explore solution spaces and employs specialized agents for different ML tasks.

**Paper**: [MLZero: A Multi-Agent System for End-to-end Machine Learning Automation](https://arxiv.org/abs/2505.13941) (NeurIPS 2025)

## Build and Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run linting (all three must pass for CI)
black --check src/ tests/
isort --check-only src/ tests/
ruff check src/ tests/

# Auto-format code
black src/ tests/
isort src/ tests/
ruff check --fix src/ tests/

# Run all tests
pytest tests/

# Run specific test file
python3 -m pytest tests/unittests/path_to_file.py

# Run specific test
python3 -m pytest tests/unittests/path_to_file.py::test_name

# MCP tests require single worker
pytest -n 1 -vv -s tests/unittests/mcp/test_mcp_integration.py
```

## CLI Entry Points

| Command | Purpose |
|---------|---------|
| `mlzero -i <data_folder>` | Main coding agent (generates and executes ML code) |
| `mlzero chat` | Conversational Q&A mode (no code execution) |
| `mlzero-frontend` | Streamlit web UI |
| `mlzero-backend` | WebUI backend server |
| `mlzero-mcp-server` | MCP server for IDE integration |

Key CLI options:
- `--provider`: LLM provider (bedrock, openai, anthropic, sagemaker, claude-code)
- `-n, --max-iterations`: Max MCTS iterations (default: 5)
- `-c, --config`: Custom YAML config file
- `-v, --verbosity`: Log verbosity 0-4

## Architecture

### MCTS-Based Solution Search

The system uses Monte Carlo Tree Search orchestrated by `NodeManager` (`managers/node_manager.py`):

```
Input Data → NodeManager (MCTS) → Agent Pipeline → Solution Tree → Best Solution
```

Node stages: `root` (initial attempt) → `debug` (error fixing) → `evolve` (optimization)

### Agent Pipeline (in `agents/`)

Each MCTS node runs this agent sequence:
1. **DataPerceptionAgent** - Analyzes input data characteristics
2. **TaskDescriptorAgent** - Refines task understanding
3. **ToolSelectorAgent** - Selects ML frameworks/tools
4. **RetrieverAgent** - Finds relevant tutorials from registry
5. **RerankerAgent** - Ranks tutorials by relevance
6. **ErrorAnalyzerAgent** - Analyzes previous execution errors
7. **CoderAgent** - Generates Python/Bash code
8. **ExecuterAgent** - Runs code and evaluates results
9. **MetaPromptingAgent** (optional) - Refines prompts

### LLM Integration (`llm/`)

- `BaseChatLLM`: Abstract base class for all LLM providers
- `ChatLLMFactory`: Factory for creating LLM instances
- Providers: Bedrock, OpenAI, Anthropic, SageMaker, ClaudeCode

### Configuration System (`configs/`)

YAML configs use OmegaConf for hierarchical merging. Provider-specific configs (bedrock.yaml, openai.yaml, etc.) extend default.yaml using YAML anchors.

Key MCTS parameters in configs:
- `exploration_constant`: UCT exploration/exploitation trade-off (default: 1.414)
- `max_debug_depth`: Maximum debug node depth (default: 3)
- `initial_root_children`, `max_debug_children`, `max_evolve_children`: Tree expansion limits

### Tools Registry (`tools_registry/`)

Central registry of ML tools (AutoGluon, Qwen, Wav2Vec2, etc.) with tutorials for the RetrieverAgent to use.

## Code Style

- Line length: 119 characters
- Python 3.10-3.12 compatible
- Tools: black, isort (profile: black), ruff
- CI requires all linters to pass

## Key Architectural Patterns

### Manager as State Container

All execution state is held by managers. Agents/prompts are stateless and access state via manager properties:
- **NodeManager** (`managers/node_manager.py`): MCTS-based code generation with tree of Node objects
- **ChattingManager** (`managers/chatting_manager.py`): Session-based Q&A without code execution

Both implement the same interface contract (properties like `user_input`, `task_description`, `python_code`, `tutorial_prompt`, etc.) enabling polymorphic agent usage.

### Prompt System Architecture

The prompts system (`prompts/`) uses a **variable provider pattern**:

1. **BasePrompt**: Abstract base with `build()` → `render()` → `parse()` lifecycle
2. **VariableProvider** (`variable_provider.py`): Resolves `{variable_name}` placeholders to manager state
3. **VariableRegistry** (`variables.py`): Central mapping of canonical names to manager properties with alias support
4. **Template directives**: `{var_name_truncate_mid_2048}` syntax for truncation

Each prompt class implements:
- `default_template()`: Template with `{variable}` placeholders
- `_build()`: Calls `self.render()` to substitute variables
- `parse()`: Extracts structured data from LLM response
- `meta_instructions()`: Guidance for meta-prompting agent (optional)

### Two-Stage Tutorial Retrieval

1. **RetrieverAgent**: Generates search query via LLM → embeds with BGE model → FAISS retrieval
2. **RerankerAgent**: LLM ranks retrieved tutorials by relevance → formats for CoderAgent

**TutorialIndexer** (`tools_registry/indexing.py`): Lazy-loads BGE embeddings, maintains per-tool FAISS indices in `indices/BAAI_bge-base-en-v1.5/`

### Node State Tracking (MCTS)

Each `Node` in the tree tracks:
- **Tree structure**: `parent`, `children`, `depth`, `time_step` (global ID)
- **Stage**: `root` (initial) → `debug` (error fixing) → `evolve` (optimization)
- **MCTS stats**: `visits`, `validated_visits`, `failure_visits`, `validated_reward`
- **Artifacts**: `python_code`, `bash_script`, `tool_used`, `tutorial_prompt`
- **Results**: `stdout`, `stderr`, `validation_score`, `error_analysis`, `is_successful`

MCTS lifecycle: `select_node()` → `expand()` → `simulate()` → `backpropagate()`

Terminal nodes stop expansion after max debug depth or when all children terminal.

### WebUI Backend Architecture

Queue-based design with SQLite persistence:

- **QueueManager** (`webui/backend/queue/manager.py`): Singleton with background executor thread
- **TaskDatabase**: SQLite at `~/.autogluon_assistant/webui_queue.db`, task lifecycle: `queued` → `running` → `completed`
- **Routes** (`webui/backend/routes.py`):
  - POST `/api/run`: Submit task
  - GET `/api/logs`: Stream subprocess output
  - GET `/api/status`: Check task state
  - POST `/api/cancel`: Cancel task

Frontend polls `/api/logs` and `/api/status` while backend executor runs mlzero subprocess.

### MCP Integration

**MCP Server** (`mcp/server/server.py`): FastMCP-based tool interface exposing `start_task()`, `check_status()`, `cancel_task()`, `list_outputs()`

**Task Manager** (`mcp/server/task_manager.py`): Async wrapper around Flask API using aiohttp

Integration: IDE/Client ← MCP Server → Task Manager → Flask `/api/run` → QueueManager → mlzero subprocess

### Error Recovery Pattern

Failed execution → ErrorAnalyzerAgent analyzes → stores in `node.error_analysis` → creates debug node child with same tool → successful debug replaces failed parent via tree restructuring

All error analyses stored in manager's `_all_error_analyses` for learning across iterations.

### LLM Integration

- **ChatLLMFactory** (`llm/`): Factory pattern creates provider-specific instances
- **GlobalTokenTracker**: Singleton tracks token usage across all sessions
- Providers: Bedrock, OpenAI, Anthropic, SageMaker, ClaudeCode
- Each inherits from `BaseChatLLM` with `assistant_chat()` interface

## Key Files

- `coding_agent.py`: Entry point - creates NodeManager, runs MCTS loop
- `chatting_agent.py`: Entry point - creates ChattingManager, runs chat loop
- `managers/node_manager.py`: MCTS orchestration, tree management, agent pipeline
- `managers/chatting_manager.py`: Session-based conversation, no code execution
- `prompts/base_prompt.py`: Base prompt abstraction with variable provider
- `prompts/variable_provider.py`: Template variable resolution system
- `agents/base_agent.py`: Base agent abstraction, LLM interaction pattern
- `llm/chat_llm_factory.py`: LLM provider abstraction and token tracking
- `tools_registry/indexing.py`: Tutorial embedding and FAISS retrieval
- `constants.py`: Configuration constants and defaults
- `cli/app.py`: CLI interface (typer-based)

## Important Implementation Notes

### Adding New Managers
Implement these properties for agent/prompt compatibility:
- State: `user_input`, `task_description`, `data_prompt`, `python_code`, `bash_script`, `error_message`, `tutorial_prompt`, `selected_tool`, `tool_prompt`
- Folders: `per_iteration_output_folder`, `iteration_folder`
- Methods: `save_and_log_states()`, `log_agent_start()`, `log_agent_end()`

### Adding New Prompts
1. Extend `BasePrompt`
2. Override `default_template()` with `{variable}` placeholders
3. Override `_build()` to call `self.render()`
4. Override `parse()` to extract structure from LLM output
5. Optionally override `meta_instructions()` classmethod for meta-prompting

### Adding New Agents
1. Extend `BaseAgent`
2. Initialize prompt in `__init__`
3. Implement `__call__()`: build prompt → LLM chat → parse response
4. Use `init_llm()` utility from `agents/utils.py`

### Thread Safety
- `Node._lock`: Protects MCTS statistics updates
- `NodeManager._node_lock`: Protects tree modifications
- QueueManager uses daemon thread for background execution

### State Persistence
Each iteration saves to `output_folder/node_X/states/`:
- Prompts: `*_prompt.txt`
- Responses: `*_response.txt`
- Code: `generated_code.py`, `execution_script.sh`
- Results: `stdout`, `stderr`, `tutorial_retrievals.txt`

Best solution linked at `output_folder/best_run/` (symlink)
