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

## Key Files

- `coding_agent.py`: Main entry point for coding agent
- `chatting_agent.py`: Entry point for chat mode
- `managers/node_manager.py`: MCTS implementation
- `constants.py`: Configuration constants and defaults
- `cli/app.py`: CLI interface (typer-based)
