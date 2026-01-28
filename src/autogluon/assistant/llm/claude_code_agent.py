import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseAssistantChat):
    """
    Integration with Claude Agent SDK for agentic capabilities.

    This provides access to Claude Code's full agentic capabilities including:
    - Web search (WebSearch, WebFetch)
    - Code execution (Bash)
    - File operations (Read, Write, Edit, Glob, Grep)
    - Task spawning and parallel execution
    - MCP server integration
    - Custom tools and skills

    Uses the Claude Agent SDK which provides built-in tool execution.
    """

    # Pydantic fields
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    verbose: bool = Field(default=False)
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"]
    )
    working_directory: str = Field(default_factory=os.getcwd)
    use_bedrock: bool = Field(default=False)
    aws_region: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None, exclude=True)
    sdk_available: bool = Field(default=False, exclude=True)
    agent: bool = Field(default=False, exclude=True)
    query: Optional[Any] = Field(default=None, exclude=True)
    ClaudeAgentOptions: Optional[Any] = Field(default=None, exclude=True)

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        session_name: str = "default_session",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        verbose: bool = False,
        allowed_tools: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
        api_key: Optional[str] = None,
        use_bedrock: bool = False,
        aws_region: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Claude Agent SDK integration.

        Args:
            model: Claude model to use (e.g., 'claude-3-5-sonnet-20241022')
            session_name: Name for the session
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            verbose: Enable verbose logging
            allowed_tools: List of tools to allow (default: all standard tools)
            working_directory: Working directory for file operations
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            use_bedrock: Use AWS Bedrock instead of Anthropic API
            aws_region: AWS region for Bedrock (uses AWS_DEFAULT_REGION if not provided)
        """
        # Setup API credentials before calling super().__init__()
        if use_bedrock:
            aws_region = aws_region or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
            if "AWS_DEFAULT_REGION" not in os.environ:
                os.environ["AWS_DEFAULT_REGION"] = aws_region
            os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
            logger.info(f"Using AWS Bedrock with region: {aws_region}")
        else:
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set in environment or provided as parameter")
            os.environ["ANTHROPIC_API_KEY"] = api_key
            logger.info("Using Anthropic API")

        # Default tools if not specified
        if allowed_tools is None:
            allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"]

        # Import Claude Agent SDK (required) - do this BEFORE super().__init__()
        try:
            from claude_agent_sdk import ClaudeAgentOptions as sdk_options
            from claude_agent_sdk import query as sdk_query

            query_func = sdk_query
            options_class = sdk_options
            sdk_available = True
            agent_initialized = True
            logger.info(f"Initialized Claude Agent SDK with model: {model}")
        except ImportError as e:
            error_msg = (
                "Claude Agent SDK is required for claude-code provider.\n"
                "Install with:\n"
                "  1. Install Claude Code CLI: curl -fsSL https://claude.ai/install.sh | bash\n"
                "  2. Install Claude Agent SDK: pip install claude-agent-sdk\n"
                "For more information, see: https://docs.anthropic.com/en/docs/agent-sdk"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        super().__init__(
            session_name=session_name,
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
            allowed_tools=allowed_tools,
            working_directory=working_directory or os.getcwd(),
            use_bedrock=use_bedrock,
            aws_region=aws_region,
            api_key=api_key,
            sdk_available=sdk_available,
            agent=agent_initialized,
            query=query_func,
            ClaudeAgentOptions=options_class,
            **kwargs,
        )

    def _log_internal_process(self, internal_process: Dict) -> None:
        """
        Log Claude Code's internal thinking process.

        Args:
            internal_process: Dictionary containing thinking blocks, tool calls, etc.
        """
        # Count activities
        thinking_count = len(internal_process["thinking"])
        tool_call_count = len(internal_process["tool_calls"])
        tool_result_count = len(internal_process["tool_results"])

        logger.info(
            f"[Claude Code Internal Process] "
            f"Thinking blocks: {thinking_count}, "
            f"Tool calls: {tool_call_count}, "
            f"Tool results: {tool_result_count}"
        )

        # If verbose, log details
        if self.verbose and (thinking_count > 0 or tool_call_count > 0):
            logger.debug("=== Claude Code Internal Process Details ===")

            if thinking_count > 0:
                logger.debug(f"Thinking blocks ({thinking_count}):")
                for i, thinking in enumerate(internal_process["thinking"], 1):
                    logger.debug(f"  [{i}] {thinking[:300]}...")

            if tool_call_count > 0:
                logger.debug(f"Tool calls ({tool_call_count}):")
                for i, tool_call in enumerate(internal_process["tool_calls"], 1):
                    logger.debug(f"  [{i}] {tool_call}")

            if tool_result_count > 0:
                logger.debug(f"Tool results ({tool_result_count}):")
                for i, result in enumerate(internal_process["tool_results"], 1):
                    logger.debug(f"  [{i}] {result[:200]}...")

    async def _send_via_sdk(self, message: str) -> tuple:
        """Send message using Claude Agent SDK.

        Returns:
            Tuple of (response_text, internal_process_dict)
        """
        try:
            # Set model via environment variable (SDK reads from env)
            original_model = os.environ.get("CLAUDE_MODEL")
            os.environ["CLAUDE_MODEL"] = self.model_name

            # Create options for the query (only supported parameters)
            options = self.ClaudeAgentOptions(
                allowed_tools=self.allowed_tools,
                permission_mode="bypassPermissions",  # Auto-approve for programmatic use
            )

            result_text = ""
            input_tokens = 0
            output_tokens = 0

            # Storage for internal process tracking
            internal_process = {
                "thinking": [],
                "tool_calls": [],
                "tool_results": [],
                "text_blocks": [],
                "raw_messages": [],
            }

            try:
                # Run the query and collect results
                async for msg in self.query(prompt=message, options=options):
                    msg_type = type(msg).__name__

                    # Store raw message for debugging
                    if self.verbose:
                        logger.debug(f"SDK message type: {msg_type}")
                        msg_attrs = list(vars(msg).keys()) if hasattr(msg, "__dict__") else []
                        internal_process["raw_messages"].append({"type": msg_type, "attributes": msg_attrs})

                        # Log full message content for debugging
                        logger.debug(f"Message attributes: {msg_attrs}")

                    # Extract content based on message type
                    # AssistantMessage: has 'content' attribute
                    if hasattr(msg, "content"):
                        content = msg.content
                        if content:
                            content_str = str(content)
                            internal_process["text_blocks"].append(content_str)
                            logger.info(f"[Claude Code Message] {content_str[:200]}...")

                            # Try to parse content for thinking or tool use
                            # Content might be structured (list of dicts) or plain text
                            if isinstance(content, (list, tuple)):
                                for block in content:
                                    # Check for thinking blocks
                                    if isinstance(block, dict):
                                        if "thinking" in block or block.get("type") == "thinking":
                                            thinking_text = block.get("thinking") or block.get("text", str(block))
                                            internal_process["thinking"].append(thinking_text)
                                            logger.info(f"[Claude Code Thinking] {thinking_text[:200]}...")
                                        # Check for tool use
                                        elif "tool_use" in block or block.get("type") == "tool_use":
                                            tool_info = {
                                                "name": block.get("name", "unknown"),
                                                "id": block.get("id", ""),
                                                "input": str(block.get("input", ""))[:200],
                                            }
                                            internal_process["tool_calls"].append(tool_info)
                                            logger.info(
                                                f"[Claude Code Tool] {tool_info['name']}: {tool_info['input']}"
                                            )

                    # SystemMessage: has 'data' attribute
                    if hasattr(msg, "data") and msg.data:
                        data_str = str(msg.data)[:500]
                        logger.debug(f"[Claude Code System] {data_str}")

                    # ResultMessage: has 'result' attribute with final output
                    if hasattr(msg, "result"):
                        result_text = msg.result
                        logger.debug(f"[Claude Code Result] Got result ({len(result_text)} chars)")

                    # Legacy: check for old-style attributes (in case SDK changes)
                    if hasattr(msg, "thinking") and msg.thinking:
                        thinking_content = msg.thinking if isinstance(msg.thinking, str) else str(msg.thinking)
                        internal_process["thinking"].append(thinking_content)
                        logger.info(f"[Claude Code Thinking] {thinking_content[:200]}...")

                    if hasattr(msg, "tool_use") or hasattr(msg, "tool_calls"):
                        tool_info = getattr(msg, "tool_use", None) or getattr(msg, "tool_calls", None)
                        if tool_info:
                            internal_process["tool_calls"].append({"tool": str(tool_info)[:200]})
                            logger.info(f"[Claude Code Tool Call] {str(tool_info)[:200]}")

                    if hasattr(msg, "tool_result"):
                        result_info = str(msg.tool_result)[:500]
                        internal_process["tool_results"].append(result_info)
                        logger.info(f"[Claude Code Tool Result] {result_info}")

                    # Extract token usage if available
                    if hasattr(msg, "usage"):
                        usage_obj = msg.usage
                        input_tokens = getattr(usage_obj, "input_tokens", 0)
                        output_tokens = getattr(usage_obj, "output_tokens", 0)
                        if self.verbose:
                            logger.debug(f"[Claude Code Tokens] Input: {input_tokens}, Output: {output_tokens}")

            finally:
                # Restore original model env var
                if original_model is not None:
                    os.environ["CLAUDE_MODEL"] = original_model
                elif "CLAUDE_MODEL" in os.environ:
                    del os.environ["CLAUDE_MODEL"]

            # Log internal process summary
            self._log_internal_process(internal_process)

            # Update token tracking
            if input_tokens > 0 or output_tokens > 0:
                self.input_tokens_ += input_tokens
                self.output_tokens_ += output_tokens
                self.token_tracker.add_tokens(self.conversation_id, self.session_name, input_tokens, output_tokens)

            return (result_text if result_text else "", internal_process)

        except Exception as e:
            logger.error(f"Error sending message via Claude Agent SDK: {e}")
            raise

    def save_internal_process_to_file(self, filepath: str, include_raw_dump: bool = False) -> None:
        """
        Save the most recent internal process to a file.

        Args:
            filepath: Path where to save the internal process log
            include_raw_dump: If True, append full raw message dump for debugging
        """
        if not hasattr(self, "history_") or not self.history_:
            logger.warning("No history available to save internal process")
            return

        last_entry = self.history_[-1]
        if "internal_process" not in last_entry:
            logger.warning("No internal process data in last history entry")
            return

        import json

        internal_process = last_entry["internal_process"]

        # Create a readable format
        output = ["=== Claude Code Internal Process Log ===\n"]

        # Text blocks (full message content)
        if internal_process["text_blocks"]:
            output.append(f"\n## Message Content ({len(internal_process['text_blocks'])} messages)\n")
            for i, text in enumerate(internal_process["text_blocks"], 1):
                output.append(f"\n### Message {i}:\n{text[:1000]}\n")  # First 1000 chars
                if len(text) > 1000:
                    output.append(f"... ({len(text) - 1000} more characters)\n")

        # Thinking blocks
        if internal_process["thinking"]:
            output.append(f"\n## Thinking Blocks ({len(internal_process['thinking'])})\n")
            for i, thinking in enumerate(internal_process["thinking"], 1):
                output.append(f"\n### Thinking Block {i}:\n{thinking}\n")

        # Tool calls
        if internal_process["tool_calls"]:
            output.append(f"\n## Tool Calls ({len(internal_process['tool_calls'])})\n")
            for i, tool_call in enumerate(internal_process["tool_calls"], 1):
                if isinstance(tool_call, dict):
                    output.append(f"\n### Tool Call {i}:\n{json.dumps(tool_call, indent=2)}\n")
                else:
                    output.append(f"\n### Tool Call {i}:\n{tool_call}\n")

        # Tool results
        if internal_process["tool_results"]:
            output.append(f"\n## Tool Results ({len(internal_process['tool_results'])})\n")
            for i, result in enumerate(internal_process["tool_results"], 1):
                output.append(f"\n### Tool Result {i}:\n{result}\n")

        # Raw messages (for debugging)
        if internal_process["raw_messages"]:
            output.append(f"\n## Raw Message Types ({len(internal_process['raw_messages'])})\n")
            output.append("(For debugging SDK message structure)\n")
            for i, msg_info in enumerate(internal_process["raw_messages"], 1):
                output.append(f"{i}. Type: {msg_info['type']}, Attributes: {msg_info['attributes']}\n")

        # If nothing captured, add explanation
        if (
            not internal_process["thinking"]
            and not internal_process["tool_calls"]
            and not internal_process["text_blocks"]
        ):
            output.append(
                "\n## No Internal Process Captured\n\n"
                "This may be normal if:\n"
                "- The task was simple and didn't require explicit reasoning\n"
                "- The Claude Agent SDK doesn't expose thinking blocks in messages\n"
                "- The response was generated directly without intermediate steps\n\n"
                "Check console logs with -v 4 for real-time message stream.\n"
            )

        try:
            with open(filepath, "w") as f:
                f.write("".join(output))
            logger.info(f"Saved Claude Code internal process to {filepath}")

            # Save raw dump if requested
            if include_raw_dump:
                raw_dump_path = filepath.replace(".txt", "_raw_dump.json")
                try:
                    with open(raw_dump_path, "w") as f:
                        json.dump(internal_process, f, indent=2, default=str)
                    logger.info(f"Saved raw dump to {raw_dump_path}")
                except Exception as e2:
                    logger.warning(f"Could not save raw dump: {e2}")

        except Exception as e:
            logger.error(f"Failed to save internal process to {filepath}: {e}")

    def save_debug_dump(self, output_dir: str) -> None:
        """
        Save detailed debug information about SDK messages.

        Args:
            output_dir: Directory to save debug files
        """
        if not hasattr(self, "history_") or not self.history_:
            logger.warning("No history available for debug dump")
            return

        import json
        import os

        os.makedirs(output_dir, exist_ok=True)

        last_entry = self.history_[-1]
        if "internal_process" not in last_entry:
            logger.warning("No internal process data for debug dump")
            return

        internal_process = last_entry["internal_process"]

        # Save raw messages structure
        debug_file = os.path.join(output_dir, "claude_code_debug.json")
        try:
            debug_data = {
                "message_types": internal_process["raw_messages"],
                "text_blocks_count": len(internal_process["text_blocks"]),
                "thinking_blocks_count": len(internal_process["thinking"]),
                "tool_calls_count": len(internal_process["tool_calls"]),
                "tool_results_count": len(internal_process["tool_results"]),
                "full_internal_process": internal_process,
            }

            with open(debug_file, "w") as f:
                json.dump(debug_data, f, indent=2, default=str)

            logger.info(f"Saved debug dump to {debug_file}")

            # Also save a human-readable analysis
            analysis_file = os.path.join(output_dir, "claude_code_analysis.txt")
            with open(analysis_file, "w") as f:
                f.write("=== Claude Code SDK Message Analysis ===\n\n")
                f.write("Message Types Observed:\n")
                for msg_info in internal_process["raw_messages"]:
                    f.write(f"  - {msg_info['type']}\n")
                    f.write(f"    Attributes: {', '.join(msg_info['attributes'])}\n")

                f.write(f"\nText Blocks: {len(internal_process['text_blocks'])}\n")
                f.write(f"Thinking Blocks: {len(internal_process['thinking'])}\n")
                f.write(f"Tool Calls: {len(internal_process['tool_calls'])}\n")
                f.write(f"Tool Results: {len(internal_process['tool_results'])}\n")

                if internal_process["text_blocks"]:
                    f.write("\n--- Sample Text Block ---\n")
                    f.write(str(internal_process["text_blocks"][0])[:500] + "\n")

            logger.info(f"Saved analysis to {analysis_file}")

        except Exception as e:
            logger.error(f"Failed to save debug dump: {e}")

    def assistant_chat(self, message: str) -> str:
        """
        Send a message to Claude Agent SDK and get response.

        Args:
            message: The user message/prompt

        Returns:
            The agent's response text
        """
        if not message:
            raise ValueError("Message cannot be empty")

        if not self.agent:
            raise RuntimeError("Claude Agent SDK not initialized. Please check installation.")

        try:
            # Import asyncio and run the async function
            import asyncio

            # Send message via SDK - returns (response_text, internal_process)
            response_text, internal_process = asyncio.run(self._send_via_sdk(message))

            # Record in history with internal process
            history_entry = {
                "input": message,
                "output": response_text,
                "input_tokens": 0,  # Updated by send methods
                "output_tokens": 0,
                "internal_process": internal_process,  # Store internal process
            }
            self.history_.append(history_entry)

            return response_text

        except Exception as e:
            logger.error(f"Error in assistant_chat: {e}")
            raise

    def describe(self) -> Dict[str, Any]:
        """Get model description and conversation info."""
        base_desc = super().describe()
        return {
            **base_desc,
            "provider": "claude-code",
            "model": self.model_name,
            "backend": "bedrock" if self.use_bedrock else "anthropic",
            "working_directory": self.working_directory,
            "allowed_tools": self.allowed_tools,
        }


def create_claude_code_agent(config, session_name: str) -> ClaudeCodeAgent:
    """
    Create a Claude Agent SDK instance.

    Args:
        config: Configuration object with the following attributes:
            - model: Model ID
            - max_tokens: Maximum tokens
            - temperature: Sampling temperature
            - verbose: Enable verbose logging
            - backend: 'anthropic' or 'bedrock' (optional, default: 'anthropic')
            - allowed_tools: List of tools to allow (optional)
            - working_directory: Working directory (optional)
            - aws_region: AWS region for Bedrock (optional)
        session_name: Session name for tracking

    Returns:
        ClaudeCodeAgent instance
    """
    model = config.model
    backend = getattr(config, "backend", "anthropic")
    use_bedrock = backend == "bedrock"

    logger.info(f"Creating Claude Agent SDK instance with model={model}, backend={backend}, session={session_name}")

    kwargs = {
        "model": model,
        "session_name": session_name,
        "max_tokens": config.max_tokens,
        "use_bedrock": use_bedrock,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    if hasattr(config, "allowed_tools"):
        kwargs["allowed_tools"] = config.allowed_tools

    if hasattr(config, "working_directory"):
        kwargs["working_directory"] = config.working_directory

    if hasattr(config, "aws_region"):
        kwargs["aws_region"] = config.aws_region

    # API key handling
    if not use_bedrock and hasattr(config, "api_key"):
        kwargs["api_key"] = config.api_key

    return ClaudeCodeAgent(**kwargs)


def get_claude_code_models() -> List[str]:
    """
    Get available Claude Agent SDK models.

    Returns:
        List of available model IDs
    """
    # Claude Agent SDK supports all Claude models
    return [
        "claude-sonnet-4-5-20250929",
        # Bedrock model IDs
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ]
