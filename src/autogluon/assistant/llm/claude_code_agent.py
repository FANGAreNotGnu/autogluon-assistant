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
    allowed_tools: List[str] = Field(default_factory=lambda: [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"
    ])
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
        **kwargs
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
            allowed_tools = [
                "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"
            ]

        # Import Claude Agent SDK (required) - do this BEFORE super().__init__()
        try:
            from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions as sdk_options
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
            **kwargs
        )

    async def _send_via_sdk(self, message: str) -> str:
        """Send message using Claude Agent SDK."""
        try:
            # Set model via environment variable (SDK reads from env)
            original_model = os.environ.get("CLAUDE_MODEL")
            os.environ["CLAUDE_MODEL"] = self.model_name

            # Create options for the query (only supported parameters)
            options = self.ClaudeAgentOptions(
                allowed_tools=self.allowed_tools,
                permission_mode="bypassPermissions"  # Auto-approve for programmatic use
            )

            result_text = ""
            input_tokens = 0
            output_tokens = 0

            try:
                # Run the query and collect results
                async for msg in self.query(prompt=message, options=options):
                    if self.verbose:
                        logger.info(f"SDK message type: {type(msg).__name__}")
                        if hasattr(msg, '__dict__'):
                            logger.info(f"SDK message: {msg.__dict__}")

                    # Extract result from final message
                    if hasattr(msg, "result"):
                        result_text = msg.result
                    # Some messages might have 'content' instead
                    elif hasattr(msg, "content") and isinstance(msg.content, str):
                        result_text = msg.content

                    # Extract token usage if available
                    if hasattr(msg, "usage"):
                        input_tokens = getattr(msg.usage, "input_tokens", 0)
                        output_tokens = getattr(msg.usage, "output_tokens", 0)
            finally:
                # Restore original model env var
                if original_model is not None:
                    os.environ["CLAUDE_MODEL"] = original_model
                elif "CLAUDE_MODEL" in os.environ:
                    del os.environ["CLAUDE_MODEL"]

            # Update token tracking
            if input_tokens > 0 or output_tokens > 0:
                self.input_tokens_ += input_tokens
                self.output_tokens_ += output_tokens
                self.token_tracker.add_tokens(
                    self.conversation_id,
                    self.session_name,
                    input_tokens,
                    output_tokens
                )

            return result_text if result_text else ""

        except Exception as e:
            logger.error(f"Error sending message via Claude Agent SDK: {e}")
            raise

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

            # Send message via SDK
            response_text = asyncio.run(self._send_via_sdk(message))

            # Record in history
            self.history_.append({
                "input": message,
                "output": response_text,
                "input_tokens": 0,  # Updated by send methods
                "output_tokens": 0,
            })

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
    use_bedrock = (backend == "bedrock")

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
