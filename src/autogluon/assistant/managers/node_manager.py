"""
Node-based manager using pure Monte Carlo Tree Search. It implements a tree-based
search strategy that allows for more flexible exploration and exploitation of solution
space. It also ensures all available tools are tried during the exploration process.
"""

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Set

from ..llm import ChatLLMFactory
from ..tools_registry import registry

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """
    A node in the solution tree representing a single iteration.
    Stores code, execution results, and evaluation information.
    """

    # Node creation time
    ctime: float = field(default_factory=lambda: time.time())

    # Tree structure
    parent: Optional["Node"] = None
    children: Set["Node"] = field(default_factory=set)

    # Node position in tree
    time_step: int = None  # Corresponds to the global time step when created
    depth: int = 0  # Depth in the tree (root=0, increases with each level)

    # Solution stage
    stage: Literal["root", "debug", "evolve"] = "root"

    # MCTS statistics
    visits: int = 0
    validated_visits: int = 0  # Number of successful runs with validation scores
    failure_visits: int = 0  # Number of failed runs
    unvalidated_visits: int = 0  # Number of successful runs without validation scores
    validated_reward: float = 0.0  # Total reward from validated runs
    # total_reward: float = 0.0  # Replaced by separate reward tracking

    # Node state tracking
    is_successful: bool = False  # Did the execution succeed?
    is_debug_successful: bool = False  # Did the debug in the subtree succeed?
    is_terminal: bool = False  # Should this node not be expanded further?
    debug_attempts: int = 0  # Number of debug attempts on this node

    # Solution artifacts
    python_code: str = ""
    bash_script: str = ""
    tool_used: str = ""  # The primary tool used for this solution
    tools_available: List[str] = field(
        default_factory=list
    )  # All tools available for this solution, in priority order
    tutorial_retrieval: str = ""  # Retrieved tutorials for this node
    tutorial_prompt: str = ""  # Processed tutorial prompt for this node

    # Execution results
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    error_analysis: str = ""

    # Evaluation metrics
    validation_score: Optional[float] = None

    # Instructions (for checkpoint-resume orchestration)
    local_instructions: List[str] = field(default_factory=list)

    # Locking for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    expected_child_count: int = 0

    @property
    def id(
        self,
    ):
        return self.time_step

    def __post_init__(self):
        """
        Initialize a node, adding it to parent's children if parent exists.
        Set depth based on parent's depth.
        """
        if self.parent is not None:
            self.parent.add_child(self)
            self.depth = self.parent.depth + 1

    def add_child(self, child: "Node") -> None:
        """
        Add a child node to this node.
        """
        logger.detail(f"Node {child.id} is added to children of Node {self.id}.")
        self.children.add(child)

    def remove_child(self, child: "Node") -> None:
        """
        Remove a child node of this node.
        """
        logger.detail(f"Node {child.id} is removed from children of Node {self.id}.")
        self.children.remove(child)

    @property
    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node (has no children).
        """
        return len(self.children) == 0

    @property
    def num_children(self) -> int:
        """
        Get the number of child nodes.
        """
        return len(self.children)

    @property
    def prev_tutorial_prompt(self) -> str:
        if self.parent and self.parent.tutorial_prompt:
            return self.parent.tutorial_prompt

    def update(self, reward: float, is_validated: bool = False, is_failure: bool = False) -> None:
        """
        Update the node's statistics with a new reward.

        Args:
            reward: The raw validation score (for validated runs) or None
            is_validated: Whether this reward comes from a validated run
            is_failure: Whether this run was a failure
        """
        with self._lock:
            self.visits += 1

            if is_failure:
                self.failure_visits += 1
            elif is_validated and reward is not None:
                # For validated runs, store the raw validation score
                self.validated_visits += 1
                self.validated_reward += reward  # Sum up the raw scores, will be normalized in UCT
            else:
                # For successful runs without validation
                self.unvalidated_visits += 1

    def uct_value(
        self,
        exploration_constant: float = 1.414,
        failure_offset: float = 0,
        failure_penalty_weight: float = 0.5,
        score_min: float = 0.0,
        score_max: float = 0.0,
        score_temperature: float = 0.3,
    ) -> float:
        """
        Calculate the UCT (Upper Confidence Bound for Trees) value of the node.

        Uses min-max exponential normalization for the exploitation term:
        normalized = (avg_score - score_min) / (score_max - score_min)   # [0, 1]
        shaped     = (exp(normalized / T) - 1) / (exp(1/T) - 1)         # [0, 1]

        This is scale-invariant: the same temperature works regardless of absolute score range.

        Args:
            exploration_constant: The constant that controls exploration vs exploitation
            failure_offset: Number of failures to forgive before penalizing
            failure_penalty_weight: Weight of the failure penalty
            score_min: Minimum avg score across all scored nodes
            score_max: Maximum avg score across all scored nodes
            score_temperature: Temperature for shaping (lower = sharper, favors high scores)

        Returns:
            The UCT value
        """
        # For unvisited nodes, return infinity to ensure they are visited
        if self.visits == 0:
            return float("inf")

        # Get parent visits for UCT calculation
        if self.parent:
            parent_visits = max(1, self.parent.visits)
        else:
            parent_visits = 1

        # Calculate exploitation term based on node stats
        self.normalized_failure_visit = max(0, self.failure_visits - failure_offset)
        self.failure_penalty = -failure_penalty_weight * self.normalized_failure_visit / self.visits

        # Calculate the validated rewards part using min-max exponential normalization
        if self.validated_visits > 0 and score_max > score_min:
            self.avg_raw_score = self.validated_reward / self.validated_visits
            # Min-max normalize to [0, 1], then apply exponential shaping
            normalized = (self.avg_raw_score - score_min) / (score_max - score_min)
            normalized = max(0.0, min(1.0, normalized))  # clamp for safety
            T = score_temperature
            exp_inv_T = math.exp(1.0 / T)
            self.softmax_score = (math.exp(normalized / T) - 1.0) / (exp_inv_T - 1.0)
            self.validated_weight = self.validated_visits / self.visits
            self.validated_contribution = self.validated_weight * self.softmax_score
        elif self.validated_visits > 0:
            # All scores equal or only one scored node — treat as 1.0
            self.avg_raw_score = self.validated_reward / self.validated_visits
            self.softmax_score = 1.0
            self.validated_contribution = self.validated_visits / self.visits
        else:
            self.validated_contribution = 0.0

        # Total exploitation is the weighted sum of all components
        self.exploitation = self.validated_contribution + self.failure_penalty

        # Calculate exploration term
        self.exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

        return self.exploitation + self.exploration

    def to_dict(self) -> dict:
        """Serialize this node and its entire subtree to a dict."""
        return {
            "time_step": self.time_step,
            "depth": self.depth,
            "stage": self.stage,
            "visits": self.visits,
            "validated_visits": self.validated_visits,
            "failure_visits": self.failure_visits,
            "unvalidated_visits": self.unvalidated_visits,
            "validated_reward": self.validated_reward,
            "is_successful": self.is_successful,
            "is_debug_successful": self.is_debug_successful,
            "is_terminal": self.is_terminal,
            "debug_attempts": self.debug_attempts,
            "python_code": self.python_code,
            "bash_script": self.bash_script,
            "tool_used": self.tool_used,
            "tools_available": self.tools_available,
            "tutorial_retrieval": self.tutorial_retrieval,
            "tutorial_prompt": self.tutorial_prompt,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "error_analysis": self.error_analysis,
            "validation_score": self.validation_score,
            "local_instructions": self.local_instructions,
            "expected_child_count": self.expected_child_count,
            "children": [child.to_dict() for child in sorted(self.children, key=lambda c: c.time_step)],
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional["Node"] = None) -> "Node":
        """Reconstruct a node and its subtree from a dict.

        Args:
            data: Serialized node dict (from to_dict)
            parent: Parent node (None for root)

        Returns:
            Reconstructed Node with children
        """
        children_data = data.pop("children", [])
        # Create node without triggering __post_init__ parent linkage yet
        node = cls(parent=parent, **data)
        # Recursively restore children (they link to this node via __post_init__)
        for child_data in children_data:
            cls.from_dict(child_data, parent=node)
        return node

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class NodeManager:
    """
    Manages a tree of nodes representing different iterations of solution development.
    Uses Monte Carlo Tree Search (MCTS) to explore the solution space more effectively.
    """

    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: Any,
        initial_user_input: str,
        enable_per_iteration_instruction: bool,
    ):
        """
        Initialize the NodeManager with required paths and configuration.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config: Configuration object
            initial_user_input: Initial user instruction
            enable_per_iteration_instruction: If asking for per iteration user input
        """
        # Store required paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config
        self.enable_per_iteration_instruction = enable_per_iteration_instruction
        self.initial_user_input = initial_user_input

        # Track time_step
        self.time_step = -1
        # Create root node
        self.root_node = Node(stage="root", time_step=self.time_step, depth=0)
        self.current_node = self.root_node

        # Track best nodes and metrics
        self._best_node = None
        self._best_validation_score = None
        self._worst_validation_score = None
        self.last_successful_node = None

        # Key node tracking
        self.best_step = -1
        self.last_successful_step = -1

        # MCTS parameters
        self.exploration_constant = self.config.exploration_constant
        self.max_debug_depth = self.config.max_debug_depth
        self.failure_offset = self.config.failure_offset
        self.failure_penalty_weight = self.config.failure_penalty_weight
        self.score_temperature = getattr(self.config, "score_temperature", 0.3)

        # Score range cache for UCT computation
        self._cached_score_range = (0.0, 0.0)
        self._cached_score_range_step = -1

        # Tracking for thread safety
        self._node_lock = threading.Lock()
        self.search_start_time = time.time()

        # User inputs storage
        self.user_inputs = []

        # Error analysis storage
        self._all_error_analyses = []

        # Tool tracking
        self.used_tools = set()

        # Checkpoint-resume: instructions
        self.global_instructions: List[str] = []
        self.pending_local_instruction: Optional[str] = None

        # Target prompt instance for meta-prompting
        self.target_prompt_instance = None

        # Initialize the agent components
        self._init_agents()

    def _init_agents(self):
        """Initialize all required agents."""
        from ..agents import (
            CoderAgent,
            DataPerceptionAgent,
            DescriptionFileRetrieverAgent,
            ErrorAnalyzerAgent,
            ExecuterAgent,
            MetaPromptingAgent,
            RerankerAgent,
            RetrieverAgent,
            TaskDescriptorAgent,
            ToolSelectorAgent,
        )

        # Data perception agent
        self.dp_agent = DataPerceptionAgent(
            config=self.config,
            manager=self,
            input_data_folder=self.input_data_folder,
            reader_llm_config=self.config.reader,
            reader_prompt_template=None,
        )

        # Description file retriever agent
        self.dfr_agent = DescriptionFileRetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.description_file_retriever,
            prompt_template=None,
        )

        # Task descriptor agent
        self.td_agent = TaskDescriptorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.task_descriptor,
            prompt_template=None,
        )

        # Tool selector agent
        self.ts_agent = ToolSelectorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.tool_selector,
            prompt_template=None,
        )

        # Initialize meta-prompting
        self.enable_meta_prompting = self.config.enable_meta_prompting
        self.meta_prompting_agent = MetaPromptingAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.meta_prompting,
        )

        # Error analyzer
        self.error_analyzer = ErrorAnalyzerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.error_analyzer,
            prompt_template=None,
        )

        # Retriever
        self.retriever = RetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.retriever,
            prompt_template=None,
        )

        # Reranker
        self.reranker = RerankerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.reranker,
            prompt_template=None,
        )

        # Python coder
        self.python_coder = CoderAgent(
            config=self.config,
            manager=self,
            language="python",
            coding_mode="coder",
            llm_config=self.config.python_coder,
            prompt_template=None,
        )

        # Bash coder
        self.bash_coder = CoderAgent(
            config=self.config,
            manager=self,
            language="bash",
            coding_mode="coder",
            llm_config=self.config.bash_coder,
            prompt_template=None,
        )

        # Executer
        self.executer = ExecuterAgent(
            config=self.config,
            manager=self,
            language="bash",
            timeout=self.config.per_execution_timeout,
            executer_llm_config=self.config.executer,
            executer_prompt_template=None,
        )

    def initialize(self):
        """Initialize the manager."""
        self.data_prompt = self.dp_agent()
        self.description_files = self.dfr_agent()
        self.task_description = self.td_agent()

        # Use tool selector to get prioritized list of tools
        self.available_tools = self.ts_agent()

    def get_iteration_folder(self, node: Node) -> str:
        """
        Get the folder for storing iteration artifacts.

        Args:
            node: The node to get the folder for

        Returns:
            Path to the iteration folder
        """
        if node.id < 0:
            iter_folder = os.path.join(self.output_folder, "node_init")
        else:
            iter_folder = os.path.join(self.output_folder, f"node_{node.id}")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    def get_per_iteration_output_folder(self, node: Node) -> str:
        """
        Get the folder for storing iteration output artifacts.

        Args:
            node: The node to get the output folder for

        Returns:
            Path to the iteration output folder
        """
        iter_output_folder = os.path.join(self.get_iteration_folder(node), "output")
        os.makedirs(iter_output_folder, exist_ok=True)
        return iter_output_folder

    def save_and_log_states(self, content, save_name, per_iteration=False, add_uuid=False, node=None):
        """
        Save states to a file and log them.

        Args:
            content: Content to save
            save_name: Name for the saved file
            per_iteration: Whether this is for a specific iteration (backward compatibility)
            add_uuid: Whether to add a UUID to the filename
            node: Node to associate with the saved content (required if per_iteration is False)
        """
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        # Determine the save directory
        if per_iteration and self.current_node:
            states_dir = os.path.join(self.get_iteration_folder(self.current_node), "states")
        elif node:
            states_dir = os.path.join(self.get_iteration_folder(node), "states")
        else:
            states_dir = os.path.join(self.output_folder, "states")

        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        """Log agent start message."""
        logger.info(message)

    def log_agent_end(self, message: str):
        """Log agent end message."""
        logger.info(message)

    def select_node(self) -> Node:
        """
        Select a node for expansion using UCT with prior-calibrated new child.

        At each node, a potential "new child" competes with existing children.
        The new child uses the parent's own validation score as its expected
        quality (Q_prior) and fair-share visits to normalize exploration:

          UCT(new)   = Q_prior + C * sqrt(ln(N_parent) / N_fair)
          UCT(child) = Q(child) + C * sqrt(ln(N_parent) / N_child)

        This is score-dependent: expansion happens when children underperform
        the parent; deepening happens when children improve over the parent.

        Returns:
            The selected node for expansion
        """
        # Ensure score range is cached for this step
        if self._cached_score_range_step != self.time_step:
            self._cached_score_range = self._compute_score_range()
            self._cached_score_range_step = self.time_step

        node = self.root_node

        while node is not None:
            # Leaf nodes: always expand
            if node.is_leaf:
                return node

            # Check if we can still create new children at this node
            can_expand = self._can_add_child(node)

            non_terminal_children = [child for child in node.children if not child.is_terminal]

            if not non_terminal_children and not can_expand:
                # All children terminal and no room for new ones
                if node.is_terminal:
                    return None
                # Mark terminal and return None
                self.mark_node_terminal(node)
                return None

            if not non_terminal_children:
                # All existing children are terminal, but we can still expand
                return node

            if can_expand:
                # Compute new child's UCT using parent's own score as prior
                parent_visits = max(1, node.visits)
                num_children = len(non_terminal_children)
                n_fair = max(1, parent_visits / (num_children + 1))

                # Parent's own validation score as Q_prior for new child
                score_min, score_max = self._cached_score_range
                q_prior = 0.0
                if node.validation_score is not None and score_max > score_min:
                    norm = (node.validation_score - score_min) / (score_max - score_min)
                    norm = max(0.0, min(1.0, norm))
                    T = self.score_temperature
                    exp_inv_T = math.exp(1.0 / T)
                    q_prior = (math.exp(norm / T) - 1.0) / (exp_inv_T - 1.0)

                new_child_uct = q_prior + self.exploration_constant * math.sqrt(
                    math.log(parent_visits) / n_fair
                )

                # Compute best existing child UCT (with tool-specific exploration at root)
                best_child = max(
                    non_terminal_children, key=lambda c: self._compute_child_uct(node, c)
                )
                best_child_uct = self._compute_child_uct(node, best_child)

                logger.detail(
                    f"Node {node.id}: new_child UCT={new_child_uct:.3f} (Q_prior={q_prior:.3f}) vs "
                    f"best_child (Node {best_child.id}) UCT={best_child_uct:.3f}"
                )

                if new_child_uct > best_child_uct:
                    # New child wins -> expand at this node
                    return node
                else:
                    # Existing child wins -> descend
                    node = best_child
            else:
                # Can't expand, must descend into best existing child
                best_child = max(
                    non_terminal_children, key=lambda c: self._compute_child_uct(node, c)
                )
                node = best_child

        return None

    def _can_add_child(self, node: Node) -> bool:
        """
        Check if a node can accept a new child (not at hard limit).

        Args:
            node: The node to check

        Returns:
            True if a new child can be added, False otherwise
        """
        if node.is_terminal:
            return False

        # Root node
        if node.stage == "root":
            if self._get_unused_tool() is None:
                return False
            return node.num_children < self.config.initial_root_children

        # For debug nodes, stop expanding after getting a successful node
        if node.stage == "debug":
            if node.is_debug_successful:
                return False
            return node.num_children < self.config.max_debug_children

        # For evolve nodes
        if node.stage == "evolve":
            return node.num_children < self.config.max_evolve_children

        return False

    def _compute_child_uct(self, parent: Node, child: Node) -> float:
        """
        Compute UCT value for a child node, applying tool-specific exploration
        constants when the parent is the root node.

        Args:
            parent: The parent node
            child: The child node to compute UCT for

        Returns:
            The UCT value
        """
        score_min, score_max = self._cached_score_range

        if parent == self.root_node:
            # Tools earlier in the list get higher exploration constants
            tool_index = self.available_tools.index(child.tool_used)
            tool_specific_exploration = self.exploration_constant * max(0.25, 1.0 - 0.25 * tool_index)
            return child.uct_value(
                tool_specific_exploration,
                failure_offset=self.failure_offset,
                failure_penalty_weight=self.failure_penalty_weight,
                score_min=score_min,
                score_max=score_max,
                score_temperature=self.score_temperature,
            )
        else:
            return child.uct_value(
                self.exploration_constant,
                failure_offset=self.failure_offset,
                failure_penalty_weight=self.failure_penalty_weight,
                score_min=score_min,
                score_max=score_max,
                score_temperature=self.score_temperature,
            )

    def expand(self) -> Node:
        """
        Expand the current node by creating a child node.

        Returns:
            The newly created child node
        """
        if self.current_node.stage == "root":
            return self._create_evolve_node()
        elif self.current_node.is_successful:
            return self._create_evolve_node()
        else:
            return self._create_debug_node()

    def _get_unused_tool(self) -> Optional[str]:
        """
        Get a tool that has not been used yet in the tree.

        Returns:
            An unused tool, or None if all tools have been used
        """
        unused_tools = [tool for tool in self.available_tools if tool not in self.used_tools]
        if unused_tools:
            # return random.choice(unused_tools)
            return unused_tools[0]  # TODO: enable random selection of available tools
        return None

    def _create_debug_node(
        self,
    ) -> Node:
        """
        Create a debug node to fix issues in a failed node.

        Returns:
            The newly created debug node
        """
        # Increment global time step for this new node
        self.time_step += 1

        # Inherit local instructions from parent + apply any pending instruction
        inherited_instructions = list(self.current_node.local_instructions)
        if self.pending_local_instruction:
            inherited_instructions.append(self.pending_local_instruction)
            self.pending_local_instruction = None

        # Create a new node
        self.current_node = Node(
            parent=self.current_node,
            stage="debug",
            # Use the same tool as the parent for debugging
            tool_used=self.current_node.tool_used,
            tools_available=self.available_tools,
            time_step=self.time_step,
            debug_attempts=self.current_node.debug_attempts + 1,
            local_instructions=inherited_instructions,
        )

        # Check if we've exceeded the maximum debug attempts for this node
        if self.current_node.debug_attempts >= self.max_debug_depth:
            logger.warning(
                f"Node {self.current_node.id} has reached the maximum debug depth ({self.max_debug_depth}). Marking as terminal."
            )
            self.mark_node_terminal(self.current_node)

        # Generate code for the node
        self._generate_code()

    def _create_evolve_node(
        self,
    ) -> Node:
        """
        Create an evolve node to improve a successful node.

        Returns:
            The newly created evolve node
        """
        # Increment global time step for this new node
        self.time_step += 1

        # Check if there's an unused tool to try
        unused_tool = self._get_unused_tool()
        if unused_tool:
            # If there's an unused tool, create a node from the root with that tool
            logger.info(f"Found unused tool {unused_tool}, creating evolve node from root using this tool")
            parent = self.root_node
            tool_used = unused_tool
        else:
            # Otherwise evolve from the parent node
            logger.info(f"Creating evolve node from Node {self.current_node.id} using {self.current_node.tool_used}.")
            parent = self.current_node
            tool_used = self.current_node.tool_used

        # Inherit local instructions from parent + apply any pending instruction
        inherited_instructions = list(parent.local_instructions) if parent != self.root_node else []
        if self.pending_local_instruction:
            inherited_instructions.append(self.pending_local_instruction)
            self.pending_local_instruction = None

        self.current_node = Node(
            parent=parent,
            stage="evolve",
            tool_used=tool_used,
            tools_available=self.available_tools,
            time_step=self.time_step,
            local_instructions=inherited_instructions,
        )

        # Generate code for the node
        self._generate_code()

    def _update_tutorials(self=None):
        """
        Retrieve and update tutorials for the current selected tool.

        Args:
            node: Node to associate the tutorials with (optional)
        """
        # Retrieve tutorials
        self.current_node.tutorial_retrieval = self.retriever()

        # Rerank the retrieved tutorials
        self.current_node.tutorial_prompt = self.reranker()

        # Save to node's folder
        self.save_and_log_states(
            content=self.current_node.tutorial_retrieval,
            save_name="tutorial_retrievals.txt",
            node=self.current_node,
            add_uuid=False,
        )
        self.save_and_log_states(
            content=self.current_node.tutorial_prompt,
            save_name="tutorial_prompt.txt",
            node=self.current_node,
            add_uuid=False,
        )

    def _generate_code(self):
        """
        Generate Python and Bash code for the current node after the tool to use is specified.
        """
        logger.debug(f"Starting code generation for Node {self.current_node.id}")

        # Mark this tool as used
        self.used_tools.add(self.current_node.tool_used)
        logger.debug(f"  Tool being used: {self.current_node.tool_used}")

        # Always get user input for this step (handles both initial and per-iteration instructions)
        logger.debug(f"  Getting user input for step {self.time_step}")
        self._get_user_input_for_step()

        # Get the tool-specific prompt for the node's selected tool
        from ..tools_registry import registry

        logger.debug("  Retrieving tool info from registry")
        tool_info = registry.get_tool(self.current_node.tool_used)
        if not tool_info:
            print(self.current_node.state)
            raise ValueError(f"Tool {self.current_node.tool_used} not found in registry")

        # Get tool-specific prompt
        self.tool_prompt = tool_info.get("prompt_template", "")
        if isinstance(self.tool_prompt, list):
            self.tool_prompt = "\n".join(self.tool_prompt)

        # Get tutorials specific to this node
        logger.debug("  Starting tutorial retrieval and reranking (this may take time)...")
        self._update_tutorials()
        logger.debug("  Finished tutorial retrieval and reranking")

        # Generate Python code
        logger.debug("  Calling Python coder agent...")
        self.current_node.python_code = self.python_coder()
        logger.debug("  Finished Python code generation")

        # Write the Python code to a file
        python_file_path = os.path.join(self.get_iteration_folder(self.current_node), "generated_code.py")
        logger.debug(f"  Writing Python code to: {python_file_path}")
        with open(python_file_path, "w") as file:
            file.write(self.current_node.python_code)

        # Generate Bash script
        logger.debug("  Calling Bash coder agent...")
        self.current_node.bash_script = self.bash_coder()
        logger.debug("  Finished Bash script generation")

        # Write the Bash script to a file
        bash_file_path = os.path.join(self.get_iteration_folder(self.current_node), "execution_script.sh")
        logger.debug(f"  Writing Bash script to: {bash_file_path}")
        with open(bash_file_path, "w") as file:
            file.write(self.current_node.bash_script)

        logger.debug(f"Completed code generation for Node {self.current_node.id}")

    def _get_user_input_for_step(self):
        """Get user input for the current step.

        Combines: initial_user_input + per-iteration input + global_instructions + local_instructions.
        """
        if self.time_step == 0:
            user_input = self.initial_user_input or ""
        else:
            if self.enable_per_iteration_instruction:
                logger.info(f"Previous iteration info is stored in: {self.get_iteration_folder(self.current_node)}")
                user_input = self.initial_user_input or ""
                user_input += "\n" + input(
                    f"Enter your inputs for current node (step {self.time_step}) (press Enter to skip): "
                )
            else:
                user_input = self.initial_user_input or ""

        # Append global instructions
        if self.global_instructions:
            user_input += "\n\n### Global Instructions\n" + "\n".join(self.global_instructions)

        # Append local instructions from current node's ancestor chain
        local = self._collect_local_instructions()
        if local:
            user_input += "\n\n### Local Instructions (for this subtree)\n" + "\n".join(local)

        self.user_inputs.append(user_input)

    def _collect_local_instructions(self) -> List[str]:
        """Collect local instructions from the current node and its ancestors."""
        instructions = []
        seen = set()
        node = self.current_node
        while node is not None:
            for instr in node.local_instructions:
                if instr not in seen:
                    instructions.append(instr)
                    seen.add(instr)
            node = node.parent
        return instructions

    def simulate(self) -> tuple:
        """
        Simulate execution of current node and evaluate the result.

        Returns:
            Tuple containing: (validation_score, is_validated, is_failure)
                validation_score: The raw validation score (or None if not available)
                is_validated: True if this run has a validation score
                is_failure: True if this run failed
        """
        # Execute the code
        planner_decision, error_summary, validation_score, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.current_node.bash_script,
            code_to_analyze=self.current_node.python_code,
            execution_task=self.task_description,
            execution_data=self.data_prompt,
        )

        # Store execution results
        self.current_node.stdout = stdout
        self.current_node.stderr = stderr

        # Save execution outputs
        self.save_and_log_states(stderr, "stderr", node=self.current_node, add_uuid=False)
        self.save_and_log_states(stdout, "stdout", node=self.current_node, add_uuid=False)

        # Update validation score
        self.current_node.validation_score = validation_score

        # Track the best and worst validation scores for scaling in UCT calculation
        if validation_score is not None:
            # Update best validation score
            if self._best_node is None or validation_score > self._best_validation_score:
                self._best_node = self.current_node
                self._best_validation_score = validation_score
                self.best_step = self.time_step

            # Track worst validation score (initialize if not set yet)
            if not hasattr(self, "_worst_validation_score") or self._worst_validation_score is None:
                self._worst_validation_score = validation_score
            else:
                self._worst_validation_score = min(self._worst_validation_score, validation_score)

        # Determine if the execution was successful
        if planner_decision == "SUCCESS":
            self.current_node.is_successful = True
            self.last_successful_node = self.current_node
            self.last_successful_step = self.time_step
            self.current_node.error_message = ""

            # If this is a debug node, find the origin of the debug chain
            if self.current_node.stage == "debug":
                # Find the original node that started this debugging chain
                debug_origin = self._find_debug_origin(self.current_node)

                # Add this successful node as a sibling to the original buggy node
                self.current_node.parent.remove_child(self.current_node)
                self.current_node.parent = debug_origin.parent
                debug_origin.parent.add_child(self.current_node)

                self.mark_node_terminal(debug_origin)

                logger.info(
                    f"Replaced debug origin node {debug_origin.id} with successful debug node {self.current_node.id}"
                )

            # Return the raw validation score (for tracking), is_validated flag, and is_failure flag
            return (validation_score, validation_score is not None, False)
        else:
            self.current_node.is_successful = False
            self.current_node.error_message = f"stderr: {stderr}\n\n" if stderr else ""
            self.current_node.error_message += f"Error summary: {error_summary}"

            # Get error analysis
            self.current_node.error_analysis = self.error_analyzer()

            self._all_error_analyses.append(self.current_node.error_analysis)

            # If this is a debug node and it failed, check parent's debug attempts
            if self.current_node.stage == "debug" and self.current_node.parent:
                self.current_node.parent.debug_attempts += 1
                logger.warning(
                    f"Debug attempt failed. Debug attempts on parent node {self.current_node.parent.id}: {self.current_node.parent.debug_attempts}/{self.max_debug_depth}"
                )

                # If parent has reached max debug attempts, mark it as terminal
                if self.current_node.parent.debug_attempts >= self.max_debug_depth:
                    logger.warning(
                        f"Parent node {self.current_node.parent.id} has reached the maximum debug depth. Marking as terminal."
                    )
                    self.mark_node_terminal(self.current_node.parent)

            # For failures, we return None score, not validated, and is_failure=True
            return (None, False, True)

    def backpropagate(self, simulation_result):
        """
        Backpropagate the reward up the tree and update terminal status.

        Args:
            simulation_result: Tuple of (validation_score, is_validated, is_failure)
        """
        # Extract simulation results
        validation_score, is_validated, is_failure = simulation_result

        node = self.current_node
        while node is not None:
            node.update(validation_score, is_validated, is_failure)
            node = node.parent

    def step(self):
        """
        Perform one step of the Monte Carlo Tree Search.

        Returns:
            True if a successful node was found, False otherwise
        """
        # Selection: select a node to expand
        self.current_node = self.select_node()
        if self.current_node is None:
            return None

        # Expansion: create a new child node
        # Note: time_step is now incremented in the creation methods
        self.expand()

        # Simulation: execute the code and get results
        simulation_result = self.simulate()

        # Backpropagation: update node statistics
        self.backpropagate(simulation_result)

        # Generate a visualization of the node tree after each iteration
        from .node_visualizer import visualize_tree_only

        visualize_tree_only(self)

        return self.current_node.is_successful

    def mark_node_terminal(self, node):
        """
        Mark a node and all its descendants as terminal.
        Then check if any ancestors should be marked terminal.

        Args:
            node: The node to mark as terminal
        """
        # Mark the node itself and all descendants as terminal
        self._mark_subtree_terminal(node)

        # Check if any ancestors should be marked terminal
        self._check_ancestors_terminal(node.parent)

    def _mark_subtree_terminal(self, node):
        """
        Recursively mark a node and all its descendants as terminal.

        Args:
            node: The node to mark as terminal
        """
        if node.is_terminal:
            return

        node.is_terminal = True
        logger.info(f"Marking node {node.id} as terminal")

        # Recursively mark all children
        for child in node.children:
            self._mark_subtree_terminal(child)

    def _check_ancestors_terminal(self, node):
        """
        Recursively check if ancestors should be marked as terminal.
        An ancestor is terminal if fully expanded and all children are terminal.

        Args:
            node: The ancestor node to check
        """
        if node is None:
            return

        if not self._can_add_child(node) and all(child.is_terminal for child in node.children):
            node.is_terminal = True
            logger.info(f"Marking ancestor node {node.id} as terminal (all children terminal)")

            # Continue checking up the tree
            self._check_ancestors_terminal(node.parent)

    def _get_all_nodes(self) -> List[Node]:
        """
        Get all nodes in the tree.

        Returns:
            List of all nodes
        """
        all_nodes = []

        def _collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                _collect_nodes(child)

        _collect_nodes(self.root_node)
        return all_nodes

    def create_best_run_copy(self):
        """Create a 'best_run' folder that symlinks to the best node folder."""
        # Determine which node to link
        target_node = None
        link_reason = ""

        if self._best_node:
            target_node = self._best_node
            link_reason = f"best validation score ({self._best_validation_score:.4f})"
        elif self.last_successful_node:
            target_node = self.last_successful_node
            link_reason = "last successful execution"
        else:
            logger.warning("No best node or successful node found. Cannot create best_run link.")
            return

        # Create paths
        source_folder = self.get_iteration_folder(target_node)
        best_run_folder = os.path.join(self.output_folder, "best_run")

        # Verify source folder exists
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder does not exist: {source_folder}")
            return

        # Check if source folder has an 'output' subdirectory
        source_output_folder = os.path.join(source_folder, "output")
        if not os.path.exists(source_output_folder):
            logger.warning(f"Source output folder does not exist: {source_output_folder}")
            return

        # Handle existing best_run folder/link
        old_best_folder = None
        if os.path.exists(best_run_folder) or os.path.islink(best_run_folder):
            try:
                if os.path.islink(best_run_folder):
                    # Save the old target for potential cleanup
                    logger.debug(f"Reading existing best_run symlink target: {best_run_folder}")
                    old_link_target = os.readlink(best_run_folder)
                    old_best_folder = os.path.abspath(os.path.join(os.path.dirname(best_run_folder), old_link_target))
                    logger.debug(
                        f"Unlinking existing best_run symlink: {best_run_folder} (pointed to {old_best_folder})"
                    )
                    os.unlink(best_run_folder)
                    logger.info("Removed existing best_run symlink")
                else:
                    import shutil

                    logger.debug(f"Removing existing best_run folder (not a symlink): {best_run_folder}")
                    shutil.rmtree(best_run_folder)
                    logger.info("Removed existing best_run folder")
            except Exception as e:
                logger.error(f"Failed to remove existing best_run folder/link: {e}")
                return

        try:
            # Log completion marker
            logger.brief(
                f"Task completed successfully! Best node: {target_node.id} with validation score {target_node.validation_score}"
            )

            # Copy all files from source_output_folder to self.output_folder
            import shutil

            logger.debug(
                f"Starting copy of output folder contents from {source_output_folder} to {self.output_folder}"
            )
            for item in os.listdir(source_output_folder):
                source_item = os.path.join(source_output_folder, item)
                dest_item = os.path.join(self.output_folder, item)

                logger.debug(f"Copying item: {item}")
                if os.path.isfile(source_item):
                    logger.debug(f"  Copying file: {source_item} -> {dest_item}")
                    shutil.copy2(source_item, dest_item)
                elif os.path.isdir(source_item):
                    logger.debug(f"  Copying directory: {source_item} -> {dest_item}")
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
            logger.debug("Finished copying output folder contents")

            # Create symbolic link to the source folder instead of copying
            logger.debug(f"About to create symlink: {best_run_folder} -> {source_folder}")
            logger.info("Creating best_run symlink to best solution folder (instant operation, saves disk space)")
            os.symlink(source_folder, best_run_folder, target_is_directory=True)
            logger.debug("Successfully created best_run symlink")

            logger.info(f"Created best_run symlink (linked to node {target_node.id} - {link_reason})")

            # Save summary information in the target node folder
            summary_content = [
                "Best Run Summary",
                "================",
                f"Linked to: node_{target_node.id}",
                f"Reason: {link_reason}",
                f"Tool used: {target_node.tool_used}",
                f"Symlink created at: {os.path.basename(best_run_folder)}",
                "",
                self.get_validation_score_summary(),
                "",
                "Tool Usage Summary:",
                "==================",
                f"Available tools: {', '.join(self.available_tools)}",
                f"Tools used: {', '.join(self.used_tools)}",
                f"Tools not used: {', '.join(set(self.available_tools) - self.used_tools)}",
            ]

            # Save summary in both the main output folder and the target node folder
            summary_text = "\n".join(summary_content)

            self.save_and_log_states(
                content=summary_text, save_name="best_run_summary.txt", node=target_node, add_uuid=False
            )

            # Clean up old best folder if cleanup is enabled and it's not the same as the new one
            if old_best_folder and self.config.remove_current_iteration_folder:
                source_folder_abs = os.path.abspath(source_folder)
                logger.debug(f"Checking if old best folder should be removed: {old_best_folder}")
                logger.debug(f"  New best folder: {source_folder_abs}")
                logger.debug(f"  Are they different? {old_best_folder != source_folder_abs}")
                logger.debug(f"  Does old folder exist? {os.path.exists(old_best_folder)}")
                if old_best_folder != source_folder_abs and os.path.exists(old_best_folder):
                    try:
                        import shutil

                        logger.debug(f"About to remove old best folder: {old_best_folder}")
                        shutil.rmtree(old_best_folder)
                        logger.info(
                            f"Removed old best node folder {old_best_folder} (superseded by new best node {target_node.id})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to remove old best folder {old_best_folder}: {e}")

        except Exception as e:
            logger.error(f"Failed to create symlink: {e}")

    def remove_current_iteration_folder(self):
        """Remove the current iteration folder to save disk space.

        IMPORTANT: This should NOT be called if the current node is linked by best_run symlink,
        as it would break the symlink. Only call this for intermediate nodes that are not the best.
        """
        if not self.current_node:
            logger.warning("Current node is None.")
            return

        source_folder = self.get_iteration_folder(self.current_node)
        best_run_folder = os.path.join(self.output_folder, "best_run")

        logger.debug(f"Checking if iteration folder can be removed: {source_folder}")
        logger.debug(f"  Current node: {self.current_node.id}")
        logger.debug(f"  Best run folder: {best_run_folder}")

        # Check if best_run symlink exists and points to the current folder
        if os.path.islink(best_run_folder):
            logger.debug("  best_run is a symlink, checking target...")
            link_target = os.readlink(best_run_folder)
            # Resolve to absolute paths for comparison
            link_target_abs = os.path.abspath(os.path.join(os.path.dirname(best_run_folder), link_target))
            source_folder_abs = os.path.abspath(source_folder)

            logger.debug(f"  Symlink target (absolute): {link_target_abs}")
            logger.debug(f"  Current folder (absolute): {source_folder_abs}")
            logger.debug(f"  Are they the same? {link_target_abs == source_folder_abs}")

            if link_target_abs == source_folder_abs:
                logger.info(
                    f"Skipping removal of Node {self.current_node.id} folder - it is linked by best_run symlink."
                )
                return

        if os.path.exists(source_folder):
            import shutil

            try:
                logger.debug(f"About to remove iteration folder: {source_folder}")
                shutil.rmtree(source_folder)
                logger.info(f"Removed iteration folder of Node {self.current_node.id} to save disk space.")
            except Exception as e:
                logger.error(f"Failed to remove existing current iteration folder: {e}")
                return
        else:
            logger.debug(f"Iteration folder does not exist, nothing to remove: {source_folder}")

    def get_validation_score_summary(self) -> str:
        """
        Get a summary of all validation scores.

        Returns:
            A summary string
        """
        all_nodes = self._get_all_nodes()
        nodes_with_scores = [node for node in all_nodes if node.validation_score is not None]

        if not nodes_with_scores:
            return "No validation scores available."

        summary = ["Validation Score Summary:"]
        for node in nodes_with_scores:
            marker = " (BEST)" if node == self._best_node else ""
            summary.append(f"Node {node.id} ({node.tool_used}): {node.validation_score}{marker}")

        if self._best_node:
            summary.append(
                f"\nBest score: {self._best_validation_score:.4f} from node {self._best_node.id} using {self._best_node.tool_used}"
            )

        return "\n".join(summary)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    # ==================== Checkpoint-Resume ====================

    def save_checkpoint(self, phase: str) -> str:
        """Save full manager state to a JSON checkpoint file.

        Args:
            phase: The phase at which we're checkpointing ("init" or "step")

        Returns:
            Path to the saved checkpoint file
        """
        import json
        from datetime import datetime

        from omegaconf import OmegaConf

        last_step_result = None
        if phase == "step" and self.current_node and self.current_node != self.root_node:
            last_step_result = {
                "node_id": self.current_node.id,
                "success": self.current_node.is_successful,
                "validation_score": self.current_node.validation_score,
                "tool_used": self.current_node.tool_used,
                "stage": self.current_node.stage,
                "error_message": self.current_node.error_message[:2000] if self.current_node.error_message else "",
            }

        checkpoint = {
            "version": "1.0",
            "phase": phase,
            "step_number": self.time_step,
            "timestamp": datetime.now().isoformat(),
            "input_data_folder": self.input_data_folder,
            "output_folder": self.output_folder,
            "config": OmegaConf.to_container(self.config, resolve=True),
            "manager_state": {
                "data_prompt": self.data_prompt,
                "description_files": getattr(self, "description_files", ""),
                "task_description": getattr(self, "task_description", ""),
                "available_tools": getattr(self, "available_tools", []),
                "used_tools": list(self.used_tools),
                "time_step": self.time_step,
                "best_validation_score": self._best_validation_score,
                "worst_validation_score": self._worst_validation_score,
                "best_node_id": self._best_node.id if self._best_node else None,
                "last_successful_node_id": self.last_successful_node.id if self.last_successful_node else None,
                "best_step": self.best_step,
                "last_successful_step": self.last_successful_step,
                "global_instructions": self.global_instructions,
                "all_error_analyses": self._all_error_analyses,
                "user_inputs": self.user_inputs,
                "initial_user_input": self.initial_user_input or "",
                "enable_per_iteration_instruction": self.enable_per_iteration_instruction,
            },
            "tree": self.root_node.to_dict(),
            "last_step_result": last_step_result,
        }

        checkpoint_path = os.path.join(self.output_folder, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.brief(f"Checkpoint saved: {checkpoint_path} (phase={phase}, step={self.time_step})")
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, path: str, config=None) -> "NodeManager":
        """Restore a NodeManager from a checkpoint file.

        Args:
            path: Path to checkpoint JSON file
            config: Optional config override. If None, uses config from checkpoint.

        Returns:
            Restored NodeManager instance
        """
        import json

        from omegaconf import OmegaConf

        with open(path, "r") as f:
            checkpoint = json.load(f)

        state = checkpoint["manager_state"]

        # Use provided config or restore from checkpoint
        if config is None:
            config = OmegaConf.create(checkpoint["config"])

        # Create instance without calling __init__
        manager = cls.__new__(cls)

        # Restore paths
        manager.input_data_folder = checkpoint["input_data_folder"]
        manager.output_folder = checkpoint["output_folder"]
        manager.config = config

        # Restore scalars
        manager.enable_per_iteration_instruction = state.get("enable_per_iteration_instruction", False)
        manager.initial_user_input = state.get("initial_user_input", "")
        manager.time_step = state["time_step"]
        manager._best_validation_score = state["best_validation_score"]
        manager._worst_validation_score = state["worst_validation_score"]
        manager.best_step = state["best_step"]
        manager.last_successful_step = state["last_successful_step"]

        # Restore collections
        manager.data_prompt = state["data_prompt"]
        manager.description_files = state.get("description_files", "")
        manager.task_description = state.get("task_description", "")
        manager.available_tools = state.get("available_tools", [])
        manager.used_tools = set(state.get("used_tools", []))
        manager.global_instructions = state.get("global_instructions", [])
        manager.pending_local_instruction = None
        manager._all_error_analyses = state.get("all_error_analyses", [])
        manager.user_inputs = state.get("user_inputs", [])

        # MCTS parameters
        manager.exploration_constant = config.exploration_constant
        manager.max_debug_depth = config.max_debug_depth
        manager.failure_offset = config.failure_offset
        manager.failure_penalty_weight = config.failure_penalty_weight

        # Threading
        manager._node_lock = threading.Lock()
        manager.search_start_time = time.time()
        manager.target_prompt_instance = None

        # Rebuild tree
        manager.root_node = Node.from_dict(checkpoint["tree"])
        manager.current_node = manager.root_node

        # Re-derive best_node and last_successful_node by scanning the tree
        all_nodes = manager._get_all_nodes()
        best_node_id = state.get("best_node_id")
        last_successful_id = state.get("last_successful_node_id")
        manager._best_node = None
        manager.last_successful_node = None

        for node in all_nodes:
            if best_node_id is not None and node.id == best_node_id:
                manager._best_node = node
            if last_successful_id is not None and node.id == last_successful_id:
                manager.last_successful_node = node
            # Also set current_node to the latest node
            if node.time_step == manager.time_step:
                manager.current_node = node

        # Create output folder if needed
        Path(manager.output_folder).mkdir(parents=True, exist_ok=True)

        # Initialize agents (stateless — recreated fresh)
        manager._init_agents()

        logger.brief(f"Checkpoint loaded: {path} (phase={checkpoint['phase']}, step={manager.time_step})")
        return manager

    def rollback_last_step(self):
        """Discard the last MCTS step (most recent node) and re-derive state.

        Removes the node with the highest time_step from the tree,
        cleans up its disk folder, and re-derives best/worst scores.
        """
        import shutil

        # Find the node with the highest time_step
        all_nodes = self._get_all_nodes()
        non_root = [n for n in all_nodes if n.id is not None and n.id >= 0]
        if not non_root:
            logger.warning("No nodes to rollback.")
            return

        last_node = max(non_root, key=lambda n: n.time_step)
        logger.brief(f"Rolling back node {last_node.id} (stage={last_node.stage}, tool={last_node.tool_used})")

        # Remove from parent
        if last_node.parent:
            last_node.parent.remove_child(last_node)

        # Decrement time_step
        self.time_step = max(n.time_step for n in all_nodes if n != last_node)

        # Remove disk folder
        node_folder = os.path.join(self.output_folder, f"node_{last_node.id}")
        if os.path.exists(node_folder):
            shutil.rmtree(node_folder)
            logger.info(f"Removed folder: {node_folder}")

        # Re-derive best/worst scores
        remaining = [n for n in all_nodes if n != last_node]
        scored = [n for n in remaining if n.validation_score is not None]
        if scored:
            best = max(scored, key=lambda n: n.validation_score)
            self._best_node = best
            self._best_validation_score = best.validation_score
            self.best_step = best.time_step
            self._worst_validation_score = min(n.validation_score for n in scored)
        else:
            self._best_node = None
            self._best_validation_score = None
            self._worst_validation_score = None
            self.best_step = -1

        # Re-derive last_successful_node
        successful = [n for n in remaining if n.is_successful and n.id >= 0]
        if successful:
            self.last_successful_node = max(successful, key=lambda n: n.time_step)
            self.last_successful_step = self.last_successful_node.time_step
        else:
            self.last_successful_node = None
            self.last_successful_step = -1

        # Pop last error analysis if last node had an error
        if last_node.error_analysis and self._all_error_analyses:
            self._all_error_analyses.pop()

        # Pop last user_input
        if self.user_inputs:
            self.user_inputs.pop()

        # Un-mark terminal ancestors if needed (rollback may open up expansion)
        if last_node.parent and last_node.parent.is_terminal:
            last_node.parent.is_terminal = False
            self._check_ancestors_terminal(last_node.parent.parent)

        # Remove tool from used_tools if no other node uses it
        remaining_tools = {n.tool_used for n in remaining if n.tool_used}
        self.used_tools = remaining_tools

        self.current_node = self.root_node
        logger.brief(f"Rollback complete. Tree now has {len(remaining) - 1} nodes (excluding root).")

    def _find_debug_origin(self, node: Node) -> Optional[Node]:
        """
        Find the original node that started this debugging chain.

        Args:
            node: The current node in the debug chain

        Returns:
            The original node that started the debug chain
        """
        # Go up the tree until we find a non-debug node
        current = node
        while current.parent and current.parent.stage == "debug":
            current = current.parent

        debug_origin = current.parent
        assert not debug_origin.is_successful

        return debug_origin

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def visualize_results(self, output_path: Optional[str] = None) -> str:
        """
        Generate a PDF visualization of the node structure.

        Args:
            output_path: Path to save the PDF. If not provided, it will be saved to
                        the output folder.

        Returns:
            The path to the generated PDF file
        """
        from .node_visualizer import visualize_results

        return visualize_results(self, output_path)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def _compute_score_range(self) -> tuple:
        """Compute the min and max average scores across all scored nodes.

        Used for min-max exponential normalization in UCT:
        - normalized = (score - min) / (max - min)   -> [0, 1]
        - shaped = (exp(normalized/T) - 1) / (exp(1/T) - 1)  -> [0, 1]

        Returns:
            (min_score, max_score) tuple across all scored nodes,
            or (0.0, 0.0) if no scored nodes exist.
        """
        all_nodes = self._get_all_nodes()
        scored_nodes = [n for n in all_nodes if n.validated_visits > 0]
        if not scored_nodes:
            return (0.0, 0.0)
        avg_scores = [n.validated_reward / n.validated_visits for n in scored_nodes]
        return (min(avg_scores), max(avg_scores))

    def compute_uct_value(self, node):
        # Cache score range per step to avoid recomputing for every node
        if not hasattr(self, "_cached_score_range") or self._cached_score_range_step != self.time_step:
            self._cached_score_range = self._compute_score_range()
            self._cached_score_range_step = self.time_step
        score_min, score_max = self._cached_score_range
        return node.uct_value(
            self.exploration_constant,
            failure_offset=self.failure_offset,
            failure_penalty_weight=self.failure_penalty_weight,
            score_min=score_min,
            score_max=score_max,
            score_temperature=self.score_temperature,
        )

    # Properties to maintain compatibility with Manager API
    @property
    def user_input(self) -> str:
        """Get the user input for the current step."""
        if self.time_step < 0 or self.time_step >= len(self.user_inputs):
            return ""
        return self.user_inputs[self.time_step]

    @property
    def best_validation_score(self) -> float:
        """Get the best validation score."""
        return self._best_validation_score if self._best_validation_score is not None else 0.0

    @property
    def best_node(self) -> Node:
        """Get the best node."""
        return self._best_node

    @property
    def python_code(self) -> str:
        """Get the Python code from the current node."""
        return self.current_node.python_code if self.current_node else ""

    @property
    def python_file_path(self) -> str:
        """Get the Python file path for the current node."""
        if not self.current_node:
            return ""
        return os.path.join(self.get_iteration_folder(self.current_node), "generated_code.py")

    @property
    def previous_python_code(self) -> str:
        """Get the Python code from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.python_code
        return ""

    @property
    def bash_script(self) -> str:
        """Get the Bash script from the current node."""
        return self.current_node.bash_script if self.current_node else ""

    @property
    def previous_bash_script(self) -> str:
        """Get the Bash script from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.bash_script
        return ""

    @property
    def error_message(self) -> str:
        """Get the error message from the current node."""
        return self.current_node.error_message if self.current_node else ""

    @property
    def previous_error_message(self) -> str:
        """Get the error message from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.error_message
        return ""

    @property
    def error_analysis(self) -> str:
        """Get the error analysis from the current node."""
        return self.current_node.error_analysis if self.current_node else ""

    @property
    def previous_error_analysis(self) -> str:
        """Get the error analysis from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.error_analysis
        return ""

    @property
    def all_previous_error_analyses(self) -> str:
        """Get all error analyses from previous nodes."""
        # TODO: make this recursive, handle debugging code and successful ones differently
        return "\n\n".join(self._all_error_analyses)

        if not self.current_node:
            return ""

        analyses = []
        node = self.current_node
        while node.parent:
            node = node.parent
            if node.error_analysis:
                analyses.append(node.error_analysis)

        return "\n\n".join(analyses)

    @property
    def per_iteration_output_folder(self) -> str:
        """Get the output folder for the current iteration."""
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization", "output")
        return self.get_per_iteration_output_folder(self.current_node)

    @property
    def iteration_folder(self) -> str:
        """Get the folder for the current iteration."""
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization")
        return self.get_iteration_folder(self.current_node)

    @property
    def tutorial_retrieval(self) -> str:
        """Get the tutorial retrieval for the current step."""
        if self.current_node:
            return self.current_node.tutorial_retrieval
        else:
            logger.warning("Invalid node while asking for tutorial_retrieval")

    @property
    def tutorial_prompt(self) -> str:
        """Get the tutorial prompt for the current step."""
        return self.current_node.tutorial_prompt if self.current_node else ""

    @property
    def previous_tutorial_prompt(self) -> str:
        """Get the tutorial prompt from the previous step."""
        return self.current_node.prev_tutorial_prompt

    @property
    def global_instructions_prompt(self) -> str:
        """Get global instructions formatted for prompts."""
        if not self.global_instructions:
            return ""
        return "\n".join(self.global_instructions)

    @property
    def local_instructions_prompt(self) -> str:
        """Get local instructions for the current node's subtree."""
        local = self._collect_local_instructions()
        if not local:
            return ""
        return "\n".join(local)

    @property
    def common_env_file(self) -> str:
        return registry.registry_path / "_common" / "requirements.txt"

    @property
    def selected_tool(self) -> str:
        return self.current_node.tool_used

    @property
    def selected_tool_env_file(self) -> str:
        tool_path = registry.get_tool(self.selected_tool)["path"]
        return registry.registry_path / tool_path / "requirements.txt"

    @property
    def configure_env(
        self,
    ):
        if self.selected_tool.lower() in ["machine learning", "huggingface", "fairseq"]:
            return True
        else:
            return self.config.configure_env

    @property
    def code_to_improve(
        self,
    ):
        if self.current_node.stage == "evolve":
            return self.current_node.parent.python_code
        else:
            return None

    @property
    def code_to_debug(
        self,
    ):
        if self.current_node.stage == "debug":
            return self.current_node.parent.python_code
        else:
            return None
