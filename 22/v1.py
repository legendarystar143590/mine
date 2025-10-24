
from __future__ import annotations
# IMPORTS
# =============================================================================
import textwrap as tw  # Import as tw to avoid conflict with IndentationHelper alias
import re
import os
import sys
import uuid
import json
import time
import ast
import random
import asyncio
import logging
import requests
import subprocess
from json import JSONDecodeError
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast, get_args, Literal
from enum import Enum
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
from dataclasses import dataclass, field
from typing_extensions import final
from uuid import uuid4

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1800"))
# DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://31.22.104.92:8000")
EVALUATION_RUN_ID = os.getenv("RUN_ID") or str(uuid4())
MAX_FIX_TASK_STEPS = 400
DEFAULT_INDENT_SIZE = 4

# Model configuration - Updated with best performing models
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
AGENT_MODELS=[GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]

GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = tw.dedent(
"""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
Return only the final Python code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)


INFINITE_LOOP_CHECK_PROMPT = tw.dedent(
"""
You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.

CRITICAL INFINITE LOOP DETECTION:
1. Check for while True: loops without guaranteed exit conditions
2. Verify all while loops have clear termination conditions
3. Ensure recursive functions have proper base cases
4. Look for loops that depend on external state that might never change
5. Check for patterns that could lead to infinite iteration

If you find potential infinite loops:
- Provide a corrected version of the code
- Ensure all loops have finite termination conditions
- Add reasonable iteration limits or timeout mechanisms where appropriate

If no infinite loops are detected:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

example:
```python
a.py
contents of a.py

b.py
contents of b.py
```
"""
)


GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = tw.dedent(
"""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)

GENERATE_INITIAL_SOLUTION_PROMPT = tw.dedent("""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.

Return only the final python files code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)

TESTCASES_CHECK_PROMPT = tw.dedent(
"""
You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

Important:
1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
3. Minimize all testcases as you have context and generation limit

If no invalid testcases are detected and covered all critical edge cases:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.

example:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)


GENERATE_INITIAL_TESTCASES_PROMPT = tw.dedent("""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)

PROBLEM_TYPE_CHECK_PROMPT = tw.dedent(
'''
You are the problem type checker that will categories problem type into:

1. CREATE: If the problem statement is about creating a new functionality from scratch.
2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

Only respond with the "FIX" or "CREATE".
'''
)

class Logger:
    """Custom logger with colored messages and color tag support."""
    
    # ANSI color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'gray': '\033[90m',
        'light_gray': '\033[37m', 
        'reset': '\033[0m',
        'faint': '\033[2m',   
        'very_faint': '\033[2m\033[90m',
    }
    
    # Default colors for each log level
    LEVEL_COLORS = {
        'DEBUG': 'very_faint',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
    
    def __init__(self, name="agent_main"):
        self.name = name
    
    def _colorize(self, text, color):
        """Apply color to text."""
        if color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text
    
    def _parse_color_tags(self, message):
        """Parse custom color tags like <red>text</red>."""
        import re
        # Pattern to match only valid color tags (not XML tags)
        valid_colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'gray', 'light_gray', 'bold', 'dim', 'faint', 'very_faint']
        color_pattern = '|'.join(valid_colors)
        pattern = rf'<({color_pattern})>(.*?)</\1>'
        
        def replace_color(match):
            color = match.group(1)
            text = match.group(2)
            return self._colorize(text, color)
        
        return re.sub(pattern, replace_color, message)
    
    def _log(self, level, message):
        """Internal logging method."""
        # Get default color for this level
        default_color = self.LEVEL_COLORS.get(level, 'white')
        
        # Check if message has color tags
        has_color_tags = any(f'<{color}>' in message for color in self.COLORS.keys())
        
        if has_color_tags:
            # Message has color tags, parse them first, then apply default color to entire result
            parsed_message = self._parse_color_tags(message)
            colored_message = self._colorize(parsed_message, default_color)
        else:
            # No color tags, apply default level color to entire message
            colored_message = self._colorize(message, default_color)
        
        print(colored_message)
    
    def debug(self, message):
        """Log debug message."""
        self._log('DEBUG', message)
    
    def info(self, message):
        """Log info message."""
        self._log('INFO', message)
    
    def warning(self, message):
        """Log warning message."""
        self._log('WARNING', message)
    
    def error(self, message):
        """Log error message."""
        self._log('ERROR', message)
    
    def critical(self, message):
        """Log critical message."""
        self._log('CRITICAL', message)
    
    # Convenience methods
    def warn(self, message):
        """Alias for warning."""
        self.warning(message)
    
    def err(self, message):
        """Alias for error."""
        self.error(message)

# Create global logger instance
logger = Logger("agent_main")

# Optional imports with fallbacks
try:
    import jsonschema
except ImportError:
    jsonschema = None

# CONSTANTS
# =============================================================================

SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 200000

# UTILITY CLASS
# =============================================================================

class Utils:
    """Utility class containing all helper functions."""
    
    @staticmethod
    def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
        """Truncate content and append a notice if content exceeds the specified length."""
        TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
        return (
            content
            if not truncate_after or len(content) <= truncate_after
            else content[:truncate_after] + TRUNCATED_MESSAGE
        )
        
    @staticmethod
    def is_path_in_directory(directory: Path, path: Path) -> bool:
        """Check if a path is within a directory."""
        directory = directory.resolve()
        path = path.resolve()
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False

    @staticmethod
    def ensure_git_initialize():
        """Initialize git repository if not already initialized, with temporary config."""
        logger.info("Starting git initialization check...")
        try:
            work_dir = os.getcwd()
            # Initialize git repo if not already initialized
            if not os.path.exists(".git"):
                logger.debug("Initializing git repository...")
                subprocess.run(["git", "init"], check=True, capture_output=True)
                subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir], capture_output=True)
                
                # Verify .git was created in current directory
                logger.debug(f".git exists: {os.path.exists('.git')}")
                logger.debug(f"Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
                
                # Set local git config (only for this repo)
                logger.debug("Setting git config...")
                subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True, capture_output=True)
                subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True, capture_output=True)

                # Add all files
                logger.debug("Adding all files...")
                subprocess.run(["git", "add", "."], check=True, capture_output=True)
                
                # Commit (ignore error if nothing to commit)
                logger.debug("Creating initial commit...")  
                result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.debug("Initial commit created successfully")
                else:
                    logger.debug(f"Commit result: {result.stderr.strip()}")
                    
                logger.info("Git initialization completed successfully")
            else:
                logger.info("Git repository already exists")
                subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
        except Exception as e:
            logger.error(f"Could not initialize git repository: {e}")


    @staticmethod
    def set_env_for_agent():
        if os.getcwd() not in os.environ.get("PYTHONPATH",""):
            os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
        if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
            os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"
    
    @staticmethod
    def create_final_git_patch(temp_files: list[str] = []) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in temp_files:
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )

            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                logger.warning(f"git diff (stderr): {diff.stderr.strip()}")

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.error("Error generating git patch")
            return f"Error generating git patch: {e}"
        
    @staticmethod
    def format_log(text: str, label: str):
        logger.info(f"\n<yellow>-------------------------------- [{label}] --------------------------------</yellow>")
        logger.info(text)
        logger.info(f"<yellow>------------------------------ End [{label}] ------------------------------</yellow>")
    

# This ensures the module is properly registered in sys.modules for dataclass compatibility
def ensure_module_registration():
    """Ensure module is properly registered for dataclass compatibility."""
    # Create a proper module object
    import types
    current_module = types.ModuleType('agent-main')
    current_module.__name__ = 'agent-main'
    current_module.__file__ = __file__
    current_module.__dict__.update(globals())
    
    # Register the module
    sys.modules['agent-main'] = current_module
    sys.modules['agent'] = current_module  # Also register as 'agent' for compatibility

ensure_module_registration()

class Types:
    """Container for all custom data type classes used by the agent."""

    @dataclass
    class ToolParam:
        """Internal representation of LLM tool."""
        name: str
        description: str
        input_schema: dict[str, Any]

    @dataclass
    class ToolCall:
        """Internal representation of LLM-generated tool call."""
        tool_call_id: str
        tool_name: str
        tool_input: Any

    @dataclass
    class ToolResult:
        """Internal representation of LLM tool result."""
        tool_call_id: str
        tool_name: str
        tool_output: Any

    @dataclass
    class ToolFormattedResult:
        """Internal representation of formatted LLM tool result."""
        tool_call_id: str
        tool_name: str
        tool_output: str

    @dataclass
    class TextPrompt:
        """Internal representation of user-generated text prompt."""
        text: str

    @dataclass
    class TextResult:
        """Internal representation of LLM-generated text result."""
        text: str

    @dataclass
    class ToolCallParameters:
        tool_call_id: str
        tool_name: str
        tool_input: Any

    @dataclass
    class ToolImplOutput:
        """Output from an LLM tool implementation."""
        tool_output: str
        tool_result_message: str
        auxiliary_data: dict[str, Any] = field(default_factory=dict)
        
    @dataclass
    class LLMMessage:
        role: str
        content: str
        
    @dataclass(frozen=True)
    class IndentType:
        """Class representing indentation type with size attribute."""
        type: Literal["space", "tab", "mixed"]
        size: int = 4
        most_used: Types.IndentType | None = None  # Tracks predominant indent type for mixed

        @property
        def is_tab(self) -> bool:
            return self.type == "tab"

        @property
        def is_mixed(self) -> bool:
            return self.type == "mixed"

        @property
        def is_space(self) -> bool:
            return self.type == "space"

        @classmethod
        def space(cls, size: int = 4) -> Types.IndentType:
            """Create a space indentation type with the specified size."""
            return cls(type="space", size=size)

        @classmethod
        def tab(cls, size: int = 1) -> Types.IndentType:
            """Create a tab indentation type (size is typically 1)."""
            return cls(type="tab", size=size)

        @classmethod
        def mixed(cls, most_used: Types.IndentType | None = None) -> Types.IndentType:
            """Create a mixed indentation type."""
            return cls(type="mixed", size=1, most_used=most_used)

        def __repr__(self):
            if self.is_mixed:
                most_used_str = f", most_used={self.most_used}" if self.most_used else ""
                return f"IndentType({self.type}{most_used_str})"
            if self.is_tab:
                return f"IndentType({self.type})"
            return f"IndentType({self.type}, size={self.size})"

    class ExtendedToolImplOutput(ToolImplOutput):
        @property
        def success(self) -> bool:
            """Get success status from metadata."""
            return bool(self.auxiliary_data.get("success", False))

    class ToolError(Exception):
        def __init__(self, message: str):
            self.message = message
            super().__init__(message)

        def __str__(self):
            return self.message
        
    Command = Literal[
        "view",
        "create",
        "str_replace",
        "insert",
        "undo_edit",
    ]

    class ThoughtData(TypedDict, total=False):
        """Type definition for thought data."""
        thought: str
        thoughtNumber: int
        totalThoughts: int
        isRevision: Optional[bool]
        revisesThought: Optional[int]
        branchFromThought: Optional[int]
        branchId: Optional[str]
        needsMoreThoughts: Optional[bool]
        nextThoughtNeeded: bool

    # Type aliases
    AssistantContentBlock = TextResult | ToolCall
    UserContentBlock = TextPrompt | ToolFormattedResult
    GeneralContentBlock = UserContentBlock | AssistantContentBlock
    LLMMessages = list[list[GeneralContentBlock]]
    ToolInputSchema = dict[str, Any]
    """A JSON schema describing the input to a tool."""

# TOKEN COUNTER
# =============================================================================

class ClaudeTokenCounter:
    def count_tokens(self, prompt_chars: str) -> int:
        return len(prompt_chars) // 3

# DIALOG MESSAGES
# =============================================================================

class DialogMessages:
    """Keeps track of messages that compose a dialog.

    A dialog alternates between user and assistant turns. Each turn consists
    of one or more messages, represented by Types.GeneralContentBlock.

    A user turn consists of one or more prompts and tool results.
    An assistant turn consists of a model answer and tool calls.
    """

    def __init__(
        self,
        use_prompt_budgeting: bool = False,
    ):
        self._message_lists: list[list[Types.GeneralContentBlock]] = []
        self.token_counter = ClaudeTokenCounter()
        self.use_prompt_budgeting = use_prompt_budgeting
        self.truncation_history_token_cts: list[int] = []
        self.token_budget_to_trigger_truncation = 100_000
        self.truncate_all_but_N = 5

    def add_user_prompt(
        self, message: str, allow_append_to_tool_call_results: bool = False
    ):
        """Add a user prompt to the dialog."""
        if self.is_user_turn():
            self._message_lists.append([Types.TextPrompt(message)])
        else:
            if allow_append_to_tool_call_results:
                user_messages = self._message_lists[-1]
                for user_message in user_messages:
                    if isinstance(user_message, Types.TextPrompt):
                        raise ValueError(
                            f"Last user turn already contains a text prompt: {user_message}"
                        )
                user_messages.append(Types.TextPrompt(message))
            else:
                self._assert_user_turn()

    def add_tool_call_result(self, parameters: Types.ToolCallParameters, result: str):
        """Add the result of a tool call to the dialog."""
        self.add_tool_call_results([parameters], [result])

    def add_tool_call_results(
        self, parameters: list[Types.ToolCallParameters], results: list[str]
    ):
        """Add the result of a tool call to the dialog."""
        self._assert_user_turn()
        self._message_lists.append(
            [
                Types.ToolFormattedResult(
                    tool_call_id=params.tool_call_id,
                    tool_name=params.tool_name,
                    tool_output=result,
                )
                for params, result in zip(parameters, results)
            ]
        )

    def add_model_response(self, response: list[Types.AssistantContentBlock]):
        """Add the result of a model call to the dialog."""
        self._assert_assistant_turn()
        self._message_lists.append(cast(list[Types.GeneralContentBlock], response))

    def count_tokens(self) -> int:
        """Count the total number of tokens in the dialog."""
        total_tokens = 0
        for i, message_list in enumerate(self._message_lists):
            is_last_message_list = i == len(self._message_lists) - 1
            for message in message_list:
                if isinstance(message, (Types.TextPrompt, Types.TextResult)):
                    total_tokens += self.token_counter.count_tokens(message.text)
                elif isinstance(message, Types.ToolFormattedResult):
                    total_tokens += self.token_counter.count_tokens(message.tool_output)
                elif isinstance(message, Types.ToolCall):
                    total_tokens += self.token_counter.count_tokens(
                        json.dumps(message.tool_input)
                    )
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
        return total_tokens

    def run_truncation_strategy(self) -> None:
        """Truncate all the tool results apart from the last N turns."""
        logger.info(f"<yellow>Truncating all but the last {self.truncate_all_but_N} turns as we hit the token budget {self.token_budget_to_trigger_truncation}.</yellow>")

        old_token_ct = self.count_tokens()
        new_message_lists: list[list[Types.GeneralContentBlock]] = deepcopy(
            self._message_lists
        )

        for message_list in new_message_lists[: -self.truncate_all_but_N]:
            for message in message_list:
                if isinstance(message, Types.ToolFormattedResult):
                    message.tool_output = (
                        "[Truncated...re-run tool if you need to see output again.]"
                    )
                elif isinstance(message, Types.ToolCall):
                    if message.tool_name == "sequential_thinking":
                        message.tool_input["thought"] = (
                            "[Truncated...re-run tool if you need to see input/output again.]"
                        )
                    elif message.tool_name == "str_replace_editor":
                        if "file_text" in message.tool_input:
                            message.tool_input["file_text"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )
                        if "old_str" in message.tool_input:
                            message.tool_input["old_str"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )
                        if "new_str" in message.tool_input:
                            message.tool_input["new_str"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )

        self._message_lists = new_message_lists
        new_token_ct = self.count_tokens()
        logger.info("<yellow>[dialog_messages] Token count after truncation: {new_token_ct}</yellow>")
        self.truncation_history_token_cts.append(old_token_ct - new_token_ct)

    def get_messages_for_inference(self) -> list[Types.LLMMessage]:
        """Returns messages in the format the LM client expects."""
        if (
            self.use_prompt_budgeting
            and self.count_tokens() > self.token_budget_to_trigger_truncation
        ):
            self.run_truncation_strategy()
            
        messages = []
            
        for idx, message_list in enumerate(self._message_lists):
            role = "assistant" if idx % 2 == 0 else "user"
            content_parts = []
            
            for message in message_list:
                if str(type(message)) == str(Types.TextPrompt):
                    message = cast(Types.TextPrompt, message)
                    content_parts.append(message.text)
                elif str(type(message)) == str(Types.TextResult):
                    message = cast(Types.TextResult, message)
                    content_parts.append(message.text)
                elif str(type(message)) == str(Types.ToolCall):
                    message = cast(Types.ToolCall, message)
                    # Convert tool call to new <tool_call> XML format
                    try:
                        import json as _json
                        params_json = _json.dumps(message.tool_input, ensure_ascii=False)
                    except Exception:
                        params_json = str(message.tool_input)
                    tool_xml = (
                        f"<tool_call>\n"
                        f"<tool_name>{message.tool_name}</tool_name>\n"
                        f"<tool_id>{message.tool_call_id}</tool_id>\n"
                        f"<parameters>{params_json}</parameters>\n"
                        f"</tool_call>"
                    )
                    content_parts.append(tool_xml)
                elif str(type(message)) == str(Types.ToolFormattedResult):
                    message = cast(Types.ToolFormattedResult, message)
                    # Convert tool result to text format
                    result_text = f"Tool Result: {message.tool_output}"
                    content_parts.append(result_text)
                else:
                    # Handle other message types as text
                    content_parts.append(str(message))
            
            if content_parts:
                messages.append({
                    "role": role,
                    "content": "\n".join(content_parts)
                })
        return messages

    def drop_final_assistant_turn(self):
        """Remove the final assistant turn."""
        if self.is_user_turn():
            self._message_lists.pop()

    def drop_tool_calls_from_final_turn(self):
        """Remove tool calls from the final assistant turn."""
        if self.is_user_turn():
            new_turn_messages = [
                message
                for message in self._message_lists[-1]
                if not isinstance(message, Types.ToolCall)
            ]
            self._message_lists[-1] = cast(list[Types.GeneralContentBlock], new_turn_messages)

    def get_pending_tool_calls(self) -> list[Types.ToolCallParameters]:
        """Returns the tool calls from the last assistant turn."""
        self._assert_user_turn()
        if len(self._message_lists) == 0:
            return []
        tool_calls = []
        for message in self._message_lists[-1]:
            if isinstance(message, Types.ToolCall):
                tool_calls.append(
                    Types.ToolCallParameters(
                        tool_call_id=message.tool_call_id,
                        tool_name=message.tool_name,
                        tool_input=message.tool_input,
                    )
                )
        return tool_calls

    def get_last_model_text_response(self):
        """Returns the last model response as a string."""
        self._assert_user_turn()
        for message in self._message_lists[-1]:
            if isinstance(message, Types.TextResult):
                return message.text
        raise ValueError("No text response found in last model response")

    def get_last_user_prompt(self) -> str:
        """Returns the last user prompt."""
        self._assert_assistant_turn()
        for message in self._message_lists[-1]:
            if isinstance(message, Types.TextPrompt):
                return message.text
        raise ValueError("No text prompt found in last user prompt")

    def replace_last_user_prompt(self, new_prompt: str):
        """Replace the last user prompt with a new one."""
        self._assert_assistant_turn()
        for i, message in enumerate(self._message_lists[-1]):
            if isinstance(message, Types.TextPrompt):
                self._message_lists[-1][i] = Types.TextPrompt(new_prompt)
                return
        raise ValueError("No text prompt found in last user prompt")

    def clear(self):
        """Delete all messages."""
        self._message_lists = []

    def is_user_turn(self):
        return len(self._message_lists) % 2 == 1

    def is_assistant_turn(self):
        return len(self._message_lists) % 2 == 0

    def _assert_user_turn(self):
        assert self.is_user_turn(), "Can only add user prompts on user's turn"

    def _assert_assistant_turn(self):
        assert self.is_assistant_turn(), (
            "Can only get/replace last user prompt on assistant's turn"
        )

class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls, raw_text: str) -> Tuple[bool, Optional[str]]:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if raw_text is None or len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        if raw_text.startswith("ERROR:"):
            return False, "API Error: " + raw_text
        return True, None

    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def make_request(cls, messages: list[Types.LLMMessage], model: str, attempt: int=0, temperature: float=0.0, max_retries: int=5) -> str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"

        # Cache miss - make the actual request
        # Convert LLMMessage objects to dictionaries for JSON serialization
        messages_dict = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                messages_dict.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict):
                messages_dict.append(msg)
            else:
                # Fallback for other types
                messages_dict.append({"role": "user", "content": str(msg)})
        
        request_data = {
            "run_id": EVALUATION_RUN_ID,
            "messages": messages_dict,
                "temperature": temperature,
            "model": model
            }

        headers = {
            "Content-Type": "application/json"
        }
        
        for retry_attempt in range(max_retries + 1):
            try:
                response = requests.post(url, data=json.dumps(request_data), timeout=120, headers=headers)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return f"ERROR: Request timeout for model {model}"
            except requests.exceptions.ConnectionError as e:
                return f"ERROR: Connection failed for model {model}"
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                # Retry for 500 (Internal Server Error) or 504 (Gateway Timeout)
                if status_code in [500, 504] and retry_attempt < max_retries:
                    sleep_time = 2 ** retry_attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(sleep_time)
                    continue  # Retry the request
                return f"ERROR: HTTP error {status_code} for model {model}"
            except requests.exceptions.RequestException as e:
                return f"ERROR: Request failed for model {model}"
            
            try:
                response_json = response.json()
            except JSONDecodeError as e:
                return f"ERROR: Invalid JSON response for model {model}"
            
            try:
                # Safe OAI interface detection with proper null checking
                choices = response_json.get('choices') if isinstance(response_json, dict) else None
                is_oai_interface = (
                    isinstance(response_json, dict) and 
                    choices is not None and 
                    len(choices) > 0 and 
                    isinstance(choices[0], dict) and
                    choices[0].get('message') is not None
                )
                
                if is_oai_interface:
                    if choices and isinstance(choices, list) and choices[0] and isinstance(choices[0], dict):
                        raw_text = choices[0].get('message', {}).get('content', '')
                    else:
                        raise ValueError("Invalid response structure: 'choices' is None or improperly formatted")
                else:
                    if isinstance(response_json, str):
                        raw_text = response_json.strip("\n").strip()
                    else:
                        raw_text = response_json
                
                if not isinstance(raw_text, dict):
                    raw_text = str(raw_text).lstrip()
                return str(raw_text)
            except (KeyError, IndexError, TypeError) as e:
                return f"ERROR: Invalid response structure for model {model}"
            except Exception as e:
                return f"ERROR: Unexpected error for model {model}"
        
        # If we exhausted all retries
        return f"ERROR: Max retries exceeded for model {model}"

    @classmethod
    def inference(
        cls,
        system_prompt: str,
        instruction_prompt: str,
        messages: list[Types.LLMMessage],
        model: str,
        return_json: bool=False,
        temperature: float=0.0
    ) -> list[Types.AssistantContentBlock]:
        """Prod inference with caching - returns list of AssistantContentBlock"""
        # Log evaluation run ID once at the start of inference
        logger.info(f"<yellow>🚀 [EVALUATION_RUN_ID]: {EVALUATION_RUN_ID}</yellow>")
        
        proxy_messages: list[Types.LLMMessage] = []
        
        # Add system prompt if provided
        if system_prompt:
            proxy_messages.append(cast(Types.LLMMessage, {"role": "system", "content": system_prompt}))
        
        # Add instruction prompt if provided
        if instruction_prompt:
            proxy_messages.append(cast(Types.LLMMessage, {"role": "user", "content": instruction_prompt}))
        
        proxy_messages.extend(messages)

        # Make request to proxy API with retry logic and re-try on parse failures
        parse_retry_limit = 3
        last_error: str | None = None
        parse_start_time = time.time()
        
        for parse_attempt in range(parse_retry_limit):
            try:
                raw_text = cls._make_request_with_retry(proxy_messages, model=model, temperature=temperature)
            except RuntimeError as e:
                # Log the error and re-raise with more context
                elapsed = time.time() - parse_start_time
                logger.error(f"<red>  ├──💥 Network request failed after {elapsed:.1f}s: {str(e)}</red>")
                last_error = str(e)
                continue

            # Parse response and convert back to AssistantContentBlock format
            try:
                logger.info(f"<blue>  ├──🔍 Parsing response...</blue>")
                parse_start = time.time()
                tool_call, text_content = cls.parse_response(raw_text)
                parse_duration = time.time() - parse_start
                total_elapsed = time.time() - parse_start_time
                logger.info(f"<green>  ├──✅ Response parsed successfully in {parse_duration:.1f}s (total: {total_elapsed:.1f}s)</green>")
                break
            except Exception as e:
                last_error = str(e)
                elapsed = time.time() - parse_start_time
                logger.warning(f"<red>  ├──❌ Parse failed after {elapsed:.1f}s: {last_error}</red>")
                logger.info(f"<yellow>  ├──🔄 Retrying again...</yellow>")
                continue
        else:
            total_elapsed = time.time() - parse_start_time
            logger.error(f"<red>  ├──💥 Response parsing failed after {parse_retry_limit} retries ({total_elapsed:.1f}s): {last_error}</red>")
            raise RuntimeError(f"Response tool call parsing failed after retries: {last_error}")

        # Build response in AssistantContentBlock format
        response_blocks = []

        # Add text content if present
        if text_content.strip():
            response_blocks.append(Types.TextResult(text=text_content))

        response_blocks.append(tool_call)

        return response_blocks
    
    @classmethod
    def _make_request_with_retry(cls, messages: list[Types.LLMMessage], model: str, temperature: float = 0.0, max_retries: int = 5) -> str:
        """Make request with retry logic, cycling through different models on failure."""
        raw_text = 'not defined'
        error_counter = cls.get_error_counter()
        total_attempts = 0
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                current_model = AGENT_MODELS[(index + attempt) % len(AGENT_MODELS)]
                
                # Show sending message with timing
                if attempt == 0:
                    logger.info(f"<blue>  ├──📤 Sending request to {current_model}...</blue>")
                else:
                    elapsed = time.time() - start_time
                    logger.info(f"<yellow>  ├──🔄 Retrying with {current_model} (attempt {attempt + 1}/{max_retries}) after {elapsed:.1f}s...</yellow>")
                
                request_start = time.time()
                raw_text = cls.make_request(messages, model=current_model, temperature=temperature)
                request_duration = time.time() - request_start
                
                # Check if response is valid using the existing validation logic
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                
                # Additional validation: Check if response has proper format for this project
                if not cls._is_valid_response_format(raw_text):
                    raise Exception("Invalid response format - missing required XML structure")

                # Success message with timing
                total_elapsed = time.time() - start_time
                logger.info(f"<green>  ├──✅ Completed after {request_duration:.1f}s (total: {total_elapsed:.1f}s)</green>")
                
                # Note: single-tool enforcement is handled exclusively in parse_response at call site
                
                break
                
            except Exception as e:
                error_body = str(e)
                if attempt < max_retries - 1:
                    # Show failure and retry message with timing
                    elapsed = time.time() - start_time
                    logger.warning(f"<red>  ├──❌ Failed after {elapsed:.1f}s: {error_body}</red>")
                    
                    delay = 1.0
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name] += 1
                        delay = 2.0  # Longer delay for rate limits
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name] += 1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name] += 1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name] += 1
                        delay = 1.5  # Longer delay for timeouts
                    elif "Invalid JSON" in error_body or "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    elif "Network unreachable" in error_body or "Connection refused" in error_body:
                        error_counter[cls.ErrorType.NETWORK_ERROR.name] += 1
                        delay = 2.0  # Longer delay for network issues
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    
                    # Add context for retry (except for certain error types)
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and "TIMEOUT" not in error_body:
                        messages.append(cast(Types.LLMMessage, {"role": "assistant", "content": raw_text}))
                        messages.append(cast(Types.LLMMessage, {"role": "user", "content": f"Error occurred: {error_body}. Please try again with a different approach."}))
                    
                    # Show retry delay message
                    retry_delay = random.uniform(1.2 * delay, 1.5 * delay)
                    logger.info(f"<yellow>  ├──⏳ Retrying after {retry_delay:.1f}s...</yellow>")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Final failure with total timing
                    total_elapsed = time.time() - start_time
                    logger.error(f"<red>  ├──💥 Max retries exceeded after {total_elapsed:.1f}s. Last error: {error_body}</red>")
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(f"Max retries exceeded. Last error: {error_body}")
        
        return raw_text
    
    @classmethod
    def _is_valid_response_format(cls, response_text: str) -> bool:
        """Validate exactly one <tool_call> with name, id and parameters."""
        blocks = re.findall(r'<tool_call>([\s\S]*?)</tool_call>', response_text)
        if len(blocks) != 1:
            return False
        block = blocks[0]
        has_name = re.search(r'<tool_name>[\s\S]*?</tool_name>', block) is not None
        has_id = re.search(r'<tool_id>[\s\S]*?</tool_id>', block) is not None
        has_params = re.search(r'<parameters>[\s\S]*?</parameters>', block) is not None
        return has_name and has_id and has_params
    
    @classmethod
    def parse_response(cls, response_text: str) -> Tuple[Types.ToolCall, str]:
        """Parse response enforcing a single <tool_call> block with JSON parameters."""
        def _coerce(tool_name: str, param_name: str, value: str) -> Any:
            v = value.strip()
            if tool_name == "sequential_thinking":
                if param_name in ["totalThoughts", "thoughtNumber"]:
                    try:
                        return int(v)
                    except ValueError:
                        return v
                if param_name in ["nextThoughtNeeded", "isRevision", "needsMoreThoughts"]:
                    if v.lower() in ["true", "false"]:
                        return v.lower() == "true"
            if tool_name == "str_replace_editor":
                if param_name == "view_range":
                    try:
                        return ast.literal_eval(v)
                    except Exception:
                        return v
                if param_name == "insert_line":
                    try:
                        return int(v)
                    except ValueError:
                        return v
            return v

        def _parse_xml_blocks(text: str) -> Tuple[list[Types.ToolCall], str]:
            blocks = re.findall(r'<tool_call>([\s\S]*?)</tool_call>', text)
            tool_calls: list[Types.ToolCall] = []
            if len(blocks) == 0:
                return tool_calls, text
            if len(blocks) > 1:
                raise ValueError("Multiple <tool_call> blocks detected; only one is allowed")
            block = blocks[0]
            name_match = re.search(r'<tool_name>([\s\S]*?)</tool_name>', block)
            id_match = re.search(r'<tool_id>([\s\S]*?)</tool_id>', block)
            params_match = re.search(r'<parameters>([\s\S]*?)</parameters>', block)
            if not (name_match and id_match and params_match):
                raise ValueError("Incomplete <tool_call> block; expected tool_name, tool_id, parameters")
            tool_name = name_match.group(1).strip()
            tool_id = id_match.group(1).strip()
            params_raw = params_match.group(1).strip()
            # Coerce parameter values to appropriate types
            def _coerce_param(tool_name: str, param_name: str, value: Any) -> Any:
                if isinstance(value, str):
                    v = value.strip()
                    if tool_name == "sequential_thinking":
                        if param_name in ["totalThoughts", "thoughtNumber"]:
                            try:
                                return int(v)
                            except ValueError:
                                return v
                        if param_name in ["nextThoughtNeeded", "isRevision", "needsMoreThoughts"]:
                            if v.lower() in ["true", "false"]:
                                return v.lower() == "true"
                    if tool_name == "str_replace_editor":
                        if param_name == "view_range":
                            try:
                                import ast
                                return ast.literal_eval(v)
                            except Exception:
                                return v
                        if param_name == "insert_line":
                            try:
                                return int(v)
                            except ValueError:
                                return v
                return value
            
            try:
                params_obj = json.loads(params_raw)
            except Exception as e:
                logger.error(f"<red>  ├── ⏰ Invalid JSON params, trying to fix...</red>")
                params_obj = cls.fix_json_string(params_raw)
                if params_obj:
                    logger.error(f"<green>  ├── ✅ Fixed JSON params</green>")
                else:
                    raise ValueError(f"Parameters must be valid JSON: {params_raw}")
                
                              
            # Apply coercion to all parameters
            coerced_params = {}
            for param_name, param_value in params_obj.items():
                coerced_params[param_name] = _coerce_param(tool_name, param_name, param_value)
                            
            tool_calls.append(Types.ToolCall(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_input=coerced_params
            ))
            cleaned_text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text, flags=re.DOTALL).strip()
            return tool_calls, cleaned_text

        # XML parsing (required)
        xml_tool_calls, text_after_xml = _parse_xml_blocks(response_text)
        if not xml_tool_calls:
            raise ValueError("No tool call detected; XML tool call is required")
        if len(xml_tool_calls) > 1:
            raise ValueError("Multiple tool calls detected; only one tool call is supported per turn")
        return xml_tool_calls[0], text_after_xml
    
    @classmethod
    def fix_json_string(cls, json_string: str, attempt: int = 0 ) -> dict | None:
        messages = cast(list[Types.LLMMessage], [
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ])
        response=cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            return None
        
class IndentationHelper:
    """
    🧠 SMART INDENTATION HELPER CLASS
    
    This class provides all indentation-related functionality in one organized place.
    Think of it as your indentation toolkit!
    """
    
    # =============================================================================
    # 🔍 DETECTION METHODS
    # =============================================================================
    
    @classmethod
    def detect_line_indent(cls, line: str) -> Tuple[int, int]:
        """
        🔍 Detect the indentation of a single line.
        
        Args:
            line: The line to analyze
            
        Returns:
            Tuple of (num_tabs, num_spaces_after_tabs)
            
        Example:
            helper = IndentationHelper()
            tabs, spaces = helper.detect_line_indent("    print('Hello')")
            # Returns: (0, 4) - 0 tabs, 4 spaces
        """
        if not line:
            return (0, 0)

        # Count leading tabs
        num_tabs = 0
        for char in line:
            if char != "\t":
                break
            num_tabs += 1

        # Count spaces after tabs
        num_spaces = 0
        for char in line[num_tabs:]:
            if char != " ":
                break
            num_spaces += 1

        return (num_tabs, num_spaces)
    
    @classmethod
    def detect_indent_type(cls, code: str | None) -> Types.IndentType | None:
        """
        🔍 Detect the indentation type and size used in the entire code.
        
        Args:
            code: The code to analyze
            
        Returns:
            Types.IndentType object with type and size information
            
        Example:
            helper = IndentationHelper()
            code = "def hello():\\n    print('Hello')"
            indent_info = helper.detect_indent_type(code)
            # Returns: Types.IndentType(type="space", size=4)
        """
        if not code or not isinstance(code, str):
            return None

        lines = code.splitlines()
        space_diff_counts = defaultdict(int)
        tab_indents = 0
        space_indents = 0
        mixed_indent_in_one_line = False
        prev_indent_level = 0
        prev_indent_type = "space"

        for line in lines:
            if not line.strip():
                continue

            num_tabs, num_spaces = cls.detect_line_indent(line)
            if num_tabs == 0 and num_spaces == 0:
                continue

            if num_tabs > 0:
                if num_spaces > 0:
                    mixed_indent_in_one_line = True
                tab_indents += 1
                current_indent_type = "tab"
            else:
                space_indents += 1
                current_indent_type = "space"
                if prev_indent_type == "space":
                    diff = abs(num_spaces - prev_indent_level)
                    if diff > 1:
                        space_diff_counts[diff] += 1

            prev_indent_level = num_spaces if num_spaces > 0 else num_tabs
            prev_indent_type = current_indent_type

        if mixed_indent_in_one_line or (tab_indents > 0 and space_indents > 0):
            if tab_indents > space_indents:
                most_used = Types.IndentType.tab()
            else:
                if space_diff_counts:
                    most_common_diff = max(space_diff_counts.items(), key=lambda x: x[1])[0]
                    most_used = Types.IndentType.space(most_common_diff)
                else:
                    most_used = Types.IndentType.space()
            return Types.IndentType.mixed(most_used=most_used)
        elif tab_indents > 0:
            return Types.IndentType.tab()
        elif space_diff_counts:
            most_common_diff = max(space_diff_counts.items(), key=lambda x: x[1])[0]
            return Types.IndentType.space(most_common_diff)
        else:
            return None
    
    # =============================================================================
    # 🔧 NORMALIZATION METHODS
    # =============================================================================
    
    @classmethod
    def force_normalize_indent(cls, code: str) -> str:
        """
        🔧 Force normalize indentation to 4 spaces regardless of original style.
        
        Args:
            code: The code to normalize
            
        Returns:
            Code with 4-space indentation
            
        Example:
            helper = IndentationHelper()
            messy_code = "def hello():\n\tprint('Hello')"  # Mixed tabs/spaces
            clean_code = helper.force_normalize_indent(messy_code)
            # Result: "def hello():\n    print('Hello')"  # 4 spaces
        """
        lines = code.splitlines()
        normalized_lines = []
        for line in lines:
            if not line.strip():
                normalized_lines.append(line.strip())
                continue

            num_tabs, num_spaces = cls.detect_line_indent(line)
            normalized_lines.append(" " * (4 * num_tabs) + " " * num_spaces + line.lstrip())
        return "\n".join(normalized_lines)
    
    @classmethod
    def normalize_indent(cls, code: str | None, indent_type: Types.IndentType) -> str | None:
        """
        🔧 Normalize indentation to match the specified type.
        
        Args:
            code: The code to normalize
            indent_type: The target indentation type
            
        Returns:
            Code with normalized indentation
            
        Example:
            helper = IndentationHelper()
            code = "def hello():\n\tprint('Hello')"  # Tabs
            space_type = Types.IndentType.space(4)
            normalized = helper.normalize_indent(code, space_type)
            # Result: "def hello():\n    print('Hello')"  # 4 spaces
        """
        assert not indent_type.is_mixed, "Cannot normalize mixed indentation"
        if not code or not isinstance(code, str):
            return code

        lines = code.splitlines()
        normalized_lines = []

        for line in lines:
            if not line.strip():
                normalized_lines.append(line)
                continue

            num_tabs, num_spaces = cls.detect_line_indent(line)
            if num_tabs == 0 and num_spaces == 0:
                normalized_lines.append(line)
                continue

            indent_level = 0
            remainder = 0
            if indent_type.is_tab:
                indent_level = num_tabs
                remainder = num_spaces
                assert line[: num_tabs + num_spaces] == "\t" * num_tabs + " " * num_spaces
            else:
                total_spaces = num_spaces + (num_tabs * indent_type.size)
                indent_level = total_spaces // indent_type.size
                remainder = total_spaces % indent_type.size
                assert line[: num_tabs + num_spaces] == " " * (num_tabs + num_spaces)

            assert remainder < 2, f"Unexpected remainder: {remainder} for line: {line}"
            new_indent = " " * (4 * indent_level) + " " * remainder
            normalized_line = new_indent + line.lstrip()
            normalized_lines.append(normalized_line)

        return "\n".join(normalized_lines)
    
    # =============================================================================
    # 🎯 MATCHING METHODS
    # =============================================================================
    
    @classmethod
    def match_indent_by_first_line(cls, code: str | None, line: str) -> str | None:
        """
        🎯 Match the indentation of the first line in code to the given line.
        
        Args:
            code: The code to adjust
            line: The line to match indentation with
            
        Returns:
            Code with adjusted indentation
            
        Example:
            helper = IndentationHelper()
            code = "print('Hello')\nprint('World')"
            target_line = "    if True:"  # 4 spaces
            matched = helper.match_indent_by_first_line(code, target_line)
            # Result: "    print('Hello')\n    print('World')"  # 4 spaces
        """
        if not code or not isinstance(code, str):
            return code

        lines = code.splitlines()
        if not lines:
            return code

        # Get target and current indentation levels
        _, target_spaces = cls.detect_line_indent(line)
        _, current_spaces = cls.detect_line_indent(lines[0])

        # Calculate the indentation difference
        indent_diff = target_spaces - current_spaces

        modified_lines = []

        for line in lines:
            if not line.strip():  # Preserve empty lines
                modified_lines.append(line)
                continue

            _, num_spaces = cls.detect_line_indent(line)
            new_indent_size = max(0, num_spaces + indent_diff)
            modified_lines.append(" " * new_indent_size + line.lstrip())

        return "\n".join(modified_lines)
    
    @classmethod
    def match_indent(cls, code: str | None, code_to_match: str) -> str | None:
        """
        🎯 Match the indentation style of the target code.
        
        Args:
            code: The code to adjust
            code_to_match: The code whose indentation style to match
            
        Returns:
            Code with matched indentation
            
        Example:
            helper = IndentationHelper()
            new_code = "print('Hello')\nprint('World')"
            existing_code = "def main():\n    print('Test')"  # 4 spaces
            matched = helper.match_indent(new_code, existing_code)
            # Result: "    print('Hello')\n    print('World')"  # 4 spaces
        """
        if not code or not isinstance(code, str):
            return code

        indent_type = cls.detect_indent_type(code_to_match)
        if indent_type is not None and indent_type.is_mixed:
            indent_type = indent_type.most_used
        if indent_type is not None:
            return cls.apply_indent_type(code, indent_type)

        return code
    
    # =============================================================================
    # 🔄 CONVERSION METHODS
    # =============================================================================
    
    @classmethod
    def apply_indent_type(
        cls,
        code: str | None,
        indent_type: Types.IndentType,
        original_indent_type: Types.IndentType | None = None,
    ) -> str | None:
        """
        🔄 Apply the specified indentation type to code.
        
        Args:
            code: The code to convert
            indent_type: The target indentation type
            original_indent_type: The original indentation type (auto-detected if None)
            
        Returns:
            Code with applied indentation type
            
        Example:
            helper = IndentationHelper()
            code = "def hello():\n    print('Hello')"  # 4 spaces
            tab_type = Types.IndentType.tab()
            converted = helper.apply_indent_type(code, tab_type)
            # Result: "def hello():\n\tprint('Hello')"  # Tabs
        """
        assert not indent_type.is_mixed, "Cannot apply mixed indentation"
        if not code or not isinstance(code, str):
            return code

        if original_indent_type is None:
            original_indent_type = cls.detect_indent_type(code)
            if original_indent_type is None or original_indent_type.is_mixed:
                return code
            else:
                return cls.apply_indent_type(code, indent_type, original_indent_type)

        if original_indent_type == indent_type:
            return code

        lines = code.splitlines()
        modified_lines = []

        for line in lines:
            if not line.strip():  # Empty line
                modified_lines.append(line)
                continue

            num_tabs, num_spaces = cls.detect_line_indent(line)

            if original_indent_type.is_tab:
                indent_levels = num_tabs
                remainder = num_spaces
            else:
                assert num_tabs == 0, f"Unexpected tab in line: {line}"
                indent_levels = num_spaces // original_indent_type.size
                remainder = num_spaces % original_indent_type.size

            if indent_levels == 0:  # No indentation
                modified_lines.append(line)
                continue

            if indent_type.is_tab:
                new_indent = "\t" * indent_levels
            else:
                new_indent = " " * (indent_type.size * indent_levels)

            new_indent += " " * remainder

            modified_line = new_indent + line.lstrip()
            modified_lines.append(modified_line)

        return "\n".join(modified_lines)
    
    @classmethod
    def dedent(cls, text: str) -> str:
        """
        🧠 Smart dedent that removes common leading whitespace while preserving structure.
        
        Args:
            text: The text to dedent
            
        Returns:
            Text with common leading whitespace removed
            
        Example:
            indented_text = '''
                This is indented
                    This is more indented
                This is back to normal
            '''
            result = textwrap.dedent(indented_text)
            # Result: "This is indented\n    This is more indented\nThis is back to normal"
        """
        lines = text.split('\n')
        if not lines:
            return text
        
        # Remove empty lines from beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        if not lines:
            return ""
            
        # Find the minimum indentation (excluding empty lines)
        min_indent = float('inf')
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            min_indent = 0
            
        # Remove the minimum indentation from all lines and add default spaces
        dedented_lines = []
        for line in lines:
            if line.strip():  # Non-empty lines
                dedented = line[min_indent:]
                # Add default spaces to the beginning
                dedented_lines.append(dedented)
            else:  # Empty lines
                dedented_lines.append('')
        
        return '\n'.join(dedented_lines)

textwrap = IndentationHelper

class PromptManager():
    """Manager class for all LLM prompts used throughout the system."""
    
    @classmethod
    def get_system_prompt_with_tools(cls, tool_manager) -> str:
        """Get system prompt with dynamic tool documentation."""
        tool_docs = tool_manager.generate_tool_documentation()
        if tool_docs is None:
            tool_docs = ""
        return cls.SYSTEM_PROMPT.format(avaialable_tools=tool_docs)

    SYSTEM_PROMPT = textwrap.dedent("""
        <system_prompt>
            <role>You are an AI assistant helping a software engineer implement pull requests, and you have access to tools to interact with the engineer's codebase.</role>

            <environment>
                <operating_system>Linux</operating_system>
            </environment>

            <guidelines>
                <guideline>You are working in a codebase with multiple modules, classes, and functions. Be careful that changes you make in one module don't break other modules that depend on it.</guideline>
                <guideline>When designing changes, follow these specific best practices: 1) Keep data access separate from business logic, 2) Don't expose internal variables through public methods, 3) Use clear function and variable names, 4) Add error handling for edge cases.</guideline>
                <guideline>Choose the solution that is both simple and follows best practices. Example: Use a simple loop instead of complex recursion when both work.</guideline>
                <guideline>Use your bash tool to set up environment variables needed for testing. Example: export PYTHONPATH=/path/to/project before running tests.</guideline>
                <guideline>Run tests that cover the specific functionality you changed. Example: If you modify a function in user_service.py, run tests for user_service.py and any integration tests that use it.</guideline>
                
                <guideline>🧪 CRITICAL - TESTING WORKFLOW GUIDELINES 🧪</guideline>
                <guideline>USE the enhanced testing tools strategically during these steps:</guideline>
                <guideline>1. Step 4 (Baseline): Call test_validation with validation_type="baseline" BEFORE making changes</guideline>
                <guideline>2. Step 5 (Pre-Fix Test Generation): Call generate_tests with test_type="reproduction" to create test cases that reproduce the reported issue</guideline>
                <guideline>3. Step 7 (Fail-to-Pass): Call test_validation with validation_type="fail_to_pass" after implementing fix</guideline>
                <guideline>4. Step 8 (Pass-to-Pass): Call test_validation with validation_type="pass_to_pass" to check for regressions</guideline>
                <guideline>5. Step 9 (Post-Fix Test Generation): Call generate_tests with test_type="edge_cases" and test_type="error_cases" to create comprehensive test cases for your fix</guideline>
                <guideline>6. Step 10 (Comprehensive Validation): Call test_validation with validation_type="comprehensive" for final validation</guideline>
                <guideline>7. When encountering dependency errors: Call test_validation with validation_type="dependency_check"</guideline>
                <guideline>8. CRITICAL: Always create and run the generated test cases to validate your fix thoroughly</guideline>
                <guideline>9. TEST GENERATION WORKFLOW: Generate reproduction tests BEFORE fixing, then generate comprehensive tests AFTER fixing</guideline>
                <guideline>10. TEST EXECUTION: Always create the generated test files using str_replace_editor and run them with bash commands</guideline>
                
                <guideline>🚨 CRITICAL - ANTI-REPETITION RULES 🚨</guideline>
                <guideline>NEVER repeat the same tool call more than 2 times in a row.</guideline>
                <guideline>If a command fails with "No such file or directory", DO NOT repeat it. Instead, use 'ls' to check what files exist first.</guideline>
                <guideline>If you get the exact same error message twice in a row, STOP and try a different approach. Example: If file reading fails twice, try creating the file instead.</guideline>
                <guideline>Always check file existence before running files: use 'ls -la' or 'find . -name "*.py"' first.</guideline>
                <guideline>If you need a file that doesn't exist, CREATE it with str_replace_editor (this tool creates/modifies files by replacing text content) instead of trying to run it.</guideline>
                <guideline>Use sequential_thinking to break down complex problems into clear steps before taking action. Example: Step 1: Check what files exist, Step 2: Read the main file, Step 3: Identify what needs to be changed.</guideline>
            </guidelines>

            <restrictions>
                <title>CRITICAL - No Internet Access:</title>
                <restriction>You CANNOT access the internet or download packages.</restriction>
                <restriction>Do NOT try to install packages with pip, apt, or any package manager.</restriction>
                <restriction>Do NOT use wget, curl, or any other tool to download files from the internet.</restriction>
            </restrictions>

            <reminder>You're finished when: 1) All requested changes are implemented, 2) Tests pass without errors, 3) No new errors are introduced. Then use the complete tool to signal completion.</reminder>
        </system_prompt>
        """)

    TOOL_CALL_FORMAT_PROMPT = textwrap.dedent("""
        <tool_call_format_requirements>
            <important>YOU MUST RETURN EXACTLY ONE TOOL CALL USING THE SCHEMA BELOW.</important>
            <must>Only one <tool_call> per response</must>
            <do_not>Do not include any extra XML blocks, JSON outside <parameters>, or commentary inside <tool_call>.</do_not>
        
            <required_format>
                <tool_call>
                    <tool_name>tool_name</tool_name>
                    <tool_id>unique_id_string</tool_id>
                    <parameters>{"param_name": "param_value", "another": 123}</parameters>
                </tool_call>
            </required_format>
        
            <format_rules>
                <rule>Provide exactly one <tool_call> per response</rule>
                <rule>Include <tool_name>, <tool_id>, and JSON object in <parameters></rule>
                <rule>Parameters must be valid JSON (no XML inside)</rule>
                <rule>Do not include any additional <tool_call> blocks</rule>
            </format_rules>
        
            <correct_examples>
                <example>
                    <tool_call>
                        <tool_name>bash</tool_name>
                        <tool_id>1</tool_id>
                        <parameters>{"command": "ls -la"}</parameters>
                    </tool_call>
                </example>
                <example>
                    <tool_call>
                        <tool_name>str_replace_editor</tool_name>
                        <tool_id>2</tool_id>
                        <parameters>{"command": "view", "path": "app.py"}</parameters>
                    </tool_call>
                </example>
                <example>
                    <tool_call>
                        <tool_name>sequential_thinking</tool_name>
                        <tool_id>3</tool_id>
                        <parameters>{"thought": "Analyze", "thoughtNumber": 1, "totalThoughts": 5, "nextThoughtNeeded": true}</parameters>
                    </tool_call>
                </example>
            </correct_examples>
        
            <forbidden_formats>
                <forbidden>Tool: bash (missing XML)</forbidden>
                <forbidden>{"command": "ls"} (bare JSON without <tool_call>)</forbidden>
                <forbidden><tool_call /><tool_call>... (multiple tool_call blocks)</forbidden>
            </forbidden_formats>
        
            <reminder>XML format MANDATORY! Only ONE tool call per turn!</reminder>
        </tool_call_format_requirements>
    """)
    
    INSTRUCTION_PROMPT = textwrap.dedent("""
        <instruction_prompt>
            <context>
                I have uploaded all files of a python repository. Your current working directory is at the root of that repo. Consider the following problem statement:
            </context>

            <problem_statement>
            {problem_statement}
            </problem_statement>

            <task_description>
                Can you help me implement the necessary changes to the repository so that the requirements specified in the <problem_statement> are met?
                I've already taken care of all changes to any of the test files described in the <problem_statement>. This means you DON'T have to modify the testing logic or any of the tests in any way!
                Your task is to make the minimal changes to non-tests files in the current directory to ensure the <problem_statement> is satisfied.
            </task_description>

            <critical_anti_repetition_rules>
                <title>🚨 CRITICAL: ANTI-REPETITION RULES - FOLLOW THESE EXACTLY 🚨</title>
                <rule>NEVER repeat the same tool call more than 2 times in a row</rule>
                <rule>If a command fails with "No such file or directory", DO NOT repeat it</rule>
                <rule>Instead, first use 'ls' or 'find' to see what files actually exist</rule>
                <rule>If you need a Python file that doesn't exist, CREATE it with str_replace_editor first</rule>
                <rule>If you get the same error twice, STOP and try a completely different approach</rule>
                <rule>Always check file existence before attempting to run files</rule>
                <rule>Use sequential_thinking to plan your approach and avoid loops</rule>
                <warning>VIOLATING THESE RULES WILL CAUSE INFINITE LOOPS - FOLLOW THEM STRICTLY</warning>
            </critical_anti_repetition_rules>

            <resolution_steps>
                <step number="1">As a first step, it would be a good idea to explore the repo to familiarize yourself with its structure using bash commands like 'ls', 'find', and 'grep'.</step>
                
                <step number="2">Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error. If you encounter dependency issues, use the dependency_analysis tool to understand what's missing.</step>
                
                <step number="3">Use the "sequential_thinking" tool to plan your fix. Reflect on 5-7 different possible sources of the problem, distill those down to 1-2 most likely sources, and then add logs to validate your assumptions before moving onto implementing the actual code fix.</step>
                
                <step number="4">BEFORE making any changes, establish a baseline by calling test_validation with validation_type="baseline" to understand the current test state. This is crucial for later validation.</step>
                
                <step number="5">PRE-FIX TEST GENERATION: Use generate_tests with test_type="reproduction" to create test cases that specifically reproduce the reported issue. Create these test files using str_replace_editor and run them to confirm they fail as expected.</step>
                
                <step number="6">Edit the sourcecode of the repo to resolve the issue using str_replace_editor.</step>
                
                <step number="7">After implementing your fix, validate that the reported issue is resolved by calling test_validation with validation_type="fail_to_pass" and providing your reproduction script.</step>
                
                <step number="8">Ensure your fix doesn't break existing functionality by calling test_validation with validation_type="pass_to_pass" to run existing tests and check for regressions.</step>
                
                <step number="9">POST-FIX TEST GENERATION: Use generate_tests with test_type="edge_cases" and test_type="error_cases" to create comprehensive test cases that validate your fix handles various scenarios. Create these test files and run them to ensure comprehensive coverage.</step>
                
                <step number="10">Run the generated test cases from steps 5 and 9 to thoroughly validate your fix. Use bash commands to execute these tests and ensure they all pass.</step>
                
                <step number="11">For final validation, call test_validation with validation_type="comprehensive" to ensure both fail-to-pass and pass-to-pass validation succeed.</step>
                
                <step number="12">If you encounter any dependency or import errors during testing, use dependency_analysis to understand the issues and get specific recommendations.</step>
            </resolution_steps>

            <sequential_thinking_guide>
                <important>THIS SECTION IS CRITICAL. FOLLOW IT EXACTLY.</important>
                <description>GUIDE FOR HOW TO USE "sequential_thinking" TOOL:</description>
                <must>NEVER embed <tool_call> XML inside the "thought" parameter.</must>
                <must>ALWAYS make tool calls as a separate <tool_call> block only.</must>
                <warning>Violations will be rejected and you must try again.</warning>
                <tip>Your thinking should be thorough and so it's fine if it's very long. Set "totalThoughts" to at least 5, but setting it up to 25 is fine as well. You'll need more total thoughts when you are considering multiple possible solutions or root causes for an issue.</tip>
                <tip>Use this tool as much as you find necessary to improve the quality of your answers.</tip>
                <tip>You can run bash commands (like tests, a reproduction script, or 'grep'/'find' to find relevant context) in between thoughts.</tip>
                <tip>The "sequential_thinking" tool can help you break down complex problems, analyze issues step-by-step, and ensure a thorough approach to problem-solving.</tip>
                <tip>Don't hesitate to use it multiple times throughout your thought process to enhance the depth and accuracy of your solutions.</tip>
            </sequential_thinking_guide>

            <enhanced_tools_guide>
                <important>🧪 ENHANCED TESTING TOOLS - USE THESE STRATEGICALLY</important>
                <description>GUIDE FOR USING THE ENHANCED TESTING AND ERROR ANALYSIS TOOLS:</description>
                
                <tool_usage>
                    <tool_name>test_validation</tool_name>
                    <purpose>Comprehensive test validation with detailed error reporting</purpose>
                    <when_to_use>
                        - Step 4: validation_type="baseline" (BEFORE making changes)
                        - Step 7: validation_type="fail_to_pass" (after implementing fix)
                        - Step 8: validation_type="pass_to_pass" (check for regressions)
                        - Step 10: validation_type="comprehensive" (final validation)
                        - Anytime: validation_type="dependency_check" (when encountering import/dependency errors)
                    </when_to_use>
                    <key_benefits>
                        - Establishes baseline test state before changes
                        - Validates that reported issues are actually fixed
                        - Prevents regression bugs by checking existing functionality
                        - Provides detailed error analysis and recommendations
                    </key_benefits>
                </tool_usage>
                
                <tool_usage>
                    <tool_name>generate_tests</tool_name>
                    <purpose>Generate comprehensive test cases for various scenarios</purpose>
                    <when_to_use>
                        - Step 5 (PRE-FIX): test_type="reproduction" to create test cases that reproduce the reported issue
                        - Step 9 (POST-FIX): test_type="edge_cases" and test_type="error_cases" for comprehensive validation
                        - Additional scenarios: test_type="integration" for component interaction, test_type="regression" for preventing future issues
                    </when_to_use>
                    <key_benefits>
                        - Creates targeted test cases for specific scenarios
                        - Helps reproduce and validate issues before and after fixes
                        - Provides comprehensive test coverage with edge cases
                        - Generates actionable test code that can be executed
                        - CRITICAL: Always create and run the generated test files to validate your fix
                    </key_benefits>
                </tool_usage>
                
                <tool_usage>
                    <tool_name>dependency_analysis</tool_name>
                    <purpose>Analyze dependency and import issues with detailed recommendations</purpose>
                    <when_to_use>
                        - Step 2: When reproduction script fails due to missing dependencies
                        - Step 11: When encountering import errors during testing
                        - Anytime: When "No module named" or similar errors occur
                    </when_to_use>
                    <key_benefits>
                        - Identifies specific missing modules and dependencies
                        - Provides alternative solutions and workarounds
                        - Suggests static analysis approaches when dynamic execution fails
                        - Offers mock implementation strategies
                    </key_benefits>
                </tool_usage>
                
                <tool_usage>
                    <tool_name>bash (Enhanced)</tool_name>
                    <purpose>Execute commands with comprehensive error analysis</purpose>
                    <when_to_use>
                        - Throughout all steps for command execution
                        - Automatically provides detailed error analysis
                        - Offers specific recommendations for different error types
                    </when_to_use>
                    <key_benefits>
                        - Automatic error detection and categorization
                        - Specific guidance for import/dependency errors
                        - Test execution analysis with recommendations
                        - File/directory error handling
                        - Permission and timeout error guidance
                    </key_benefits>
                </tool_usage>
                
                <test_generation_workflow>
                    <important>🧪 TEST GENERATION WORKFLOW - FOLLOW THIS EXACTLY</important>
                    <description>Critical workflow for generating and using test cases effectively:</description>
                    
                    <phase_1_pre_fix>
                        <title>PHASE 1: PRE-FIX TEST GENERATION (Step 5)</title>
                        <actions>
                            1. Call generate_tests with test_type="reproduction"
                            2. Target the specific function/module mentioned in the problem statement
                            3. Include issue_description from the problem statement
                            4. Create the generated test files using str_replace_editor
                            5. Run the reproduction tests to confirm they FAIL (reproducing the issue)
                        </actions>
                        <purpose>Create test cases that reproduce the reported issue before fixing it</purpose>
                    </phase_1_pre_fix>
                    
                    <phase_2_post_fix>
                        <title>PHASE 2: POST-FIX TEST GENERATION (Step 9)</title>
                        <actions>
                            1. Call generate_tests with test_type="edge_cases" for boundary conditions
                            2. Call generate_tests with test_type="error_cases" for exception handling
                            3. Create the generated test files using str_replace_editor
                            4. Run all generated tests to ensure they PASS
                        </actions>
                        <purpose>Create comprehensive test cases to validate the fix handles various scenarios</purpose>
                    </phase_2_post_fix>
                    
                    <phase_3_validation>
                        <title>PHASE 3: COMPREHENSIVE VALIDATION (Step 10)</title>
                        <actions>
                            1. Run ALL generated test cases from both phases
                            2. Ensure reproduction tests now PASS (issue is fixed)
                            3. Ensure edge case and error case tests PASS (comprehensive coverage)
                            4. Use test_validation with validation_type="comprehensive" for final check
                        </actions>
                        <purpose>Thoroughly validate that the fix works and handles edge cases</purpose>
                    </phase_3_validation>
                    
                    <critical_notes>
                        <note>🚨 CRITICAL: Always create and execute the generated test files</note>
                        <note>🚨 CRITICAL: Reproduction tests should FAIL before fix, PASS after fix</note>
                        <note>🚨 CRITICAL: Edge case tests should PASS after fix to ensure comprehensive coverage</note>
                        <note>💡 TIP: Use the issue_description from problem statement in generate_tests</note>
                        <note>💡 TIP: Generate tests for the specific function/module mentioned in the problem</note>
                    </critical_notes>
                </test_generation_workflow>
            </enhanced_tools_guide>

            <tips>
                <tip>You must make changes in the project directory in order to ensure the requirements specified in the <problem_statement> are met. Leaving the directory unchanged is not a valid solution.</tip>
                <tip>Do NOT embed tool calls inside the 'thought' parameter passed to the "sequential_thinking" tool. 
                    The 'thought' parameter should contain only plain text reasoning, not XML tool calls.
                    
                    CORRECT usage:
                
                <tool_call>
                <tool_name>sequential_thinking</tool_name>
                <tool_id>3</tool_id>
                <parameters>{{"thought": "Analyze", "thoughtNumber": 1, "totalThoughts": 5, "nextThoughtNeeded": true}}</parameters>
                </tool_call>
                    INCORRECT: Do not put XML tool calls inside the thought parameter text.
                </tip>
                <tip>Respect the tool specifications. If a field is required, make sure to provide a value for it. For example "thoughtNumber" is required by the "sequential_thinking" tool.</tip>
                <tip>When you run "ls" with the bash tool, the "view" command with the "str_replace_editor" tool, or variants of those, you may see a symlink like "fileA -> /home/augment/docker/volumes/_data/fileA". You can safely ignore the symlink and just use "fileA" as the path when read, editing, or executing the file.</tip>
                <tip>When you need to find information about the codebase, use "grep" and "find" to search for relevant files and code with the bash tool</tip>
                <tip>Use your bash tool to set up any necessary environment variables, such as those needed to run tests.</tip>
            </tips>

            <restrictions>
                <title>CRITICAL - No Internet Access:</title>
                <restriction>You CANNOT access the internet or download packages.</restriction>
                <restriction>Do NOT try to install packages with pip, apt, or any package manager.</restriction>
                <restriction>Do NOT try to download files from the internet.</restriction>
                <restriction>Do NOT try to access external services or APIs.</restriction>
                <restriction>Work only with the files and tools available in the local repository.</restriction>
                <restriction>If you need a package that's not available, find an alternative solution or work around it.</restriction>
            <restriction>Do NOT create any documentation files, summary files, README files, or SOLUTION_SUMMARY.md files. Your job is to fix the code, not create documentation.</restriction>
            </restrictions>
    
            {available_tools}

            {tool_call_format}

            <anti_repetition_guidelines>
                <title>🚨 CRITICAL: ANTI-REPETITION GUIDELINES 🚨</title>
                <guideline>Keep track of what you've already tried to avoid repeating the same actions.</guideline>
                <guideline>If you've already attempted a solution, try a different approach.</guideline>
                <guideline>If you find yourself repeating the same tool calls, stop and think of a different strategy.</guideline>
                <guideline>Document your progress in your thinking to avoid loops.</guideline>
                <guideline>Use "sequential_thinking" to plan your approach before executing.</guideline>
                <guideline>NEVER repeat the same bash command more than 2 times in a row.</guideline>
                <guideline>If a file doesn't exist, CREATE it first instead of trying to run it repeatedly.</guideline>
                <warning>REPEATING THE SAME FAILED ACTION WILL CAUSE INFINITE LOOPS</warning>
            </anti_repetition_guidelines>
        </instruction_prompt>
    """)

class ToolManager:
    """Manager class for all tools in the agent system."""
    
    class LLMTool:
        """A tool that fits into the standard LLM tool-calling paradigm."""
        name: str
        description: str
        input_schema: Types.ToolInputSchema

        @property
        def should_stop(self) -> bool:
            """Whether the tool wants to stop the current agentic run."""
            return False

        @final
        def run(
            self,
            tool_input: dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> str:
            """Run the tool."""
            if dialog_messages:
                assert dialog_messages.is_user_turn()

            try:
                self._validate_tool_input(tool_input)
                result = self.run_impl(tool_input, dialog_messages)
                tool_output = result.tool_output
            except Exception as exc:
                if jsonschema and isinstance(exc, jsonschema.ValidationError):
                    tool_output = "Invalid tool input: " + exc.message
                else:
                    raise RuntimeError("Bad request: " + str(exc))

            return tool_output

        def get_tool_start_message(self, tool_input: Types.ToolInputSchema) -> str:
            """Return a user-friendly message to be shown to the model when the tool is called."""
            return f"Calling tool '{self.name}'"

        def run_impl(
            self,
            tool_input: dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> Types.ToolImplOutput:
            """Subclasses should implement this."""
            raise NotImplementedError()

        def get_tool_param(self) -> Types.ToolParam:
            return Types.ToolParam(
                name=self.name,
                description=self.description,
                input_schema=self.input_schema,
            )
        
        def get_tool_doc(self) -> dict[str, Any]:
            """Get the tool documentation as a dictionary."""
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.input_schema,
            }

        def _validate_tool_input(self, tool_input: dict[str, Any]):
            """Validates the tool input."""
            if jsonschema:
                jsonschema.validate(instance=tool_input, schema=self.input_schema)

    def __init__(self):
        """Initialize the tool manager."""
        self._tools: Dict[str, ToolManager.LLMTool] = {}
        self._register_default_tools()
        self.temp_files: List[str] = []

    def add_temp_file(self, file_path: str):
        """Add a temporary file to the tool manager."""
        self.temp_files.append(file_path)
    
    def _register_default_tools(self):
        """Register all default tools."""
        # Register Enhanced BashTool
        self.register_tool(ToolManager.EnhancedBashTool(tool_manager=self))
        
        # Register CompleteTool
        self.register_tool(ToolManager.CompleteTool())
        
        # Register SequentialThinkingTool
        self.register_tool(ToolManager.SequentialThinkingTool(tool_manager=self))
        
        # Register StrReplaceEditorTool
        self.register_tool(ToolManager.StrReplaceEditorTool(tool_manager=self))
        
        # Register Enhanced Test Validation Tool
        self.register_tool(ToolManager.TestValidationTool(tool_manager=self))
        
        # Register Dependency Analysis Tool
        self.register_tool(ToolManager.DependencyAnalysisTool(tool_manager=self))
        
        # Register Test Generation Tool
        self.register_tool(ToolManager.TestGenerationTool(tool_manager=self))
    
    def register_tool(self, tool: 'LLMTool'):
        """Register a tool with the manager."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional['LLMTool']:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tool_params(self) -> List[Types.ToolParam]:
        """Get all tool parameters."""
        return [tool.get_tool_param() for tool in self._tools.values()]
    
    def get_tool_docs(self, tool_choice: dict[str, str] | None = None) -> str:
        _docs: list[str] = []
        _docs.append("<available_tools>")
        _docs.append("You have access to the following tools. When you need to use a tool, respond with the exact format shown below. And also please make sure to follow the `<tool_call_format>` section for more information about how to use the tools.")
        for tool in self._tools.values():
            _tool = (
                f"<tool>\n"
                f"<name>{tool.name}</name>\n"
                f"<description>\n"
                f"{tool.description}\n"
                f"</description>\n"
                f"<parameters>\n"
                f"{json.dumps(tool.input_schema, indent=2, ensure_ascii=False)}\n"
                f"</parameters>\n"
                f"</tool>"
            )
            _docs.append(_tool)
        _docs.append("</available_tools>")
        return textwrap.dedent("\n".join(_docs))
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    class TestValidationTool(LLMTool):
        """Enhanced test validation tool for comprehensive testing."""
        
        name = "test_validation"
        
        def __init__(self, tool_manager: Optional[ToolManager] = None):
            """Initialize the TestValidationTool."""
            super().__init__()
            self.tool_manager = tool_manager
        description = textwrap.dedent("""
            Comprehensive test validation tool that the LLM should call during testing phases.
            
            The LLM should call this tool at these specific steps:
            
            1. BASELINE ESTABLISHMENT (Step 4):
               - Call with validation_type="baseline" to establish current test state
               - Run this BEFORE making any changes to understand what's currently working
            
            2. FAIL-TO-PASS VALIDATION (Step 7):
               - Call with validation_type="fail_to_pass" after implementing the fix
               - Verify that the reported issue is actually resolved
            
            3. PASS-TO-PASS VALIDATION (Step 8):
               - Call with validation_type="pass_to_pass" to ensure existing functionality still works
               - Run this to prevent regression bugs
            
            4. COMPREHENSIVE VALIDATION (Final Step):
               - Call with validation_type="comprehensive" for final validation
               - Combines both fail-to-pass and pass-to-pass checks
            
            5. DEPENDENCY CHECK (When needed):
               - Call with validation_type="dependency_check" when encountering import/dependency errors
               - Use this to understand what dependencies are available vs missing
            
            The LLM should use this tool strategically throughout the workflow, not automatically.
        """)
        
        input_schema = {
            "type": "object",
            "properties": {
                "validation_type": {
                    "type": "string",
                    "enum": ["baseline", "fail_to_pass", "pass_to_pass", "comprehensive", "dependency_check"],
                    "description": "Type of validation to perform. LLM should choose based on current workflow step."
                },
                "test_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific test files to run (optional, for pass_to_pass validation)"
                },
                "issue_reproduction": {
                    "type": "string",
                    "description": "Script or command to reproduce the reported issue (for fail_to_pass validation)"
                },
                "detailed_analysis": {
                    "type": "boolean",
                    "description": "Whether to provide detailed error analysis and recommendations (default: true)"
                }
            },
            "required": ["validation_type"]
        }
        
        def run_impl(self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages] = None) -> Types.ToolImplOutput:
            validation_type = tool_input["validation_type"]
            detailed_analysis = tool_input.get("detailed_analysis", True)
            
            try:
                if validation_type == "baseline":
                    return self._establish_baseline(detailed_analysis)
                elif validation_type == "fail_to_pass":
                    return self._validate_fail_to_pass(tool_input.get("issue_reproduction", ""), detailed_analysis)
                elif validation_type == "pass_to_pass":
                    return self._validate_pass_to_pass(tool_input.get("test_files", []), detailed_analysis)
                elif validation_type == "comprehensive":
                    return self._validate_comprehensive(tool_input, detailed_analysis)
                elif validation_type == "dependency_check":
                    return self._check_dependencies(detailed_analysis)
                else:
                    return Types.ToolImplOutput(
                        f"Unknown validation type: {validation_type}",
                        "Validation failed",
                        {"success": False, "error": f"Unknown validation type: {validation_type}"}
                    )
                    
            except Exception as e:
                error_msg = f"Test validation failed: {str(e)}"
                if detailed_analysis:
                    error_msg += f"\n\n{self._analyze_validation_error(str(e))}"
                return Types.ToolImplOutput(
                    error_msg,
                    "Test validation failed",
                    {"success": False, "error": str(e)}
                )
        
        def _establish_baseline(self, detailed_analysis: bool = True) -> Types.ToolImplOutput:
            """Establish baseline - LLM calls this at Step 4."""
            try:
                # Run all existing tests to establish baseline
                result = subprocess.run(
                    ["python", "-m", "pytest", "-v", "--tb=short"],
                    capture_output=True, text=True, timeout=120
                )
                
                baseline_data = self._parse_test_results(result.stdout, result.stderr, result.returncode)
                
                output = f"BASELINE ESTABLISHMENT COMPLETED:\n"
                output += f"✅ Tests found: {baseline_data.get('total_tests', baseline_data['passed'] + baseline_data['failed'])}\n"
                output += f"✅ Currently passing: {baseline_data['passed']}\n"
                output += f"❌ Currently failing: {baseline_data['failed']}\n"
                output += f"⚠️ Errors: {baseline_data['errors']}\n\n"
                
                if detailed_analysis and baseline_data['failed'] > 0:
                    output += self._analyze_baseline_failures(result.stdout, result.stderr)
                
                return Types.ToolImplOutput(
                    output,
                    "Baseline established successfully",
                    {"success": True, "baseline": baseline_data}
                )
            except Exception as e:
                error_msg = f"Failed to establish baseline: {str(e)}"
                if detailed_analysis:
                    error_msg += f"\n\n{self._analyze_baseline_error(str(e))}"
                return Types.ToolImplOutput(
                    error_msg,
                    "Baseline establishment failed",
                    {"success": False, "error": str(e)}
                )
        
        def _validate_fail_to_pass(self, issue_reproduction: str, detailed_analysis: bool = True) -> Types.ToolImplOutput:
            """Validate that the reported issue is fixed."""
            try:
                if issue_reproduction:
                    # Run the reproduction script
                    result = subprocess.run(
                        issue_reproduction.split(),
                        capture_output=True, text=True, timeout=60
                    )
                    
                    # If reproduction script runs without error, the issue is fixed
                    issue_fixed = result.returncode == 0 and "error" not in result.stderr.lower()
                    
                    output = f"FAIL-TO-PASS VALIDATION: {'PASSED' if issue_fixed else 'FAILED'}\n"
                    output += f"📋 Reproduction command: {issue_reproduction}\n"
                    output += f"📊 Return code: {result.returncode}\n"
                    
                    if detailed_analysis and not issue_fixed:
                        output += f"\n{self._analyze_f2p_failure(result.stdout, result.stderr)}"
                    
                    return Types.ToolImplOutput(
                        output,
                        f"Fail-to-Pass validation completed",
                        {"success": issue_fixed, "f2p_result": issue_fixed}
                    )
                else:
                    return Types.ToolImplOutput(
                        "No issue reproduction script provided",
                        "Fail-to-Pass validation skipped",
                        {"success": False, "error": "No reproduction script"}
                    )
                    
            except Exception as e:
                error_msg = f"Fail-to-Pass validation failed: {str(e)}"
                if detailed_analysis:
                    error_msg += f"\n\n{self._analyze_f2p_error(str(e))}"
                return Types.ToolImplOutput(
                    error_msg,
                    "Fail-to-Pass validation failed",
                    {"success": False, "error": str(e)}
                )
        
        def _validate_pass_to_pass(self, test_files: list[str], detailed_analysis: bool = True) -> Types.ToolImplOutput:
            """Validate that existing functionality still works."""
            try:
                # Run existing tests
                cmd = ["python", "-m", "pytest", "-v"]
                if test_files:
                    cmd.extend(test_files)
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                # Parse results
                test_data = self._parse_test_results(result.stdout, result.stderr, result.returncode)
                
                # Pass-to-Pass validation passes if no new failures introduced
                p2p_success = test_data['failed'] == 0
                
                output = f"PASS-TO-PASS VALIDATION: {'PASSED' if p2p_success else 'FAILED'}\n"
                output += f"📊 Results: {test_data['passed']} passed, {test_data['failed']} failed, {test_data['errors']} errors\n"
                output += f"📋 Command: {' '.join(cmd)}\n"
                
                if detailed_analysis and not p2p_success:
                    output += f"\n{self._analyze_p2p_failures(result.stdout, result.stderr)}"
                
                return Types.ToolImplOutput(
                    output,
                    f"Pass-to-Pass validation completed",
                    {"success": p2p_success, "p2p_result": p2p_success, "test_data": test_data}
                )
                
            except Exception as e:
                error_msg = f"Pass-to-Pass validation failed: {str(e)}"
                if detailed_analysis:
                    error_msg += f"\n\n{self._analyze_p2p_error(str(e))}"
                return Types.ToolImplOutput(
                    error_msg,
                    "Pass-to-Pass validation failed",
                    {"success": False, "error": str(e)}
                )
        
        def _validate_comprehensive(self, tool_input: dict, detailed_analysis: bool = True) -> Types.ToolImplOutput:
            """Perform comprehensive validation combining both F2P and P2P."""
            f2p_result = self._validate_fail_to_pass(tool_input.get("issue_reproduction", ""), detailed_analysis)
            p2p_result = self._validate_pass_to_pass(tool_input.get("test_files", []), detailed_analysis)
            
            f2p_success = f2p_result.auxiliary_data.get("success", False)
            p2p_success = p2p_result.auxiliary_data.get("success", False)
            
            overall_success = f2p_success and p2p_success
            
            output = f"COMPREHENSIVE VALIDATION: {'PASSED' if overall_success else 'FAILED'}\n"
            output += f"🔍 F2P (Fail-to-Pass): {'PASS' if f2p_success else 'FAIL'}\n"
            output += f"🔍 P2P (Pass-to-Pass): {'PASS' if p2p_success else 'FAIL'}\n\n"
            output += f"F2P Details:\n{f2p_result.tool_output}\n\n"
            output += f"P2P Details:\n{p2p_result.tool_output}"
            
            return Types.ToolImplOutput(
                output,
                f"Comprehensive validation completed",
                {
                    "success": overall_success,
                    "f2p_success": f2p_success,
                    "p2p_success": p2p_success,
                    "f2p_details": f2p_result.auxiliary_data,
                    "p2p_details": p2p_result.auxiliary_data
                }
            )
        
        def _parse_test_results(self, stdout: str, stderr: str, return_code: int) -> dict:
            """Parse pytest output to extract test results."""
            import re
            
            # Count passed tests
            passed_matches = re.findall(r'(\d+) passed', stdout)
            passed = int(passed_matches[0]) if passed_matches else 0
            
            # Count failed tests
            failed_matches = re.findall(r'(\d+) failed', stdout)
            failed = int(failed_matches[0]) if failed_matches else 0
            
            # Count errors
            error_matches = re.findall(r'(\d+) error', stdout)
            errors = int(error_matches[0]) if error_matches else 0
            
            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total_tests": passed + failed + errors,
                "return_code": return_code,
                "has_failures": "FAILED" in stdout or "ERROR" in stdout or return_code != 0
            }
        
        def _check_dependencies(self, detailed_analysis: bool = True) -> Types.ToolImplOutput:
            """Check dependencies - LLM calls this when encountering dependency issues."""
            try:
                # Check for common dependency issues
                checks = []
                
                # Check Python version
                python_version = subprocess.run(["python", "--version"], capture_output=True, text=True)
                checks.append(f"Python version: {python_version.stdout.strip()}")
                
                # Check for common missing modules
                common_modules = ["pytest", "unittest", "mock", "json", "os", "sys", "re"]
                missing_modules = []
                
                for module in common_modules:
                    try:
                        result = subprocess.run(["python", "-c", f"import {module}"], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode != 0:
                            missing_modules.append(module)
                    except:
                        missing_modules.append(module)
                
                if missing_modules:
                    checks.append(f"Missing modules: {', '.join(missing_modules)}")
                else:
                    checks.append("All common modules available")
                
                # Check for test files
                test_files = subprocess.run(["find", ".", "-name", "*test*.py"], 
                                          capture_output=True, text=True)
                if test_files.stdout:
                    checks.append(f"Found test files: {len(test_files.stdout.strip().split())}")
                else:
                    checks.append("No test files found")
                
                output = "DEPENDENCY ANALYSIS REPORT:\n" + "\n".join(f"- {check}" for check in checks)
                
                if detailed_analysis and missing_modules:
                    output += f"\n\n{self._analyze_missing_dependencies(missing_modules)}"
                
                return Types.ToolImplOutput(
                    output,
                    "Dependency check completed",
                    {"success": True, "missing_modules": missing_modules}
                )
                
            except Exception as e:
                return Types.ToolImplOutput(
                    f"Dependency check failed: {str(e)}",
                    "Dependency check failed",
                    {"success": False, "error": str(e)}
                )
        
        def _analyze_baseline_failures(self, stdout: str, stderr: str) -> str:
            """Analyze baseline test failures."""
            return textwrap.dedent("""
            📊 BASELINE FAILURE ANALYSIS:
            
            ⚠️ Some tests are currently failing before any changes.
            🔍 This is normal - focus on fixing the specific issue mentioned in the problem statement.
            💡 Only worry about new failures introduced by your changes.
            """)
        
        def _analyze_baseline_error(self, error_msg: str) -> str:
            """Analyze baseline establishment errors."""
            return textwrap.dedent(f"""
            🔍 BASELINE ERROR ANALYSIS:
            
            ❌ Error: {error_msg}
            🔧 RECOMMENDATIONS:
            
            1. Check if pytest is available: bash "python -c 'import pytest'"
            2. Try alternative test runner: bash "python -m unittest discover"
            3. Look for test files manually: bash "find . -name '*test*.py'"
            4. Focus on static analysis if tests can't run
            """)
        
        def _analyze_f2p_failure(self, stdout: str, stderr: str) -> str:
            """Analyze fail-to-pass validation failures."""
            return textwrap.dedent("""
            🔍 FAIL-TO-PASS ANALYSIS:
            
            ❌ The reproduction script still shows the issue exists.
            🔧 RECOMMENDATIONS:
            
            1. Review your fix implementation
            2. Check if the fix addresses the root cause
            3. Verify the fix is applied to the correct files
            4. Test the fix manually with different inputs
            """)
        
        def _analyze_f2p_error(self, error_msg: str) -> str:
            """Analyze fail-to-pass validation errors."""
            return textwrap.dedent(f"""
            🔍 FAIL-TO-PASS ERROR ANALYSIS:
            
            ❌ Error: {error_msg}
            🔧 RECOMMENDATIONS:
            
            1. Check if the reproduction script exists and is executable
            2. Verify the script doesn't have syntax errors
            3. Include the script in your fix if it's missing
            4. Use static analysis to verify your fix
            """)
        
        def _analyze_p2p_failures(self, stdout: str, stderr: str) -> str:
            """Analyze pass-to-pass validation failures."""
            return textwrap.dedent("""
            🔍 PASS-TO-PASS ANALYSIS:
            
            ❌ Some existing tests are now failing after your changes.
            🚨 This indicates your fix may have introduced regression bugs.
            🔧 RECOMMENDATIONS:
            
            1. Review which tests are failing and why
            2. Check if your changes affect unrelated functionality
            3. Make your fix more targeted to avoid side effects
            4. Consider alternative approaches that don't break existing tests
            """)
        
        def _analyze_p2p_error(self, error_msg: str) -> str:
            """Analyze pass-to-pass validation errors."""
            return textwrap.dedent(f"""
            🔍 PASS-TO-PASS ERROR ANALYSIS:
            
            ❌ Error: {error_msg}
            🔧 RECOMMENDATIONS:
            
            1. Check if pytest is available and working
            2. Try running tests individually to isolate issues
            3. Use static analysis to verify your fix doesn't break existing logic
            4. Focus on minimal changes that don't affect other components
            """)
        
        def _analyze_missing_dependencies(self, missing_modules: list[str]) -> str:
            """Analyze missing dependencies and provide recommendations."""
            return textwrap.dedent(f"""
            🔍 DEPENDENCY ANALYSIS:
            
            ❌ Missing modules: {', '.join(missing_modules)}
            🔧 RECOMMENDATIONS:
            
            1. FOR MISSING PYTEST/UNITTEST:
               - Use static analysis instead of dynamic testing
               - Create simple test scripts using basic Python
               - Focus on logic verification rather than full test execution
            
            2. FOR MISSING STANDARD LIBRARIES:
               - Check if alternative standard library modules can be used
               - Implement minimal versions of missing functionality
               - Use mock objects for external dependencies
            
            3. WORKAROUND STRATEGIES:
               - Create minimal reproduction scripts
               - Use static code analysis to verify fixes
               - Test core logic with simple Python scripts
            """)
        
        def _analyze_validation_error(self, error_msg: str) -> str:
            """Analyze general validation errors."""
            return textwrap.dedent(f"""
            🔍 VALIDATION ERROR ANALYSIS:
            
            ❌ Error: {error_msg}
            🔧 RECOMMENDATIONS:
            
            1. Check if the command or script exists and is executable
            2. Verify file permissions and paths
            3. Use alternative validation approaches if dynamic execution fails
            4. Focus on static analysis and manual verification
            """)

    class DependencyAnalysisTool(LLMTool):
        """Tool for LLM to call when encountering dependency issues."""
        
        name = "dependency_analysis"
        
        def __init__(self, tool_manager: Optional[ToolManager] = None):
            """Initialize the DependencyAnalysisTool."""
            super().__init__()
            self.tool_manager = tool_manager
        description = textwrap.dedent("""
            Dependency analysis tool that the LLM should call when encountering import errors or missing modules.
            
            The LLM should call this tool when:
            - Import errors occur during test execution
            - "No module named" errors appear
            - Test commands fail due to missing dependencies
            - Need to understand what's available in the environment
            
            This tool provides detailed analysis and recommendations for handling dependency issues.
        """)
        
        input_schema = {
            "type": "object",
            "properties": {
                "error_output": {
                    "type": "string",
                    "description": "The error output that contains dependency issues"
                },
                "command_attempted": {
                    "type": "string", 
                    "description": "The command that failed due to dependency issues"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["import_error", "test_failure", "command_failure", "general"],
                    "description": "Type of dependency analysis needed"
                }
            },
            "required": ["error_output", "command_attempted"]
        }
        
        def run_impl(self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages] = None) -> Types.ToolImplOutput:
            error_output = tool_input["error_output"]
            command_attempted = tool_input["command_attempted"]
            analysis_type = tool_input.get("analysis_type", "general")
            
            try:
                analysis = self._analyze_dependency_issues(error_output, command_attempted, analysis_type)
                recommendations = self._generate_recommendations(error_output, command_attempted, analysis_type)
                
                output = f"DEPENDENCY ANALYSIS REPORT:\n\n"
                output += f"🔍 Command attempted: {command_attempted}\n"
                output += f"📋 Analysis type: {analysis_type}\n\n"
                output += f"📊 ANALYSIS:\n{analysis}\n\n"
                output += f"🔧 RECOMMENDATIONS:\n{recommendations}\n\n"
                output += f"💡 NEXT STEPS FOR LLM:\n{self._get_llm_next_steps(error_output, analysis_type)}"
                
                return Types.ToolImplOutput(
                    output,
                    "Dependency analysis completed",
                    {"success": True, "analysis": analysis, "recommendations": recommendations}
                )
            except Exception as e:
                return Types.ToolImplOutput(
                    f"Dependency analysis failed: {str(e)}",
                    "Dependency analysis failed",
                    {"success": False, "error": str(e)}
                )
        
        def _analyze_dependency_issues(self, error_output: str, command_attempted: str, analysis_type: str) -> str:
            """Analyze dependency issues from error output."""
            analysis = []
            error_lower = error_output.lower()
            
            # Check for import errors
            if any(keyword in error_lower for keyword in ["no module named", "import error", "module not found"]):
                import_match = re.search(r"No module named '([^']+)'", error_output)
                module_name = import_match.group(1) if import_match else "unknown module"
                analysis.append(f"🔍 Import error detected: Missing module '{module_name}'")
            
            # Check for test framework issues
            if any(keyword in error_lower for keyword in ["pytest", "unittest"]) and "error" in error_lower:
                analysis.append("🧪 Test framework issues detected")
            
            # Check for permission issues
            if "permission denied" in error_lower:
                analysis.append("🔒 Permission issues detected")
            
            # Check for file/directory issues
            if any(keyword in error_lower for keyword in ["no such file", "file not found", "directory not found"]):
                analysis.append("📁 File/directory issues detected")
            
            return "\n".join(analysis) if analysis else "No specific dependency issues identified"
        
        def _generate_recommendations(self, error_output: str, command_attempted: str, analysis_type: str) -> str:
            """Generate recommendations based on error analysis."""
            recommendations = []
            error_lower = error_output.lower()
            
            if "no module named" in error_lower:
                recommendations.extend([
                    "1. Check if the module is part of standard Python library",
                    "2. Look for alternative implementations in the codebase",
                    "3. Create mock implementations for testing",
                    "4. Use static analysis instead of dynamic execution"
                ])
            elif "pytest" in error_lower or "unittest" in error_lower:
                recommendations.extend([
                    "1. Try alternative test runners: 'python -m unittest discover'",
                    "2. Use static analysis to verify your fix",
                    "3. Create simple test scripts using basic Python",
                    "4. Focus on logic verification rather than full test execution"
                ])
            elif "permission denied" in error_lower:
                recommendations.extend([
                    "1. Check file permissions with 'ls -la'",
                    "2. Use read-only operations where possible",
                    "3. Create files instead of modifying protected ones",
                    "4. Focus on code analysis without execution"
                ])
            else:
                recommendations.extend([
                    "1. Analyze the error message for specific issues",
                    "2. Try alternative approaches or commands",
                    "3. Use static analysis when dynamic execution fails",
                    "4. Focus on core logic verification"
                ])
            
            return "\n".join(recommendations)
        
        def _get_llm_next_steps(self, error_output: str, analysis_type: str) -> str:
            """Provide specific next steps for the LLM."""
            return textwrap.dedent("""
            1. READ the complete error analysis above
            2. FOLLOW the specific recommendations provided
            3. ADAPT your approach based on the dependency issues found
            4. USE static analysis and mock implementations when dynamic execution fails
            5. FOCUS on core logic verification rather than full test execution
            6. CREATE minimal reproduction scripts when full test suites can't run
            """)

    class TestGenerationTool(LLMTool):
        """Tool for generating comprehensive test cases."""
        
        name = "generate_tests"
        description = textwrap.dedent("""
            Generate comprehensive test cases for the problem statement.
            Creates test scenarios that cover edge cases, error conditions, and integration scenarios.
            
            This tool helps create test cases when the existing test suite is insufficient
            or when you need to create reproduction scripts for the reported issue.
        """)
        
        input_schema = {
            "type": "object",
            "properties": {
                "test_type": {
                    "type": "string",
                    "enum": ["edge_cases", "error_cases", "integration", "regression", "reproduction", "all"],
                    "description": "Type of tests to generate"
                },
                "target_function": {
                    "type": "string",
                    "description": "Specific function or module to test"
                },
                "issue_description": {
                    "type": "string",
                    "description": "Description of the issue to create reproduction tests for"
                }
            },
            "required": ["test_type"]
        }
        
        def __init__(self, tool_manager: Optional[ToolManager] = None):
            """Initialize the TestGenerationTool."""
            super().__init__()
            self.tool_manager = tool_manager
        
        def run_impl(self, tool_input: dict[str, Any], dialog_messages: Optional[DialogMessages] = None) -> Types.ToolImplOutput:
            test_type = tool_input["test_type"]
            target_function = tool_input.get("target_function", "")
            issue_description = tool_input.get("issue_description", "")
            
            try:
                # Generate test cases based on type
                test_cases = self._generate_test_cases(test_type, target_function, issue_description)
                
                output = f"TEST GENERATION COMPLETED:\n"
                output += f"📊 Generated {len(test_cases)} test cases for {test_type}\n"
                output += f"🎯 Target: {target_function if target_function else 'General'}\n\n"
                
                for i, test_case in enumerate(test_cases, 1):
                    output += f"Test Case {i}:\n"
                    output += f"Type: {test_case['type']}\n"
                    output += f"Description: {test_case['description']}\n"
                    output += f"Code: {test_case['code']}\n\n"
                
                output += "💡 RECOMMENDATIONS:\n"
                output += "1. Review the generated test cases above\n"
                output += "2. Create the test files using str_replace_editor\n"
                output += "3. Run the tests using bash commands\n"
                output += "4. Use test_validation tool to verify results"
                
                return Types.ToolImplOutput(
                    output,
                    "Test generation completed",
                    {"success": True, "test_cases": test_cases}
                )
            except Exception as e:
                return Types.ToolImplOutput(
                    f"Test generation failed: {str(e)}",
                    "Test generation failed",
                    {"success": False, "error": str(e)}
                )
        
        def _generate_test_cases(self, test_type: str, target_function: str, issue_description: str) -> list[dict]:
            """Generate test cases based on the specified type."""
            test_cases = []
            
            if test_type == "reproduction" or test_type == "all":
                # Generate reproduction test cases
                test_cases.append({
                    "type": "reproduction",
                    "target": target_function,
                    "description": f"Reproduction test for the reported issue",
                    "code": f"def test_reproduction_{target_function.replace('.', '_')}():\n    # Test case to reproduce the reported issue\n    # {issue_description}\n    pass"
                })
            
            if test_type == "edge_cases" or test_type == "all":
                test_cases.append({
                    "type": "edge_cases",
                    "target": target_function,
                    "description": f"Edge case test for {target_function}",
                    "code": f"def test_edge_cases_{target_function.replace('.', '_')}():\n    # Test edge cases like empty inputs, boundary values, etc.\n    pass"
                })
            
            if test_type == "error_cases" or test_type == "all":
                test_cases.append({
                    "type": "error_cases",
                    "target": target_function,
                    "description": f"Error handling test for {target_function}",
                    "code": f"def test_error_handling_{target_function.replace('.', '_')}():\n    # Test error conditions and exception handling\n    pass"
                })
            
            if test_type == "integration" or test_type == "all":
                test_cases.append({
                    "type": "integration",
                    "target": target_function,
                    "description": f"Integration test for {target_function}",
                    "code": f"def test_integration_{target_function.replace('.', '_')}():\n    # Test integration with other components\n    pass"
                })
            
            if test_type == "regression" or test_type == "all":
                test_cases.append({
                    "type": "regression",
                    "target": target_function,
                    "description": f"Regression test for {target_function}",
                    "code": f"def test_regression_{target_function.replace('.', '_')}():\n    # Test to prevent regression of fixed issues\n    pass"
                })
            
            return test_cases

    class EnhancedBashTool(LLMTool):
        """Enhanced BashTool with comprehensive error reporting for test cases."""
        
        name = "bash"
        description = textwrap.dedent("""
            Enhanced bash tool with comprehensive error reporting for test cases.
            Run commands in a bash shell with detailed error analysis.
            
            * When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
            * You don't have access to the internet via this tool.
            * You do have access to a mirror of common linux and python packages via apt and pip.
            * State is persistent across command calls and discussions with the user.
            * Provides detailed error analysis and recommendations when commands fail.
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
            },
            "required": ["command"],
        }

        def __init__(
            self,
            tool_manager: Optional[ToolManager] = None,
            workspace_root: Optional[Path] = None,
            timeout: int = 60,
            additional_banned_command_strs: Optional[List[str]] = None,
        ):
            """Initialize the Enhanced BashTool."""
            super().__init__()
            self.tool_manager = tool_manager
            self.workspace_root = workspace_root
            self.timeout = timeout

            self.banned_command_strs = [
                "git init",
                "git commit",
                "git add",
            ]
            if additional_banned_command_strs is not None:
                self.banned_command_strs.extend(additional_banned_command_strs)

            self.workspace_cwd = str(workspace_root) if workspace_root else None
        
        def run_command_simple(self, cmd: str, cwd: str | None = None, timeout: int = 60) -> str:
            """Run a command with enhanced error reporting."""
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=timeout,
                    env=None
                )
                
                # Enhanced error analysis
                error_analysis = self._analyze_command_result(cmd, result)
                
                # Combine stdout and stderr with error analysis
                output = result.stdout
                if result.stderr:
                    output += f"\n{result.stderr}"
                
                # Add error analysis to output
                if error_analysis:
                    output += f"\n\n{error_analysis}"
                    
                # Clean ANSI escape codes
                ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                clean_output = ansi_escape.sub("", output)
                
                return clean_output.strip()
                
            except subprocess.TimeoutExpired:
                return self._format_timeout_error(cmd, timeout)
            except Exception as e:
                return self._format_execution_error(cmd, str(e))
        
        def _analyze_command_result(self, cmd: str, result: subprocess.CompletedProcess) -> str:
            """Analyze command result and provide detailed error reporting."""
            analysis = []
            
            # Check for common dependency issues
            if result.returncode != 0:
                stderr_lower = result.stderr.lower() if result.stderr else ""
                stdout_lower = result.stdout.lower() if result.stdout else ""
                combined_output = f"{stderr_lower} {stdout_lower}"
                
                # Dependency-related errors
                if any(keyword in combined_output for keyword in [
                    "module not found", "no module named", "import error", "import failed"
                ]):
                    analysis.append(self._analyze_import_error(cmd, result))
                
                # Test framework errors
                elif any(keyword in combined_output for keyword in [
                    "pytest", "unittest", "test", "failed", "error"
                ]):
                    analysis.append(self._analyze_test_error(cmd, result))
                
                # Permission errors
                elif "permission denied" in combined_output:
                    analysis.append(self._analyze_permission_error(cmd, result))
                
                # File/directory errors
                elif any(keyword in combined_output for keyword in [
                    "no such file", "file not found", "directory not found"
                ]):
                    analysis.append(self._analyze_file_error(cmd, result))
                
                # Generic error analysis
                else:
                    analysis.append(self._analyze_generic_error(cmd, result))
            
            return "\n".join(analysis) if analysis else ""
        
        def _analyze_import_error(self, cmd: str, result: subprocess.CompletedProcess) -> str:
            """Analyze import/dependency errors and provide specific guidance."""
            stderr = result.stderr if result.stderr else ""
            stdout = result.stdout if result.stdout else ""
            
            # Extract missing module name
            import_match = re.search(r"No module named '([^']+)'", stderr + stdout)
            module_name = import_match.group(1) if import_match else "unknown module"
            
            return textwrap.dedent(f"""
            🔍 DEPENDENCY ERROR ANALYSIS:
            
            ❌ Missing dependency: {module_name}
            📋 Command: {cmd}
            🔧 RECOMMENDED ACTIONS:
            
            1. CHECK IF MODULE IS PART OF STANDARD LIBRARY:
               - Use: bash "python -c 'import {module_name}'" to verify
               - If it fails, the module is not available in standard Python
            
            2. FIND ALTERNATIVE IMPLEMENTATION:
               - Search for existing implementations in the codebase: bash "grep -r '{module_name}' ."
               - Look for similar functionality in standard library modules
               - Consider implementing a minimal version using standard libraries only
            
            3. MOCK THE DEPENDENCY:
               - Create a mock implementation for testing purposes
               - Use unittest.mock to replace the missing module
            
            4. STATIC ANALYSIS APPROACH:
               - Analyze the code without executing it
               - Use AST parsing to understand the code structure
               - Focus on the logic rather than execution
            
            💡 Remember: You cannot install packages, so find alternative solutions or implement missing functionality yourself.
            """)
        
        def _analyze_test_error(self, cmd: str, result: subprocess.CompletedProcess) -> str:
            """Analyze test execution errors and provide specific guidance."""
            stderr = result.stderr if result.stderr else ""
            stdout = result.stdout if result.stdout else ""
            
            # Count test results
            passed_match = re.search(r'(\d+) passed', stdout)
            failed_match = re.search(r'(\d+) failed', stdout)
            error_match = re.search(r'(\d+) error', stdout)
            
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            errors = int(error_match.group(1)) if error_match else 0
            
            analysis = textwrap.dedent(f"""
            🧪 TEST EXECUTION ANALYSIS:
            
            📊 Results: {passed} passed, {failed} failed, {errors} errors
            📋 Command: {cmd}
            """)
            
            if failed > 0 or errors > 0:
                analysis += textwrap.dedent(f"""
            🔧 RECOMMENDED ACTIONS:
            
            1. ANALYZE FAILING TESTS:
               - Look for specific test failure messages in the output above
               - Identify which tests are failing and why
            
            2. CHECK TEST DEPENDENCIES:
               - Some tests might require external dependencies
               - Consider mocking external dependencies or using static analysis
            
            3. FOCUS ON CORE FUNCTIONALITY:
               - If tests can't run due to dependency issues, focus on the core logic
               - Use static code analysis to verify your changes are correct
            
            4. CREATE MINIMAL REPRODUCTION:
               - Create a simple script that reproduces the core issue
               - Test your fix with this minimal script instead of full test suite
            """)
            
            return analysis
        
        def _analyze_permission_error(self, cmd: str, result: subprocess.CompletedProcess) -> str:
            """Analyze permission errors."""
            return textwrap.dedent(f"""
            🔒 PERMISSION ERROR ANALYSIS:
            
            ❌ Permission denied for command: {cmd}
            🔧 RECOMMENDED ACTIONS:
            
            1. CHECK FILE PERMISSIONS:
               - Use: bash "ls -la" to check file permissions
               - Look for files that might need execute permissions
            
            2. USE ALTERNATIVE APPROACHES:
               - Try running Python files directly: bash "python filename.py"
               - Use read-only operations where possible
            
            3. FOCUS ON CODE ANALYSIS:
               - If execution is blocked, use static analysis
               - Read and analyze code without executing it
            """)
        
        def _analyze_file_error(self, cmd: str, result: subprocess.CompletedProcess) -> str:
            """Analyze file/directory errors."""
            return textwrap.dedent(f"""
            📁 FILE/DIRECTORY ERROR ANALYSIS:
            
            ❌ File or directory not found
            📋 Command: {cmd}
            🔧 RECOMMENDED ACTIONS:
            
            1. VERIFY FILE EXISTENCE:
               - Use: bash "ls -la" to see what files exist
               - Use: bash "find . -name '*.py'" to find Python files
            
            2. CHECK WORKING DIRECTORY:
               - Use: bash "pwd" to see current directory
               - Use: bash "ls" to see directory contents
            
            3. CREATE MISSING FILES:
               - If a file is missing and needed, create it
               - Use str_replace_editor to create new files
            """)
        
        def _analyze_generic_error(self, cmd: str, result: subprocess.CompletedProcess) -> str:
            """Analyze generic errors."""
            return textwrap.dedent(f"""
            ⚠️ GENERIC ERROR ANALYSIS:
            
            ❌ Command failed with return code: {result.returncode}
            📋 Command: {cmd}
            🔧 RECOMMENDED ACTIONS:
            
            1. ANALYZE ERROR MESSAGE:
               - Look at the stderr output above for specific error details
               - Identify the root cause of the failure
            
            2. TRY ALTERNATIVE APPROACHES:
               - Use different commands to achieve the same goal
               - Break down complex commands into simpler steps
            
            3. USE STATIC ANALYSIS:
               - If dynamic execution fails, use static code analysis
               - Focus on understanding the code structure and logic
            """)
        
        def _format_timeout_error(self, cmd: str, timeout: int) -> str:
            """Format timeout errors with specific guidance."""
            return textwrap.dedent(f"""
            ⏰ TIMEOUT ERROR:
            
            ❌ Command timed out after {timeout} seconds
            📋 Command: {cmd}
            🔧 RECOMMENDED ACTIONS:
            
            1. COMMAND MIGHT BE LONG-RUNNING:
               - The command might be waiting for input or running indefinitely
               - Try a simpler version of the command
            
            2. USE QUICKER ALTERNATIVES:
               - Use faster commands for the same purpose
               - Break down complex operations into smaller steps
            
            3. FOCUS ON ESSENTIAL OPERATIONS:
               - Skip non-essential commands that might timeout
               - Use static analysis instead of dynamic execution when possible
            """)
        
        def _format_execution_error(self, cmd: str, error_msg: str) -> str:
            """Format execution errors with specific guidance."""
            return textwrap.dedent(f"""
            💥 EXECUTION ERROR:
            
            ❌ Error executing command: {error_msg}
            📋 Command: {cmd}
            🔧 RECOMMENDED ACTIONS:
            
            1. CHECK COMMAND SYNTAX:
               - Verify the command syntax is correct
               - Try simpler variations of the command
            
            2. USE ALTERNATIVE TOOLS:
               - Try different tools or approaches
               - Use static analysis instead of dynamic execution
            
            3. FOCUS ON CORE FUNCTIONALITY:
               - Skip problematic commands and focus on the main task
               - Use available tools to achieve the same goal
            """)

        def run_impl(
            self,
            tool_input: Dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> Types.ToolImplOutput:
            """Execute a bash command with enhanced error reporting."""
            command = tool_input["command"]
            aux_data = {
                "original_command": command,
                "executed_command": command,
            }

            # Check for banned commands
            for banned_str in self.banned_command_strs:
                if banned_str in command:
                    return Types.ToolImplOutput(
                        f"Command not executed due to banned string in command: {banned_str} found in {command}.",
                        f"Command not executed due to banned string in command: {banned_str} found in {command}.",
                        aux_data | {"success": False, "reason": "Banned command"},
                    )

            # Execute the command using enhanced method
            try:
                result = self.run_command_simple(command, cwd=self.workspace_cwd, timeout=self.timeout)
                
                # Check if command failed
                if "Error executing command:" in result or "Command timed out" in result:
                    return Types.ToolImplOutput(
                        result,
                        f"Failed to execute command '{command}'",
                        aux_data | {"success": False, "error": result},
                    )
                else:
                    # Command executed successfully
                    return Types.ToolImplOutput(
                        result,
                        f"Command '{command}' executed.",
                        aux_data | {"success": True},
                    )

            except Exception as e:
                return Types.ToolImplOutput(
                    f"Error executing command: {str(e)}",
                    f"Failed to execute command '{command}'",
                    aux_data | {
                        "success": False,
                        "error": str(e),
                    },
                )

        def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
            """Get a message to display when the tool starts."""
            return f"Executing bash command: {tool_input['command']}"

    class BashTool(LLMTool):
        """A tool for executing bash commands.

        This tool allows the agent to run shell commands and get their output.
        Commands are executed in a controlled environment with appropriate safeguards.
        Command filters can be added to transform commands before execution.
        """

        name = "bash"
        description = textwrap.dedent("""
            Run commands in a bash shell
            * When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.
            * You don't have access to the internet via this tool.
            * You do have access to a mirror of common linux and python packages via apt and pip.
            * State is persistent across command calls and discussions with the user.
            * To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
            * Please avoid commands that may produce a very large amount of output.
            * Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
            },
            "required": ["command"],
        }

        def __init__(
            self,
            tool_manager: Optional[ToolManager] = None,
            workspace_root: Optional[Path] = None,
            timeout: int = 60,
            additional_banned_command_strs: Optional[List[str]] = None,
        ):
            """Initialize the BashTool.

            Args:
                workspace_root: Root directory of the workspace
                timeout: Timeout for command execution in seconds
                additional_banned_command_strs: Additional commands to ban
            """
            super().__init__()
            self.tool_manager = tool_manager
            self.workspace_root = workspace_root
            self.timeout = timeout

            self.banned_command_strs = [
                "git init",
                "git commit",
                "git add",
            ]
            if additional_banned_command_strs is not None:
                self.banned_command_strs.extend(additional_banned_command_strs)

            # No persistent shell - using simple subprocess approach
            self.workspace_cwd = str(workspace_root) if workspace_root else None
            
        def run_command_simple(self, cmd: str, cwd: str | None = None, timeout: int = 60) -> str:
            """Run a command using subprocess.run() - simple alternative to pexpect."""
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=timeout,
                    env=None  # Use current environment
                )
                
                # Combine stdout and stderr
                output = result.stdout
                if result.stderr:
                    output += f"\n{result.stderr}"
                    
                # Clean ANSI escape codes
                ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                clean_output = ansi_escape.sub("", output)
                
                return clean_output.strip()
                
            except subprocess.TimeoutExpired:
                return f"Command timed out after {timeout} seconds"
            except Exception as e:
                return f"Error executing command: {str(e)}"


        # Command filter methods removed - simplified BashTool

        def run_impl(
            self,
            tool_input: Dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> Types.ToolImplOutput:
            """Execute a bash command and return its output.

            Args:
                tool_input: Dictionary containing the command to execute
                dialog_messages: Optional dialog messages for context

            Returns:
                ToolImplOutput containing the command output
            """
            command = tool_input["command"]
            aux_data = {
                "original_command": command,
                "executed_command": command,
            }

            # Show the command in the confirmation prompt
            display_command = command

            for banned_str in self.banned_command_strs:
                if banned_str in command:
                    return Types.ToolImplOutput(
                        f"Command not executed due to banned string in command: {banned_str} found in {command}.",
                        f"Command not executed due to banned string in command: {banned_str} found in {command}.",
                        aux_data | {"success": False, "reason": "Banned command"},
                    )

            # Execute the command using simple subprocess
            try:
                result = self.run_command_simple(command, cwd=self.workspace_cwd, timeout=self.timeout)
                
                # Check if command failed
                if "Error executing command:" in result or "Command timed out" in result:
                    return Types.ToolImplOutput(
                        result,
                        f"Failed to execute command '{command}'",
                        aux_data | {"success": False, "error": result},
                    )
                else:
                    # Command executed successfully
                    return Types.ToolImplOutput(
                        result,
                        f"Command '{command}' executed.",
                        aux_data | {"success": True},
                    )

            except Exception as e:
                    return Types.ToolImplOutput(
                    f"Error executing command: {str(e)}",
                    f"Failed to execute command '{command}'",
                    aux_data | {
                        "success": False,
                        "error": str(e),
                    },
                    )

        def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
            """Get a message to display when the tool starts.

            Args:
                tool_input: Dictionary containing the command to execute

            Returns:
                A message describing the command being executed
            """
            return f"Executing bash command: {tool_input['command']}"

    class CompleteTool(LLMTool):
        name = "complete"
        """The model should call this tool when it is done with the task."""

        description = "Call this tool when you are done with the task, and supply your answer or summary."
        input_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the question, or final summary of actions taken to accomplish the task.",
                },
            },
            "required": ["answer"],
        }

        def __init__(
            self, 
            tool_manager: Optional[ToolManager] = None,
        ):
            super().__init__()
            self.answer: str = ""
            self.tool_manager = tool_manager

        @property
        def should_stop(self):
            return self.answer != ""

        def reset(self):
            self.answer = ""

        def run_impl(
            self,
            tool_input: dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> Types.ToolImplOutput:
            """Handle completion."""
            self.answer = tool_input["answer"]
            return Types.ToolImplOutput(
                tool_output=f"Task completed: {self.answer}",
                tool_result_message=f"Task completed: {self.answer}",
            )

        def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
            return ""

    class SequentialThinkingTool(LLMTool):
        """A tool for sequential thinking that helps break down complex problems."""

        name = "sequential_thinking"
        description = textwrap.dedent("""
            A detailed tool for dynamic and reflective problem-solving through thoughts.
            This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
            Each thought can build on, question, or revise previous insights as understanding deepens.

            When to use this tool:
            - Breaking down complex problems into steps
            - Planning and design with room for revision
            - Analysis that might need course correction
            - Problems where the full scope might not be clear initially
            - Problems that require a multi-step solution
            - Tasks that need to maintain context over multiple steps
            - Situations where irrelevant information needs to be filtered out

            Key features:
            - You can adjust total_thoughts up or down as you progress
            - You can question or revise previous thoughts
            - You can add more thoughts even after reaching what seemed like the end
            - You can express uncertainty and explore alternative approaches
            - Not every thought needs to build linearly - you can branch or backtrack
            - Generates a solution hypothesis
            - Verifies the hypothesis based on the Chain of Thought steps
            - Repeats the process until satisfied
            - Provides a correct answer

            Parameters explained:
            - thought: Your current thinking step, which can include:
                * Analysis of the problem
                * Breaking down complex parts
                * Questioning assumptions
                * Exploring alternatives
                * Revising previous thoughts
                * Synthesizing information
                * Making connections
                * Identifying patterns
                * Formulating hypotheses
                * Planning next steps
            - thoughtNumber: The current thought number (1-based)
            - totalThoughts: How many thoughts you plan to have (can be adjusted)
            - isRevision: Set to true if this thought revises a previous one
            - revisesThought: If isRevision is true, specify which thought number this revises
            - branchFromThought: If this thought branches from a previous one, specify the thought number
            - branchId: A unique identifier for this branch (e.g., "alternative_approach")
            - needsMoreThoughts: Set to true if you need more thoughts after this one
            - nextThoughtNeeded: Set to true if you want to continue with another thought
        """)

        input_schema = {
            "type": "object",
            "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your current thinking step or analysis.",
                },
                "thoughtNumber": {
                    "type": "integer",
                        "description": "The current thought number (1-based).",
                },
                "totalThoughts": {
                    "type": "integer",
                        "description": "How many thoughts you plan to have (can be adjusted).",
                },
                "isRevision": {
                    "type": "boolean",
                        "description": "Set to true if this thought revises a previous one.",
                },
                "revisesThought": {
                    "type": "integer",
                        "description": "If isRevision is true, specify which thought number this revises.",
                },
                "branchFromThought": {
                    "type": "integer",
                        "description": "If this thought branches from a previous one, specify the thought number.",
                },
                    "branchId": {
                        "type": "string",
                        "description": "A unique identifier for this branch (e.g., 'alternative_approach').",
                    },
                "needsMoreThoughts": {
                    "type": "boolean",
                        "description": "Set to true if you need more thoughts after this one.",
                },
                    "nextThoughtNeeded": {
                        "type": "boolean",
                        "description": "Set to true if you want to continue with another thought.",
            },
                },
                "required": ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"],
        }

        def __init__(
            self,
            tool_manager: Optional[ToolManager] = None,
        ):
            """Initialize the sequential thinking tool."""
            super().__init__()
            self.tool_manager = tool_manager
            self.thought_history: List[Types.ThoughtData] = []
            self.branches: Dict[str, List[Types.ThoughtData]] = {}

        def _validate_thought_data(self, input_data: Dict[str, Any]) -> Types.ThoughtData:
            """Validate the thought data input.

            Args:
                input_data: The input data to validate

            Returns:
                Validated ThoughtData

            Raises:
                ValueError: If the input data is invalid
            """
            if not input_data.get("thought") or not isinstance(input_data["thought"], str):
                raise ValueError("Invalid thought: must be a string")

            if not input_data.get("thoughtNumber") or not isinstance(
                input_data["thoughtNumber"], int
            ):
                raise ValueError("Invalid thoughtNumber: must be a number")

            if not input_data.get("totalThoughts") or not isinstance(
                input_data["totalThoughts"], int
            ):
                raise ValueError("Invalid totalThoughts: must be a number")

            if not isinstance(input_data.get("nextThoughtNeeded"), bool):
                raise ValueError("Invalid nextThoughtNeeded: must be a boolean")

            return {
                "thought": input_data["thought"],
                "thoughtNumber": input_data["thoughtNumber"],
                "totalThoughts": input_data["totalThoughts"],
                "nextThoughtNeeded": input_data["nextThoughtNeeded"],
                "isRevision": input_data.get("isRevision"),
                "revisesThought": input_data.get("revisesThought"),
                "branchFromThought": input_data.get("branchFromThought"),
                "branchId": input_data.get("branchId"),
                "needsMoreThoughts": input_data.get("needsMoreThoughts"),
            }

        def _format_thought(self, thought_data: Types.ThoughtData) -> str:
            """Format a thought for display.

            Args:
                thought_data: The thought data to format

            Returns:
                Formatted thought string
            """
            thought_number = thought_data["thoughtNumber"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            total_thoughts = thought_data["totalThoughts"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            thought = thought_data["thought"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            is_revision = thought_data.get("isRevision", False)
            revises_thought = thought_data.get("revisesThought")
            branch_from_thought = thought_data.get("branchFromThought")
            branch_id = thought_data.get("branchId")

            prefix = ""
            context = ""

            if is_revision:
                prefix = "🔄 Revision"
                context = f" (revising thought {revises_thought})"
            elif branch_from_thought:
                prefix = "🌿 Branch"
                context = f" (from thought {branch_from_thought}, ID: {branch_id})"
            else:
                prefix = "💭 Thought"
                context = ""

            header = f"{prefix} {thought_number}/{total_thoughts}{context}"
            border_length = max(len(header), len(thought)) + 4
            border = "─" * border_length

            return textwrap.dedent(f"""
            ┌{border}┐
            │ {header.ljust(border_length)} │
            ├{border}┤
            │ {thought.ljust(border_length)} │
            └{border}┘
            """)
            # return textwrap.dedent(f"""
            #     ┌{border}┐
            #     │ {header.ljust(border_length)} │
            #     ├{border}┤
            #     │ {thought.ljust(border_length)} │
            #     └{border}┘
            # """)

        def run_impl(
            self,
            tool_input: Dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> Types.ToolImplOutput:
            """Run the sequential thinking tool.

            Args:
                tool_input: The input data for the tool
                dialog_messages: Optional dialog messages

            Returns:
                Tool output with the result
            """
            try:
                validated_input = self._validate_thought_data(tool_input)

                # Adjust total thoughts if needed
                if validated_input["thoughtNumber"] > validated_input["totalThoughts"]:  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    validated_input["totalThoughts"] = validated_input["thoughtNumber"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

                # Add to thought history
                self.thought_history.append(validated_input)

                # Handle branches
                branch_id = validated_input.get("branchId")
                if validated_input.get("branchFromThought") and branch_id:
                    if branch_id not in self.branches:
                        self.branches[branch_id] = []  # pyright: ignore[reportArgumentType]
                    self.branches[branch_id].append(validated_input)  # pyright: ignore[reportArgumentType]

                # Format and log the thought
                formatted_thought = self._format_thought(validated_input)
                logger.info(formatted_thought)

                # Prepare response
                response = {
                    "thoughtNumber": validated_input["thoughtNumber"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    "totalThoughts": validated_input["totalThoughts"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    "nextThoughtNeeded": validated_input["nextThoughtNeeded"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    "branches": list(self.branches.keys()),
                    "thoughtHistoryLength": len(self.thought_history),
                }

                return Types.ToolImplOutput(
                    tool_output=json.dumps(response, indent=2),
                    tool_result_message=f"Processed thought {validated_input['thoughtNumber']}/{validated_input['totalThoughts']}",  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    auxiliary_data={"thought_data": validated_input},
                )
            except Exception as e:
                error_response = {"error": str(e), "status": "failed"}
                return Types.ToolImplOutput(
                    tool_output=json.dumps(error_response, indent=2),
                    tool_result_message=f"Error processing thought: {str(e)}",
                    auxiliary_data={"error": str(e)},
                )

        def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
            """Return a user-friendly message when the tool is called.

            Args:
                tool_input: The input data for the tool

            Returns:
                A user-friendly message
            """
            thought_number = tool_input.get("thoughtNumber", "?")
            total_thoughts = tool_input.get("totalThoughts", "?")
            return f"Processing sequential thought {thought_number}/{total_thoughts}"

    class StrReplaceEditorTool(LLMTool):
        name = "str_replace_editor"

        description = textwrap.dedent("""
            Custom editing tool for viewing, creating and editing files
            * State is persistent across command calls and discussions with the user
            * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
            * The `create` command cannot be used if the specified `path` already exists as a file
            * If a `command` generates a long output, it will be truncated and marked with `<response clipped>` 
            * The `undo_edit` command will revert the last edit made to the file at `path`

            Notes for using the `str_replace` command:
            * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
            * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
            * The `new_str` parameter should contain the edited lines that should replace the `old_str`
            """
        )
        
        input_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                },
                "file_text": {
                    "description": "Required parameter of `create` command, with the content of the file to be created.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                    "type": "integer",
                },
                "new_str": {
                    "description": "Required parameter of `str_replace` command containing the new string. Required parameter of `insert` command containing the string to insert.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    "type": "string",
                },
                "path": {
                    "description": "Path to file or directory.",
                    "type": "string",
                },
                "view_range": {
                    "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                    "items": {"type": "integer"},
                    "type": "array",
                },
            },
            "required": ["command", "path"],
        }

        # Track file edit history for undo operations
        _file_history = defaultdict(list)

        def __init__(
            self,
            tool_manager: Optional[ToolManager] = None,
            ignore_indentation_for_str_replace: bool = False,
            expand_tabs: bool = False,
        ):
            super().__init__()
            self.tool_manager = tool_manager
            self.ignore_indentation_for_str_replace = ignore_indentation_for_str_replace
            self.expand_tabs = expand_tabs
            self._file_history = defaultdict(list)

        def run_impl(
            self,
            tool_input: dict[str, Any],
            dialog_messages: Optional[DialogMessages] = None,
        ) -> Types.ExtendedToolImplOutput:
            command = tool_input["command"]
            path = tool_input["path"]
            file_text = tool_input.get("file_text")
            view_range = tool_input.get("view_range")
            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str")
            insert_line = tool_input.get("insert_line")

            try:
                # Use current working directory for all path operations
                _ws_path = Path(path).resolve()
                if not isinstance(_ws_path, Path):
                    _ws_path = Path(_ws_path)
                self.validate_path(command, _ws_path)

                # Security check - ensure path is within current working directory
                current_dir = Path.cwd()
                if not Utils.is_path_in_directory(current_dir, _ws_path):
                    return Types.ExtendedToolImplOutput(
                        f"Path {_ws_path} is outside the current working directory: {current_dir}. You can only access files within the current directory.",
                        f"Path {_ws_path} is outside the current working directory: {current_dir}. You can only access files within the current directory.",
                        {"success": False},
                    )
                if command == "view":
                    return self.view(_ws_path, view_range)
                elif command == "create":
                    if file_text is None:
                        raise Types.ToolError(
                            "Parameter `file_text` is required for command: create"
                        )
                    self.write_file(_ws_path, file_text)
                    self._file_history[_ws_path].append(file_text)
                    
                    # Track created file for git patch exclusion
                    if self.tool_manager:
                        self.tool_manager.add_temp_file(str(_ws_path))
                    
                    return Types.ExtendedToolImplOutput(
                        f"File created successfully at: {_ws_path}",
                        f"File created successfully at: {_ws_path}",
                        {"success": True},
                    )
                elif command == "str_replace":
                    if old_str is None:
                        raise Types.ToolError(
                            "Parameter `old_str` is required for command: str_replace"
                        )
                    if self.ignore_indentation_for_str_replace:
                        return self._str_replace_ignore_indent(_ws_path, old_str, new_str)
                    else:
                        try:
                            return self.str_replace(_ws_path, old_str, new_str)
                        except PermissionError:
                            return Types.ExtendedToolImplOutput(
                                f"The file {path} could not be edited due to lack of permission. Try changing the file permissions.",
                                f"The file {path} could not be edited due to lack of permission. Try changing the file permissions.",
                                {"success": True},
                            )
                elif command == "insert":
                    if insert_line is None:
                        raise Types.ToolError(
                            "Parameter `insert_line` is required for command: insert"
                        )
                    if new_str is None:
                        raise Types.ToolError(
                            "Parameter `new_str` is required for command: insert"
                        )
                    return self.insert(_ws_path, insert_line, new_str)
                elif command == "undo_edit":
                    return self.undo_edit(_ws_path)
                raise Types.ToolError(
                    f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: {', '.join(get_args(Types.Command))}"
                )
            except Exception as e:
                return Types.ExtendedToolImplOutput(
                    str(e),
                    str(e),
                    {"success": False},
                )

        def validate_path(self, command: str, path: Path):
            """
            Check that the path/command combination is valid.
            """
            # Check if path exists
            if not path.exists() and command != "create":
                raise Types.ToolError(
                    f"The path {path} does not exist. Please provide a valid path."
                )
            if path.exists() and command == "create":
                content = self.read_file(path)
                if content.strip():
                    raise Types.ToolError(
                        f"File already exists and is not empty at: {path}. Cannot overwrite non empty files using command `create`."
                    )
            # Check if the path points to a directory
            if path.is_dir():
                if command != "view":
                    raise Types.ToolError(
                        f"The path {path} is a directory and only the `view` command can be used on directories"
                    )

        def view(
            self, path: Path, 
            view_range: list[int] | None = None
        ) -> Types.ExtendedToolImplOutput:
            if path.is_dir():
                if view_range:
                    raise Types.ToolError(
                        "The `view_range` parameter is not allowed when `path` points to a directory."
                    )

                result = subprocess.run(
                    ["find", str(path), "-maxdepth", "2", "-not", "-path", "*/.*"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    output = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{result.stdout}\n"
                else:
                    output = f"stderr: {result.stderr}\nstdout: {result.stdout}\n"
                return Types.ExtendedToolImplOutput(
                    output, "Listed directory contents", {"success": result.returncode == 0}
                )

            file_content = self.read_file(path)
            file_lines = file_content.split("\n")  # Split into lines early for total line count
            init_line = 1
            if view_range:
                if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                    raise Types.ToolError(
                        "Invalid `view_range`. It should be a list of two integers."
                    )
                n_lines_file = len(file_lines)
                init_line, final_line = view_range
                if init_line < 1 or init_line > n_lines_file:
                    raise Types.ToolError(
                        f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                    )
                if final_line > n_lines_file:
                    raise Types.ToolError(
                        f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                    )
                if final_line != -1 and final_line < init_line:
                    raise Types.ToolError(
                        f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                    )

                if final_line == -1:
                    file_content = "\n".join(file_lines[init_line - 1 :])
                else:
                    file_content = "\n".join(file_lines[init_line - 1 : final_line])

            output = self._make_output(
                file_content=file_content,
                file_descriptor=str(path),
                total_lines=len(
                    file_lines
                ),  # Use total lines in file, not just the viewed range
                init_line=init_line,
            )
            return Types.ExtendedToolImplOutput(
                output, "Displayed file content", {"success": True}
            )

        def _str_replace_ignore_indent(self, path: Path, old_str: str, new_str: str | None):
            """Replace old_str with new_str in content, ignoring indentation.

            Finds matches in stripped version of text and uses those line numbers
            to perform replacements in original indented version.
            """
            if new_str is None:
                new_str = ""

            content = self.read_file(path)
            if self.expand_tabs:
                content = content.expandtabs()
                old_str = old_str.expandtabs()
                new_str = new_str.expandtabs()

            new_str = textwrap.match_indent(new_str, content)
            assert new_str is not None, "new_str should not be None after match_indent"

            # Split into lines for processing
            content_lines = content.splitlines()
            stripped_content_lines = [line.strip() for line in content.splitlines()]
            stripped_old_str_lines = [line.strip() for line in old_str.splitlines()]

            # Find all potential starting line matches
            matches = []
            for i in range(len(stripped_content_lines) - len(stripped_old_str_lines) + 1):
                is_match = True
                for j, pattern_line in enumerate(stripped_old_str_lines):
                    if j == len(stripped_old_str_lines) - 1:
                        if stripped_content_lines[i + j].startswith(pattern_line):
                            # it's a match but last line in old_str is not the full line
                            # we need to append the rest of the line to new_str
                            new_str += stripped_content_lines[i + j][len(pattern_line) :]
                        else:
                            is_match = False
                            break
                    elif stripped_content_lines[i + j] != pattern_line:
                        is_match = False
                        break
                if is_match:
                    matches.append(i)

            if not matches:
                raise Types.ToolError(
                    f"No replacement was performed, old_str \n ```\n{old_str}\n```\n did not appear in {path}."
                )
            if len(matches) > 1:
                # Add 1 to convert to 1-based line numbers for error message
                match_lines = [idx + 1 for idx in matches]
                raise Types.ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str \n ```\n{old_str}\n```\n starting at lines {match_lines}. Please ensure it is unique"
                )

            # Get the matching range in the original content
            match_start = matches[0]
            match_end = match_start + len(stripped_old_str_lines)

            # Get the original indented lines
            original_matched_lines = content_lines[match_start:match_end]

            indented_new_str = textwrap.match_indent_by_first_line(
                new_str, original_matched_lines[0]
            )
            assert indented_new_str is not None, "indented_new_str should not be None"

            # Create new content by replacing the matched lines
            new_content = [
                *content_lines[:match_start],
                *indented_new_str.splitlines(),
                *content_lines[match_end:],
            ]
            new_content_str = "\n".join(new_content)

            self._file_history[path].append(content)  # Save old content for undo
            path.write_text(new_content_str)

            # Create a snippet of the edited section
            start_line = max(0, match_start - SNIPPET_LINES)
            end_line = match_start + SNIPPET_LINES + new_str.count("\n")
            snippet = "\n".join(new_content[start_line : end_line + 1])

            # Prepare the success message
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                file_content=snippet,
                file_descriptor=f"a snippet of {path}",
                total_lines=len(new_content),
                init_line=start_line + 1,
            )
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

            return Types.ExtendedToolImplOutput(
                success_msg,
                f"The file {path} has been edited.",
                {"success": True},
            )

        def str_replace(
            self, path: Path, old_str: str, new_str: str | None
        ) -> Types.ExtendedToolImplOutput:
            if new_str is None:
                new_str = ""

            content = self.read_file(path)
            if self.expand_tabs:
                content = content.expandtabs()
                old_str = old_str.expandtabs()
                new_str = new_str.expandtabs()

            if not old_str.strip():
                if content.strip():
                    raise Types.ToolError(
                        f"No replacement was performed, old_str is empty which is only allowed when the file is empty. The file {path} is not empty."
                    )
                else:
                    # replace the whole file with new_str
                    new_content = new_str
                    self._file_history[path].append(content)  # Save old content for undo
                    path.write_text(new_content)
                    # Prepare the success message
                    success_msg = f"The file {path} has been edited. "
                    success_msg += self._make_output(
                        file_content=new_content,
                        file_descriptor=f"{path}",
                        total_lines=len(new_content.split("\n")),
                    )
                    success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

                    return Types.ExtendedToolImplOutput(
                        success_msg,
                        f"The file {path} has been edited.",
                        {"success": True},
                    )

            occurrences = content.count(old_str)

            if occurrences == 0:
                raise Types.ToolError(
                    f"No replacement was performed, old_str \n ```\n{old_str}\n```\n did not appear verbatim in {path}."
                )
            elif occurrences > 1:
                file_content_lines = content.split("\n")
                lines = [
                    idx + 1
                    for idx, line in enumerate(file_content_lines)
                    if old_str in line
                ]
                raise Types.ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str \n ```\n{old_str}\n```\n in lines {lines}. Please ensure it is unique"
                )

            new_content = content.replace(old_str, new_str)
            self._file_history[path].append(content)  # Save old content for undo
            path.write_text(new_content)

            # Create a snippet of the edited section
            replacement_line = content.split(old_str)[0].count("\n")
            start_line = max(0, replacement_line - SNIPPET_LINES)
            end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
            snippet = "\n".join(new_content.split("\n")[start_line : end_line + 1])

            # Prepare the success message
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                file_content=snippet,
                file_descriptor=f"a snippet of {path}",
                total_lines=len(new_content.split("\n")),
                init_line=start_line + 1,
            )
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

            return Types.ExtendedToolImplOutput(
                success_msg,
                f"The file {path} has been edited.",
                {"success": True},
            )

        def insert(
            self, path: Path, insert_line: int, new_str: str
        ) -> Types.ExtendedToolImplOutput:
            """Implement the insert command, which inserts new_str at the specified line in the file content."""
            file_text = self.read_file(path)
            if self.expand_tabs:
                file_text = file_text.expandtabs()
                new_str = new_str.expandtabs()
            file_text_lines = file_text.split("\n")
            n_lines_file = len(file_text_lines)

            if insert_line < 0 or insert_line > n_lines_file:
                raise Types.ToolError(
                    f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
                )

            new_str_lines = new_str.split("\n")
            new_file_text_lines = (
                file_text_lines[:insert_line]
                + new_str_lines
                + file_text_lines[insert_line:]
            )
            snippet_lines = (
                file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
                + new_str_lines
                + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
            )

            new_file_text = "\n".join(new_file_text_lines)
            snippet = "\n".join(snippet_lines)

            self.write_file(path, new_file_text)
            self._file_history[path].append(file_text)

            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                file_content=snippet,
                file_descriptor="a snippet of the edited file",
                total_lines=len(new_file_text_lines),
                init_line=max(1, insert_line - SNIPPET_LINES + 1),
            )
            success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

            return Types.ExtendedToolImplOutput(
                success_msg,
                "Insert successful",
                {"success": True},
            )

        def undo_edit(self, path: Path) -> Types.ExtendedToolImplOutput:
            """Implement the undo_edit command."""
            if not self._file_history[path]:
                raise Types.ToolError(f"No edit history found for {path}.")

            old_text = self._file_history[path].pop()
            self.write_file(path, old_text)

            formatted_file = self._make_output(
                file_content=old_text,
                file_descriptor=str(path),
                total_lines=len(old_text.split("\n")),
            )
            output = f"Last edit to {path} undone successfully.\n{formatted_file}"

            return Types.ExtendedToolImplOutput(
                output,
                "Undo successful",
                {"success": True},
            )

        def read_file(self, path: Path):
            """Read the content of a file from a given path; raise a ToolError if an error occurs."""
            try:
                return path.read_text()
            except Exception as e:
                raise Types.ToolError(f"Ran into {e} while trying to read {path}") from None

        def write_file(self, path: Path, file: str):
            """Write the content of a file to a given path; raise a ToolError if an error occurs."""
            try:
                path.write_text(file)
            except Exception as e:
                raise Types.ToolError(f"Ran into {e} while trying to write to {path}") from None

        def _make_output(
            self,
            file_content: str,
            file_descriptor: str,
            total_lines: int,
            init_line: int = 1,
        ):
            """Generate output for the CLI based on the content of a file."""
            file_content = Utils.maybe_truncate(file_content)
            if self.expand_tabs:
                file_content = file_content.expandtabs()
            file_content = "\n".join(
                [
                    f"{i + init_line:6}\t{line}"
                    for i, line in enumerate(file_content.split("\n"))
                ]
            )
            return (
                f"Here's the result of running `cat -n` on {file_descriptor}:\n"
                + file_content
                + "\n"
                + f"Total lines in file: {total_lines}\n"
            )

        def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
            """Get a message to display when the tool starts."""
            command = tool_input.get("command", "unknown")
            path = tool_input.get("path", "unknown")
            return f"Executing {command} on {path}"

class Agent:
    """Main agent class for problem solving."""

    def __init__(
        self,
        tool_manager: ToolManager,
        max_output_tokens_per_turn: int = 8192,
        max_turns: int = 200,
        use_prompt_budgeting: bool = True,
    ):
        """Initialize the agent."""
        # Create LLM client internally
        self.max_output_tokens = max_output_tokens_per_turn
        self.max_turns = max_turns
        self.dialog = DialogMessages(
            use_prompt_budgeting=use_prompt_budgeting,
        )

        # Create ToolManager to manage all tools
        self.tool_manager = tool_manager
        self.turn_counter = 0

    def run_agent(
        self,
        problem_statement: str,
    ) -> str:
        """Start a new agent run.
        
        Args:
            instruction: The instruction to the agent.
            
        Returns:
            The result of the agent run.
        """
        # # Reset complete tool through tool_manager
        # complete_tool = self.tool_manager.get_tool('complete')
        # if complete_tool:
        #     complete_tool.reset()
        # self.dialog.clear()

        # user_input_delimiter = "-" * 45 + " USER INPUT " + "-" * 45 + "\n" + str(instruction)
        # logger.info(f"\n{user_input_delimiter}\n")

        # Add instruction to dialog before getting mode
        
        # Get tool parameters for available tools
        tool_docs = self.tool_manager.get_tool_docs()
        # system & instruction prompts
        instruction_prompt = PromptManager.INSTRUCTION_PROMPT.format(
            problem_statement=problem_statement,
            available_tools=tool_docs,
            tool_call_format=PromptManager.TOOL_CALL_FORMAT_PROMPT,
        )
        system_prompt = PromptManager.SYSTEM_PROMPT
        
        remaining_turns = self.max_turns
        
        while remaining_turns > 0:
            remaining_turns -= 1

            self.turn_counter += 1
            delimiter = "=" * 60 + f" TURN {self.turn_counter} " + "=" * 60
            logger.info(f"\n\n<blue>{delimiter}</blue>\n\n")
            
            logger.info(f"Temp files: {self.tool_manager.temp_files}")

            if self.dialog.use_prompt_budgeting:
                current_tok_count = self.dialog.count_tokens()
                logger.info(
                    f"(Current token count: {current_tok_count})\n"
                )

            try:
                model_response = EnhancedNetwork.inference(
                    system_prompt,
                    instruction_prompt,
                    self.dialog.get_messages_for_inference(),
                    model=QWEN_MODEL_NAME,
                    temperature=0
                )
                
                self.dialog.add_model_response(model_response)

                # Handle tool calls
                pending_tool_calls = self.dialog.get_pending_tool_calls()

                if len(pending_tool_calls) == 0:
                    # No tools were called, so assume the task is complete
                    logger.info("[no tools were called]")
                    return self.dialog.get_last_model_text_response()

                if len(pending_tool_calls) > 1:
                    raise ValueError("Only one tool call per turn is supported")

                assert len(pending_tool_calls) == 1
                tool_call = pending_tool_calls[0]
                

                text_results = [
                    item for item in model_response if isinstance(item, Types.TextResult)
                ]
                if len(text_results) > 0:
                    text_result = text_results[0]
                    logger.info(
                        f"<yellow>Agent planning next step:</yellow>\n{text_result.text}",
                    )

                Utils.format_log(json.dumps(tool_call.__dict__, indent=2, ensure_ascii=False), "🚀 Tool call")
                # Get tool from tool_manager
                tool = self.tool_manager.get_tool(tool_call.tool_name)
                if tool is None:
                    raise ValueError(
                        f"Tool with name {tool_call.tool_name} not found"
                    )

                result = tool.run(tool_call.tool_input, deepcopy(self.dialog))
                Utils.format_log(result, "🔧 Tool output")

                # Handle both ToolResult objects and tuples
                if isinstance(result, tuple):
                    tool_result, _ = result
                else:
                    tool_result = result

                self.dialog.add_tool_call_result(tool_call, tool_result)

                # Debug: Check if complete tool was called
                if tool_call.tool_name == "complete":
                    logger.info(f"Complete tool called with answer: {tool_call.tool_input.get('answer', 'No answer')}")

                # Check if complete tool should stop
                complete_tool = self.tool_manager.get_tool('complete')
                if complete_tool and complete_tool.should_stop:
                    # Add a fake model response, so the next turn is the user's
                    self.dialog.add_model_response(
                        [Types.TextResult(text="Completed the task.")]
                    )
                    return cast(ToolManager.CompleteTool, complete_tool).answer

            except Exception as e:
                error_msg = str(e)
                logger.info(f"Error in agent run: {error_msg}")
                
                # Add error to dialog and continue to next turn instead of exiting
                error_response = f"Error occurred: {error_msg}"
                
                # Add error as a model response to the dialog
                try:
                    self.dialog.add_model_response([Types.TextResult(text=error_response)])
                    logger.info(f"Added error to dialog and continuing to next turn: {error_response}")
                except Exception as dialog_error:
                    # If dialog structure doesn't allow adding model response, add as user message
                    logger.info(f"Could not add error as model response, adding as user message: {dialog_error}")
                    self.dialog.add_user_prompt(f"System Error: {error_response}")
                
                continue  # Continue to next turn instead of returning/exiting

        agent_answer = "Agent did not complete after max turns"
        return agent_answer


    def clear(self):
        self.dialog.clear()
        
def solve_fix_task(problem_statement: str):
    try:
        tool_manager = ToolManager()
        agent = Agent(
            tool_manager=tool_manager,
            max_output_tokens_per_turn=32768,
            max_turns=400
        )
        # Run the agent
        result = agent.run_agent(problem_statement)
        
        logger.info("Agent completed successfully!")
        logger.info(f"<yellow>Result: {result}</yellow>")
        
        # Generate the git patch
        patch = Utils.create_final_git_patch(tool_manager.temp_files)
        logger.info(f"<yellow>Generated patch with {len(patch)} characters...</yellow>")
        logger.info(patch)
        
        return patch
        
    except Exception as e:
        logger.error(f"Agent failed with error: {str(e)}")
        
        # Still try to generate a patch even if agent failed
        try:
            patch = Utils.create_final_git_patch([])
            logger.error(f"Generated patch despite agent failure: {len(patch)} characters")
            return patch
        except Exception as patch_error:
            logger.error(f"Failed to generate patch: {str(patch_error)}")
            return ""
 
def get_code_skeleton() -> str:
    # Initialize the result string
    result = ""
    
    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"
    
    return result

def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re
    
    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:

        ```text
        This is a test.

        This is another test!
        ```text

        Then the text block should be:

        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split('\n')
        processed_lines = []
        
        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                # Mark trailing and leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))
                
                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
            
            processed_lines.append(f"\"{processed_line}\"")
        
        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
            
    # Pattern to match ```text...``` blocks
    pattern = r'```text\n(.*?)\n```'
    
    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)
        
        return f'```text\n{processed_content}\n```'
    
    # Replace all text blocks with processed versions
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction

def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    while retry < 10:
        try:
            code_response = EnhancedNetwork.make_request(code_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Code Generation completed")
            
            # Step 5: Infinite Loop Check and Validation
            loop_check_messages = [
                {
                    "role": "system",
                    "content": INFINITE_LOOP_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code."
                }   
            ]
            
            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Infinite Loop Check completed")

            # Clean up the final response (use loop check response as it's the final validated version)
            solution = loop_check_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()
            
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning solution generation failed")
        return ""
    
    return ""

def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting multi-step reasoning solution generation")
            
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton)
            
            if solution:
                logger.info("Generated initial solution successfully using multi-step reasoning")
                return solution
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                
                # Clean up the response
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()
                
                logger.info("Generated initial solution successfully using fallback approach")
                return solution
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    import re
    
    created_files = []
    
    if not initial_solution.strip():
        print("No solution content to process")
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()
        
        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):  # Not a comment
            
            # Write the previous file if we have one
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                # Create directory if needed (for subdirectories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Join content and remove empty lines at start/end
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            
            # Start new file
            current_filename = stripped_line
            current_content = []
        else:
            # This line is content for the current file
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    
    return created_files

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Testcase Generation completed")
            
            # Step 5: Infinite Loop Check and Validation
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                }   
            ]
            
            testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Testcase check completed")

            # Clean up the final response (use loop check response as it's the final validated version)
            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning testcase generation failed")
        return ""
    
    return ""

def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting test cases generation")
            
            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)
            
            if testcases:
                logger.info("Generated testcases successfully using multi-step reasoning")
                return testcases
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                
                # Clean up the response
                testcases = response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                
                logger.info("Generated testcases successfully using fallback approach")
                return testcases
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def process_create_task(input_dict):
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)
    print(problem_statement)

    code_skeleton = get_code_skeleton()
    start_time = time.time()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton)
    print(initial_solution)
    
    # Extract and write files from the solution
    created_files = extract_and_write_files(initial_solution)
    print(f"Created or Updated {len(created_files)} files: {created_files}")

    
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    print(test_cases)
    # Extract and write files from test cases
    test_files = extract_and_write_files(test_cases)
    print(f"Created or Updated {len(test_files)} files: {test_files}")

    timeout = DEFAULT_TIMEOUT - (time.time()-start_time) - 60
    
    patch = solve_fix_task(problem_statement)

    if patch is None: # Failed to fix by testcases, maybe testcases are wrong so try to use original solution
        extract_and_write_files(initial_solution)

    # Create a temporary tool manager to get list of created files
    temp_tool_manager = ToolManager()
    # Add test files to temp files list so they're excluded from patch
    for test_file in test_files:
        temp_tool_manager.add_temp_file(test_file)
    
    # Generate final patch excluding test files
    patch = Utils.create_final_git_patch(temp_tool_manager.temp_files)
    return patch

def get_directory_tree(start_path: str = '.') -> str:

    tree_lines = []
    
    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            # Get the directory name
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            
            # Add current directory to tree (skip for root directory)
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            
            # Get all items in directory
            try:
                items = os.listdir(path)
                # Filter out hidden directories and files starting with '.'
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                
                # Separate directories and files
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                
                # Process directories first
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                
                # Then process files
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                    
            except PermissionError:
                # Handle directories we can't read
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")
                
        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    
    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)

def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}"}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)

            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            retry += 1
        
        time.sleep(2)

    return response

def agent_main(input_dict: dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """
    Main agent function that generates a git patch to solve the problem statement.
    
    Args:
        input_dict: Dictionary containing the problem statement and other parameters
        repo_dir: Path to the repository directory
        test_mode: Whether to run in test mode
    
    Returns:
        str: Git patch as a string
    """
    
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)
    
    Utils.ensure_git_initialize()
    Utils.set_env_for_agent()
    
    logger.info(f"<blue>📁 Repository:</blue> {repo_dir}")
    
    try:
        problem_type = check_problem_type(input_dict.get("problem_statement"))

        if problem_type == PROBLEM_TYPE_FIX:
            result = solve_fix_task(input_dict.get("problem_statement"))
        else:
            result = process_create_task(input_dict)
    except Exception as e:
        result = solve_fix_task(input_dict.get("problem_statement"))

    os.system("git reset --hard")
    
    return result