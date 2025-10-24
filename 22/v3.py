#V3.4

from __future__ import annotations
import re
import os
import ast
import sys
import json
import uuid
import time
import httpx
import logging
import asyncio
import subprocess
import requests
from ast import literal_eval
from json import JSONDecodeError
from enum import Enum
from typing import get_origin, NamedTuple
from typing import TypedDict, Any, Callable, Dict, List, Optional, Set, Tuple, AsyncGenerator, Sequence, cast, Union, get_args, get_origin, Literal
from pathlib import Path
from functools import partial
from typing_extensions import final
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from autogen_core import CancellationToken
from autogen_core.models import ModelFamily
from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import ToolCallExecutionEvent, ToolCallSummaryMessage
from autogen_core.models._types import FunctionExecutionResult
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage

SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 200000

## all global variables here -------------------------------------------------------->>
RUN_ID = os.getenv("RUN_ID") or str(uuid.uuid4())
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://31.22.104.92:8000")
# DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
GLM_MODEL_NAME_46="zai-org/GLM-4.6-FP8"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"

AGENT_MODELS=[QWEN_MODEL_NAME, GLM_MODEL_NAME,KIMI_MODEL_NAME,DEEPSEEK_MODEL_NAME]

JSON_LLM_USED = 0
JSON_LITERAL_USED = 0
MARKDOWN_FAILED = 0
TOOL_CALL_FAILED = 0
MAX_EMBED_TOKENS = 128000
MAX_EMBED_CHARS = MAX_EMBED_TOKENS*4
DISABLE_TEST_FILE_REMOVAL = False
TOO_MANY_SECTIONS_FOUND = 0
MAX_RESPONSE_LEN: int = 200_000

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

class logger:
    """Custom logger with colored messages and color tag support."""
    
    # ANSI color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m',
        'light_gray': '\033[37m', 
        'reset': '\033[0m',
        'faint': '\033[2m',   
        'very_faint': '\033[2m\033[90m',
        'bold': '\033[1m',
        'dim': '\033[2m',
    }
    
    # Default colors for each log level
    LEVEL_COLORS = {
        'DEBUG': 'very_faint',
        'INFO': 'white',
        'WARNING': 'yellow',
        'EXCEPTION': 'magenta',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
    
    @classmethod
    def _colorize(cls, text, color):
        """Apply color to text."""
        if color in cls.COLORS:
            return f"{cls.COLORS[color]}{text}{cls.COLORS['reset']}"
        return text
    
    @classmethod
    def _parse_color_tags(cls, message):
        """Parse custom color tags like <red>text</red>."""
        import re
        # Pattern to match only valid color tags (not XML tags)
        valid_colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'gray', 'light_gray', 'bold', 'dim', 'faint', 'very_faint']
        color_pattern = '|'.join(valid_colors)
        pattern = rf'<({color_pattern})>(.*?)</\1>'
        
        def replace_color(match):
            color = match.group(1)
            text = match.group(2)
            return cls._colorize(text, color)
        
        return re.sub(pattern, replace_color, message)
    
    @classmethod
    def _log(cls, level, message):
        """Internal logging method."""
        # Get default color for this level
        default_color = cls.LEVEL_COLORS.get(level, 'white')
        
        # Check if message has color tags
        has_color_tags = any(f'<{color}>' in message for color in cls.COLORS.keys())
        
        if has_color_tags:
            # Message has color tags, parse them first, then apply default color to entire result
            parsed_message = cls._parse_color_tags(message)
            colored_message = cls._colorize(parsed_message, default_color)
        else:
            # No color tags, apply default level color to entire message
            colored_message = cls._colorize(message, default_color)
        
        print(colored_message)
    
    @classmethod
    def debug(cls, message):
        """Log debug message."""
        cls._log('DEBUG', message)
    
    @classmethod
    def info(cls, message):
        """Log info message."""
        cls._log('INFO', message)
    
    @classmethod
    def warning(cls, message):
        """Log warning message."""
        cls._log('WARNING', message)
        
    @classmethod
    def exception(cls, message):
        """Log warning message."""
        cls._log('EXCEPTION', message)
    
    @classmethod
    def error(cls, message):
        """Log error message."""
        cls._log('ERROR', message)
    
    @classmethod
    def critical(cls, message):
        """Log critical message."""
        cls._log('CRITICAL', message)
    
    # Convenience methods
    @classmethod
    def warn(cls, message):
        """Alias for warning."""
        cls.warning(message)
    
    @classmethod
    def err(cls, message):
        """Alias for error."""
        cls.error(message)

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
    class ToolImplOutput:
        """Output from an LLM tool implementation."""
        tool_output: str
        tool_result_message: str
        auxiliary_data: dict[str, Any] = field(default_factory=dict)
        
    @dataclass
    class ProxyMessage:
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
        GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"

    Model = Literal[
        "zai-org/GLM-4.5-FP8", 
        "moonshotai/Kimi-K2-Instruct", 
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 
        "deepseek-ai/DeepSeek-V3-0324"
    ]
    
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
    ToolInputSchema = dict[str, Any]
    """A JSON schema describing the input to a tool."""

# INDENT UTILITY HELPER
# =============================================================================

class textwrap:
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

class Utils:
    """Utility class containing all helper functions."""

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
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Simple JSON schema validation without external dependencies"""
        
        def validate_object(obj: Any, schema_props: Dict[str, Any]) -> bool:
            """Validate object against schema properties"""
            
            if not isinstance(obj, dict):
                return False
            
            # Check required fields
            required = schema_props.get("required", [])
            for field in required:
                if field not in obj:
                    return False
            
            # Check each property
            for prop_name, prop_schema in schema_props.get("properties", {}).items():
                if prop_name in obj:
                    if not validate_value(obj[prop_name], prop_schema):
                        return False
            
            return True
        
        def validate_value(value: Any, schema: Dict[str, Any]) -> bool:
            """Validate a single value against its schema"""
            
            # Check type
            expected_type = schema.get("type")
            if expected_type == "string":
                if not isinstance(value, str):
                    return False
            elif expected_type == "integer":
                if not isinstance(value, int):
                    return False
            elif expected_type == "boolean":
                if not isinstance(value, bool):
                    return False
            elif expected_type == "array":
                if not isinstance(value, list):
                    return False
                # Validate array items
                items_schema = schema.get("items", {})
                for item in value:
                    if not validate_value(item, items_schema):
                        return False
            elif expected_type == "object":
                if not isinstance(value, dict):
                    return False
                # Validate object properties
                properties = schema.get("properties", {})
                return validate_object(value, {"properties": properties})
            
            return True
        
        # Main validation
        if schema.get("type") == "object":
            return validate_object(data, schema)
        
        return validate_value(data, schema)    
    
    @staticmethod
    def format_log(content: Any, label: str):
        logger.info(f"\n\n<yellow>-------------------------------- [{label}] --------------------------------</yellow>")
        print(content)
        logger.info(f"<yellow>------------------------------ [End {label}] ------------------------------</yellow>")
  
class ProxyClient(OpenAIChatCompletionClient):
    
    class Utils:
        @classmethod  
        def is_json_string(cls, raw_text: str) -> bool:
            return ("{" in raw_text[:10] and "}" in raw_text[len(raw_text)-10:]) or ("[" in raw_text[:10] and "]" in raw_text[len(raw_text)-10:])
        
        @classmethod
        def stable_tool_call_id(cls, name: str, args: dict | list | str) -> str:
            key = f"{name}:{json.dumps(args, sort_keys=True)}"
            return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

        @classmethod
        def parse_response(cls, raw_text: str, expected_format: str | None = None) -> Tuple[str, list[dict]|None, str]:
            global JSON_LLM_USED, JSON_LITERAL_USED
            raw_text2 = raw_text
            #logger.info("raw_text:{}".format(raw_text))
            raw_text=cls._strip_code_fences(raw_text)
            parse_error = None
            error_msg = ""
            try:
                if cls.is_json_string(raw_text):
                    raw_text = json.loads(raw_text)
                    if isinstance(raw_text, str): # sometimes server returns leading quotes.
                        raw_text = json.loads(raw_text)
            except Exception as e:
                parse_error = e
                try:
                    with open("raw_text.txt", "w") as f:
                        f.write(raw_text)
                    with open("raw_text2.txt", "w") as f:
                        f.write(raw_text2)
                    JSON_LITERAL_USED += 1
                    raw_text = literal_eval(raw_text)
                    parse_error = None  # Successfully recovered
                    if isinstance(raw_text, str):
                        raw_text = json.loads(raw_text)
                        if isinstance(raw_text, str):
                            raw_text = json.loads(raw_text)
                except Exception as e:
                    parse_error = e
                    if isinstance(raw_text, str):
                        JSON_LLM_USED += 1
                        logger.info("Trying to fix json string with llm")
                        logger.info(raw_text)
                        raw_text_n = EnhancedNetwork.fix_json_string_with_llm(raw_text)
                        if raw_text_n:
                            raw_text = json.dumps(raw_text_n)
                            parse_error = None  # Successfully recovered
                        else:
                            logger.info("json load failed")
                            # Build comprehensive error message with context
                            error_msg = f"Invalid JSON format. Error: {str(e)}"
                            if expected_format:
                                error_msg += f"\n\nExpected format:\n{expected_format}"
                            error_msg += f"\n\nYour response:\n{raw_text[:500]}{'...' if len(raw_text) > 500 else ''}"
                            error_msg += "\n\nPlease fix the JSON format and respond again."
            
            content_text = ""
            tool_calls = None
            error = error_msg if parse_error is not None else ""
            
            if isinstance(raw_text, (dict, list)):
               
                if type(raw_text) == dict and raw_text.get("response_type")=="tool":
                    if raw_text.get("tool_calls") is not None and isinstance(raw_text.get("tool_calls"), list) and len(raw_text.get("tool_calls")) > 0:
                        
                        tool_calls=raw_text.get("tool_calls")
                        try:
                            logger.info("<green>🤖 Found tool calls</green>\n{tool_calls}")
                            tool_calls=[{"id":cls.stable_tool_call_id(call.get("name"),call.get("arguments")),"type":"function","function":{"name":call.get("name"),"arguments":json.dumps(call.get("arguments") if isinstance(call.get("arguments"), (dict, list)) else {"input":call.get("arguments")})}} for call in tool_calls]
                            content_text=""
                        except Exception as e:
                            error_detail = f"Invalid tool_calls arguments. Error: {str(e)}"
                            if expected_format:
                                error_detail += f"\n\nExpected format:\n{expected_format}"
                            error_detail += "\n\nPlease ensure tool_calls is properly formatted."
                            error = error_detail
                            logger.error(f"cannot fix tool_calls arguments: {e}")
                            content_text=json.dumps(raw_text)
                            tool_calls=None
                        
                    else:
                        logger.info("found no tool calls, invalid tool_calls arguments.")
                        error_detail = "Invalid tool_calls arguments - tool_calls is missing, not a list, or empty."
                        if expected_format:
                            error_detail += f"\n\nExpected format:\n{expected_format}"
                        error = error_detail
                        content_text=json.dumps(raw_text)
                    
                else:
                    #logger.info("json but not a tool call")
                    #logger.info("json but not a tool call, json load succeeded")
                    content_text=json.dumps(raw_text)
            else:
                if (raw_text[0]=="\"" or raw_text[0]=="'") and (raw_text[-1]=="\"" or raw_text[-1]=="'"):
                    try:
                        raw_text=literal_eval(raw_text)
                    except Exception as e:
                        logger.error("literal eval failed..")
                        pass
                content_text=raw_text
            
            return content_text, tool_calls, error

        @classmethod
        def is_empty_response(cls, response:str)->bool:
            return not response or response=="null" or response.strip()==""
        
        @classmethod
        def is_network_error(cls, response:str)->bool:
            return  "<|reserved_token_" in response or "API request failed with status 429" in response or "Read timed out" in response or "Network unreachable" in response or "Connection refused" in response
        
        @classmethod
        def _strip_code_fences(cls, text: str) -> str:
            if re.search(r"^=+\s*[A-Z_]+$",text,re.MULTILINE): # ignore if its a markdown text #^=+\s*[A-Z_]+$
                return text
            if text:
                text=text.strip().strip("\n")
            fenced = re.search(r"```(?:json)\s*(.*?)```$", text, flags=re.DOTALL | re.IGNORECASE)
            if fenced:
                fenced=fenced.group(1).strip().strip("\n")
                return fenced
            m = re.search(r"[^`]*```python\r?\n(.*?)\n?```.*", text, flags=re.DOTALL|re.IGNORECASE)
            text = m.group(1) if m else text
              
            return text.strip()

        @classmethod
        def _extract_text_from_message(cls, message):
            try:
                if isinstance(message, TextMessage):
                    return message.content
                if isinstance(message, FunctionExecutionResult):
                    return message.content
                if isinstance(message, ToolCallExecutionEvent):
                    # Tool calls sometimes return a list of chunks
                    try:
                        return message.content[0].content
                    except Exception:
                        return str(message.content)
                # Fallback: try generic .content
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    return "".join([getattr(c, "content", str(c)) for c in content])
                if content is not None:
                    return content
                return str(message)
            except Exception as e:
                return str(message)
    
    def __init__(self, model: str, base_url: str, agent_prefix: str):
        self.model_name = model
        super().__init__(
            model = model,
            base_url = base_url,
            api_key = "",
            model_info = {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            }, 
            timeout=180
        )
        self.parsing_error = None
        self.agent_prefix = agent_prefix
        self._client._client._event_hooks['request'] = [self.request_modify]
        self._client._client._event_hooks['response'] = [self.response_modify]
    
    async def request_modify(self, request):
        
        self.parsing_error = None
        await request.aread()
        
        try:
            raw = request.content.decode('utf-8') if request.content else '{}'
            body_data = json.loads(raw)
        except Exception as e:
            logger.error(f"Error parsing request content: {e}")
            body_data = {}
        messages=[]
 
        for m in body_data.get("messages", []):
            if m.get("content") != None:
                messages.append(m)
            else:
                content=""
                for k in m.keys():
                    if k not in ["role","content"]:
                        content+=f"{k}: {m[k]}\n"
                messages.append({"role":m.get("role"),"content":content})
        # Build body with only model, messages, and run_id

        new_body = {
            "model": self.model_name,
            "messages": messages,
            "run_id": RUN_ID, 
            "temperature": 0.0,
            "agent_id": self.agent_prefix + ":" + RUN_ID
        }
        new_bytes = json.dumps(new_body).encode('utf-8')
 
        # Update URL and replace stream safely
        request.url = request.url.copy_with(path="/api/inference")
        request.headers["content-type"] = "application/json"
        request.headers["content-length"] = str(len(new_bytes))
        request._content = new_bytes
        request.stream = httpx.ByteStream(new_bytes)  # provide body bytes without httpx
 
        return request

    async def response_modify(self, response):
        data = await response.aread()
        #raw_text=response
        
        raw_text = data.decode('utf-8') if data else ""
        raw_text=raw_text.strip()

        content_text, tool_calls, _ = self.__class__.Utils.parse_response(raw_text)
        # logger.info(f"Content text: {content_text}")
        
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content_text
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
            # logger.info(message)
        oai_response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop" if message.get("tool_calls") is None else "tool_calls",
                    "message": message,
                }
            ],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

        new_bytes = json.dumps(oai_response).encode("utf-8")
        response.headers["content-type"] = "application/json"
        response.headers["content-length"] = str(len(new_bytes))
        response._content = new_bytes
        response.stream = httpx.ByteStream(new_bytes)
        return response
    
class CustomAssistantAgent(AssistantAgent):
    
    class ResponseValidator:
        @classmethod
        def check_tool_call_section(cls, response: str, raw_response: str,correct_format: str)->str:
            if not("TOOL_CALL" in raw_response and re.search(r"^=+\s*[A-Z_]+$", raw_response, re.MULTILINE)):
                return "Invalid response, please respond in correct format: {correct_format}"
            return "success"
    
    def __init__(
        self,
        agent_name = "assistant",
        model_name: str = QWEN_MODEL_NAME,
        system_message: str | None = None,
        tools: list | None = None,
        max_tokens: int = 100_000
    ):
        self.semaphore=asyncio.Semaphore(3)
        self.agent_idx=0
        self.agent_name=agent_name
        self.model_name=model_name
        self.system_message=system_message
        #self.register_hook("process_all_messages_before_reply",CustomAssistantAgent.Utils.drop_last_assistant)
        self.model_client = ProxyClient(
            model=self.model_name, 
            base_url=DEFAULT_PROXY_URL, 
            agent_prefix=self.agent_name
        )
        if not tools:
            self.agent: AssistantAgent = AssistantAgent(
                name=self.agent_name,
                model_client=self.model_client,
                reflect_on_tool_use=False,
                system_message=self.system_message
            )
        else:
            self.agent: AssistantAgent = AssistantAgent(
                name=self.agent_name,
                model_client=self.model_client,
                reflect_on_tool_use=False,
                system_message=self.system_message,
                tools=tools
            )
    
    def parse_markdown(self, text: str, return_type: type | None = None) -> Any:
        """Parse markdown text with ==== section headers into a tuple based on return_type."""
        global MARKDOWN_FAILED
        global TOO_MANY_SECTIONS_FOUND
        
        # regex = r"^=+\s*.*[A-Z_]+.*=+$"
        regex = r"^=+\s*.*[A-Z_]+.*=+$"
        
        if get_origin(return_type) is Union:
            # determine what we have..
            args = get_args(return_type)
            no_sections = re.findall(r"^=+\s*[A-Z_]+$", text, re.MULTILINE)
            if len(no_sections) <= 1 and str in args:
                return_type = str
            
        if not return_type:
            return True, None, text
        if return_type == str:
            start_line = [idx for idx, t in enumerate(text.split("\n")) if re.search(r"^=+\s*[A-Z_]+$",t)]
            if start_line:
                start_line = start_line[0]
            else:
                start_line = 0
            return True, None, "\n".join([t for t in text.split("\n")[start_line:] if (not re.search(r"^=+\s*[A-Z_]+$", t) and not re.search(r"^={3,}$",t))])
        if return_type == list:
            return True, None, [t for t in text.split("\n") if not re.search(r"^=+\s*[A-Z_]+$",t)]
        if return_type == tuple or get_origin(return_type) == tuple:
            sections = []
            current_content = []
            
            lines = text.split("\n")
            start_line = 0
            #skipping the start section which does not belong to any section..
            for idx, line in enumerate(lines):
                if re.search(r"^=+\s*[A-Z_]+$",line):
                    start_line = idx
                    break
            for line in lines[start_line:]:
                if re.search(r"^=+\s*[A-Z_]+$", line):
                    # If we have accumulated content, save it
                    if current_content:
                        sections.append("\n".join(current_content).strip())
                        current_content = []
                    # Extract section name from the header line
                    match = re.search(r"([A-Z]+)", line)
                    if match:
                        # Section header found, reset content accumulator
                        current_content = []
                else:
                    # Accumulate content for current section
                    current_content.append(line)
            
            # Don't forget the last section
            if current_content:
                sections.append("\n".join(current_content).strip())
            
            # Check if return_type is a tuple and get expected length
        
            # Try to extract tuple length from return_type hints like tuple[str, str]
            if hasattr(return_type, '__args__'):
                expected_length = len(return_type.__args__)
            else:
                # If no hints, just return tuple of all sections
                return True,None,tuple(sections)
            
            if len(sections) != expected_length:
                MARKDOWN_FAILED += 1
                return False, f"Expected {expected_length} markdown sections but found {len(sections)}", None
            else:
                return True, None, tuple(sections)
        
        else:
            MARKDOWN_FAILED += 1
            raise ValueError(f"Invalid return type: {return_type}")
                    
    async def solve_task(
        self,
        task: str,   
        response_format: str,
        is_json: bool,
        regex: str | None = None,
        post_process_func: Callable | None = None,
        max_attempts: int = 3,
        is_parallel: bool = False,
        disable_reset: bool = False,
        return_type = None
    ):
        # write code here to make the request to assistant agent , fetch response, check the response format matches with regex shared. If not it must through an error and ask to regenerate. It must then convert the reponse to json if is_json is True else return. Also, if there is any error in parsing json it must send an error to LLM and ask to regerate. The LLM must have less than 3 attempts to generate the response successfully.
        
        async with self.semaphore:
            if is_parallel:
                logger.info("<blue>🤖 Creating new agent..</blue> {}".format(self.agent_idx))
                self.agent_idx += 1
                agent = AssistantAgent(
                    name=self.agent_name,
                    model_client=ProxyClient(
                        model=self.model_name, 
                        base_url=DEFAULT_PROXY_URL, 
                        agent_prefix=self.agent_name
                    ),
                    reflect_on_tool_use=False,
                    system_message=self.system_message
                )
            else:
                agent = self.agent
            
            if not disable_reset:
                # Create a simple mock CancellationToken since we can't use autogen_core
                await agent.on_reset(CancellationToken())
                
            attempts = 0
            
            full_task = (
                    f"{task}\n\n"
                    f"\n{response_format}\n\n"
                )
            while attempts < max_attempts:
                
                logger.info(f"<green>😴 Agent trying to answer with attempt [{attempts}]...</green>\n")
                attempts += 1
                try:
                    result: TaskResult = await Console(agent.run_stream(task=full_task))
                except Exception as e:
                    logger.error(f"Agent call failed: {type(e)}:{e}, sleeping for 2 seconds before retrying..")
                    time.sleep(2)
                    continue
                
                # Utils.format_log(result, "Assistant Result")
                # # Find the last non-summary message
                last_message = None
                try:
                    for m in result.messages[::-1]:
                        if isinstance(m, ToolCallSummaryMessage):
                            continue
                        last_message = m
                        break
                except Exception:
                    last_message = None
                    
                # Utils.format_log(last_message, "Last Message")
                if last_message is None:
                    last_error = "No response message returned by assistant."
                    logger.error("No response message returned by assistant. This should not happen..")
                    continue
                
                candidate_text = ProxyClient.Utils._extract_text_from_message(last_message).strip()
                
                candidate_text, _, error = ProxyClient.Utils.parse_response(candidate_text, expected_format=response_format if response_format else None)
                
                if error:
                    full_task=error
                    logger.info(f"Assistant attempt {attempts} error: {error}")
                    continue
                    
                    
                if ProxyClient.Utils.is_empty_response(candidate_text) or ProxyClient.Utils.is_network_error(candidate_text):
                    full_task="network error. please try again."
                    continue
                # Regex validation
                if regex and not re.search(regex, candidate_text, flags=re.DOTALL):
                    full_task=f"Response did not match the required response format. You need to respond with this format: {response_format}"
                    logger.info(f"assistant attempt {attempts} failed regex. Text: {candidate_text[:2000]}")
                    continue
                
                #if "Error: " in candidate_text:
                #    full_task=candidate_text
                #    logger.info(f"assistant attempt {attempts} error: {candidate_text}")
                #    continue
                
                if not is_json:
                    cleaned = self.model_client.Utils._strip_code_fences(candidate_text)
                    is_success, error_message, cleaned = self.parse_markdown(cleaned, return_type)
                    if not is_success:
                        if error_message and "expected" in error_message.lower():  
                            logger.info(f"context length before rejection: {len(agent.model_context._messages)}")
                            logger.info("removing the last assistant messages from context")
                            agent.model_context._messages = [m for m in agent.model_context._messages if  isinstance(last_message,TextMessage) and m.content!=last_message.content]
                            logger.warning(f"context length after rejection: [{len(agent.model_context._messages)}]")
                            sections = re.findall(r"^=+\s*[A-Z_]+$",candidate_text,re.MULTILINE)
                            if sections:
                                sections = [s.replace("=","").strip() for s in sections if s.strip()]
                                logger.info(f"len(sections): {len(sections)}, return_type: {return_type}")
                                if len(sections) > 2:
                                    sections = list(set(sections))
                                    full_task = "Respond in correct format. You must not have multiple sections of {}".format(",".join(sections))
                                elif return_type == tuple[str,str] and len(sections) < 2:
                                   full_task = "Respond in the correct format. You are missing a section in your response. Check if THOUGHT or any other section is missing."
                                else:
                                    full_task = None
                            else:
                                full_task = None
                        else:
                            full_task = error_message
                        logger.info(f"assistant attempt {attempts} error: {error_message}")
                        continue
                    if post_process_func:
                        resp_post_process = post_process_func(cleaned,candidate_text)
                        if resp_post_process != "success":        
                            full_task = f"Invalid response:{resp_post_process}"
                            logger.info(f"assistant attempt {attempts} invalid response: {resp_post_process}")
                            continue
                    return cleaned
                
                # Parse JSON with best-effort cleanup
                try:
                    cleaned = ProxyClient.Utils._strip_code_fences(candidate_text)
                    
                    parsed = json.loads(cleaned)
                    if post_process_func:
                        resp_post_process = post_process_func(parsed,candidate_text)
                        if resp_post_process != "success":
                            full_task = f"Invalid response:{resp_post_process}"
                            logger.info(f"assistant attempt {attempts} invalid response: {resp_post_process}")
                            continue
                    return parsed
                except Exception as e:
                    full_task=f"Invalid JSON: {e}. Please respond with the exact same format as the response format: {response_format}"
                    logger.info(f"Unexpected JSON format: {e}")
                    continue
            return None

class CreateProblemSolver:
    
    SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL=textwrap.dedent("""
    You are an expert Python developer. You will be given a problem statement and a python solution. You need to evaluate if the solution is correct or not as per the problem statement.
    
    WorkFlow:-
        - **Plan:** After understanding the problem statement, create a initial list of all the requirements mentioned in problem statement that you need to evaluate.
        - **Evaluate:** Begin evaluating the solution for each of those cases. Create test cases to confirm if the solution is correct.
        - **Adapt:** As you discover new information or encounter obstacles, update your plan.
        - **Verify (Tests):** Check test_cases.txt file for additional scenerios you can test.
        - **Comprehensive Testing:** Think about all the edge cases. Ensure the solution handles all of them. Run comprehensive test to ensure solution fully satisfies all the requirements.
        - **Finish:** Call complete tool once the solution fully satisfies all the requirements.

    *GUIDE FOR HOW TO USE "sequential_thinking" TOOL:*
        1. Your thinking should be thorough and so it's fine if it's very long. Set totalThoughts to at least 5, but setting it up to 25 is fine as well. You'll need more total thoughts when you are considering multiple possible solutions or root causes for an issue.
        2. Use this tool as much as you find necessary to improve the quality of your answers.
        3. You can run bash commands (like tests, a reproduction script, or 'grep'/'find' to find relevant context) in between thoughts.
        4. The "sequential_thinking" tool can help you break down complex problems, analyze issues step-by-step, and ensure a thorough approach to problem-solving.
        5. Don't hesitate to use it multiple times throughout your thought process to enhance the depth and accuracy of your solutions.

    Tool Usage:-
        - Use bash tool to create and run unittests.
        - Use str_replace_editor to fix the solution if it fails.
        - Use str_replace_editor to fix the test case if they are not as per the problem statement.
        - Use complete to finish the task.
    
    Rules:-
        1. Test code must always import functionality from the repository—never duplicate or reimplement the code within the test itself.
        2. Use verbosity level 2 while running the tests to ensure you see the full output.
        3. If bash tool throws syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases and try again.
        4. Must ensure you have tested **ALL scenarios** listed in test_cases.txt file. Even if some  scenarios are not mentioned in problem statement, you must test them.
    
    Tips:-
        - You must make changes in the project directory in order to ensure the requirements specified in the <problem_statement> are met. Leaving the directory unchanged is not a valid solution.
        - Respect the tool specifications. If a field is required, make sure to provide a value for it. For example "thoughtNumber" is required by the sequential_thinking tool.
        - When you run "ls" with the bash tool, the "view" command with the "str_replace_editor" tool, or variants of those, you may see a symlink like "fileA -> /home/augment/docker/volumes/_data/fileA". You can safely ignore the symlink and just use "fileA" as the path when read, editing, or executing the file.
        - When you need to find information about the codebase, use "grep" and "find" to search for relevant files and code with the bash tool
        - Use your bash tool to set up any necessary environment variables, such as those needed to run tests.
    
    Here are the tools you have access to:-
    {tools_docs}
    
    You must follow the following response format.
    {format_prompt}
    """
    )
    
    INSTANCE_PROMPT_INITIAL_SOLUTION_EVAL=textwrap.dedent("""
    Problem Statement:
    {problem_statement}
    
    Key Python Files:
    {initial_solution}
    
    """
    )
    
    SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

    Strict Requirements:
    1. Output the full content of Python files along with their file names.
    2. Do not include explanations, comments, or markdown formatting.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    
    """
    )
    
    INSTANCE_PROMPT=textwrap.dedent("""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files.""")
    
    RESPONSE_FORMAT="""Return only the final python files code.

    Response Examples:
    ```python
    a.py
    {content}

    b.py
    {content}
    ```"""
    
    RESPONSE_FORMAT_JSON="""Return only the final python files code in JSON format.
    Response Examples:
    [{"file_name":"a.py","code":"contents of a.py"},{"file_name":"b.py","content":"contents of b.py"}]
    """
    
    RESPONSE_FORMAT_SOLUTION_EVAL="""
    Your response must not contain multiple THOUGHT or TOOL_CALL sections. You must respond in the following format. You must not add anything before THOUGHT section.
    ===================THOUGHT
    <<your detailed thought process>>
    ===================TOOL_CALL
    {"name":"<tool_name>","arguments":{...}}
    """
    RESPONSE_FORMAT_SOLUTION_EVAL_2="""
    Your response must not contain multiple TOOL_CALL sections. You must add your detailed analysis before TOOL_CALL section. You must respond in the following format.
    ===================THOUGHT
    <<your detailed thought process>>
    ===================TOOL_CALL
    {"name":"<tool_name>","arguments":{...}}
    """
    
    TEST_CASE_GENERATOR_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Python unittest testcase developer. 
    Important points:-
    - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
    - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
    
    You must respond directly with the test cases in the following format. 
    =========TEST_CASES
    <<test cases>>
    Do not include anything else. For Example:
    =========TEST_CASES
    # These tests are auto-generated with test data from:
    # https://github.com/xxxx.json
    # File last updated on 2023-07-20
    import unittest
    from main_module import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()
    """)
    
    TEST_CASES_GEN_INSTANCE_PROMPT=textwrap.dedent("""Problem Statement:\n{problem_statement}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases.""")
    
    TESTCASES_CHECK_PROMPT = textwrap.dedent(
    """
    You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.
    WorfFlow:-
    1. Read the problem statement carefully and note down all the requirements/edge cases/worflows/mathmatical formulas/data/etc.
    2. Derive expected output for each test case one by one. You must include working, reflection, reflection2, reflection3 and final expected output in your response for each test case.
    3. Reply with 'DONE' once you have verfied all the test cases.
    
    
    For Example:
    Generated Test Case Code:
    def test_add_two_numbers(self):
        self.assertEqual(add_two_numbers(1, 2), 3)
    Output:
    Test case 1:test_add_two_numbers:
    - Working - I need to add two numbers and check output. As per the problem statement, all arguments needs to be positive integers.
    Both 1 and 2 are integers. So 1+2=3. So the expected output is 3.
    - Reflection - Let me double check my work above. I checked if both are positive integers and then i checked the calculation. I am confident about my work. 1+2=3. So the expected output is 3.
    - Reflection2 - <<double check your work on reflection, must recheck all the steps and ensure you have not missed anything>>
    - Reflection3 - <<double check your work on reflection2, must recheck all the steps and ensure you have not missed anything>>
    - Final Expected Output:- 3
    
    Now no more test cases to check, let me end my work here.
    DONE
    """)
    
    INSTANCE_TESTCASES_CHECK_PROMPT=textwrap.dedent(""""Problem statement: {problem_statement}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}
                                 
    Now check all the test cases above. Please note you must check each test case carefully one by one as per the plan you created.
                                     
    """)
    
    FINAL_OUTPUT_TEST_CHECK=textwrap.dedent("""
    Now share the final code in the following format.
    =========TEST_CASES
    <<final code here>>
    For example:
    # These tests are auto-generated with test data from:
    # https://github.com/xxxx.json
    # File last updated on 2023-07-20
    import unittest
    from main_module import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()       
    """)
    
    class ResponseValidator:
        @classmethod
        def check_syntax_error(cls, content:str, raw_response:str)->str:
            try:
                if "```python" in content:
                    return """Do not include any markups like ```python, ```, etc. in the response.
                Follow the response format strictly like below:-
                =======TEST_CASES
                import unittest
                from main_module import (
                    main_func
                )

                class TestFuncA(unittest.TestCase):
                    def test_main_func(self):
                        self.assertEqual(main_func(), "expected_output")

                if __name__ == "__main__":
                    unittest.main()
                """
                ast.parse(content)
                return "success"
            except Exception as e:
                logger.error(f"Syntax error: {e}")
                logger.error(f"Content: {content}")
                # check the length of content and if high advise to reduce the number of test cases.
                no_tokens=len(content.split(" "))
                if "unittest.main()" not in content:
                    return "Generation limit reached. Response truncated.. Skip last couple of test cases from your last response.."
                return f"Syntax error: {e}\n If the syntax error is due to the response getting truncated skip last couple of test cases and try again."
    
    def process_response(self, response) -> Tuple[str | None, str]:
        resp = None
        tool_name = ""
        tool_call = None
        global TOOL_CALL_FAILED
        #logger.info(f"response: {response}")
        if response is None:
            logger.error("response NONE received..")
            return None, ""
        if type(response) is tuple and len(response) == 2:
            _, tool_call = response
        elif "{" in response:
            tool_call = response
        if tool_call:
            expected_tool_format = """Expected JSON format for tool calls:
{"name": "<tool_name>", "arguments": {<arguments_dict>}}

Example:
{"name": "bash", "arguments": {"command": "ls -la"}}"""
            json_obj, _, error = ProxyClient.Utils.parse_response(str(tool_call), expected_format=expected_tool_format)
            if error:
                resp=error
            elif json_obj:
                try:
                    json_obj=json.loads(json_obj)
                    logger.info("\n\n<yellow>🤖 calling tool:</yellow> " + json_obj.get("name") + " with arguments: " + json.dumps(json_obj.get("arguments")))
                    # resp = self.tool_map[json_obj.get("name")](**json_obj.get("arguments"))
                    tool_name = str(json_obj.get("name", ""))
                    tool = self.tool_manager.get_tool(tool_name)
                    if tool is None:
                        resp=f"Error: {json_obj.get('name')} tool not found"
                    else:
                        resp = tool.run(tool_input = json_obj.get("arguments"))
                except Exception as e:
                    TOOL_CALL_FAILED += 1
                    logger.error(f"Error calling tool: {e}")
                    resp=f"Error: {e}"
        return resp, tool_name
      
    def _sanity_check_code(self,code:str)-> Tuple[bool, str|None]:
        try:
            # check for syntax errors
            tree = ast.parse(code)
        except Exception as e:
            return False,str(e)
        # build parent map to determine enclosing classes for functions
        parent_map = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        for node in ast.walk(tree):
            
            if isinstance(node, ast.FunctionDef):
                # find enclosing class name if any
                class_name = None
                current_parent = parent_map.get(node)
                while current_parent is not None:
                    if isinstance(current_parent, ast.ClassDef):
                        class_name = current_parent.name
                        break
                    current_parent = parent_map.get(current_parent)
                #logger.info(f"node: {node}, name: {node.name}, type: {type(node)}, class: {class_name}")
                body = list(node.body)
                # drop leading docstring
                # Check if first statement is a docstring (string literal)
                if body and isinstance(body[0], ast.Expr):
                    expr_value = getattr(body[0], "value", None)
                    if isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
                        body = body[1:]
                #logger.info(f"body: {body}, type: {type(body)}, len: {len(body)},type of body[0]: {type(body[0])}")
                
                if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    return False, f"function {node.name} has empty body"
        return True, None
    
    def check_code_for_common_errors(self, response: Union[str, list], raw_response: str) -> str|None:
        if isinstance(response, list):
            for r in response:
                if not r.get("code"):
                    return "'code' key is missing in the response"
                is_success, error_message = self._sanity_check_code(r.get("code"))
                logger.info(f"sanity check code for {r.get('file_name')}: {is_success} {error_message}")
                if not is_success:
                    return error_message
        return "success"
                
    def __init__(self, problem_statement: str, tool_manager: ToolManager):
        self.problem_statement = problem_statement
        self.problem_statement = self.post_process_instruction()
        self.code_skeleton = self.get_code_skeleton()
        self.tool_manager = tool_manager

        self.agent_initial_solution_eval=CustomAssistantAgent(
            system_message=CreateProblemSolver.SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL.format(tools_docs=tool_manager.get_tool_docs(), format_prompt=self.RESPONSE_FORMAT_SOLUTION_EVAL_2),
            model_name=QWEN_MODEL_NAME
        )
        
    def get_code_skeleton(self) -> str:
        # Initialize the result string
        result = ""
        
        # Walk through the current directory
        for root, _, files in os.walk("."):
            for file in files:
                # Check if the file is a Python file
                if file.endswith(".py") and "solution" not in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        content = f.read()
                    # Concatenate the file name and content
                    result += f"{file}\n{{\n{content}\n}}\n\n"
        
        return result

    def post_process_instruction(self):
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
        processed_instruction = re.sub(pattern, replace_text_block, self.problem_statement, flags=re.DOTALL)
        logger.info(f"Processed instruction: {processed_instruction}")
        return processed_instruction
    
    def extract_and_write_files(self,initial_solution: str, base_dir: str = ".") -> list:
        created_files = []
        
        if not initial_solution.strip():
            logger.info("No solution content to process")
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
                if current_filename and not line.startswith('{') and not line.startswith('}'):  # Only collect content if we have a filename
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
    
    async def generate_initial_solution(self):
        logger.info("Generating initial solution")
        agent=CustomAssistantAgent(system_message=self.SYSTEM_PROMPT.format(),model_name=QWEN_MODEL_NAME)
        
        response=await agent.solve_task(self.INSTANCE_PROMPT.format(problem_statement=self.problem_statement,code_skeleton=self.code_skeleton),response_format=self.RESPONSE_FORMAT_JSON,is_json=True,regex=None,post_process_func=self.check_code_for_common_errors,max_attempts=3,is_parallel=False,disable_reset=False)
        
        if response is None:
            logger.info("Failed to generate initial solution")
            return None
        
        logger.info("Initial solution generated successfully")
        logger.info(response)
        initial_solution="\n".join([r["file_name"]+"\n"+r["code"] for r in response])
        return initial_solution
    
    async def generate_test_cases(self):
        #generate test cases.
       
        agent=CustomAssistantAgent(system_message=self.TEST_CASE_GENERATOR_SYSTEM_PROMPT.format(),model_name=QWEN_MODEL_NAME)
        response=await agent.solve_task(self.TEST_CASES_GEN_INSTANCE_PROMPT.format(problem_statement=self.problem_statement,code_skeleton=self.code_skeleton),response_format="",is_json=False,regex=None,post_process_func=self.ResponseValidator.check_syntax_error,max_attempts=10,is_parallel=False,disable_reset=False,return_type=str)
        
        if response is None:
            logger.info("Failed to generate test cases")
            return None
        logger.info(response)
        if False:
            logger.info("Now verifying the test cases...")
            agent=CustomAssistantAgent(system_message=self.TESTCASES_CHECK_PROMPT.format(),model_name=QWEN_MODEL_NAME)
            no_steps=10
            response=await agent.solve_task(self.INSTANCE_TESTCASES_CHECK_PROMPT.format(problem_statement=self.problem_statement,code_skeleton=self.code_skeleton,testcode_response=response),response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
            no_of_rechecks=1
            while True:
                while "DONE" not in response:
                    response=await agent.solve_task("continue for other test cases",response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                no_of_rechecks+=1
                if no_of_rechecks<=1:
                    response=await agent.solve_task(f"recheck all your test cases once again.. This is your attempt %{no_of_rechecks}th attempt",response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                else:
                    break
            response=await agent.solve_task(self.FINAL_OUTPUT_TEST_CHECK,response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
        
            if response is None or self.ResponseValidator.check_syntax_error(response,None)!="success":
                logger.info("Failed to verify test cases")
                return None
    
        all_test_cases=[]
        for line in response.split("\n"):
            if "def test_" in line.strip() and "(" in line.strip():
                all_test_cases.append(line.strip().split("def test_")[1].split("(")[0].strip())
        with open("test_cases.txt", "w") as f:
            f.write("\n".join(all_test_cases))
        logger.info("test_cases.txt file created successfully")
            
        return response
    
    async def solve_problem(self):
        start_time = time.time()
        
        initial_solution=await self.generate_initial_solution()
        if initial_solution is None:
            return None
        initial_solutions=[self.generate_initial_solution() for _ in range(5)]
        initial_solutions=await asyncio.gather(*initial_solutions)
        logger.info(Counter(initial_solutions))
        initial_solution=max(initial_solutions, key=initial_solutions.count)
        self.extract_and_write_files(str(initial_solution))
        await self.generate_test_cases()
        
        response=await self.agent_initial_solution_eval.solve_task(CreateProblemSolver.INSTANCE_PROMPT_INITIAL_SOLUTION_EVAL.format(problem_statement=self.problem_statement,initial_solution=initial_solution),response_format="",is_json=False,regex=None,post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
        response, tool_name=self.process_response(response)
        finish_called_earlier=True
        while True:
            if tool_name != "complete":
                logger.info(f"\n\n<yellow>⚡ Execution</yellow>")
                response=await self.agent_initial_solution_eval.solve_task(str(response),response_format="",is_json=False,regex=None,post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                response, tool_name=self.process_response(response)
            else:
                if not finish_called_earlier:
                    response=await self.agent_initial_solution_eval.solve_task("Check the problem statement and find out the cases which have not been tested yet. You must check all the mentioned scenarios (outputs, any edge cases, errors, any workflows). Create those test cases and test your solution.",response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=3,is_parallel=False,disable_reset=True,return_type=str)
                    response, tool_name=self.process_response(response)
                    finish_called_earlier=True
                    continue
                break
        logger.info(f"<yellow>Total time taken: {time.time()-start_time} seconds</yellow>")
        
        final_patch = Utils.create_final_git_patch(self.tool_manager.temp_files)
        logger.info(f"\n<yellow>😊 Generated patch with {len(final_patch)} characters...</yellow>")
        logger.info(final_patch)
        return final_patch
    
    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
        try:
            # Update to include cfg, txt, and toml files along with py files
            # Check whether ignore_files is a property of this clas
            command = f"""
            shopt -s globstar

            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            print("Generating git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            
            # output = output.stdout.decode("utf-8") + '\n' + output.stderr.decode("utf-8")
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"

class BugFixSolver:
    
    FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
    <system_prompt>
        <role>
            You are an AI assistant helping a software engineer develop pull requests, working in a Linux environment.
        </role>

        <resolution_steps>
            *Follow these steps to resolve the issue in sequence:*

            **Step 1: Explore the Repository**
            - Understand the repository structure comprehensively.
            - Identify key files and directories relevant to the problem.
            - Locate code areas related to the problem statement.
            - If there is clear search term, use str_replace_editor tool.
            - If str_replace_editor tool fails to find the relevant files several times, explore project structure and find the relevant files.

            **Step 2: Identify Test Framework and Files**
            - First, detect the project type by examining repository files (language files, dependencies).
            - Identify test directories and files using standard naming patterns (`test`, `spec`) and `find` commands.
            - Detect the testing framework by examining configuration and metadata files.
            - Find the exact command(s) used to run the tests in CI/CD, Makefiles, or docs.
            - Document these findings: project type, test framework, test file locations, and test command.

            **Step 3: Create Reproduction Script**
            
            3.1 **Analyze the Problem Statement Thoroughly:**
            - Read the problem statement carefully and extract:
            * Expected behavior (what SHOULD happen)
            * Actual behavior (what DOES happen - the bug)
            * Specific error messages or exceptions mentioned
            * Input data or scenarios that trigger the issue
            * Any mentioned test cases that fail
            
            3.2 **Create Reproduction Script:**
            - Based on project type, write a minimal script that:
            * Uses the exact inputs/scenarios from problem statement
            * Calls the relevant functions/code paths
            * Should fail in the same way as described
            - Keep it simple and focused on reproducing the issue
            
            3.3 **Validate Syntax (if Python):**
            - **If you created a Python reproduction script**, validate syntax:
                * Call: `{"name": "validate_syntax", "arguments": {"file_path": "reproduce_issue.py"}}`
            - **If syntax errors:** Fix them before running
            - **If syntax valid:** Proceed to run the script
            
            3.4 **Run and Document Initial Results:**
            - Execute the script
            - Capture the output/error
            - Document what happened vs. what was expected
            
            **Step 3.5: CRITICAL - Verify Reproduction Script Accuracy**
            ⚠️ **DO NOT SKIP THIS STEP - Incorrect reproduction wastes significant time!**
            
            **You MUST validate the reproduction script before proceeding:**
            
            3.5.1 **Use sequential_thinking to analyze the reproduction:**
            - Question: "Does my reproduction script accurately capture the problem described?"
            - Compare the error/behavior from your script with the problem statement
            - List all symptoms mentioned in the problem statement
            - Verify each symptom is reproduced by your script
            
            3.5.2 **Cross-check with existing tests (if available):**
            - If the problem statement mentions specific failing tests, run those tests
            - Compare the error from existing tests with your reproduction script error
            - Ensure they fail for the SAME reason
            
            3.5.3 **Explicit verification checklist:**
            Answer ALL these questions before proceeding:
            - ✅ Does the reproduction script produce the EXACT error message mentioned in the problem statement?
            - ✅ Does it fail in the SAME way as described (exception type, behavior, output)?
            - ✅ Are ALL issues from the problem statement reproducible (not just one)?
            - ✅ If problem statement mentions specific inputs/scenarios, did you test those?
            - ✅ Did you verify the reproduction fails BEFORE any fix (baseline)?
            
            3.5.4 **If ANY verification fails:**
            - **STOP** - Do not proceed to Step 4
            - Revise your reproduction script
            - Re-run Step 3 and 3.5 until ALL verifications pass
            
            3.5.5 **Document your verification:**
            - Explicitly state: "Reproduction verified - script accurately captures the problem"
            - List what you verified
            - Only then proceed to Step 4
            
            **Step 4: Plan Your Fix**
            - Use the "sequential_thinking" tool to brainstorm and analyze potential root causes.
            - Consider 5-7 plausible sources of the issue.
            - Narrow down to 1-2 most likely root causes.
            - Insert logging where needed to validate your hypotheses before applying code fixes.

            **Step 5: Implement Solution**
            
            5.1 **Make Code Changes**
            - Edit source code minimally and only in necessary files
            - Maintain code quality and best practices
            - Follow project coding conventions
            
            5.2 **CRITICAL: Validate Syntax Immediately**
            - **After creating or editing ANY Python file**, use validate_syntax tool:
                * Call: `validate_syntax` with `file_path` parameter
                * Example: `{"name": "validate_syntax", "arguments": {"file_path": "module/fixed_file.py"}}`
            - **If syntax errors found:**
                * Fix the syntax error immediately using str_replace_editor
                * Re-validate with validate_syntax until it passes
                * Do NOT proceed to Step 6 until syntax is valid
            - **If syntax is valid:**
                * Proceed to Step 6
            
            **Why this matters:** Syntax errors caught early prevent wasted time debugging test failures

            **Step 6: Fail-to-Pass (F2P) Validation**
            *Objective: Confirm your fix solves the stated problem.*

            6.1 **Run Your Reproduction Script**
            - Execute the reproduction script again.
            - Confirm the original error no longer occurs.
            - Verify output matches expected correct behavior.

            6.2 **Alternative Functional Testing (When Dependencies Fail)**
            - If your reproduction script fails due to missing dependencies or environment issues:
                * Create a simplified test script that validates core logic without dependencies.
                * Use mocks or stubs if necessary.
                * Test the modified functions directly.
            - This ensures your fix works even if environment constraints block full reproduction.

            6.3 **Verify F2P Success**
            - Confirm problem described in the problem statement is solved.
            - If not solved, return to Step 4 for re-planning.
            - If solved, proceed to Step 7.

            **Step 7: Pass-to-Pass (P2P) Validation**
            *Objective: Confirm your fix does NOT break any existing functionality.*

            7.1 **Run All Existing Tests (Quick Pass)**
            - Run the full automated regression test suite using the discovered test command with quiet/summary flags.
            - Purpose: Quickly identify any test failures without detailed analysis.
            - Integrate this into your validation workflow for automatic execution.
            - If the full test suite fails, I'd like to find all test files that contains the related senario and make a new script that includes all test cases from these test files.
            7.2 **Identify Failed Tests**
            - Parse the output to list failed test files/test cases.
            - If no failures, all tests pass → proceed to Step 8.
            - If failures found, continue with detailed investigation.

            7.3 **Detailed Rerun of Failed Tests**
            - Rerun each failing test file with verbose/detailed logging.
            - Capture detailed errors, stack traces, and assertion failures.
            - Document all failures for analysis.

            7.4 **Analyze Failures with Risk Assessment**
            - Use "sequential_thinking" to investigate each failure carefully.
            - **Perform risk analysis:**
                * Identify code dependencies affected by your fix
                * Determine which functionalities are most likely impacted
                * Prioritize investigation on high-risk impacted components
            - **Categorize each failure:**
                * **Regression** (caused by your changes) → Critical, must fix
                * **Pre-existing failure** (unrelated to your fix) → Document, continue
                * **Dependency/Environment issue** → Handle via alternative testing

            7.5 **Handle Dependency/Environment Failures**
            - For failures due to missing dependencies/version mismatches:
                * Verify test environment is clean and correctly configured
                * Review test scripts to confirm they accurately reflect expected behavior
                * Create alternative, standalone functional tests for the same logic without dependencies
                * Use mocks/stubs/direct calls to prove your changes are valid
            - Document any environment issues encountered.

            7.6 **Fix Regressions**
            - If your fix caused regressions:
                * Return to Step 4 to revise your solution
                * Ensure fix doesn't break existing functionality
                * Re-run Steps 6 and 7 after applying fixes

            7.7 **Create Comprehensive Test Script (If Repeated Failures)**
            
            **Trigger:** If existing test suite fails multiple times (3+ attempts) OR cannot be executed reliably
            
            **You MUST create a NEW comprehensive test script that:**
            
            7.7.1 **Covers All Relevant Test Cases:**
            - **Identify impacted areas** via risk analysis:
                * What code did you modify?
                * What functions/classes depend on your changes?
                * What downstream components might be affected?
            
            - **Include all necessary test cases:**
                * **Bug fix validation:** Tests that explicitly cover the bug scenario you fixed
                * **Regression tests:** Tests for functionality that depends on your changes
                * **Edge cases:** Boundary conditions and corner cases
                * **Integration tests:** How your fix interacts with other components
            
            7.7.2 **Write Comprehensive Test Functions:**
            - Create a new test file (e.g., `test_comprehensive_fix_validation.py`)
            - Include ALL necessary functions and imports:
                * Import the modified modules/functions
                * Import any dependencies needed (or mock them if unavailable)
                * Set up test fixtures and data
            
            - **Test structure should include:**
                * Unit tests for modified functions
                * Integration tests for component interactions
                * End-to-end tests for complete workflows (if applicable)
            
            7.7.3 **Ensure Test Quality:**
            - **Verify test environment:**
                * Clean test environment setup
                * Correct configuration
                * Consistent state between test runs
            
            - **Review test code:**
                * Tests accurately reflect expected behavior
                * Clear assertions with meaningful error messages
                * Tests are independent and can run in any order
                * Proper setup and teardown
            
            7.7.4 **Validate Syntax of Test Script:**
            - **BEFORE running tests**, validate syntax:
                * Call: `{"name": "validate_syntax", "arguments": {"file_path": "test_comprehensive_fix_validation.py"}}`
            - **If syntax errors:**
                * Fix immediately with str_replace_editor
                * Re-validate until passes
            - **If syntax valid:**
                * Proceed to execute tests
            
            7.7.5 **Execute Comprehensive Tests:**
            - Run your new comprehensive test script
            - Verify ALL tests pass
            - Document results
            
            7.7.6 **Analyze and Review:**
            - Review test results carefully for any failures
            - Investigate and debug any new failures immediately
            - Address all failures before concluding fix is stable
            - Document what was tested and results
            
            7.8 **Iteration and Completion**
            
            **Standard Iteration:**
            - **Repeat Steps 4 → 5 → 6 → 7 until:**
                * ✅ All F2P tests pass (problem solved)
                * ✅ All P2P tests pass (no regressions via existing tests OR comprehensive test script)
                * ✅ Comprehensive test script passes (if created)
                * ✅ All failures analyzed and addressed
            
            **🚨 ESCALATION: When to START OVER (Last Resort)**
            
            After each iteration, evaluate if you should continue or start over:
            
            **Consider start_over if ANY of these apply:**
            1. **Iteration Count**: You've completed 5+ iterations (Step 4→5→6→7) with no meaningful progress
            2. **Cascading Failures**: Each fix introduces 3+ new test failures
            3. **Regression Spiral**: Fixing one P2P failure consistently creates 2+ new P2P failures
            4. **Complexity Explosion**: What should be a simple fix now touches 5+ files
            5. **Wrong Root Cause**: Evidence suggests your initial root cause analysis was incorrect
            6. **Diminishing Returns**: Each iteration makes things worse, not better
            
            **If you identify need to start over:**
            1. Use sequential_thinking to analyze:
                - Why did the current approach fail?
                - What was fundamentally wrong with the approach?
                - What specific new approach would address these issues?
                - Why will the new approach succeed?
            
            2. Call start_over tool with:
                - `problem_with_old_approach`: Detailed analysis of failures
                - `new_approach_to_try`: Clear alternative strategy
            
            3. After start_over:
                - Return to Step 3 (may reuse reproduction if valid)
                - Proceed to Step 4 with new approach
                - Apply lessons learned from first attempt
            
            **Example start_over scenarios:**
            - "Modified 6 files trying to fix data validation, but each change breaks 4 new tests. Root cause is actually in data parsing, not validation. Need to start over focusing on parser logic."
            - "After 7 iterations, F2P passes but P2P creates new failures each time. Current approach modifies core class incorrectly. Need to subclass instead of modifying base class."

            **Step 8: Comprehensive Validation & Completion**

            8.1 **Final Verification Checklist**
            
            Confirm ALL of the following before completing:
            
            **F2P Validation:**
            - ✅ Problem statement fully resolved
            - ✅ Reproduction script now passes (or alternative functional tests pass)
            - ✅ All issues mentioned in problem statement are fixed
            
            **P2P Validation:**
            - ✅ Existing test suite passes (if executable)
            - ✅ OR comprehensive test script passes (if created in Step 7.7)
            - ✅ No regressions introduced by your changes
            - ✅ All impacted areas tested via risk analysis
            
            **Test Quality:**
            - ✅ If comprehensive test script created:
                * Covers bug fix validation
                * Includes regression tests for impacted areas
                * Tests edge cases and boundary conditions
                * Includes integration tests (if applicable)
                * All necessary functions and imports included
                * Test environment verified clean and consistent
            - ✅ Test code quality confirmed (accurate assertions, clear error messages)
            - ✅ All test failures analyzed and addressed
            
            **Code Quality:**
            - ✅ Edge cases handled appropriately
            - ✅ Changes minimal and focused
            - ✅ Code follows best practices
            
            8.2 **If Any Test Fails or Verification Incomplete**
            - **DO NOT proceed to complete**
            - Return to appropriate step:
                * Reproduction wrong → Step 3
                * F2P fails → Step 4 (revise fix)
                * P2P fails (regressions) → Step 4 (revise fix)
                * Test quality issues → Step 7.7 (improve comprehensive test)
            - Continue iteration until ALL verifications pass

            8.3 **Completion**
            - **Only when ALL verification items checked**, call the complete tool
            - Provide comprehensive summary including:
                * What was fixed
                * How it was validated (existing tests, comprehensive test script, or both)
                * Any test scripts created
                * Confirmation all F2P and P2P tests pass

        </resolution_steps>

        <sequential_thinking_tool_usage_guide>
            *GUIDE FOR USING "sequential_thinking" TOOL:*
            1. Use thorough thinking; set totalThoughts ≥5 for simple cases, more if multiple fixes or root causes possible.
            2. Apply the tool repeatedly as needed throughout problem solving.
            3. Run commands and gather context between thoughts.
            4. Use it to break down complex problems and ensure thorough fix plans.
        </sequential_thinking_tool_usage_guide>

        <tips>
            - Changes must be made in the project directory to meet the problem statement.
            - Follow all tool specifications strictly.
            - Use "ls" or "grep" to verify files before retrying.
            - Set environment variables for test commands as needed.
            - Avoid repeating the same tool call more than twice.
        </tips>

        <guidelines>
            1. Implement changes carefully; maintain separation of concerns.
            2. Follow senior engineer best practices.
            3. **CRITICAL - Syntax Validation:**
                - ALWAYS use validate_syntax tool IMMEDIATELY after creating or editing Python files
                - This catches syntax errors early before they cause test failures
                - If syntax validation fails, fix the syntax error before proceeding
                - Do NOT proceed to next steps if syntax errors exist
            4. **CRITICAL - Reproduction Script Verification (Step 3.5):**
                - ALWAYS verify your reproduction script accuracy before planning any fix
                - Use sequential_thinking to validate reproduction matches problem statement
                - Cross-check with existing failing tests if mentioned
                - Wrong reproduction = wasted time solving wrong problem
                - Do NOT skip Step 3.5 verification checklist
            5. Use bash tool to set test environment variables.
            6. **Critical test validation rules:**
                - ALL F2P tests must pass (problem is solved)
                - ALL P2P tests must pass (no regressions)
                - If existing test suite fails 3+ times, create comprehensive test script (Step 7.7)
                - If dependency issues occur, create alternative functional tests
                - Iterate until both F2P and P2P fully pass
                - Do not complete until all tests pass
            7. **Comprehensive Test Script (Step 7.7):**
                - Trigger: If existing tests fail repeatedly (3+ attempts) or cannot be executed reliably
                - Must cover: Bug fix validation, regression tests, edge cases, integration tests
                - Must include ALL necessary functions, imports, and test fixtures
                - Perform risk analysis to identify all impacted areas
                - Verify test environment and review test code quality
                - Execute and analyze results thoroughly
            8. Alternative testing:
                - Create minimal tests to verify logic directly without dependencies when tests fail due to env issues
            9. **START OVER Decision (Last Resort):**
                - Use start_over tool ONLY when current approach is fundamentally flawed
                - **Triggers for start_over:**
                    * 5+ iterations of Step 4→7 with no progress
                    * Cascading failures (1 fix creates 3+ new bugs)
                    * Regression spiral (fixing P2P creates 2+ new P2P failures repeatedly)
                    * Complexity explosion (simple fix touching 5+ files)
                    * Wrong root cause confirmed after multiple attempts
                - **Before calling start_over:**
                    * Use sequential_thinking to analyze why approach failed
                    * Identify specific new approach and why it will succeed
                    * Document learnings from failed approach
                - **After start_over:**
                    * Return to Step 3 with fresh perspective
                    * Apply new approach from the start
            10. **AUTOMATIC LOOP DETECTION:**
                - System automatically monitors for repetitive tool calls
                - **If ANY tool is called 5+ times with IDENTICAL arguments and results:**
                    * Automatic start_over is triggered
                    * Codebase is reset to initial state
                    * You receive notification of the reset
                    * You MUST try a fundamentally different approach
                - **This prevents infinite loops when:**
                    * Same file is viewed repeatedly without progress
                    * Same command is run repeatedly with identical output
                    * Same edit fails repeatedly in the same way
                - **After automatic reset:**
                    * Do NOT repeat the same actions
                    * Analyze why you were stuck in a loop
                    * Choose a completely different strategy
            11. Avoid tool-call repetition and file-related errors.
        </guidelines>

        <restrictions>
            - No internet access or installing packages.
            - No downloading or external API calls.
            - Cannot modify existing test files.
            - Can create new temporary test files as needed.
        </restrictions>

        <completion_criteria>
            You may call the complete tool ONLY when ALL of these are true:
            ✅ Problem fully resolved (F2P tests passed).
            ✅ No regression (P2P tests passed via existing tests OR comprehensive test script).
            ✅ If existing tests failed 3+ times, comprehensive test script created and passing (Step 7.7).
            ✅ Comprehensive test script includes: bug fix validation, regression tests, edge cases, integration tests.
            ✅ Alternative tests prove fix correctness if dependencies failed.
            ✅ Edge cases handled.
            ✅ All failures analyzed and addressed.
            ✅ Test environment verified and test code quality confirmed.
            
            If ANY fail, return to Step 4 and iterate.
        </completion_criteria>
    </system_prompt>
    """)
    
    RESPONSE_FORMAT=textwrap.dedent("""
    ===================THOUGHT
    <<your detailed thought process>>
    ===================TOOL_CALL
    {"name":"<tool_name>","arguments":{...}}
    """)

    FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
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

    <available_tools>
    {available_tools}
    </available_tools>
    
    <response_format>
    {tool_call_format}
    </response_format>
    </instruction_prompt>
    """)
    
    TIMEOUT=900
    MAX_FIX_TASK_STEPS=300
    
    def __init__(
        self,
        problem_statement:str,
        tool_manager:ToolManager,
        top_k:int=30
    ):
        self.problem_statement=problem_statement
        self.tool_manager = tool_manager
        
        # Track tool calls to detect infinite loops
        self.tool_call_history: List[Dict[str, Any]] = []
        self.max_identical_calls = 5
        
        self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
            problem_statement=self.problem_statement,
            available_tools=self.tool_manager.get_tool_docs(),
            tool_call_format=self.RESPONSE_FORMAT
        )
        
        self.agent=CustomAssistantAgent(
            system_message=self.FIX_TASK_SYSTEM_PROMPT,
            model_name=GLM_MODEL_NAME
        )
        #logger.info("system message: "+self.agent.system_message)
    
    def _track_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: str) -> bool:
        """
        Track tool calls and detect infinite loops.
        Returns True if automatic start_over should be triggered.
        """
        # Create a hashable representation of the tool call
        call_signature = {
            "tool": tool_name,
            "args": json.dumps(arguments, sort_keys=True),
            "result": result[:500]  # First 500 chars of result to avoid memory issues
        }
        
        # Add to history
        self.tool_call_history.append(call_signature)
        
        # Count identical calls in recent history (last 20 calls)
        recent_history = self.tool_call_history[-20:]
        identical_count = sum(
            1 for call in recent_history 
            if call["tool"] == tool_name 
            and call["args"] == call_signature["args"]
            and call["result"] == call_signature["result"]
        )
        
        # If we've seen this exact call 5+ times, trigger auto start_over
        if identical_count >= self.max_identical_calls:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"⚠️  INFINITE LOOP DETECTED")
            logger.warning(f"{'='*60}")
            logger.warning(f"Tool '{tool_name}' called {identical_count} times with identical arguments and results")
            logger.warning(f"Arguments: {arguments}")
            logger.warning(f"Result (truncated): {result[:200]}")
            logger.warning(f"{'='*60}\n")
            return True
        
        return False
    
    def _auto_start_over(self) -> str:
        """
        Automatically trigger start_over when infinite loop is detected.
        """
        problem_msg = (
            f"Automatic start_over triggered due to infinite loop detection.\n\n"
            f"The same tool was called {self.max_identical_calls}+ times with identical arguments and results, "
            f"indicating the agent is stuck in a repetitive loop.\n\n"
            f"Recent tool call pattern shows no progress is being made."
        )
        
        new_approach_msg = (
            f"Will restart from Step 3 with a fresh perspective.\n\n"
            f"Key changes:\n"
            f"1. Re-analyze the problem statement to ensure correct understanding\n"
            f"2. Consider alternative root causes not previously explored\n"
            f"3. Use different debugging/exploration strategies\n"
            f"4. Avoid repeating the same actions that led to the loop"
        )
        
        logger.info("\n🔄 AUTO START OVER - Resetting due to detected infinite loop\n")
        
        # Call the start_over tool
        start_over_tool = self.tool_manager.get_tool("start_over")
        if start_over_tool:
            result = start_over_tool.run({
                "problem_with_old_approach": problem_msg,
                "new_approach_to_try": new_approach_msg
            })
            
            # Clear tool call history after reset
            self.tool_call_history = []
            
            return result.tool_output
        else:
            logger.error("start_over tool not found in tool_manager")
            return "Error: Could not trigger automatic start_over"
    
    def process_response(self, response) -> Tuple[str | None, str]:
        resp = None
        tool_name = ""
        tool_call = None
        global TOOL_CALL_FAILED
        #logger.info(f"response: {response}")
        if response is None:
            logger.error("response NONE received..")
            return None, ""
        if type(response) is tuple and len(response) == 2:
            _, tool_call = response
        elif "{" in response:
            tool_call = response
        if tool_call:
            expected_tool_format = """Expected JSON format for tool calls:
{"name": "<tool_name>", "arguments": {<arguments_dict>}}

Example:
{"name": "bash", "arguments": {"command": "ls -la"}}"""
            json_obj, _, error = ProxyClient.Utils.parse_response(str(tool_call), expected_format=expected_tool_format)
            if error:
                resp=error
            elif json_obj:
                try:
                    json_obj=json.loads(json_obj)
                    logger.info("\n\n<yellow>🤖 calling tool:</yellow> " + json_obj.get("name") + " with arguments: " + json.dumps(json_obj.get("arguments")))
                    # resp = self.tool_map[json_obj.get("name")](**json_obj.get("arguments"))
                    tool_name = str(json_obj.get("name", ""))
                    tool = self.tool_manager.get_tool(tool_name)
                    if tool is None:
                        resp=f"Error: {json_obj.get('name')} tool not found"
                    else:
                        resp = tool.run(tool_input = json_obj.get("arguments"))
                        
                        # Track tool call and check for infinite loops
                        # Skip tracking for sequential_thinking and complete tools
                        if tool_name not in ["sequential_thinking", "complete", "start_over"]:
                            result_str = str(resp) if resp else ""
                            should_auto_start_over = self._track_tool_call(
                                tool_name, 
                                json_obj.get("arguments", {}), 
                                result_str
                            )
                            
                            # If infinite loop detected, auto start over
                            if should_auto_start_over:
                                auto_reset_msg = self._auto_start_over()
                                resp = (
                                    f"{auto_reset_msg}\n\n"
                                    f"⚠️ AUTOMATIC RESET PERFORMED ⚠️\n\n"
                                    f"You were stuck in an infinite loop, calling the same tool repeatedly.\n"
                                    f"The codebase has been reset to initial state.\n\n"
                                    f"Please proceed with a different approach:\n"
                                    f"1. Return to Step 3 to verify your reproduction\n"
                                    f"2. Re-analyze the root cause with fresh perspective\n"
                                    f"3. Try a fundamentally different solution strategy\n"
                                    f"4. Avoid repeating the same actions that caused the loop"
                                )
                                tool_name = "start_over"  # Mark as start_over for workflow
                                
                except Exception as e:
                    TOOL_CALL_FAILED += 1
                    logger.error(f"Error calling tool: {e}")
                    resp=f"Error: {e}"
        return resp, tool_name
        
    async def solve_problem(self):
        logger.info(f"<yellow>🗯  Starting main agent execution...</yellow>")
        
        start_time = time.time()
        logs: List[str] = []
        logs.append(f"cwd: {os.getcwd()}")
        response = await self.agent.solve_task(
            self.instruction_prompt,
            response_format="",
            is_json=False,regex=None,
            post_process_func=None,
            max_attempts=3,
            is_parallel=False,
            disable_reset=True,
            return_type=tuple[str,str]
        )
        for step in range(self.MAX_FIX_TASK_STEPS):
            resp, tool_name = self.process_response(response)
            if tool_name != "complete":
                logger.info(f"\n\n<yellow>⚡ Execution step {step + 1}/{self.MAX_FIX_TASK_STEPS}</yellow>")
                response = await self.agent.solve_task(str(resp), response_format="", is_json=False, regex=None, post_process_func=None, max_attempts=10, is_parallel=False, disable_reset=True, return_type=tuple[str,str])
            else:
                # one final check to see if no pass_to_pass test are failing..
                # files_to_test = FixTaskEnhancedToolManager.generated_test_files
                # files_to_test = [f for f in files_to_test if "/" in f]
                
                # if not files_to_test:
                #     logger.info(f"generated test files are empty, finding relevant test files..")
                #     files_to_test=await self.find_relevant_test_files()
                #     files_to_test=[f for f in files_to_test if "/" in f] if files_to_test else []
                #     logger.info(f"relevant test files: {files_to_test} found..")
                # if files_to_test:
                #     # Normalize paths and deduplicate to avoid treating ./tests/abc.py and tests/abc.py as different
                #     if st.get("status")=="success":
                #         logger.info(f"switched to initial_commit")
                #     else:
                #         logger.error(f"initial commit switching failed {st}")
                #     initial_test_response=FixTaskEnhancedToolManager.run_repo_tests(list(set(os.path.normpath(f) for f in files_to_test)))
                #     st=restore_stashed_changes(".",0,False)
                #     if st.get("status")=="success":
                #         logger.info(f"restored to working state..")
                #     else:
                #         logger.error(f"stashed changes restoration failed {st}")
                #     # failures=FixTaskEnhancedToolManager.parse_run_repo_tests_response(repo_test_response,initial_test_response)
                    
                #     logger.info(f"files_to_test: {files_to_test}")
                #     logger.info(f"repo_test_response: {repo_test_response}")
                #     logger.info(f"failures: {failures}")
                #     if len(failures)>0:
                #         logger.info(f"total {len(failures)} failed tests detected...")
                #         repo_failures="Your fix has broken some tests. Please fix them and then call the finish tool to finish the task.\n"+("\n\n===============================================\n\n".join(failures))
                #         response=await self.agent.solve_task(repo_failures,response_format="",is_json=False,regex=None,post_process_func=None,max_attempts=10,is_parallel=False,disable_reset=True,return_type=Union[tuple[str,str],str])
                #         continue
                break
            
        # final_patch=FixTaskEnhancedToolManager.get_final_git_patch(initial_checkpoint="initial_commit")
        # if not DISABLE_TEST_FILE_REMOVAL:
        #     FixTaskEnhancedToolManager.remove_any_generated_test_files()
        final_patch = Utils.create_final_git_patch(self.tool_manager.temp_files)
        
        logger.info(f"\n<yellow>😊 Generated patch with {len(final_patch)} characters...</yellow>")
        logger.info(final_patch)
        return final_patch

class ProblemTypeClassifier:
    
    PROBLEM_TYPE_CREATE="CREATE"
    PROBLEM_TYPE_FIX="FIX"
    
    SYSTEM_PROMPT=textwrap.dedent("""
    You are the problem type checker that will categories problem type into:

    1. CREATE: If the problem statement is about creating a new functionality from scratch.
    2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

    Only respond with the "FIX" or "CREATE".
    """)
    
    @classmethod
    def get_directory_tree(cls,start_path: str = '.') -> str:

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
    
    @classmethod
    async def check_problem_type(cls, problem_statement: str) -> str:
        system_message = textwrap.dedent("""
            You are the problem type checker that will categories problem type into:

            1. CREATE: If the problem statement is about creating a new functionality from scratch. The codebase shared would be very small with no more than few files.
            2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase. Codebase for this **MUST contain multiple files and directories**.

            Only respond with the "FIX" or "CREATE". Your response cannot contain multiple THOUGHT or TOOL_CALL sections.
            """)
        instance_prompt = f"{problem_statement}\n# Project Tree Structure: \n{cls.get_directory_tree()[:10000]}..."
        
        agent = CustomAssistantAgent(
            agent_name="problem_type_classifier_agent",
            model_name=GLM_MODEL_NAME,
            system_message=system_message
        )
        response = await agent.solve_task(
            instance_prompt,
            response_format="=======PROBLEM_TYPE\n<<problem type>>",
            is_json=False,
            regex=None,
            post_process_func=None,
            max_attempts=10,
            is_parallel=False,
            disable_reset=True,
            return_type=Union[tuple[str,str],str]
        )
        
        while True:
            if isinstance(response, tuple) and len(response) == 2 and isinstance(response[1], str):
                if response[1].strip() == "FIX":
                    return cls.PROBLEM_TYPE_FIX
                elif response[1].strip() == "CREATE":
                    return cls.PROBLEM_TYPE_CREATE
            elif isinstance(response, str):
                if response.strip() == "FIX":
                    return cls.PROBLEM_TYPE_FIX
                elif response.strip() == "CREATE":
                    return cls.PROBLEM_TYPE_CREATE
            response = await agent.solve_task("Invalid response, please respond problem_type with the 'FIX' or 'CREATE'.",response_format="===================THOUGHT\n<<your thought>>\n===================PROBLEM_TYPE\n<<problem type>>", is_json=False, regex=None, post_process_func=None, max_attempts=10, is_parallel=False, disable_reset=True, return_type=Union[tuple[str,str], str])
            logger.info("<blue>classifier response</blue>:\n{response}")

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
    def get_error_counter(cls) -> dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }
          
    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 3) -> dict | None:
        
        messages = cast(list[Types.ProxyMessage], [
            {"role": "system", "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role": "user", "content": json_string}
        ])
        
        try:
            response = cls.make_request(messages)
            response = response.replace('```json','').strip('```')
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e}, trying again..[{attempt}]")
            attempt -= 1
            if attempt <= 0:
                return None
            return cls.fix_json_string_with_llm(json_string, attempt)
        except Exception as e:
            logger.error(f"❌ Error fixing json string: {e}, trying again..[{attempt}]")
            attempt -= 1
            if attempt <= 0:
                return None
            return cls.fix_json_string_with_llm(json_string, attempt)
            
    @classmethod
    def make_request(cls, messages: list[Types.ProxyMessage], model: Types.Model = KIMI_MODEL_NAME, temperature: float=0.0, attempt: int=5) -> str:
        
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"

        request_data = {
            "run_id": RUN_ID,
            "messages": messages,
            "temperature": temperature,
            "model": model
        }

        headers = {
            "Content-Type": "application/json"
        }
        
        for retry_attempt in range(attempt + 1):
            try:
                response = requests.post(url, json=request_data, timeout=120, headers=headers)
                logger.info(f"<green>📡 [INFERENCE] HTTP {response.status_code} from {url} ({len(response.content)} bytes)</green>")
                response.raise_for_status()
                
                # Process response - if this fails, we'll retry
                try:
                    response_json = response.json()
                except JSONDecodeError as e:
                    if retry_attempt < attempt:
                        sleep_time = 2 ** retry_attempt
                        logger.error(f"❌ Invalid JSON response error, retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                    return f"ERROR: Invalid JSON response for model [{model}]"
                
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
                    
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    if retry_attempt < attempt:
                        sleep_time = 2 ** retry_attempt
                        logger.error(f"❌ Invalid response error, retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                    return f"ERROR: Invalid response structure for model [{model}]"
                except Exception as e:
                    if retry_attempt < attempt:
                        sleep_time = 2 ** retry_attempt
                        logger.error(f"❌ Unexpected error, retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                    return f"ERROR: Unexpected error for model [{model}]"
                
            except requests.exceptions.Timeout:
                if retry_attempt < attempt:
                    sleep_time = 2 ** retry_attempt
                    logger.error(f"❌ Timeout error, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return f"ERROR: Request timeout for model [{model}]"
                
            except requests.exceptions.ConnectionError as e:
                if retry_attempt < attempt:
                    sleep_time = 2 ** retry_attempt
                    logger.error(f"❌ Connection error, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return f"ERROR: Connection failed for model [{model}]"
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                # Retry for 500 (Internal Server Error) or 504 (Gateway Timeout)
                if  retry_attempt < attempt:
                    sleep_time = 2 ** retry_attempt
                    # if status_code in [500, 504]
                    logger.error(f"❌ Http {status_code} error, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return f"ERROR: HTTP error {status_code} for model [{model}]"
                
            except requests.exceptions.RequestException as e:
                if retry_attempt < attempt:
                    sleep_time = 2 ** retry_attempt
                    logger.error(f"❌ Request failed, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return f"ERROR: Request failed for model [{model}]"
        
        # If we exhausted all retries
        return f"ERROR: Max retries exceeded for model [{model}]"
    
    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
class ToolManager:
    """Manager class for all tools in the agent system."""
    
    class Utils:
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
        ) -> str:
            """Run the tool."""

            try:
                self._validate_tool_input(tool_input)
                result = self.run_impl(tool_input)
                tool_output = result.tool_output
            except Exception as exc:
                tool_output = "Invalid tool input: " + exc.__str__()
               
            return tool_output

        def get_tool_start_message(self, tool_input: Types.ToolInputSchema) -> str:
            return f"Calling tool '{self.name}'"

        def run_impl(
            self,
            tool_input: dict[str, Any],
        ) -> Types.ToolImplOutput:
            """Subclasses should implement this."""
            raise NotImplementedError()

        def _validate_tool_input(self, tool_input: dict[str, Any]):
            """Validates the tool input."""
            if Utils.validate_json_schema(tool_input, self.input_schema):
                return
            else:
                raise ValueError("Tool input does not match schema.")

    def __init__(self):
        """Initialize the tool manager."""
        self._tools: Dict[str, ToolManager.LLMTool] = {}
        self._register_default_tools()
        self.temp_files: List[str] = []

    def add_temp_file(self, file_path: str):
        self.temp_files.append(file_path)
    
    def _register_default_tools(self):
        """Register all default tools."""
        # Register BashTool
        self.register_tool(ToolManager.BashTool(tool_manager=self))
        # Register CompleteTool
        self.register_tool(ToolManager.CompleteTool())
        # Register SequentialThinkingTool
        self.register_tool(ToolManager.SequentialThinkingTool(tool_manager=self))
        # Register StartOverTool
        self.register_tool(ToolManager.StartOverTool(tool_manager=self))
        # Register SyntaxValidatorTool
        self.register_tool(ToolManager.SyntaxValidatorTool(tool_manager=self))
        # Register StrReplaceEditorTool
        self.register_tool(ToolManager.StrReplaceEditorTool(tool_manager=self))
    
    def register_tool(self, tool: 'LLMTool'):
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional['LLMTool']:
        return self._tools.get(name)
    
    def get_tool_docs(self, tool_choice: dict[str, str] | None = None) -> str:
        _docs: list[str] = []
        for tool in self._tools.values():
            _tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            _docs.append(json.dumps(_tool, indent=2))
        return textwrap.dedent("\n\n".join(_docs))
    
    class BashTool(LLMTool):
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

        def run_impl(
            self,
            tool_input: Dict[str, Any],
        ) -> Types.ToolImplOutput:

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
            return f"Executing bash command: {tool_input['command']}"

    class CompleteTool(LLMTool):
        """The model should call this tool when it is done with the task."""
        name = "complete"

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
            border = "─" * 100

            return textwrap.dedent(f"""
┌{border}
│ {header}
├{border}
│ {thought}
└{border}
            """)

        def run_impl(
            self,
            tool_input: Dict[str, Any],
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
            thought_number = tool_input.get("thoughtNumber", "?")
            total_thoughts = tool_input.get("totalThoughts", "?")
            return f"Processing sequential thought {thought_number}/{total_thoughts}"

    class StartOverTool(LLMTool):
        """Tool to revert all changes and start fresh with a new approach."""
        name = "start_over"
        
        description = textwrap.dedent("""
            Revert all changes to the codebase and start over with a new approach.
            
            ⚠️ **USE THIS TOOL ONLY AS A LAST RESORT** ⚠️
            
            When to use:
            1. **Cascading Failures**: Your fix created 3+ new bugs or failures
            2. **Iteration Deadlock**: After 5+ iterations of Step 4→5→6→7 with no progress
            3. **Regression Spiral**: Fixing one P2P failure consistently causes 2+ new P2P failures
            4. **Complexity Explosion**: A simple fix has grown to touch 5+ files with no end in sight
            5. **Wrong Root Cause**: After multiple attempts, clear the fix doesn't address the core issue
            6. **Fundamental Flaw**: Discovered architecture/design incompatibility with current approach
            
            When NOT to use:
            - Single syntax error or test failure (just fix it)
            - First or second attempt at solution
            - Minor tweaks needed
            - Haven't tried debugging thoroughly
            
            This tool executes `git reset --hard` to revert ALL changes.
            You will lose all current modifications and return to initial state.
            Only use when you have a clear, better alternative approach.
        """)
        
        input_schema = {
            "type": "object",
            "properties": {
                "problem_with_old_approach": {
                    "type": "string",
                    "description": "Detailed explanation of what you tried and the key issues encountered. Must be specific about why the approach failed (e.g., 'Modified 6 files but each fix created 2 new P2P failures, indicating wrong root cause').",
                },
                "new_approach_to_try": {
                    "type": "string",
                    "description": "Clear description of the new approach and how it addresses the failures of the old approach. Must explain why this approach will succeed where the previous one failed.",
                },
            },
            "required": ["problem_with_old_approach", "new_approach_to_try"],
        }
        
        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager
        
        def run_impl(
            self,
            tool_input: dict[str, Any],
        ) -> Types.ToolImplOutput:
            problem_with_old_approach = tool_input["problem_with_old_approach"]
            new_approach_to_try = tool_input["new_approach_to_try"]
            
            try:
                logger.info("\n" + "="*60)
                logger.info("🔄 START OVER - REVERTING ALL CHANGES")
                logger.info("="*60)
                logger.info(f"\n<yellow>Problem with old approach:</yellow>\n{problem_with_old_approach}")
                logger.info(f"\n<green>New approach to try:</green>\n{new_approach_to_try}")
                logger.info("="*60 + "\n")
                
                # Execute git reset --hard to revert all changes
                result = subprocess.run(
                    ["git", "reset", "--hard"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    success_msg = "✅ Codebase successfully reverted to initial state.\n\n"
                    success_msg += "All changes have been discarded.\n"
                    success_msg += "You can now start fresh with your new approach.\n\n"
                    success_msg += "**Next Steps:**\n"
                    success_msg += "1. Return to Step 3 to create a new reproduction script (if needed)\n"
                    success_msg += "2. Proceed to Step 4 with your new approach\n"
                    success_msg += "3. Implement the solution following the new strategy\n"
                    
                    logger.info(f"<green>Git reset output: {result.stdout}</green>")
                    
                    # Clear temp files list since we've reverted everything
                    if self.tool_manager:
                        self.tool_manager.temp_files = []
                    
                    return Types.ToolImplOutput(
                        tool_output=success_msg,
                        tool_result_message="Codebase reverted successfully",
                        auxiliary_data={
                            "success": True,
                            "reverted": True,
                            "problem_with_old_approach": problem_with_old_approach,
                            "new_approach_to_try": new_approach_to_try
                        }
                    )
                else:
                    error_msg = f"❌ Failed to revert codebase.\n\n"
                    error_msg += f"Git reset error: {result.stderr}\n"
                    error_msg += f"Return code: {result.returncode}\n\n"
                    error_msg += "You may need to manually check the repository state."
                    
                    logger.error(f"Git reset failed: {result.stderr}")
                    
                    return Types.ToolImplOutput(
                        tool_output=error_msg,
                        tool_result_message=f"Git reset failed: {result.stderr}",
                        auxiliary_data={"success": False, "reverted": False}
                    )
                    
            except Exception as e:
                error_msg = f"❌ Unexpected error during start_over: {str(e)}\n"
                error_msg += "Codebase state may be uncertain. Please verify manually."
                
                logger.error(f"start_over tool error: {str(e)}")
                
                return Types.ToolImplOutput(
                    tool_output=error_msg,
                    tool_result_message=f"Error: {str(e)}",
                    auxiliary_data={"success": False, "error": str(e)}
                )
        
        def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
            return "⚠️ Reverting all changes and starting over with new approach..."
    
    class SyntaxValidatorTool(LLMTool):
        """Tool to validate Python syntax in files."""
        name = "validate_syntax"
        
        description = textwrap.dedent("""
            Validate Python syntax in a file using ast.parse()
            * Use this tool IMMEDIATELY after creating or editing Python files
            * Detects syntax errors before they cause issues
            * Provides detailed error messages with line numbers
            * Supports Python files only (.py extension)
            
            When to use:
            - After creating a new Python file with str_replace_editor
            - After making significant edits to Python files
            - Before running tests to catch syntax errors early
            
            Returns:
            - Success message if syntax is valid
            - Detailed error with line number and description if syntax is invalid
        """)
        
        input_schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file to validate (.py extension required)",
                },
            },
            "required": ["file_path"],
        }
        
        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager
        
        def run_impl(
            self,
            tool_input: dict[str, Any],
        ) -> Types.ToolImplOutput:
            file_path = tool_input["file_path"]
            
            try:
                # Convert to Path object
                path = Path(file_path).resolve()
                
                # Security check - ensure path is within current working directory
                current_dir = Path.cwd()
                if not ToolManager.Utils.is_path_in_directory(current_dir, path):
                    return Types.ToolImplOutput(
                        tool_output=f"Error: Path {path} is outside the current working directory",
                        tool_result_message=f"Path {path} is outside the current working directory",
                        auxiliary_data={"success": False, "valid_syntax": False}
                    )
                
                # Check if file exists
                if not path.exists():
                    return Types.ToolImplOutput(
                        tool_output=f"Error: File {file_path} does not exist",
                        tool_result_message=f"File {file_path} not found",
                        auxiliary_data={"success": False, "valid_syntax": False}
                    )
                
                # Check if it's a Python file
                if not str(path).endswith('.py'):
                    return Types.ToolImplOutput(
                        tool_output=f"Skipped: {file_path} is not a Python file. Syntax validation only supports .py files.",
                        tool_result_message="Not a Python file",
                        auxiliary_data={"success": True, "valid_syntax": True, "skipped": True}
                    )
                
                # Read file content
                try:
                    content = path.read_text(encoding='utf-8')
                except Exception as e:
                    return Types.ToolImplOutput(
                        tool_output=f"Error reading file {file_path}: {str(e)}",
                        tool_result_message=f"Failed to read file: {str(e)}",
                        auxiliary_data={"success": False, "valid_syntax": False}
                    )
                
                # Validate syntax using ast.parse
                try:
                    ast.parse(content)
                    success_msg = f"✅ Syntax validation passed for {file_path}\n"
                    success_msg += f"File contains valid Python syntax ({len(content.splitlines())} lines)"
                    
                    return Types.ToolImplOutput(
                        tool_output=success_msg,
                        tool_result_message=f"Syntax valid for {file_path}",
                        auxiliary_data={"success": True, "valid_syntax": True, "lines": len(content.splitlines())}
                    )
                    
                except SyntaxError as e:
                    # Detailed syntax error information
                    error_msg = f"❌ Syntax Error in {file_path}\n\n"
                    error_msg += f"Error: {e.msg}\n"
                    error_msg += f"Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}\n"
                    if e.offset:
                        error_msg += " " * (e.offset - 1) + "^\n"
                    error_msg += f"\nLocation: Line {e.lineno}, Column {e.offset}\n"
                    error_msg += f"\nPlease fix the syntax error before proceeding."
                    
                    return Types.ToolImplOutput(
                        tool_output=error_msg,
                        tool_result_message=f"Syntax error at line {e.lineno}",
                        auxiliary_data={
                            "success": False,
                            "valid_syntax": False,
                            "error": e.msg,
                            "line": e.lineno,
                            "offset": e.offset
                        }
                    )
                    
            except Exception as e:
                return Types.ToolImplOutput(
                    tool_output=f"Unexpected error during syntax validation: {str(e)}",
                    tool_result_message=f"Validation failed: {str(e)}",
                    auxiliary_data={"success": False, "valid_syntax": False}
                )
        
        def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
            return f"Validating syntax for {tool_input.get('file_path', 'unknown')}"
    
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
                    "description": "Optional parameter of 'view' command when 'path' points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting '[start_line, -1]' shows all lines from 'start_line' to the end of the file.",
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
                if not ToolManager.Utils.is_path_in_directory(current_dir, _ws_path):
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
            file_content = ToolManager.Utils.maybe_truncate(file_content)
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
            command = tool_input.get("command", "unknown")
            path = tool_input.get("path", "unknown")
            return f"Executing {command} on {path}"

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """
    Main agent function that generates a git patch to solve the problem statement.
    
    Args:
        input_dict: Dictionary containing the problem statement and other parameters
        repo_dir: Path to the repository directory
        test_mode: Whether to run in test mode
    
    Returns:
        str: Git patch as a string
    """
    global DEFAULT_PROXY_URL, REPO_DIR, DEFAULT_TIMEOUT, MAX_TEST_PATCH_TIMEOUT, RUN_ID
    
    REPO_DIR = repo_dir
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    Utils.ensure_git_initialize()
    Utils.set_env_for_agent()
    
    logger.info(f"<blue>📁 Repository:</blue> {repo_dir}")

    problem_statement = input_dict.get("problem_statement", "")
    # Check problem type first
    problem_type = asyncio.run(ProblemTypeClassifier.check_problem_type(problem_statement))
    # problem_type = "FIX"
    
    tool_manager = ToolManager()
    if problem_type == ProblemTypeClassifier.PROBLEM_TYPE_FIX:
        fix_prb_task = BugFixSolver(problem_statement, tool_manager).solve_problem()
        try:
            result = asyncio.run(asyncio.wait_for(fix_prb_task, timeout=2280))
        except asyncio.TimeoutError as e:
            logger.error(f"Timed out after 2280 seconds..")
            result = Utils.create_final_git_patch(tool_manager.temp_files)
        except Exception as e:
            logger.error(f"Error: {e}")
            result = Utils.create_final_git_patch(tool_manager.temp_files)
            
    else:
        # Use traditional approach for CREATE tasks
        create_problem_task = CreateProblemSolver(problem_statement, tool_manager).solve_problem()
        try:
            result = asyncio.run(asyncio.wait_for(create_problem_task, timeout=2280))
        except asyncio.TimeoutError as e:
            logger.error(f"Timed out after 60 seconds..")
            result = Utils.create_final_git_patch(tool_manager.temp_files)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            result = Utils.create_final_git_patch(tool_manager.temp_files)
            
    os.system("git reset --hard")  
    logger.info(f"🌕🌕🌕 Final result: {result}")
    return result
