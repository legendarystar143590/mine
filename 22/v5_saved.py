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

GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"

JSON_LLM_USED = 0
JSON_LITERAL_USED = 0
TOO_MANY_SECTIONS_FOUND = 0
MAX_EMBED_TOKENS = 128000
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
    def error(cls, message):
        """Log error message."""
        cls._log('ERROR', message)

class Types:
    """Container for all custom data type classes used by the agent."""

    @dataclass
    class ToolImplOutput:
        """Output from an LLM tool implementation."""
        tool_output: str
        tool_result_message: str
        auxiliary_data: dict[str, Any] = field(default_factory=dict)
        
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

    class ToolError(Exception):
        def __init__(self, message: str):
            self.message = message
            super().__init__(message)

        def __str__(self):
            return self.message

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

class ToolUtils:
    """Shared utilities for all LLMTool classes following DRY principles."""
    
    @staticmethod
    def run_subprocess(
        command: List[str],
        timeout: int = 60,
        cwd: Optional[str] = None
    ) -> Types.ToolImplOutput:
        """
        Execute a subprocess command and return standardized output.
        
        Args:
            command: Command to execute as list (e.g., ['python', 'file.py'])
            timeout: Timeout in seconds
            cwd: Working directory for command execution
            
        Returns:
            ToolImplOutput with execution results
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=cwd
            )
            
            if result.returncode != 0:
                output = f"Error running command: {result.stderr}\n"
                logger.error(output)
                return Types.ToolImplOutput(
                    output,
                    "Command execution failed",
                    {"success": False, "returncode": result.returncode}
                )
            
            output = f"{result.stdout}\n"
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            logger.info(f"Command execution output: {output}")
            return Types.ToolImplOutput(
                output,
                "Command executed successfully",
                {"success": True, "returncode": 0}
            )
            
        except subprocess.TimeoutExpired:
            return Types.ToolImplOutput(
                f"Error: Command execution timed out after {timeout} seconds",
                "Execution timeout",
                {"success": False, "error": "timeout"}
            )
        except Exception as e:
            return Types.ToolImplOutput(
                f"Error executing command: {e}",
                "Execution failed",
                {"success": False, "error": str(e)}
            )
    
    @staticmethod
    def validate_file_exists(file_path: str) -> Optional[Types.ToolImplOutput]:
        """
        Check if file exists, return error response if not.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            ToolImplOutput with error if file doesn't exist, None if it exists
        """
        if not os.path.exists(file_path):
            return Types.ToolImplOutput(
                f"Error: file '{file_path}' does not exist.",
                "File not found",
                {"success": False, "error": "file_not_found"}
            )
        return None
    
    @staticmethod
    def validate_syntax(
        content: str,
        file_path: str,
        language: str = "python"
    ) -> Optional[Types.ToolImplOutput]:
        """
        Validate code syntax, return error response if invalid.
        
        Args:
            content: Code content to validate
            file_path: File path for error reporting
            language: Programming language (currently only 'python' supported)
            
        Returns:
            ToolImplOutput with error if syntax is invalid, None if valid
        """
        if language == "python":
            try:
                ast.parse(content, filename=file_path)
                return None
            except SyntaxError as e:
                error_msg = f"Syntax error: {e}"
                logger.error(error_msg)
                return Types.ToolImplOutput(
                    f"Error: {error_msg}",
                    "Syntax error in code",
                    {"success": False, "error": error_msg}
                )
        return None
    
    @staticmethod
    def check_dependencies(content: str, file_path: str) -> Tuple[bool, Set[str]]:
        """
        Check for disallowed third-party modules in code.
        
        Args:
            content: Code content to check
            file_path: File path for parsing
            
        Returns:
            Tuple of (has_disallowed, set_of_disallowed_modules)
        """
        try:
            tree = ast.parse(content, filename=file_path)
            disallowed_modules = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        mod = node.module.split(".")[0]
                    else:
                        mod = node.names[0].name.split(".")[0]

                    # Skip built-in modules
                    if mod in sys.builtin_module_names:
                        continue

                    # Skip relative imports
                    if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                        continue

                    # Check if it's a local module
                    cwd = os.getcwd()
                    local_file = os.path.join(cwd, f"{mod}.py")
                    local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                    local_pkg_dir = os.path.join(cwd, mod)
                    lib_dir = os.path.join(cwd, 'lib')
                    lib_file = os.path.join(lib_dir, f"{mod}.py")
                    lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                    lib_pkg_dir = os.path.join(lib_dir, mod)

                    if (os.path.isfile(local_file) or os.path.isfile(local_pkg_init) or 
                        os.path.isdir(local_pkg_dir) or os.path.isfile(lib_file) or 
                        os.path.isfile(lib_pkg_init) or os.path.isdir(lib_pkg_dir)):
                        continue

                    disallowed_modules.add(mod)
            
            return len(disallowed_modules) > 0, disallowed_modules
            
        except Exception as e:
            logger.warning(f"Could not check dependencies: {e}")
            return False, set()
    
    @staticmethod
    def error_response(
        message: str,
        summary: str = None,
        error_type: str = "error",
        **kwargs
    ) -> Types.ToolImplOutput:
        """
        Create standardized error response.
        
        Args:
            message: Error message to display
            summary: Short summary for tool_result_message
            error_type: Type of error for auxiliary_data
            **kwargs: Additional data for auxiliary_data
            
        Returns:
            Standardized error ToolImplOutput
        """
        aux_data = {"success": False, "error": error_type}
        aux_data.update(kwargs)
        
        return Types.ToolImplOutput(
            message,
            summary or "Operation failed",
            aux_data
        )
    
    @staticmethod
    def success_response(
        message: str,
        summary: str = None,
        **kwargs
    ) -> Types.ToolImplOutput:
        """
        Create standardized success response.
        
        Args:
            message: Success message to display
            summary: Short summary for tool_result_message
            **kwargs: Additional data for auxiliary_data
            
        Returns:
            Standardized success ToolImplOutput
        """
        aux_data = {"success": True}
        aux_data.update(kwargs)
        
        return Types.ToolImplOutput(
            message,
            summary or "Operation successful",
            aux_data
        )

class textwrap:
    """
    🧠 SMART INDENTATION HELPER CLASS
    
    This class provides all indentation-related functionality in one organized place.
    Think of it as your indentation toolkit!
    """
    
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

    @classmethod
    def delete_files_from_repo(cls, file_list: list[str]) -> str:
        """
        Delete specified files from the repository.
        
        Args:
            file_list: List of file paths to delete (relative to repo root)
            
        Returns:
            String with deletion results
        """
        if not file_list:
            return "No files to delete"
        
        deleted_files = []
        failed_files = []
        
        for file_path in file_list:
            try:
                # Convert to absolute path
                abs_path = Path(file_path).resolve()
                
                # Check if file exists
                if abs_path.exists():
                    # Delete the file
                    abs_path.unlink()
                    deleted_files.append(str(file_path))
                    logger.info(f"<green>🗑️ Deleted file: {file_path}</green>")
                else:
                    logger.warning(f"<yellow>⚠️ File not found: {file_path}</yellow>")
                    failed_files.append(f"{file_path} (not found)")
                    
            except Exception as e:
                error_msg = f"{file_path} ({str(e)})"
                failed_files.append(error_msg)
                logger.error(f"<red>❌ Failed to delete {file_path}: {e}</red>")
        
        # Generate result summary
        result_parts = []
        if deleted_files:
            result_parts.append(f"✅ Successfully deleted {len(deleted_files)} files: {', '.join(deleted_files)}")
        if failed_files:
            result_parts.append(f"❌ Failed to delete {len(failed_files)} files: {', '.join(failed_files)}")
        
        return "\n".join(result_parts) if result_parts else "No files processed"
    @staticmethod
    def set_env_for_agent():
        if os.getcwd() not in os.environ.get("PYTHONPATH",""):
            os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
        if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
            os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"
    
    @classmethod
    def create_final_git_patch(cls, temp_files: list[str] = []) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        if len(temp_files) > 0:
            cls.delete_files_from_repo(temp_files)

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
    
    @staticmethod
    def parse_test_results(test_command: str, test_output: str) -> Dict[str, Any]:
        """
        Parse test results from various testing frameworks and extract failed test cases.
        
        Args:
            test_command: The command used to run tests (e.g., "pytest", "python -m unittest")
            test_output: The output from running the tests
            
        Returns:
            Dictionary containing:
            - framework: Detected testing framework
            - total_tests: Total number of tests run
            - passed: Number of passed tests
            - failed: Number of failed tests
            - errors: Number of errors
            - skipped: Number of skipped tests
            - failed_tests: List of failed test details with names and error messages
        """
        result = {
            "framework": "unknown",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "failed_tests": []
        }
        
        # Detect framework from command (for informational purposes)
        command_lower = test_command.lower()
        
        if "pytest" in command_lower or "py.test" in command_lower:
            result["framework"] = "pytest"
        elif "unittest" in command_lower or ("python" in command_lower and "-m unittest" in command_lower):
            result["framework"] = "unittest"
        elif "jest" in command_lower or "npm test" in command_lower or "yarn test" in command_lower:
            result["framework"] = "jest"
        elif "mocha" in command_lower:
            result["framework"] = "mocha"
        elif "go test" in command_lower:
            result["framework"] = "go_test"
        elif "cargo test" in command_lower or "rust" in command_lower:
            result["framework"] = "cargo_test"
        else:
            # Try to detect from output patterns
            output_lower = test_output.lower()
            if "pytest" in output_lower or "test session starts" in output_lower:
                result["framework"] = "pytest"
            elif "ran " in output_lower and " test" in output_lower:
                result["framework"] = "unittest"
            elif "tests:" in output_lower and "total" in output_lower:
                result["framework"] = "jest"
            elif "passing" in output_lower or "failing" in output_lower:
                result["framework"] = "mocha"
            elif "--- fail:" in output_lower:
                result["framework"] = "go_test"
            elif "test result:" in output_lower:
                result["framework"] = "cargo_test"
            else:
                result["framework"] = "generic"
        
        # Use generic parser for all frameworks
        return Utils._parse_generic_output(test_output, result)
    
    @staticmethod
    def _parse_generic_output(output: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generic parser for all test frameworks - looks for result status keywords."""
        lines = output.split('\n')
        
        # Pattern 1: Look for summary lines with counts
        # Examples: "5 passed, 2 failed", "X passed; Y failed", "Tests: 5 failed, 10 passed"
        for line in lines:
            line_lower = line.lower()
            
            # Extract passed count
            passed_patterns = [
                r'(\d+)\s+passed',           # "5 passed"
                r'(\d+)\s+passing',          # "5 passing" (mocha)
                r'ok\s+(\d+)',               # "ok 5" (TAP)
            ]
            for pattern in passed_patterns:
                match = re.search(pattern, line_lower)
                if match and result["passed"] == 0:
                    result["passed"] = int(match.group(1))
                    break
            
            # Extract failed count
            failed_patterns = [
                r'(\d+)\s+failed',           # "2 failed"
                r'(\d+)\s+failing',          # "2 failing" (mocha)
                r'failures?=(\d+)',          # "failures=2" (unittest)
                r'not ok\s+(\d+)',           # "not ok 2" (TAP)
            ]
            for pattern in failed_patterns:
                match = re.search(pattern, line_lower)
                if match and result["failed"] == 0:
                    result["failed"] = int(match.group(1))
                    break
            
            # Extract error count
            error_patterns = [
                r'(\d+)\s+errors?',          # "1 error"
                r'errors?=(\d+)',            # "errors=1"
            ]
            for pattern in error_patterns:
                match = re.search(pattern, line_lower)
                if match and result["errors"] == 0:
                    result["errors"] = int(match.group(1))
                    break
            
            # Extract skipped count
            skipped_patterns = [
                r'(\d+)\s+skipped',          # "3 skipped"
                r'(\d+)\s+pending',          # "3 pending" (mocha)
                r'(\d+)\s+ignored',          # "3 ignored" (rust)
            ]
            for pattern in skipped_patterns:
                match = re.search(pattern, line_lower)
                if match and result["skipped"] == 0:
                    result["skipped"] = int(match.group(1))
                    break
            
            # Extract total count
            total_patterns = [
                r'ran\s+(\d+)\s+tests?',     # "Ran 5 tests" (unittest)
                r'(\d+)\s+total',            # "5 total" (jest)
                r'(\d+)\s+examples?',        # "5 examples" (rspec)
            ]
            for pattern in total_patterns:
                match = re.search(pattern, line_lower)
                if match and result["total_tests"] == 0:
                    result["total_tests"] = int(match.group(1))
                    break
        
        # Pattern 2: Parse individual test result lines (only if no summary found)
        # Look for lines with test status at the end (but not in file paths)
        # Supported patterns:
        #   - pytest: "test_file.py::test_name PASSED [ 50%]"
        #   - unittest dot notation: "..F.E.."
        #   - mocha: "  ✓ test name" or "  1) test name"
        #   - go test: "--- FAIL: TestName"
        #   - rust: "test test_name ... ok/FAILED"
        #   - TAP: "ok 1 - test description" or "not ok 1 - test description"
        #   - generic ok: "test_name ... ok" or "test_name ok"
        
        counted_in_summary = result["passed"] > 0 or result["failed"] > 0
        
        for line in lines:
            # Skip individual counting if we already got counts from summary
            if counted_in_summary:
                break
                
            # pytest/unittest style
            match = re.search(r'^(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED)\s*(\[|$)', line)
            if match:
                status = match.group(2)
                if status == 'PASSED':
                    result["passed"] += 1
                elif status == 'FAILED':
                    result["failed"] += 1
                elif status == 'ERROR':
                    result["errors"] += 1
                elif status == 'SKIPPED':
                    result["skipped"] += 1
            
            # Handle "ok" as pass indicator (TAP, Rust, Go, etc.)
            # Pattern: "ok 1 - test name" or "test name ... ok" or "test name ok"
            line_lower = line.lower()
            if not match:  # Only check if not already matched above
                # TAP format: "ok 1 - test description" or "ok 1 test description"
                ok_tap = re.match(r'^ok\s+\d+', line_lower)
                # Generic ok: "test_name ... ok" or "test_name ok"
                ok_generic = re.search(r'\.\.\.\s+ok\s*$', line_lower) or re.search(r'\s+ok\s*$', line_lower)
                
                if ok_tap or ok_generic:
                    result["passed"] += 1
                # TAP fail: "not ok 1 - test description"
                elif re.match(r'^not\s+ok\s+\d+', line_lower):
                    result["failed"] += 1
            
            # Dot notation (unittest)
            if re.match(r'^[\.FEs]+$', line.strip()):
                result["passed"] += line.count('.')
                result["failed"] += line.count('F')
                result["errors"] += line.count('E')
                result["skipped"] += line.count('s')
        
        # Calculate total if not found
        if result["total_tests"] == 0:
            result["total_tests"] = result["passed"] + result["failed"] + result["errors"] + result["skipped"]
        
        # Pattern 3: Extract failed test details
        # Look for lines with FAILED, ERROR, FAIL:, ERROR: at beginning or as status
        current_failed_test = None
        error_lines = []
        
        for line in lines:
            # Check for test failure indicators (must be test result, not file path)
            # Pattern: "test_name FAILED" or "FAILED test_name" or "FAIL: test_name" or "ERROR: test_name"
            failed_patterns = [
                r'^(.+?)\s+(FAILED|ERROR)\s*(\[|$)',  # "test_name FAILED"
                r'^(FAIL|ERROR):\s+(.+)$',             # "FAIL: test_name"
                r'^---\s+FAIL:\s+(.+)$',               # "--- FAIL: TestName" (go)
                r'^test\s+(.+?)\s+\.\.\.\s+FAILED$',   # "test name ... FAILED" (rust)
                r'^\s*\d+\)\s+(.+)$',                  # "  1) test name" (mocha)
                r'●\s+(.+)$',                          # "● Test suite › test" (jest)
                r'^not\s+ok\s+\d+\s*-?\s*(.+)$',      # "not ok 1 - test name" (TAP)
            ]
            
            matched = False
            for pattern in failed_patterns:
                match = re.search(pattern, line)
                if match:
                    matched = True
                    # Save previous failed test
                    if current_failed_test and error_lines:
                        result["failed_tests"].append({
                            "name": current_failed_test,
                            "error": "\n".join(error_lines).strip()
                        })
                    
                    # Extract test name
                    if len(match.groups()) == 3 and match.group(1):  # Pattern with status at end
                        current_failed_test = match.group(1).strip()
                    elif len(match.groups()) >= 2:  # Pattern with status at start
                        current_failed_test = match.group(2).strip() if match.group(2) else match.group(1).strip()
                    else:
                        current_failed_test = match.group(1).strip()
                    
                    error_lines = []
                    break
            
            if not matched and current_failed_test:
                # Capture error lines (stop at separators or next test)
                if re.match(r'^(===|---|___){3,}$', line) or re.match(r'^(PASS|OK)', line):
                    # End of error section
                    if error_lines:
                        result["failed_tests"].append({
                            "name": current_failed_test,
                            "error": "\n".join(error_lines).strip()
                        })
                        current_failed_test = None
                        error_lines = []
                else:
                    error_lines.append(line)
        
        # Add last failed test if exists
        if current_failed_test and error_lines:
            result["failed_tests"].append({
                "name": current_failed_test,
                "error": "\n".join(error_lines).strip()
            })
        
        return result
  
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
        def parse_response(cls, raw_text: str) -> Tuple[str, list[dict]|None, str]:
            global JSON_LLM_USED, JSON_LITERAL_USED
            raw_text2 = raw_text
            #logger.info("raw_text:{}".format(raw_text))
            raw_text=cls._strip_code_fences(raw_text)
            try:
                if cls.is_json_string(raw_text):
                    raw_text = json.loads(raw_text)
                    if isinstance(raw_text, str): # sometimes server returns leading quotes.
                        raw_text = json.loads(raw_text)
            except Exception as e:
                try:
                    with open("raw_text.txt", "w") as f:
                        f.write(raw_text)
                    with open("raw_text2.txt", "w") as f:
                        f.write(raw_text2)
                    JSON_LITERAL_USED += 1
                    raw_text = literal_eval(raw_text)
                    if isinstance(raw_text, str):
                        raw_text = json.loads(raw_text)
                        if isinstance(raw_text, str):
                            raw_text = json.loads(raw_text)
                except Exception as e:
                    if isinstance(raw_text, str):
                        JSON_LLM_USED += 1
                        logger.info("Trying to fix json string with llm")
                        logger.info(raw_text)
                        raw_text_n = EnhancedNetwork.fix_json_string_with_llm(raw_text)
                        if raw_text_n:
                            raw_text = json.dumps(raw_text_n)
                        else:
                            logger.info("json load failed")
                            error="Invalid JSON: "+str(e)
            
            content_text = ""
            tool_calls = None
            error = ""
            if isinstance(raw_text, (dict, list)):
               
                if type(raw_text) == dict and raw_text.get("response_type")=="tool":
                    if raw_text.get("tool_calls") is not None and isinstance(raw_text.get("tool_calls"), list) and len(raw_text.get("tool_calls")) > 0:
                        
                        tool_calls=raw_text.get("tool_calls")
                        try:
                            logger.info("<green>🤖 Found tool calls</green>\n{tool_calls}")
                            tool_calls=[{"id":cls.stable_tool_call_id(call.get("name"),call.get("arguments")),"type":"function","function":{"name":call.get("name"),"arguments":json.dumps(call.get("arguments") if isinstance(call.get("arguments"), (dict, list)) else {"input":call.get("arguments")})}} for call in tool_calls]
                            content_text=""
                        except Exception as e:
                            error="Invalid tool_calls arguments."
                            logger.error(f"cannot fix tool_calls arguments: {e}")
                            content_text=json.dumps(raw_text)
                            tool_calls=None
                        
                    else:
                        logger.info("found no tool calls, invalid tool_calls arguments.")
                        error="Invalid tool_calls arguments."
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
        def check_tool_call_section(cls, response: str, raw_response: str, correct_format: str)->str:
            """Validate that the response has proper THOUGHT and TOOL_CALL sections."""
            # Check for required sections
            if not re.search(r"^=+\s*THOUGHT\s*$", raw_response, re.MULTILINE):
                return f"Missing THOUGHT section. You must respond in this exact format:\n{correct_format}"
            
            if not re.search(r"^=+\s*TOOL_CALL\s*$", raw_response, re.MULTILINE):
                return f"Missing TOOL_CALL section. You must respond in this exact format:\n{correct_format}"
            
            # Check for multiple sections (which is not allowed)
            thought_count = len(re.findall(r"^=+\s*THOUGHT\s*$", raw_response, re.MULTILINE))
            tool_call_count = len(re.findall(r"^=+\s*TOOL_CALL\s*$", raw_response, re.MULTILINE))
            
            if thought_count > 1:
                return f"ERROR: Found {thought_count} THOUGHT sections. You must have EXACTLY ONE THOUGHT section.\n{correct_format}"
            
            if tool_call_count > 1:
                return f"ERROR: Found {tool_call_count} TOOL_CALL sections. You must have EXACTLY ONE TOOL_CALL section.\n{correct_format}"
            
            # Check if there's content before THOUGHT section
            first_thought_pos = raw_response.find("=")
            if first_thought_pos > 0 and raw_response[:first_thought_pos].strip():
                return f"ERROR: Do not add any content before the THOUGHT section. Your response must start with ===================THOUGHT\n{correct_format}"
            
            return "success"
    
    def __init__(
        self,
        agent_name = "assistant",
        model_name: str = GLM_MODEL_NAME,
        system_message: str | None = None,
        tools: list | None = None
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
                return False, f"Expected {expected_length} markdown sections but found {len(sections)}", None
            else:
                return True, None, tuple(sections)
        
        else:
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
                    
                if last_message is None:
                    logger.error("No response message returned by assistant. This should not happen..")
                    continue
                
                candidate_text = ProxyClient.Utils._extract_text_from_message(last_message).strip()
                
                candidate_text, _, error = ProxyClient.Utils.parse_response(candidate_text)
                
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

class BaseSolver:
    """Base class for problem solvers sharing common functionality."""
    
    def __init__(self, problem_statement: str, tool_manager: ToolManager):
        """Initialize base solver with problem statement and tool manager."""
        self.problem_statement = problem_statement
        self.tool_manager = tool_manager
    
    def process_response(self, response, step_number: int = 0, elapsed_time: float = 0.0) -> Tuple[str | None, str]:
        """
        Process LLM response and execute tool calls.
        
        Args:
            response: Response from LLM (either tuple or string)
            step_number: Current step number for logging
            elapsed_time: Time elapsed since start for logging
            
        Returns:
            Tuple of (tool_output, tool_name)
        """
        resp = None
        tool_name = ""
        tool_call = None
        
        if response is None:
            logger.error("response NONE received..")
            return None, ""
            
        if type(response) is tuple and len(response) == 2:
            _, tool_call = response
        elif "{" in response:
            tool_call = response
            
        if tool_call:
            json_obj, _, error = ProxyClient.Utils.parse_response(str(tool_call))
            if error:
                resp = error
            elif json_obj:
                try:
                    json_obj = json.loads(json_obj)
                    logger.info(f"Step {step_number} | Tool: {json_obj.get('name')} | Time: {elapsed_time:.2f}s")
                    tool_name = str(json_obj.get("name", ""))
                    tool = self.tool_manager.get_tool(tool_name)
                    if tool is None:
                        resp = f"Error: {json_obj.get('name')} tool not found"
                    else:
                        resp = tool.run(tool_input=json_obj.get("arguments"))
                except Exception as e:
                    logger.error(f"Error calling tool: {e}")
                    resp = f"Error: {e}"
                    
        return resp, tool_name

class CreateProblemSolver(BaseSolver):
    
    SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL=textwrap.dedent("""
    You are an expert Python developer. You will be given a problem statement and a python solution. You need to evaluate if the solution is correct or not as per the problem statement.
    
    WorkFlow:-
        - **Plan:** After understanding the problem statement, create a initial list of all the requirements mentioned in problem statement that you need to evaluate.
        - **Evaluate:** Begin evaluating the solution for each of those cases. Create test cases to confirm if the solution is correct.
        - **Adapt:** As you discover new information or encounter obstacles, update your plan.
        - **Verify (Tests):** Check test_cases.txt file for additional scenerios including edge cases you can test.
        - **Comprehensive Testing:** Think about all possible edge cases then find new solution. Run comprehensive test to ensure solution fully satisfies all the requirements.
        - **Finish:** Call complete tool once the solution fully satisfies all the requirements.

    *GUIDE FOR HOW TO USE "sequential_thinking" TOOL:*
        1. Your thinking should be thorough and so it's fine if it's very long. Set totalThoughts to at least 5, but setting it up to 25 is fine as well. You'll need more total thoughts when you are considering multiple possible solutions or root causes for an issue.
        2. Use this tool as much as you find necessary to improve the quality of your answers.
        3. You can run bash commands (like tests, a reproduction script, or 'grep'/'find' to find relevant context) in between thoughts.
        4. The "sequential_thinking" tool can help you break down complex problems, analyze issues step-by-step, and ensure a thorough approach to problem-solving.
        5. Don't hesitate to use it multiple times throughout your thought process to enhance the depth and accuracy of your solutions.

    Tool Usage:-
        - Use run_code to create and run unittests.
        - Use apply_code_edit to fix the solution if it fails.
        - Use apply_code_edit to fix the test case if they are not as per the problem statement.
        - Use complete to finish the task.
    
    Rules:-
        1. Test code must always import functionality from the repository—never duplicate or reimplement the code within the test itself.
        2. Use verbosity level 2 while running the tests to ensure you see the full output.
        3. If run_code tool throws syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases and try again.
        4. Must ensure you have tested **ALL scenarios** listed in test_cases.txt file. Even if some  scenarios are not mentioned in problem statement, you must test them.
        5. **CRITICAL OUTPUT FORMAT:** You MUST respond in the exact format specified below. Do not add extra sections or deviate from this format.
        6. Test code must include all edge cases.
        7. Implement all variables,classes and functions from the problem statement.
        8. Generate working solution for the problem with as much detail as possible from the context.
    
    Here are the tools you have access to:-
    {tools_docs}
    
    **STRICT RESPONSE FORMAT - YOU MUST FOLLOW THIS EXACTLY:**
    {format_prompt}
    
    **Remember:** Your response MUST contain exactly ONE THOUGHT section followed by ONE TOOL_CALL section. Do not add anything before the THOUGHT section or after the TOOL_CALL section.
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
    
    RESPONSE_FORMAT_JSON="""Return only the final python files code in JSON format.
    Response Examples:
    [{"file_name":"a.py","code":"contents of a.py"},{"file_name":"b.py","content":"contents of b.py"}]
    """
    
    RESPONSE_FORMAT_SOLUTION_EVAL_2=textwrap.dedent("""
    **📝 STRICT RESPONSE FORMAT - MANDATORY FOR EVERY RESPONSE**
    
    **1. Required Structure:**
    Every response MUST contain EXACTLY TWO sections in this order:
    
    ===================THOUGHT
    <<Your detailed reasoning including:
    - Current evaluation status and what you're checking
    - Analysis of test results or code quality
    - Why you're choosing this specific tool/action
    - What you expect to achieve>>
    
    ===================TOOL_CALL
    {{"name":"<exact_tool_name>","arguments":{{<valid_json_args>}}}}
    
    **2. THOUGHT Section Requirements:**
    - Clearly state what you're evaluating or testing
    - Include test results analysis when relevant
    - Explain your reasoning for the next action
    - Reference problem statement requirements
    - State expected outcome
    
    **3. TOOL_CALL Section Requirements:**
    - Must be valid JSON object
    - "name" must exactly match: run_code, apply_code_edit, get_file_content, run_python_file, search_in_specified_file_v2, complete
    - "arguments" must match the tool's input schema
    - Properly escape special characters in JSON strings
    - No trailing commas
    - No comments in JSON
    
    **4. ✅ VALID RESPONSE EXAMPLES:**
    
    Example 1 - Running test code:
    ===================THOUGHT
    I need to verify the solution handles edge case with empty input. I'll create a test to check if the function returns the expected default value when input is empty. This is mentioned in the problem statement requirements.
    ===================TOOL_CALL
    {{"name":"run_code","arguments":{{"content":"import unittest\\nfrom solution import process_input\\n\\nclass TestEmpty(unittest.TestCase):\\n    def test_empty_input(self):\\n        self.assertEqual(process_input(''), 'default')\\n\\nif __name__ == '__main__':\\n    unittest.main()","file_path":"test_empty.py"}}}}
    
    Example 2 - Fixing code:
    ===================THOUGHT
    The test revealed that the function crashes on None input. I need to add a null check before processing. This will prevent the TypeError and match the problem statement requirement that all edge cases should be handled gracefully.
    ===================TOOL_CALL
    {{"name":"apply_code_edit","arguments":{{"file_path":"solution.py","search":"def process_input(data):\\n    return data.strip()","replace":"def process_input(data):\\n    if data is None:\\n        return 'default'\\n    return data.strip()"}}}}
    
    Example 3 - Reading file to understand:
    ===================THOUGHT
    Before running more tests, I should examine the current solution implementation to understand the data flow. This will help me identify where the edge case handling should be added.
    ===================TOOL_CALL
    {{"name":"get_file_content","arguments":{{"file_path":"solution.py"}}}}
    
    Example 4 - Searching for function:
    ===================THOUGHT
    I need to locate where the validation logic is implemented. I'll search for the validate function to understand its current behavior before modifying it.
    ===================TOOL_CALL
    {{"name":"search_in_specified_file_v2","arguments":{{"file_path":"solution.py","search_term":"def validate"}}}}
    
    Example 5 - Completion:
    ===================THOUGHT
    All test cases pass successfully. I've verified the solution handles all edge cases mentioned in the problem statement including empty inputs, None values, and special characters. The solution is complete and correct.
    ===================TOOL_CALL
    {{"name":"complete","arguments":{{"answer":"Solution complete. All test cases pass, including edge cases for empty input, None values, and special characters."}}}}
    
    **5. ❌ INVALID RESPONSE EXAMPLES (DO NOT DO THIS):**
    
    ❌ Missing THOUGHT section:
    ===================TOOL_CALL
    {{"name":"run_code","arguments":{{"content":"test","file_path":"test.py"}}}}
    
    ❌ Missing TOOL_CALL section:
    ===================THOUGHT
    I need to run tests
    
    ❌ Text before THOUGHT section:
    Let me test the solution.
    ===================THOUGHT
    Testing now...
    
    ❌ Text after TOOL_CALL section:
    ===================TOOL_CALL
    {{"name":"complete","arguments":{{"answer":"done"}}}}
    This completes the task.
    
    ❌ Invalid JSON (missing quotes):
    ===================TOOL_CALL
    {{"name":"run_code","arguments":{{content:"test",file_path:"test.py"}}}}
    
    ❌ Invalid JSON (trailing comma):
    ===================TOOL_CALL
    {{"name":"complete","arguments":{{"answer":"done",}}}}
    
    ❌ Wrong tool name:
    ===================TOOL_CALL
    {{"name":"execute_code","arguments":{{"content":"test","file_path":"test.py"}}}}
    
    ❌ Multiple THOUGHT sections:
    ===================THOUGHT
    First thought
    ===================THOUGHT
    Second thought
    ===================TOOL_CALL
    {{"name":"complete","arguments":{{"answer":"done"}}}}
    
    ❌ Not escaped strings in code content:
    ===================TOOL_CALL
    {{"name":"run_code","arguments":{{"content":"print("hello")","file_path":"test.py"}}}}
    
    **6. Critical Formatting Rules:**
    - Section headers MUST be exactly: ===================THOUGHT and ===================TOOL_CALL
    - No extra spaces, no missing equals signs
    - JSON must be valid (use double quotes, no trailing commas)
    - When including Python code in "content", escape quotes and newlines properly
    - Use \\n for newlines in Python code strings
    - Use \\" for quotes inside Python code strings
    - Tool names are case-sensitive and must match exactly
    
    **7. Error Recovery Format:**
    If you receive an error about your format, respond with:
    ===================THOUGHT
    Error acknowledged. My previous response had invalid format: [explain the error]. I will now respond correctly using the proper two-section format with valid JSON.
    ===================TOOL_CALL
    {{"name":"<appropriate_tool>","arguments":{{...}}}}
    """)
    
    TEST_CASE_GENERATOR_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

    Important things:
    1. Test functions declared in code skeleton, don't customized those prototypes.
    2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
    3. Do not generate testcases that are not mentioned in problem statement.
    4. Generate test cases for all possible error scenarios and edge cases of the problem statement.


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
                if "unittest.main()" not in content:
                    return "Generation limit reached. Response truncated.. Skip last couple of test cases from your last response.."
                return f"Syntax error: {e}\n If the syntax error is due to the response getting truncated skip last couple of test cases and try again."
    
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
        super().__init__(problem_statement, tool_manager)
        self.problem_statement = self.post_process_instruction()
        self.code_skeleton = self.get_code_skeleton()

        self.agent_initial_solution_eval=CustomAssistantAgent(
            system_message=CreateProblemSolver.SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL.format(tools_docs=tool_manager.get_tool_docs(), format_prompt=self.RESPONSE_FORMAT_SOLUTION_EVAL_2),
            model_name=GLM_MODEL_NAME
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
        
        response=await self.agent_initial_solution_eval.solve_task(
            CreateProblemSolver.INSTANCE_PROMPT_INITIAL_SOLUTION_EVAL.format(problem_statement=self.problem_statement,initial_solution=initial_solution),
            response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2,
            is_json=False,
            regex=None,
            post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),
            max_attempts=3,
            is_parallel=False,
            disable_reset=True,
            return_type=tuple[str,str]
        )
        response, tool_name=self.process_response(response)
        finish_called_earlier=True
        while True:
            if tool_name != "complete":
                logger.info(f"\n\n<yellow>⚡ Execution</yellow>")
                response=await self.agent_initial_solution_eval.solve_task(
                    str(response),
                    response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2,
                    is_json=False,
                    regex=None,
                    post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),
                    max_attempts=3,
                    is_parallel=False,
                    disable_reset=True,
                    return_type=tuple[str,str]
                )
                response, tool_name=self.process_response(response)
            else:
                if not finish_called_earlier:
                    response=await self.agent_initial_solution_eval.solve_task(
                        "Check the problem statement and find out the cases which have not been tested yet. You must check all the mentioned scenarios (outputs, any edge cases, errors, any workflows). Create those test cases and test your solution.",
                        response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2,
                        is_json=False,
                        regex=None,
                        post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),
                        max_attempts=3,
                        is_parallel=False,
                        disable_reset=True,
                        return_type=tuple[str,str]
                    )
                    response, tool_name=self.process_response(response)
                    finish_called_earlier=True
                    continue
                break
        logger.info(f"<yellow>Total time taken: {time.time()-start_time} seconds</yellow>")
        
        final_patch = Utils.create_final_git_patch(self.tool_manager.temp_files)
        logger.info(f"\n<yellow>😊 Generated patch with {len(final_patch)} characters...</yellow>")
        logger.info(final_patch)
        return final_patch

class BugFixSolver(BaseSolver):
    
    FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
    <role>
        You are an expert AI software engineer specializing in bug fixing. You follow a systematic test-driven approach: reproduce first, fix second, validate thoroughly.
    </role>

    <core_principles>
        1. **Reproduce Before Fix**: Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error
        2. **Non-Invasive Testing**: NEVER edit existing test files - always create NEW test scripts
        3. **Verification-Driven**: Every claim must be backed by actual test execution output
        4. **Fail-to-Pass First, Pass-to-Pass Second**: Validate the fix works AND doesn't break existing functionality
    </core_principles>

    <critical_rules>
        ⚠️ **ABSOLUTELY MANDATORY - NEVER VIOLATE:**
        1. **DO NOT edit or modify existing test files** - Always create NEW test scripts
        2. **MUST create a script to reproduce the error** and execute it with `python <filename.py>` using the BashTool, to confirm the error BEFORE fixing
        3. **MUST run Fail-to-Pass validation** - verify your test script passes after the fix
        4. **MUST run Pass-to-Pass validation** - verify existing tests still pass after the fix
        5. **MUST see actual test output** - Never assume tests pass without seeing results
        6. **DON'T** install any packages since you don't have access to the internet
    </critical_rules>

    <workflow_enforcement>
        **Execute steps sequentially. Each step must be completed before moving to the next:**
        
        Phase 1: UNDERSTAND (Steps 1-2)
          - Step 1: Understand problem & explore codebase
          - Step 2: Analyze test framework & establish baseline
        Phase 2: CREATE TEST SCRIPT (Steps 3-5)
          - Step 3: Create test file
          - Step 4: VERIFY test completeness (CRITICAL)
          - Step 5: Run test and confirm it fails
        Phase 3: FIX (Steps 6-7)
          - Step 6: Analyze root cause & plan fix
          - Step 7: Implement the fix
        Phase 4: VALIDATE (Steps 8-11)
          - Step 8: Fail-to-Pass validation
          - Step 9: Pass-to-Pass validation
          - Step 10: Existing functionality validation
          - Step 11: Final validation & completion
        
        ⚠️ Do not skip phases. Do not skip steps within phases.
        ⚠️ Step 4 is CRITICAL - never skip test verification.
    </workflow_enforcement>

    <restrictions>
        ❌ **Cannot do:**
        - Access the internet or install packages
        - Make external API calls or downloads
        - Edit existing test files (tests/, test_*.py, *_test.py)
        
        ✅ **Can do:**
        - Read any project files including test files (to understand, not edit)
        - Create NEW test scripts for reproduction
        - Run existing test commands
        - Read configuration files (pytest.ini, setup.py, tox.ini, etc.)
    </restrictions>

    <test_editing_policy>
        **STRICT RULE:** Do NOT edit existing test files under any circumstances.
        
        ✅ **Correct approach:**
        - Create: test_issue.py (new file)
        - Create: test_bug_fix.py (new file)
        - Create: verify_fix.py (new file)
        
        ❌ **Incorrect approach:**
        - Edit: tests/test_module.py (existing file)
        - Modify: test_feature.py (existing file)
        - Update: any file in tests/ directory
    </test_editing_policy>

    <test_framework_discovery>
        **CRITICAL: Test Framework Discovery Strategy**
        
        **Step 1: Project Structure Analysis**
        - Look for framework indicators: management scripts, package configuration files, dependency files
        - Check for test configuration files: test runner configs, test suite definitions
        - Examine dependency management files: requirements files, package managers, build configurations
        
        **Step 2: Existing Test Pattern Analysis**
        - Read existing test files to identify framework patterns
        - Look for test framework imports and base classes
        - Check test file naming conventions and directory structure
        - Analyze test setup and teardown patterns
        
        **Step 3: Native Test Command Discovery**
        - Try framework-specific commands first by analyzing project structure
        - Look for management scripts that might run tests
        - Check for standard test runner commands in the project
        - Look for custom test scripts in configuration files or build systems
        
        **Step 4: Fallback Strategy**
        - **ONLY** use standalone scripts if:
          * Native framework fails due to missing dependencies that cannot be installed
          * Native framework has configuration issues that cannot be resolved
          * Native framework is not available in the environment
        - Document why native framework failed before using standalone scripts
        
        **Step 5: Validation Priority**
        1. **FIRST**: Try native test framework
        2. **SECOND**: Diagnose and fix framework issues if package installation does not required
        3. **LAST RESORT**: Use standalone scripts only if framework is truly unusable
        4. Save the analysis result in the context using save_context
    </test_framework_discovery>

    <validation_requirements>
        **Three-Phase Validation is MANDATORY:**
        
        **Phase A: Fail-to-Pass (F2P) Validation**
        - Run YOUR test script (the one you created)
        - Confirm it NOW PASSES (it should have failed before the fix)
        - If it still fails → iterate on the fix
        
        **Phase B: Pass-to-Pass (P2P) Validation**
        - Find ALL relevant existing test files
        - **CRITICAL: Discover and use the project's native test framework**
        - **Test Framework Discovery Process:**
          1. Analyze project structure (look for management scripts, configuration files, dependency files)
          2. Check existing test files for framework patterns and imports
          3. Try native test commands first by examining project structure and existing test patterns
          4. Only use standalone scripts if native framework fails due to dependency issues
        - Run existing test suite using native framework
        - ALL tests MUST pass - no exceptions
        - If ANY test fails:
          * Analyze why your fix caused the failure
          * Refine your fix to satisfy both F2P and P2P requirements
          * Iterate until all tests pass
        
        **Phase C: Existing Functionality Validation (CRITICAL)**
        - **Understand Current Testing System**: Analyze how tests are organized and run in the project
        - **Identify Test Dependencies**: Map out which components are tested by existing tests
        - **Run Comprehensive Test Suite**: Execute all tests to ensure no regressions
        - **Validate Component Integration**: Ensure your fix doesn't break other components
        - **Check Test Coverage**: Verify that existing functionality is still properly tested
        - **Environment Validation**: Ensure test environment is properly configured
        - **Documentation Validation**: Check if any documentation needs updates due to changes
    </validation_requirements>

    <best_practices>
        ✅ **Always do:**
        - Use 'sequential_thinking' for root cause analysis
        - Read existing test files to understand patterns and conventions
        - **Discover and use the project's native test framework** - analyze project structure to find built-in test commands
        - Run existing test suite FIRST to establish baseline (Step 2)
        - Create new test file within test suite directory
        - Follow exact naming conventions (test_*.py or *_test.py as appropriate)
        - Use same imports, fixtures, and assertion style as existing tests
        - Run tests multiple times to ensure consistency
        - Verify new test is discoverable by test runner
        - **Use 'save_context' tool** to save summaries after completing each step
        - **Understand the current testing system thoroughly** - analyze test organization, frameworks, and patterns
        - **Map test dependencies** - identify which components are tested by existing tests
        - **Validate component integration** - ensure your fix doesn't break other components
        - **Check test coverage** - verify existing functionality is still properly tested
        - **Follow senior software engineer practices** - separate concerns, avoid leaky interfaces
        - **Choose simpler solutions** when possible
        - **Set up necessary environment variables** for test execution
        
        ❌ **Never do:**
        - Skip running existing tests before creating your new test
        - Create standalone scripts outside the test suite UNLESS native test framework fails due to dependency issues
        - Edit existing test files
        - Assume test commands without trying them
        - Move forward without seeing actual test output
        - Repeat failed commands more than twice without changing approach
        - Skip validation of existing functionality
        - Break other components with your changes
        - Decrease test coverage
        - Ignore test dependencies and component relationships
        - Skip saving context after completing steps
        - Use standalone scripts when native test framework is available and working
    </best_practices>

    <completion_criteria>
        ✅ **Call 'complete' tool ONLY when ALL criteria are met:**
        1. Existing test suite was run and baseline established (Step 2) ✓
        2. New test file was created within test suite (Step 3) ✓
        3. Test completeness was verified - all scenarios covered (Step 4) ✓
        4. New test initially failed, confirming it catches the bug (Step 5) ✓
        5. Code fix was implemented (Step 7) ✓
        6. New test file now passes (F2P validation - bug is fixed) (Step 8) ✓
        7. ALL tests pass including new test (P2P validation - no regressions) (Step 9) ✓
        8. **Existing functionality validation complete** - no regressions in existing tests (Step 10) ✓
        9. All scenarios in problem statement are addressed ✓
        10. New test file follows existing test conventions and is discoverable ✓
        11. **Current testing system understood** - patterns, frameworks, and organization mapped ✓
        12. **Component integration validated** - shared interfaces and APIs work correctly ✓
        13. **Test coverage maintained** - no decrease in existing test coverage ✓
    </completion_criteria>

    <response_format>
        **📝 STRICT RESPONSE FORMAT - MANDATORY FOR EVERY RESPONSE**
        
        **1. Required Structure:**
        Every response MUST contain EXACTLY TWO sections in this order:
        
        ===================THOUGHT
        <<Your detailed reasoning including:
        - Current step number and success criteria
        - Analysis of the situation
        - Why you're choosing this specific tool
        - What you expect to achieve>>
        
        ===================TOOL_CALL
        {{"name":"<exact_tool_name>","arguments":{{<valid_json_args>}}}}
        
        **2. THOUGHT Section Requirements:**
        - Start with current step context (e.g., "Step 2: Locating source files...")
        - Include what success looks like for this step
        - Explain your reasoning thoroughly
        - Reference previous findings when relevant
        - State what you'll do and why
        
        **3. TOOL_CALL Section Requirements:**
        - Must be valid JSON object
        - "name" must exactly match available tool names
        - "arguments" must match the tool's input schema
        - Properly escape special characters in JSON strings
        - No trailing commas
        - No comments in JSON
        
        **4. Critical Formatting Rules:**
        - Section headers MUST be exactly: ===================THOUGHT and ===================TOOL_CALL
        - No extra spaces, no missing equals signs
        - JSON must be on a single line (or properly formatted multi-line)
        - Always use double quotes in JSON, never single quotes
        - Escape backslashes and quotes inside JSON strings
        - When using str_replace, preserve exact indentation and line breaks using \\n
        - Test command parameters: use "is_test_command" not "parse_test_results"
        
        **5. Error Recovery Format:**
        If you receive an error about your format, respond with:
        ===================THOUGHT
        Error acknowledged. I provided an invalid format: [explain the error]. I will now respond correctly in the proper format.
        ===================TOOL_CALL
        {{"name":"<appropriate_tool>","arguments":{{...}}}}
    </response_format>
    """)
    
    FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
    <context>
        You are working in a Python repository. Your current working directory is the repository root.
        All project files are available for inspection and modification.
    </context>

    <task>
        Fix the bug described in the problem statement below.
        Follow the mandatory workflow defined in your system prompt.
    </task>

    <critical_reminders>
        • Start with Step 1 of the mandatory workflow
        • Follow each step in exact order - do not skip any step
        • Run tests to verify - never assume without seeing output
        • Create NEW test files within the test suite - never edit existing tests
        • All tests must pass before calling complete tool
    </critical_reminders>

    <tool_usage_reminders>
        bash:
        • Use is_test_command=true when running tests to see only failures
        • Use is_test_command=false to see full test output
        • Always check file existence before running files
        
        str_replace_editor:
        • Use "view" command to read files
        • Use "str_replace" command to make changes
        • Preserve exact indentation and line breaks using \\n
        
        sequential_thinking:
        • Use for complex analysis and planning
        • Never embed tool calls inside the "thought" parameter
        • Set totalThoughts to at least 5 for thorough analysis
        
        save_context:
        • Use after completing each step to save progress
        • Include step_number, step_name, and summary (required)
        • Add findings and next_steps for better context
        • This helps maintain continuity across the workflow
    </tool_usage_reminders>

    <anti_repetition_guidelines>
        🚨 CRITICAL: ANTI-REPETITION GUIDELINES 🚨
        • Keep track of what you've already tried to avoid repeating the same actions
        • If you've already attempted a solution, try a different approach
        • If you find yourself repeating the same tool calls, stop and think of a different strategy
        • Document your progress in your thinking to avoid loops
        • Use "sequential_thinking" to plan your approach before executing
        • NEVER repeat the same bash command more than 2 times in a row
        • If a file doesn't exist, CREATE it first instead of trying to run it repeatedly
        • REPEATING THE SAME FAILED ACTION WILL CAUSE INFINITE LOOPS
    </anti_repetition_guidelines>

    <restrictions>
        CRITICAL - No Internet Access:
        • You CANNOT access the internet or download packages
        • Do NOT try to install packages with pip, apt, or any package manager
        • Do NOT try to download files from the internet
        • Do NOT try to access external services or APIs
        • Work only with the files and tools available in the local repository
        • If you need a package that's not available, find an alternative solution or work around it
        • Do NOT create any documentation files, summary files, README files, or SOLUTION_SUMMARY.md files
    </restrictions>

    <mandatory_workflow>
    **CRITICAL: Follow these steps in EXACT order. Do NOT skip any step.**

    **Step 1: Understand Problem & Explore Codebase**
    - Read problem statement carefully
    - Use `ls`, `find`, `grep` to explore repository structure
    - Find source files related to the problem
    - Read relevant code to understand current implementation
    - Identify what needs to be fixed

    **Step 2: Create Test Script**
    - Create NEW test file within test directory structure
    - Follow existing test naming conventions and patterns
    - Write test cases that cover ALL scenarios in problem statement
    - Use same imports, structure, and assertion style as existing tests
    - DO NOT edit existing test files

    **Step 3: Verify Test Completeness**
    - Use `sequential_thinking` to analyze test completeness
    - Ask: "If ALL these test cases pass, does that GUARANTEE the problem is fixed?"
    - Verify ALL scenarios from problem statement are covered
    - Check ALL edge cases are included
    - Ensure ALL code paths that trigger the bug are tested

    **Step 4: Run Test Script & Confirm Error**
    - Understand existing test code pattern and how to configure and run the test from project structure if there is any
    - Create a standalone script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to reproduce the error
    - The script MUST FAIL (proving it catches the bug)
    - Check if the failure matches the problem described and if all are pass the problem is completed fixed
    - If test passes when it should fail → go back to Step 4

    **Step 6: Analyze Root Cause & Plan Fix**
    - Use `sequential_thinking` tool (set totalThoughts: 10-25)
    - Analyze why the bug is happening (root cause)
    - Analyze all related files (may need to change multiple files)
    - Brainstorm 5-7 possible solutions
    - Choose the best fix approach
    - Plan implementation steps

    **Step 7: Implement the Fix**
    - Edit the source code file(s) identified in Step 1
    - Implement the fix planned in Step 6
    - Keep changes minimal and focused
    - Ensure code handles all edge cases

    **Step 8: Fail-to-Pass (F2P) Validation**
    - Run your new test script using the same test command as existing tests
    - The new test should now PASS (bug is fixed)
    - If any test fails → go back to Step 6, refine the fix
    - Iterate Steps 6→7→8 until all tests pass

    **Step 9: Pass-to-Pass (P2P) Validation**
    - **CRITICAL: Discover the project's native test framework**
    - Analyze project structure: look for management scripts, configuration files, dependency files
    - Check existing test files to identify framework patterns and imports
    - Try native test commands first by examining project structure and existing test patterns
    - **ONLY use standalone scripts if native framework fails due to dependency issues**
    - Run the FULL existing test suite using the discovered native framework
    - Include your new test script in the run
    - ALL tests must pass (existing + your new test)
    - If ANY test fails → analyze why and refine fix
    - Iterate Steps 6→7→8→9 until ALL tests pass

    **Step 10: Existing Functionality Validation**
    - Understand current testing system (patterns, frameworks, organization)
    - Identify test dependencies and component relationships
    - Run comprehensive test suite to ensure no regressions
    - Validate component integration (shared interfaces work correctly)
    - Check test coverage maintained
    - Ensure environment properly configured

    **Step 11: Final Validation & Completion**
    - Run full test suite one final time
    - Verify ALL tests passing
    - Review all problem statement requirements → all addressed
    - Confirm no regressions introduced
    - If ALL criteria met: Call the 'complete' tool

    </mandatory_workflow>
    
    ## Multi-file awareness (critical):
    - Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
    - Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
    - Re-run tests only after covering all discovered occurrences to avoid partial fixes.

    <response_format>
        **📝 STRICT RESPONSE FORMAT - MANDATORY FOR EVERY RESPONSE**
        
        **1. Required Structure:**
        Every response MUST contain EXACTLY TWO sections in this order:
        
        ===================THOUGHT
        <<Your detailed reasoning including:
        - Current step number and success criteria
        - Analysis of the situation
        - Why you're choosing this specific tool
        - What you expect to achieve>>
        
        ===================TOOL_CALL
        {{"name":"<exact_tool_name>","arguments":{{<valid_json_args>}}}}
        
        **2. THOUGHT Section Requirements:**
        - Start with current step context (e.g., "Step 2: Locating source files...")
        - Include what success looks like for this step
        - Explain your reasoning thoroughly
        - Reference previous findings when relevant
        - State what you'll do and why
        
        **3. TOOL_CALL Section Requirements:**
        - Must be valid JSON object
        - "name" must exactly match available tool names
        - "arguments" must match the tool's input schema
        - Properly escape special characters in JSON strings
        - No trailing commas
        - No comments in JSON
        
        **4. ✅ VALID RESPONSE EXAMPLES:**
        
        Example 1 - Using bash tool:
        ===================THOUGHT
        Step 2: Locating source files related to authentication
        Success criteria: Find the files that contain authentication logic
        I need to search for files containing "auth" to understand the codebase structure. This will help me identify where the bug might be located.
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{"command":"find . -name '*.py' | grep -i auth"}}}}
        
        Example 2 - Using str_replace_editor to view:
        ===================THOUGHT
        Step 2: Examining the authentication module
        Success criteria: Understand the current implementation of login function
        Now I'll read the auth.py file to see the current implementation and identify where the password validation is failing.
        ===================TOOL_CALL
        {{"name":"str_replace_editor","arguments":{{"command":"view","path":"src/auth.py"}}}}
        
        Example 3 - Using str_replace_editor to fix:
        ===================THOUGHT
        Step 6: Implementing the fix for password validation
        Success criteria: Update the code to properly handle empty passwords
        I'll add a check for empty passwords before the hash comparison. This prevents the TypeError when password is None.
        ===================TOOL_CALL
        {{"name":"str_replace_editor","arguments":{{"command":"str_replace","path":"src/auth.py","old_str":"def validate_password(password, hash):\\n    return bcrypt.checkpw(password, hash)","new_str":"def validate_password(password, hash):\\n    if not password:\\n        return False\\n    return bcrypt.checkpw(password, hash)"}}}}
        
        Example 4 - Using bash with test parsing:
        ===================THOUGHT
        Step 8: Running Pass-to-Pass validation
        Success criteria: Confirm all existing tests still pass after the fix
        I'll run the existing test suite with is_test_command=true to get a summary of results and see only failures if any exist.
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{"command":"pytest tests/test_auth.py -v","is_test_command":true}}}}
        
        Example 5 - Completion:
        ===================THOUGHT
        Step 9: Task completion
        Success criteria: All validations passed, ready to finish
        All tests pass (F2P and P2P validation complete). The fix handles empty passwords correctly and doesn't break any existing functionality. Ready to complete.
        ===================TOOL_CALL
        {{"name":"complete","arguments":{{"answer":"Fixed password validation by adding null check before hash comparison. All tests pass."}}}}
        
        Example 6 - Saving context:
        ===================THOUGHT
        Step 2: Completed test script creation
        Success criteria: Test script created and follows existing patterns
        I've successfully created a test script that reproduces the bug. The script follows the existing test patterns and covers all scenarios from the problem statement. Ready to save this progress.
        ===================TOOL_CALL
        {{"name":"save_context","arguments":{{"step_number":2,"step_name":"Create Test Script","summary":"Created test_issue.py with comprehensive test cases","findings":"Found existing test patterns use unittest framework with specific naming conventions","next_steps":"Verify test completeness and run the test to confirm it fails"}}}}
        
        **5. ❌ INVALID RESPONSE EXAMPLES (DO NOT DO THIS):**
        
        ❌ Missing THOUGHT section:
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{"command":"ls"}}}}
        
        ❌ Missing TOOL_CALL section:
        ===================THOUGHT
        I need to list files
        
        ❌ Text before THOUGHT section:
        Let me start by exploring the code.
        ===================THOUGHT
        Step 1: Exploring...
        
        ❌ Text after TOOL_CALL section:
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{"command":"ls"}}}}
        This will show us the files.
        
        ❌ Invalid JSON (missing quotes):
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{command:"ls"}}}}
        
        ❌ Invalid JSON (trailing comma):
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{"command":"ls",}}}}
        
        ❌ Wrong tool name:
        ===================TOOL_CALL
        {{"name":"run_bash","arguments":{{"command":"ls"}}}}
        
        ❌ Multiple THOUGHT sections:
        ===================THOUGHT
        First thought
        ===================THOUGHT
        Second thought
        ===================TOOL_CALL
        {{"name":"bash","arguments":{{"command":"ls"}}}}
        
        ❌ Not escaped JSON strings:
        ===================TOOL_CALL
        {{"name":"str_replace_editor","arguments":{{"command":"str_replace","path":"test.py","old_str":"print("hello")","new_str":"print("world")"}}}}
        
        **6. Critical Formatting Rules:**
        - Section headers MUST be exactly: ===================THOUGHT and ===================TOOL_CALL
        - No extra spaces, no missing equals signs
        - JSON must be on a single line (or properly formatted multi-line)
        - Always use double quotes in JSON, never single quotes
        - Escape backslashes and quotes inside JSON strings
        - When using str_replace, preserve exact indentation and line breaks using \\n
        - Test command parameters: use "is_test_command" not "parse_test_results"
        
        **7. Error Recovery Format:**
        If you receive an error about your format, respond with:
        ===================THOUGHT
        Error acknowledged. I provided an invalid format: [explain the error]. I will now respond correctly in the proper format.
        ===================TOOL_CALL
        {{"name":"<appropriate_tool>","arguments":{{...}}}}
    </response_format>
    
    <available_tools>
    {available_tools}
    </available_tools>
    
    <problem_statement>
    {problem_statement}
    </problem_statement>
    """)
    
    MAX_FIX_TASK_STEPS=300
    
    def __init__(
        self,
        problem_statement:str,
        tool_manager:ToolManager
    ):
        super().__init__(problem_statement, tool_manager)
        
        # Format instance prompt with problem statement and available tools
        self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
            problem_statement=self.problem_statement,
            available_tools=self.tool_manager.get_tool_docs()
        )
        
        # System prompt doesn't need formatting - it contains the methodology
        formatted_system_prompt = self.FIX_TASK_SYSTEM_PROMPT
        
        self.agent=CustomAssistantAgent(
            system_message=formatted_system_prompt,
            model_name=GLM_MODEL_NAME
        )
        #logger.info("system message: "+self.agent.system_message)
        
    async def solve_problem(self):
        start_time = time.time()
        
        # Get saved context and include it in the initial prompt
        saved_context = self.tool_manager.get_saved_context()
        initial_prompt = self.instruction_prompt
        if saved_context:
            initial_prompt = f"{saved_context}\n\n{self.instruction_prompt}"
        
        response = await self.agent.solve_task(
            initial_prompt,
            response_format="",
            is_json=False,regex=None,
            post_process_func=None,
            max_attempts=3,
            is_parallel=False,
            disable_reset=True,
            return_type=tuple[str,str]
        )
        for step in range(self.MAX_FIX_TASK_STEPS):
            step_start_time = time.time()
            elapsed_time = step_start_time - start_time
            
            resp, tool_name = self.process_response(response, step + 1, elapsed_time)
            if tool_name != "complete":
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                logger.info(f"Step {step + 1}/{self.MAX_FIX_TASK_STEPS} | Iteration {step + 1} | Time: {elapsed_time:.2f}s")
                
                # Include saved context in subsequent LLM calls
                saved_context = self.tool_manager.get_saved_context()
                next_prompt = str(resp)
                if saved_context:
                    next_prompt = f"{saved_context}\n\n{str(resp)}"
                
                response = await self.agent.solve_task(
                    next_prompt, 
                    response_format="", 
                    is_json=False, 
                    regex=None, 
                    post_process_func=None, 
                    max_attempts=10, 
                    is_parallel=False, 
                    disable_reset=True, 
                    return_type=tuple[str,str]
                )
            else:
                logger.info(f"Task completed at step {step + 1} | Total time: {elapsed_time:.2f}s")
                break
            
        final_patch = Utils.create_final_git_patch(self.tool_manager.temp_files)
        
        total_time = time.time() - start_time
        logger.info(f"Generated patch ({len(final_patch)} chars) | Total time: {total_time:.2f}s")
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

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 3) -> dict | None:
        
        messages = [
            {"role": "system", "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role": "user", "content": json_string}
        ]
        
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
    def make_request(cls, messages: list[dict], model: Types.Model = KIMI_MODEL_NAME, temperature: float=0.0, attempt: int=5) -> str:
        
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
                tool_output = "❌ Failed to run tool:\n" + str(exc)
               
            return tool_output

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
                raise ValueError(f"Tool input does not match schema.\ntool_input: {tool_input}\ninput_schema: {self.input_schema}")

    def __init__(self):
        """Initialize the tool manager."""
        self._tools: Dict[str, ToolManager.LLMTool] = {}
        self.temp_files: List[str] = []
        self.saved_context: List[dict] = []

    def add_temp_file(self, file_path: str):
        self.temp_files.append(file_path)

    def register_tool(self, tool: 'LLMTool'):
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional['LLMTool']:
        return self._tools.get(name)
    
    def get_tool_docs(self) -> str:
        _docs: list[str] = []
        for tool in self._tools.values():
            _tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            _docs.append(json.dumps(_tool, indent=2))
        return textwrap.dedent("\n\n".join(_docs))
    
    def get_saved_context(self) -> str:
        """Get formatted saved context for LLM calls."""
        if not self.saved_context:
            return ""
        
        context_lines = ["<saved_context>"]
        context_lines.append("**Previous Steps Summary:**")
        
        for entry in self.saved_context:
            context_lines.append(f"\n**Step {entry['step_number']}: {entry['step_name']}**")
            context_lines.append(f"Summary: {entry['summary']}")
            if entry.get('findings'):
                context_lines.append(f"Findings: {entry['findings']}")
            if entry.get('next_steps'):
                context_lines.append(f"Next Steps: {entry['next_steps']}")
        
        context_lines.append("</saved_context>")
        return "\n".join(context_lines)
    
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
            
            **Smart Test Result Parsing:**
            * When running test commands (pytest, unittest, jest, mocha, go test, cargo test, npm test, etc.),
              set the 'is_test_command' parameter to true to get intelligent parsing:
              - Shows test summary (total, passed, failed, errors, skipped)
              - Shows ONLY the failed test details with error messages
              - Hides all passing tests to reduce noise and focus on failures
            * This is especially useful for Pass-to-Pass (P2P) validation where you need to:
              1. First run: is_test_command=false to see full output and understand test framework
              2. Subsequent runs: is_test_command=true to see only failures
            * Example: Running 'pytest tests/' with is_test_command=true on 100 tests (98 pass, 2 fail)
              will show a summary and only the 2 failed tests, not all 98 passing tests.
            * Leave is_test_command=false (default) for non-test commands or when you need full output.
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
                "is_test_command": {
                    "type": "boolean",
                    "description": "Set to true when running test commands to get parsed output showing only failed tests. Set to false (default) for full output or non-test commands.",
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
            is_test_command = tool_input.get("is_test_command", False)
            
            aux_data = {
                "original_command": command,
                "executed_command": command,
                "is_test_command": is_test_command,
            }

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
                    # Check if user explicitly requested test result parsing
                    if is_test_command:
                        # Parse test results to extract summary and failed tests only
                        parsed_results = Utils.parse_test_results(command, result)
                        
                        # Format output to show summary and only failed tests
                        formatted_output = f"""Test Command: {command}

                        Test Summary:
                        Framework: {parsed_results['framework']}
                        Total Tests: {parsed_results['total_tests']}
                        ✓ Passed: {parsed_results['passed']}
                        ✗ Failed: {parsed_results['failed']}
                        ⚠ Errors: {parsed_results['errors']}
                        ⊘ Skipped: {parsed_results['skipped']}
                        """
                        
                        # Add failed test details if any
                        if parsed_results['failed'] > 0 and parsed_results['failed_tests']:
                            formatted_output += "\nFailed Tests (showing only failures):\n"
                            formatted_output += "=" * 80 + "\n"
                            for idx, failed_test in enumerate(parsed_results['failed_tests'], 1):
                                formatted_output += f"\n{idx}. {failed_test['name']}\n"
                                formatted_output += "-" * 80 + "\n"
                                if failed_test['error']:
                                    # Limit error message length to avoid too much output
                                    error_msg = failed_test['error']
                                    if len(error_msg) > 1000:
                                        error_msg = error_msg[:1000] + "\n... (error truncated, run without is_test_command=true for full output)"
                                    formatted_output += f"{error_msg}\n"
                                formatted_output += "-" * 80 + "\n"
                        elif parsed_results['failed'] == 0 and parsed_results['errors'] == 0:
                            formatted_output += "\n✅ All tests passed! No failures to show.\n"
                        
                        # Store full output in aux_data for reference if needed
                        aux_data["full_output"] = result
                        aux_data["parsed_results"] = parsed_results
                        
                        return Types.ToolImplOutput(
                            formatted_output.strip(),
                            f"Test executed - {parsed_results['passed']} passed, {parsed_results['failed']} failed",
                            aux_data | {"success": True, "is_test": True, "parsed": True},
                        )
                    else:
                        # Return raw output (default behavior)
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

    class SaveContextTool(LLMTool):
        """Save context/summary for each step to maintain continuity."""
        name = "save_context"

        description = textwrap.dedent("""
            Save a summary or context for the current step to maintain continuity across the workflow.
            This tool helps track progress and provides context for subsequent steps.
            
            Use this tool to:
            - Save findings from each step
            - Document important discoveries
            - Record decisions made
            - Track progress through the workflow
            - Maintain context for complex multi-step processes
        """)
        
        input_schema = {
            "type": "object",
            "properties": {
                "step_number": {
                    "type": "integer",
                    "description": "The current step number (1-11)"
                },
                "step_name": {
                    "type": "string",
                    "description": "Name of the current step (e.g., 'Understand Problem', 'Create Test Script')"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of what was accomplished in this step"
                },
                "findings": {
                    "type": "string",
                    "description": "Key findings, discoveries, or important information from this step"
                },
                "next_steps": {
                    "type": "string",
                    "description": "What needs to be done in the next step(s)"
                }
            },
            "required": ["step_number", "step_name", "summary"]
        }

        def __init__(
            self, 
            tool_manager: Optional[ToolManager] = None,
        ):
            super().__init__()
            self.tool_manager = tool_manager

        def run_impl(
            self,
            tool_input: dict[str, Any],
        ) -> Types.ToolImplOutput:
            step_number = tool_input.get("step_number", 0)
            step_name = tool_input.get("step_name", "")
            summary = tool_input.get("summary", "")
            findings = tool_input.get("findings", "")
            next_steps = tool_input.get("next_steps", "")
            
            # Create context entry
            context_entry = {
                "step_number": step_number,
                "step_name": step_name,
                "summary": summary,
                "findings": findings,
                "next_steps": next_steps,
                "timestamp": time.time()
            }
            
            # Save to tool manager's context storage
            if self.tool_manager:
                self.tool_manager.saved_context.append(context_entry)
            
            return ToolUtils.success_response(
                f"Context saved for Step {step_number}: {step_name}",
                summary=f"Step {step_number} context saved"
            )

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
        ) -> Types.ToolImplOutput:
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
                    return Types. ToolImplOutput(
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
                    
                    return Types. ToolImplOutput(
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
                            return Types. ToolImplOutput(
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
                return Types. ToolImplOutput(
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
        ) -> Types.ToolImplOutput:
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
                return Types. ToolImplOutput(
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
            return Types. ToolImplOutput(
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

            return Types. ToolImplOutput(
                success_msg,
                f"The file {path} has been edited.",
                {"success": True},
            )

        def str_replace(
            self, path: Path, old_str: str, new_str: str | None
        ) -> Types.ToolImplOutput:
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

                    return Types. ToolImplOutput(
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

            return Types. ToolImplOutput(
                success_msg,
                f"The file {path} has been edited.",
                {"success": True},
            )

        def insert(
            self, path: Path, insert_line: int, new_str: str
        ) -> Types.ToolImplOutput:
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

            return Types. ToolImplOutput(
                success_msg,
                "Insert successful",
                {"success": True},
            )

        def undo_edit(self, path: Path) -> Types. ToolImplOutput:
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

            return Types. ToolImplOutput(
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


    # ============================================================================
    # CREATE TASK SPECIFIC TOOLS (from miner-261.py)
    # ============================================================================

    class RunCodeTool(LLMTool):
        """Execute Python code for testing (CREATE tasks)."""
        name = "run_code"
        description = textwrap.dedent("""
            Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
            Saves the code at the given file_path and then runs it.
            
            * Returns the stdout/stderr from the executed file
            * Checks for syntax errors before execution
            * Tracks generated test files automatically
            * Returns error message if there are any third party dependencies
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to save and run the code (should be in current directory)",
                },
            },
            "required": ["content", "file_path"],
        }

        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager

        def run_impl(self, tool_input: dict[str, Any]) -> Types.ToolImplOutput:
            content = tool_input["content"]
            file_path = tool_input["file_path"]

            # Check syntax using ToolUtils
            syntax_error = ToolUtils.validate_syntax(content, file_path)
            if syntax_error:
                return syntax_error

            # Save file
            try:
                Path(file_path).write_text(content)
                if self.tool_manager:
                    self.tool_manager.add_temp_file(file_path)
            except Exception as e:
                return ToolUtils.error_response(
                    f"Error saving file: {e}",
                    "Failed to save file",
                    error=str(e)
                )

            # Check for disallowed third-party dependencies using ToolUtils
            has_disallowed, disallowed_modules = ToolUtils.check_dependencies(content, file_path)
            # Note: Currently we just check but don't block execution

            # Run the code using ToolUtils
            return ToolUtils.run_subprocess(["python", file_path], timeout=60)

    class ApplyCodeEditTool(LLMTool):
        """Apply search/replace code edits (CREATE tasks)."""
        name = "apply_code_edit"
        description = textwrap.dedent("""
            Performs targeted text replacement within source files.
            Checks for syntax errors before applying changes.
            
            * Requires exact match for search string
            * Validates Python syntax after replacement
            * Provides clear error messages for multiple matches or no matches
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Target file for modification",
                },
                "search": {
                    "type": "string",
                    "description": "Exact text pattern to locate and replace",
                },
                "replace": {
                    "type": "string",
                    "description": "New text content to substitute",
                },
            },
            "required": ["file_path", "search", "replace"],
        }

        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager

        def run_impl(self, tool_input: dict[str, Any]) -> Types.ToolImplOutput:
            file_path = tool_input["file_path"]
            search = tool_input["search"]
            replace = tool_input["replace"]

            # Check file exists using ToolUtils
            file_error = ToolUtils.validate_file_exists(file_path)
            if file_error:
                return file_error

            # Read original content
            try:
                with open(file_path, 'r') as f:
                    original = f.read()
            except Exception as e:
                return ToolUtils.error_response(
                    f"Error reading file: {e}",
                    "Read failed",
                    error=str(e)
                )

            # Count occurrences
            count = original.count(search)
            
            if count == 0:
                return ToolUtils.error_response(
                    f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.",
                    "Search string not found",
                    "search_not_found"
                )
            elif count > 1:
                return ToolUtils.error_response(
                    f"Error: search string found {count} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.",
                    "Multiple matches found",
                    "multiple_matches",
                    count=count
                )

            # Apply replacement
            new_content = original.replace(search, replace)

            # Check syntax if it's a Python file using ToolUtils
            if file_path.endswith('.py'):
                syntax_error = ToolUtils.validate_syntax(new_content, file_path)
                if syntax_error:
                    return syntax_error

            # Save file
            try:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                logger.info("Code edit applied successfully")
                return ToolUtils.success_response(
                    "ok, code edit applied successfully",
                    "Edit successful"
                )
            except Exception as e:
                return ToolUtils.error_response(
                    f"Error saving file: {e}",
                    "Save failed",
                    error=str(e)
                )

    class GetFileContentTool(LLMTool):
        """Read file content with optional line range (CREATE tasks)."""
        name = "get_file_content"
        description = textwrap.dedent("""
            Retrieves file contents with optional filtering based on line numbers.
            
            * Can read entire file or specific line range
            * Returns content with line numbers
            * Supports Python files
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Filesystem path to target file",
                },
                "search_start_line": {
                    "type": "integer",
                    "description": "Optional start line number (1-indexed)",
                },
                "search_end_line": {
                    "type": "integer",
                    "description": "Optional end line number (1-indexed)",
                },
            },
            "required": ["file_path"],
        }

        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager

        def run_impl(self, tool_input: dict[str, Any]) -> Types.ToolImplOutput:
            file_path = tool_input["file_path"]
            search_start_line = tool_input.get("search_start_line")
            search_end_line = tool_input.get("search_end_line")

            # Check file exists using ToolUtils
            file_error = ToolUtils.validate_file_exists(file_path)
            if file_error:
                return file_error

            try:
                with open(file_path, 'r') as f:
                    if search_start_line is not None or search_end_line is not None:
                        lines = f.readlines()
                        start = max(0, (search_start_line or 1) - 1)
                        end = min(len(lines), search_end_line or len(lines))
                        content = ''.join(lines[start:end])
                        output = f"Lines {start+1}-{end} of {file_path}:\n{content}"
                    else:
                        content = f.read()
                        output = f"Content of {file_path}:\n{content}"

                return ToolUtils.success_response(
                    output,
                    "File read successfully"
                )
            except Exception as e:
                return ToolUtils.error_response(
                    f"Error reading file: {e}",
                    "Read failed",
                    error=str(e)
                )

    class RunPythonFileTool(LLMTool):
        """Execute a Python file (CREATE tasks)."""
        name = "run_python_file"
        description = textwrap.dedent("""
            Runs any existing python file.
            
            * Executes the file with Python interpreter
            * Returns stdout/stderr from execution
            * Returns error if file doesn't exist
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the python file to run",
                },
            },
            "required": ["file_path"],
        }

        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager

        def run_impl(self, tool_input: dict[str, Any]) -> Types.ToolImplOutput:
            file_path = tool_input["file_path"]

            # Check file exists using ToolUtils
            file_error = ToolUtils.validate_file_exists(file_path)
            if file_error:
                return file_error

            # Run the file using ToolUtils
            return ToolUtils.run_subprocess(["python", file_path], timeout=60)

    class SearchInFileTool(LLMTool):
        """Search for text patterns in a file (CREATE tasks)."""
        name = "search_in_specified_file_v2"
        description = textwrap.dedent("""
            Locates text patterns within a specific Python file.
            
            * Searches for text patterns or function/class names
            * Returns matching locations with line numbers
            * Only works with Python files
        """)

        input_schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Target Python file for pattern matching",
                },
                "search_term": {
                    "type": "string",
                    "description": "Text pattern to find (e.g., 'def test_function', 'SomeClass')",
                },
            },
            "required": ["file_path", "search_term"],
        }

        def __init__(self, tool_manager: Optional[ToolManager] = None):
            super().__init__()
            self.tool_manager = tool_manager

        def run_impl(self, tool_input: dict[str, Any]) -> Types.ToolImplOutput:
            file_path = tool_input["file_path"]
            search_term = tool_input["search_term"]

            if not file_path.endswith(".py"):
                return ToolUtils.error_response(
                    f"Error: file '{file_path}' is not a python file.",
                    "Not a Python file",
                    "invalid_file_type"
                )

            # Check file exists using ToolUtils
            file_error = ToolUtils.validate_file_exists(file_path)
            if file_error:
                return file_error

            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                matches = []
                for i, line in enumerate(lines, 1):
                    if search_term in line:
                        matches.append(f"Line {i}: {line.rstrip()}")

                if matches:
                    output = f"Found {len(matches)} match(es) in {file_path}:\n" + "\n".join(matches[:50])
                    if len(matches) > 50:
                        output += f"\n... and {len(matches) - 50} more matches"
                else:
                    output = f"No matches found for '{search_term}' in {file_path}"

                return ToolUtils.success_response(
                    output,
                    "Search completed",
                    matches=len(matches)
                )

            except Exception as e:
                return ToolUtils.error_response(
                    f"Error searching file: {e}",
                    "Search failed",
                    error=str(e)
                )

class CreateTaskToolManager(ToolManager):
    """Tool manager specifically for CREATE tasks with miner-261 tools."""
    
    def __init__(self):
        """Initialize CreateTaskToolManager with CREATE-specific tools."""
        # Call parent init but don't register default tools yet
        super().__init__()
        # Clear default tools
        self._tools.clear()
        # Register CREATE-specific tools
        self._register_create_tools()
    
    def _register_create_tools(self):
        """Register CREATE-specific tools from miner-261."""
        # Register CREATE-specific tools
        self.register_tool(ToolManager.RunCodeTool(tool_manager=self))
        self.register_tool(ToolManager.ApplyCodeEditTool(tool_manager=self))
        self.register_tool(ToolManager.GetFileContentTool(tool_manager=self))
        self.register_tool(ToolManager.RunPythonFileTool(tool_manager=self))
        self.register_tool(ToolManager.SearchInFileTool(tool_manager=self))
        self.register_tool(ToolManager.CompleteTool(tool_manager=self))

class FixTaskToolManager(ToolManager):
    """Tool manager specifically for CREATE tasks with miner-261 tools."""
    
    def __init__(self):
        """Initialize CreateTaskToolManager with CREATE-specific tools."""
        # Call parent init but don't register default tools yet
        super().__init__()
        # Clear default tools
        self._tools.clear()
        # Register CREATE-specific tools
        self._register_fix_tools()
    
    def _register_fix_tools(self):
        """Register all default tools."""
        # Register BashTool
        self.register_tool(ToolManager.BashTool(tool_manager=self))
        # Register CompleteTool
        self.register_tool(ToolManager.CompleteTool())
        # Register SaveContextTool
        self.register_tool(ToolManager.SaveContextTool(tool_manager=self))
        # Register SequentialThinkingTool
        self.register_tool(ToolManager.SequentialThinkingTool(tool_manager=self))
        # Register StrReplaceEditorTool
        self.register_tool(ToolManager.StrReplaceEditorTool(tool_manager=self))

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

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    """
    Main agent function that generates a git patch to solve the problem statement.
    
    Args:
        input_dict: Dictionary containing the problem statement and other parameters
        repo_dir: Path to the repository directory
        test_mode: Whether to run in test mode
    
    Returns:
        str: Git patch as a string
    """
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, MAX_TEST_PATCH_TIMEOUT, RUN_ID
    
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
    
    if problem_type == ProblemTypeClassifier.PROBLEM_TYPE_FIX:
        # Use ToolManager with BugFix tools for FIX tasks
        tool_manager = FixTaskToolManager()
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
        tool_manager = CreateTaskToolManager()
        create_problem_task = CreateProblemSolver(problem_statement, tool_manager).solve_problem()
        try:
            result = asyncio.run(asyncio.wait_for(create_problem_task, timeout=2280))
        except asyncio.TimeoutError as e:
            logger.error(f"Timed out after 2280 seconds..")
            result = Utils.create_final_git_patch(tool_manager.temp_files)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            result = Utils.create_final_git_patch(tool_manager.temp_files)
            
    os.system("git reset --hard")  
    logger.info(f"🌕🌕🌕 Final result: {result}")
    return result
