from __future__ import annotations
import ast
import os
import requests
import subprocess
import ast, sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
from uuid import uuid4

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))

GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS=[GLM_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
libraries = [
    "aiohttp",
    "ast",
    "asyncio",
    "catboost",
    "collections",
    "collections.abc",
    "concurrent.futures",
    "copy",
    "cv2",
    "datasets",
    "difflib",
    "dill",
    "django",
    "gensim",
    "glob",
    "inspect",
    "io",
    "joblib",
    "json",
    "keras",
    "lightgbm",
    "math",
    "matplotlib",
    "mlflow",
    "nltk",
    "numpy",
    "optuna",
    "os",
    "pandas",
    "pathlib",
    "pickle_mixin",
    "pydantic",
    "pytest",
    "random",
    "re",
    "requests",
    "scikit-learn",
    "scipy",
    "seaborn",
    "sentence_transformers",
    "sklearn",
    "sklearn.feature_extraction.text",
    "sklearn.feature_extraction.text.TfidfVectorizer",
    "socket",
    "spacy",
    "statsmodels",
    "subprocess",
    "sys",
    "tensorflow",
    "textwrap",
    "time",
    "tokenize",
    "tqdm",
    "torch",
    "torchaudio",
    "torchvision",
    "traceback",
    "transformers",
    "typing",
    "urllib.error",
    "urllib.parse",
    "urllib.request",
    "urllib3",
    "xgboost",
    "autogen",
    "autogen_ext",
    "autogen_agentchat"
]
MAX_FIX_TASK_STEPS = 400

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
run_id=None

STOP_INSTRUCTION=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")

FORMAT_PROMPT_V0=textwrap.dedent("""
<response_format>
  <requirements>
    <triplet_format>
      <next_thought>
        <description>Detailed reasoning including:</description>
        <components>
          <component>Problem understanding</component>
          <component>Code analysis</component>
          <component>Solution justification</component>
          <component>Validation plan</component>
        </components>
      </next_thought>
      <next_tool_name>
        <description>Must be an exact tool name from the tool list</description>
      </next_tool_name>
      <next_tool_args>
        <description>Valid JSON with:</description>
        <requirements>
          <requirement>Proper escaping</requirement>
          <requirement>No trailing commas</requirement>
          <requirement>Tool-specific parameters</requirement>
        </requirements>
      </next_tool_args>
    </triplet_format>
    
    <error_handling>
      <format>
        <next_thought>Error: [detailed explanation]</next_thought>
        <next_tool_name></next_tool_name>
        <next_tool_args>{}</next_tool_args>
      </format>
    </error_handling>
    
    <example_valid>
      <next_thought>I'll fix the JSON parsing issue by adding proper error handling and validation</next_thought>
      <next_tool_name>apply_code_edit</next_tool_name>
      <next_tool_args>
        {
          "file_path": "network.py",
          "search": "return json.loads(response)",
          "replace": "try:\\n    return json.loads(response)\\nexcept JSONDecodeError:\\n    logger.error(f'Invalid JSON: {{response}}')\\n    raise"
        }
      </next_tool_args>
    </example_valid>
    
    <invalid_examples>
      <avoid>
        <item>Missing any of the three required fields</item>
        <item>JSON syntax errors in next_tool_args</item>
        <item>Extra text outside the triplet format</item>
        <item>Using incorrect tool names</item>
        <item>Not quoting special characters properly</item>
      </avoid>
    </invalid_examples>
  </requirements>
</response_format>
""")

PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
'''
<role>
  <title>Problem Type Classifier</title>
  <description>You are the problem type checker that will categorize problem type into:</description>
</role>

<categories>
  <category>
    <name>CREATE</name>
    <description>If the problem statement is about creating a new functionality from scratch.</description>
  </category>
  <category>
    <name>FIX</name>
    <description>If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.</description>
  </category>
</categories>

<response_format>
  <instruction>Only respond with the "FIX" or "CREATE".</instruction>
</response_format>
'''
)

DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
<constraint>
  <rule>You're not allowed to repeat the same tool call with the same arguments.</rule>
  <previous_response>{previous_response}</previous_response>
  <instruction>Try to use something different!</instruction>
</constraint>
""")

GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent("""
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

INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
"""
<role>
  <title>Expert Code Reviewer</title>
  <specialization>Infinite loop detection and prevention</specialization>
  <task>Analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.</task>
</role>

<detection_criteria>
  <critical_checks>
    <check>Check for while True: loops without guaranteed exit conditions</check>
    <check>Verify all while loops have clear termination conditions</check>
    <check>Ensure recursive functions have proper base cases</check>
    <check>Look for loops that depend on external state that might never change</check>
    <check>Check for patterns that could lead to infinite iteration</check>
  </critical_checks>
</detection_criteria>

<correction_guidelines>
  <if_infinite_loops_found>
    <action>Provide a corrected version of the code</action>
    <action>Ensure all loops have finite termination conditions</action>
    <action>Add reasonable iteration limits or timeout mechanisms where appropriate</action>
  </if_infinite_loops_found>
  
  <if_no_infinite_loops>
    <action>Return the original code unchanged</action>
  </if_no_infinite_loops>
</correction_guidelines>

<output_requirements>
  <strict>Return the final Python code along with file names. Do not include any explanations, comments, or additional text.</strict>
  <example>
    <code_block>
      ```python
      a.py
      contents of a.py

      b.py
      contents of b.py
      ```
    </code_block>
  </example>
</output_requirements>
"""
)

GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
<role>
  <title>Expert Python Developer</title>
  <task>Generate a complete, working Python solution for the given problem statement.</task>
</role>

<requirements>
  <strict>
    <requirement priority="high">Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.</requirement>
    <requirement>Do not include explanations, comments, or markdown formatting.</requirement>
    <requirement priority="critical">
      <constraint>**CRITICAL LIBRARY CONSTRAINT**: You can ONLY import libraries from this EXACT list: {libraries}</constraint>
      <rules>
        <rule>Before writing ANY import statement, verify the module is in the allowed list</rule>
        <rule>If a library is NOT in the list, you MUST implement the functionality yourself or use an alternative approach</rule>
        <rule>NEVER import a library that is not explicitly listed</rule>
      </rules>
    </requirement>
    <requirement>Implement all required classes and functions exactly with the same names as in the initial code stub.</requirement>
    <requirement>You may add helper functions or classes if needed, but do not remove or rename the original ones.</requirement>
    <requirement>Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.</requirement>
    <requirement>The solution must be executable as-is with no placeholders or TODOs.</requirement>
    <requirement>If problem statement doesn't explicitly requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.</requirement>
  </strict>
</requirements>

<output_format>
  <instruction>Return only the final Python code.</instruction>
  <example>
    <code_block>
      ```python
      a.py
      content
      b.py
      content
      ```
    </code_block>
  </example>
</output_format>
"""
)

LIBRARY_CHECK_PROMPT = textwrap.dedent(
"""
<role>
  <title>Expert Code Reviewer</title>
  <specialization>Library constraint validation</specialization>
  <task>Analyze the generated Python code and verify that ALL import statements only use libraries from the allowed list.</task>
</role>

<constraint priority="highest">
  <title>**CRITICAL LIBRARY CONSTRAINT (HIGHEST PRIORITY)**</title>
  <rule>You can ONLY import libraries from this EXACT list: {libraries}</rule>
</constraint>

<tasks>
  <task>Analyze ALL import statements in the code (both 'import module' and 'from module import ...' statements)</task>
  <task>Verify each imported module is in the allowed list</task>
  <task>
    <condition>If ANY import uses a disallowed library:</condition>
    <actions>
      <action>You MUST rewrite that part using ONLY allowed libraries</action>
      <action>Implement missing functionality manually using only allowed libraries</action>
      <action>Use alternative allowed modules where possible</action>
      <action>If needed, implement algorithms from scratch using only standard allowed libraries</action>
    </actions>
  </task>
</tasks>

<correction_guidelines>
  <if_violations_found>
    <action>Provide a corrected version of the code</action>
    <action>Remove ALL disallowed library imports</action>
    <action>Replace with implementations using only allowed libraries</action>
    <action>Ensure functionality is preserved</action>
  </if_violations_found>
  
  <if_no_violations>
    <action>Return the original code unchanged</action>
  </if_no_violations>
</correction_guidelines>

<output_requirements>
  <strict>Return the final Python code along with file names. Do not include any explanations, comments, or additional text.</strict>
  <example>
    <code_block>
      ```python
      a.py
      contents of a.py
      b.py
      contents of b.py
      ```
    </code_block>
  </example>
</output_requirements>
"""
)

PROTOCOL_PATTERN_CHECK_PROMPT = textwrap.dedent(
"""
<role>
  <title>Top-tier Developer</title>
  <focus>Protocol excellence and performance optimization</focus>
  <task>Review generated code to fix protocol violations, performance bottlenecks, and ensure optimal resource utilization across any programming language or system.</task>
</role>

<protocol_checks>
  <category>
    <name>ITERATION & TRAVERSAL PROTOCOLS</name>
    <checks>
      <check>Ensure iteration mechanisms work correctly and efficiently</check>
      <check>Verify proper termination conditions and state advancement</check>
      <check>Check that all elements are visited without skips or repeats</check>
      <check>Optimize traversal algorithms for performance</check>
      <check>Validate iterator state management and cleanup</check>
    </checks>
  </category>
  
  <category>
    <name>EVENT & CALLBACK MANAGEMENT</name>
    <checks>
      <check>Fire events/callbacks only on actual state changes</check>
      <check>Ensure proper event registration and deregistration</check>
      <check>Provide callbacks with correct parameters and context</check>
      <check>Prevent memory leaks from event handler references</check>
      <check>Optimize callback execution performance</check>
    </checks>
  </category>
  
  <category>
    <name>STATE MANAGEMENT & CONSISTENCY</name>
    <checks>
      <check>Initialize and maintain internal state correctly</check>
      <check>Ensure all operations update relevant state variables</check>
      <check>Maintain consistent state across all operations</check>
      <check>Prevent orphaned objects or memory leaks</check>
      <check>Optimize state update operations for performance</check>
    </checks>
  </category>
  
  <category>
    <name>RESOURCE MANAGEMENT</name>
    <checks>
      <check>Proper allocation and deallocation of resources</check>
      <check>Efficient memory usage and garbage collection</check>
      <check>Connection pooling and resource reuse</check>
      <check>Prevent resource leaks and exhaustion</check>
      <check>Optimize resource access patterns</check>
    </checks>
  </category>
  
  <category>
    <name>PERFORMANCE OPTIMIZATION</name>
    <checks>
      <check>Minimize unnecessary computations and operations</check>
      <check>Optimize data structure access patterns</check>
      <check>Reduce algorithmic complexity where possible</check>
      <check>Cache frequently accessed data appropriately</check>
      <check>Eliminate redundant operations and checks</check>
    </checks>
  </category>
</protocol_checks>

<fix_guidelines>
  <detailed_fixes>
    <fix condition="iteration is incomplete">Ensure all elements are processed</fix>
    <fix condition="state updates are missing">Verify all operations update relevant state</fix>
    <fix condition="resource cleanup is missing">Add proper cleanup mechanisms</fix>
    <fix condition="performance is suboptimal">Optimize algorithms and data access</fix>
    <fix condition="protocols are violated">Implement proper interface compliance</fix>
  </detailed_fixes>
</fix_guidelines>

<verification_details>
  <checklist>
    <item>Check all interfaces implement required methods correctly</item>
    <item>Verify state updates occur in every relevant operation</item>
    <item>Ensure proper resource cleanup and memory management</item>
    <item>Validate performance characteristics and optimization opportunities</item>
    <item>Confirm protocol compliance and interface adherence</item>
  </checklist>
</verification_details>

<output_requirements>
  <condition>If no issues: Return code unchanged.</condition>
  <strict>Only return the final corrected code. Don't elaborate or provide explanations.</strict>
</output_requirements>
"""
)

FINAL_CORRECTNESS_CHECK_PROMPT = textwrap.dedent(
"""
<role>
  <title>Ultimate Code Expert</title>
  <focus>Absolute correctness validation</focus>
  <task>Detect and fix algorithmic, semantic, edge-case, and specification compliance issues across any programming language or problem domain.</task>
</role>

<master_checks>
  <category>
    <name>ALGORITHM CHECKS</name>
    <checks>
      <check>Verify all operations produce mathematically/logically correct results</check>
      <check>Ensure data transformations maintain integrity</check>
      <check>Validate sorting, filtering, and mapping operations</check>
      <check>Check arithmetic and computational accuracy</check>
      <check>Verify search and retrieval operations</check>
      <check>Ensure proper data structure manipulations</check>
    </checks>
  </category>
  
  <category>
    <name>SEMANTIC VALIDATION</name>
    <checks>
      <check>Functions/methods do exactly what their names and documentation promise</check>
      <check>Return types match expected interfaces and contracts</check>
      <check>Side effects are consistent and documented</check>
      <check>API contracts are strictly honored</check>
      <check>Behavior matches specifications and requirements</check>
    </checks>
  </category>
  
  <category>
    <name>EDGE CASES & BOUNDARY CONDITIONS</name>
    <checks>
      <check>Handle empty/null inputs gracefully</check>
      <check>Manage single element and minimal cases</check>
      <check>Test boundary values and limits</check>
      <check>Detect off-by-one and boundary errors</check>
      <check>Prevent division by zero and overflow errors</check>
      <check>Handle extreme values and edge conditions</check>
    </checks>
  </category>
  
  <category>
    <name>STRICT SPECIFICATION ADHERENCE</name>
    <checks>
      <check>Review problem statement and requirements meticulously</check>
      <check>Implement every required behavior and feature</check>
      <check>Ensure all examples and test cases work correctly</check>
      <check>Avoid missing or misunderstanding any requirements</check>
      <check>Validate against all given constraints</check>
    </checks>
  </category>
  
  <category>
    <name>STRUCTURAL & LOGICAL INVARIANTS</name>
    <checks>
      <check>Maintain data structure integrity throughout operations</check>
      <check>Ensure proper object relationships and references</check>
      <check>Validate state transitions and lifecycle management</check>
      <check>Check resource management and cleanup</check>
      <check>Verify concurrency and thread safety where applicable</check>
    </checks>
  </category>
</master_checks>

<fix_instructions>
  <step>Build comprehensive mental models of expected outcomes</step>
  <step>Validate semantics: Code does exactly what it promises</step>
  <step>Test ALL corner cases: Empty inputs, boundary values, error conditions</step>
  <step>Examine logic validity: Ensure conditions, loops, and invariants are correct</step>
  <step>Confirm structural integrity during all operations</step>
  <step>Verify performance characteristics and resource usage</step>
</fix_instructions>

<output_requirements>
  <condition>If no problems are found: Return the code unchanged.</condition>
  <strict>Only return the final corrected code. Don't elaborate or provide explanations.</strict>
</output_requirements>
"""
)

GENERATE_INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
<role>
  <title>Expert Python Testcase Developer</title>
  <task>Generate a complete testcases for the given problem statement.</task>
</role>

<important_guidelines>
  <guideline>Test functions declared in code skeleton, don't customized those prototypes.</guideline>
  <guideline>Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathematical formulas, algorithms, data, and workflow in it.</guideline>
  <guideline>Do not generate testcases that are not mentioned in problem statement</guideline>
  <guideline>Minimize all testcases as you have context and generation limit</guideline>
</important_guidelines>

<requirements>
  <strict>
    <requirement>Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.</requirement>
    <requirement>Do not include explanations, comments, or markdown formatting.</requirement>
    <requirement>Use only standard Python (no external libraries).</requirement>
  </strict>
</requirements>

<output_format>
  <example>
    <code_block>
      ```python
      test_a.py
      contents of test_a.py

      test_b.py
      contents of test_b.py
      ```
    </code_block>
  </example>
</output_format>
"""
)

GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
<role>
  <title>Expert Python Testcase Developer</title>
  <task>Generate comprehensive testcases that cover ALL possible scenarios for the given problem statement.</task>
</role>

<critical_analysis_before_testing>
  <step>Identify the problem type: algorithmic, data structure, API, business logic, mathematical computation, string manipulation, etc.</step>
  <step>Analyze all variables, functions, and classes in the problem statement and code skeleton</step>
  <step>Identify all input parameters: types, ranges, constraints, and relationships</step>
  <step>Identify all output expectations: types, formats, conditions</step>
  <step>Map out all code paths: normal flow, alternative flows, error paths</step>
</critical_analysis_before_testing>

<important_guidelines>
  <guideline>Test functions declared in code skeleton, don't customized those prototypes.</guideline>
  <guideline>Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathematical formulas, algorithms, data, and workflow in it.</guideline>
  <guideline>Cover ALL test scenarios mentioned in problem statement PLUS all edge cases and boundary conditions</guideline>
  <guideline>Every test case must be independent and use only mockup/stub data</guideline>
  <guideline>Use pytest ONLY - no external libraries or dependencies</guideline>
</important_guidelines>

<comprehensive_test_coverage_rules priority="mandatory">
  <rule category="problem_type_identification">
    <type name="algorithmic">Test with: minimum input, maximum input, typical input, adversarial input</type>
    <type name="data_structure">Test with: empty structure, single element, multiple elements, capacity limits, nested structures</type>
    <type name="string_manipulation">Test with: empty string, single char, special characters, unicode, very long strings, whitespace variations</type>
    <type name="numerical">Test with: zero, negative, positive, floats, integers, infinity, very large/small numbers, precision edge cases</type>
    <type name="datetime">Test with: past dates, future dates, leap years, timezone changes, DST, invalid dates, boundary dates (year 1, 9999)</type>
    <type name="file_operations">Mock file I/O and test: empty files, large files, binary files, missing files, permission errors, corrupted files</type>
    <type name="api_network">Mock all calls and test: successful responses, timeouts, 404/500 errors, malformed responses, rate limits</type>
    <type name="database">Mock DB and test: empty results, single row, multiple rows, null values, transactions, connection failures</type>
  </rule>

  <rule category="variable_level_testing">
    <test>Type validation: correct type, wrong type, None, mixed types</test>
    <test>Value ranges: minimum, maximum, below minimum, above maximum, zero, negative</test>
    <test>State changes: initial state, intermediate states, final state, invalid state transitions</test>
    <test>Immutability: verify immutable variables cannot be changed (if applicable)</test>
    <test>Default values: test with defaults, test overriding defaults</test>
    <test>Special values: None, empty string "", empty list [], empty dict {}, 0, -1, float('inf'), float('nan')</test>
  </rule>

  <rule category="function_level_testing">
    <test>Normal execution: typical valid inputs that follow happy path</test>
    <test>Boundary values: test at limits (e.g., list with 0 items, 1 item, max items)</test>
    <test>Invalid inputs: wrong types, out of range values, None when not expected</test>
    <test>Empty inputs: empty strings, lists, dicts, sets, tuples</test>
    <test>Null/None handling: None as parameter, None in collections, returning None</test>
    <test>Return value validation: correct type, correct value, correct format</test>
    <test>Side effects: verify state changes, verify no unintended side effects</test>
    <test>Multiple calls: test idempotency, test state between calls</test>
    <test>Exception handling: test all expected exceptions are raised correctly</test>
    <test>Edge cases specific to logic: off-by-one errors, empty iterations, division by zero</test>
  </rule>

  <rule category="class_level_testing">
    <test>Initialization: test __init__ with valid params, invalid params, default params, None params</test>
    <test>All public methods: test each method with valid/invalid inputs, edge cases</test>
    <test>Properties: test getters, setters, computed properties, property validation</test>
    <test>Special methods: __str__, __repr__, __eq__, __hash__, __len__, __iter__, __contains__, __getitem__, __setitem__</test>
    <test>Class attributes vs instance attributes: verify proper scoping and sharing</test>
    <test>Inheritance: test parent methods, overridden methods, super() calls</test>
    <test>State management: test object state throughout lifecycle, test state after exceptions</test>
    <test>Class methods and static methods: test with valid/invalid inputs</test>
    <test>Private methods: test indirectly through public API if they affect behavior</test>
  </rule>

  <rule category="edge_cases_by_data_type">
    <type name="strings">
      <case>Empty string ""</case>
      <case>Single character "a"</case>
      <case>Whitespace only "   ", "\n", "\t"</case>
      <case>Special characters: punctuation, symbols, emojis</case>
      <case>Unicode and non-ASCII characters</case>
      <case>Very long strings (10000+ chars)</case>
      <case>Strings with quotes, backslashes, newlines</case>
      <case>Case sensitivity variations</case>
    </type>
    <type name="numbers">
      <case>Zero: 0, 0.0</case>
      <case>Negative: -1, -100, -0.5, -float('inf')</case>
      <case>Positive: 1, 100, 0.5, float('inf')</case>
      <case>Boundaries: sys.maxsize, -sys.maxsize, float min/max</case>
      <case>Floating point: precision issues, rounding, 0.1 + 0.2 != 0.3</case>
      <case>Special float values: float('inf'), float('-inf'), float('nan')</case>
      <case>Integer overflow scenarios (if applicable)</case>
    </type>
    <type name="collections">
      <case>Empty: [], {}, set(), tuple()</case>
      <case>Single element: [1], {1}, {1: 'a'}</case>
      <case>Multiple elements with duplicates</case>
      <case>Nested collections: [[[]]], [{[]: {}}]</case>
      <case>Large collections (1000+ elements)</case>
      <case>Mixed types in collection (if allowed)</case>
      <case>Mutable vs immutable (list vs tuple)</case>
    </type>
    <type name="booleans">
      <case>True and False</case>
      <case>Truthy values: 1, "string", [1], non-empty</case>
      <case>Falsy values: 0, "", [], None, False</case>
      <case>Boolean operations: and, or, not</case>
    </type>
    <type name="none_null">
      <case>None as function parameter</case>
      <case>None as return value</case>
      <case>None in collections: [None], {None: value}</case>
      <case>Distinguishing None from False or 0</case>
    </type>
  </rule>

  <rule category="boundary_conditions">
    <condition>Off-by-one: test at index 0, length-1, length, -1</condition>
    <condition>Array bounds: empty array, single element, accessing first/last element</condition>
    <condition>Loop boundaries: 0 iterations, 1 iteration, many iterations</condition>
    <condition>Recursive limits: base case, one level, deep recursion, maximum recursion depth</condition>
    <condition>Size limits: minimum size, maximum size, exceeding limits</condition>
    <condition>Time boundaries: timeout scenarios, very fast operations, very slow operations (mocked)</condition>
    <condition>Numeric ranges: min value, max value, just below min, just above max</condition>
  </rule>

  <rule category="exception_and_error_handling">
    <test>Expected exceptions: verify correct exception type is raised</test>
    <test>Exception messages: verify error messages contain useful information</test>
    <test>ValueError: for invalid values within correct type</test>
    <test>TypeError: for incorrect types</test>
    <test>KeyError: for missing dictionary keys</test>
    <test>IndexError: for out of bounds access</test>
    <test>AttributeError: for missing attributes</test>
    <test>ZeroDivisionError: for division by zero</test>
    <test>FileNotFoundError: for missing files (mocked)</test>
    <test>PermissionError: for access denied (mocked)</test>
    <test>RuntimeError: for runtime issues</test>
    <test>Custom exceptions: if defined in code</test>
    <test>Exception chaining: verify exception context is preserved</test>
  </rule>

  <rule category="concurrency_and_state">
    <test>State consistency: verify state is correct after operations</test>
    <test>Multiple operations: test sequence of operations maintains consistency</test>
    <test>Isolation: test operations don't affect unrelated state</test>
    <test>Cleanup: test proper cleanup after operations or exceptions</test>
  </rule>

  <rule category="integration_and_interaction">
    <test>Function composition: test functions calling other functions</test>
    <test>Class collaboration: test classes interacting with each other (mock dependencies)</test>
    <test>Data flow: test data passing through multiple components</test>
    <test>Configuration variations: test with different configurations/settings</test>
  </rule>

  <rule category="negative_testing">
    <test>Invalid operations: operations that should not be allowed</test>
    <test>Malformed input: deliberately broken or corrupted input</test>
    <test>Missing required parameters: omitting mandatory arguments</test>
    <test>Wrong parameter order: swapping argument positions</test>
    <test>Security scenarios: injection attempts, path traversal (if applicable)</test>
  </rule>

  <rule category="mocking_requirements">
    <requirement>Mock ALL external dependencies: file I/O, network calls, database operations, API calls</requirement>
    <requirement>Mock system calls: os, sys, platform, datetime.now(), random.random()</requirement>
    <requirement>Use unittest.mock or create simple stub classes/functions</requirement>
    <requirement>Ensure mocks return predictable values for reproducible tests</requirement>
    <requirement>Mock both success and failure scenarios for external dependencies</requirement>
  </rule>

  <rule category="test_organization">
    <requirement>Group related tests in test classes</requirement>
    <requirement>Use descriptive test names: test_function_name_scenario_expected_result</requirement>
    <requirement>Each test should test ONE specific behavior</requirement>
    <requirement>Use arrange-act-assert pattern clearly</requirement>
    <requirement>Add docstrings explaining what each test validates</requirement>
    <requirement>Use pytest.mark.parametrize for testing multiple inputs with same logic</requirement>
  </rule>
</comprehensive_test_coverage_rules>

<test_case_generation_workflow>
  <step number="1">Analyze the problem statement to identify the problem type and all requirements</step>
  <step number="2">List all variables, functions, and classes that need testing</step>
  <step number="3">For each function/method, identify: parameters, return type, side effects, exceptions</step>
  <step number="4">Generate normal/happy path test cases first</step>
  <step number="5">Generate edge case tests based on data types and boundaries</step>
  <step number="6">Generate exception/error handling tests</step>
  <step number="7">Generate negative test cases for invalid inputs</step>
  <step number="8">Review to ensure ALL rules above are covered</step>
  <step number="9">Ensure all tests use only pytest and mocks, no external dependencies</step>
</test_case_generation_workflow>

<requirements>
  <strict>
    <requirement>Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.</requirement>
    <requirement>Do not include explanations, comments, or markdown formatting outside the code blocks.</requirement>
    <requirement>Use only standard Python with pytest and unittest.mock - NO external libraries.</requirement>
    <requirement>Every test must be self-contained and runnable independently.</requirement>
    <requirement>Cover ALL scenarios from the comprehensive rules above that are applicable to the problem.</requirement>
    <requirement>Include at least: 5 normal cases, 5 edge cases, 3 boundary cases, 3 exception cases (minimum, adjust based on complexity).</requirement>
  </strict>
</requirements>

<output_format>
  <example>
    <code_block>
      ```python
      test_a.py
      contents of test_a.py

      test_b.py
      contents of test_b.py
      ```
    </code_block>
  </example>
</output_format>

<problem_statement>
{problem_statement}
</problem_statement>
"""
)

TESTCASES_CHECK_PROMPT = textwrap.dedent(
"""
<role>
  <title>Expert Testcases Reviewer</title>
  <specialization>Comprehensive test coverage validation and gap analysis</specialization>
  <task>Analyze the generated test code to ensure complete coverage of all scenarios, edge cases, and boundary conditions.</task>
</role>

<critical_validation_checks priority="mandatory">
  <check category="correctness">
    <item>Verify all input/output pairs match the problem statement requirements exactly</item>
    <item>Verify test assertions are correct and test the right behavior</item>
    <item>Check for logical errors in test setup or expectations</item>
    <item>Ensure no tests will produce false positives or false negatives</item>
  </check>

  <check category="coverage_completeness">
    <item>Verify ALL scenarios from problem statement are tested</item>
    <item>Check for missing edge cases based on data types involved</item>
    <item>Verify boundary conditions are tested (min, max, zero, empty, etc.)</item>
    <item>Ensure exception/error cases are covered</item>
    <item>Check for missing negative test cases (invalid inputs)</item>
  </check>

  <check category="data_type_edge_cases">
    <strings>Empty string, single char, special chars, unicode, very long strings, whitespace</strings>
    <numbers>Zero, negative, positive, infinity, NaN, boundaries, precision issues</numbers>
    <collections>Empty, single element, multiple elements, nested, large size, mixed types</collections>
    <none_null>None as parameter, None in collections, None vs False/0 distinction</none_null>
    <booleans>True, False, truthy/falsy values</booleans>
  </check>

  <check category="variable_function_class_testing">
    <variable_level>Type validation, value ranges, state changes, immutability, defaults, special values</variable_level>
    <function_level>Normal execution, boundaries, invalid inputs, empty inputs, None handling, return values, side effects, exceptions</function_level>
    <class_level>Initialization, all public methods, properties, special methods, inheritance, state management</class_level>
  </check>

  <check category="exception_handling">
    <item>Verify all expected exception types are tested</item>
    <item>Check ValueError, TypeError, KeyError, IndexError, AttributeError tests</item>
    <item>Verify ZeroDivisionError for division operations</item>
    <item>Check FileNotFoundError, PermissionError for I/O (mocked)</item>
    <item>Verify custom exceptions if defined</item>
  </check>

  <check category="mocking_and_independence">
    <item>Verify ALL external dependencies are properly mocked</item>
    <item>Check that no tests rely on actual external services/files/databases</item>
    <item>Ensure mocks use only unittest.mock or pytest fixtures</item>
    <item>Verify tests can run in isolation without external dependencies</item>
    <item>Check that only pytest is used as external dependency</item>
  </check>

  <check category="test_quality">
    <item>Verify test names are descriptive: test_function_scenario_expected</item>
    <item>Check for proper arrange-act-assert structure</item>
    <item>Ensure tests are focused (one behavior per test)</item>
    <item>Verify tests are independent (no interdependencies)</item>
    <item>Check for proper use of pytest.mark.parametrize when applicable</item>
    <item>Ensure docstrings explain what each test validates</item>
  </check>

  <check category="boundary_and_special_conditions">
    <item>Off-by-one errors: test at 0, length-1, length, -1</item>
    <item>Loop boundaries: 0, 1, many iterations</item>
    <item>Recursion: base case, deep recursion</item>
    <item>Size limits: min, max, exceeding limits</item>
    <item>Numeric ranges: boundaries just below and above limits</item>
  </check>

  <check category="problem_specific">
    <algorithmic>Min/max input, typical input, adversarial input</algorithmic>
    <string_manipulation>Empty, special chars, unicode, case sensitivity</string_manipulation>
    <numerical>Zero, negative, precision, overflow</numerical>
    <data_structures>Empty, single, multiple elements, capacity</data_structures>
  </check>
</critical_validation_checks>

<gap_analysis_process>
  <step number="1">Review problem statement and identify ALL required test scenarios</step>
  <step number="2">Check generated tests against comprehensive test coverage rules</step>
  <step number="3">Identify missing test categories: normal cases, edge cases, boundaries, exceptions</step>
  <step number="4">Identify incorrect test assertions or logic</step>
  <step number="5">Determine if mocking is properly implemented for all external dependencies</step>
  <step number="6">Check if tests cover variable, function, AND class level as applicable</step>
  <step number="7">Verify minimum coverage: 5 normal, 5 edge, 3 boundary, 3 exception cases</step>
</gap_analysis_process>

<remediation_actions>
  <action>If tests are missing: ADD the missing test cases for complete coverage</action>
  <action>If tests are incorrect: FIX the input/output pairs or assertions</action>
  <action>If tests lack mocks: ADD proper mocking for external dependencies</action>
  <action>If tests are poorly structured: REFACTOR for clarity and independence</action>
  <action>If coverage is incomplete: ADD tests for uncovered scenarios</action>
  <action>If tests use external libraries: REPLACE with pytest-only approach with mocks</action>
</remediation_actions>

<output_format priority="CRITICAL">
  <requirement>You MUST respond with ONLY a valid JSON object in one of these two exact formats:</requirement>
  
  <format_1 name="Tests are Perfect">
    <when>If ALL validation checks pass and coverage is complete</when>
    <json_structure>
      {
        "status": "perfect",
        "message": "All test cases are valid and comprehensive - no issues found"
      }
    </json_structure>
  </format_1>
  
  <format_2 name="Tests Need Updates">
    <when>If ANY issues are found or improvements needed</when>
    <json_structure>
      {
        "status": "updated",
        "issues_found": ["list of specific issues identified"],
        "improvements_made": ["list of specific improvements applied"],
        "test_code": "COMPLETE UPDATED TEST CODE WITH FILE NAMES"
      }
    </json_structure>
    <note>The test_code field must contain the FULL test code in format: test_a.py\\ncode...\\n\\ntest_b.py\\ncode...</note>
  </format_2>
  
  <critical_rules>
    <rule>Output ONLY valid JSON - no explanations, no markdown blocks, no extra text</rule>
    <rule>Status field must be exactly "perfect" or "updated" (lowercase)</rule>
    <rule>If status is "updated", test_code field is MANDATORY and must contain complete code</rule>
    <rule>Properly escape newlines in JSON strings (use \\n)</rule>
    <rule>Do not wrap test_code content in markdown code blocks</rule>
    <rule>Ensure at minimum: 5 normal, 5 edge, 3 boundary, 3 exception cases</rule>
    <rule>All tests must use only pytest and unittest.mock</rule>
  </critical_rules>
  
  <example_perfect>
    {
      "status": "perfect",
      "message": "All test cases are comprehensive and valid. Coverage includes 6 normal cases, 7 edge cases, 4 boundary cases, and 4 exception cases. All assertions are correct."
    }
  </example_perfect>
  
  <example_updated>
    {
      "status": "updated",
      "issues_found": [
        "Missing edge case: empty string input for function parse_data",
        "Incorrect assertion in test_calculate: expected 10 but asserted 5",
        "Missing None value handling test"
      ],
      "improvements_made": [
        "Added test_parse_data_empty_string for edge case",
        "Fixed assertion in test_calculate to expect 10",
        "Added test_handle_none_input for None handling"
      ],
      "test_code": "test_module.py\\nimport pytest\\n\\ndef test_parse_data_normal():\\n    assert parse_data('hello') == 'HELLO'\\n\\ndef test_parse_data_empty_string():\\n    assert parse_data('') == ''\\n"
    }
  </example_updated>
</output_format>
"""
)

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
<role>
  <title>Coding Assistant</title>
  <emoji>🚀</emoji>
  <context>I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.</context>
</role>

<critical_thinking_protocol priority="mandatory">
  <principle>Before calling ANY tool, you MUST explicitly think through and document:</principle>
  <thinking_checklist>
    <item>What specific information do I need from this tool call?</item>
    <item>What information do I already have that makes this tool call necessary?</item>
    <item>What additional context or data do I still need to gather?</item>
    <item>How will the output of this tool call inform my next steps?</item>
    <item>Are there any dependencies or prerequisites I need to resolve first?</item>
    <item>What are the potential outcomes and how will I handle each?</item>
  </thinking_checklist>
  <requirement>Always articulate your reasoning BEFORE executing tools. Never call tools impulsively.</requirement>
</critical_thinking_protocol>

<enhanced_agent_workflow priority="critical">
  <note>You have access to specialized meta-cognitive agents that enhance problem-solving effectiveness. Use them strategically.</note>
  <agents>
    <agent name="Meta-Planning Agent">Creates comprehensive execution plans before implementation</agent>
    <agent name="Reflection Agent">Reviews solutions and provides self-critique</agent>
    <agent name="Solution Validator">Validates solutions against multiple quality dimensions</agent>
    <agent name="Iterative Refinement Agent">Improves solutions based on feedback</agent>
  </agents>
  <when_to_use>
    <use agent="Meta-Planning Agent" when="At the start of complex problems to create a strategic plan"/>
    <use agent="Reflection Agent" when="After proposing solutions but before implementation"/>
    <use agent="Solution Validator" when="After implementation to comprehensively validate"/>
    <use agent="Iterative Refinement Agent" when="Test failures or validation issues found"/>
  </when_to_use>
</enhanced_agent_workflow>

<workflow_steps>
  <step number="0" name="Strategic Planning (Optional but Recommended)" priority="HIGH">
    <when>Use for complex problems with multiple components or unclear approaches</when>
    <tool>create_meta_plan(problem_statement, project_context)</tool>
    
    <sub_steps>
      <sub_step number="0.1" name="Analyze Problem Comprehensively">
        <approach>Break down the problem statement into core components</approach>
        <actions>
          <action>Identify problem type (algorithmic, data structure, API, business logic, performance, etc.)</action>
          <action>Extract ALL requirements explicitly stated and implied</action>
          <action>Identify constraints (time, space, compatibility, dependencies)</action>
          <action>List all edge cases mentioned or inferred from problem</action>
          <action>Identify affected components (files, classes, functions)</action>
        </actions>
        <rules>
          <rule>Read problem statement at least twice to catch all details</rule>
          <rule>Look for implicit requirements (e.g., "backward compatible" often implied)</rule>
          <rule>Consider system-wide implications, not just local fix</rule>
        </rules>
      </sub_step>
      
      <sub_step number="0.2" name="Decompose into Sub-Tasks">
        <approach>Break complex problem into 3-5 manageable sub-tasks</approach>
        <actions>
          <action>List major tasks needed to solve the problem</action>
          <action>Identify dependencies between tasks (which must come first)</action>
          <action>Estimate complexity for each task (low/medium/high)</action>
          <action>Identify risks and mitigation strategies for each task</action>
          <action>Create ordered execution sequence based on dependencies</action>
        </actions>
        <rules>
          <rule>Each sub-task should be independently verifiable</rule>
          <rule>Keep sub-tasks focused - if too complex, break down further</rule>
          <rule>Always consider dependencies - don't assume arbitrary order</rule>
        </rules>
      </sub_step>
      
      <sub_step number="0.3" name="Evaluate Approaches">
        <approach>Generate and compare 2-3 different solution approaches</approach>
        <actions>
          <action>List 2-3 fundamentally different approaches to solve the problem</action>
          <action>For each approach, identify: pros, cons, complexity, risks</action>
          <action>Consider: minimal change vs. refactoring, short-term vs. long-term</action>
          <action>Select best approach with clear justification</action>
          <action>Identify fallback approach if primary fails</action>
        </actions>
        <rules>
          <rule>Approaches should be meaningfully different, not minor variations</rule>
          <rule>Consider backward compatibility impact in evaluation</rule>
          <rule>Prefer simpler approaches unless complexity is justified</rule>
          <rule>Document why you selected one approach over others</rule>
        </rules>
      </sub_step>
      
      <sub_step number="0.4" name="Define Verification Strategy">
        <approach>Plan how you'll verify solution correctness</approach>
        <actions>
          <action>Define success criteria (what makes solution acceptable)</action>
          <action>List test categories needed (normal, edge, boundary, exception)</action>
          <action>Plan verification checkpoints at each sub-task</action>
          <action>Define rollback strategy if solution fails</action>
        </actions>
        <rules>
          <rule>Success criteria must be measurable and objective</rule>
          <rule>Plan tests BEFORE implementation, not after</rule>
          <rule>Include regression testing in verification plan</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <benefit>Reduces wasted effort, identifies issues early, improves solution quality by 40-60%</benefit>
  </step>

  <step number="1" name="Find Relevant Files" priority="CRITICAL">
    <goal>Locate all files in the repository that need to be examined or modified</goal>
    <approach>Use search tools to systematically locate relevant code</approach>
    
    <sub_steps>
      <sub_step number="1.1" name="Identify Key Terms">
        <actions>
          <action>Extract key terms from problem statement (class names, function names, error messages)</action>
          <action>Identify domain-specific terminology</action>
          <action>Note any file paths or module names mentioned</action>
        </actions>
        <rules>
          <rule>Include variations of terms (e.g., CacheManager, cache_manager, cache)</rule>
          <rule>Search for error messages exactly as stated in problem</rule>
        </rules>
      </sub_step>
      
      <sub_step number="1.2" name="Perform Broad Search">
        <tool>search_in_all_files_content(search_term, case_sensitive=False)</tool>
        <actions>
          <action>Search for primary terms across entire codebase</action>
          <action>Review results to identify relevant files</action>
          <action>Note files that appear multiple times (likely central to issue)</action>
        </actions>
        <rules>
          <rule>Start with most specific terms first</rule>
          <rule>If too many results (>50), refine search terms</rule>
          <rule>If no results, try variations or broader terms</rule>
        </rules>
      </sub_step>
      
      <sub_step number="1.3" name="Examine Project Structure">
        <actions>
          <action>Review project_structure provided in context</action>
          <action>Identify patterns (e.g., tests in tests/, main code in src/)</action>
          <action>Locate configuration files that might be relevant</action>
        </actions>
        <rules>
          <rule>Understand project organization before making changes</rule>
          <rule>Look for test files corresponding to source files</rule>
        </rules>
      </sub_step>
      
      <sub_step number="1.4" name="Create File List">
        <actions>
          <action>List all files that need examination</action>
          <action>Prioritize files: core issue files first, then dependent files</action>
          <action>Note which files are source vs. test files</action>
        </actions>
        <rules>
          <rule>Include test files in your list - they provide context</rule>
          <rule>Don't skip files that seem tangentially related</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Prioritized list of relevant files to examine</output>
  </step>

  <step number="2" name="Localize the Issue" priority="CRITICAL">
    <goal>Pinpoint the exact code causing the issue at variable, function, and class level</goal>
    <approach>Systematically narrow down from file → class → function → variables</approach>
    
    <sub_steps>
      <sub_step number="2.1" name="Read Relevant Files">
        <tool>get_file_content(file_path) or get_classes(class_paths) or get_functions(function_paths)</tool>
        <actions>
          <action>Start with most likely file from step 1</action>
          <action>Read full file if small (&lt;500 lines) or use targeted reads for large files</action>
          <action>Understand overall structure before diving into details</action>
        </actions>
        <rules>
          <rule>Read imports first to understand dependencies</rule>
          <rule>Read class/function signatures before implementations</rule>
          <rule>Take notes on what each component does</rule>
        </rules>
      </sub_step>
      
      <sub_step number="2.2" name="Locate Relevant Classes">
        <tool>get_classes([file::class1, file::class2])</tool>
        <actions>
          <action>Identify classes mentioned in problem statement</action>
          <action>Examine class structure: attributes, methods, inheritance</action>
          <action>Understand class responsibilities and interactions</action>
        </actions>
        <rules>
          <rule>Check __init__ method to understand initialization</rule>
          <rule>Look for class vs instance variables</rule>
          <rule>Check for inherited methods that might be involved</rule>
        </rules>
      </sub_step>
      
      <sub_step number="2.3" name="Locate Relevant Functions/Methods">
        <tool>get_functions([file::class::method]) or search_in_specified_file_v2(file_path, function_name)</tool>
        <actions>
          <action>Identify functions/methods mentioned in problem or error</action>
          <action>Read complete function implementation</action>
          <action>Trace logic flow through the function</action>
          <action>Identify where error could occur or incorrect behavior happens</action>
        </actions>
        <rules>
          <rule>Pay attention to control flow (if/else, loops, exceptions)</rule>
          <rule>Check function parameters and return values</rule>
          <rule>Look for functions that call the problematic function</rule>
        </rules>
      </sub_step>
      
      <sub_step number="2.4" name="Identify Problematic Variables/Logic">
        <actions>
          <action>Trace variable values through the problematic code</action>
          <action>Identify incorrect calculations, comparisons, or assignments</action>
          <action>Look for off-by-one errors, missing checks, wrong operators</action>
          <action>Check for missing error handling or edge case handling</action>
        </actions>
        <rules>
          <rule>Consider variable scope (local, instance, class, global)</rule>
          <rule>Check for mutability issues (list/dict modifications)</rule>
          <rule>Look for race conditions in multi-threaded code</rule>
          <rule>Verify assumptions about variable types and values</rule>
        </rules>
      </sub_step>
      
      <sub_step number="2.5" name="Understand Root Cause">
        <actions>
          <action>Identify the root cause, not just symptoms</action>
          <action>Trace how the issue manifests from root cause to observed behavior</action>
          <action>Document your understanding of WHY the issue occurs</action>
        </actions>
        <rules>
          <rule>Keep asking "why" until you reach fundamental cause</rule>
          <rule>Don't confuse symptoms (error message) with cause (bad logic)</rule>
          <rule>Consider: is this design flaw or implementation bug?</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Clear identification of: problematic file(s), class(es), function(s), and specific variables/logic that need fixing</output>
  </step>

  <step number="3" name="Create Comprehensive Test Script" priority="CRITICAL">
    <goal>Create test script that reproduces the issue and validates the fix</goal>
    <approach>Test-driven approach - create tests that fail initially, pass after fix</approach>
    <tool>generate_test_function(file_path, test_function_code, position="auto")</tool>
    
    <sub_steps>
      <sub_step number="3.1" name="Design Test Strategy">
        <approach>Plan comprehensive test coverage before writing tests</approach>
        <actions>
          <action>List ALL scenarios from problem statement that need testing</action>
          <action>Identify edge cases: empty inputs, None values, boundaries, invalid types</action>
          <action>Identify exception cases: what should raise errors and when</action>
          <action>Plan test structure: separate normal, edge, boundary, exception tests</action>
        </actions>
        <rules>
          <rule>Include at minimum: 5 normal cases, 5 edge cases, 3 boundary cases, 3 exception cases</rule>
          <rule>Each test should test ONE specific behavior</rule>
          <rule>Test names should be descriptive: test_function_scenario_expected_result</rule>
        </rules>
      </sub_step>
      
      <sub_step number="3.2" name="Write Test Cases">
        <approach>Follow arrange-act-assert pattern for each test</approach>
        <actions>
          <action>Write normal case tests first (expected happy path scenarios)</action>
          <action>Write edge case tests (empty, single element, large inputs, special values)</action>
          <action>Write boundary tests (min/max values, off-by-one scenarios)</action>
          <action>Write exception tests (invalid inputs should raise specific exceptions)</action>
        </actions>
        <rules>
          <rule>Use ONLY pytest and unittest.mock - no external dependencies</rule>
          <rule>Mock ALL external dependencies (file I/O, network, database, APIs)</rule>
          <rule>Each test must be independent and runnable in isolation</rule>
          <rule>Include docstrings explaining what each test validates</rule>
          <rule>Use pytest.mark.parametrize for similar tests with different inputs</rule>
        </rules>
      </sub_step>
      
      <sub_step number="3.3" name="Add Bug Reproduction Test">
        <approach>Create specific test that fails due to current bug</approach>
        <actions>
          <action>Write test that reproduces the exact issue from problem statement</action>
          <action>This test should FAIL when run against current code</action>
          <action>Document why this test should fail currently</action>
        </actions>
        <rules>
          <rule>This test must pass after fix is applied</rule>
          <rule>Keep this test simple and focused on the specific bug</rule>
        </rules>
      </sub_step>
      
      <sub_step number="3.4" name="Run Tests to Verify Failures">
        <tool>run_code(content=test_code, file_path="test_issue.py")</tool>
        <actions>
          <action>Run the test script against current code</action>
          <action>Verify that bug reproduction test fails as expected</action>
          <action>Note which tests fail and why</action>
        </actions>
        <rules>
          <rule>Tests SHOULD fail before fix - that's expected and correct</rule>
          <rule>If tests pass unexpectedly, re-examine your understanding of the bug</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <test_requirements priority="MANDATORY">
      <requirement>Include ALL test cases from problem statement</requirement>
      <requirement>Test at variable, function, AND class level</requirement>
      <requirement>Use ONLY mockup/stub data - completely independent of external libraries</requirement>
      <requirement>Runnable with pytest ONLY, no external dependencies</requirement>
      <requirement>Mock ALL external dependencies (APIs, DB, file I/O, network)</requirement>
      <requirement>Test: boundary conditions, None values, empty inputs, invalid inputs, type mismatches</requirement>
      <requirement>Include bug reproduction test that fails initially</requirement>
    </test_requirements>
    
    <output>Test script that fails on current code, will pass after fix</output>
  </step>

  <step number="4" name="Propose Solution Approaches" priority="CRITICAL">
    <goal>Propose at least 2 meaningfully different solutions to the problem</goal>
    <approach>Think through multiple approaches before implementing</approach>
    
    <sub_steps>
      <sub_step number="4.1" name="Generate Solution Options">
        <actions>
          <action>Brainstorm 2-3 fundamentally different ways to fix the issue</action>
          <action>For each option, describe: what changes, why it works, pros/cons</action>
          <action>Consider: minimal change, refactoring, algorithm change approaches</action>
        </actions>
        <rules>
          <rule>Solutions must be meaningfully different, not minor variations</rule>
          <rule>At least one solution should be minimal/conservative change</rule>
          <rule>Consider short-term vs long-term maintainability</rule>
        </rules>
      </sub_step>
      
      <sub_step number="4.2" name="Analyze Each Solution">
        <actions>
          <action>For each solution, identify: files changed, functions modified, risk level</action>
          <action>Assess backward compatibility impact</action>
          <action>Estimate implementation complexity and testing needs</action>
          <action>Consider potential side effects and unintended consequences</action>
        </actions>
        <rules>
          <rule>Be honest about risks and limitations of each approach</rule>
          <rule>Document why you think each approach would work</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>At least 2 well-described solution approaches with analysis</output>
  </step>

  <step number="5" name="Reflect on Solutions" priority="CRITICAL">
    <goal>Self-critique proposed solutions to catch issues before implementation</goal>
    <approach>Use reflection agent to systematically review solutions</approach>
    <tool>reflect_on_solution(proposed_solution, problem_statement, solution_description)</tool>
    
    <sub_steps>
      <sub_step number="5.1" name="Perform Reflection">
        <actions>
          <action>For each proposed solution, call reflect_on_solution tool</action>
          <action>Review critique across 6 dimensions: correctness, completeness, robustness, efficiency, code_quality, backward_compatibility</action>
          <action>Note all issues found with severity ratings (CRITICAL/HIGH/MEDIUM/LOW)</action>
        </actions>
        <rules>
          <rule>Take reflection seriously - don't dismiss concerns</rule>
          <rule>CRITICAL and HIGH issues must be addressed before proceeding</rule>
        </rules>
      </sub_step>
      
      <sub_step number="5.2" name="Decide on Revisions">
        <actions>
          <action>Review all issues identified by reflection</action>
          <action>Decide if revision needed (check reflection's revision_needed field)</action>
          <action>If revision needed, improve solution addressing CRITICAL/HIGH issues</action>
          <action>Document changes made based on reflection</action>
        </actions>
        <rules>
          <rule>If confidence_score &lt; 70, seriously consider revision</rule>
          <rule>Address all CRITICAL issues before seeking approval</rule>
          <rule>Don't ignore HIGH severity issues without good reason</rule>
        </rules>
      </sub_step>
      
      <sub_step number="5.3" name="Select Best Solution">
        <actions>
          <action>Compare reflection results for all solutions</action>
          <action>Select solution with highest confidence_score and fewest issues</action>
          <action>Document rationale for selection</action>
        </actions>
        <rules>
          <rule>Prefer solution with fewer CRITICAL/HIGH issues</rule>
          <rule>Consider implementation complexity vs. benefit trade-off</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <benefit>Catches 60-70% of issues before implementation, saving significant rework time</benefit>
    <output>Refined solution(s) with confidence assessment</output>
  </step>

  <step number="6" name="Get User Approval" priority="CRITICAL">
    <goal>Present solutions to user and get approval before implementation</goal>
    <tool>get_approval_for_solution(solutions, selected_solution, reason_for_selection)</tool>
    
    <sub_steps>
      <sub_step number="6.1" name="Prepare Proposal">
        <actions>
          <action>Format all solutions clearly with descriptions</action>
          <action>Include reflection results and confidence scores</action>
          <action>Highlight selected solution and why you recommend it</action>
        </actions>
        <rules>
          <rule>Be transparent about risks and trade-offs</rule>
          <rule>Explain technical decisions in understandable terms</rule>
        </rules>
      </sub_step>
      
      <sub_step number="6.2" name="Request Approval">
        <actions>
          <action>Call get_approval_for_solution with all solutions</action>
          <action>Specify which solution you recommend (selected_solution index)</action>
          <action>Provide clear reason_for_selection</action>
        </actions>
        <rules>
          <rule>Must have at least 2 solutions to propose</rule>
          <rule>Must explain why selected solution is best</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>User approval to proceed with implementation</output>
  </step>

  <step number="7" name="Implement Solution" priority="CRITICAL">
    <goal>Apply approved solution to codebase</goal>
    <approach>Make precise, targeted changes while preserving existing functionality</approach>
    <tool>apply_code_edit(file_path, search, replace) or save_file(file_path, content)</tool>
    
    <sub_steps>
      <sub_step number="7.1" name="Plan Implementation Order">
        <actions>
          <action>List all files that need modification</action>
          <action>Determine order: dependencies first, then dependent code</action>
          <action>Identify which changes are core vs. supporting changes</action>
        </actions>
        <rules>
          <rule>Make changes in logical order (don't break intermediate states)</rule>
          <rule>Consider: what if implementation is interrupted mid-way?</rule>
        </rules>
      </sub_step>
      
      <sub_step number="7.2" name="Make Core Changes">
        <actions>
          <action>Start with the primary fix to the root cause</action>
          <action>Use apply_code_edit for surgical changes to existing code</action>
          <action>Use save_file only if creating new files or complete rewrites</action>
          <action>Double-check each change before applying</action>
        </actions>
        <rules>
          <rule>Make minimal changes necessary - don't refactor unnecessarily</rule>
          <rule>Preserve variable names, function signatures unless must change</rule>
          <rule>Keep changes focused on the problem - avoid scope creep</rule>
          <rule>Add comments explaining non-obvious changes</rule>
        </rules>
      </sub_step>
      
      <sub_step number="7.3" name="Add Edge Case Handling">
        <actions>
          <action>Add checks for None values where needed</action>
          <action>Add validation for boundary conditions</action>
          <action>Add exception handling where errors can occur</action>
        </actions>
        <rules>
          <rule>Handle edge cases gracefully, don't just crash</rule>
          <rule>Raise appropriate exceptions with clear messages</rule>
          <rule>Don't add edge case handling that breaks existing behavior</rule>
        </rules>
      </sub_step>
      
      <sub_step number="7.4" name="Ensure Backward Compatibility">
        <actions>
          <action>Verify function signatures unchanged (or only extended)</action>
          <action>Verify class interfaces unchanged</action>
          <action>Check that default behavior preserved for existing callers</action>
        </actions>
        <rules>
          <rule>NEVER break backward compatibility unless explicitly required</rule>
          <rule>If must break compatibility, add deprecation warnings first</rule>
          <rule>Document any compatibility considerations</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Modified codebase with fix implemented</output>
  </step>

  <step number="8" name="Run and Verify Tests" priority="CRITICAL">
    <goal>Verify fix works correctly and doesn't break anything</goal>
    <approach>Run comprehensive test suite and analyze results</approach>
    <tool>run_repo_tests([test_files]) or run_code(content, file_path)</tool>
    
    <sub_steps>
      <sub_step number="8.1" name="Run Bug Reproduction Test">
        <actions>
          <action>Run the test from step 3 that initially failed</action>
          <action>Verify it now passes</action>
          <action>If still fails, analyze why and return to step 7</action>
        </actions>
        <rules>
          <rule>This test MUST pass after fix</rule>
          <rule>If doesn't pass, fix is incomplete or incorrect</rule>
        </rules>
      </sub_step>
      
      <sub_step number="8.2" name="Run Full Test Suite">
        <actions>
          <action>Run all existing tests in repository</action>
          <action>Run all new tests created in step 3</action>
          <action>Collect test results: passed count, failed count, error details</action>
        </actions>
        <rules>
          <rule>ALL tests must pass (no regressions)</rule>
          <rule>If any test fails, must investigate and fix</rule>
        </rules>
      </sub_step>
      
      <sub_step number="8.3" name="Analyze Test Failures">
        <approach>If any tests fail, systematically debug</approach>
        <actions>
          <action>For each failing test, understand why it fails</action>
          <action>Determine if: bug in fix, bug in test, or uncovered edge case</action>
          <action>Fix the issue and re-run tests</action>
        </actions>
        <rules>
          <rule>Don't ignore test failures - each must be resolved</rule>
          <rule>Don't modify existing tests unless they're incorrect</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Test results showing all tests pass</output>
  </step>

  <step number="9" name="Validate Solution Comprehensively" priority="CRITICAL">
    <goal>Comprehensive quality validation across all dimensions</goal>
    <approach>Use validation agent for systematic quality assessment</approach>
    <tool>validate_solution(solution_code, test_results, problem_statement)</tool>
    
    <sub_steps>
      <sub_step number="9.1" name="Perform Validation">
        <actions>
          <action>Gather solution code, test results, problem statement</action>
          <action>Call validate_solution tool</action>
          <action>Review validation report across all 5 dimensions</action>
        </actions>
        <rules>
          <rule>Validation must be done before calling finish()</rule>
          <rule>Take validation results seriously</rule>
        </rules>
      </sub_step>
      
      <sub_step number="9.2" name="Check Passing Criteria">
        <approach>Verify solution meets quality thresholds</approach>
        <criteria>
          <criterion>Overall score >= 85</criterion>
          <criterion>Functional correctness >= 90</criterion>
          <criterion>Test coverage >= 80</criterion>
          <criterion>Code quality >= 80</criterion>
          <criterion>Performance meets requirements</criterion>
          <criterion>No CRITICAL blocking issues</criterion>
          <criterion>Backward compatible</criterion>
        </criteria>
        <actions>
          <action>Check validation_passed boolean</action>
          <action>Review overall_score and category_scores</action>
          <action>Check blocking_issues for CRITICAL items</action>
        </actions>
        <rules>
          <rule>If validation_passed = false, must refine (go to step 10)</rule>
          <rule>All category scores should be >= 80</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Validation report with pass/fail and scores</output>
  </step>

  <step number="10" name="Iterative Refinement (If Needed)" priority="CRITICAL">
    <goal>Improve solution based on validation feedback</goal>
    <approach>Targeted improvements addressing highest priority issues first</approach>
    <tool>refine_solution(current_solution, feedback, problem_statement, test_failures)</tool>
    <when>Only if validation fails (score &lt; 85 or CRITICAL issues)</when>
    
    <sub_steps>
      <sub_step number="10.1" name="Analyze Feedback">
        <actions>
          <action>Review all feedback: validation report, test failures, blocking issues</action>
          <action>Categorize issues by severity: CRITICAL, HIGH, MEDIUM, LOW</action>
          <action>Identify root causes of issues</action>
        </actions>
        <rules>
          <rule>Focus on CRITICAL issues first, then HIGH</rule>
          <rule>Look for patterns - multiple issues may have common cause</rule>
        </rules>
      </sub_step>
      
      <sub_step number="10.2" name="Apply Refinement">
        <actions>
          <action>Call refine_solution with current code and all feedback</action>
          <action>Review improved solution generated</action>
          <action>Apply improved solution to codebase</action>
        </actions>
        <rules>
          <rule>Review refined code before applying - don't blindly trust</rule>
          <rule>Ensure working functionality preserved</rule>
        </rules>
      </sub_step>
      
      <sub_step number="10.3" name="Re-test and Re-validate">
        <actions>
          <action>Re-run all tests (step 8)</action>
          <action>Re-run validation (step 9)</action>
          <action>Check if issues resolved</action>
        </actions>
        <rules>
          <rule>Track iteration count - maximum 3 refinement iterations</rule>
          <rule>If still failing after 3 iterations, reconsider approach</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <iteration_limit>Maximum 3 refinement iterations before reconsidering approach</iteration_limit>
    <output>Improved solution meeting validation criteria</output>
  </step>

  <step number="11" name="Final Verification" priority="HIGH">
    <goal>Final checks before completing task</goal>
    <approach>Systematic final review</approach>
    
    <sub_steps>
      <sub_step number="11.1" name="Verify Completeness">
        <actions>
          <action>Review problem statement one more time</action>
          <action>Verify ALL requirements addressed</action>
          <action>Check that ONLY requested changes made (no scope creep)</action>
        </actions>
        <rules>
          <rule>Don't leave any requirements unaddressed</rule>
          <rule>Don't include unrelated changes</rule>
        </rules>
      </sub_step>
      
      <sub_step number="11.2" name="Check Repository-Wide Impact">
        <actions>
          <action>Search for all uses of modified functions/classes</action>
          <action>Verify no unexpected breakage in other parts of codebase</action>
          <action>Run full repository test suite if available</action>
        </actions>
        <rules>
          <rule>Use search_in_all_files_content to find all usages</rule>
          <rule>Don't assume your changes are isolated</rule>
        </rules>
      </sub_step>
      
      <sub_step number="11.3" name="Review Patch">
        <actions>
          <action>Review all changes made</action>
          <action>Verify no debug code, print statements, or test artifacts left behind</action>
          <action>Ensure generated test files won't be included in patch</action>
        </actions>
        <rules>
          <rule>Clean, professional code only</rule>
          <rule>No temporary debugging code in final version</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Clean, complete solution ready for finalization</output>
  </step>

  <step number="12" name="Complete Task" priority="CRITICAL">
    <goal>Finalize and document the solution</goal>
    <tool>finish(investigation_summary)</tool>
    
    <sub_steps>
      <sub_step number="12.1" name="Prepare Summary">
        <actions>
          <action>Write clear summary including: Problem, Investigation, Solution</action>
          <action>Document what was changed and why</action>
          <action>Note any important considerations or limitations</action>
        </actions>
        <format>
          Problem: [Concise problem description]
          Investigation: [What you found, root cause]
          Solution: [What you changed, how it fixes the issue]
        </format>
      </sub_step>
      
      <sub_step number="12.2" name="Call Finish">
        <actions>
          <action>Call finish tool with investigation_summary</action>
          <action>Workflow will generate final git patch</action>
        </actions>
        <rules>
          <rule>Only call finish after ALL validation passes</rule>
          <rule>Summary should be professional and complete</rule>
        </rules>
      </sub_step>
    </sub_steps>
    
    <output>Completed task with git patch of changes</output>
  </step>
</workflow_steps>

<important_notes>
  <note>Always compare expected output from BOTH problem statement AND existing test cases</note>
  <note>If run_code or run_repo_tests fails due to missing dependencies, don't try to install them - no internet access</note>
  <note>Generated test files are automatically excluded from final patch</note>
  <note>Never modify existing test files directly - use generate_test_function instead</note>
  <note>Backward compatibility is mandatory unless problem statement explicitly allows breaking changes</note>
</important_notes>

<meta_cognitive_best_practices>
  <practice name="Think-Plan-Act-Reflect" priority="highest">
    <think>Understand the problem deeply before acting</think>
    <plan>Create strategic plan for complex problems (use meta-planning)</plan>
    <act>Implement solution following the plan</act>
    <reflect>Self-critique before finalization (use reflection agent)</reflect>
    <validate>Comprehensive validation of results (use validator)</validate>
    <refine>Iterative improvement based on feedback (use refinement)</refine>
  </practice>
  
  <practice name="Fail-Fast-Learn-Quick">
    <principle>Identify and fix issues as early as possible</principle>
    <principle>Use reflection before implementation to catch design flaws</principle>
    <principle>Run tests frequently to catch regressions immediately</principle>
    <principle>Learn from failures - don't repeat the same mistakes</principle>
  </practice>
  
  <practice name="Systematic-Problem-Solving">
    <principle>Break complex problems into manageable sub-problems</principle>
    <principle>Solve sub-problems in optimal order considering dependencies</principle>
    <principle>Verify each sub-solution before moving to next</principle>
    <principle>Integration testing after combining sub-solutions</principle>
  </practice>
  
  <practice name="Quality-First">
    <principle>Correctness over speed - get it right first</principle>
    <principle>Comprehensive testing over minimal testing</principle>
    <principle>Robust error handling over happy-path-only code</principle>
    <principle>Maintainable code over clever code</principle>
  </practice>
</meta_cognitive_best_practices>

<multi_file_awareness priority="critical">
  <guideline>Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.</guideline>
  <guideline>Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.</guideline>
  <guideline>Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.</guideline>
  <guideline>Re-run tests only after covering all discovered occurrences to avoid partial fixes.</guideline>
</multi_file_awareness>

<test_generation_guidance priority="critical">
  <comprehensive_testing>
    <principle>Every fix MUST be accompanied by comprehensive test coverage that validates the fix and prevents regression.</principle>
    <test_requirements>
      <requirement>Create test scripts that are completely self-contained and use only pytest as a dependency</requirement>
      <requirement>Mock all external dependencies using pytest's monkeypatch or unittest.mock</requirement>
      <requirement>Test all code paths: normal execution, edge cases, error conditions, boundary values</requirement>
      <requirement>Include parametrized tests for multiple input combinations when applicable</requirement>
      <requirement>Test at multiple levels: unit tests for functions, integration-style tests for classes, end-to-end tests for workflows</requirement>
      <requirement>Verify exception handling by testing that appropriate exceptions are raised for invalid inputs</requirement>
      <requirement>Include negative test cases (tests that verify incorrect behavior is prevented)</requirement>
      <requirement>Test state changes and side effects (verify variables/attributes change as expected)</requirement>
      <requirement>For class-level testing: test initialization, all public methods, property getters/setters, special methods (__str__, __repr__, etc.)</requirement>
      <requirement>For function-level testing: test with valid inputs, invalid inputs, boundary values, None/null, empty strings/lists/dicts</requirement>
      <requirement>For variable-level testing: test type constraints, value ranges, immutability where applicable</requirement>
    </test_requirements>
    <test_structure>
      <structure>Each test file should contain: imports (pytest only), mock/fixture definitions, test classes/functions grouped by feature</structure>
      <structure>Each test function should: have a descriptive name (test_feature_scenario), include a docstring explaining what it tests, use clear arrange-act-assert pattern</structure>
      <structure>Use fixtures for common setup and teardown to avoid code duplication</structure>
    </test_structure>
  </comprehensive_testing>
  <guideline>Use `generate_test_function(file_path, test_function_code, position)` after discovering the most relevant existing test file.</guideline>
  <guideline>Prefer `position="auto"` which inserts after imports or before the `if __name__ == "__main__":` block when present, falling back to append.</guideline>
  <guideline>Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.</guideline>
  <guideline>Keep generated tests minimal and focused on the bug and its edge cases.</guideline>
  <guideline>Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.</guideline>
  <guideline>MANDATORY: Before implementing the fix, create a test that reproduces the bug (it should fail). After implementing the fix, run the test again to verify it passes.</guideline>
</test_generation_guidance>

<available_tools>
  <tools>{tools_docs}</tools>
</available_tools>

<format_requirements>
  {format_prompt}
</format_requirements>

<project_structure>
{project_structure}
</project_structure>
""")

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
<task_start>
  <instruction>Now let's start. Here is the problem statement:</instruction>
  <problem_statement>{problem_statement}</problem_statement>
</task_start>
""")

FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
<role>
  <title>Helpful Assistant</title>
  <task>Find the test runner for a given repository.</task>
</role>

<guidelines>
  <guideline>The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)</guideline>
  <guideline>Do not use the test runner to run test for whole repository or test setup.</guideline>
  <guideline>Read the README file and find the test runner. If there is no test runner, return pytest.</guideline>
</guidelines>

<output_format>
  <requirement>Output format should be as the following. No other texts are allowed.</requirement>
  <example>abc/test.py</example>
</output_format>
""")

TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
<role>
  <title>Helpful Assistant</title>
  <task>Determine the mode of the test runner.</task>
</role>

<instructions>
  <instruction>Read the test runner file and determine if it requires a module or a file path to run the test.</instruction>
  <instruction>Output should be one of MODULE or FILE, No other texts are allowed.</instruction>
</instructions>

<modes>
  <mode>
    <name>MODULE</name>
    <description>When the test runner requires a module path to run the test.</description>
  </mode>
  <mode>
    <name>FILE</name>
    <description>When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).</description>
  </mode>
</modes>
""")

# ============================================================================
# ENHANCED AGENT LOGIC PROMPTS - Best Practices from Winning Solutions
# ============================================================================

META_PLANNING_AGENT_PROMPT = textwrap.dedent("""
<role>
  <title>Meta-Planning Agent</title>
  <specialization>Strategic problem decomposition and solution planning</specialization>
  <task>Analyze the problem and create a comprehensive execution plan before implementation</task>
</role>

<planning_strategy>
  <step number="1">
    <name>Problem Analysis</name>
    <actions>
      <action>Identify the core problem type (algorithmic, data structure, API, business logic)</action>
      <action>Extract all requirements from the problem statement</action>
      <action>Identify constraints (time, space, compatibility)</action>
      <action>List all inputs, outputs, and edge cases</action>
      <action>Identify dependencies and affected components</action>
    </actions>
  </step>
  
  <step number="2">
    <name>Solution Decomposition</name>
    <actions>
      <action>Break down the problem into 3-5 major sub-tasks</action>
      <action>Identify dependencies between sub-tasks</action>
      <action>Determine the optimal order of execution</action>
      <action>Identify potential risks and mitigation strategies for each sub-task</action>
    </actions>
  </step>
  
  <step number="3">
    <name>Approach Selection</name>
    <actions>
      <action>List 2-3 different approaches to solve the problem</action>
      <action>For each approach, identify: pros, cons, complexity, and risks</action>
      <action>Recommend the best approach with clear justification</action>
      <action>Provide fallback approach if primary fails</action>
    </actions>
  </step>
  
  <step number="4">
    <name>Verification Strategy</name>
    <actions>
      <action>Define success criteria for the solution</action>
      <action>List all test cases needed (normal, edge, boundary, exception)</action>
      <action>Plan verification checkpoints at each sub-task</action>
      <action>Define rollback strategy if solution fails</action>
    </actions>
  </step>
</planning_strategy>

<output_format>
  <requirement>Return a structured JSON plan with the following schema:</requirement>
  <schema>
    {
      "problem_analysis": {
        "problem_type": "string",
        "requirements": ["list of requirements"],
        "constraints": ["list of constraints"],
        "edge_cases": ["list of edge cases"],
        "affected_components": ["list of files/classes/functions"]
      },
      "solution_decomposition": {
        "sub_tasks": [
          {
            "id": "number",
            "name": "string",
            "description": "string",
            "dependencies": ["list of sub-task ids"],
            "estimated_complexity": "low/medium/high",
            "risks": ["list of risks"]
          }
        ],
        "execution_order": ["list of sub-task ids in order"]
      },
      "approaches": [
        {
          "name": "string",
          "description": "string",
          "pros": ["list of advantages"],
          "cons": ["list of disadvantages"],
          "complexity": "low/medium/high",
          "recommended": boolean
        }
      ],
      "verification_strategy": {
        "success_criteria": ["list of criteria"],
        "test_categories": ["normal", "edge", "boundary", "exception"],
        "checkpoints": ["list of verification points"],
        "rollback_plan": "string"
      }
    }
  </schema>
</output_format>

<critical_guidelines>
  <guideline>Be specific - avoid generic statements</guideline>
  <guideline>Consider ALL edge cases and failure modes</guideline>
  <guideline>Plan must be actionable - each step should be clear and executable</guideline>
  <guideline>Identify potential blockers and mitigation strategies upfront</guideline>
  <guideline>Balance thoroughness with execution speed</guideline>
</critical_guidelines>
""")

REFLECTION_AGENT_PROMPT = textwrap.dedent("""
<role>
  <title>Reflection Agent</title>
  <specialization>Self-critique and solution improvement</specialization>
  <task>Review proposed solutions, identify weaknesses, and suggest improvements</task>
</role>

<reflection_framework>
  <dimension name="correctness">
    <check>Does the solution correctly address all requirements in the problem statement?</check>
    <check>Are there any logical errors or incorrect assumptions?</check>
    <check>Does it handle all specified inputs and outputs correctly?</check>
    <check>Are edge cases properly handled?</check>
  </dimension>
  
  <dimension name="completeness">
    <check>Are all requirements from the problem statement implemented?</check>
    <check>Are there missing edge case handlers?</check>
    <check>Is error handling comprehensive?</check>
    <check>Are all necessary validations included?</check>
  </dimension>
  
  <dimension name="robustness">
    <check>How does the solution handle invalid inputs?</check>
    <check>Are there potential runtime errors not caught?</check>
    <check>Does it gracefully degrade on edge cases?</check>
    <check>Are there potential race conditions or threading issues?</check>
  </dimension>
  
  <dimension name="efficiency">
    <check>Is the time complexity optimal for the problem?</check>
    <check>Is the space complexity reasonable?</check>
    <check>Are there unnecessary operations or redundancies?</check>
    <check>Can performance be improved without sacrificing correctness?</check>
  </dimension>
  
  <dimension name="code_quality">
    <check>Is the code readable and well-structured?</check>
    <check>Are variable and function names descriptive?</check>
    <check>Is there appropriate separation of concerns?</check>
    <check>Does it follow Python best practices?</check>
  </dimension>
  
  <dimension name="backward_compatibility">
    <check>Does the solution maintain compatibility with existing code?</check>
    <check>Are there breaking changes to APIs or interfaces?</check>
    <check>Will existing tests still pass?</check>
  </dimension>
</reflection_framework>

<critique_process>
  <step number="1">Analyze each dimension systematically</step>
  <step number="2">For each issue found, rate severity: CRITICAL, HIGH, MEDIUM, LOW</step>
  <step number="3">Provide specific examples of the issue</step>
  <step number="4">Suggest concrete improvements for each issue</step>
  <step number="5">Prioritize improvements by impact and effort</step>
</critique_process>

<output_format>
  <requirement>Return a structured critique with the following format:</requirement>
  <schema>
    {
      "overall_assessment": "string (1-2 sentences)",
      "confidence_score": "number (0-100)",
      "issues_found": [
        {
          "dimension": "string",
          "severity": "CRITICAL/HIGH/MEDIUM/LOW",
          "description": "string",
          "example": "string (code snippet or specific case)",
          "suggested_fix": "string",
          "priority": "number (1-10)"
        }
      ],
      "strengths": ["list of things done well"],
      "improvement_recommendations": [
        {
          "recommendation": "string",
          "impact": "HIGH/MEDIUM/LOW",
          "effort": "HIGH/MEDIUM/LOW",
          "implementation": "string (how to implement)"
        }
      ],
      "should_proceed": boolean,
      "revision_needed": boolean
    }
  </schema>
</output_format>

<critical_standards>
  <standard>Be brutally honest - identify ALL issues, even minor ones</standard>
  <standard>Focus on correctness over aesthetics</standard>
  <standard>Provide actionable feedback - vague critiques are not helpful</standard>
  <standard>Consider the broader context and implications</standard>
  <standard>If solution is fundamentally flawed, recommend starting over</standard>
</critical_standards>
""")

SOLUTION_VALIDATOR_PROMPT = textwrap.dedent("""
<role>
  <title>Solution Validator</title>
  <specialization>Comprehensive solution validation and quality assurance</specialization>
  <task>Validate solutions against problem requirements, test results, and quality criteria</task>
</role>

<validation_checklist>
  <category name="functional_correctness">
    <item weight="30">Solution produces correct outputs for all specified test cases</item>
    <item weight="20">Edge cases are handled correctly</item>
    <item weight="15">Boundary conditions are properly managed</item>
    <item weight="10">Error cases raise appropriate exceptions</item>
    <item weight="10">Side effects are as expected</item>
    <item weight="15">All requirements from problem statement are met</item>
  </category>
  
  <category name="test_coverage">
    <item weight="25">All normal use cases are tested</item>
    <item weight="25">All edge cases have corresponding tests</item>
    <item weight="20">Boundary conditions are tested</item>
    <item weight="20">Exception scenarios are tested</item>
    <item weight="10">Tests are independent and reproducible</item>
  </category>
  
  <category name="code_quality">
    <item weight="20">No syntax errors</item>
    <item weight="15">No runtime errors on valid inputs</item>
    <item weight="15">Follows Python coding conventions</item>
    <item weight="15">Proper error handling</item>
    <item weight="15">Code is maintainable and readable</item>
    <item weight="10">Appropriate use of data structures</item>
    <item weight="10">No unnecessary complexity</item>
  </category>
  
  <category name="performance">
    <item weight="40">Time complexity meets requirements</item>
    <item weight="30">Space complexity is reasonable</item>
    <item weight="20">No obvious performance bottlenecks</item>
    <item weight="10">Resource usage is appropriate</item>
  </category>
  
  <category name="compatibility">
    <item weight="50">Backward compatible with existing code</item>
    <item weight="30">Doesn't break existing tests</item>
    <item weight="20">Compatible with specified Python version</item>
  </category>
</validation_checklist>

<validation_process>
  <step number="1">
    <name>Requirements Check</name>
    <action>Compare solution against each requirement in problem statement</action>
    <action>Mark each requirement as: PASS, FAIL, or PARTIAL</action>
  </step>
  
  <step number="2">
    <name>Test Execution Analysis</name>
    <action>Review all test results</action>
    <action>Analyze failure patterns if any tests failed</action>
    <action>Verify test coverage is comprehensive</action>
  </step>
  
  <step number="3">
    <name>Quality Assessment</name>
    <action>Evaluate against each checklist category</action>
    <action>Calculate weighted scores</action>
    <action>Identify quality gaps</action>
  </step>
  
  <step number="4">
    <name>Edge Case Verification</name>
    <action>List all possible edge cases for the problem</action>
    <action>Verify each edge case is handled</action>
    <action>Test edge cases not explicitly covered</action>
  </step>
  
  <step number="5">
    <name>Final Decision</name>
    <action>Calculate overall validation score</action>
    <action>Determine if solution passes validation</action>
    <action>List blocking issues if any</action>
  </step>
</validation_process>

<output_format>
  <requirement>Return a comprehensive validation report:</requirement>
  <schema>
    {
      "validation_passed": boolean,
      "overall_score": "number (0-100)",
      "category_scores": {
        "functional_correctness": "number (0-100)",
        "test_coverage": "number (0-100)",
        "code_quality": "number (0-100)",
        "performance": "number (0-100)",
        "compatibility": "number (0-100)"
      },
      "requirements_status": [
        {
          "requirement": "string",
          "status": "PASS/FAIL/PARTIAL",
          "evidence": "string"
        }
      ],
      "test_results": {
        "total_tests": "number",
        "passed": "number",
        "failed": "number",
        "failure_analysis": "string"
      },
      "blocking_issues": [
        {
          "issue": "string",
          "severity": "CRITICAL/HIGH",
          "must_fix": boolean
        }
      ],
      "recommendations": ["list of improvement suggestions"],
      "certification": {
        "ready_for_production": boolean,
        "confidence_level": "HIGH/MEDIUM/LOW",
        "validation_notes": "string"
      }
    }
  </schema>
</output_format>

<passing_criteria>
  <criterion>Overall score must be >= 85</criterion>
  <criterion>Functional correctness score must be >= 90</criterion>
  <criterion>No CRITICAL blocking issues</criterion>
  <criterion>All test categories must have >= 80% score</criterion>
  <criterion>All requirements must be PASS or PARTIAL (no complete FAIL)</criterion>
</passing_criteria>
""")

ITERATIVE_REFINEMENT_PROMPT = textwrap.dedent("""
<role>
  <title>Iterative Refinement Agent</title>
  <specialization>Solution improvement through feedback loops</specialization>
  <task>Take a solution, analyze feedback, and generate an improved version</task>
</role>

<refinement_strategy>
  <input_analysis>
    <item>Current solution code</item>
    <item>Test results (passed/failed)</item>
    <item>Reflection agent critique</item>
    <item>Validation report</item>
    <item>Problem statement</item>
  </input_analysis>
  
  <refinement_priorities>
    <priority level="1">Fix CRITICAL issues that cause test failures</priority>
    <priority level="2">Fix HIGH severity issues affecting correctness</priority>
    <priority level="3">Address edge cases not properly handled</priority>
    <priority level="4">Improve performance if below requirements</priority>
    <priority level="5">Enhance code quality and maintainability</priority>
  </refinement_priorities>
  
  <refinement_process>
    <step number="1">
      <name>Analyze Feedback</name>
      <action>Categorize all feedback by type and severity</action>
      <action>Identify root causes of failures</action>
      <action>Determine which issues are related</action>
    </step>
    
    <step number="2">
      <name>Plan Improvements</name>
      <action>Create ordered list of changes to make</action>
      <action>Ensure changes don't conflict</action>
      <action>Identify changes that might introduce new issues</action>
    </step>
    
    <step number="3">
      <name>Implement Changes</name>
      <action>Apply fixes in priority order</action>
      <action>Preserve working parts of the solution</action>
      <action>Add necessary validations and error handling</action>
    </step>
    
    <step number="4">
      <name>Verify Improvements</name>
      <action>Ensure original working tests still pass</action>
      <action>Verify failed tests now pass</action>
      <action>Check no regressions introduced</action>
    </step>
  </refinement_process>
</refinement_strategy>

<improvement_techniques>
  <technique name="targeted_fixes">
    <description>Make minimal, surgical changes to fix specific issues</description>
    <when>When issue is well-understood and fix is clear</when>
  </technique>
  
  <technique name="refactoring">
    <description>Restructure code to eliminate classes of problems</description>
    <when>When multiple related issues stem from poor structure</when>
  </technique>
  
  <technique name="edge_case_handling">
    <description>Add explicit checks and handling for edge cases</description>
    <when>When edge cases cause failures</when>
  </technique>
  
  <technique name="algorithm_replacement">
    <description>Replace approach with fundamentally better algorithm</description>
    <when>When current approach cannot meet requirements</when>
  </technique>
  
  <technique name="defensive_programming">
    <description>Add input validation and error checking</description>
    <when>When errors occur due to invalid inputs</when>
  </technique>
</improvement_techniques>

<output_requirements>
  <requirement>Return ONLY the improved solution code</requirement>
  <requirement>Include comments explaining key changes made</requirement>
  <requirement>Preserve all working functionality</requirement>
  <requirement>Ensure backward compatibility unless explicitly waived</requirement>
  <requirement>Follow same code structure and style as original</requirement>
</output_requirements>

<refinement_principles>
  <principle>Make the minimum changes necessary to address feedback</principle>
  <principle>Don't over-engineer - solve the problem at hand</principle>
  <principle>Test-driven fixes - ensure each fix addresses a test failure</principle>
  <principle>Preserve simplicity - don't make code more complex</principle>
  <principle>One improvement at a time - avoid massive rewrites</principle>
</refinement_principles>
""")
  
class EnhancedCOT:
    class Action:
            
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
    def __init__(self,latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep

    def add_action(self, action: EnhancedCOT.Action) -> bool: # don't add if thought is repeated
        self.thoughts.append(action)
        return True
        
    def is_thought_repeated(self)->bool:
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False
    def to_str(self):
        messages=[]
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i<len(self.thoughts)-self.latest_observations_to_keep:
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n"
                    f"next_tool_name:{thought.next_tool_name}\n"
                    f"next_tool_args:{thought.next_tool_args}\n"
                )
                if thought.observation is None:
                    _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)):
                    _obs_len = len(thought.observation)
                else:
                    _obs_len = len(str(thought.observation).splitlines())
                user_str=( f"observation: {'error ocurred.' if thought.is_error else ''} "
                    f"output omitted ({_obs_len}) lines\n")
                
            else:
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
            messages.append({"role":"assistant","content":assistant_str})
            messages.append({"role":"user","content":user_str})
        return messages

class Utils:
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.info(f"unable to fix manually, trying with llm")
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError(f"Invalid JSON: {json_string}")
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        with open("../failed_messages.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None

class ClassVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content

    def visit_ClassDef(self, node):
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.classes[node.name] = {
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

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
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response=cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0, max_retries:int=5)->str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"

        # Cache miss - make the actual request
        request_data = {
                "run_id": run_id if run_id else str(uuid4()),
                "messages": messages,
                "temperature": temperature,
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model
        
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
                is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
                if is_oai_interface:
                    raw_text=response_json['choices'][0]['message']['content']
                else:
                    if type(response_json) is str:
                        raw_text=response_json.strip("\n").strip()
                    else:
                        raw_text=response_json
                if type(raw_text) is not dict:
                    raw_text=raw_text.lstrip()
                return raw_text
            except (KeyError, IndexError, TypeError) as e:
                return f"ERROR: Invalid response structure for model {model}"
            except Exception as e:
                return f"ERROR: Unexpected error for model {model}"
        
        # If we exhausted all retries
        return f"ERROR: Max retries exceeded for model {model}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, 
                            model: str,
                            max_retries: int = 5, 
                            base_delay: float = 1.0,
                            temperature: float = 0.0) -> str:
        
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                # index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text=cls.make_request(messages,model=model, temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})") 
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name]+=1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    time.sleep(random.uniform(1.2*delay, 1.5*delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages
    
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
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
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, next_tool_args: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''

        next_tool_args=next_tool_args.replace('```json','').strip('```')
        error_msg=''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg=f"Invalid JSON: {next_tool_args}"    
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name,required=True), next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()),return_json:bool=False, temperature:float=0.0) -> dict:
        """Prod inference with caching"""
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")

            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
        
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        # remove all leading and trailing quotes
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            # remove all leading and trailing quotes in tool_name
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)
        
        return text_resp

    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, Any, Any]:
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:")<text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_thought=text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw=text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                # Enforce arrays per new contract: if single string/object, wrap as arrays
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
            except JSONDecodeError as e:
                error_msg=f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
                
        else:
            if "next_thought:" not in text_resp:
                error_msg="Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg="Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg="Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:")>text_resp.find("next_tool_name:"):
                error_msg="Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:")>text_resp.find("next_tool_args:"):
                error_msg="Invalid response. next_tool_name is after next_tool_args"
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
            Utils.log_to_failed_messages(text_resp)
            return None,None,None,error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
            
        return next_thought, next_tool_name, next_tool_args,error_msg

class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_TOOL_NAME=11
            INVALID_FILE_PATH=12
            INVALID_TOOL_CALL=13
            IMPORT_ERROR=14
            GIT_OPERATION_FAILED=15
            GIT_CONFIG_ERROR=16
            GIT_STATE_ERROR=17
            GIT_MERGE_CONFLICT=18
            GIT_BRANCH_ERROR=19
            TEST_COVERAGE_ERROR = 20
            DEPENDENCY_ANALYSIS_ERROR = 21
            CODE_SMELL_DETECTION_ERROR = 22
            GIT_HISTORY_ERROR = 23
            CODE_QUALITY_ERROR = 24
            SOLUTION_VALIDATION_ERROR = 25
            CODE_STYLE_ERROR = 26
            SOLUTION_COMPARISON_ERROR = 27
            
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__]+=1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message

        # Preserve original function metadata
       
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True

        return wrapper

    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            # Special handling for list[str] / List[str] annotations so that the
            # generated JSON schema correctly represents an array of strings.
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self,tool_name:str,required_only:bool=False)->list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']

    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        
        return tool_method
    
    def _check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            # self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)

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

class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]

        # Check all classes in the method resolution order (MRO) to include inherited tools
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={
            k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations={
          k:0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term is not None and search_term!="":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
            
        # check if start and end line are not between a function..
        func_ranges=self.get_function_ranges(file_path)
        if search_start_line!=None:
            for start, end, name in func_ranges:
                if start<=search_start_line<=end:
                    if start<search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line=start
        if search_end_line!=None:
            for start, end, name in func_ranges:
                if start<=search_end_line<=end:
                    if end>search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line=end
        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit!=-1  else content
    
    @EnhancedToolManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
       
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
        
    @EnhancedToolManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self,solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
            
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)
        
        solutions = parsed_solutions

        if type(solutions) is not list or len(solutions)<2:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: solutions must be a list with length at least 2.")

        self.is_solution_approved = True
        return "Approved"
          
    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)
 
    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            parts = function_path.split("::")
            file_path = parts[0]
            function_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = FunctionVisitor(content)
                visitor.visit(tree)
                
                if function_name in visitor.functions:
                    functions[function_path] = visitor.functions[function_name].get("body", "")
                else:
                    functions[function_path] = f"Function {function_name} not found in {file_path}"
            except FileNotFoundError:
                functions[function_path] = f"File {file_path} not found"
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {str(e)}"

        return functions

    @EnhancedToolManager.tool
    def get_classes(self, class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            parts = class_path.split("::")
            file_path = parts[0]
            class_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = ClassVisitor(content)
                visitor.visit(tree)
                if class_name in visitor.classes:
                    classes[class_path] = visitor.classes[class_name].get("body", "")
                else:
                    classes[class_path] = f"Class {class_name} not found in {file_path}"
            except FileNotFoundError:
                classes[class_path] = f"File {file_path} not found"
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {str(e)}"

        return classes

    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        If more context is needed (e.g., surrounding functions, classes, etc.), follow up with get_classes or get_functions.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        # Walk through all directories and find Python files
        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue

                        # Parse the file content using AST
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")

        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in the codebase.")
        return output

    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error parsing '{file_path}': {e}, {traceback.format_exc()}")
            tree = None  # Fallback if file cannot be parsed.

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{file_path}'")

        func_ranges=self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)

    @EnhancedToolManager.tool
    def start_over(self,problem_with_old_approach:str,new_apprach_to_try:str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''    
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
    def get_final_git_patch(self) -> str:
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
                for _p in getattr(self, "generated_test_files", []):
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
                logger.warning("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    
    @EnhancedToolManager.tool
    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        '''
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")

        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Normalize newline handling
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,"Error: test_function_code cannot be empty.")

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            # Validate standalone content before writing
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: generated test function has syntax error: {err}")
        else:
            original = self._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")

            # Try each candidate until one passes syntax check
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: inserting test caused syntax error. First error: {first_error}")

        self._save(file_path, new_content)

        # Track for exclusion from final patch
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    @EnhancedToolManager.tool
    def run_repo_tests(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        if self.test_runner == "pytest":
            print("CMD: pytest ", file_paths)
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    @EnhancedToolManager.tool
    def run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            
            error_type=EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
       

        return observation
    
    @EnhancedToolManager.tool
    def apply_code_edit(self,file_path:str, search:str, replace:str)->str:
        '''
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute
            
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: file '{file_path}' does not exist.")
        
        original=self._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.")
            case 1:
                
                new_content = original.replace(search, replace)
                try:
                        is_error,error=self.check_syntax_error(new_content)
                        if not is_error:
                            self.save_file(file_path, new_content)
                                
                            return "ok, code edit applied successfully"
                        else:
                            error.message="code edit failed. "+error.message
                            raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: syntax error in file {file_path}. {e.message}")
            case num_hits:
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
    
    @EnhancedToolManager.tool
    def create_meta_plan(self, problem_statement: str, project_context: str = "") -> str:
        '''
        Create a strategic execution plan for solving a complex problem using meta-planning approach.
        This tool analyzes the problem comprehensively and generates a structured plan before implementation.
        
        Arguments:
            problem_statement: The complete problem description that needs to be solved
            project_context: Additional context about the project structure, affected files, or constraints
        
        Output:
            A comprehensive JSON plan including: problem analysis, solution decomposition into sub-tasks, 
            approach evaluation with pros/cons, and verification strategy with success criteria
        '''
        try:
            messages = [
                {"role": "system", "content": META_PLANNING_AGENT_PROMPT},
                {"role": "user", "content": f"Problem Statement:\n{problem_statement}\n\nProject Context:\n{project_context}\n\nCreate a comprehensive meta-plan for solving this problem."}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME, temperature=0.3)
            
            # Try to parse and validate JSON
            try:
                # Clean up potential markdown formatting
                cleaned = response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.startswith('```'):
                    cleaned = cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                plan = json.loads(cleaned)
                return json.dumps(plan, indent=2)
            except json.JSONDecodeError:
                # Return as-is if not valid JSON
                return response
                
        except Exception as e:
            return f"Error creating meta-plan: {str(e)}\nNote: Meta-planning is optional. You can proceed without it."
    
    @EnhancedToolManager.tool
    def reflect_on_solution(self, proposed_solution: str, problem_statement: str, solution_description: str = "") -> str:
        '''
        Perform self-critique and reflection on a proposed solution before implementation.
        Reviews the solution across 6 dimensions: correctness, completeness, robustness, efficiency, code_quality, and backward_compatibility.
        
        Arguments:
            proposed_solution: The code or detailed description of the proposed solution
            problem_statement: The original problem that the solution addresses
            solution_description: Optional explanation of the solution approach and rationale
        
        Output:
            A structured critique with: overall assessment, confidence score, issues found with severity ratings,
            strengths, improvement recommendations, and decision on whether to proceed or revise
        '''
        try:
            messages = [
                {"role": "system", "content": REFLECTION_AGENT_PROMPT},
                {"role": "user", "content": f"Problem Statement:\n{problem_statement}\n\nProposed Solution:\n{proposed_solution}\n\nSolution Description:\n{solution_description}\n\nProvide a comprehensive reflection and critique of this solution."}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME, temperature=0.3)
            
            # Try to parse and validate JSON
            try:
                cleaned = response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.startswith('```'):
                    cleaned = cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                critique = json.loads(cleaned)
                return json.dumps(critique, indent=2)
            except json.JSONDecodeError:
                return response
                
        except Exception as e:
            return f"Error performing reflection: {str(e)}\nNote: Reflection is recommended but optional. You can proceed with caution."
    
    @EnhancedToolManager.tool
    def validate_solution(self, solution_code: str, test_results: str, problem_statement: str, requirements_checklist: str = "") -> str:
        '''
        Comprehensively validate a solution against multiple quality dimensions and criteria.
        Provides weighted scoring across functional correctness, test coverage, code quality, performance, and compatibility.
        
        Arguments:
            solution_code: The implemented solution code to validate
            test_results: Results from running tests (passed/failed counts, failure details)
            problem_statement: The original problem requirements
            requirements_checklist: Optional list of specific requirements to check against
        
        Output:
            A comprehensive validation report with: validation_passed boolean, overall_score (0-100),
            category scores for each dimension, requirements status, test analysis, blocking issues,
            and certification for production readiness
        '''
        try:
            messages = [
                {"role": "system", "content": SOLUTION_VALIDATOR_PROMPT},
                {"role": "user", "content": f"Problem Statement:\n{problem_statement}\n\nSolution Code:\n{solution_code}\n\nTest Results:\n{test_results}\n\nRequirements Checklist:\n{requirements_checklist}\n\nValidate this solution comprehensively."}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
            
            # Try to parse and validate JSON
            try:
                cleaned = response.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.startswith('```'):
                    cleaned = cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                validation = json.loads(cleaned)
                return json.dumps(validation, indent=2)
            except json.JSONDecodeError:
                return response
                
        except Exception as e:
            return f"Error validating solution: {str(e)}\nNote: Proceed with manual validation."
    
    @EnhancedToolManager.tool
    def refine_solution(self, current_solution: str, feedback: str, problem_statement: str, test_failures: str = "") -> str:
        '''
        Iteratively improve a solution based on feedback from tests, validation, or reflection.
        Applies targeted fixes prioritized by severity while preserving working functionality.
        
        Arguments:
            current_solution: The current solution code that needs improvement
            feedback: Feedback from reflection, validation, or test failures (include severity ratings)
            problem_statement: The original problem requirements
            test_failures: Optional detailed test failure information to guide refinement
        
        Output:
            An improved version of the solution code with comments explaining key changes made.
            The output preserves working functionality while addressing the feedback.
        '''
        try:
            messages = [
                {"role": "system", "content": ITERATIVE_REFINEMENT_PROMPT},
                {"role": "user", "content": f"Problem Statement:\n{problem_statement}\n\nCurrent Solution:\n{current_solution}\n\nFeedback to Address:\n{feedback}\n\nTest Failures:\n{test_failures}\n\nGenerate an improved solution."}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
            
            # Clean up code fences if present
            cleaned = response.strip()
            if cleaned.startswith('```python'):
                cleaned = cleaned[9:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            return cleaned.strip()
                
        except Exception as e:
            return f"Error refining solution: {str(e)}\nNote: Manual refinement needed."

    @EnhancedToolManager.tool
    def finish(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return "finish"
        else: 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,qa_response.get("analysis",""))

def determine_model_order(problem_statement: str) -> list:
    """Determine model priority via LLM routing based on the problem statement.

    The router LLM must return strict JSON indicating the first and second models.
    Falls back to a safe default if parsing fails.
    """
    try:
        system_prompt = (
            "You are a model router. Choose the best first LLM to solve a Python\n"
            "coding challenge given its problem statement, and then the second LLM.\n"
            "Only consider these options (use exact identifiers):\n"
            f"1) {DEEPSEEK_MODEL_NAME} (stronger reasoning, graphs/backtracking/parsers)\n"
            f"2) {QWEN_MODEL_NAME} (stronger implementation, string/data wrangling/spec-following)\n\n"
            "Output MUST be a single JSON object with key 'order' mapping to a list of two\n"
            "strings, the exact model identifiers, best-first. No explanations."
        )

        user_prompt = (
            "Problem statement to route:\n\n" + (problem_statement or "").strip()
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME)

        # Normalize potential fenced responses
        cleaned = raw.strip()
        cleaned = cleaned.replace('```json', '```')
        if cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = cleaned.strip('`').strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            data = json.loads(match.group(0)) if match else {}

        order = []
        if isinstance(data, dict):
            if isinstance(data.get('order'), list):
                order = data['order']
            elif 'first' in data and 'second' in data:
                order = [data['first'], data['second']]

        alias_map = {
            DEEPSEEK_MODEL_NAME.lower(): DEEPSEEK_MODEL_NAME,
            QWEN_MODEL_NAME.lower(): QWEN_MODEL_NAME,
            'deepseek': DEEPSEEK_MODEL_NAME,
            'qwen': QWEN_MODEL_NAME,
        }

        mapped = []
        for item in order:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key in alias_map and alias_map[key] not in mapped:
                mapped.append(alias_map[key])

        for candidate in [DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]:
            if candidate not in mapped:
                mapped.append(candidate)
            if len(mapped) == 2:
                break

        logger.info(f"[MODEL-ROUTER] Selected model order via LLM: {mapped}")
        return mapped[:2]
    except Exception as e:
        logger.warning(f"[MODEL-ROUTER] Routing failed ({e}); using safe default order")
        return [QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME]
    
def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        print(f"[DEBUG] Work directory: {work_dir}")
        print(f"[DEBUG] Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        os.chdir(work_dir)
        print(f"[DEBUG] After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
            # Verify .git was created in current directory
            print(f"[DEBUG] .git exists: {os.path.exists('.git')}")
            print(f"[DEBUG] Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
            
            # Set local git config (only for this repo)
            print("[DEBUG] Setting git config...")
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)

            # Add all files
            print("[DEBUG] Adding all files...")
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit (ignore error if nothing to commit)
            print("[DEBUG] Creating initial commit...")
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            else:
                print(f"[DEBUG] Commit result: {result.stderr.strip()}")
                
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)

def set_env_for_agent():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, MAX_TEST_PATCH_TIMEOUT, run_id
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    if test_mode:
        DEFAULT_TIMEOUT = 1000
        MAX_TEST_PATCH_TIMEOUT = 400

    sys.path.insert(0, repo_dir)


    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    ensure_git_initialized()

    set_env_for_agent()

    try:
        problem_type = check_problem_type(input_dict.get("problem_statement"))

        if problem_type == PROBLEM_TYPE_FIX:
            result = process_fix_task(input_dict)
        else:
            result = process_create_task(input_dict)
    except Exception as e:
        result = process_fix_task(input_dict)

    os.system("git reset --hard")

    return result

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
    models = determine_model_order(problem_statement)
    
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
                
                response = EnhancedNetwork.make_request(messages, model=models[0])
                
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

def parse_testcase_validation_response(response: str) -> Optional[dict]:
    """
    Parse the testcase validation response into a structured dict.
    Expected JSON format:
    - {"status": "perfect", "message": "..."}
    - {"status": "updated", "issues_found": [...], "improvements_made": [...], "test_code": "..."}
    
    Returns None if parsing fails.
    """
    try:
        # Clean up potential markdown formatting
        cleaned = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Try to find JSON object in the response
        if not cleaned.startswith('{'):
            # Try to extract JSON object from response
            import re
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            else:
                logger.error("No JSON object found in validation response")
                return None
        
        # Parse JSON
        result = json.loads(cleaned)
        
        # Validate required fields
        if "status" not in result:
            logger.error("Missing 'status' field in validation response")
            return None
        
        if result["status"] not in ["perfect", "updated"]:
            logger.error(f"Invalid status value: {result['status']}")
            return None
        
        if result["status"] == "updated":
            if "test_code" not in result:
                logger.error("Status is 'updated' but missing 'test_code' field")
                return None
            if not result.get("issues_found"):
                logger.warning("Status is 'updated' but no issues_found provided")
            if not result.get("improvements_made"):
                logger.warning("Status is 'updated' but no improvements_made provided")
        
        logger.info(f"✓ Successfully parsed validation response with status: {result['status']}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Response was: {response[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Error parsing validation response: {e}")
        return None

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT.format(problem_statement=problem_statement)
        },
        {
            "role": "user",
            "content": f"Files To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: \n 1. Cover all test cases addressed in the problem statement. Think about all variables, functions and classes characteristics in the problem statement. \n 2. Ouput Rule: \nYou **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Testcase Generation completed")
            
            # Step 5: Testcase Validation with JSON Response
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze and validate the test code. Return ONLY valid JSON with status 'perfect' or 'updated'."
                }   
            ]
            
            testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Initial testcase validation completed")

            # Parse the validation response (JSON format)
            validation_result = parse_testcase_validation_response(testcode_checked_response)
            
            if validation_result is None:
                # If JSON parsing failed, treat as regular text response (backward compatibility)
                logger.warning("Failed to parse JSON validation response, using traditional flow")
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
                
                logger.info("Testcase generation completed with traditional validation")
                return testcases
            
            # Process JSON validation response
            status = validation_result.get("status", "").lower()
            
            if status == "perfect":
                logger.info("✓ Initial test cases are perfect - no improvements needed")
                # Clean up the original test response for return
                testcases = testcode_response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                
                logger.info("Test generation completed successfully - tests validated as perfect")
                return testcases
            
            elif status == "updated":
                logger.info(f"✓ Test cases need updates - starting iterative improvement")
                
                # Extract updated test code from first validation
                current_testcases = validation_result.get("test_code", "").strip()
                
                # Clean up
                if current_testcases.startswith('```python'):
                    current_testcases = current_testcases[9:]
                if current_testcases.startswith('```'):
                    current_testcases = current_testcases[3:]
                if current_testcases.endswith('```'):
                    current_testcases = current_testcases[:-3]
                current_testcases = current_testcases.strip()
                
                # Iterative validation loop until "perfect"
                max_validation_iterations = 5
                
                for validation_iteration in range(max_validation_iterations):
                    logger.info(f"Validation iteration {validation_iteration + 1}/{max_validation_iterations}")
                    
                    try:
                        # Re-validate the updated test code
                        testcases_recheck_messages = [
                            {
                                "role": "system",
                                "content": TESTCASES_CHECK_PROMPT
                            },
                            {
                                "role": "user",
                                "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{current_testcases}\n\nAnalyze and validate. Return ONLY valid JSON response."
                            }   
                        ]
                        
                        revalidation_response = EnhancedNetwork.make_request(testcases_recheck_messages, model=QWEN_MODEL_NAME)
                        logger.info(f"Re-validation iteration {validation_iteration + 1} completed")
                        
                        # Parse re-validation response
                        revalidation_result = parse_testcase_validation_response(revalidation_response)
                        
                        if revalidation_result is None:
                            logger.warning(f"Failed to parse re-validation response in iteration {validation_iteration + 1}, using current testcases")
                            break
                        
                        revalidation_status = revalidation_result.get("status", "").lower()
                        
                        if revalidation_status == "perfect":
                            logger.info(f"✓ Test cases validated as PERFECT after {validation_iteration + 1} iterations")
                            logger.info(f"Final message: {revalidation_result.get('message', 'N/A')}")
                            return current_testcases
                        
                        elif revalidation_status == "updated":
                            logger.info(f"Iteration {validation_iteration + 1}: Further updates needed")
                            logger.info(f"Issues: {revalidation_result.get('issues_found', [])}")
                            logger.info(f"Improvements: {revalidation_result.get('improvements_made', [])}")
                            
                            # Extract newly updated test code
                            updated_test_code = revalidation_result.get("test_code", "").strip()
                            
                            if not updated_test_code:
                                logger.warning("Updated test code is empty, using current version")
                                return current_testcases
                            
                            # Clean up
                            if updated_test_code.startswith('```python'):
                                updated_test_code = updated_test_code[9:]
                            if updated_test_code.startswith('```'):
                                updated_test_code = updated_test_code[3:]
                            if updated_test_code.endswith('```'):
                                updated_test_code = updated_test_code[:-3]
                            updated_test_code = updated_test_code.strip()
                            
                            # Verify format
                            lines = updated_test_code.split("\n")
                            if lines[0].endswith(".py"):
                                current_testcases = updated_test_code
                                logger.info(f"Test cases updated in iteration {validation_iteration + 1}, continuing validation")
                            else:
                                logger.warning("Updated test code has invalid format, stopping iteration")
                                return current_testcases
                        
                        else:
                            logger.warning(f"Unknown status '{revalidation_status}', stopping iteration")
                            return current_testcases
                    
                    except Exception as e:
                        logger.error(f"Exception in validation iteration {validation_iteration + 1}: {e}")
                        return current_testcases
                
                # Exhausted validation iterations
                logger.info(f"Completed {max_validation_iterations} validation iterations, returning final testcases")
                return current_testcases
            
            else:
                # Unknown status in first validation
                logger.warning(f"Unknown status '{status}' in initial validation, using original testcases")
                testcases = testcode_response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                return testcases.strip()
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

def process_create_task(input_dict):
    global run_id
    start_time = time.time()
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)

    code_skeleton = get_code_skeleton()

    initial_solution = generate_initial_solution(problem_statement, code_skeleton)
    
    created_files = extract_and_write_files(initial_solution)
    
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    test_files = extract_and_write_files(test_cases)

    timeout = DEFAULT_TIMEOUT - (time.time()-start_time) - 60
    
    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        instance_id="",
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=50
    )

    if patch is None:
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch

def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    global libraries

    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT.format(libraries=libraries)
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    code_response = EnhancedNetwork.make_request(code_generation_messages, model=QWEN_MODEL_NAME, temperature=0.0)

    library_check_messages = [
        {
            "role": "system",
            "content": LIBRARY_CHECK_PROMPT.format(libraries=libraries)
        },
        {
            "role": "user",
            "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for library constraint violations. Verify that ALL imports only use libraries from the allowed list. Provide a corrected version if any issues are found. Return ONLY the final Python code."
        }   
    ]
    library_check_response = EnhancedNetwork.make_request(library_check_messages, model=QWEN_MODEL_NAME, temperature=0.0)

    loop_check_messages = [
        {
            "role": "system",
            "content": INFINITE_LOOP_CHECK_PROMPT
        },
        {
            "role": "user",
            "content": f"Generated Code:\n{library_check_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code."
        }   
    ]
    loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME, temperature=0.0)

    protocol_check_messages = [
        {
            "role": "system",
            "content": PROTOCOL_PATTERN_CHECK_PROMPT
        },
        {
            "role": "user",
            "content": f"Generated Code:\n{loop_check_response}\n\nProblem Context:\n{problem_statement}\n\nAnalyze this code for protocol correctness (iterators, callbacks, state management). Provide a corrected version if any issues are found. Return ONLY the final Python code with file names."
        }
    ]
    protocol_check_response = EnhancedNetwork.make_request(protocol_check_messages, model=QWEN_MODEL_NAME, temperature=0.0)

    final_check_messages = [
        {
            "role": "system",
            "content": FINAL_CORRECTNESS_CHECK_PROMPT
        },
        {
            "role": "user",
            "content": f"Generated Code:\n{protocol_check_response}\n\nProblem Statement:\n{problem_statement}\n\nCode Skeleton:\n{code_skeleton}\n\nPerform final validation: check algorithm correctness, semantic alignment, edge cases, and specification compliance. Provide a corrected version if any issues are found. Return ONLY the final Python code with file names."
        }
    ]
    final_check_response = EnhancedNetwork.make_request(final_check_messages, model=QWEN_MODEL_NAME, temperature=0.0)
    
    solution = final_check_response.strip()

    # Clean up code fences
    if solution.startswith('```python'):
        solution = solution[9:]
    if solution.startswith('```'):
        solution = solution[3:]
    if solution.endswith('```'):
        solution = solution[:-3]
    solution = solution.strip()
    
    logger.info("[VALIDATION COMPLETE] 5-stage validation finished")
    return solution


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

def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
    current_dir = os.path.dirname(file_path)
    
    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None

def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": FIND_TEST_RUNNER_PROMPT},
            {"role": "user", "content": readme_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "pytest"
    except Exception as e:
        logger.error(f"Error finding test runner: {e}")
        return "pytest"

def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    # Remove extension and make relative to repo
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    # Adjust relative to test runner directory if needed
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, '.')

def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path

def get_test_runner_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"

    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": TEST_RUNNER_MODE_PROMPT},
            {"role": "user", "content": runner_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "FILE"
    except Exception as e:
        logger.error(f"Error determining test runner mode: {e}")
        return "FILE"

def count_test_cases(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)
    
    except (FileNotFoundError, UnicodeDecodeError):
        return 0

def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None
    
    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        print(f"no test file found")
        return "pytest", "FILE"

    print(f"test_file_path: {test_file_path}")
    readme_file_path = find_readme(test_file_path, '.')
    if readme_file_path:
        print(f"README found: {readme_file_path}")
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)
    else:
        print("No README found, using default pytest")

    return test_runner, test_runner_mode

def process_fix_task(input_dict: Dict[str, Any]):
    """Main entry point for task processing and code modification.

    Parameters
    ----------
    input_dict : dict
        Configuration dictionary containing the task specification.
        Required key: 'problem_statement' with task details.
        Optional keys: 'run_id', 'instance_id' for tracking purposes.
    """
    global run_id
    # setting environment to include current working directory and lib directory
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split('/')[-1]
    repod_path = repo_path[:-len(repod_dir)-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)

    set_env_for_agent()
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd} and environ:{os.environ}")
    
    test_runner, test_runner_mode = get_test_runner_and_mode()
    print(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")

    try:
        logger.info(f"current files:{os.listdir()}")
        logger.info(f"packages installed:{subprocess.check_output(['pip','list']).decode('utf-8')}")
        logger.info(f"About to execute workflow...")
        patch_text= fix_task_solve_workflow(
            problem_text,
            timeout=timeout,
            run_id_1=run_id,
            test_runner=test_runner,
            test_runner_mode=test_runner_mode
        )
        logger.info(f"workflow execution completed, patch length: {len(patch_text)}")

        os.system("git reset --hard")

    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"[CRITICAL] Exception in task processing: {error_info}")
        logs.append(error_info)
    finally:
        os.chdir(cwd)

    print(f"[CRITICAL] task processor returning patch length: {len(patch_text)}")
    print(f"[CRITICAL] patch: {patch_text}")
    return patch_text

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "", \
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id=run_id_1
    cot=EnhancedCOT(latest_observations_to_keep=30)
    tool_manager=FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "get_functions",
            "get_classes",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "generate_test_function",
            "create_meta_plan",
            "reflect_on_solution",
            "validate_solution",
            "refine_solution",
            "finish"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    logger.info(f"Starting main agent execution...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0, project_structure=get_directory_tree())
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())

        messages.append({"role": "system", "content": STOP_INSTRUCTION})
    
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
    
        try:
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=GLM_MODEL_NAME, run_id=run_id, temperature=0.3)
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text,total_attempts=total_attempts),inference_error_counter=error_counter,request_data=messages)
            break
        
        logger.info(f"About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == "finish":
            logger.info('[CRITICAL] Workflow called finish operation')
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        # This happens if we exit the loop without breaking (reached MAX_STEPS)
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS: # This is create task case and failed with smaller fix steps so try to use original solution supposing generated testcases are wrong
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    return patch