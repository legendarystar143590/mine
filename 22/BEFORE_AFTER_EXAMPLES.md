# Before/After Optimization Examples

## 1. Prompt Management

### Before ❌
```python
# Scattered global prompts
GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = tw.dedent("""...""")
INFINITE_LOOP_CHECK_PROMPT = tw.dedent("""...""")
TESTCASES_CHECK_PROMPT = tw.dedent("""...""")
# ... 4 more prompts

# Usage in functions
messages = [
    {"role": "system", "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT},
    {"role": "user", "content": f"Problem: {problem}"}
]
```

### After ✅
```python
# Centralized in PromptManager class
class PromptManager:
    GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent("""...""")
    INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent("""...""")
    TESTCASES_CHECK_PROMPT = textwrap.dedent("""...""")
    # ... all prompts in one place

    @classmethod
    def create_system_user_messages(cls, system_prompt: str, user_content: str) -> list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

# Clean usage in functions
messages = PromptManager.create_system_user_messages(
    PromptManager.GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT,
    f"Problem: {problem}"
)
```

**Benefits**: Single source of truth, easier to update, cleaner imports

---

## 2. Response Cleaning

### Before ❌
```python
# In generate_solution_with_multi_step_reasoning()
solution = loop_check_response.strip()
if solution.startswith('```python'):
    solution = solution[9:]
if solution.startswith('```'):
    solution = solution[3:]
if solution.endswith('```'):
    solution = solution[:-3]
solution = solution.strip()

# Same code repeated in generate_initial_solution()
solution = response.strip()
if solution.startswith('```python'):
    solution = solution[9:]
if solution.startswith('```'):
    solution = solution[3:]
if solution.endswith('```'):
    solution = solution[:-3]
solution = solution.strip()

# Same code repeated in generate_testcases_with_multi_step_reasoning()
testcases = testcode_checked_response.strip()
if testcases.startswith('```python'):
    testcases = testcases[9:]
# ... repeated 2 more times
```

### After ✅
```python
# Single utility method in Utils class
@staticmethod
def clean_code_response(response: str) -> str:
    """Clean code response by removing markdown code blocks."""
    if not response:
        return ""
    
    cleaned = response.strip()
    
    if cleaned.startswith('```python'):
        cleaned = cleaned[9:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
    
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    
    return cleaned.strip()

# Usage everywhere - single line
solution = Utils.clean_code_response(loop_check_response)
testcases = Utils.clean_code_response(testcode_checked_response)
```

**Benefits**: DRY compliance, consistent behavior, easier to fix bugs

---

## 3. Filename Validation

### Before ❌
```python
# In generate_solution_with_multi_step_reasoning()
lines = solution.split("\n")
if lines[0].endswith(".py") == False:
    retry += 1
    # retry logic
    continue

# Similar code in generate_testcases_with_multi_step_reasoning()
lines = testcases.split("\n")
if lines[0].endswith(".py") == False:
    retry += 1
    # retry logic
    continue
```

### After ✅
```python
# Single utility method
@staticmethod
def validate_python_filename(text: str) -> bool:
    """Check if the first line appears to be a Python filename."""
    if not text:
        return False
    
    lines = text.split("\n")
    if not lines:
        return False
    
    first_line = lines[0].strip()
    return (first_line.endswith('.py') and 
            ' ' not in first_line and 
            len(first_line) > 3)

# Usage - clear and consistent
if not Utils.validate_python_filename(solution):
    retry += 1
    logger.warning("Retrying because the first line is not a python file name")
    continue
```

**Benefits**: More robust validation, clearer intent, reusable

---

## 4. Duplicate Code Removal

### Before ❌
```python
# Two separate BashTool implementations
class EnhancedBashTool(LLMTool):
    """Enhanced bash tool with error analysis"""
    # 360+ lines of code
    def _analyze_import_error(self, cmd, result):
        """Detailed error analysis"""
        # ...
    
class BashTool(LLMTool):
    """Basic bash tool"""
    # 162 lines of duplicate/similar code
    def run_command_simple(self, cmd):
        """Simple command execution"""
        # Similar to EnhancedBashTool but without analysis
```

### After ✅
```python
# Only one implementation - the enhanced one
class EnhancedBashTool(LLMTool):
    """Enhanced bash tool with comprehensive error analysis"""
    # 360+ lines - only one implementation needed
    
    def _analyze_import_error(self, cmd, result):
        """Detailed error analysis"""
        # ...
    
    def run_command_simple(self, cmd):
        """Command execution with error analysis"""
        # ...
```

**Benefits**: -162 lines, no duplicate maintenance, clearer which to use

---

## 5. Message Creation Pattern

### Before ❌
```python
# In generate_solution_with_multi_step_reasoning()
code_generation_messages = [
    {
        "role": "system",
        "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
    },
    {
        "role": "user",
        "content": f"Problem: {problem}"
    }
]

# In check_problem_type()
messages = [
    {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
    {"role": "user", "content": f"{problem}\nTree: {tree}"}
]

# Repeated 10+ times across different functions
```

### After ✅
```python
# In generate_solution_with_multi_step_reasoning()
code_generation_messages = PromptManager.create_system_user_messages(
    PromptManager.GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT,
    f"Problem: {problem}"
)

# In check_problem_type()
messages = PromptManager.create_system_user_messages(
    PromptManager.PROBLEM_TYPE_CHECK_PROMPT,
    f"{problem}\nTree: {tree}"
)

# Same pattern everywhere - consistent and clean
```

**Benefits**: Consistent format, easier to modify, less typing

---

## 6. Error Handling Improvements

### Before ❌
```python
def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            # ...
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error: {e}")  # Generic error
            retry += 1
        time.sleep(2)
    
    return response  # Could be invalid if loop exits!
```

### After ✅
```python
def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            # ...
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
                logger.warning(f"Invalid problem type response: {response}, retrying...")
            else:
                logger.info(f"Problem type determined: {response}")
                return response  # Return on success
        except Exception as e:
            logger.error(f"Error checking problem type: {e}")  # Specific error
            retry += 1
        time.sleep(2)
    
    # Safe fallback with clear logging
    logger.warning(f"Failed to determine problem type after max retries, defaulting to FIX")
    return PROBLEM_TYPE_FIX
```

**Benefits**: Safer, clearer intent, better debugging, no invalid states

---

## 7. Logging Improvements

### Before ❌
```python
# Mixed print and logger
print(f"Retrying because the first line is not a python file name:\n {solution}")
print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")

# Generic error messages
logger.error("Multi-step reasoning solution generation failed")
```

### After ✅
```python
# Consistent logger usage with context
logger.warning(f"Retrying because the first line is not a python file name")
logger.error(f"Exception in generate_solution_with_multi_step_reasoning: {e}")

# Specific, actionable messages
logger.error("Multi-step reasoning solution generation failed after max retries")
logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
```

**Benefits**: Better debugging, clearer logs, consistent output format

---

## Summary Statistics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines of Code | ~4985 | ~4755 | -230 lines (-4.6%) |
| Prompt Locations | 7 global | 1 class | 86% reduction |
| Code Cleaning Logic | 4 copies | 1 method | 75% reduction |
| BashTool Classes | 2 classes | 1 class | 50% reduction |
| Message Creation Pattern | 10+ variations | 1 method | 90% reduction |
| Maintainability Score | Medium | High | Significant improvement |

## Key Principles Applied

### DRY (Don't Repeat Yourself)
✅ Eliminated repeated code patterns
✅ Created reusable utility methods
✅ Centralized common logic
✅ Single source of truth for prompts

### YAGNI (You Aren't Gonna Need It)
✅ Removed unused BashTool class
✅ Simplified complex patterns
✅ Eliminated redundant implementations
✅ Focused on what's actually used

### Clean Code
✅ Better naming conventions
✅ Consistent error handling
✅ Improved logging
✅ Clearer code organization

