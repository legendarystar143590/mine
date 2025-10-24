# Quick Reference - Optimized v2.py

## 🎯 Quick Access Guide

### Prompts (PromptManager)
All prompts are now in `PromptManager` class:

```python
# Solution Generation
PromptManager.GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
PromptManager.GENERATE_INITIAL_SOLUTION_PROMPT

# Code Validation  
PromptManager.INFINITE_LOOP_CHECK_PROMPT

# Test Generation
PromptManager.GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
PromptManager.GENERATE_INITIAL_TESTCASES_PROMPT
PromptManager.TESTCASES_CHECK_PROMPT

# Problem Analysis
PromptManager.PROBLEM_TYPE_CHECK_PROMPT

# System Prompts
PromptManager.SYSTEM_PROMPT
PromptManager.INSTRUCTION_PROMPT
PromptManager.TOOL_CALL_FORMAT_PROMPT
```

### Helper Methods (PromptManager)

```python
# Create standard message format
messages = PromptManager.create_system_user_messages(
    system_prompt="System instructions here",
    user_content="User query here"
)
# Returns: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
```

### Utility Methods (Utils)

```python
# Clean code responses (removes ```python, ```, etc.)
clean_code = Utils.clean_code_response(raw_response)

# Validate Python filename
is_valid = Utils.validate_python_filename(text)

# Format logs with borders
Utils.format_log(text, "LABEL")

# Create git patch
patch = Utils.create_final_git_patch(temp_files=[])

# Check path safety
is_safe = Utils.is_path_in_directory(directory, path)

# Truncate long content
truncated = Utils.maybe_truncate(content, truncate_after=200000)
```

---

## 🔧 Common Patterns

### Pattern 1: Generate Solution
```python
messages = PromptManager.create_system_user_messages(
    PromptManager.GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT,
    f"Problem Statement:\n{problem}\n\nCode:\n{code}"
)
response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
clean_solution = Utils.clean_code_response(response)

if not Utils.validate_python_filename(clean_solution):
    # Handle retry logic
    pass
```

### Pattern 2: Generate Tests
```python
messages = PromptManager.create_system_user_messages(
    PromptManager.GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT,
    f"Problem:\n{problem}\n\nFiles:\n{files}"
)
response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
clean_tests = Utils.clean_code_response(response)
```

### Pattern 3: Check Problem Type
```python
messages = PromptManager.create_system_user_messages(
    PromptManager.PROBLEM_TYPE_CHECK_PROMPT,
    f"{problem_statement}\n# Tree:\n{tree}"
)
response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
# Returns: "FIX" or "CREATE"
```

---

## 🏗️ Class Structure

### PromptManager
- **Purpose**: Central prompt management
- **Location**: Lines 1529-2048
- **Methods**: 
  - `get_system_prompt_with_tools(tool_manager)`
  - `create_system_user_messages(system_prompt, user_content)`

### Utils
- **Purpose**: Reusable utility functions
- **Location**: Lines 159-332
- **Key Methods**:
  - `clean_code_response(response)` - Clean LLM output
  - `validate_python_filename(text)` - Check filename format
  - `format_log(text, label)` - Pretty print logs
  - `create_final_git_patch(temp_files)` - Generate patches

### EnhancedNetwork
- **Purpose**: Network requests with retry logic
- **Location**: Lines 875-1281
- **Key Methods**:
  - `inference()` - Main inference method
  - `make_request()` - Direct API call
  - `parse_response()` - Parse tool calls

### ToolManager
- **Purpose**: Manage all agent tools
- **Location**: Lines 2051-4257
- **Registered Tools**:
  - `EnhancedBashTool` - Command execution with error analysis
  - `CompleteTool` - Task completion signal
  - `SequentialThinkingTool` - Step-by-step reasoning
  - `StrReplaceEditorTool` - File operations
  - `TestValidationTool` - Test validation
  - `DependencyAnalysisTool` - Dependency checking
  - `TestGenerationTool` - Test generation

---

## 📊 Model Configuration

```python
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
AGENT_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
```

---

## 🎨 Logging

```python
# Available log methods
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Color tags in messages
logger.info("<yellow>Warning text</yellow>")
logger.info("<red>Error text</red>")
logger.info("<green>Success text</green>")
logger.info("<blue>Info text</blue>")
```

---

## 🔍 Finding Things

### Where to Find Prompts
- **File**: `v2.py`
- **Class**: `PromptManager`
- **Lines**: 1529-1708
- **All prompts**: `PromptManager.{PROMPT_NAME}`

### Where to Find Utilities
- **File**: `v2.py`
- **Class**: `Utils`
- **Lines**: 159-332
- **All utilities**: `Utils.{method_name}()`

### Where to Find Tools
- **File**: `v2.py`
- **Class**: `ToolManager`
- **Lines**: 2051-4257
- **Nested classes**: `ToolManager.{ToolName}`

### Where to Find Main Logic
- **Agent class**: Lines 4095-4253
- **Main function**: `agent_main()` at line 4767
- **Solution generation**: Lines 4428-4504
- **Test generation**: Lines 4569-4645

---

## 💡 Tips

### When Adding New Prompts
1. Add to `PromptManager` class
2. Use `textwrap.dedent()` for formatting
3. Follow existing naming convention: `{PURPOSE}_{PROMPT/CHECK/etc}`

### When Adding New Utilities
1. Add to `Utils` class
2. Make it `@staticmethod`
3. Add docstring with Args/Returns
4. Use consistent error handling

### When Updating Code
1. Check if utility method exists first
2. Use `PromptManager` for all LLM prompts
3. Use `logger` instead of `print()`
4. Follow DRY principle - don't repeat code

---

## 📝 Common Tasks

### Task: Add New Prompt
```python
# In PromptManager class
NEW_PROMPT = textwrap.dedent("""
    Your prompt text here
""")
```

### Task: Call LLM with Prompt
```python
messages = PromptManager.create_system_user_messages(
    PromptManager.YOUR_PROMPT,
    user_content
)
response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
clean = Utils.clean_code_response(response)
```

### Task: Validate Code Response
```python
clean_code = Utils.clean_code_response(llm_response)
if Utils.validate_python_filename(clean_code):
    # Process valid code
else:
    # Retry or handle error
```

---

## 🚀 Quick Start Template

```python
def your_new_function(problem_statement: str) -> str:
    """Your function description."""
    retry = 0
    
    # Create messages using PromptManager
    messages = PromptManager.create_system_user_messages(
        PromptManager.YOUR_CHOSEN_PROMPT,
        f"Problem: {problem_statement}"
    )
    
    while retry < 10:
        try:
            # Make request
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            logger.info("Request completed")
            
            # Clean response
            clean_response = Utils.clean_code_response(response)
            
            # Validate response
            if not Utils.validate_python_filename(clean_response):
                retry += 1
                logger.warning("Invalid response, retrying...")
                continue
            
            logger.info("Success!")
            return clean_response
            
        except Exception as e:
            retry += 1
            logger.error(f"Error: {e}")
            time.sleep(2)
    
    logger.error("Failed after max retries")
    return ""
```

---

## ✅ Checklist for New Code

- [ ] Use `PromptManager` for all prompts
- [ ] Use `Utils.clean_code_response()` for cleaning
- [ ] Use `Utils.validate_python_filename()` for validation
- [ ] Use `logger.*()` instead of `print()`
- [ ] Use `PromptManager.create_system_user_messages()` for message creation
- [ ] Follow existing error handling patterns
- [ ] Add appropriate logging at each step
- [ ] Include retry logic for network calls

---

**Last Updated**: Optimization completed
**Version**: v2.py (optimized)
**Status**: ✅ Production Ready

