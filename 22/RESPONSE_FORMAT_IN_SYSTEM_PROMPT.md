# Response Format Moved to System Prompt

## Overview
Moved the response format instructions from user prompts (instance prompts) to system prompts for both `BugFixSolver` and `CreateProblemSolver`. This ensures the LLM always has the format instructions available without needing to append them to every user message.

## Changes Made

### 1. BugFixSolver Updates

#### A. Added Response Format to FIX_TASK_SYSTEM_PROMPT

**Location:** Lines 2569-2583

Added a new `<response_format>` section at the end of the system prompt (before `</system_prompt>`):

```xml
<response_format>
    **CRITICAL: You MUST respond EXACTLY in this format for EVERY response:**
    
    ===================THOUGHT
    <<your detailed thought process>>
    ===================TOOL_CALL
    {{"name":"<tool_name>","arguments":{{...}}}}
    
    **RULES:**
    - ALWAYS start with ===================THOUGHT
    - Then ALWAYS follow with ===================TOOL_CALL
    - Do NOT add any text before THOUGHT
    - Do NOT add any text after TOOL_CALL
    - TOOL_CALL must be valid JSON
</response_format>
```

**Note:** Used double braces `{{` and `}}` for JSON examples to escape them in Python format strings.

#### B. Removed Response Format from FIX_TASK_INSTANCE_PROMPT_TEMPLATE

**Location:** Lines 2867-2869 (removed)

**Before:**
```xml
<response_format>
{tool_call_format}
</response_format>
```

**After:** Completely removed this section.

#### C. Updated __init__ Method

**Location:** Lines 2894-2897

**Before:**
```python
self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
    problem_statement=self.problem_statement,
    available_tools=self.tool_manager.get_tool_docs(),
    tool_call_format=self.RESPONSE_FORMAT  # Removed
)
```

**After:**
```python
self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
    problem_statement=self.problem_statement,
    available_tools=self.tool_manager.get_tool_docs()
)
```

#### D. Updated solve_task Calls

**Location:** Lines 2913, 2925

Changed `response_format` parameter from `self.RESPONSE_FORMAT` to `""` (empty string):

**Before:**
```python
response_format=self.RESPONSE_FORMAT
```

**After:**
```python
response_format=""
```

### 2. CreateProblemSolver Updates

#### A. System Prompt Already Correct

The `SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL` already had the format in the system prompt via the `{format_prompt}` placeholder (line 1993):

```python
**STRICT RESPONSE FORMAT - YOU MUST FOLLOW THIS EXACTLY:**
{format_prompt}
```

This gets filled during __init__ at line 2218:
```python
system_message=CreateProblemSolver.SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL.format(
    tools_docs=tool_manager.get_tool_docs(), 
    format_prompt=self.RESPONSE_FORMAT_SOLUTION_EVAL_2
)
```

#### B. Updated solve_task Calls

**Location:** Lines 2417, 2433, 2447

Changed all `agent_initial_solution_eval.solve_task` calls to use empty `response_format`:

**Before:**
```python
response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2
```

**After:**
```python
response_format=""
```

### 3. solve_task Method Update

**Location:** Line 1801

The user already updated this to not append response_format to the task:

**Before:**
```python
full_task = (
    f"{task}\n\n"
    f"\n{response_format}\n\n"
)
```

**After:**
```python
full_task = (task)
```

## Rationale

### Why Move to System Prompt?

1. **Consistency:** Format instructions are always available to the LLM
2. **Token Efficiency:** Don't repeat format instructions in every user message
3. **Cleaner Prompts:** Instance prompts focus on the specific task, not formatting
4. **Better Context:** System prompts are meant for persistent instructions like response format

### Before (User Prompt Approach):

```
System: You are a bug fixing expert...
User: <problem>Fix this bug...</problem> <response_format>Use this format...</response_format>
Assistant: <response>
User: <tool_result>...</tool_result> <response_format>Use this format...</response_format>
Assistant: <response>
User: <tool_result>...</tool_result> <response_format>Use this format...</response_format>
```

**Problems:**
- Format repeated in every user message
- Wastes tokens
- Can be forgotten or inconsistent

### After (System Prompt Approach):

```
System: You are a bug fixing expert... <response_format>Use this format...</response_format>
User: <problem>Fix this bug...</problem>
Assistant: <response>
User: <tool_result>...</tool_result>
Assistant: <response>
User: <tool_result>...</tool_result>
```

**Benefits:**
- Format specified once in system prompt
- Cleaner user messages
- More token efficient
- LLM always aware of format requirements

## Impact on Token Usage

For a typical bug fixing session with 50 iterations:
- **Before:** Format instructions repeated 50 times (~100 tokens each) = ~5,000 tokens
- **After:** Format instructions once in system prompt = ~100 tokens
- **Savings:** ~4,900 tokens per session

## Verification

✅ **Syntax Validation:** Passed linter checks (no errors)
✅ **BugFixSolver:** Response format in system prompt, not instance prompt
✅ **CreateProblemSolver:** Response format in system prompt via {format_prompt}
✅ **solve_task calls:** All updated to use empty response_format=""
✅ **JSON Escaping:** Double braces used in format examples

## Files Modified

- **`22/v4.py`**:
  - Lines 2569-2583: Added response_format to FIX_TASK_SYSTEM_PROMPT
  - Lines 2867-2869: Removed from FIX_TASK_INSTANCE_PROMPT_TEMPLATE
  - Lines 2894-2897: Updated BugFixSolver.__init__
  - Lines 2913, 2925: Updated BugFixSolver.solve_task calls
  - Lines 2417, 2433, 2447: Updated CreateProblemSolver.solve_task calls
  - Line 1801: Already updated by user (full_task = task only)

## Testing Checklist

When testing this change, verify:
- [ ] BugFixSolver initializes without KeyError
- [ ] CreateProblemSolver initializes without KeyError
- [ ] LLM responses follow the expected format
- [ ] No "Missing THOUGHT section" errors
- [ ] No "Missing TOOL_CALL section" errors
- [ ] Response parsing works correctly
- [ ] Both solvers complete tasks successfully

## Summary

Successfully moved response format instructions from user prompts to system prompts for both solvers. This makes the architecture cleaner, more token-efficient, and ensures format instructions are always available to the LLM without repetition. The change maintains the same functionality while improving the implementation quality.

