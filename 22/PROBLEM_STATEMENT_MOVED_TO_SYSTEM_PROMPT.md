# Problem Statement Moved to System Prompt

## Overview
Moved the problem statement from the instance prompt to the system prompt in `BugFixSolver`. This ensures the problem statement is always available in the agent's context and reduces redundancy in the instance prompt.

## Motivation

### Why Move to System Prompt?

1. **Persistent Context**: The problem statement is fundamental to the entire task and should be part of the agent's persistent context, not something that needs to be repeated in every instance prompt.

2. **Cleaner Instance Prompt**: The instance prompt becomes simpler and more focused on the immediate task execution rather than containing the full problem description.

3. **Better Context Management**: Having the problem statement in the system prompt means it's always available for reference throughout the conversation, similar to the tools and workflow.

4. **Consistency**: Follows the same pattern as available_tools, workflow, and response_format - all key contextual information is in the system prompt.

## Changes Made

### 1. System Prompt Updated

**File: `22/v4.py`**

**Location: `BugFixSolver.FIX_TASK_SYSTEM_PROMPT` (Lines 3094-3101)**

**Added:**
```python
<available_tools>
{available_tools}
</available_tools>

<problem_statement>
{problem_statement}
</problem_statement>
```

**Purpose:**
- Added `{problem_statement}` placeholder to the system prompt
- Positioned after `<available_tools>` section
- Will be populated during initialization with the actual problem statement

### 2. Instance Prompt Updated

**File: `22/v4.py`**

**Location: `BugFixSolver.FIX_TASK_INSTANCE_PROMPT_TEMPLATE` (Lines 3103-3113)**

**Before:**
```python
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
<instruction_prompt>
<context>
    You are working in a Python repository. Your current working directory is the repository root.
    All project files are available for inspection and modification.
</context>

<problem_statement>
{problem_statement}
</problem_statement>

<task>
    Fix the bug described in the problem statement above.
    Follow the mandatory_workflow defined in your system prompt.
</task>
""")
```

**After:**
```python
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
<instruction_prompt>
<context>
    You are working in a Python repository. Your current working directory is the repository root.
    All project files are available for inspection and modification.
</context>

<task>
    Fix the bug described in the problem statement in your system prompt.
    Follow the mandatory_workflow defined in your system prompt.
</task>
""")
```

**Changes:**
- Removed `<problem_statement>` section entirely
- Updated task description to reference "problem statement in your system prompt"
- Removed `{problem_statement}` placeholder

### 3. Initialization Logic Updated

**File: `22/v4.py`**

**Location: `BugFixSolver.__init__` (Lines 3142-3162)**

**Before:**
```python
def __init__(
    self,
    problem_statement:str,
    tool_manager:ToolManager
):
    super().__init__(problem_statement, tool_manager)
    
    self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
        problem_statement=self.problem_statement
    )
    
    # Format system prompt with available tools
    formatted_system_prompt = self.FIX_TASK_SYSTEM_PROMPT.format(
        available_tools=self.tool_manager.get_tool_docs()
    )
    
    self.agent=CustomAssistantAgent(
        system_message=formatted_system_prompt,
        model_name=GLM_MODEL_NAME
    )
```

**After:**
```python
def __init__(
    self,
    problem_statement:str,
    tool_manager:ToolManager
):
    super().__init__(problem_statement, tool_manager)
    
    # Instance prompt no longer needs problem_statement
    self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE
    
    # Format system prompt with available tools AND problem statement
    formatted_system_prompt = self.FIX_TASK_SYSTEM_PROMPT.format(
        available_tools=self.tool_manager.get_tool_docs(),
        problem_statement=self.problem_statement
    )
    
    self.agent=CustomAssistantAgent(
        system_message=formatted_system_prompt,
        model_name=GLM_MODEL_NAME
    )
```

**Changes:**
1. **Instance Prompt**: No longer formatted with `problem_statement` - used directly
2. **System Prompt**: Now formatted with **both** `available_tools` and `problem_statement`
3. **Comments**: Updated to clarify the new structure

## Impact Analysis

### What Changed

#### System Prompt Structure
```
<role>...</role>
<core_principles>...</core_principles>
<critical_rules>...</critical_rules>
<workflow_enforcement>...</workflow_enforcement>
<restrictions>...</restrictions>
<test_editing_policy>...</test_editing_policy>
<validation_requirements>...</validation_requirements>
<best_practices>...</best_practices>
<completion_criteria>...</completion_criteria>
<mandatory_workflow>...</mandatory_workflow>
<response_format>...</response_format>
<available_tools>...</available_tools>
<problem_statement>...</problem_statement>  ⭐ NEW
```

#### Instance Prompt Structure
```
<instruction_prompt>
  <context>...</context>
  <task>...</task>  (updated reference)
  <critical_reminders>...</critical_reminders>
  <tool_usage_reminders>...</tool_usage_reminders>
</instruction_prompt>
```

### What Stayed the Same

1. **Agent Behavior**: No change in how the agent processes the problem statement
2. **Workflow Steps**: All steps remain identical
3. **Tool Usage**: Tools work exactly the same way
4. **Validation Logic**: F2P and P2P validation unchanged
5. **Public API**: `BugFixSolver(problem_statement, tool_manager)` interface unchanged

### Benefits

#### 1. Reduced Prompt Size
- Instance prompt is now shorter (removed ~100-1000 characters depending on problem statement length)
- Only needs to be sent once in the system prompt instead of repeated

#### 2. Better Context Separation
- System prompt: Persistent context (role, workflow, tools, problem)
- Instance prompt: Execution context (current directory, task)

#### 3. Improved Maintainability
- Problem statement management is centralized
- Easier to understand what's persistent vs. transient

#### 4. Cleaner Execution Flow
```python
# Before:
instruction_prompt = template.format(problem_statement=problem)
system_prompt = template.format(available_tools=tools)

# After:
instruction_prompt = template  # No formatting needed
system_prompt = template.format(available_tools=tools, problem_statement=problem)
```

## Example Comparison

### Before

**System Prompt:**
```
<role>You are an expert AI software engineer...</role>
...
<available_tools>
  - bash: Run commands...
  - str_replace_editor: Edit files...
  - complete: Finish task...
</available_tools>
```

**Instance Prompt:**
```
<instruction_prompt>
<context>Working in Python repository...</context>

<problem_statement>
The validate_password function crashes when given None input 
and returns incorrect value for empty string. It should return 
False for both cases.
</problem_statement>

<task>Fix the bug described in the problem statement above.</task>
</instruction_prompt>
```

### After

**System Prompt:**
```
<role>You are an expert AI software engineer...</role>
...
<available_tools>
  - bash: Run commands...
  - str_replace_editor: Edit files...
  - complete: Finish task...
</available_tools>

<problem_statement>
The validate_password function crashes when given None input 
and returns incorrect value for empty string. It should return 
False for both cases.
</problem_statement>
```

**Instance Prompt:**
```
<instruction_prompt>
<context>Working in Python repository...</context>

<task>Fix the bug described in the problem statement in your 
system prompt.</task>
</instruction_prompt>
```

**Result:**
- System prompt: +problem_statement (sent once)
- Instance prompt: -problem_statement (shorter, cleaner)
- Net benefit: More efficient, cleaner separation

## Agent Perspective

### How Agent Sees Context

**System Prompt (Persistent):**
```
I am an AI software engineer
My role: Fix bugs systematically
My workflow: 9 steps (reproduce → fix → validate)
My tools: bash, str_replace_editor, sequential_thinking, complete
My problem: [Actual problem description from user]
```

**Instance Prompt (Task-Specific):**
```
Current context: Python repository, root directory
Current task: Fix the bug (see problem in system prompt)
Reminders: Follow workflow, run tests, create new test files
```

This separation makes it clear what's **permanent context** (system) vs. **execution context** (instance).

## Technical Details

### Formatting Order

**System Prompt:**
1. Static template text (role, workflow, etc.)
2. Dynamic `{available_tools}` → Populated with `tool_manager.get_tool_docs()`
3. Dynamic `{problem_statement}` → Populated with user's problem statement

**Instance Prompt:**
1. Static template text only (context, task, reminders)
2. No dynamic formatting needed

### Code Flow

```python
# BugFixSolver.__init__
def __init__(self, problem_statement, tool_manager):
    super().__init__(problem_statement, tool_manager)
    
    # Step 1: Prepare instance prompt (no formatting)
    self.instruction_prompt = self.FIX_TASK_INSTANCE_PROMPT_TEMPLATE
    
    # Step 2: Format system prompt with tools + problem
    formatted_system_prompt = self.FIX_TASK_SYSTEM_PROMPT.format(
        available_tools=self.tool_manager.get_tool_docs(),
        problem_statement=self.problem_statement
    )
    
    # Step 3: Create agent with formatted system prompt
    self.agent = CustomAssistantAgent(
        system_message=formatted_system_prompt,
        model_name=GLM_MODEL_NAME
    )
```

### solve_problem Method

**No changes needed!**

The `solve_problem` method continues to work exactly as before:
```python
async def solve_problem(self):
    response = await self.agent.solve_task(
        self.instruction_prompt,  # Still uses instance prompt
        response_format="",
        is_json=False,
        ...
    )
```

The only difference is:
- **Before**: Problem statement in instance prompt
- **After**: Problem statement in system prompt (agent already initialized with it)

## Validation

### ✅ Syntax Check
```bash
python -m py_compile 22/v4.py
# Result: No syntax errors
```

### ✅ Linter Check
```python
read_lints(["22/v4.py"])
# Result: No linter errors found
```

### ✅ Logical Consistency
- ✓ Problem statement placeholder added to system prompt
- ✓ Problem statement section removed from instance prompt
- ✓ Formatting logic updated in `__init__`
- ✓ Task description updated to reference system prompt
- ✓ No other code depends on instance prompt having problem statement

## Migration Notes

### For Existing Code
- No changes needed to code that creates `BugFixSolver` instances
- Constructor signature unchanged: `BugFixSolver(problem_statement, tool_manager)`
- Public API fully backward compatible

### For Testing
- Tests that inspect prompts will see different structure
- Problem statement now in `agent.system_message` instead of `instruction_prompt`
- Functional behavior identical

### For Future Development
- When adding new system-level context, add to system prompt
- When adding execution-specific instructions, add to instance prompt
- Problem statement is now a template in system prompt structure

## Comparison with Other Prompts

### Consistent Pattern

**CreateProblemSolver:**
- Also has system prompt and instance prompt
- May benefit from similar refactoring in future

**BugFixSolver (now):**
- System prompt: Role, workflow, tools, problem ✓
- Instance prompt: Context, task, reminders ✓

This creates a consistent pattern across solvers.

## Summary

### Changes Made
1. ✅ Added `<problem_statement>{problem_statement}</problem_statement>` to system prompt
2. ✅ Removed `<problem_statement>` section from instance prompt
3. ✅ Updated task reference in instance prompt
4. ✅ Updated `__init__` to format system prompt with problem statement
5. ✅ Removed formatting from instance prompt assignment

### Result
- **Cleaner separation**: System (persistent) vs. Instance (execution)
- **More efficient**: Problem statement sent once, not repeated
- **Better maintainability**: Clear context ownership
- **No breaking changes**: Public API unchanged, behavior identical

### Files Modified
- **`22/v4.py`**:
  - Lines 3094-3101: System prompt (added problem_statement)
  - Lines 3103-3113: Instance prompt (removed problem_statement)
  - Lines 3142-3162: `__init__` method (updated formatting logic)

This refactoring improves code organization while maintaining full backward compatibility and identical agent behavior.

