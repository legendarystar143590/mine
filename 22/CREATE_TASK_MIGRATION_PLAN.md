# Create Task Migration Plan: miner-261.py → v4.py

## Executive Summary
Migrate the `CreateProblemSolver` logic from `miner-261.py` to `v4.py` while preserving v4's tool parsing mechanism, response format structure, and bugfix logic.

---

## Current State Analysis

### v4.py CreateProblemSolver
**Location**: Lines 1420-1964

**Tool Manager**: `ToolManager` class
- Uses `LLMTool` base class with input_schema validation
- Tools registered as class instances
- Tool docs generated via `self.tool_manager.get_tool_docs()`

**Tools Used**:
1. `BashTool` - Run bash commands
2. `CompleteTool` - Mark task as complete
3. `SequentialThinkingTool` - Multi-step reasoning
4. `StrReplaceEditorTool` - File editing

**Response Format**:
```
===================THOUGHT
<<thought process>>
===================TOOL_CALL
{"name":"<tool_name>","arguments":{...}}
```

**Key Features**:
- Structured tool validation via JSON schemas
- Clean separation of tool logic in LLMTool classes
- Comprehensive error handling in tool execution

---

### miner-261.py CreateProblemSolver
**Location**: Lines 187-757

**Tool Manager**: `FixTaskEnhancedToolManager` static methods
- Uses `@staticmethod` decorated functions
- Tool docs extracted via function introspection
- Tools passed as list to agent

**Tools Used**:
1. `run_code` - Execute code and run tests
2. `apply_code_edit` - Apply search/replace edits
3. `finish` - Complete the task
4. `get_file_content` - Read file content
5. `search_in_specified_file_v2` - Search in files
6. `run_python_file` - Execute Python files

**Response Format**:
```
======THOUGHT
<<thought process>>
======TOOL_CALL
{"name":"<tool_name>","arguments":{...}}
```

**Key Features**:
- Comprehensive testing workflow with test generation
- Solution validation with multiple attempts (5x)
- Test-driven evaluation approach
- Better prompts for CREATE tasks specifically

---

## Migration Strategy

### Phase 1: Create Compatible Tools (New Class)
Create `CreateTaskToolManager` extending `ToolManager` with these tools:

#### 1. **RunCodeTool** (replaces BashTool for CREATE tasks)
```python
class RunCodeTool(LLMTool):
    name = "run_code"
    description = "Execute Python code for testing"
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Python code to execute"},
            "file_path": {"type": "string", "description": "File path for the code"}
        },
        "required": ["content", "file_path"]
    }
```

**Source**: miner-261.py lines 2811-2897
**Implementation**: Runs Python code, captures output, handles timeouts

#### 2. **ApplyCodeEditTool** (replaces StrReplaceEditorTool for CREATE tasks)
```python
class ApplyCodeEditTool(LLMTool):
    name = "apply_code_edit"
    description = "Apply search/replace code edits"
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "search": {"type": "string"},
            "replace": {"type": "string"}
        },
        "required": ["file_path", "search", "replace"]
    }
```

**Source**: miner-261.py lines 2923-2965
**Implementation**: Search and replace with validation

#### 3. **GetFileContentTool**
```python
class GetFileContentTool(LLMTool):
    name = "get_file_content"
    description = "Read file content with optional line range"
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "search_start_line": {"type": "integer"},
            "search_end_line": {"type": "integer"},
            "search_term": {"type": "string"}
        },
        "required": ["file_path"]
    }
```

**Source**: miner-261.py lines 2434-2443

#### 4. **RunPythonFileTool**
```python
class RunPythonFileTool(LLMTool):
    name = "run_python_file"  
    description = "Execute a Python file"
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"}
        },
        "required": ["file_path"]
    }
```

**Source**: miner-261.py lines 2898-2921

#### 5. **SearchInFileTool**
```python
class SearchInFileTool(LLMTool):
    name = "search_in_specified_file_v2"
    description = "Search for term in specified file"
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "search_term": {"type": "string"}
        },
        "required": ["file_path", "search_term"]
    }
```

**Source**: miner-261.py lines 2717-2728

#### 6. Keep `CompleteTool` (renamed to `FinishTool` for consistency)
**Note**: miner-261 uses "finish" name, v4 uses "complete". Adapt to use "complete" but with finish semantics.

---

### Phase 2: Update System Prompts

#### Replace System Prompts with miner-261 versions:

1. **SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL**
   - **Source**: miner-261.py lines 189-219
   - **Changes**: Update response format from `======` to `===================`
   - **Keep**: Tool docs placeholder `{tools_docs}` and format placeholder `{format_prompt}`

2. **SYSTEM_PROMPT** (for initial solution generation)
   - **Source**: miner-261.py lines 232-246
   - **No changes needed**: This is language-agnostic

3. **TEST_CASE_GENERATOR_SYSTEM_PROMPT**
   - **Source**: miner-261.py lines 280-303
   - **No changes needed**: Already comprehensive

4. **Response Formats**
   - **RESPONSE_FORMAT_SOLUTION_EVAL_2**: Change `======` to `===================`
   - Keep v4's separator style for consistency

---

### Phase 3: Update CreateProblemSolver Class

#### Constructor Changes:
```python
def __init__(self, problem_statement: str, tool_manager: CreateTaskToolManager):
    self.problem_statement = problem_statement
    self.problem_statement = self.post_process_instruction()
    self.code_skeleton = self.get_code_skeleton()
    self.tool_manager = tool_manager  # Use passed tool_manager
    
    # Initialize agent with updated system message
    self.agent_initial_solution_eval = CustomAssistantAgent(
        system_message=CreateProblemSolver.SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL.format(
            tools_docs=tool_manager.get_tool_docs(), 
            format_prompt=self.RESPONSE_FORMAT_SOLUTION_EVAL_2
        ),
        model_name=QWEN_MODEL_NAME
    )
```

#### process_response Method:
**Replace with miner-261.py version** (lines 391-417)
- Keep tool_manager.get_tool() pattern from v4
- Keep error handling structure from v4

#### solve_problem Method:
**Replace with miner-261.py version** (lines 692-722)
- Key workflow:
  1. Generate initial solution (with 5x validation)
  2. Extract and write files
  3. Generate test cases
  4. Iterative evaluation loop with agent_initial_solution_eval
  5. Final finish check with comprehensive testing
- Keep v4's `Utils.create_final_git_patch(tool_manager.temp_files)`

---

### Phase 4: Create CreateTaskToolManager Class

```python
class CreateTaskToolManager(ToolManager):
    """Tool manager specifically for CREATE tasks."""
    
    def __init__(self):
        super().__init__()
        # Don't register default tools
        self._tools.clear()
        
    def _register_default_tools(self):
        """Register CREATE-specific tools."""
        # Register tools for CREATE tasks
        self.register_tool(ToolManager.RunCodeTool(tool_manager=self))
        self.register_tool(ToolManager.ApplyCodeEditTool(tool_manager=self))
        self.register_tool(ToolManager.GetFileContentTool(tool_manager=self))
        self.register_tool(ToolManager.RunPythonFileTool(tool_manager=self))
        self.register_tool(ToolManager.SearchInFileTool(tool_manager=self))
        self.register_tool(ToolManager.CompleteTool(tool_manager=self))  # Use as "finish"
```

---

### Phase 5: Update agent_main Function

```python
def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    # ... existing setup ...
    
    problem_type = asyncio.run(ProblemTypeClassifier.check_problem_type(problem_statement))
    
    if problem_type == ProblemTypeClassifier.PROBLEM_TYPE_FIX:
        # Use existing BugFixSolver (unchanged)
        tool_manager = ToolManager()
        fix_prb_task = BugFixSolver(problem_statement, tool_manager).solve_problem()
        # ... existing code ...
    else:
        # Use CreateProblemSolver with CreateTaskToolManager
        tool_manager = CreateTaskToolManager()  # Different tool manager!
        create_problem_task = CreateProblemSolver(problem_statement, tool_manager).solve_problem()
        # ... existing code ...
```

---

## Implementation Checklist

### Step 1: Tool Creation ✓
- [ ] Create `RunCodeTool` class
- [ ] Create `ApplyCodeEditTool` class  
- [ ] Create `GetFileContentTool` class
- [ ] Create `RunPythonFileTool` class
- [ ] Create `SearchInFileTool` class
- [ ] Adapt `CompleteTool` for CREATE tasks

### Step 2: Tool Manager ✓
- [ ] Create `CreateTaskToolManager` class
- [ ] Override `_register_default_tools()`
- [ ] Test tool registration

### Step 3: Update Prompts ✓
- [ ] Update `SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL`
- [ ] Update `RESPONSE_FORMAT_SOLUTION_EVAL_2`
- [ ] Update `TEST_CASE_GENERATOR_SYSTEM_PROMPT`
- [ ] Verify response format consistency (===================)

### Step 4: Update CreateProblemSolver ✓
- [ ] Update `__init__` to accept tool_manager
- [ ] Update `process_response` method
- [ ] Update `solve_problem` method (5x solution generation)
- [ ] Keep helper methods: `get_code_skeleton`, `post_process_instruction`, etc.

### Step 5: Integration ✓
- [ ] Update `agent_main` to use `CreateTaskToolManager` for CREATE tasks
- [ ] Keep `ToolManager` for FIX tasks (BugFixSolver)
- [ ] Test both paths work independently

### Step 6: Validation ✓
- [ ] Verify BugFixSolver unchanged
- [ ] Test CREATE task with new logic
- [ ] Ensure response formats consistent
- [ ] Check git patch generation

---

## Key Preservation Points

### MUST KEEP from v4.py:
1. ✅ `ToolManager.LLMTool` base class structure
2. ✅ Input schema validation mechanism
3. ✅ `Types.ToolImplOutput` return type
4. ✅ Tool registration pattern
5. ✅ `BugFixSolver` class (100% unchanged)
6. ✅ Response format separator style (`===================`)
7. ✅ `Utils.create_final_git_patch()` for patch generation
8. ✅ All enhanced tools for BugFix (TestValidationTool, etc.)

### MUST IMPORT from miner-261.py:
1. ✅ System prompts for CREATE tasks
2. ✅ Workflow logic in `solve_problem` (5x generation, testing)
3. ✅ Tool implementations (run_code, apply_code_edit logic)
4. ✅ Test generation and validation approach
5. ✅ `process_response` logic

---

## Testing Strategy

### Test Case 1: CREATE Task
- Problem: "Create a function to calculate fibonacci"
- Expected: Uses CreateTaskToolManager, generates solution, runs tests

### Test Case 2: FIX Task  
- Problem: "Fix bug in existing code"
- Expected: Uses ToolManager, uses BugFixSolver (unchanged behavior)

### Test Case 3: Tool Validation
- Verify all tools have proper input_schema
- Verify tool_docs generation works
- Verify error handling

---

## Risk Mitigation

### Risk 1: Tool Compatibility
**Mitigation**: Create adapters if miner-261 tools don't fit LLMTool pattern

### Risk 2: Response Format Mismatch
**Mitigation**: Normalize separators to v4 style (===================)

### Risk 3: Breaking BugFixSolver
**Mitigation**: Zero changes to BugFixSolver, only CreateProblemSolver affected

---

## Success Criteria

1. ✅ CREATE tasks use miner-261 logic
2. ✅ FIX tasks use existing v4 logic (unchanged)
3. ✅ All tools follow LLMTool pattern
4. ✅ Response formats consistent across both task types
5. ✅ No regression in BugFixSolver functionality
6. ✅ Git patch generation works for both task types

---

## Timeline Estimate

- **Phase 1 (Tools)**: 30-45 minutes
- **Phase 2 (Prompts)**: 15 minutes
- **Phase 3 (CreateProblemSolver)**: 20-30 minutes
- **Phase 4 (ToolManager)**: 15 minutes
- **Phase 5 (Integration)**: 10 minutes
- **Phase 6 (Testing)**: 20 minutes

**Total**: ~2 hours

---

## Next Steps

1. Start with Phase 1: Create all 6 tool classes
2. Test each tool individually
3. Create CreateTaskToolManager
4. Update CreateProblemSolver class
5. Integrate into agent_main
6. Comprehensive testing

