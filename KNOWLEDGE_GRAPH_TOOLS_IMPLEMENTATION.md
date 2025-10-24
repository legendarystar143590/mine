# Knowledge Graph Tools Implementation

## Overview
Added two new tools to `FixTaskEnhancedToolManager` that leverage the `KnowledgeGraph` class for efficient code navigation and analysis.

## Changes Made

### 1. KnowledgeGraph Instance Initialization
**Location:** `FixTaskEnhancedToolManager.__init__` (line 1511)

Added initialization of a `KnowledgeGraph` instance:
```python
self.knowledge_graph = KnowledgeGraph(project_path=".", allowed_extensions=["*.py"])
```

This instance is automatically built when first accessed via `ensure_knowledge_graph_ready()`.

### 2. New Tool: `localize_code_by_symbol`
**Location:** Lines 2170-2206

**Purpose:** Quickly find the exact line range where a function or class is defined in a file.

**Usage Example:**
```json
{
  "file_path": "src/utils.py",
  "symbol": "process_data"
}
```

**Returns:**
```
Symbol 'process_data' found at lines 45-67 in src/utils.py
```

**Use Cases:**
- When you know a function/class name and need to locate it precisely
- Before reading specific code sections with `get_file_content`
- To validate that a symbol exists in a file

### 3. New Tool: `get_file_detailed_summary`
**Location:** Lines 2208-2241

**Purpose:** Generate a comprehensive structural overview of a Python file without reading the entire content.

**Usage Example:**
```json
{
  "file_path": "src/processor.py"
}
```

**Returns:**
```markdown
# File: src/processor.py

## Dependencies (Imports)
**Local Imports:**
  • utils
  • models

**External Libraries:**
  • json
  • logging
  • pathlib

## Classes Defined
  • DataProcessor (inherits: BaseProcessor)
    - Defined at line 15
  • ResultCache
    - Defined at line 89

## Functions Defined
  • validate_input(data, schema)
    - Line 125
  • process_batch(items, config)
    - Line 156

## Module Variables
  • DEFAULT_CONFIG
  • MAX_RETRIES
  • CACHE_SIZE

## Inferred Purpose
  • Data models/ORM, Utility/Helper functions
```

**Use Cases:**
- Initial exploration of unfamiliar files
- Understanding file structure before diving into details
- Identifying which file contains specific functionality
- Quick overview of dependencies and exports

## 4. Updated System Prompt
**Location:** Lines 507-511

Added guidance section "Code Navigation Tools" explaining when and how to use the new tools:

```markdown
## Code Navigation Tools:
- Use `get_file_detailed_summary` to quickly understand a file's structure, imports, 
  classes, and functions without reading the entire content. This is especially useful 
  when exploring unfamiliar code.
- Use `localize_code_by_symbol` when you know the name of a function or class and need 
  to find its exact location (line range) in a file. This helps you target specific 
  code segments efficiently.
- Use `search_in_all_files_content` to find where a function, class, or variable is 
  referenced across the entire codebase.
- Use `get_functions` or `get_classes` when you need the full implementation of 
  specific functions or classes.
```

## 5. Added to Available Tools List
**Location:** Lines 3194-3195

Both tools are now included in the workflow's available tools:
```python
available_tools=[
    # ... existing tools ...
    "localize_code_by_symbol",
    "get_file_detailed_summary",
    # ... more tools ...
]
```

## How KnowledgeGraph Works

### Initialization Flow
1. `KnowledgeGraph` instance is created in `FixTaskEnhancedToolManager.__init__`
2. On first tool call, `ensure_knowledge_graph_ready()` is invoked
3. This triggers `_build_knowledge_graph()` which:
   - Walks through the project directory
   - Parses all `.py` files using AST
   - Extracts imports, functions, classes, and variables
   - Stores metadata in `self.knowledge_graph` dictionary

### Performance Benefits
- **One-time parsing**: Files are parsed once and cached
- **Fast lookups**: Symbol localization is O(1) after initial parsing
- **Lazy loading**: Knowledge graph builds only when needed
- **Memory efficient**: Stores only metadata, not full file contents

## Error Handling

Both tools properly handle errors through the `EnhancedToolManager.Error` system:

- **FILE_NOT_FOUND**: File doesn't exist in knowledge graph
- **SEARCH_TERM_NOT_FOUND**: Symbol not found in specified file
- **UNKNOWN**: Other parsing or runtime errors

## Integration with Existing Tools

These new tools complement existing functionality:

| Existing Tool | New Tool Alternative | When to Use New Tool |
|--------------|---------------------|---------------------|
| `get_file_content` | `get_file_detailed_summary` | Need overview, not full content |
| `search_in_specified_file_v2` | `localize_code_by_symbol` | Know exact symbol name |
| Reading full file | `get_file_detailed_summary` + `localize_code_by_symbol` | Two-step: overview then precise location |

## Example Workflow

**Scenario:** Fix a bug in the `validate_input` function

1. **Explore**: Use `get_file_detailed_summary` on suspected files
   ```json
   {"file_path": "src/validation.py"}
   ```

2. **Locate**: Use `localize_code_by_symbol` to find the function
   ```json
   {"file_path": "src/validation.py", "symbol": "validate_input"}
   ```
   → Returns: "Lines 45-67"

3. **Read**: Use `get_file_content` with line range
   ```json
   {"file_path": "src/validation.py", "search_start_line": 45, "search_end_line": 67}
   ```

4. **Fix**: Use `apply_code_edit` to implement the solution

## Testing

No linter errors detected. All tools follow the existing pattern:
- Use `@EnhancedToolManager.tool` decorator
- Proper docstrings with Arguments and Output sections
- Consistent error handling
- Type hints in function signatures

## Future Enhancements

Potential improvements:
- Add caching for repeated `get_file_detailed_summary` calls
- Support for locating nested symbols (e.g., `ClassName.method_name`)
- Cross-file symbol tracking (imports and usages)
- Integration with test file discovery

