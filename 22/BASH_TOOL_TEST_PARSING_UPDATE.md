# BashTool Test Result Parsing Update

## Overview
Updated the `BashTool` in `v4.py` to include an explicit parameter for controlling intelligent test result parsing. This gives the LLM complete control over when to use smart parsing versus receiving raw output.

## Changes Made

### 1. New Input Parameter: `parse_test_results`

**Location:** `ToolManager.BashTool.input_schema` (Lines 3232-3245)

Added a new optional boolean parameter to the bash tool:
```python
"parse_test_results": {
    "type": "boolean",
    "description": "Set to true when running test commands to get parsed output showing only failed tests. Set to false (default) for full output or non-test commands.",
}
```

**Default Value:** `false` (maintains backward compatibility)

### 2. Updated Tool Description

**Location:** Lines 3206-3230

Enhanced the bash tool description to explain the new parameter:

```markdown
**Smart Test Result Parsing:**
* When running test commands (pytest, unittest, jest, mocha, go test, cargo test, npm test, etc.),
  set the 'parse_test_results' parameter to true to get intelligent parsing:
  - Shows test summary (total, passed, failed, errors, skipped)
  - Shows ONLY the failed test details with error messages
  - Hides all passing tests to reduce noise and focus on failures
* This is especially useful for Pass-to-Pass (P2P) validation where you need to:
  1. First run: parse_test_results=false to see full output and understand test framework
  2. Subsequent runs: parse_test_results=true to see only failures
* Example: Running 'pytest tests/' with parse_test_results=true on 100 tests (98 pass, 2 fail)
  will show a summary and only the 2 failed tests, not all 98 passing tests.
* Leave parse_test_results=false (default) for non-test commands or when you need full output.
```

### 3. Modified Implementation Logic

**Location:** `ToolManager.BashTool.run_impl` (Lines 3298-3391)

**Key Changes:**

1. **Parameter Extraction:**
   ```python
   parse_test_results = tool_input.get("parse_test_results", False)
   ```

2. **Conditional Parsing:**
   - Removed automatic test command detection
   - Now only parses when explicitly requested via parameter
   - If `parse_test_results=true`:
     - Calls `Utils.parse_test_results(command, result)`
     - Formats output to show summary and only failed tests
     - Truncates error messages > 1000 chars
     - Stores full output in `aux_data` for reference
   - If `parse_test_results=false` (default):
     - Returns raw output as-is
     - No parsing or filtering

3. **Enhanced Output Format:**
   ```
   Test Command: <command>
   
   Test Summary:
     Framework: pytest
     Total Tests: 100
     ✓ Passed: 98
     ✗ Failed: 2
     ⚠ Errors: 0
     ⊘ Skipped: 0
   
   Failed Tests (showing only failures):
   ================================================================================
   
   1. test_file.py::test_function_name
   --------------------------------------------------------------------------------
   <error message>
   --------------------------------------------------------------------------------
   ```

### 4. Updated Tool Usage Guide in BugFixSolver

**Location:** Lines 2771-2794

Updated the prompt to explain the new parameter usage:

```markdown
bash:
• When: Steps 2, 4, 7, 8 (exploration, running tests)
• Commands: ls, find, grep, python, pytest
• Can set environment variables if needed
• **Smart Test Parsing Parameter:** When running test commands, use parse_test_results parameter:
  - Set parse_test_results=true to get ONLY failed tests with summary
    Example: {"command": "pytest tests/", "parse_test_results": true}
  - Set parse_test_results=false (or omit) to get full raw output
    Example: {"command": "pytest tests/", "parse_test_results": false}
  - Best practice for P2P validation:
    * First run: parse_test_results=false to understand test framework
    * If failures found: parse_test_results=true to focus only on failures
```

## Benefits

### 1. **LLM Control**
- The LLM explicitly decides when to use parsing
- No guessing or automatic detection
- More predictable behavior

### 2. **Flexibility**
- Can get raw output when needed (e.g., to understand test framework)
- Can get parsed output to focus on failures
- Works for both reproduction tests and existing test suites

### 3. **Token Efficiency**
- When `parse_test_results=true`, only failed tests are shown
- For large test suites (100+ tests), this dramatically reduces token usage
- Example: 100 tests with 2 failures = only 2 test outputs + summary shown

### 4. **Better Debugging**
- First run with `parse_test_results=false` to see everything
- Subsequent runs with `parse_test_results=true` to focus on failures
- Full output always available in `aux_data` if needed

### 5. **Backward Compatible**
- Default is `false`, so existing behavior unchanged
- Only activates when explicitly requested

## Usage Examples

### Example 1: Initial Test Run (Full Output)
```json
{
  "command": "pytest tests/test_feature.py",
  "parse_test_results": false
}
```
**Returns:** Full pytest output with all test results

### Example 2: Subsequent Run (Failures Only)
```json
{
  "command": "pytest tests/",
  "parse_test_results": true
}
```
**Returns:** Summary + only failed tests

### Example 3: Non-Test Command (Default)
```json
{
  "command": "ls -la"
}
```
**Returns:** Raw ls output (no parsing)

## Integration with `Utils.parse_test_results`

The BashTool now integrates seamlessly with the existing `Utils.parse_test_results` function (Lines 1087-1324):

1. **Framework Detection:** Automatically detects testing framework from command and output
2. **Generic Parsing:** Uses the robust `_parse_generic_output` parser that handles all frameworks
3. **Failed Test Extraction:** Extracts test names and error messages for failed tests only
4. **Summary Generation:** Provides counts for passed/failed/errors/skipped tests

## Testing & Validation

✅ **Syntax Validation:** Passed linter checks (no errors)
✅ **Backward Compatibility:** Default behavior unchanged (`parse_test_results=false`)
✅ **Schema Validation:** Input schema properly defines the new optional parameter

## Recommended Workflow for BugFixSolver

### Step 4: Run Reproduction Script
```json
{"command": "python reproduce_issue.py", "parse_test_results": false}
```
→ See full output to confirm bug is reproduced

### Step 7: F2P Validation
```json
{"command": "python reproduce_issue.py", "parse_test_results": true}
```
→ See only failures (should be 0 if fix worked)

### Step 8.3: P2P Validation - First Run
```json
{"command": "pytest tests/", "parse_test_results": false}
```
→ See full output to understand test framework and results

### Step 8.3: P2P Validation - Subsequent Runs
```json
{"command": "pytest tests/relevant/", "parse_test_results": true}
```
→ Focus only on failures for analysis

## Files Modified

- **`22/v4.py`**:
  - Lines 3206-3245: Updated BashTool description and input schema
  - Lines 3298-3391: Updated run_impl method
  - Lines 2771-2794: Updated tool usage guide in BugFixSolver prompt

## Summary

This update provides the LLM with explicit control over test result parsing, allowing it to:
1. Get raw output when exploring or understanding test frameworks
2. Get focused, parsed output when analyzing failures
3. Dramatically reduce token usage for large test suites
4. Maintain complete backward compatibility

The implementation leverages the existing robust `Utils.parse_test_results` function and integrates seamlessly with the BugFixSolver workflow.

