# Automatic Loop Detection Implementation Summary

## Overview
Implemented automatic infinite loop detection in `BugFixSolver` that monitors tool calls and triggers `start_over` when the agent gets stuck calling the same tool repeatedly with identical arguments and results.

---

## Problem Statement

AI agents can get stuck in infinite loops where they:
- Call the same tool repeatedly (e.g., viewing the same file 10 times)
- Run the same command over and over with identical output
- Attempt the same failing edit multiple times
- Make no progress but continue iterating

This wastes time and resources, and never leads to a solution.

---

## Solution: Automatic Loop Detection

### Implementation Details

**Location**: `BugFixSolver` class (Lines 2463-2553, 2587-2614)

### Components

#### 1. Tool Call Tracking (Lines 2463-2465, 2479-2514)

**Initialization in `__init__`**:
```python
# Track tool calls to detect infinite loops
self.tool_call_history: List[Dict[str, Any]] = []
self.max_identical_calls = 5
```

**Tracking Method `_track_tool_call`**:
- Creates a hashable signature of each tool call:
  - Tool name
  - Arguments (JSON serialized, sorted keys)
  - Result (first 500 chars)
- Stores in `tool_call_history`
- Analyzes last 20 calls for patterns
- Counts identical calls
- **Returns `True` if 5+ identical calls detected**

**Key Features**:
- Only tracks last 20 calls (performance optimization)
- Truncates results to 500 chars (memory optimization)
- Compares JSON-serialized arguments (handles dict ordering)

#### 2. Automatic Start Over (Lines 2516-2553)

**Method `_auto_start_over`**:
- Constructs detailed problem description
- Outlines new approach requirements
- Calls `start_over` tool automatically
- Clears tool call history
- Returns reset confirmation message

**Message Content**:
- Problem: "Automatic start_over triggered due to infinite loop"
- Analysis: Explains what was detected
- New approach: Instructions for fresh start

#### 3. Integration into Workflow (Lines 2589-2613)

**In `process_response` method**:
- After every tool execution
- Skips tracking for:
  - `sequential_thinking` (used for analysis, not actions)
  - `complete` (termination tool)
  - `start_over` (prevents recursive loops)
- Checks for loop detection
- If loop detected:
  - Calls `_auto_start_over()`
  - Overwrites tool response with reset message
  - Changes `tool_name` to `"start_over"`
  - Agent receives clear instructions

---

## Detection Algorithm

### Pattern Matching

```python
# Create call signature
call_signature = {
    "tool": "str_replace_editor",
    "args": '{"command":"view","path":"src/module.py"}',
    "result": "File contents..."[:500]
}

# Count identical calls in last 20
identical_count = sum(
    1 for call in recent_history 
    if call["tool"] == tool_name 
    and call["args"] == call_signature["args"]
    and call["result"] == call_signature["result"]
)

# Trigger if >= 5
if identical_count >= 5:
    return True  # Auto start over
```

### Why This Works

1. **Tool Name Match**: Same tool being used
2. **Arguments Match**: Exact same parameters (e.g., viewing same file)
3. **Result Match**: Getting identical output (no progress)
4. **Threshold of 5**: Allows for legitimate retries but catches loops
5. **Recent History (20 calls)**: Focuses on current behavior, not entire session

---

## Example Scenarios

### Scenario 1: Viewing Same File Repeatedly

```
Call 1: view src/bug.py → "def process():\n    return None"
Call 2: view src/bug.py → "def process():\n    return None"
Call 3: bash ls → "bug.py\ntest.py"
Call 4: view src/bug.py → "def process():\n    return None"
Call 5: view src/bug.py → "def process():\n    return None"
Call 6: view src/bug.py → "def process():\n    return None"

🚨 LOOP DETECTED: view src/bug.py called 5 times with identical result
```

**Action**: Automatic start_over triggered

### Scenario 2: Running Same Failing Test

```
Call 1: bash pytest test.py → "FAILED: assertion error"
Call 2: bash pytest test.py → "FAILED: assertion error"
Call 3: bash pytest test.py → "FAILED: assertion error"
Call 4: bash pytest test.py → "FAILED: assertion error"
Call 5: bash pytest test.py → "FAILED: assertion error"

🚨 LOOP DETECTED: bash pytest test.py called 5 times with identical failure
```

**Action**: Automatic start_over triggered

### Scenario 3: Same Edit Fails Repeatedly

```
Call 1: str_replace_editor (edit file) → "Error: old_str not found"
Call 2: str_replace_editor (edit file) → "Error: old_str not found"
Call 3: view file.py → "contents"
Call 4: str_replace_editor (edit file) → "Error: old_str not found"
Call 5: str_replace_editor (edit file) → "Error: old_str not found"
Call 6: str_replace_editor (edit file) → "Error: old_str not found"

🚨 LOOP DETECTED: str_replace_editor called 5 times with same error
```

**Action**: Automatic start_over triggered

---

## Agent Response

### What the Agent Sees

When loop is detected, the agent receives:

```
✅ Codebase successfully reverted to initial state.

All changes have been discarded.
You can now start fresh with your new approach.

**Next Steps:**
1. Return to Step 3 to create a new reproduction script (if needed)
2. Proceed to Step 4 with your new approach
3. Implement the solution following the new strategy

⚠️ AUTOMATIC RESET PERFORMED ⚠️

You were stuck in an infinite loop, calling the same tool repeatedly.
The codebase has been reset to initial state.

Please proceed with a different approach:
1. Return to Step 3 to verify your reproduction
2. Re-analyze the root cause with fresh perspective
3. Try a fundamentally different solution strategy
4. Avoid repeating the same actions that caused the loop
```

### System Logs

```
============================================================
⚠️  INFINITE LOOP DETECTED
============================================================
Tool 'str_replace_editor' called 5 times with identical arguments and results
Arguments: {'command': 'view', 'path': 'src/bug.py'}
Result (truncated): Here's the result of running `cat -n` on src/bug.py:
     1  def process():
     2      return None
============================================================

🔄 AUTO START OVER - Resetting due to detected infinite loop

============================================================
🔄 START OVER - REVERTING ALL CHANGES
============================================================

Problem with old approach:
Automatic start_over triggered due to infinite loop detection.

The same tool was called 5+ times with identical arguments and results,
indicating the agent is stuck in a repetitive loop.

New approach to try:
Will restart from Step 3 with a fresh perspective.
============================================================
```

---

## System Prompt Integration

### Guideline #10 (Lines 2392-2407)

Added comprehensive documentation in the system prompt:

```markdown
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
```

This explicitly informs the agent of the automatic behavior and what to do after reset.

---

## Performance Considerations

### Memory Optimization

1. **Recent History Only**: Tracks last 20 calls, not entire session
   - Prevents unbounded memory growth
   - Focuses on current behavior

2. **Result Truncation**: Stores only first 500 chars of each result
   - Prevents large responses from consuming memory
   - Still sufficient for detecting identical results

3. **Signature Comparison**: Uses JSON serialization with sorted keys
   - Handles dict argument ordering
   - Creates consistent comparison strings

### Computational Efficiency

- **O(n) per tool call** where n = min(20, total_calls)
- Simple string comparison
- No complex pattern matching
- Minimal overhead on each tool execution

---

## Edge Cases Handled

### 1. Legitimate Repeated Calls

**Scenario**: Agent legitimately needs to view a file multiple times during different phases

**Handled**: If the file has changed between views, the result will be different, so no loop detected.

### 2. Similar but Different Arguments

**Scenario**: Agent views file.py line 1-10, then line 11-20, etc.

**Handled**: Different arguments (view_range) mean different signatures, no loop detected.

### 3. Tool Call Failures

**Scenario**: Tool fails and agent retries with different approach

**Handled**: Different results or arguments break the loop detection.

### 4. Recursive Start Over

**Scenario**: What if agent gets stuck in loop after auto-reset?

**Handled**: Tool call history is cleared after reset, allowing fresh detection. If loop happens again, it will trigger another reset (agent has deeper issues at that point).

### 5. Excluded Tools

**Scenario**: Agent uses sequential_thinking 10 times

**Handled**: sequential_thinking is explicitly excluded from tracking as it's for analysis, not action.

---

## Benefits

### 1. Prevents Wasted Resources
- Stops infinite loops automatically
- No human intervention needed
- Saves compute time and costs

### 2. Improves Success Rate
- Forces agent to try different approaches
- Prevents getting stuck on wrong strategies
- Encourages exploration

### 3. Clear Feedback
- Agent understands what happened
- Knows why reset occurred
- Receives guidance on next steps

### 4. Complements Manual Start Over
- Automatic for obvious loops
- Manual for strategic decisions
- Two-tier safety system

---

## Configuration

### Adjustable Parameters

```python
# In BugFixSolver.__init__
self.max_identical_calls = 5  # Threshold for detection
```

**Current Setting**: 5 identical calls

**Rationale**:
- 1-2 calls: Too sensitive, legitimate retries
- 3-4 calls: May catch some valid exploration
- 5 calls: Clear pattern of loop
- 6+ calls: Too late, already wasted time

### Window Size

```python
# In _track_tool_call
recent_history = self.tool_call_history[-20:]
```

**Current Setting**: Last 20 calls

**Rationale**:
- Too small (5-10): Misses patterns spread across multiple tool types
- Just right (20): Captures recent behavior without too much history
- Too large (50+): Dilutes recent pattern detection with old behavior

---

## Testing Recommendations

### Manual Test Scenarios

1. **File View Loop**:
   - Call `view` on same file 5 times
   - Verify auto start_over triggers

2. **Command Loop**:
   - Run `bash ls` 5 times
   - Verify detection

3. **Edit Failure Loop**:
   - Attempt same failing edit 5 times
   - Verify reset

4. **Mixed Tools**:
   - Alternate between 2 tools but repeat each 5 times
   - Verify each is tracked independently

5. **Post-Reset Behavior**:
   - Trigger auto reset
   - Verify history is cleared
   - Verify agent can proceed

---

## Future Enhancements

### Potential Improvements

1. **Adaptive Threshold**:
   - Lower threshold (3) if agent has made many iterations
   - Higher threshold (7) early in problem-solving

2. **Pattern Recognition**:
   - Detect alternating loops (A→B→A→B→A→B)
   - Catch multi-step cycles

3. **Whitelist Patterns**:
   - Some repeated calls are expected (e.g., syntax validation after each edit)
   - Could whitelist certain tool+argument combinations

4. **Analytics**:
   - Track how often auto-reset triggers
   - Identify common loop patterns
   - Improve agent prompts based on data

5. **Graduated Response**:
   - First loop: Warning message
   - Second loop: Suggestion to try different approach
   - Third loop: Automatic start_over

---

## Summary

The automatic loop detection system provides a crucial safety mechanism for the BugFixSolver:

✅ **Automatically detects** when agent is stuck in repetitive behavior  
✅ **Triggers start_over** without human intervention  
✅ **Clears state** and provides fresh start  
✅ **Guides agent** to try different approaches  
✅ **Optimized** for performance and memory  
✅ **Configurable** thresholds and windows  
✅ **Complements** manual start_over tool  

This prevents infinite loops that would otherwise waste resources and never converge to a solution, improving overall success rate and efficiency of the bug-fixing agent.

