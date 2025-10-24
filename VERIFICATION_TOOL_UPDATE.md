# read_file Tool Update - Verification Rules Support

## Summary

Updated the `read_file` tool in `a.py` to support an optional verification mode that returns both file contents and a comprehensive 7-step verification protocol for code review.

---

## Changes Made

### 1. New Constant: `VERIFICATION_PROTOCOL` (Lines 710-837)

Created a new constant containing the complete 7-step verification protocol extracted from the system prompt. This protocol includes:

- **Step 1:** Follow the Data Flow (parameter usage analysis)
- **Step 2:** Check the Source of Your Value (trace origins)
- **Step 3:** Search How Others Use This Attribute (learn from codebase)
- **Step 4:** Verify Type Compatibility (type checking)
- **Step 5:** Write Defensive Code (safety patterns)
- **Step 6:** Verify the Complete Flow (end-to-end verification)
- **Step 7:** Read All Updated Files (comprehensive review)
- **Common Pitfalls:** Key mistakes to avoid

**Size:** 4,774 characters  
**Location:** Between `FIX_TASK_INSTANCE_PROMPT_TEMPLATE` and `FIND_TEST_RUNNER_PROMPT`

### 2. Updated `read_file` Tool (Lines 3758-3791)

Added a new optional parameter `include_verification_rules` to the `read_file` method:

```python
def read_file(self, target_file: str, should_read_entire_file: bool, 
              start_line_one_indexed: int, end_line_one_indexed_inclusive: int,
              include_verification_rules: bool = False) -> str:
```

**New Parameter:**
- `include_verification_rules` (bool, default=False): When True, appends the verification protocol to the file contents

**Behavior:**
- When `include_verification_rules=False`: Returns file contents only (backward compatible)
- When `include_verification_rules=True`: Returns file contents + separator + verification protocol + separator

**Output Format:**
```
<file contents>

================================================================================
# CODE VERIFICATION PROTOCOL
...
(all 7 steps and common pitfalls)
...
================================================================================
```

---

## Use Cases

### Use Case 1: Normal File Reading
```python
content = read_file(
    target_file="example.py",
    should_read_entire_file=True,
    start_line_one_indexed=1,
    end_line_one_indexed_inclusive=100,
    include_verification_rules=False  # or omit (default)
)
# Returns: Just the file content
```

### Use Case 2: File Reading with Verification Guide
```python
content_with_rules = read_file(
    target_file="example.py",
    should_read_entire_file=True,
    start_line_one_indexed=1,
    end_line_one_indexed_inclusive=100,
    include_verification_rules=True  # Enable verification mode
)
# Returns: File content + verification protocol
```

**When to use verification mode:**
- After making code changes and before validation
- When reviewing code for potential bugs
- When the agent needs guidance on defensive coding
- During the verification phase of bug fixing

---

## Benefits

### 1. Self-Contained Verification
The agent receives both the code to review AND the guidelines for how to review it in a single tool call, reducing context switching.

### 2. Consistent Verification Standards
Every verification follows the same 7-step process, ensuring thorough and systematic code review.

### 3. Backward Compatible
Existing code that uses `read_file` without the new parameter continues to work unchanged.

### 4. Reduces Prompt Size
The verification protocol is extracted as a constant, making it reusable and reducing duplication in the system prompt.

### 5. On-Demand Guidance
The agent only receives the detailed verification rules when explicitly requested, avoiding information overload during normal file reading.

---

## Example Workflow

1. **Agent makes code changes** (e.g., fixes a bug)
   
2. **Agent reads file with verification:**
   ```python
   read_file(
       target_file="fixed_file.py",
       should_read_entire_file=True,
       start_line_one_indexed=1,
       end_line_one_indexed_inclusive=200,
       include_verification_rules=True
   )
   ```

3. **Agent receives:**
   - The complete updated code
   - The 7-step verification protocol
   - Common pitfalls to avoid

4. **Agent follows the protocol:**
   - Step 1: Checks function signatures
   - Step 2: Traces value origins
   - Step 3: Searches for patterns
   - Step 4: Verifies types
   - Step 5: Adds defensive code
   - Step 6: Traces complete flow
   - Step 7: Reviews all changes

5. **Agent validates the solution** with confidence

---

## Technical Details

### Constants

```python
# Line 710-837
VERIFICATION_PROTOCOL = textwrap.dedent("""
# CODE VERIFICATION PROTOCOL
...
""")
```

### Tool Schema Update

The tool's schema is automatically updated to include the new parameter:

```json
{
  "name": "read_file",
  "description": "Read the contents of a file with optional line range and verification rules.",
  "input_schema": {
    "properties": {
      "target_file": {...},
      "should_read_entire_file": {...},
      "start_line_one_indexed": {...},
      "end_line_one_indexed_inclusive": {...},
      "include_verification_rules": {
        "type": "boolean",
        "description": "If True, also return the 7-step verification protocol for code review"
      }
    }
  }
}
```

### Implementation

```python
# Lines 3774-3791
if should_read_entire_file:
    file_content = self._get_file_content(target_file, limit=-1)
else:
    file_content = self._get_file_content(
        target_file, 
        search_start_line=start_line_one_indexed, 
        search_end_line=end_line_one_indexed_inclusive, 
        limit=-1
    )

# Append verification rules if requested
if include_verification_rules:
    return f"{file_content}\n\n{'='*80}\n{VERIFICATION_PROTOCOL}\n{'='*80}\n"
else:
    return file_content
```

---

## Testing Results

✅ **Test 1:** VERIFICATION_PROTOCOL constant created (4,774 chars)  
✅ **Test 2:** read_file tool registered with new parameter  
✅ **Test 3:** read_file without verification works (backward compatible)  
✅ **Test 4:** read_file with verification returns both code and rules  
✅ **Test 5:** Verification protocol includes all 7 steps  
✅ **Test 6:** Common pitfalls section included  
✅ **Test 7:** Python syntax valid  
✅ **Test 8:** No linter errors  

---

## Files Modified

- **a.py**
  - Lines 710-837: Added `VERIFICATION_PROTOCOL` constant
  - Lines 3758-3791: Updated `read_file` method with `include_verification_rules` parameter

**Total lines added:** ~130 lines  
**Total lines modified:** ~35 lines

---

## Migration Guide

### For Existing Code
No changes required! The new parameter is optional with a default value of `False`, so all existing calls to `read_file` continue to work as before.

### For New Code
When you want verification guidance, simply add the parameter:

```python
# Old way (still works)
content = read_file("file.py", True, 1, 100)

# New way (with verification)
content = read_file("file.py", True, 1, 100, include_verification_rules=True)
```

---

## Future Enhancements

Potential improvements for future iterations:

1. **Selective Steps:** Allow requesting specific verification steps (e.g., only Step 1 and Step 5)
2. **Language-Specific Rules:** Different verification protocols for different languages
3. **Custom Rules:** Allow users to provide custom verification checklists
4. **Verification History:** Track which verification steps have been completed
5. **Interactive Mode:** Guide the agent through each step with prompts

---

## Conclusion

The updated `read_file` tool provides a powerful new capability for systematic code verification. By combining file contents with verification guidelines in a single tool call, the agent can perform more thorough and consistent code reviews, leading to higher-quality bug fixes and more robust solutions.

**Status:** ✅ Production Ready  
**Backward Compatible:** ✅ Yes  
**Tested:** ✅ Yes  
**Documented:** ✅ Yes  

