# CreateProblemSolver Response Format Fix 🔧

**Date:** $(Get-Date)  
**Issue:** LLM in CreateProblemSolver was generating wrong format output  
**Status:** ✅ **FIXED**

---

## 🐛 Problem Identified

The `CreateProblemSolver` was not enforcing response format as strictly as `BugFixSolver`, causing the LLM to produce inconsistent output formats.

### Root Cause Analysis

| Aspect | BugFixSolver (Working ✅) | CreateProblemSolver (Broken ❌) |
|--------|--------------------------|----------------------------------|
| **return_type** | `tuple[str,str]` (enforces section parsing) | `str` (no section enforcement) |
| **response_format parameter** | Shows format in prompt | Empty string `""` |
| **Validation** | Strict section validation | Basic validation |
| **System prompt** | Clear format enforcement | Vague format mention |

---

## ✅ Solution Implemented

### 1. **Enhanced System Prompt** (`SYSTEM_PROMPT_INITIAL_SOLUTION_EVAL`)

**Added:**
```python
Rules:-
    ...
    5. **CRITICAL OUTPUT FORMAT:** You MUST respond in the exact format specified below. 
       Do not add extra sections or deviate from this format.

**STRICT RESPONSE FORMAT - YOU MUST FOLLOW THIS EXACTLY:**
{format_prompt}

**Remember:** Your response MUST contain exactly ONE THOUGHT section followed by 
ONE TOOL_CALL section. Do not add anything before the THOUGHT section or after 
the TOOL_CALL section.
```

**Impact:** Makes format requirements crystal clear to the LLM.

---

### 2. **Stricter Response Format** (`RESPONSE_FORMAT_SOLUTION_EVAL_2`)

**Before:**
```python
RESPONSE_FORMAT_SOLUTION_EVAL_2="""
Your response must not contain multiple TOOL_CALL sections. You must add your 
detailed analysis before TOOL_CALL section. You must respond in the following format.
===================THOUGHT
<<your detailed thought process>>
===================TOOL_CALL
{"name":"<tool_name>","arguments":{...}}
"""
```

**After:**
```python
RESPONSE_FORMAT_SOLUTION_EVAL_2=textwrap.dedent("""
**CRITICAL: You MUST respond EXACTLY in this format. No deviations allowed.**

Your response must contain EXACTLY TWO sections in this order:
1. ONE THOUGHT section
2. ONE TOOL_CALL section

===================THOUGHT
<<your detailed thought process and analysis>>
===================TOOL_CALL
{"name":"<tool_name>","arguments":{<your arguments here>}}

**RULES:**
- Do NOT add any text before the THOUGHT section
- Do NOT add multiple THOUGHT sections
- Do NOT add multiple TOOL_CALL sections
- Do NOT add any text after the TOOL_CALL section
- The TOOL_CALL must be valid JSON
""")
```

**Impact:** 
- ✅ Crystal clear about exact requirements
- ✅ Lists specific rules
- ✅ Uses bold and capitalization for emphasis
- ✅ Numbers the sections

---

### 3. **Enhanced Validation** (`ResponseValidator.check_tool_call_section`)

**Before:**
```python
def check_tool_call_section(cls, response: str, raw_response: str, correct_format: str)->str:
    if not("TOOL_CALL" in raw_response and re.search(r"^=+\s*[A-Z_]+$", raw_response, re.MULTILINE)):
        return "Invalid response, please respond in correct format: {correct_format}"
    return "success"
```

**After:**
```python
def check_tool_call_section(cls, response: str, raw_response: str, correct_format: str)->str:
    """Validate that the response has proper THOUGHT and TOOL_CALL sections."""
    # Check for required sections
    if not re.search(r"^=+\s*THOUGHT\s*$", raw_response, re.MULTILINE):
        return f"Missing THOUGHT section. You must respond in this exact format:\n{correct_format}"
    
    if not re.search(r"^=+\s*TOOL_CALL\s*$", raw_response, re.MULTILINE):
        return f"Missing TOOL_CALL section. You must respond in this exact format:\n{correct_format}"
    
    # Check for multiple sections (which is not allowed)
    thought_count = len(re.findall(r"^=+\s*THOUGHT\s*$", raw_response, re.MULTILINE))
    tool_call_count = len(re.findall(r"^=+\s*TOOL_CALL\s*$", raw_response, re.MULTILINE))
    
    if thought_count > 1:
        return f"ERROR: Found {thought_count} THOUGHT sections. You must have EXACTLY ONE THOUGHT section.\n{correct_format}"
    
    if tool_call_count > 1:
        return f"ERROR: Found {tool_call_count} TOOL_CALL sections. You must have EXACTLY ONE TOOL_CALL section.\n{correct_format}"
    
    # Check if there's content before THOUGHT section
    first_thought_pos = raw_response.find("=")
    if first_thought_pos > 0 and raw_response[:first_thought_pos].strip():
        return f"ERROR: Do not add any content before the THOUGHT section. Your response must start with ===================THOUGHT\n{correct_format}"
    
    return "success"
```

**Improvements:**
- ✅ Explicitly checks for both THOUGHT and TOOL_CALL sections
- ✅ Detects multiple sections and provides specific error
- ✅ Checks for content before THOUGHT section
- ✅ Provides detailed, actionable error messages
- ✅ Shows the correct format when validation fails

---

### 4. **Updated solve_task Calls**

Changed **3 locations** in `CreateProblemSolver.solve_problem()`:

#### Change 1: Initial solve_task call
```python
# BEFORE:
response=await self.agent_initial_solution_eval.solve_task(
    ...,
    response_format="",  # ❌ Empty
    ...,
    return_type=str      # ❌ No section enforcement
)

# AFTER:
response=await self.agent_initial_solution_eval.solve_task(
    ...,
    response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2,  # ✅ Show format
    ...,
    return_type=tuple[str,str]  # ✅ Enforce sections
)
```

#### Change 2: Loop solve_task call
```python
# BEFORE:
response=await self.agent_initial_solution_eval.solve_task(
    str(response),
    response_format="",  # ❌ Empty
    ...,
    return_type=str      # ❌ No enforcement
)

# AFTER:
response=await self.agent_initial_solution_eval.solve_task(
    str(response),
    response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2,  # ✅ Show format
    ...,
    return_type=tuple[str,str]  # ✅ Enforce sections
)
```

#### Change 3: Final check solve_task call
```python
# BEFORE:
response=await self.agent_initial_solution_eval.solve_task(
    "Check the problem statement...",
    response_format="",           # ❌ Empty
    post_process_func=None,       # ❌ No validation
    ...,
    return_type=str               # ❌ No enforcement
)

# AFTER:
response=await self.agent_initial_solution_eval.solve_task(
    "Check the problem statement...",
    response_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2,  # ✅ Show format
    post_process_func=partial(CustomAssistantAgent.ResponseValidator.check_tool_call_section,
                             correct_format=CreateProblemSolver.RESPONSE_FORMAT_SOLUTION_EVAL_2),  # ✅ Validate
    ...,
    return_type=tuple[str,str]    # ✅ Enforce sections
)
```

---

## 📊 Impact Analysis

### Before Fix

```
LLM Output Examples (Wrong):
----------------------------
1. Missing sections:
   "I think we should test the function..."
   
2. Multiple THOUGHT sections:
   ===================THOUGHT
   Analysis 1
   ===================THOUGHT
   Analysis 2
   ===================TOOL_CALL
   {...}
   
3. Content before THOUGHT:
   "Let me analyze this first."
   ===================THOUGHT
   ...
   
4. Multiple TOOL_CALL sections:
   ===================THOUGHT
   ...
   ===================TOOL_CALL
   {...}
   ===================TOOL_CALL
   {...}
```

### After Fix

```
LLM Output (Correct):
--------------------
===================THOUGHT
My detailed analysis of the problem. I will test the function
by creating comprehensive test cases.
===================TOOL_CALL
{"name":"run_code","arguments":{"file_path":"test.py","content":"..."}}
```

---

## ✅ Validation Improvements

| Check | Before | After |
|-------|--------|-------|
| **THOUGHT section exists** | ❌ Generic check | ✅ Specific regex check |
| **TOOL_CALL section exists** | ❌ Generic check | ✅ Specific regex check |
| **Multiple THOUGHT sections** | ❌ Not checked | ✅ Counted and rejected |
| **Multiple TOOL_CALL sections** | ❌ Not checked | ✅ Counted and rejected |
| **Content before THOUGHT** | ❌ Not checked | ✅ Detected and rejected |
| **Error messages** | ❌ Generic | ✅ Specific with format |
| **Format shown in error** | ❌ Not shown | ✅ Full format shown |

---

## 🎯 Key Improvements

### 1. **Consistency with BugFixSolver**
Now `CreateProblemSolver` uses the same strict enforcement strategy as `BugFixSolver`:
- ✅ `return_type=tuple[str,str]` forces section parsing
- ✅ `response_format` parameter shows the expected format
- ✅ Post-processing validation catches errors

### 2. **Clearer Error Messages**
```python
# OLD:
"Invalid response, please respond in correct format: {correct_format}"

# NEW:
"ERROR: Found 3 THOUGHT sections. You must have EXACTLY ONE THOUGHT section.
**CRITICAL: You MUST respond EXACTLY in this format. No deviations allowed.**
..."
```

### 3. **Multiple Validation Layers**

1. **Prompt Level:** System prompt clearly states format requirements
2. **Response Format:** Detailed format with rules shown to LLM
3. **Return Type:** `tuple[str,str]` enforces 2-section parsing
4. **Validation Function:** Comprehensive checks with specific errors

### 4. **Actionable Feedback**
When LLM makes a mistake, it receives:
- ✅ What went wrong (specific error)
- ✅ What was expected (full format)
- ✅ How to fix it (clear rules)

---

## 🔍 Technical Details

### Section Parsing (`return_type=tuple[str,str]`)

When `return_type=tuple[str,str]` is used, the `parse_markdown()` method:
1. Searches for section headers matching `^=+\s*[A-Z_]+$`
2. Extracts content between sections
3. Returns a tuple of (thought_content, tool_call_content)
4. **Fails** if it can't find exactly 2 sections

This provides an **additional layer of enforcement** beyond just validation.

### Regex Patterns Used

```python
# Section header detection:
r"^=+\s*THOUGHT\s*$"     # Matches: ===================THOUGHT
r"^=+\s*TOOL_CALL\s*$"   # Matches: ===================TOOL_CALL

# Generic section detection:
r"^=+\s*[A-Z_]+$"        # Matches: ===================ANY_CAPS_TEXT
```

---

## 📝 Summary

### Changes Made
1. ✅ Enhanced system prompt with explicit format requirements
2. ✅ Improved `RESPONSE_FORMAT_SOLUTION_EVAL_2` with clear rules
3. ✅ Rewrote `ResponseValidator.check_tool_call_section()` with comprehensive checks
4. ✅ Updated all 3 `solve_task` calls to use `tuple[str,str]` and show format
5. ✅ Added format to all validation calls

### Expected Outcomes
- ✅ LLM produces consistent, correctly formatted output
- ✅ Format errors are caught immediately with actionable feedback
- ✅ `CreateProblemSolver` now matches `BugFixSolver` quality
- ✅ Reduced retry attempts due to clearer requirements

### Testing
```python
# Test the validator:
validator = CustomAssistantAgent.ResponseValidator

# Should pass:
valid = """
===================THOUGHT
My analysis
===================TOOL_CALL
{"name":"test"}
"""
assert validator.check_tool_call_section("", valid, "") == "success"

# Should fail:
invalid1 = """
Some text
===================THOUGHT
My analysis
"""
assert "Missing TOOL_CALL" in validator.check_tool_call_section("", invalid1, "")

invalid2 = """
===================THOUGHT
Analysis 1
===================THOUGHT
Analysis 2
===================TOOL_CALL
{"name":"test"}
"""
assert "EXACTLY ONE THOUGHT" in validator.check_tool_call_section("", invalid2, "")
```

---

## 🚀 Migration Notes

**No breaking changes** - This is purely a fix for existing functionality:
- Existing code structure unchanged
- Only enforcement improved
- Backward compatible with proper format responses

---

## ✅ Completion Checklist

- [x] System prompt updated with format requirements
- [x] Response format clarified with explicit rules
- [x] Validation function enhanced with specific checks
- [x] All solve_task calls updated to enforce format
- [x] `return_type` changed from `str` to `tuple[str,str]`
- [x] `response_format` parameter populated (not empty)
- [x] Post-processing validation added to all calls
- [x] Linter checks passed (0 errors)
- [x] Documentation created

---

*The CreateProblemSolver now enforces strict response format matching BugFixSolver's quality! 🎉*

