# BugFix Workflow System Prompt Update 🔧

**Date:** 2025-10-23  
**Objective:** Restructure BugFixSolver prompts to follow a clear, systematic 9-step workflow  
**Status:** ✅ **COMPLETED**

---

## 🎯 User Requirements

The user requested a specific workflow where the agent:

1. **NEVER edits existing test files** - Only creates NEW test scripts
2. Creates reproduction scripts that catch ALL problem statement cases
3. Iterates on reproduction until it fully captures the problem
4. Fixes the code after confirming reproduction
5. Validates with **Fail-to-Pass (F2P)** - reproduction script passes
6. Validates with **Pass-to-Pass (P2P)** - existing tests still pass
7. Analyzes P2P failures intelligently:
   - **Case A**: Test expects old buggy behavior → Ignore (acceptable)
   - **Case B**: Test expects correct behavior → Fix is wrong, iterate

---

## 📋 New Workflow Structure

### **Three-Phase Approach**

```
Phase 1: REPRODUCE (Steps 1-4)
    ↓
Phase 2: FIX (Steps 5-6)
    ↓
Phase 3: VALIDATE (Steps 7-9)
```

---

## 🔄 Detailed Step-by-Step Workflow

### **PHASE 1: REPRODUCE THE BUG** 🐛

#### **Step 1: Understand the Problem Statement**
- Read problem statement carefully
- Identify expected vs. actual behavior
- List ALL edge cases and scenarios
- Note expected inputs and outputs

**Success Criteria:**
- ✓ Can explain the bug clearly
- ✓ Know what behavior needs fixing

---

#### **Step 2: Locate Relevant Source Code Files**
- Use `ls`, `find`, `grep` to explore repo
- Search for related files/functions/classes
- Read and understand current implementation
- Identify files containing the bug

**Tools:** `bash`, `str_replace_editor (view)`

**Success Criteria:**
- ✓ Identified 1-5 source files to fix
- ✓ Understand how buggy code works

---

#### **Step 3: Create NEW Reproduction Test Script**
- **CRITICAL:** Create NEW file (e.g., `reproduce_issue.py`)
- **DO NOT** edit existing test files
- Cover ALL problem statement scenarios
- Include all edge cases

**Example Structure:**
```python
# reproduce_issue.py
# This script reproduces the bug in issue #XXX

def test_case_1():
    # Test scenario 1 from problem statement
    assert buggy_function(input1) == expected1

def test_case_2():
    # Test scenario 2 - edge case
    assert buggy_function(input2) == expected2

if __name__ == "__main__":
    test_case_1()
    test_case_2()
    print("All tests passed!")
```

**Tools:** `str_replace_editor (create)`

**Success Criteria:**
- ✓ NEW test script created (not editing existing)
- ✓ Script covers all scenarios

---

#### **Step 4: Run Reproduction Script & Confirm Bug**
- Run: `python reproduce_issue.py`
- **VERIFY:** Script should FAIL (catching the bug)
- Error should match problem statement
- If passes → bug not reproduced → back to Step 3
- If misses scenarios → update script → run again
- Iterate until fully reproduces problem

**Tools:** `bash`

**Success Criteria:**
- ✓ Reproduction script FAILS with matching error
- ✓ All problem scenarios caught
- ✓ Confirmed output showing failure

**⚠️ MANDATORY:** Do NOT proceed until reproduction reliably fails!

---

### **PHASE 2: FIX THE BUG** 🔧

#### **Step 5: Analyze Root Cause & Plan Fix**
- Use `sequential_thinking` tool (totalThoughts: 10-25)
- Analyze: Why is the bug happening?
- Brainstorm 5-7 possible solutions
- Evaluate each (pros/cons, edge cases)
- Choose best approach
- Plan specific code changes

**Tools:** `sequential_thinking`

**Success Criteria:**
- ✓ Root cause identified
- ✓ Fix approach chosen and justified
- ✓ Implementation plan clear

---

#### **Step 6: Implement the Fix**
- Edit source code file(s) from Step 2
- Implement fix planned in Step 5
- Keep changes minimal and focused
- Handle all edge cases
- Follow best practices

**Tools:** `str_replace_editor (str_replace)`

**Success Criteria:**
- ✓ Source code updated
- ✓ Changes minimal and targeted
- ✓ Code clean and well-structured

---

### **PHASE 3: VALIDATE THE FIX** ✅

#### **Step 7: Fail-to-Pass (F2P) Validation**
- Run reproduction script: `python reproduce_issue.py`
- **VERIFY:** Script should now PASS
- Check ALL test cases pass
- If fails:
  - Analyze why fix didn't work
  - Go back to Step 5, refine fix
  - Iterate 5→6→7 until passes

**Tools:** `bash`

**Success Criteria:**
- ✓ Reproduction script NOW PASSES (was failing)
- ✓ All problem scenarios resolved
- ✓ Output confirms all tests pass

**⚠️ MANDATORY:** Script must pass before Step 8!

---

#### **Step 8: Pass-to-Pass (P2P) Validation**

**8.1. Understand Test Framework**
- Check: pytest.ini, setup.py, tox.ini, pyproject.toml
- Read: README.md, CONTRIBUTING.md
- Identify test framework (pytest, unittest, etc.)
- Determine test command

**8.2. Find Relevant Existing Test Files**
- Search: `find . -name "test_*.py"`
- Identify tests related to fixed code
- Look in: tests/, test/, <module>/tests/
- Grep for relevant functions/classes

**8.3. Run Existing Test Suite**
- Run: `pytest` (or appropriate command)
- Run: `pytest path/to/relevant_test.py`
- **VERIFY:** Check which pass/fail

**8.4. Analyze Any Failures**

For EACH failing test, determine:

**Case A: Test expects OLD (buggy) behavior**
- ✅ Failure is EXPECTED and ACCEPTABLE
- Test was validating buggy behavior
- **Action:** Ignore this failure
- Document why it's expected

**Case B: Test expects CORRECT behavior**
- ❌ Failure is PROBLEMATIC
- Your fix broke something that should work
- **Action:** Go back to Step 5, refine fix
- Fix should not break correct functionality

**How to distinguish Case A vs B:**
1. Read test code carefully
2. Understand what test validates
3. Check if test aligns with problem statement
4. If test contradicts fix requirements → Case A
5. If test validates legitimate functionality → Case B

**Tools:** `bash`, `str_replace_editor (view)`

**Success Criteria:**
- ✓ Understand test framework and commands
- ✓ Found and ran all relevant tests
- ✓ All pass OR failures analyzed as Case A
- ✓ No Case B failures (fix doesn't break valid code)

---

#### **Step 9: Final Validation & Completion**
- Run reproduction script final time → PASS
- Run full test suite → PASS (or Case A fails)
- Review all problem requirements → all addressed
- Verify edge cases handled
- Confirm no regressions

**Final Checklist:**
- ☐ Reproduction script passes (F2P ✓)
- ☐ Existing tests pass or have justified Case A failures (P2P ✓)
- ☐ All problem statement scenarios resolved
- ☐ Code changes minimal and clean
- ☐ No new bugs introduced

**If ALL checked:**
- ✅ Call 'complete' tool

**If ANY unchecked:**
- ❌ Go back to relevant step and iterate

---

## 🚫 Critical Rules & Restrictions

### **DO NOT Edit Existing Test Files**

❌ **CANNOT edit:**
- `tests/test_*.py`
- `test_*.py` in project directories
- Any file in `tests/` directory
- `*_test.py` files

✅ **CAN create:**
- `reproduce_issue.py` (your test script)
- `verify_fix.py` (additional validation)
- `test_scenario_X.py` (new test file in root)

### **Test Editing Policy**

**STRICT RULE:** Do NOT edit existing test files under ANY circumstances.

✅ **Correct approach:**
```
Create: reproduce_issue.py      (new file)
Create: test_bug_fix.py         (new file)
Create: verify_fix.py           (new file)
```

❌ **Incorrect approach:**
```
Edit: tests/test_module.py      (existing file)
Modify: test_feature.py         (existing file)
Update: anything in tests/      (existing directory)
```

---

## ⚠️ Critical Warnings

### **1. DO NOT SKIP REPRODUCTION**
- MUST create and run reproduction script BEFORE fixing
- Fixing without reproduction = HIGH RISK of incomplete/wrong fix

### **2. DO NOT ASSUME TESTS PASS**
- ALWAYS run tests and check actual output
- "The tests should pass" ≠ "The tests do pass"

### **3. DO NOT IGNORE P2P FAILURES**
- If existing tests fail (Case B), fix is incomplete
- Properly analyze EACH failure before concluding

### **4. DO NOT REPEAT FAILED COMMANDS**
- If command fails twice, change approach
- Don't repeat same failed command >2x

---

## 🎯 Core Principles

1. **Reproduce Before Fix**
   - Create test script that reproduces problem BEFORE attempting fix
   
2. **Non-Invasive Testing**
   - NEVER edit existing test files - always create NEW scripts
   
3. **Verification-Driven**
   - Every claim must be backed by actual test execution output
   
4. **Fail-to-Pass First, Pass-to-Pass Second**
   - Validate fix works AND doesn't break existing functionality

---

## 📊 Validation Requirements

### **Two-Phase Validation is MANDATORY**

#### **Phase A: Fail-to-Pass (F2P) Validation**
1. Run YOUR reproduction script (the one you created)
2. Confirm it NOW PASSES (it should have failed before the fix)
3. If still fails → iterate on the fix

#### **Phase B: Pass-to-Pass (P2P) Validation**
1. Find ALL relevant existing test files
2. Run existing test suite
3. Analyze any failures:
   - Test expects old (buggy) behavior → Ignore (expected)
   - Test expects correct behavior but fails → Fix is wrong, iterate

---

## ✅ Completion Criteria

Call 'complete' tool ONLY when ALL criteria met:

1. ✓ Reproduction script was created and initially failed
2. ✓ Code fix was implemented
3. ✓ Reproduction script now passes (F2P validation)
4. ✓ Existing relevant tests still pass (P2P validation)
5. ✓ All scenarios in problem statement are addressed

---

## 🛠️ Tool Usage Guide

### **sequential_thinking**
- **When:** Step 5 (root cause analysis)
- **Set totalThoughts:** 10-25 for thorough analysis
- **Use for:** Exploring solution options, analyzing failures

### **bash**
- **When:** Steps 2, 4, 7, 8 (exploration, running tests)
- **Commands:** ls, find, grep, python, pytest
- **Can:** Set environment variables if needed

### **str_replace_editor**
- **When:** Steps 2, 3, 6, 8 (viewing/creating/editing files)
- **Commands:**
  - `view`: Read file contents
  - `create`: Make NEW reproduction script
  - `str_replace`: Edit source code for fix
- **Note:** Use specific old_str for unambiguous replacements

---

## 📈 Prompt Structure Changes

### **System Prompt (FIX_TASK_SYSTEM_PROMPT)**

**Before:**
- Generic guidelines about testing
- Mixed advice about test running
- No clear test editing policy
- Vague validation requirements

**After:**
- ✅ Clear three-phase approach
- ✅ Explicit "DO NOT edit existing tests" policy
- ✅ Structured F2P and P2P validation requirements
- ✅ Core principles and completion criteria
- ✅ Detailed restrictions and best practices

### **Instance Prompt (FIX_TASK_INSTANCE_PROMPT_TEMPLATE)**

**Before:**
- 11 loosely organized steps
- Mixed instructions
- Unclear about test editing
- Vague validation process

**After:**
- ✅ 9 clearly defined steps organized in 3 phases
- ✅ Visual structure with boxes and sections
- ✅ Explicit instructions for each step
- ✅ Clear success criteria per step
- ✅ Detailed P2P failure analysis (Case A vs B)
- ✅ Comprehensive tool usage guide
- ✅ Strong critical warnings section

---

## 🎨 Visual Improvements

The new prompt uses:
- Box drawing characters (`╔═══╗`, `┌───┐`, etc.)
- Clear phase headers
- Step-by-step formatting
- Emoji indicators (🧠, 🔍, 🧪, etc.)
- Success criteria checkboxes (✓, ☐)
- Warning symbols (⚠️)
- Action indicators (✅, ❌)

This makes the workflow:
- **More scannable** - Easy to see current step
- **More structured** - Clear phases and progression
- **More actionable** - Specific tasks and tools per step
- **More enforceable** - Hard to skip or misunderstand

---

## 💡 Key Improvements

### **1. Non-Invasive Testing**
- Explicit rule: NEVER edit existing test files
- Always create NEW reproduction scripts
- Clear examples of correct vs incorrect approach

### **2. Systematic Reproduction**
- Reproduction is mandatory BEFORE fixing
- Iterate on reproduction until perfect
- Must confirm failure before proceeding

### **3. Intelligent P2P Analysis**
- Case A vs Case B framework
- Clear guidance on which failures to ignore
- Instructions for analyzing each failure

### **4. Structured Validation**
- F2P: Your script passes
- P2P: Existing tests still pass (or justified failures)
- Clear success criteria for completion

### **5. Workflow Enforcement**
- Sequential phases with dependencies
- Cannot skip phases or steps
- Success criteria gate progression

---

## 🔍 Before vs After Comparison

| Aspect | Before | After |
|--------|--------|-------|
| **Test Editing** | Vague "don't break tests" | Explicit "NEVER edit existing tests" |
| **Reproduction** | Mentioned but not enforced | Mandatory Phase 1 with validation |
| **F2P Validation** | Implied | Explicit Step 7 with criteria |
| **P2P Validation** | Generic "run tests" | Detailed Step 8 with Case A/B analysis |
| **Workflow Structure** | 11 loose steps | 9 steps in 3 clear phases |
| **Completion Criteria** | Vague | 5-point checklist |
| **Visual Clarity** | Plain text | Structured with boxes and emojis |
| **Tool Guidance** | Scattered | Dedicated guide section |

---

## 📝 Example Workflow Execution

```
User provides problem statement: "Function returns wrong value for negative inputs"

Step 1: Agent reads and understands the problem
  ↓
Step 2: Agent finds relevant source files (math_utils.py)
  ↓
Step 3: Agent creates reproduce_issue.py
  ```python
  # reproduce_issue.py
  from math_utils import calculate
  
  def test_negative_input():
      result = calculate(-5)
      expected = 10
      assert result == expected, f"Expected {expected}, got {result}"
  
  if __name__ == "__main__":
      test_negative_input()
      print("All tests passed!")
  ```
  ↓
Step 4: Agent runs: python reproduce_issue.py
  Output: AssertionError: Expected 10, got -10
  ✓ Bug confirmed
  ↓
Step 5: Agent uses sequential_thinking to analyze root cause
  Finds: Missing abs() call
  ↓
Step 6: Agent edits math_utils.py to add abs()
  ↓
Step 7: Agent runs: python reproduce_issue.py
  Output: All tests passed!
  ✓ F2P validation complete
  ↓
Step 8: Agent runs: pytest tests/
  Output: 42 passed, 1 failed (test_legacy_behavior)
  Agent analyzes: test_legacy_behavior expects negative output (Case A)
  ✓ P2P validation complete (justified failure)
  ↓
Step 9: Final checklist → All ✓
  Agent calls complete tool
```

---

## ✅ Implementation Status

- [x] System prompt updated with clear principles
- [x] Instance prompt restructured into 9 steps
- [x] Three-phase workflow defined
- [x] Test editing policy made explicit
- [x] F2P validation detailed
- [x] P2P validation with Case A/B analysis
- [x] Completion criteria specified
- [x] Critical warnings added
- [x] Tool usage guide created
- [x] Visual formatting improved
- [x] Linter checks passed (0 errors)
- [x] Documentation created

---

## 🎉 Summary

The BugFixSolver prompts have been completely restructured to follow a systematic, test-driven workflow:

**✅ What Changed:**
- Clear 9-step workflow in 3 phases
- Explicit "no editing existing tests" policy
- Mandatory reproduction before fixing
- Detailed F2P and P2P validation
- Intelligent failure analysis (Case A vs B)
- Visual structure with boxes and formatting

**✅ Why It Matters:**
- Reduces risk of incomplete fixes
- Prevents breaking existing functionality
- Ensures thorough validation
- Makes workflow impossible to skip or misunderstand
- Provides clear guidance for complex scenarios

**✅ Expected Outcomes:**
- Higher quality bug fixes
- Better test coverage
- Fewer regressions
- More systematic approach
- Clearer agent behavior

---

*The bug fixing workflow is now crystal clear, systematic, and enforced! 🚀*

