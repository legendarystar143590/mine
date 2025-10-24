# Agent Workflow Structure - Hierarchical Step-by-Step Guide

## Overview
This document outlines the complete hierarchical workflow structure implemented in `v3.py`. Each step is broken down into detailed sub-steps with clear approaches and rules.

---

## 📋 Workflow Steps Summary

| Step | Name | Priority | Tool(s) Used | Output |
|------|------|----------|--------------|---------|
| 0 | Strategic Planning | HIGH (Optional) | create_meta_plan | JSON execution plan |
| 1 | Find Relevant Files | CRITICAL | search_in_all_files_content | List of relevant files |
| 2 | Localize the Issue | CRITICAL | get_file_content, get_classes, get_functions | Identified problematic code |
| 3 | Create Test Script | CRITICAL | generate_test_function, run_code | Test script (initially failing) |
| 4 | Propose Solutions | CRITICAL | None | 2+ solution approaches |
| 5 | Reflect on Solutions | CRITICAL | reflect_on_solution | Critique with confidence scores |
| 6 | Get User Approval | CRITICAL | get_approval_for_solution | User approval |
| 7 | Implement Solution | CRITICAL | apply_code_edit, save_file | Modified codebase |
| 8 | Run and Verify Tests | CRITICAL | run_repo_tests, run_code | Test results (all passing) |
| 9 | Validate Solution | CRITICAL | validate_solution | Validation report (score ≥85) |
| 10 | Iterative Refinement | CRITICAL (If needed) | refine_solution | Improved solution |
| 11 | Final Verification | HIGH | search_in_all_files_content, run_repo_tests | Final verification |
| 12 | Complete Task | CRITICAL | finish | Git patch |

---

## 🔍 Detailed Step Breakdown

### Step 0: Strategic Planning (Optional but Recommended)
**Priority:** HIGH  
**When:** Use for complex problems with multiple components or unclear approaches  
**Tool:** `create_meta_plan(problem_statement, project_context)`

#### Sub-steps:
**0.1 - Analyze Problem Comprehensively**
- **Approach:** Break down problem statement into core components
- **Actions:**
  1. Identify problem type (algorithmic, data structure, API, business logic, performance)
  2. Extract ALL requirements (explicit and implicit)
  3. Identify constraints (time, space, compatibility, dependencies)
  4. List all edge cases mentioned or inferred
  5. Identify affected components (files, classes, functions)
- **Rules:**
  - Read problem statement at least twice
  - Look for implicit requirements
  - Consider system-wide implications

**0.2 - Decompose into Sub-Tasks**
- **Approach:** Break into 3-5 manageable sub-tasks
- **Actions:**
  1. List major tasks needed
  2. Identify dependencies between tasks
  3. Estimate complexity (low/medium/high)
  4. Identify risks and mitigation strategies
  5. Create ordered execution sequence
- **Rules:**
  - Each sub-task should be independently verifiable
  - Keep focused - break down if too complex
  - Always consider dependencies

**0.3 - Evaluate Approaches**
- **Approach:** Generate and compare 2-3 different approaches
- **Actions:**
  1. List 2-3 fundamentally different approaches
  2. For each: identify pros, cons, complexity, risks
  3. Consider minimal change vs. refactoring
  4. Select best approach with justification
  5. Identify fallback approach
- **Rules:**
  - Approaches should be meaningfully different
  - Consider backward compatibility impact
  - Prefer simpler approaches unless justified
  - Document selection rationale

**0.4 - Define Verification Strategy**
- **Approach:** Plan how to verify correctness
- **Actions:**
  1. Define success criteria
  2. List test categories needed
  3. Plan verification checkpoints
  4. Define rollback strategy
- **Rules:**
  - Success criteria must be measurable
  - Plan tests BEFORE implementation
  - Include regression testing

**Output:** Comprehensive JSON plan with problem analysis, sub-tasks, approaches, verification strategy  
**Benefit:** Reduces wasted effort 40-60%, identifies issues early

---

### Step 1: Find Relevant Files
**Priority:** CRITICAL  
**Goal:** Locate all files that need examination or modification

#### Sub-steps:
**1.1 - Identify Key Terms**
- **Actions:**
  1. Extract key terms from problem statement
  2. Identify domain-specific terminology
  3. Note file paths or module names mentioned
- **Rules:**
  - Include term variations
  - Search error messages exactly as stated

**1.2 - Perform Broad Search**
- **Tool:** `search_in_all_files_content(search_term, case_sensitive=False)`
- **Actions:**
  1. Search for primary terms across codebase
  2. Review results to identify relevant files
  3. Note frequently appearing files
- **Rules:**
  - Start with most specific terms
  - If >50 results, refine search
  - If no results, try variations

**1.3 - Examine Project Structure**
- **Actions:**
  1. Review provided project_structure
  2. Identify patterns (tests/, src/, etc.)
  3. Locate relevant configuration files
- **Rules:**
  - Understand organization before changes
  - Look for corresponding test files

**1.4 - Create File List**
- **Actions:**
  1. List all files needing examination
  2. Prioritize: core issue files first
  3. Note source vs. test files
- **Rules:**
  - Include test files (provide context)
  - Don't skip tangentially related files

**Output:** Prioritized list of relevant files

---

### Step 2: Localize the Issue
**Priority:** CRITICAL  
**Goal:** Pinpoint exact code causing issue at variable, function, class level  
**Approach:** Narrow from file → class → function → variables

#### Sub-steps:
**2.1 - Read Relevant Files**
- **Tools:** `get_file_content`, `get_classes`, `get_functions`
- **Actions:**
  1. Start with most likely file
  2. Read full if <500 lines, targeted for large files
  3. Understand overall structure first
- **Rules:**
  - Read imports first
  - Read signatures before implementations
  - Take notes on components

**2.2 - Locate Relevant Classes**
- **Tool:** `get_classes([file::class1, file::class2])`
- **Actions:**
  1. Identify classes from problem statement
  2. Examine structure: attributes, methods, inheritance
  3. Understand responsibilities and interactions
- **Rules:**
  - Check `__init__` for initialization
  - Look for class vs instance variables
  - Check inherited methods

**2.3 - Locate Relevant Functions/Methods**
- **Tools:** `get_functions`, `search_in_specified_file_v2`
- **Actions:**
  1. Identify functions from problem/error
  2. Read complete implementation
  3. Trace logic flow
  4. Identify error location
- **Rules:**
  - Pay attention to control flow
  - Check parameters and return values
  - Look for calling functions

**2.4 - Identify Problematic Variables/Logic**
- **Actions:**
  1. Trace variable values through code
  2. Identify incorrect calculations/comparisons
  3. Look for off-by-one, missing checks
  4. Check for missing error handling
- **Rules:**
  - Consider variable scope
  - Check mutability issues
  - Look for race conditions
  - Verify type/value assumptions

**2.5 - Understand Root Cause**
- **Actions:**
  1. Identify root cause, not symptoms
  2. Trace from root cause to observed behavior
  3. Document WHY issue occurs
- **Rules:**
  - Keep asking "why"
  - Don't confuse symptoms with cause
  - Consider: design flaw vs. implementation bug

**Output:** Clear identification of problematic file(s), class(es), function(s), variables/logic

---

### Step 3: Create Comprehensive Test Script
**Priority:** CRITICAL  
**Goal:** Create test script that reproduces issue and validates fix  
**Approach:** Test-driven - tests fail initially, pass after fix  
**Tool:** `generate_test_function(file_path, test_function_code, position="auto")`

#### Sub-steps:
**3.1 - Design Test Strategy**
- **Approach:** Plan comprehensive coverage before writing
- **Actions:**
  1. List ALL scenarios from problem statement
  2. Identify edge cases (empty, None, boundaries, invalid types)
  3. Identify exception cases
  4. Plan test structure
- **Rules:**
  - Minimum: 5 normal, 5 edge, 3 boundary, 3 exception
  - One behavior per test
  - Descriptive names: `test_function_scenario_expected`

**3.2 - Write Test Cases**
- **Approach:** Follow arrange-act-assert pattern
- **Actions:**
  1. Write normal case tests first
  2. Write edge case tests
  3. Write boundary tests
  4. Write exception tests
- **Rules:**
  - Use ONLY pytest and unittest.mock
  - Mock ALL external dependencies
  - Each test independent
  - Include docstrings
  - Use pytest.mark.parametrize for similar tests

**3.3 - Add Bug Reproduction Test**
- **Approach:** Create test that fails due to current bug
- **Actions:**
  1. Write test reproducing exact issue
  2. Test should FAIL on current code
  3. Document why it should fail
- **Rules:**
  - Must pass after fix
  - Keep simple and focused

**3.4 - Run Tests to Verify Failures**
- **Tool:** `run_code(content=test_code, file_path="test_issue.py")`
- **Actions:**
  1. Run test script against current code
  2. Verify bug reproduction test fails
  3. Note which tests fail and why
- **Rules:**
  - Tests SHOULD fail before fix
  - If pass unexpectedly, re-examine understanding

**Test Requirements (MANDATORY):**
- Include ALL test cases from problem statement
- Test at variable, function, AND class level
- Use ONLY mockup/stub data
- Runnable with pytest ONLY
- Mock ALL external dependencies
- Test boundaries, None, empty, invalid, type mismatches
- Include bug reproduction test

**Output:** Test script that fails on current code, will pass after fix

---

### Step 4: Propose Solution Approaches
**Priority:** CRITICAL  
**Goal:** Propose at least 2 meaningfully different solutions

#### Sub-steps:
**4.1 - Generate Solution Options**
- **Actions:**
  1. Brainstorm 2-3 fundamentally different fixes
  2. For each: describe what changes, why it works, pros/cons
  3. Consider: minimal change, refactoring, algorithm change
- **Rules:**
  - Solutions must be meaningfully different
  - At least one minimal/conservative change
  - Consider short-term vs long-term maintainability

**4.2 - Analyze Each Solution**
- **Actions:**
  1. For each: identify files changed, functions modified, risk level
  2. Assess backward compatibility impact
  3. Estimate complexity and testing needs
  4. Consider side effects and unintended consequences
- **Rules:**
  - Be honest about risks and limitations
  - Document why each approach would work

**Output:** At least 2 well-described solution approaches with analysis

---

### Step 5: Reflect on Solutions
**Priority:** CRITICAL  
**Goal:** Self-critique to catch issues before implementation  
**Approach:** Use reflection agent for systematic review  
**Tool:** `reflect_on_solution(proposed_solution, problem_statement, solution_description)`

#### Sub-steps:
**5.1 - Perform Reflection**
- **Actions:**
  1. For each solution, call reflect_on_solution tool
  2. Review critique across 6 dimensions: correctness, completeness, robustness, efficiency, code_quality, backward_compatibility
  3. Note issues with severity ratings (CRITICAL/HIGH/MEDIUM/LOW)
- **Rules:**
  - Take reflection seriously
  - CRITICAL and HIGH issues must be addressed

**5.2 - Decide on Revisions**
- **Actions:**
  1. Review all identified issues
  2. Check revision_needed field
  3. If needed, improve solution addressing CRITICAL/HIGH issues
  4. Document changes made
- **Rules:**
  - If confidence_score < 70, seriously consider revision
  - Address all CRITICAL issues before approval
  - Don't ignore HIGH issues without reason

**5.3 - Select Best Solution**
- **Actions:**
  1. Compare reflection results for all solutions
  2. Select solution with highest confidence_score and fewest issues
  3. Document rationale
- **Rules:**
  - Prefer solution with fewer CRITICAL/HIGH issues
  - Consider complexity vs. benefit trade-off

**Benefit:** Catches 60-70% of issues before implementation  
**Output:** Refined solution(s) with confidence assessment

---

### Step 6: Get User Approval
**Priority:** CRITICAL  
**Goal:** Present solutions and get approval  
**Tool:** `get_approval_for_solution(solutions, selected_solution, reason_for_selection)`

#### Sub-steps:
**6.1 - Prepare Proposal**
- **Actions:**
  1. Format all solutions clearly
  2. Include reflection results and confidence scores
  3. Highlight selected solution and why
- **Rules:**
  - Be transparent about risks and trade-offs
  - Explain decisions in understandable terms

**6.2 - Request Approval**
- **Actions:**
  1. Call get_approval_for_solution
  2. Specify recommended solution (index)
  3. Provide clear reason_for_selection
- **Rules:**
  - Must have at least 2 solutions
  - Must explain why selected is best

**Output:** User approval to proceed

---

### Step 7: Implement Solution
**Priority:** CRITICAL  
**Goal:** Apply approved solution to codebase  
**Approach:** Precise, targeted changes while preserving functionality  
**Tools:** `apply_code_edit(file_path, search, replace)` or `save_file(file_path, content)`

#### Sub-steps:
**7.1 - Plan Implementation Order**
- **Actions:**
  1. List all files needing modification
  2. Determine order: dependencies first
  3. Identify core vs. supporting changes
- **Rules:**
  - Make changes in logical order
  - Consider: what if interrupted mid-way?

**7.2 - Make Core Changes**
- **Actions:**
  1. Start with primary fix to root cause
  2. Use apply_code_edit for surgical changes
  3. Use save_file only for new files/rewrites
  4. Double-check each change
- **Rules:**
  - Make minimal necessary changes
  - Preserve names/signatures unless must change
  - Keep focused - avoid scope creep
  - Add comments for non-obvious changes

**7.3 - Add Edge Case Handling**
- **Actions:**
  1. Add None value checks
  2. Add boundary validation
  3. Add exception handling
- **Rules:**
  - Handle gracefully, don't crash
  - Raise appropriate exceptions with clear messages
  - Don't break existing behavior

**7.4 - Ensure Backward Compatibility**
- **Actions:**
  1. Verify function signatures unchanged (or only extended)
  2. Verify class interfaces unchanged
  3. Check default behavior preserved
- **Rules:**
  - NEVER break compatibility unless required
  - If must break, add deprecation warnings first
  - Document compatibility considerations

**Output:** Modified codebase with fix implemented

---

### Step 8: Run and Verify Tests
**Priority:** CRITICAL  
**Goal:** Verify fix works and doesn't break anything  
**Approach:** Run comprehensive test suite  
**Tools:** `run_repo_tests([test_files])` or `run_code(content, file_path)`

#### Sub-steps:
**8.1 - Run Bug Reproduction Test**
- **Actions:**
  1. Run test from step 3 that initially failed
  2. Verify it now passes
  3. If still fails, analyze and return to step 7
- **Rules:**
  - This test MUST pass after fix
  - If doesn't pass, fix is incomplete/incorrect

**8.2 - Run Full Test Suite**
- **Actions:**
  1. Run all existing tests in repository
  2. Run all new tests from step 3
  3. Collect results: passed count, failed count, error details
- **Rules:**
  - ALL tests must pass (no regressions)
  - If any fail, must investigate

**8.3 - Analyze Test Failures**
- **Approach:** Systematically debug failures
- **Actions:**
  1. For each failure, understand why
  2. Determine: bug in fix, bug in test, or uncovered edge case
  3. Fix issue and re-run
- **Rules:**
  - Don't ignore failures - each must be resolved
  - Don't modify existing tests unless incorrect

**Output:** Test results showing all tests pass

---

### Step 9: Validate Solution Comprehensively
**Priority:** CRITICAL  
**Goal:** Comprehensive quality validation across all dimensions  
**Approach:** Use validation agent for systematic assessment  
**Tool:** `validate_solution(solution_code, test_results, problem_statement)`

#### Sub-steps:
**9.1 - Perform Validation**
- **Actions:**
  1. Gather solution code, test results, problem statement
  2. Call validate_solution tool
  3. Review validation report across all 5 dimensions
- **Rules:**
  - Must validate before calling finish()
  - Take results seriously

**9.2 - Check Passing Criteria**
- **Approach:** Verify solution meets quality thresholds
- **Criteria:**
  - Overall score ≥ 85
  - Functional correctness ≥ 90
  - Test coverage ≥ 80
  - Code quality ≥ 80
  - Performance meets requirements
  - No CRITICAL blocking issues
  - Backward compatible
- **Actions:**
  1. Check validation_passed boolean
  2. Review overall_score and category_scores
  3. Check blocking_issues for CRITICAL items
- **Rules:**
  - If validation_passed = false, must refine (go to step 10)
  - All category scores should be ≥ 80

**Output:** Validation report with pass/fail and scores

---

### Step 10: Iterative Refinement (If Needed)
**Priority:** CRITICAL  
**Goal:** Improve solution based on validation feedback  
**Approach:** Targeted improvements, highest priority first  
**Tool:** `refine_solution(current_solution, feedback, problem_statement, test_failures)`  
**When:** Only if validation fails (score < 85 or CRITICAL issues)

#### Sub-steps:
**10.1 - Analyze Feedback**
- **Actions:**
  1. Review all feedback: validation, test failures, blocking issues
  2. Categorize by severity: CRITICAL, HIGH, MEDIUM, LOW
  3. Identify root causes
- **Rules:**
  - Focus on CRITICAL first, then HIGH
  - Look for patterns - multiple issues may share cause

**10.2 - Apply Refinement**
- **Actions:**
  1. Call refine_solution with current code and feedback
  2. Review improved solution generated
  3. Apply improved solution to codebase
- **Rules:**
  - Review refined code before applying
  - Ensure working functionality preserved

**10.3 - Re-test and Re-validate**
- **Actions:**
  1. Re-run all tests (step 8)
  2. Re-run validation (step 9)
  3. Check if issues resolved
- **Rules:**
  - Track iteration count - maximum 3 iterations
  - If still failing after 3, reconsider approach

**Iteration Limit:** Maximum 3 refinement iterations  
**Output:** Improved solution meeting validation criteria

---

### Step 11: Final Verification
**Priority:** HIGH  
**Goal:** Final checks before completion  
**Approach:** Systematic final review

#### Sub-steps:
**11.1 - Verify Completeness**
- **Actions:**
  1. Review problem statement one more time
  2. Verify ALL requirements addressed
  3. Check ONLY requested changes made
- **Rules:**
  - Don't leave requirements unaddressed
  - Don't include unrelated changes

**11.2 - Check Repository-Wide Impact**
- **Actions:**
  1. Search for all uses of modified functions/classes
  2. Verify no unexpected breakage elsewhere
  3. Run full repository test suite if available
- **Rules:**
  - Use search_in_all_files_content to find usages
  - Don't assume changes are isolated

**11.3 - Review Patch**
- **Actions:**
  1. Review all changes made
  2. Verify no debug code/print statements left
  3. Ensure generated test files won't be in patch
- **Rules:**
  - Clean, professional code only
  - No temporary debugging code

**Output:** Clean, complete solution ready for finalization

---

### Step 12: Complete Task
**Priority:** CRITICAL  
**Goal:** Finalize and document the solution  
**Tool:** `finish(investigation_summary)`

#### Sub-steps:
**12.1 - Prepare Summary**
- **Actions:**
  1. Write clear summary: Problem, Investigation, Solution
  2. Document what changed and why
  3. Note important considerations or limitations
- **Format:**
  ```
  Problem: [Concise problem description]
  Investigation: [What you found, root cause]
  Solution: [What you changed, how it fixes the issue]
  ```

**12.2 - Call Finish**
- **Actions:**
  1. Call finish tool with investigation_summary
  2. Workflow generates final git patch
- **Rules:**
  - Only call finish after ALL validation passes
  - Summary should be professional and complete

**Output:** Completed task with git patch of changes

---

## 🎯 Workflow Navigation

### Decision Points

**After Step 0 (Meta-Planning):** → Proceed to Step 1

**After Step 5 (Reflection):**
- If revision_needed = true → Improve solution → Step 5 again
- If revision_needed = false → Step 6

**After Step 8 (Testing):**
- If all tests pass → Step 9
- If tests fail → Analyze → Step 7

**After Step 9 (Validation):**
- If validation_passed = true → Step 11
- If validation_passed = false → Step 10

**After Step 10 (Refinement):**
- → Step 8 (re-test)
- If iteration count = 3 and still failing → Reconsider approach (return to Step 4)

**After Step 11 (Final Verification):** → Step 12

---

## 📊 Success Metrics

Track these to measure workflow effectiveness:
- **Meta-Planning Usage:** % of complex problems using Step 0
- **Reflection Effectiveness:** % of issues caught by Step 5
- **First-Time Validation Pass:** % passing Step 9 on first attempt
- **Average Refinement Iterations:** Average loops through Step 10
- **Overall Success Rate:** % of tasks completed successfully

**Target Benchmarks:**
- Meta-planning for complex problems: 80%+
- Reflection effectiveness: 70%+
- First-time validation pass: 60%+
- Average refinement iterations: < 2
- Overall success rate: 90%+

---

## 🔍 Quick Reference

**When to use each enhanced tool:**
- **create_meta_plan:** Complex problems, unclear approaches
- **reflect_on_solution:** Before implementation, multiple options
- **validate_solution:** After implementation, before finish
- **refine_solution:** When validation fails, test failures

**Key Rules to Remember:**
1. Always think before calling tools (Critical Thinking Protocol)
2. Test-driven: create tests before fixing
3. Minimum 2 solution proposals required
4. Reflection before implementation catches 60-70% of issues
5. Validation score must be ≥ 85 before finish
6. Maximum 3 refinement iterations
7. Backward compatibility unless explicitly waived

---

*Workflow Structure v1.0*  
*For: v3.py Enhanced Agent*  
*Last Updated: 2025-10-20*

