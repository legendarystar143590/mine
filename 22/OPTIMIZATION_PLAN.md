# Code Optimization Plan - DRY & YAGNI Principles

## 1. Move All Prompts to PromptManager (DRY Violation)
**Issue**: 7 prompts defined globally instead of in PromptManager class
- GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT (line 46)
- INFINITE_LOOP_CHECK_PROMPT (line 73)
- GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT (line 106)
- GENERATE_INITIAL_SOLUTION_PROMPT (line 132)
- TESTCASES_CHECK_PROMPT (line 157)
- GENERATE_INITIAL_TESTCASES_PROMPT (line 183)
- PROBLEM_TYPE_CHECK_PROMPT (line 208)

**Solution**: Move all to PromptManager class as class attributes

## 2. Remove Duplicate BashTool Class (YAGNI Violation)
**Issue**: Two BashTool implementations exist:
- EnhancedBashTool (lines 2886-3247) - has enhanced error reporting
- BashTool (lines 3253-3415) - basic version, appears unused

**Solution**: Remove the basic BashTool class entirely, keep only EnhancedBashTool

## 3. Create Utility for Response Cleaning (DRY Violation)
**Issue**: Response cleaning logic repeated 4+ times:
```python
if solution.startswith('```python'):
    solution = solution[9:]
if solution.startswith('```'):
    solution = solution[3:]
if solution.endswith('```'):
    solution = solution[:-3]
solution = solution.strip()
```

**Solution**: Create `Utils.clean_code_response(text: str) -> str` method

## 4. Consolidate Retry Logic (DRY Violation)
**Issue**: Retry loops with similar structure in multiple functions:
- generate_solution_with_multi_step_reasoning()
- generate_initial_solution()
- generate_testcases_with_multi_step_reasoning()
- generate_test_files()
- check_problem_type()

**Solution**: Create `EnhancedNetwork.make_request_with_retry()` wrapper

## 5. Consolidate Error Analysis Methods (DRY Violation)
**Issue**: Multiple similar error analysis methods in TestValidationTool:
- _analyze_baseline_failures()
- _analyze_baseline_error()
- _analyze_f2p_failure()
- _analyze_f2p_error()
- _analyze_p2p_failures()
- _analyze_p2p_error()

**Solution**: Create generic `_format_error_analysis(error_type, error_msg, recommendations)` method

## 6. Remove Unused Methods/Code (YAGNI)
**Issue**: Potentially unused code:
- post_process_instruction() function - seems specific, may not be needed
- Multiple constants that might not be used

**Solution**: Verify usage and remove if not needed

## 7. Simplify File Writing Logic (DRY)
**Issue**: extract_and_write_files() has complex logic that could be simplified

**Solution**: Add validation and make more robust

## 8. Consolidate Message Creation (DRY)
**Issue**: Similar message dict creation repeated:
```python
{"role": "system", "content": PROMPT}
{"role": "user", "content": f"..."}
```

**Solution**: Create helper methods in PromptManager

## 9. Remove Duplicate/Redundant Tool Functionality
**Issue**: Some tool methods have overlapping functionality

**Solution**: Consolidate where possible

## 10. Simplify Agent Logic
**Issue**: agent_main() has try-except that falls back to same function

**Solution**: Simplify the logic flow

## Implementation Priority:
1. Move prompts to PromptManager (High impact, easy)
2. Remove duplicate BashTool (High impact, easy)
3. Create response cleaning utility (Medium impact, easy)
4. Consolidate retry logic (Medium impact, medium difficulty)
5. Consolidate error analysis (Low impact, medium difficulty)
6. Other optimizations (Low impact, varies)

