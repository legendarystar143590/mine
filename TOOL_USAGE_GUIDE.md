# Enhanced Agent Tools - Quick Usage Guide

## 🎯 Overview
This guide provides practical examples of how to use the 4 new enhanced agent tools in `v3.py` to improve problem-solving effectiveness.

---

## 📋 Tool Summary Table

| Tool | When to Use | Primary Model | Returns |
|------|-------------|---------------|---------|
| `create_meta_plan` | Start of complex problems | DEEPSEEK | JSON plan |
| `reflect_on_solution` | After proposing, before implementing | DEEPSEEK | JSON critique |
| `validate_solution` | After implementation & testing | QWEN | JSON validation report |
| `refine_solution` | When tests/validation fail | QWEN | Improved code |

---

## 🚀 Tool 1: create_meta_plan

### When to Use
✅ **Use When:**
- Problem has multiple components/requirements
- Multiple approaches are viable
- Need to prevent wasted effort
- Problem is complex or unfamiliar

❌ **Skip When:**
- Problem is straightforward
- Single obvious solution
- Emergency hotfix
- Time-critical simple fix

### Usage Example

```python
# Example call
plan = create_meta_plan(
    problem_statement="Fix memory leak in cache system causing server crashes after 24h uptime",
    project_context="CacheManager class in cache.py, uses weak references, multi-threaded environment"
)

# Expected output structure:
{
  "problem_analysis": {
    "problem_type": "memory leak / performance issue",
    "requirements": ["Fix memory leak", "Preserve cache functionality", "Thread-safe"],
    "constraints": ["No downtime", "Backward compatible API"],
    "edge_cases": ["Concurrent access", "Cache overflow", "Rapid expiry"],
    "affected_components": ["cache.py::CacheManager", "tests/test_cache.py"]
  },
  "solution_decomposition": {
    "sub_tasks": [
      {
        "id": 1,
        "name": "Locate leak source",
        "description": "Profile memory and identify retention points",
        "dependencies": [],
        "estimated_complexity": "medium",
        "risks": ["May need specialized profiling tools"]
      },
      {
        "id": 2,
        "name": "Fix reference management",
        "description": "Ensure objects are properly released",
        "dependencies": [1],
        "estimated_complexity": "high",
        "risks": ["Threading issues", "Breaking existing functionality"]
      }
    ],
    "execution_order": [1, 2, 3, 4]
  },
  "approaches": [
    {
      "name": "Weak reference cleanup",
      "description": "Add explicit cleanup for weak refs",
      "pros": ["Minimal API changes", "Low risk"],
      "cons": ["May not catch all leaks"],
      "complexity": "low",
      "recommended": true
    }
  ],
  "verification_strategy": {
    "success_criteria": ["Memory stable over 48h", "All tests pass"],
    "test_categories": ["normal", "edge", "load"],
    "checkpoints": ["After each sub-task"],
    "rollback_plan": "Revert to git commit abc123"
  }
}
```

### Usage Tips
1. Include relevant project context (affected files, constraints)
2. Use the plan to guide your investigation and implementation
3. Reference the plan in your thinking
4. Update approach if plan reveals complexity

---

## 🔍 Tool 2: reflect_on_solution

### When to Use
✅ **Use When:**
- About to propose solution to user
- Before implementing complex changes
- Multiple solution options available
- Uncertain about correctness

❌ **Skip When:**
- Trivial changes (typo fixes)
- User explicitly approved approach
- Emergency situations
- Very confident in solution

### Usage Example

```python
# Example call
critique = reflect_on_solution(
    proposed_solution="""
def cache_get(self, key):
    if key in self._cache:
        return self._cache[key]
    return None
""",
    problem_statement="Fix memory leak in cache system",
    solution_description="Added explicit weak reference cleanup on get()"
)

# Expected output structure:
{
  "overall_assessment": "Solution addresses immediate leak but lacks thread safety",
  "confidence_score": 65,
  "issues_found": [
    {
      "dimension": "robustness",
      "severity": "HIGH",
      "description": "No thread synchronization in multi-threaded environment",
      "example": "Two threads calling get() simultaneously could cause race condition",
      "suggested_fix": "Add threading.Lock around cache access",
      "priority": 9
    },
    {
      "dimension": "completeness",
      "severity": "MEDIUM",
      "description": "Missing null key handling",
      "example": "get(None) would raise KeyError",
      "suggested_fix": "Add if key is None: return None at start",
      "priority": 6
    }
  ],
  "strengths": [
    "Clear and simple logic",
    "Addresses core memory leak issue"
  ],
  "improvement_recommendations": [
    {
      "recommendation": "Add thread synchronization with Lock",
      "impact": "HIGH",
      "effort": "LOW",
      "implementation": "Use threading.Lock() context manager"
    }
  ],
  "should_proceed": false,
  "revision_needed": true
}
```

### Usage Tips
1. **Always check `revision_needed` field** - if true, improve before implementing
2. **Prioritize by severity**: Fix CRITICAL and HIGH issues immediately
3. **Review strengths** to ensure you preserve them in revisions
4. **Use suggested fixes** to guide improvements
5. If confidence_score < 70, seriously consider revising

---

## ✅ Tool 3: validate_solution

### When to Use
✅ **Use When:**
- Implementation complete and tests run
- Before calling finish()
- Need comprehensive quality check
- Preparing for production

❌ **Skip When:**
- Still developing/iterating
- Tests haven't been run yet
- Experimental prototype code
- During emergency hotfixes

### Usage Example

```python
# Example call
validation_report = validate_solution(
    solution_code="""
class CacheManager:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get(self, key):
        if key is None:
            return None
        with self._lock:
            return self._cache.get(key)
""",
    test_results="""
Ran 25 tests in 2.3s
PASSED: 23
FAILED: 2
- test_concurrent_access: Race condition detected
- test_cache_cleanup: Memory not released
""",
    problem_statement="Fix memory leak in cache system",
    requirements_checklist="1. Fix leak 2. Thread-safe 3. Backward compatible"
)

# Expected output structure:
{
  "validation_passed": false,
  "overall_score": 78,
  "category_scores": {
    "functional_correctness": 85,
    "test_coverage": 75,
    "code_quality": 90,
    "performance": 80,
    "compatibility": 95
  },
  "requirements_status": [
    {
      "requirement": "Fix memory leak",
      "status": "PARTIAL",
      "evidence": "Improved but test_cache_cleanup still fails"
    },
    {
      "requirement": "Thread-safe",
      "status": "FAIL",
      "evidence": "test_concurrent_access fails with race condition"
    },
    {
      "requirement": "Backward compatible",
      "status": "PASS",
      "evidence": "All API tests pass"
    }
  ],
  "test_results": {
    "total_tests": 25,
    "passed": 23,
    "failed": 2,
    "failure_analysis": "Threading issues remain despite Lock usage"
  },
  "blocking_issues": [
    {
      "issue": "Race condition in concurrent access",
      "severity": "CRITICAL",
      "must_fix": true
    }
  ],
  "recommendations": [
    "Review Lock placement - may need to cover entire method",
    "Add explicit cleanup method for memory release"
  ],
  "certification": {
    "ready_for_production": false,
    "confidence_level": "MEDIUM",
    "validation_notes": "Close but needs threading fixes"
  }
}
```

### Usage Tips
1. **Check `validation_passed` first** - must be true before finishing
2. **Overall score must be >= 85** to pass (configurable)
3. **No CRITICAL blocking_issues** allowed
4. **All category scores should be >= 80**
5. If validation fails, use `refine_solution` to improve
6. **Maximum 3 refinement attempts** - if still failing, reconsider approach

---

## 🔧 Tool 4: refine_solution

### When to Use
✅ **Use When:**
- Validation score < 85
- Test failures present
- CRITICAL or HIGH severity issues found
- Reflection identified problems

❌ **Skip When:**
- All tests passing and validation passes
- Issues are cosmetic only (LOW severity)
- Already at 3rd refinement iteration
- Complete rewrite is needed

### Usage Example

```python
# Example call
improved_solution = refine_solution(
    current_solution="""
def get(self, key):
    if key is None:
        return None
    with self._lock:
        return self._cache.get(key)
""",
    feedback="""
CRITICAL: Race condition in concurrent access
HIGH: Memory not released in cleanup
Evidence: test_concurrent_access fails, test_cache_cleanup fails
""",
    problem_statement="Fix memory leak in cache system",
    test_failures="""
test_concurrent_access: AssertionError - got different values from same key
test_cache_cleanup: Memory usage increased by 50MB after cleanup
"""
)

# Expected output:
"""
def get(self, key):
    '''Get value from cache with proper thread safety and memory management'''
    if key is None:
        return None
    
    # FIX: Acquire lock before any cache access to prevent race conditions
    with self._lock:
        value = self._cache.get(key)
        
        # FIX: Check if weak reference needs cleanup
        if hasattr(value, '__weakref__') and value is not None:
            # Ensure strong reference maintained during return
            return value
            
        return value

def cleanup(self):
    '''Explicit cleanup method for memory release'''
    # FIX: Added explicit cleanup to address memory leak
    with self._lock:
        # Clear cache and trigger garbage collection
        self._cache.clear()
        gc.collect()
"""
```

### Usage Tips
1. **Include severity in feedback** - helps prioritize fixes
2. **Provide specific test failures** - guides refinement
3. **Track iteration count** - max 3 iterations before reconsidering
4. **Review changes** - ensure working functionality preserved
5. **Re-test after refinement** - verify fixes work
6. **Re-validate after refinement** - check if score improved

---

## 📊 Complete Workflow Example

### Scenario: Complex Bug Fix with Memory Leak

```python
# STEP 1: Create Meta-Plan (Optional but recommended)
plan = create_meta_plan(
    problem_statement="Memory leak in cache causing crashes",
    project_context="Multi-threaded cache system in cache.py"
)
# Review plan, identify sub-tasks, select approach

# STEP 2: Investigate & Localize
# ... use search tools, get_file_content, etc ...

# STEP 3: Create Test Script
# ... use generate_test_function ...

# STEP 4: Propose Solutions
solution_a = "Add weak reference cleanup"
solution_b = "Redesign cache with explicit lifecycle"

# STEP 5: Reflect on Each Solution
critique_a = reflect_on_solution(
    proposed_solution=solution_a,
    problem_statement="Memory leak in cache",
    solution_description="Minimal change using weak refs"
)

critique_b = reflect_on_solution(
    proposed_solution=solution_b,
    problem_statement="Memory leak in cache",
    solution_description="Complete redesign for better control"
)

# Review critiques, choose best approach (e.g., solution_a with improvements)

# STEP 6: Get Approval
# ... use get_approval_for_solution ...

# STEP 7: Implement Solution
# ... use apply_code_edit or save_file ...

# STEP 8: Run Tests
test_results = run_repo_tests(["tests/test_cache.py"])

# STEP 9: Validate Solution
validation = validate_solution(
    solution_code=implemented_code,
    test_results=test_results,
    problem_statement="Memory leak in cache"
)

# Check if validation passed
if validation["validation_passed"] == false:
    # STEP 10: Refine Solution (max 3 iterations)
    for iteration in range(3):
        improved_code = refine_solution(
            current_solution=implemented_code,
            feedback=validation["blocking_issues"],
            problem_statement="Memory leak in cache",
            test_failures=test_results
        )
        
        # Re-implement improved solution
        # ... apply_code_edit with improved_code ...
        
        # Re-test
        test_results = run_repo_tests(["tests/test_cache.py"])
        
        # Re-validate
        validation = validate_solution(
            solution_code=improved_code,
            test_results=test_results,
            problem_statement="Memory leak in cache"
        )
        
        if validation["validation_passed"]:
            break
    
    if not validation["validation_passed"]:
        # Consider starting over with different approach
        pass

# STEP 11: Finish
finish(investigation_summary="...")
```

---

## 💡 Best Practices

### 1. Tool Combination Patterns

**Pattern A: Strategic Planning Flow**
```
create_meta_plan → investigate → reflect_on_solution → implement → validate_solution
```
*Best for: Complex, unfamiliar problems*

**Pattern B: Quick Iteration Flow**
```
investigate → implement → validate_solution → refine_solution
```
*Best for: Familiar problems, clear approach*

**Pattern C: High-Quality Flow**
```
create_meta_plan → reflect_on_solution → implement → validate_solution → refine_solution
```
*Best for: Production-critical fixes*

### 2. Decision Trees

**Should I use create_meta_plan?**
```
Is problem complex? YES → Use meta-plan
                   NO  → Is approach unclear? YES → Use meta-plan
                                              NO  → Skip
```

**Should I use reflect_on_solution?**
```
Is solution complex? YES → Use reflection
                     NO  → Am I uncertain? YES → Use reflection
                                           NO  → Is it high-stakes? YES → Use reflection
                                                                     NO  → Skip
```

**Should I use validate_solution?**
```
Tests run? YES → Use validation
           NO  → Run tests first, then validate
```

**Should I use refine_solution?**
```
Validation passed? NO  → Use refinement (max 3 times)
                   YES → Done!
```

### 3. Error Handling

All tools gracefully handle errors and return helpful messages:
- **create_meta_plan**: "Note: Meta-planning is optional. You can proceed without it."
- **reflect_on_solution**: "Note: Reflection is recommended but optional. You can proceed with caution."
- **validate_solution**: "Note: Proceed with manual validation."
- **refine_solution**: "Note: Manual refinement needed."

If a tool fails, **you can still proceed** - tools are helpers, not blockers.

### 4. Performance Considerations

- **create_meta_plan**: ~15-30 seconds (DEEPSEEK model)
- **reflect_on_solution**: ~20-40 seconds (DEEPSEEK model)
- **validate_solution**: ~15-25 seconds (QWEN model)
- **refine_solution**: ~10-20 seconds (QWEN model)

**Budget**: Reserve ~2-3 minutes for full meta-cognitive cycle

---

## 📈 Success Metrics

### Track Your Usage
- **Planning Success Rate**: Did meta-plan lead to successful solution?
- **Reflection Effectiveness**: What % of issues caught by reflection vs later?
- **Validation Pass Rate**: First-time validation pass percentage?
- **Refinement Iterations**: Average iterations needed?

### Target Benchmarks
- Meta-plan usage for complex problems: 80%+
- Reflection before complex implementation: 90%+
- Validation before finish: 100%
- Average refinement iterations: < 2

---

## 🎓 Key Takeaways

1. **Think Before Acting**: Use `create_meta_plan` for complex problems
2. **Critique Before Implementing**: Use `reflect_on_solution` to catch issues early
3. **Validate Before Finishing**: Use `validate_solution` to ensure quality
4. **Improve Systematically**: Use `refine_solution` when validation fails

**Remember**: These tools are **force multipliers**, not requirements. Use them strategically to enhance your problem-solving effectiveness!

---

*Quick Reference Guide v1.0*  
*For: v3.py Enhanced Agent*  
*Last Updated: 2025-10-20*

