# Test Result Parser Documentation

**Created:** 2025-10-23  
**Function:** `Utils.parse_test_results()`  
**Location:** `22/v4.py` lines 1088-1548

---

## 📋 Overview

The `parse_test_results()` function is a comprehensive test output parser that automatically detects the testing framework and extracts failed test cases with their error messages. This solves the problem of needing to parse different test output formats across various programming languages and frameworks.

---

## 🎯 Purpose

When the agent needs to validate fixes with Pass-to-Pass (P2P) testing, it must:
1. Run the existing test suite
2. Identify which tests failed
3. Get detailed error messages for failed tests
4. Analyze failures to determine if they're expected (Case A) or problematic (Case B)

Since different projects use different testing frameworks with different output formats, the agent cannot hardcode a single parsing strategy. This function solves that problem by:
- **Auto-detecting** the framework from the test command
- **Parsing** the output in a framework-specific way
- **Extracting** only the failed tests with details
- **Returning** structured data in a consistent format

---

## 🔧 Function Signature

```python
@staticmethod
def parse_test_results(test_command: str, test_output: str) -> Dict[str, Any]:
    """
    Parse test results from various testing frameworks and extract failed test cases.
    
    Args:
        test_command: The command used to run tests (e.g., "pytest", "python -m unittest")
        test_output: The output from running the tests
        
    Returns:
        Dictionary containing:
        - framework: Detected testing framework
        - total_tests: Total number of tests run
        - passed: Number of passed tests
        - failed: Number of failed tests
        - errors: Number of errors
        - skipped: Number of skipped tests
        - failed_tests: List of failed test details with names and error messages
    """
```

---

## 📊 Return Value Structure

```python
{
    "framework": "pytest",           # Detected framework
    "total_tests": 15,                # Total tests run
    "passed": 12,                     # Tests that passed
    "failed": 2,                      # Tests that failed
    "errors": 1,                      # Tests with errors
    "skipped": 0,                     # Tests skipped
    "failed_tests": [                 # Detailed failure info
        {
            "name": "test_file.py::test_function_name",
            "error": "AssertionError: Expected 5 but got 3\n  File 'test.py', line 42..."
        },
        {
            "name": "test_file.py::test_another_function",
            "error": "ValueError: Invalid input..."
        }
    ]
}
```

---

## 🌐 Supported Testing Frameworks

### **Python**
1. **pytest**
   - Detection: `"pytest"` or `"py.test"` in command
   - Parses: `FAILED test_file.py::test_name` format
   - Extracts: Full error tracebacks

2. **unittest**
   - Detection: `"unittest"` or `"python -m unittest"` in command
   - Parses: `FAIL: test_name (module.TestClass)` format
   - Extracts: Failure details and error messages

### **JavaScript**
3. **Jest**
   - Detection: `"jest"`, `"npm test"`, or `"yarn test"` in command
   - Parses: `● Test suite › test name` format
   - Extracts: Test descriptions and error details

4. **Mocha**
   - Detection: `"mocha"` in command
   - Parses: Numbered failures `1) test name` format
   - Extracts: Test names and error stacks

### **Go**
5. **go test**
   - Detection: `"go test"` in command
   - Parses: `--- FAIL: TestName` format
   - Extracts: Test function names and output

### **Rust**
6. **cargo test**
   - Detection: `"cargo test"` or `"rust"` in command
   - Parses: `test test_name ... FAILED` format
   - Extracts: Test names

### **Generic**
7. **Fallback Parser**
   - Used when framework cannot be determined
   - Searches for common keywords: "failed", "error", "passed"
   - Provides best-effort parsing

---

## 💡 Usage Examples

### **Example 1: Pytest**

```python
test_command = "pytest tests/test_math.py -v"
test_output = """
================================ test session starts ================================
collected 5 items

tests/test_math.py::test_add PASSED                                          [ 20%]
tests/test_math.py::test_subtract PASSED                                     [ 40%]
tests/test_math.py::test_multiply FAILED                                     [ 60%]
tests/test_math.py::test_divide PASSED                                       [ 80%]
tests/test_math.py::test_power PASSED                                        [100%]

===================================== FAILURES ======================================
___________________________________ test_multiply ___________________________________

    def test_multiply():
>       assert multiply(3, 4) == 11
E       AssertionError: assert 12 == 11

tests/test_math.py:15: AssertionError
============================ 2 passed, 1 failed in 0.50s ============================
"""

result = Utils.parse_test_results(test_command, test_output)

print(result)
# Output:
# {
#     "framework": "pytest",
#     "total_tests": 5,
#     "passed": 4,
#     "failed": 1,
#     "errors": 0,
#     "skipped": 0,
#     "failed_tests": [
#         {
#             "name": "tests/test_math.py::test_multiply",
#             "error": "    def test_multiply:\n>       assert multiply(3, 4) == 11\n..."
#         }
#     ]
# }
```

### **Example 2: Unittest**

```python
test_command = "python -m unittest discover"
test_output = """
..F.E
======================================================================
FAIL: test_division (test_calculator.TestCalculator)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_calculator.py", line 25, in test_division
    self.assertEqual(divide(10, 2), 6)
AssertionError: 5 != 6

======================================================================
ERROR: test_sqrt (test_calculator.TestCalculator)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_calculator.py", line 30, in test_sqrt
    sqrt(-1)
ValueError: math domain error

----------------------------------------------------------------------
Ran 5 tests in 0.002s

FAILED (failures=1, errors=1)
"""

result = Utils.parse_test_results(test_command, test_output)

print(f"Failed tests: {result['failed']}")
print(f"Errors: {result['errors']}")
for failed in result['failed_tests']:
    print(f"  - {failed['name']}")
```

### **Example 3: Jest**

```python
test_command = "npm test"
test_output = """
 FAIL  src/utils.test.js
  ● Utils › parseDate › handles invalid dates

    expect(received).toBe(expected) // Object.is equality

    Expected: null
    Received: undefined

      12 |   it('handles invalid dates', () => {
      13 |     const result = parseDate('invalid');
    > 14 |     expect(result).toBe(null);
         |                    ^
      15 |   });

Tests:       1 failed, 3 passed, 4 total
Snapshots:   0 total
Time:        1.234 s
"""

result = Utils.parse_test_results(test_command, test_output)

# Access failed test details
for failed in result['failed_tests']:
    print(f"Test: {failed['name']}")
    print(f"Error: {failed['error'][:100]}...")
```

---

## 🔍 How Framework Detection Works

### **1. Command-Based Detection (Primary)**
The function first examines the test command:

```python
if "pytest" in command_lower or "py.test" in command_lower:
    result["framework"] = "pytest"
    return Utils._parse_pytest_output(test_output, result)
```

### **2. Output-Based Detection (Fallback)**
If command detection fails, it examines the output:

```python
if "pytest" in test_output or "test session starts" in test_output.lower():
    result["framework"] = "pytest"
    return Utils._parse_pytest_output(test_output, result)
```

### **3. Generic Parser (Last Resort)**
If no framework is detected, uses a generic parser that looks for common keywords.

---

## 🎨 Internal Parser Functions

Each framework has its own specialized parser:

| Framework | Parser Function | Key Patterns |
|-----------|----------------|--------------|
| pytest | `_parse_pytest_output()` | `FAILED test.py::name` |
| unittest | `_parse_unittest_output()` | `FAIL: test_name (class)` |
| Jest | `_parse_jest_output()` | `● Test suite › name` |
| Mocha | `_parse_mocha_output()` | `1) test name` |
| Go test | `_parse_go_test_output()` | `--- FAIL: TestName` |
| Cargo test | `_parse_cargo_test_output()` | `test name ... FAILED` |
| Generic | `_parse_generic_output()` | Keywords: fail, error, pass |

---

## 📝 Use Cases in Agent Workflow

### **Use Case 1: Pass-to-Pass (P2P) Validation**

In Step 8 of the BugFixSolver workflow:

```python
# Step 8.3: Run existing test suite
bash_tool = tool_manager.get_tool("bash")
test_result = bash_tool.run({"command": "pytest tests/"})

# Parse the results
parsed = Utils.parse_test_results("pytest tests/", test_result.tool_output)

if parsed["failed"] > 0:
    logger.info(f"Found {parsed['failed']} failed tests")
    
    # Analyze each failure
    for failed_test in parsed["failed_tests"]:
        logger.info(f"Failed: {failed_test['name']}")
        logger.info(f"Error: {failed_test['error']}")
        
        # Determine Case A vs Case B
        # Case A: Test expects old (buggy) behavior → Ignore
        # Case B: Test expects correct behavior → Fix is wrong
```

### **Use Case 2: Quick Failure Summary**

```python
# Run tests
output = run_command("pytest")

# Get quick summary
result = Utils.parse_test_results("pytest", output)

logger.info(f"""
Test Summary:
  Framework: {result['framework']}
  Total: {result['total_tests']}
  Passed: {result['passed']} ✓
  Failed: {result['failed']} ✗
  Errors: {result['errors']} ⚠
  Skipped: {result['skipped']} ⊘
""")

if result['failed'] > 0:
    logger.info("Failed tests:")
    for test in result['failed_tests']:
        logger.info(f"  - {test['name']}")
```

### **Use Case 3: Re-run Only Failed Tests**

```python
# First run: Get all failures
full_run = Utils.parse_test_results("pytest", run_command("pytest"))

if full_run["failed"] > 0:
    # Second run: Get detailed info on failures only
    failed_names = [t["name"] for t in full_run["failed_tests"]]
    
    detailed_results = []
    for test_name in failed_names:
        # Run with verbose output
        detailed_output = run_command(f"pytest {test_name} -vv")
        detailed = Utils.parse_test_results(f"pytest {test_name} -vv", detailed_output)
        detailed_results.append(detailed)
```

---

## ⚠️ Important Notes

### **1. Framework-Specific Quirks**

**pytest:**
- Uses `::` to separate file and test name: `file.py::test_name`
- Can have nested test names: `file.py::TestClass::test_method`
- Error messages include full tracebacks

**unittest:**
- Format: `test_name (module.ClassName)`
- Distinguishes between FAIL and ERROR
- Less structured output

**Jest:**
- Uses `●` bullet points for failures
- Shows file path and test description
- Pretty-printed error diffs

### **2. Edge Cases Handled**

- **Empty output**: Returns all zeros
- **Partial output**: Best-effort parsing
- **Mixed frameworks**: First detected framework wins
- **No test markers**: Falls back to generic parser
- **Truncated output**: Parses what's available

### **3. Limitations**

- Cannot parse binary or non-text output
- Assumes standard output format (may break with custom formatters)
- Generic parser may have false positives/negatives
- Error messages may be truncated if very long

---

## 🚀 Benefits

1. **Language Agnostic**: Works with Python, JavaScript, Go, Rust, and more
2. **Framework Agnostic**: Supports 6+ testing frameworks out of the box
3. **Automatic Detection**: No need to specify the framework manually
4. **Structured Output**: Consistent data structure regardless of framework
5. **Detailed Failures**: Captures both test names and error messages
6. **Extensible**: Easy to add new framework parsers
7. **Robust**: Falls back to generic parsing if framework unknown

---

## 🔧 Extending the Parser

To add support for a new testing framework:

### **Step 1: Add Detection Logic**

```python
elif "rspec" in command_lower:
    result["framework"] = "rspec"
    return Utils._parse_rspec_output(test_output, result)
```

### **Step 2: Create Parser Function**

```python
@staticmethod
def _parse_rspec_output(output: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse RSpec output format."""
    lines = output.split('\n')
    
    # Parse summary line
    for line in lines:
        if " examples, " in line:
            # RSpec format: "5 examples, 2 failures"
            match = re.search(r'(\d+) examples, (\d+) failures', line)
            if match:
                result["total_tests"] = int(match.group(1))
                result["failed"] = int(match.group(2))
                result["passed"] = result["total_tests"] - result["failed"]
    
    # Extract failed test details
    # ... (framework-specific parsing logic)
    
    return result
```

---

## 📊 Example Integration

Here's how to integrate this into a tool:

```python
class TestAnalysisTool(LLMTool):
    """Analyze test results and provide failure summary."""
    name = "analyze_tests"
    
    input_schema = {
        "type": "object",
        "properties": {
            "test_command": {"type": "string"},
            "test_output": {"type": "string"}
        },
        "required": ["test_command", "test_output"]
    }
    
    def run_impl(self, tool_input: dict[str, Any]) -> Types.ToolImplOutput:
        cmd = tool_input["test_command"]
        output = tool_input["test_output"]
        
        # Parse results
        result = Utils.parse_test_results(cmd, output)
        
        # Format summary
        summary = f"""
Test Analysis ({result['framework']}):
  Total: {result['total_tests']}
  ✓ Passed: {result['passed']}
  ✗ Failed: {result['failed']}
  ⚠ Errors: {result['errors']}
  ⊘ Skipped: {result['skipped']}
"""
        
        if result['failed'] > 0:
            summary += "\nFailed Tests:\n"
            for test in result['failed_tests'][:10]:  # Show first 10
                summary += f"  • {test['name']}\n"
                # Show first line of error
                error_line = test['error'].split('\n')[0]
                summary += f"    {error_line}\n"
        
        return Types.ToolImplOutput(
            summary,
            f"Analyzed {result['total_tests']} tests",
            result
        )
```

---

## ✅ Testing the Parser

You can test the parser with sample outputs:

```python
# Test with pytest sample
pytest_output = """
============================= test session starts ==============================
collected 3 items

test_sample.py::test_one PASSED                                          [ 33%]
test_sample.py::test_two FAILED                                          [ 66%]
test_sample.py::test_three PASSED                                        [100%]

=================================== FAILURES ===================================
_________________________________ test_two _____________________________________

    def test_two():
>       assert 1 == 2
E       assert 1 == 2

test_sample.py:5: AssertionError
=========================== 1 failed, 2 passed in 0.12s ========================
"""

result = Utils.parse_test_results("pytest", pytest_output)
assert result["framework"] == "pytest"
assert result["passed"] == 2
assert result["failed"] == 1
assert len(result["failed_tests"]) == 1
assert "test_two" in result["failed_tests"][0]["name"]
```

---

## 📌 Summary

The `Utils.parse_test_results()` function is a powerful, flexible test output parser that:
- ✅ **Automatically detects** testing frameworks
- ✅ **Extracts structured data** from test output
- ✅ **Filters only failed tests** with error details
- ✅ **Supports 6+ frameworks** out of the box
- ✅ **Provides consistent interface** across all frameworks
- ✅ **Enables intelligent P2P validation** in the agent workflow

This makes it easy for the agent to understand test results regardless of the project's testing setup!

---

*For questions or to add support for additional frameworks, refer to the implementation in `22/v4.py` lines 1088-1548.*

