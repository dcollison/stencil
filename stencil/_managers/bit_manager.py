from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Dict, List, Any
from enum import Enum

from stencil._managers.i_manager import IManager
from stencil._utilities import timestamp

if TYPE_CHECKING:
    from stencil import Service


class TestResult(Enum):
    """Enumeration of possible test results."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class TestType(Enum):
    """Types of Built-In Tests."""
    PBIT = "PBIT"  # Power-On Built-In Test
    CBIT = "CBIT"  # Continuous Built-In Test


@dataclass
class TestInfo:
    """Information about a registered test."""
    method: Callable[[], bool]
    name: str
    test_type: TestType
    description: Optional[str] = None
    category: Optional[str] = None
    timeout_ms: Optional[int] = None
    enabled: bool = True
    execution_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    error_count: int = 0
    total_execution_time: int = 0
    last_result: Optional[TestResult] = None
    last_execution_time: int = 0
    last_error: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.execution_count == 0:
            return 0.0
        return (self.pass_count / self.execution_count) * 100
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time in milliseconds."""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count


@dataclass
class TestExecutionResult:
    """Result of a single test execution."""
    name: str
    result: TestResult
    execution_time: int
    error_message: Optional[str] = None
    timestamp: int = field(default_factory=timestamp)


@dataclass
class TestSuiteResult:
    """Result of running a complete test suite."""
    test_type: TestType
    overall_result: TestResult
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total_execution_time: int
    individual_results: List[TestExecutionResult] = field(default_factory=list)
    timestamp: int = field(default_factory=timestamp)
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


TestMethod = Callable[[], bool]
ResultCallback = Callable[[TestSuiteResult], None]


class BitManager(IManager):
    """
    Manages Built-In Tests (BITs) including Power-On BIT (PBIT) and Continuous BIT (CBIT).
    
    Provides comprehensive test execution, result collection, and reporting capabilities.
    Tests can be categorized, enabled/disabled, and have configurable timeouts.
    """
    
    def __init__(self, service: "Service", default_timeout_ms: int = 30000):
        """
        Initialize the BitManager.
        
        :param service: The parent service object that owns this manager.
        :param default_timeout_ms: Default timeout for test execution in milliseconds.
        """
        super().__init__(service)
        
        self.default_timeout_ms = default_timeout_ms
        
        # Test storage - using dicts for O(1) lookup by name
        self._pbits: Dict[str, TestInfo] = {}
        self._cbits: Dict[str, TestInfo] = {}
        
        # Result callbacks for external communication
        self._result_callbacks: List[ResultCallback] = []
        
        # Execution history (keep last N results)
        self._max_history_size = 100
        self._pbit_history: List[TestSuiteResult] = []
        self._cbit_history: List[TestSuiteResult] = []

    def register_pbit(
        self, 
        method: TestMethod, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        enabled: bool = True
    ) -> None:
        """
        Register a Power-On Built-In Test.
        
        :param method: The test method that returns bool (True = pass, False = fail).
        :param name: Optional custom name for the test. Defaults to method.__name__.
        :param description: Optional description of what the test does.
        :param category: Optional category for grouping tests.
        :param timeout_ms: Optional timeout in milliseconds. Defaults to default_timeout_ms.
        :param enabled: Whether the test is enabled by default.
        :raises ValueError: If test name already exists or method signature is invalid.
        """
        self._register_test(method, TestType.PBIT, name, description, category, timeout_ms, enabled)

    def register_cbit(
        self, 
        method: TestMethod, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        enabled: bool = True
    ) -> None:
        """
        Register a Continuous Built-In Test.
        
        :param method: The test method that returns bool (True = pass, False = fail).
        :param name: Optional custom name for the test. Defaults to method.__name__.
        :param description: Optional description of what the test does.
        :param category: Optional category for grouping tests.
        :param timeout_ms: Optional timeout in milliseconds. Defaults to default_timeout_ms.
        :param enabled: Whether the test is enabled by default.
        :raises ValueError: If test name already exists or method signature is invalid.
        """
        self._register_test(method, TestType.CBIT, name, description, category, timeout_ms, enabled)

    def _register_test(
        self,
        method: TestMethod,
        test_type: TestType,
        name: Optional[str],
        description: Optional[str],
        category: Optional[str],
        timeout_ms: Optional[int],
        enabled: bool
    ) -> None:
        """Internal method to register a test."""
        test_name = name or method.__name__
        
        # Validate method signature
        sig = inspect.signature(method)
        if sig.return_annotation not in (bool, inspect.Signature.empty):
            self._logger.warning(
                f"{self._log_area}: Test {test_name} should return bool, "
                f"found annotation: {sig.return_annotation}"
            )
        
        # Check for existing test
        test_dict = self._pbits if test_type == TestType.PBIT else self._cbits
        if test_name in test_dict:
            raise ValueError(f"{test_type.value} '{test_name}' is already registered")
        
        # Create test info
        test_info = TestInfo(
            method=method,
            name=test_name,
            test_type=test_type,
            description=description,
            category=category,
            timeout_ms=timeout_ms or self.default_timeout_ms,
            enabled=enabled
        )
        
        test_dict[test_name] = test_info
        
        self._logger.debug(
            f"{self._log_area}: Registered {test_type.value} - {test_name}"
            f"{f' ({description})' if description else ''}"
            f"{f' [Category: {category}]' if category else ''}"
        )

    def add_result_callback(self, callback: ResultCallback) -> None:
        """
        Add a callback to be notified of test suite results.
        
        :param callback: Function that takes TestSuiteResult as parameter.
        """
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: ResultCallback) -> None:
        """Remove a result callback."""
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def enable_test(self, name: str, test_type: Optional[TestType] = None) -> bool:
        """
        Enable a test by name.
        
        :param name: Name of the test to enable.
        :param test_type: Optional test type to search in. If None, searches both.
        :return: True if test was found and enabled, False otherwise.
        """
        return self._set_test_enabled(name, True, test_type)

    def disable_test(self, name: str, test_type: Optional[TestType] = None) -> bool:
        """
        Disable a test by name.
        
        :param name: Name of the test to disable.
        :param test_type: Optional test type to search in. If None, searches both.
        :return: True if test was found and disabled, False otherwise.
        """
        return self._set_test_enabled(name, False, test_type)

    def _set_test_enabled(self, name: str, enabled: bool, test_type: Optional[TestType]) -> bool:
        """Internal method to enable/disable tests."""
        if test_type == TestType.PBIT or test_type is None:
            if name in self._pbits:
                self._pbits[name].enabled = enabled
                return True
        
        if test_type == TestType.CBIT or test_type is None:
            if name in self._cbits:
                self._cbits[name].enabled = enabled
                return True
        
        return False

    @property
    def n_pbits(self) -> int:
        """Number of registered PBITs."""
        return len(self._pbits)

    @property
    def n_cbits(self) -> int:
        """Number of registered CBITs.""" 
        return len(self._cbits)

    @property
    def n_enabled_pbits(self) -> int:
        """Number of enabled PBITs."""
        return sum(1 for test in self._pbits.values() if test.enabled)

    @property
    def n_enabled_cbits(self) -> int:
        """Number of enabled CBITs."""
        return sum(1 for test in self._cbits.values() if test.enabled)

    def get_test_info(self, name: str, test_type: Optional[TestType] = None) -> Optional[TestInfo]:
        """
        Get information about a specific test.
        
        :param name: Name of the test.
        :param test_type: Optional test type to search in. If None, searches both.
        :return: TestInfo object or None if not found.
        """
        if test_type == TestType.PBIT or test_type is None:
            if name in self._pbits:
                return self._pbits[name]
        
        if test_type == TestType.CBIT or test_type is None:
            if name in self._cbits:
                return self._cbits[name]
        
        return None

    def get_test_summary(self, test_type: Optional[TestType] = None) -> Dict[str, Any]:
        """
        Get a summary of test statistics.
        
        :param test_type: Optional test type filter. If None, includes both.
        :return: Dictionary containing test statistics.
        """
        tests = []
        if test_type == TestType.PBIT or test_type is None:
            tests.extend(self._pbits.values())
        if test_type == TestType.CBIT or test_type is None:
            tests.extend(self._cbits.values())
        
        if not tests:
            return {}
        
        total_tests = len(tests)
        enabled_tests = sum(1 for t in tests if t.enabled)
        total_executions = sum(t.execution_count for t in tests)
        total_passes = sum(t.pass_count for t in tests)
        
        return {
            'total_tests': total_tests,
            'enabled_tests': enabled_tests,
            'disabled_tests': total_tests - enabled_tests,
            'total_executions': total_executions,
            'overall_success_rate': (total_passes / total_executions * 100) if total_executions > 0 else 0.0,
            'categories': list(set(t.category for t in tests if t.category)),
            'avg_execution_time': sum(t.average_execution_time for t in tests) / total_tests if total_tests > 0 else 0.0
        }

    def run_pbits(self, category: Optional[str] = None) -> TestSuiteResult:
        """
        Run all enabled Power-On Built-In Tests.
        
        :param category: Optional category filter. If specified, only runs tests in that category.
        :return: TestSuiteResult containing execution results.
        """
        result = self._run_tests(self._pbits, TestType.PBIT, category)
        self._pbit_history.append(result)
        if len(self._pbit_history) > self._max_history_size:
            self._pbit_history.pop(0)
        return result

    def run_cbits(self, category: Optional[str] = None) -> TestSuiteResult:
        """
        Run all enabled Continuous Built-In Tests.
        
        :param category: Optional category filter. If specified, only runs tests in that category.
        :return: TestSuiteResult containing execution results.
        """
        result = self._run_tests(self._cbits, TestType.CBIT, category)
        self._cbit_history.append(result)
        if len(self._cbit_history) > self._max_history_size:
            self._cbit_history.pop(0)
        return result

    def run_single_test(self, name: str, test_type: Optional[TestType] = None) -> Optional[TestExecutionResult]:
        """
        Run a single test by name.
        
        :param name: Name of the test to run.
        :param test_type: Optional test type to search in. If None, searches both.
        :return: TestExecutionResult or None if test not found.
        """
        test_info = self.get_test_info(name, test_type)
        if not test_info:
            self._logger.error(f"{self._log_area}: Test '{name}' not found")
            return None
        
        if not test_info.enabled:
            self._logger.warning(f"{self._log_area}: Test '{name}' is disabled")
            return TestExecutionResult(
                name=name,
                result=TestResult.SKIPPED,
                execution_time=0,
                error_message="Test is disabled"
            )
        
        return self._execute_single_test(test_info)

    def _run_tests(self, tests: Dict[str, TestInfo], test_type: TestType, category: Optional[str] = None) -> TestSuiteResult:
        """Internal method to run a collection of tests."""
        # Filter tests
        tests_to_run = [
            test for test in tests.values() 
            if test.enabled and (category is None or test.category == category)
        ]
        
        if not tests_to_run:
            self._logger.warning(
                f"{self._log_area}: No enabled {test_type.value} tests"
                f"{f' in category {category}' if category else ''} to run"
            )
            return TestSuiteResult(
                test_type=test_type,
                overall_result=TestResult.SKIPPED,
                total_tests=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                total_execution_time=0
            )
        
        # Log test start
        category_str = f" [{category}]" if category else ""
        self._logger.info(f"{self._log_area}: {f' Running {test_type.value}{category_str} ':-^50}")
        
        suite_start_time = timestamp()
        results = []
        passed = failed = errors = 0
        
        # Execute tests
        for test_info in sorted(tests_to_run, key=lambda t: (t.category or "", t.name)):
            result = self._execute_single_test(test_info)
            results.append(result)
            
            if result.result == TestResult.PASS:
                passed += 1
            elif result.result == TestResult.FAIL:
                failed += 1
            elif result.result == TestResult.ERROR:
                errors += 1
        
        # Calculate overall result
        suite_execution_time = timestamp() - suite_start_time
        overall_result = TestResult.PASS if failed == 0 and errors == 0 else TestResult.FAIL
        
        # Create suite result
        suite_result = TestSuiteResult(
            test_type=test_type,
            overall_result=overall_result,
            total_tests=len(tests_to_run),
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=0,
            total_execution_time=suite_execution_time,
            individual_results=results
        )
        
        # Log completion
        result_str = f"{overall_result.value} ({passed}/{len(tests_to_run)})"
        self._logger.info(
            f"{self._log_area}: {f' {test_type.value}{category_str} Complete [{result_str}] ':-^50}"
        )
        
        # Notify callbacks
        for callback in self._result_callbacks:
            try:
                callback(suite_result)
            except Exception as e:
                self._logger.error(f"{self._log_area}: Error in result callback: {e}", exc_info=True)
        
        return suite_result

    def _execute_single_test(self, test_info: TestInfo) -> TestExecutionResult:
        """Execute a single test and return the result."""
        start_time = timestamp()
        
        try:
            # Execute the test with timeout handling (simplified - real implementation might use threading)
            result = test_info.method()
            execution_time = timestamp() - start_time
            
            # Update test statistics
            test_info.execution_count += 1
            test_info.total_execution_time += execution_time
            test_info.last_execution_time = execution_time
            
            if result is True:
                test_info.pass_count += 1
                test_info.last_result = TestResult.PASS
                test_info.last_error = None
                self._logger.info(f"{self._log_area}: [PASS] {test_info.name} ({execution_time}ms)")
                return TestExecutionResult(
                    name=test_info.name,
                    result=TestResult.PASS,
                    execution_time=execution_time
                )
            else:
                test_info.fail_count += 1
                test_info.last_result = TestResult.FAIL
                test_info.last_error = "Test returned False"
                self._logger.error(f"{self._log_area}: [FAIL] {test_info.name} ({execution_time}ms)")
                return TestExecutionResult(
                    name=test_info.name,
                    result=TestResult.FAIL,
                    execution_time=execution_time,
                    error_message="Test returned False"
                )
                
        except Exception as e:
            execution_time = timestamp() - start_time
            error_msg = str(e)
            
            # Update test statistics
            test_info.execution_count += 1
            test_info.total_execution_time += execution_time
            test_info.last_execution_time = execution_time
            test_info.error_count += 1
            test_info.last_result = TestResult.ERROR
            test_info.last_error = error_msg
            
            self._logger.error(f"{self._log_area}: [ERROR] {test_info.name} ({execution_time}ms): {e}", exc_info=True)
            return TestExecutionResult(
                name=test_info.name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=error_msg
            )

    def get_execution_history(self, test_type: TestType, limit: int = 10) -> List[TestSuiteResult]:
        """
        Get recent execution history for a test type.
        
        :param test_type: Type of tests to get history for.
        :param limit: Maximum number of recent results to return.
        :return: List of recent TestSuiteResult objects.
        """
        history = self._pbit_history if test_type == TestType.PBIT else self._cbit_history
        return history[-limit:] if history else []