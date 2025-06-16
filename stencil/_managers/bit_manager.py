from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

from stencil._managers.i_manager import IManager

if TYPE_CHECKING:
    from stencil import Service


TestMethod = Callable[[], bool]


class BitManager(IManager):
    def __init__(self, service: "Service"):
        super().__init__(service)

        self._pbits = []
        self._cbits = []

    def register_pbit(self, method: TestMethod):
        self._pbits.append(method)
        self._logger.debug(f"{self._log_area}: Registered PBIT - {method.__name__}")
        a = inspect.signature(method).return_annotation

    def register_cbit(self, method: TestMethod):
        self._cbits.append(method)
        self._logger.debug(f"{self._log_area}: Registered CBIT - {method.__name__}")
        a = inspect.signature(method).return_annotation

    @property
    def n_pbits(self) -> int:
        return len(self._pbits)

    @property
    def n_cbits(self) -> int:
        return len(self._cbits)

    def run_pbit(self):
        return self._run_tests(self._pbits, "PBIT")

    def run_cbit(self):
        return self._run_tests(self._cbits, "CBIT")

    def _run_tests(self, tests, test_type: str):
        self._logger.info(f"{self._log_area}: {f' Running {test_type} ':-^40}")
        all_passed = True
        for test in tests:
            try:
                result = test()
                if result:
                    self._logger.info(f"[PASS] {test.__name__}")
                else:
                    all_passed = False
                    self._logger.error(f"[FAIL] {test.__name__}")
            except Exception as e:
                self._logger.error(f"[FAIL] {test.__name__}: {e}", exc_info=True)
                all_passed = False
        result_str = "PASS" if all_passed else "FAIL"
        self._logger.info(f"{f' {test_type} Complete [{result_str}] ':-^40}")
        return all_passed
