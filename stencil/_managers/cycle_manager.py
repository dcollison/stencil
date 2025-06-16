from __future__ import annotations

import concurrent.futures
import threading
from typing import TYPE_CHECKING, Callable

from stencil._utilities import stop_event
from stencil._managers.i_manager import IManager
from stencil._utilities import timestamp

if TYPE_CHECKING:
    from stencil import Service


class CycleManager(IManager):
    """
    Manages a non-overlapping, multi-threaded periodic routine scheduler.

    This class runs a main loop in a dedicated thread. The loop checks at a
    regular interval (the tick rate) which registered routines are due to be run.
    Due routines are submitted to a thread pool for execution.

    A locking mechanism ensures that a new execution of a routine will not be
    scheduled if its previous execution is still in progress.
    """

    def __init__(self, service: "Service", tick_rate_ms: int = 10, max_workers: int = 4):
        """
        Initializes the CycleManager.

        :param service: The parent service object that owns this manager.
        :param tick_rate_ms: The period, in milliseconds, at which the scheduler
                             checks for due routines. This is the "tick rate".
        :param max_workers: The maximum number of threads in the worker pool.
        """
        super().__init__(service)

        # --- Scheduler State ---
        self.tick_rate_ms: int = tick_rate_ms
        self.cycle_count: int = 0

        # --- Routine Registration ---
        self._routines: list[tuple[Callable[[], None], int]] = []
        self._routine_last_run_times: dict[str, int] = {}

        # --- Threading and Concurrency Control ---
        self._main_thread: threading.Thread | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="RoutineThread"
        )
        self._routines_in_progress: set[str] = set()
        self._routine_lock = threading.Lock()  # Protects access to _routines_in_progress

    def register_routine(self, method: Callable[[], None], period_ms: int = 1000) -> None:
        """
        Registers a method to be run periodically.

        :param method: The function or method to execute. It must take no arguments.
        :param period_ms: The desired period, in milliseconds, between executions.
        """
        routine_name = method.__name__
        self._routines.append((method, period_ms))
        self._routine_last_run_times[routine_name] = 0  # Initialize last run time to 0
        self._logger.debug(
            f"{self._log_area}: Registered periodic routine: {routine_name} "
            f"[{period_ms}ms / {1000 / period_ms:.2f}Hz]"
        )

    def start(self) -> None:
        """
        Starts the main scheduler loop in a new thread.
        """
        if self._main_thread is not None and self._main_thread.is_alive():
            self._logger.warning(f"{self._log_area}: Cycle manager is already running.")
            return

        self._logger.info(f"{self._log_area}: Starting cycle manager thread.")
        self._main_thread = threading.Thread(target=self._run, name="CycleManagerThread")
        self._main_thread.start()

    def stop(self) -> None:
        """
        Signals the scheduler to stop and waits for the thread to terminate.
        """
        if self._main_thread is None or not self._main_thread.is_alive():
            self._logger.warning(f"{self._log_area}: Cycle manager is not running.")
            return

        self._logger.info(f"{self._log_area}: Stopping cycle manager thread.")
        stop_event.set()
        self._main_thread.join()
        self._logger.info(f"{self._log_area}: Cycle manager thread stopped.")

    def _run(self) -> None:
        """
        The main scheduler loop. This method should not be called directly.

        This loop continuously checks for due routines and submits them to the
        thread pool. It also handles graceful shutdown of the executor.
        """
        try:
            while not stop_event.is_set():
                self.cycle_count += 1
                current_time_ms = timestamp()

                for routine_method, period_ms in self._routines:
                    routine_name = routine_method.__name__

                    # Check if the routine's period has elapsed since it was last scheduled.
                    if self._routine_last_run_times[routine_name] + period_ms > current_time_ms:
                        continue

                    # Safely check if the routine is already running to prevent overlap.
                    with self._routine_lock:
                        if routine_name in self._routines_in_progress:
                            # self._logger.warning(
                            #     f"{self._log_area}: Skipping routine {routine_name} as its "
                            #     "previous execution is still in progress."
                            # )
                            continue

                        # Mark the routine as running before submitting it.
                        self._routines_in_progress.add(routine_name)

                    # Submit the routine to the thread pool for execution.
                    self._executor.submit(self._routine_runner_wrapper, routine_method)
                    self._routine_last_run_times[routine_name] = current_time_ms

                # Wait for the next tick, but wake up immediately if stop is requested.
                stop_event.wait(self.tick_rate_ms / 1000.0)

        except Exception as e:
            self._logger.critical(
                f"{self._log_area}: An unhandled exception occurred in the scheduler thread: {e}",
                exc_info=True,
            )
        finally:
            # Ensure the thread pool is shut down gracefully.
            self._logger.info(f"{self._log_area}: Shutting down routine executor.")
            self._executor.shutdown()

    def _routine_runner_wrapper(self, routine_method: Callable[[], None]) -> None:
        """
        A wrapper executed by the thread pool for each routine.

        This handles timing, logging, exception catching, and releasing the
        routine lock to allow the next execution to be scheduled.

        :param routine_method: The periodic routine to execute.
        """
        routine_name = routine_method.__name__
        self._logger.debug(f"{self._log_area}: [   START] Routine '{routine_name}'")
        t_start = timestamp()
        try:
            routine_method()
            t_end = timestamp()
            self._logger.info(
                f"{self._log_area}: [COMPLETE] Routine '{routine_name}' completed successfully in {t_end - t_start}ms"
            )
        except Exception as e:
            t_end = timestamp()
            self._logger.error(
                f"{self._log_area}: [   ERROR] Routine '{routine_name}' failed after {t_end - t_start}ms - {e}",
                exc_info=True,
            )
        finally:
            # This block is critical: it ensures the routine is marked as "not running"
            # even if it fails, allowing it to be scheduled again in the future.
            with self._routine_lock:
                self._routines_in_progress.remove(routine_name)
