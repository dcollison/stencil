from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from collections import defaultdict

from stencil._utilities import stop_event
from stencil._managers.i_manager import IManager
from stencil._utilities import timestamp

if TYPE_CHECKING:
    from stencil import Service


@dataclass
class RoutineInfo:
    """Encapsulates information about a registered routine."""
    method: Callable[[], None]
    period_ms: int
    name: str
    last_run_time: int = 0
    execution_count: int = 0
    total_execution_time: int = 0
    last_execution_time: int = 0
    failure_count: int = 0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time in milliseconds."""
        return self.total_execution_time / self.execution_count if self.execution_count > 0 else 0.0


class CycleManager(IManager):
    """
    Manages a non-overlapping, multi-threaded periodic routine scheduler.

    This class runs a main loop in a dedicated thread. The loop checks at a
    regular interval (the tick rate) which registered routines are due to be run.
    Due routines are submitted to a thread pool for execution.

    A locking mechanism ensures that a new execution of a routine will not be
    scheduled if its previous execution is still in progress.
    """

    def __init__(
        self, 
        service: "Service", 
        tick_rate_ms: int = 10, 
        max_workers: int = 4,
        enable_metrics: bool = True
    ):
        """
        Initializes the CycleManager.

        :param service: The parent service object that owns this manager.
        :param tick_rate_ms: The period, in milliseconds, at which the scheduler
                             checks for due routines. This is the "tick rate".
        :param max_workers: The maximum number of threads in the worker pool.
        :param enable_metrics: Whether to collect detailed execution metrics.
        """
        super().__init__(service)

        # Validate parameters
        if tick_rate_ms <= 0:
            raise ValueError("tick_rate_ms must be positive")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        # --- Scheduler State ---
        self.tick_rate_ms: int = tick_rate_ms
        self.cycle_count: int = 0
        self.enable_metrics: bool = enable_metrics

        # --- Routine Registration ---
        self._routines: dict[str, RoutineInfo] = {}
        self._routine_execution_order: list[str] = []  # Track registration order

        # --- Threading and Concurrency Control ---
        self._main_thread: Optional[threading.Thread] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._routines_in_progress: set[str] = set()
        self._routine_lock = threading.Lock()  # Protects access to _routines_in_progress
        self._metrics_lock = threading.Lock()  # Protects metrics updates
        self._is_running = False
        self._shutdown_event = threading.Event()

        # --- Performance Metrics ---
        self._scheduler_overhead_times: list[int] = []
        self._max_overhead_samples = 100  # Keep last 100 samples

    def register_routine(
        self, 
        method: Callable[[], None], 
        period_ms: int = 1000,
        name: Optional[str] = None
    ) -> None:
        """
        Registers a method to be run periodically.

        :param method: The function or method to execute. It must take no arguments.
        :param period_ms: The desired period, in milliseconds, between executions.
        :param name: Optional custom name for the routine. Defaults to method.__name__.
        :raises ValueError: If period_ms is not positive or routine name already exists.
        """
        if period_ms <= 0:
            raise ValueError("period_ms must be positive")
        
        routine_name = name or method.__name__
        
        if routine_name in self._routines:
            raise ValueError(f"Routine '{routine_name}' is already registered")

        routine_info = RoutineInfo(
            method=method,
            period_ms=period_ms,
            name=routine_name
        )
        
        self._routines[routine_name] = routine_info
        self._routine_execution_order.append(routine_name)
        
        self._logger.debug(
            f"{self._log_area}: Registered periodic routine: {routine_name} "
            f"[{period_ms}ms / {1000 / period_ms:.2f}Hz]"
        )

    def unregister_routine(self, name: str) -> bool:
        """
        Unregisters a routine by name.
        
        :param name: The name of the routine to unregister.
        :return: True if routine was found and removed, False otherwise.
        """
        if name not in self._routines:
            return False
            
        # Wait for routine to finish if it's currently running
        while name in self._routines_in_progress:
            time.sleep(0.001)  # Small sleep to avoid busy waiting
            
        del self._routines[name]
        self._routine_execution_order.remove(name)
        
        self._logger.debug(f"{self._log_area}: Unregistered routine: {name}")
        return True

    def get_routine_metrics(self, name: str) -> Optional[dict]:
        """
        Get performance metrics for a specific routine.
        
        :param name: The name of the routine.
        :return: Dictionary containing metrics or None if routine not found.
        """
        if not self.enable_metrics or name not in self._routines:
            return None
            
        routine = self._routines[name]
        return {
            'name': routine.name,
            'period_ms': routine.period_ms,
            'execution_count': routine.execution_count,
            'total_execution_time': routine.total_execution_time,
            'average_execution_time': routine.average_execution_time,
            'last_execution_time': routine.last_execution_time,
            'failure_count': routine.failure_count,
            'last_run_time': routine.last_run_time
        }

    def get_scheduler_metrics(self) -> dict:
        """Get overall scheduler performance metrics."""
        metrics = {
            'cycle_count': self.cycle_count,
            'tick_rate_ms': self.tick_rate_ms,
            'registered_routines': len(self._routines),
            'routines_in_progress': len(self._routines_in_progress),
            'is_running': self._is_running
        }
        
        if self.enable_metrics and self._scheduler_overhead_times:
            overhead_times = self._scheduler_overhead_times[-self._max_overhead_samples:]
            metrics.update({
                'average_scheduler_overhead_ms': sum(overhead_times) / len(overhead_times),
                'max_scheduler_overhead_ms': max(overhead_times),
                'min_scheduler_overhead_ms': min(overhead_times)
            })
        
        return metrics

    def start(self) -> None:
        """
        Starts the main scheduler loop in a new thread.
        """
        if self._is_running:
            self._logger.warning(f"{self._log_area}: Cycle manager is already running.")
            return

        self._logger.info(f"{self._log_area}: Starting cycle manager thread.")
        
        # Create thread pool
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,  # You might want to make this configurable
            thread_name_prefix="RoutineThread"
        )
        
        self._is_running = True
        self._shutdown_event.clear()
        self._main_thread = threading.Thread(target=self._run, name="CycleManagerThread")
        self._main_thread.start()

    def stop(self, timeout: float = 30.0) -> bool:
        """
        Signals the scheduler to stop and waits for the thread to terminate.
        
        :param timeout: Maximum time to wait for shutdown in seconds.
        :return: True if shutdown completed within timeout, False otherwise.
        """
        if not self._is_running:
            self._logger.warning(f"{self._log_area}: Cycle manager is not running.")
            return True

        self._logger.info(f"{self._log_area}: Stopping cycle manager thread.")
        
        # Signal shutdown
        self._shutdown_event.set()
        stop_event.set()
        
        # Wait for main thread to finish
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=timeout)
            if self._main_thread.is_alive():
                self._logger.error(f"{self._log_area}: Cycle manager thread did not stop within {timeout}s")
                return False
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, timeout=timeout)
            self._executor = None
        
        self._is_running = False
        self._logger.info(f"{self._log_area}: Cycle manager thread stopped.")
        return True

    def _run(self) -> None:
        """
        The main scheduler loop. This method should not be called directly.

        This loop continuously checks for due routines and submits them to the
        thread pool. It also handles graceful shutdown of the executor.
        """
        try:
            while not (stop_event.is_set() or self._shutdown_event.is_set()):
                cycle_start_time = timestamp()
                self.cycle_count += 1
                current_time_ms = timestamp()

                # Process routines in registration order for consistency
                for routine_name in self._routine_execution_order:
                    if routine_name not in self._routines:
                        continue  # Routine may have been unregistered
                        
                    routine_info = self._routines[routine_name]

                    # Check if the routine's period has elapsed since it was last scheduled.
                    if routine_info.last_run_time + routine_info.period_ms > current_time_ms:
                        continue

                    # Safely check if the routine is already running to prevent overlap.
                    with self._routine_lock:
                        if routine_name in self._routines_in_progress:
                            self._logger.debug(
                                f"{self._log_area}: Skipping routine {routine_name} as its "
                                "previous execution is still in progress."
                            )
                            continue

                        # Mark the routine as running before submitting it.
                        self._routines_in_progress.add(routine_name)

                    # Submit the routine to the thread pool for execution.
                    try:
                        self._executor.submit(self._routine_runner_wrapper, routine_info)
                        routine_info.last_run_time = current_time_ms
                    except Exception as e:
                        # If submission fails, remove from in-progress set
                        with self._routine_lock:
                            self._routines_in_progress.discard(routine_name)
                        self._logger.error(
                            f"{self._log_area}: Failed to submit routine {routine_name}: {e}"
                        )

                # Track scheduler overhead
                if self.enable_metrics:
                    cycle_end_time = timestamp()
                    overhead = cycle_end_time - cycle_start_time
                    with self._metrics_lock:
                        self._scheduler_overhead_times.append(overhead)
                        if len(self._scheduler_overhead_times) > self._max_overhead_samples:
                            self._scheduler_overhead_times.pop(0)

                # Wait for the next tick, but wake up immediately if stop is requested.
                if not (stop_event.is_set() or self._shutdown_event.is_set()):
                    stop_event.wait(self.tick_rate_ms / 1000.0)

        except Exception as e:
            self._logger.critical(
                f"{self._log_area}: An unhandled exception occurred in the scheduler thread: {e}",
                exc_info=True,
            )
        finally:
            self._is_running = False
            self._logger.info(f"{self._log_area}: Scheduler loop terminated.")

    def _routine_runner_wrapper(self, routine_info: RoutineInfo) -> None:
        """
        A wrapper executed by the thread pool for each routine.

        This handles timing, logging, exception catching, and releasing the
        routine lock to allow the next execution to be scheduled.

        :param routine_info: The routine information object containing method and metadata.
        """
        routine_name = routine_info.name
        self._logger.debug(f"{self._log_area}: [   START] Routine '{routine_name}'")
        t_start = timestamp()
        
        try:
            routine_info.method()
            t_end = timestamp()
            execution_time = t_end - t_start
            
            # Update metrics
            if self.enable_metrics:
                with self._metrics_lock:
                    routine_info.execution_count += 1
                    routine_info.total_execution_time += execution_time
                    routine_info.last_execution_time = execution_time
            
            # Create detailed completion message with metrics
            completion_msg = f"{self._log_area}: [COMPLETE] Routine '{routine_name}' completed successfully in {execution_time}ms"
            
            if self.enable_metrics:
                with self._metrics_lock:
                    avg_time = routine_info.average_execution_time
                    total_calls = routine_info.execution_count
                    
                    # Calculate performance comparison with previous execution
                    perf_indicator = ""
                    if routine_info.execution_count > 1:  # Need at least 2 executions to compare
                        # Get previous execution time from running average
                        prev_total_time = routine_info.total_execution_time - execution_time
                        prev_avg = prev_total_time / (routine_info.execution_count - 1) if routine_info.execution_count > 1 else execution_time
                        
                        if execution_time < prev_avg * 0.9:  # More than 10% faster
                            percent_faster = ((prev_avg - execution_time) / prev_avg) * 100
                            perf_indicator = f" [{percent_faster:.1f}% faster than avg]"
                        elif execution_time > prev_avg * 1.1:  # More than 10% slower
                            percent_slower = ((execution_time - prev_avg) / prev_avg) * 100
                            perf_indicator = f" [{percent_slower:.1f}% slower than avg]"
                        else:
                            perf_indicator = " [similar to avg]"
                    
                    completion_msg += f" | Calls: {total_calls}, Avg: {avg_time:.1f}ms{perf_indicator}"
                    
                    # Add failure rate if there have been failures
                    if routine_info.failure_count > 0:
                        failure_rate = (routine_info.failure_count / total_calls) * 100
                        completion_msg += f", Failures: {routine_info.failure_count} ({failure_rate:.1f}%)"
            
            self._logger.info(completion_msg)
            
        except Exception as e:
            t_end = timestamp()
            execution_time = t_end - t_start
            
            # Update failure metrics
            if self.enable_metrics:
                with self._metrics_lock:
                    routine_info.execution_count += 1
                    routine_info.total_execution_time += execution_time
                    routine_info.last_execution_time = execution_time
                    routine_info.failure_count += 1
            
            self._logger.error(
                f"{self._log_area}: [   ERROR] Routine '{routine_name}' failed after {execution_time}ms - {e}",
                exc_info=True,
            )
        finally:
            # This block is critical: it ensures the routine is marked as "not running"
            # even if it fails, allowing it to be scheduled again in the future.
            with self._routine_lock:
                self._routines_in_progress.discard(routine_name)  # Use discard instead of remove