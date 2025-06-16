import logging
import threading
import abc
from argparse import ArgumentParser
import sys
import argparse
from stencil._managers.bit_manager import BitManager
from stencil._managers.cycle_manager import CycleManager
from stencil._utilities import stop_event


class Service(abc.ABC):
    """
    An abstract base class for creating applications that run tasks periodically.

    Subclasses should define methods and decorate them with @pbit, @cbit,
    and @periodic_task to schedule them for execution. The main loop runs in
    a dedicated thread and can be stopped gracefully.
    """

    def __init__(self):
        """
        Initializes the application base.
        """
        self._logger = logging.getLogger(self.__class__.__qualname__)
        self._log_area: str = self.__class__.__name__
        self._app_thread: threading.Thread | None = None

        self._bit_manager: BitManager = BitManager(self)
        self._cycle_manager: CycleManager = CycleManager(self)

        # The subclass must define its configuration here.
        self._configure()

        self._logger.info(f"{self._log_area}: Initialised and configured.")

    @abc.abstractmethod
    def _configure(self):
        """
        Abstract method that subclasses MUST implement.

        This is the central place to call the `register_pbit`, `register_cbit`,
        and `register_periodic_task` methods to build the application's behavior.
        """
        pass

    # --- Registration Methods ---
    def register_pbit(self, method):
        return self._bit_manager.register_pbit(method)

    def register_cbit(self, method):
        return self._bit_manager.register_cbit(method)

    def register_routine(self, method, period_ms: int = 1000):
        return self._cycle_manager.register_routine(method, period_ms)

        # Discover and register decorated methods

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        """
        Hook for subclasses to add their own command-line arguments.
        """
        pass

    def start(self):
        """
        Starts the main execution loop of the application.

        This method is intended to be run in a separate thread. It first runs
        startup tests (PBITs) and then enters the main periodic loop, which
        can be gracefully stopped via the stop() method.
        """
        self._cycle_manager.start()

    def stop(self):
        """Signals the application to shut down gracefully."""
        self._logger.info("Stop signal received by application.")
        stop_event.set()
        if self._cycle_manager:
            self._cycle_manager.stop()


def main(service_class, description="A periodic Python application."):
    """
    The main entrypoint for a periodic application.

    This function sets up logging, parses arguments, and then instantiates and
    runs the application in a separate thread. The main thread waits for a
    shutdown command ('s') from the user.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-p",
        "--period",
        type=float,
        default=1.0,
        help="The cycle period in seconds. (Default: 1.0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the console logging level. (Default: INFO)",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Path to a file to write logs to.")

    service_class.add_arguments(parser)
    args = parser.parse_args()

    # --- Logging Setup ---
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s|%(name)-15s|%(levelname)-8s|%(message)s",
        stream=sys.stdout,
    )
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s"))
        logging.getLogger().addHandler(file_handler)

    main_logger = logging.getLogger("MainThread")
    service = None

    try:
        service = service_class()
        service.start()

        main_logger.info(f"'{service_class.__name__}' is running in the background.")
        print("\n*** Press 's' and then Enter to stop the application gracefully. ***\n")

        # Block main thread until user requests shutdown
        for line in sys.stdin:
            if line.strip().lower() == "s":
                break

        main_logger.info("Shutdown requested by user.")

    except KeyboardInterrupt:
        main_logger.info("KeyboardInterrupt received in main thread. Initiating shutdown.")
    except Exception as e:
        main_logger.critical(f"An error occurred during startup: {e}", exc_info=True)
    finally:
        if service:
            service.stop()  # Signal the thread to stop
        main_logger.info("Application has been stopped. Exiting.")
