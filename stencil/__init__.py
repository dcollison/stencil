import logging
import threading
import abc
from argparse import ArgumentParser
import sys
import signal
from typing import Optional, Type, Dict, Callable, Any
from contextlib import contextmanager
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
        """Initialize the service with managers and configuration."""
        self._logger = logging.getLogger(self.__class__.__qualname__)
        self._log_area: str = self.__class__.__name__
        self._app_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._shutdown_lock = threading.Lock()

        # Initialize managers
        self._bit_manager: BitManager = BitManager(self)
        self._cycle_manager: CycleManager = CycleManager(self)

        # Configure the service
        try:
            self._configure()
            self._logger.info(f"{self._log_area}: Initialized and configured successfully.")
        except Exception as e:
            self._logger.error(f"{self._log_area}: Configuration failed: {e}")
            raise

    @abc.abstractmethod
    def _configure(self):
        """
        Abstract method that subclasses MUST implement.

        This is the central place to call the `register_pbit`, `register_cbit`,
        and `register_routine` methods to build the application's behavior.
        """
        pass

    # --- Registration Methods ---
    def register_pbit(self, method):
        """Register a power-on built-in test method."""
        return self._bit_manager.register_pbit(method)

    def register_cbit(self, method):
        """Register a continuous built-in test method."""
        return self._bit_manager.register_cbit(method)

    def register_routine(self, method, period_ms: int = 1000):
        """Register a periodic routine with specified period in milliseconds."""
        return self._cycle_manager.register_routine(method, period_ms)

    def register_command(self, command: str, handler: Callable[[], Any], description: str = ""):
        """
        Register a user input command that can be executed while the service is running.
        
        Args:
            command: The command string (case-insensitive)
            handler: Function to call when command is entered
            description: Description of what the command does
        """
        if not hasattr(self, '_user_commands'):
            self._user_commands = {}
        self._user_commands[command.lower()] = {
            'handler': handler,
            'description': description
        }

    def get_user_commands(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered user commands."""
        return getattr(self, '_user_commands', {})

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        """
        Hook for subclasses to add their own command-line arguments.
        
        Args:
            parser: The ArgumentParser instance to add arguments to.
        """
        pass

    def start(self):
        """
        Start the main execution loop of the application.

        This method runs startup tests (PBITs) and then enters the main 
        periodic loop, which can be gracefully stopped via the stop() method.
        """
        with self._shutdown_lock:
            if self._is_running:
                self._logger.warning("Service is already running")
                return
            
            self._is_running = True

        try:
            self._logger.info(f"{self._log_area}: Starting service...")
            self._cycle_manager.start()
            self._logger.info(f"{self._log_area}: Service started successfully")
        except Exception as e:
            self._logger.error(f"{self._log_area}: Failed to start service: {e}")
            with self._shutdown_lock:
                self._is_running = False
            raise

    def stop(self):
        """Signal the application to shut down gracefully."""
        with self._shutdown_lock:
            if not self._is_running:
                return
            
            self._is_running = False

        self._logger.info(f"{self._log_area}: Stop signal received, initiating shutdown...")
        stop_event.set()
        
        if self._cycle_manager:
            try:
                self._cycle_manager.stop()
                self._logger.info(f"{self._log_area}: Service stopped successfully")
            except Exception as e:
                self._logger.error(f"{self._log_area}: Error during shutdown: {e}")

    @property
    def is_running(self) -> bool:
        """Check if the service is currently running."""
        with self._shutdown_lock:
            return self._is_running


class ServiceRunner:
    """Handles the execution and lifecycle of a Service instance."""
    
    def __init__(self, service_class: Type[Service], description: str = "A periodic Python service."):
        self.service_class = service_class
        self.description = description
        self.service: Optional[Service] = None
        self.logger = logging.getLogger("ServiceRunner")
        self._shutdown_requested = threading.Event()
        self._built_in_commands = {
            's': {'handler': self._stop_service, 'description': 'Stop the service gracefully'},
            'stop': {'handler': self._stop_service, 'description': 'Stop the service gracefully'},
            'h': {'handler': self._show_help, 'description': 'Show available commands'},
            'help': {'handler': self._show_help, 'description': 'Show available commands'},
            'status': {'handler': self._show_status, 'description': 'Show service status'},
        }

    def setup_argument_parser(self) -> ArgumentParser:
        """Create and configure the argument parser."""
        parser = ArgumentParser(description=self.description)
        parser.add_argument(
            "-p", "--period",
            type=float,
            default=1.0,
            help="The cycle period in seconds (default: 1.0)"
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the console logging level (default: INFO)"
        )
        parser.add_argument(
            "--log-file",
            type=str,
            help="Path to a file to write logs to"
        )
        
        # Allow service class to add custom arguments
        self.service_class.add_arguments(parser)
        return parser

    def setup_logging(self, log_level: str, log_file: Optional[str] = None):
        """Configure logging with console and optional file output."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            "%(asctime)s|%(name)-15s|%(levelname)-8s|%(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)

        # File handler (if specified)
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                self.logger.info(f"Logging to file: {log_file}")
            except Exception as e:
                self.logger.error(f"Failed to setup file logging: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self._shutdown_requested.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @contextmanager
    def service_context(self):
        """Context manager for service lifecycle."""
        try:
            self.service = self.service_class()
            yield self.service
        except Exception as e:
            self.logger.critical(f"Failed to initialize service: {e}", exc_info=True)
            raise
        finally:
            if self.service:
                self.service.stop()

    def wait_for_shutdown(self):
        """Wait for shutdown signal from user input or signal."""
        self.logger.info(f"'{self.service_class.__name__}' is running.")
        self._show_startup_help()

        # Start input monitoring in a separate thread
        input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        input_thread.start()

        # Wait for shutdown signal
        self._shutdown_requested.wait()
        self.logger.info("Shutdown requested.")

    def _show_startup_help(self):
        """Show initial help message with available commands."""
        print("\n" + "="*60)
        print("SERVICE RUNNING - Available Commands:")
        print("="*60)
        
        # Show built-in commands
        for cmd, info in self._built_in_commands.items():
            if cmd in ['s', 'h']:  # Show short versions primarily
                print(f"  {cmd:<10} - {info['description']}")
        
        # Show service-specific commands
        if self.service:
            user_commands = self.service.get_user_commands()
            if user_commands:
                print("\nService Commands:")
                for cmd, info in user_commands.items():
                    desc = info['description'] or 'Custom command'
                    print(f"  {cmd:<10} - {desc}")
        
        print("\nType any command and press Enter, or use Ctrl+C for immediate shutdown")
        print("="*60 + "\n")

    def _show_help(self):
        """Display all available commands."""
        print("\n" + "-"*50)
        print("AVAILABLE COMMANDS:")
        print("-"*50)
        
        print("Built-in Commands:")
        for cmd, info in self._built_in_commands.items():
            print(f"  {cmd:<12} - {info['description']}")
        
        if self.service:
            user_commands = self.service.get_user_commands()
            if user_commands:
                print("\nService Commands:")
                for cmd, info in user_commands.items():
                    desc = info['description'] or 'Custom command'
                    print(f"  {cmd:<12} - {desc}")
        
        print("-"*50 + "\n")

    def _show_status(self):
        """Display service status information."""
        if self.service:
            status = "RUNNING" if self.service.is_running else "STOPPED"
            print(f"\nService Status: {status}")
            print(f"Service Class: {self.service.__class__.__name__}")
            print(f"Available Commands: {len(self._get_all_commands())}")
        else:
            print("\nService Status: NOT INITIALIZED")
        print()

    def _stop_service(self):
        """Handle stop command."""
        print("Stopping service...")
        self._shutdown_requested.set()

    def _get_all_commands(self) -> Dict[str, Dict[str, Any]]:
        """Get all available commands (built-in + service-specific)."""
        all_commands = self._built_in_commands.copy()
        if self.service:
            user_commands = self.service.get_user_commands()
            all_commands.update(user_commands)
        return all_commands

    def _monitor_input(self):
        """Monitor stdin for user commands."""
        try:
            for line in sys.stdin:
                command = line.strip().lower()
                if not command:
                    continue
                
                # Get all available commands
                all_commands = self._get_all_commands()
                
                if command in all_commands:
                    try:
                        # Execute the command handler
                        handler = all_commands[command]['handler']
                        result = handler()
                        if result is not None:
                            print(f"Command result: {result}")
                    except Exception as e:
                        print(f"Error executing command '{command}': {e}")
                        self.logger.error(f"Command execution error: {e}")
                else:
                    print(f"Unknown command: '{command}'. Type 'h' for help.")
                
                # Check if shutdown was requested
                if self._shutdown_requested.is_set():
                    break
                    
        except EOFError:
            # Handle case where stdin is closed
            pass
        except Exception as e:
            self.logger.debug(f"Input monitoring error: {e}")

    def run(self):
        """Main execution method."""
        parser = self.setup_argument_parser()
        args = parser.parse_args()

        self.setup_logging(args.log_level, args.log_file)
        self.setup_signal_handlers()

        try:
            with self.service_context() as service:
                service.start()
                self.wait_for_shutdown()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)
            return 1
        finally:
            self.logger.info("Service runner exiting.")
        
        return 0


def main(service_class: Type[Service], description: str = "A periodic Python service."):
    """
    Main entry point for running a service.
    
    Args:
        service_class: The Service subclass to run
        description: Description for the argument parser
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    runner = ServiceRunner(service_class, description)
    return runner.run()


# Example usage:
if __name__ == "__main__":
    class ExampleService(Service):
        def _configure(self):
            # Register your methods here
            pass
            
            # Register custom commands
            self.register_command('restart', self._restart_components, 'Restart service components')
            self.register_command('debug', self._toggle_debug, 'Toggle debug mode')
            self.register_command('stats', self._show_stats, 'Show runtime statistics')
        
        def _restart_components(self):
            """Example custom command handler."""
            self._logger.info("Restarting components...")
            return "Components restarted successfully"
        
        def _toggle_debug(self):
            """Example debug toggle command."""
            current_level = self._logger.level
            if current_level == logging.DEBUG:
                self._logger.setLevel(logging.INFO)
                return "Debug mode OFF"
            else:
                self._logger.setLevel(logging.DEBUG)
                return "Debug mode ON"
        
        def _show_stats(self):
            """Example stats command."""
            return f"Service: {self.__class__.__name__}, Running: {self.is_running}"
    
    sys.exit(main(ExampleService, "Example periodic service"))