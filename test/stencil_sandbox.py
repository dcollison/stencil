import time

from stencil import *


class MyService(Service):

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        pass

    def _configure(self):
        self.register_routine(self.do_nothing)

    def do_nothing(self):
        time.sleep(1)


if __name__ == "__main__":
    # The `main` function now handles starting the app in a thread
    # and waiting for the user to stop it.
    main(service_class=MyService, description="A sample monitoring application.")
