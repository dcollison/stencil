import threading
import time

stop_event = threading.Event()


def timestamp() -> int:
    return int(time.time() * 1000)
