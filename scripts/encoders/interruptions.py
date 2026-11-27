"""Request a clean checkpoint at the next completed optimizer-update boundary."""

from contextlib import contextmanager
from collections.abc import Iterator
import signal


class StopRequest:
    def __init__(self):
        self.signal_number: int | None = None

    def request(self, signum: int, frame=None) -> None:
        self.signal_number = signum

    def __call__(self) -> bool:
        return self.signal_number is not None


@contextmanager
def stop_signals(request: StopRequest) -> Iterator[None]:
    previous = {
        number: signal.signal(number, request.request)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
