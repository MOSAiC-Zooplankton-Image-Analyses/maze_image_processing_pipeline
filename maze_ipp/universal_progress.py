from typing import Callable, Iterable, List, Optional, Protocol, Union

class ProgressHandler:
    def __init__(
        self, iterable: Optional[Iterable], total: Union[int, float, None]
    ) -> None:
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def update(self, n=1):


    def __iter__(self):
        with self:
            for obj in self.iterable:
                yield obj

                self.update()

class HandlerFactory(Protocol):
    def __call__(self, iterable: Optional[Iterable], total: Union[int, float, None]) -> "ProgressHandler": ...

class ProgressManager:
    def __init__(self, handler_factory: HandlerFactory) -> None:
        self.handler_factory = handler_factory

    def progress(self, iterable=None, *, total=None) -> ProgressHandler:
        return self.handler_factory(iterable, total)

    def set_handler_factory(self, handler_factory: HandlerFactory):
        self.handler_factory = handler_factory


class LoggingProgressHandler(ProgressHandler):
    ...

class ttyProgressHandler(ProgressHandler):
    ...

_root_progress_manager = ...


def get_progress_manager(name=None) -> ProgressManager: ...


###

# A per-module progress manager allows the handler to filter
pm = get_progress_manager(__name__)

total = 10
with pm.progress(total=total) as pbar:
    for i in range(total):
        pbar.update()

for i in pm.progress(range(total)):
    pass
