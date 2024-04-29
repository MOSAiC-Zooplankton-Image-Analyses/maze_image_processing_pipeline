import logging
import time
from typing import Literal, Optional
import prefixed

from morphocut.core import (
    Node,
    RawOrVariable,
    ReturnOutputs,
    Stream,
    closing_if_closable,
)

logger = logging.getLogger(__name__)

NumberFormat = Literal[None, "si", "iec"]


def format_number(x: float, format: NumberFormat):
    if format is None:
        return f"{x:.2f}"
    if format == "si":
        return f"{prefixed.Float(x):.2h}"
    if format == "iec":
        return f"{prefixed.Float(x):.2k}"

    raise ValueError(f"Unsupported format: {format!r}")


def format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)

    if h:
        return f"{h:d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


class ProgressLogger:
    def __init__(
        self,
        *,
        n_total: Optional[float] = None,
        log_interval: float = 60,
        description: Optional[str] = None,
        unit="it",
        number_format: NumberFormat,
    ) -> None:
        self.n_total = n_total
        self.log_interval = log_interval
        self.description = description
        self.unit = unit
        self.number_format: NumberFormat = number_format

        self.n_done = 0
        self.t_last_update = time.time()
        self.t_elapsed = 0
        self.t_last_update = 0

    def update(self, n=1):
        t_now = time.time()
        delta_t = self.t_last_update - t_now
        self.t_last_update = t_now

        self.t_elapsed += delta_t
        self.n_done += n

        if t_now > self.t_last_update + self.log_interval:
            rate = self.t_elapsed / self.n_done

            if self.description is not None:
                msg = f"{self.description}: "
            else:
                msg = ""

            parts = []

            if self.n_total is not None:
                t_remaining = rate * (self.n_total - self.n_done)

                parts.append(
                    f"{format_number(self.n_done, self.number_format)} / {format_number(self.n_total, self.number_format)}"
                )
                parts.append(f"{self.n_done / self.n_total:.2%}")

                parts.append(
                    f"{format_interval(self.t_elapsed)} + {format_interval(t_remaining)}"
                )
            else:
                parts.append(f"{format_number(self.n_done, self.number_format)} / ?")

                parts.append(f"{format_interval(self.t_elapsed)}")

            if rate >= 0:
                parts.append(f"{format_number(rate, self.number_format)}{self.unit}/s")
            else:
                parts.append(f"{1/rate:.2f}s/{self.unit}")

            msg += ", ".join(parts)

            logger.info(msg)


@ReturnOutputs
class LogProgress(Node):
    """ """

    def __init__(
        self,
        description: Optional[RawOrVariable[str]] = None,
        log_interval=60,
        number_format: NumberFormat = "si",
    ):
        super().__init__()
        self.description = description
        self.log_interval = log_interval
        self.number_format: NumberFormat = number_format

    def transform_stream(self, stream: Stream):
        with closing_if_closable(stream):
            progress_logger = ProgressLogger(
                log_interval=self.log_interval, number_format=self.number_format
            )

            for n_consumed, obj in enumerate(stream, 1):
                description = self.prepare_input(obj, "description")

                if description is not None:
                    progress_logger.description = str(description)

                if obj.n_remaining_hint is not None:
                    progress_logger.n_total = n_consumed + obj.n_remaining_hint

                yield obj
