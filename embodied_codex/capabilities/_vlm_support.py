"""Bounded transport helpers shared by sensor-only VLM capabilities."""
from __future__ import annotations

import io
from queue import Empty, Queue
import threading
import time
from typing import Callable, TypeVar


T = TypeVar("T")


def compact_image(data: bytes, mime: str) -> tuple[str, bytes]:
    """Return a smaller model-view preview without modifying source evidence."""
    if mime != "image/png" or len(data) <= 64 * 1024:
        return mime, data
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92)
        preview = output.getvalue()
        if len(preview) < len(data) * 0.8:
            return "image/jpeg", preview
    except Exception:
        pass
    return mime, data


def bounded_consensus(
    rounds: int,
    total_timeout: float,
    call: Callable[[int, float], T],
    *,
    error_type: type[Exception],
    operation: str,
    minimum_results: int | None = None,
    completion_grace: float = 2.0,
    decision_quorum: Callable[[list[T]], bool] | None = None,
) -> list[T]:
    """Run independent votes concurrently under one wall-clock deadline.

    Daemon workers prevent an uncooperative HTTP stream from holding the robot
    loop after the deadline. Each caller also receives the absolute deadline so
    normal API requests and retries can shorten their own per-request timeout.
    """
    count = max(1, int(rounds))
    timeout = max(0.1, float(total_timeout))
    deadline = time.monotonic() + timeout
    completed: Queue[tuple[int, T | None, BaseException | None]] = Queue()

    def worker(index: int) -> None:
        try:
            completed.put((index, call(index, deadline), None))
        except BaseException as exc:
            completed.put((index, None, exc))

    for index in range(count):
        threading.Thread(
            target=worker,
            args=(index,),
            name=f"embodied-codex-{operation}-{index}",
            daemon=True,
        ).start()

    required=count if minimum_results is None else max(1,min(int(minimum_results),count))
    results: list[T | None] = [None] * count;pending=count;successes=0
    first_error: BaseException|None=None;grace_deadline=None
    def quorum_reached():
        values=[value for value in results if value is not None]
        return bool(decision_quorum(values)) if decision_quorum is not None else False
    while pending:
        active_deadline=min(deadline,grace_deadline) if grace_deadline else deadline
        remaining = active_deadline - time.monotonic()
        if remaining <= 0:
            if ((decision_quorum is None and successes>=required)
                    or (decision_quorum is not None and quorum_reached())):break
            if first_error is not None:raise first_error
            suffix=(" without a decision quorum" if decision_quorum is not None else "")
            raise error_type(f"{operation} exceeded {timeout:g} seconds{suffix}")
        try:
            index, value, error = completed.get(timeout=remaining)
        except Empty as exc:
            if ((decision_quorum is None and successes>=required)
                    or (decision_quorum is not None and quorum_reached())):break
            raise error_type(f"{operation} exceeded {timeout:g} seconds") from exc
        pending-=1
        if error is not None:
            if first_error is None:first_error=error
            if successes+pending<required:raise first_error
            continue
        results[index] = value;successes+=1
        enough=(successes>=required if decision_quorum is None else quorum_reached())
        if enough and grace_deadline is None:
            grace_deadline=min(deadline,time.monotonic()+max(0.0,float(completion_grace)))
    values=[value for value in results if value is not None]
    if decision_quorum is not None and not decision_quorum(values):
        if first_error is not None:raise first_error
        raise error_type(f"{operation} did not reach decision quorum")
    return values


__all__ = ["bounded_consensus", "compact_image"]
