"""Thread-safe task primitives shared by the Qt frontend and workers."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class TaskCancelledError(Exception):
    """Raised inside a worker after the user requests cancellation."""


class CancellationToken:
    """A tiny, thread-safe cancellation handle safe to call from the UI."""

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event or threading.Event()

    @property
    def event(self) -> threading.Event:
        return self._event

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TaskCancelledError()


@dataclass(frozen=True)
class TaskSnapshot:
    """Values captured from Qt widgets before a worker thread is started."""

    operation: str
    values: Mapping[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def background_creation_flags() -> int:
    """Keep child consoles hidden and preserve UI scheduling on Windows."""

    if os.name != "nt":
        return 0
    return int(
        getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    )


def terminate_process_tree(proc: subprocess.Popen, grace_seconds: float = 1.5) -> None:
    """Terminate a process and its descendants without blocking the Qt thread."""

    if proc is None or proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, grace_seconds),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
                check=False,
            )
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    try:
        proc.wait(timeout=grace_seconds)
    except Exception:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


class ProcessRegistry:
    """Tracks child processes and provides cancellation-aware waits."""

    def __init__(self, cancel_token: CancellationToken) -> None:
        self.cancel_token = cancel_token
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()

    def popen(self, args: Sequence[str], **kwargs: Any) -> subprocess.Popen:
        if "creationflags" not in kwargs:
            kwargs["creationflags"] = background_creation_flags()
        if os.name != "nt" and "start_new_session" not in kwargs:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(args, **kwargs)
        self.register(proc)
        return proc

    def register(self, proc: subprocess.Popen) -> subprocess.Popen:
        with self._lock:
            self._processes.add(proc)
        return proc

    def unregister(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        with self._lock:
            self._processes.discard(proc)

    def wait(
        self,
        proc: subprocess.Popen,
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> int:
        started = time.monotonic()
        try:
            while proc.poll() is None:
                if self.cancel_token.is_cancelled():
                    terminate_process_tree(proc)
                    raise TaskCancelledError()
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
                self.cancel_token.event.wait(poll_interval)
            return int(proc.returncode or 0)
        finally:
            if proc.poll() is not None:
                self.unregister(proc)

    def terminate(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        terminate_process_tree(proc)
        self.unregister(proc)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)
        terminators = [
            threading.Thread(target=self.terminate, args=(proc,), daemon=True)
            for proc in processes
        ]
        for thread in terminators:
            thread.start()
        deadline = time.monotonic() + 3.0
        for thread in terminators:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
