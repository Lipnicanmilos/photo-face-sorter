"""Beh dlhých operácií (scan) na pozadí, aby si UI mohlo ťahať priebeh."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["running", "done", "error"]


@dataclass
class Job:
    """Stav jednej operácie na pozadí."""

    id: str
    kind: str
    status: JobStatus = "running"
    current: int = 0
    total: int = 0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobBusyError(RuntimeError):
    """Iná operácia práve beží."""


class JobManager:
    """Drží najviac jednu bežiacu úlohu - detekcia aj tak vyťaží CPU naplno."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._active_id: str | None = None

    def start(self, kind: str, target: Callable[[Job], dict[str, Any]]) -> Job:
        """Spustí `target` vo vlákne. Callback dostáva `Job` na hlásenie priebehu.

        Raises:
            JobBusyError: Ak už nejaká úloha beží.
        """
        with self._lock:
            if self._active_id is not None:
                raise JobBusyError("Iná operácia práve beží, počkaj na jej dokončenie.")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind)
            self._jobs[job.id] = job
            self._active_id = job.id

        def runner() -> None:
            try:
                job.result = target(job)
                job.status = "done"
                job.message = "Hotovo"
            except Exception as exc:  # noqa: BLE001 - chybu chceme ukázať v UI
                logger.exception("Úloha %s (%s) zlyhala", job.id, job.kind)
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                with self._lock:
                    self._active_id = None

        threading.Thread(target=runner, name=f"job-{job.kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    @property
    def active(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._active_id) if self._active_id else None
