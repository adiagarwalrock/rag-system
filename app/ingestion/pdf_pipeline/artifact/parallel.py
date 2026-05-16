from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Callable, TypeVar

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class ParallelExecutionConfig:
    max_parallel_requests: int = 4


class ParallelExecutor:
    def __init__(self, config: ParallelExecutionConfig | None = None) -> None:
        self._config = config or ParallelExecutionConfig()

    def worker_count(self, job_count: int, max_workers: int | None = None) -> int:
        limit = max_workers or self._config.max_parallel_requests
        return max(1, min(job_count, limit))

    def run(
        self,
        jobs: list[_Job],
        runner: Callable[[_Job], _Result],
        *,
        max_workers: int | None = None,
        timeout_seconds: float | None = None,
    ) -> list[tuple[_Job, _Result | None, Exception | None]]:
        if not jobs:
            return []

        workers = self.worker_count(len(jobs), max_workers=max_workers)

        def _wrapped(job: _Job) -> tuple[_Job, _Result | None, Exception | None]:
            try:
                return (job, runner(job), None)
            except Exception as exc:
                return (job, None, exc)

        if workers == 1 and timeout_seconds is None:
            return [_wrapped(job) for job in jobs]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(runner, job) for job in jobs]
            results: list[tuple[_Job, _Result | None, Exception | None]] = []
            for job, future in zip(jobs, futures):
                try:
                    result = future.result(timeout=timeout_seconds)
                    results.append((job, result, None))
                except FuturesTimeoutError:
                    future.cancel()
                    timeout_error = TimeoutError(
                        f"job exceeded timeout ({timeout_seconds}s)"
                    )
                    results.append((job, None, timeout_error))
                except Exception as exc:
                    results.append((job, None, exc))
            return results


_PARALLEL_EXECUTOR = ParallelExecutor()


def parallel_worker_count(
    job_count: int,
    *,
    max_workers: int | None = None,
) -> int:
    return _PARALLEL_EXECUTOR.worker_count(job_count, max_workers=max_workers)


def run_parallel_jobs(
    jobs: list[_Job],
    runner: Callable[[_Job], _Result],
    *,
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> list[tuple[_Job, _Result | None, Exception | None]]:
    return _PARALLEL_EXECUTOR.run(
        jobs,
        runner,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )
