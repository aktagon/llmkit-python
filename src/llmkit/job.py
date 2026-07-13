"""Job engine (ADR-062 / ADR-063) — the ONE shared poll runtime for llmkit's
async, poll-until-done capabilities. Mirror of go/job.go.

Slice 1 migrates batch + transcription onto it (video lands in slice 2).

Four "poll"-family names, kept deliberately distinct (glossary):
  - ``poll()`` (public handle method — BatchHandle.poll / TranscriptionHandle.poll):
    exactly one provider round-trip, normalized, NO loop (ADR-063 POLL-001).
  - ``poll_job`` / ``poll_job_async`` — the internal engine: the bounded loop over
    ``poll_once`` that owns the deadline backstop and the monotonic
    Running -> (Succeeded | Failed) state machine. The single writer of job state.
  - ``poll_once`` — one engine iteration (poll -> classify -> result-when-Succeeded).
    ``poll()`` IS ``poll_once`` made public; ``wait`` IS ``poll_job`` (a loop over it).
  - ``PollBody`` — the once-decoded provider poll response; confines the untyped JSON
    leaf so no ``dict[str, Any]`` crosses an adapter signature (S04).
  - the ``poll`` adapter seam performs the round-trip and returns a ``PollBody``.

The engine is generic on the result type ``T`` so no bare ``Any`` crosses the seam
(CLAUDE.md concrete-types rule; ADR-062 H1 typed-waist fix).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from .errors import APIError, PollTimeoutError
from .paths import extract_path

T = TypeVar("T")


class JobState(Enum):
    """Lifecycle state of an async job. PUBLIC because it is what ``poll``
    returns (ADR-063 POLL-004). The lifecycle is monotonic —
    RUNNING -> (SUCCEEDED | FAILED) — because ``poll_job`` returns on the FIRST
    terminal classification and no state is stored that could regress.

    There is deliberately NO ``UNKNOWN`` member (ADR-063 refinements 2): every
    ``JobStatus`` is constructed with an explicit state, so an ``UNKNOWN`` would
    be a public state the library never returns.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class JobFailure:
    """Normalized failure detail carried by a FAILED status. ONE terminal, not a
    taxonomy (ADR-062 refinements 1): the raw provider status, an optional
    provider error message, and a ``timed_out`` flag."""

    # status is the raw provider status string that classified as failure
    # (OpenAI batch failed/expired/cancelled; AssemblyAI "error"). Empty when the
    # failure is the engine's deadline backstop firing.
    status: str = ""

    # message is the provider error message when the provider reports one
    # (AssemblyAI's top-level "error"); empty otherwise.
    message: str = ""

    # timed_out is True iff this failure is the engine's deadline backstop, not a
    # provider-reported terminal.
    timed_out: bool = False


@dataclass
class JobStatus(Generic[T]):
    """Normalized result of a single ``poll`` (ADR-063 POLL-001): the state plus
    the result XOR the failure cause — never a raw provider payload. ``result`` is
    set iff ``state is SUCCEEDED``; ``cause`` is set iff ``state is FAILED``."""

    state: JobState
    result: T | None = None
    cause: JobFailure | None = None
    raw_status: str = ""


@dataclass
class LifecycleConfig:
    """Config half of the engine seam: the classification facts (status path +
    done / error value sets + the error-message path) and the poll cadence. Each
    capability assembles it from its own generated facts."""

    # noun labels the capability in the failure error string ("transcription",
    # "batch") so a FAILED terminal reads "<noun> failed: <message>".
    noun: str

    # status_path is the dotted path to the status string in the poll body.
    status_path: str

    # done_values are the status strings marking terminal success (precedence over
    # error_values).
    done_values: tuple[str, ...] = ()

    # error_values are the status strings marking terminal failure. An empty set
    # means "no failure terminal" — a stuck job then terminates at the deadline
    # backstop rather than mislabelling a FAILED terminal.
    error_values: tuple[str, ...] = ()

    # error_message_path is the dotted path to a provider error message, surfaced
    # in JobFailure.message. Empty = no message extraction.
    error_message_path: str = ""

    # poll_interval is the cadence between polls (seconds).
    poll_interval: float = 2.0

    # poll_timeout is the OVERALL wall-clock backstop for the poll LOOP (seconds)
    # — NOT a per-HTTP-request timeout (S05). Zero = no backstop.
    poll_timeout: float = 0.0


@dataclass
class PollBody:
    """The once-decoded provider poll response (S04). Confines the untyped JSON
    leaf: classification reads a config path via ``status``; ``result`` reads the
    decoded tree. No adapter signature carries a bare ``dict[str, Any]``."""

    raw: dict[str, Any] = field(default_factory=dict)

    def status(self, path: str) -> str:
        """Return the string at the given dotted path, or "" if absent."""
        return extract_path(self.raw, path)


@dataclass
class _Classification:
    """What classify returns: the state plus the failure detail when FAILED.
    Internal — the public boundary is JobState."""

    state: JobState
    failure: JobFailure | None = None
    raw_status: str = ""


class JobAdapter(Protocol[T]):
    """The capability seams the engine cannot share (ADR-062 difference table).
    ``result`` is the capability tail and MAY perform a second network hop (batch's
    output_file_id -> GET /content), so the adapter closes over the http client +
    provider config."""

    def config(self) -> LifecycleConfig: ...

    def poll(self) -> PollBody: ...

    def classify(self, body: PollBody) -> _Classification: ...

    def result(self, body: PollBody) -> T: ...


def classify_by_config(lc: LifecycleConfig, body: PollBody) -> _Classification:
    """Shared config-driven default classifier (ADR-062). Precedence
    done > error > running: a status in ``done_values`` -> SUCCEEDED; in
    ``error_values`` -> FAILED (message extracted); in NEITHER set -> RUNNING (poll
    on, bounded by the backstop). So an unmodeled/new terminal degrades to a
    bounded timeout — never a false success and never a false failure of a live
    job."""
    status = body.status(lc.status_path)
    for done in lc.done_values:
        if status == done:
            return _Classification(state=JobState.SUCCEEDED, raw_status=status)
    for err in lc.error_values:
        if status == err:
            failure = JobFailure(status=status)
            if lc.error_message_path:
                failure.message = body.status(lc.error_message_path)
            return _Classification(
                state=JobState.FAILED, failure=failure, raw_status=status
            )
    return _Classification(state=JobState.RUNNING, raw_status=status)


def job_failed_error(noun: str, cause: JobFailure) -> APIError:
    """The error a failed ``poll_job`` raises. Its message preserves each
    capability's surface via ``noun`` — transcription's "transcription failed:
    <msg>" (S02). A provider failure is an APIError, NOT a PollTimeoutError —
    the deadline backstop is the only thing that raises PollTimeoutError."""
    detail = cause.message or cause.status
    message = f"{noun} failed: {detail}" if detail else f"{noun} failed"
    return APIError(message=message, status_code=0)


def _poll_timeout_error(noun: str) -> PollTimeoutError:
    """The deadline-backstop timeout — teaches the async / handle pattern
    (ADR-062 OQ-1): a long job should be polled across requests, not synchronously
    blocked on."""
    return PollTimeoutError(
        message=(
            f"{noun} poll: timed out; the job may still be running — poll the "
            f"handle across requests, or raise the deadline with poll_deadline"
        )
    )


def poll_once(adapter: JobAdapter[T]) -> JobStatus[T]:
    """One engine iteration: poll -> classify -> (on success) the capability
    result tail, including any second network hop. ``poll``'s body and
    ``poll_job``'s per-iteration step — no loop, no deadline (ADR-063 POLL-001:
    exactly one round-trip). Blocking; run under ``asyncio.to_thread`` from the
    async surface."""
    body = adapter.poll()
    c = adapter.classify(body)
    st: JobStatus[T] = JobStatus(state=c.state, raw_status=c.raw_status)
    if c.state is JobState.SUCCEEDED:
        st.result = adapter.result(body)
    elif c.state is JobState.FAILED:
        st.cause = c.failure
    return st


def _interval_and_deadline(lc: LifecycleConfig) -> tuple[float, float | None]:
    interval = lc.poll_interval if lc.poll_interval > 0 else 2.0
    deadline = time.monotonic() + lc.poll_timeout if lc.poll_timeout > 0 else None
    return interval, deadline


def poll_job(adapter: JobAdapter[T]) -> T:
    """The shared engine (ADR-062) — synchronous. Loops ``poll_once`` on the
    configured cadence until the first terminal classification or the deadline
    backstop. Used by the blocking free-function ``wait_batch``. Monotonicity is a
    consequence of returning on the first terminal, not of any stored state."""
    lc = adapter.config()
    interval, deadline = _interval_and_deadline(lc)
    while True:
        st = poll_once(adapter)
        if st.state is JobState.SUCCEEDED:
            assert st.result is not None
            return st.result
        if st.state is JobState.FAILED:
            assert st.cause is not None
            raise job_failed_error(lc.noun, st.cause)
        if deadline is not None and time.monotonic() > deadline:
            raise _poll_timeout_error(lc.noun)
        time.sleep(interval)


async def poll_job_async(adapter: JobAdapter[T]) -> T:
    """The shared engine — asynchronous. Same state machine as ``poll_job``, but
    the between-poll wait is a cancellable ``asyncio.sleep`` so
    ``asyncio.CancelledError`` propagates (S06); each blocking round-trip runs in a
    worker thread via ``asyncio.to_thread``."""
    lc = adapter.config()
    interval, deadline = _interval_and_deadline(lc)
    while True:
        st = await asyncio.to_thread(poll_once, adapter)
        if st.state is JobState.SUCCEEDED:
            assert st.result is not None
            return st.result
        if st.state is JobState.FAILED:
            assert st.cause is not None
            raise job_failed_error(lc.noun, st.cause)
        if deadline is not None and time.monotonic() > deadline:
            raise _poll_timeout_error(lc.noun)
        await asyncio.sleep(interval)  # cancellable — CancelledError propagates


async def poll_engine_once(adapter: JobAdapter[T]) -> JobStatus[T]:
    """The public ``poll`` primitive (ADR-063 POLL-001): exactly one normalized
    round-trip, no loop, never times out. Runs the blocking iteration in a worker
    thread so the async caller is not blocked."""
    return await asyncio.to_thread(poll_once, adapter)
