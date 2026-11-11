"""Persistent whisper.cpp process for a meeting's span-by-span ASR.

whisper-cli reloads the 3 GB model on every call, ~0.9 s each, and the
multilingual path makes hundreds of calls per meeting (one per span plus one
per language-ID chunk): on a 14.8-min meeting, 120 of the 188 s spent in
whisper-cli were model loads. `whisper-batch` (tools/whisper-batch) keeps the
model loaded and answers the same requests over a pipe, on a fresh
whisper_state per request so the output is whisper-cli's, call for call (the
comment at the top of whisper_batch.cpp explains why a reused state is not).

Nothing depends on it: the caller falls back to whisper-cli when the binary is
missing, fails to start, or stops answering, and since both produce the same
text the fallback does not show in the result.
"""
from __future__ import annotations

import collections
import json
import queue
import subprocess
import threading
import time
from pathlib import Path


class WhisperBatchError(RuntimeError):
    """The process is unusable: it did not start, died, hung, or broke protocol."""


class WhisperBatchRequestError(RuntimeError):
    """The process is fine but rejected this one request."""


class WhisperBatchProcess:
    def __init__(self, binary: Path, model: Path, ready_timeout: float = 180.0):
        self.binary = binary
        self.model = model
        self.ready_timeout = ready_timeout
        self.calls = 0
        self.busy_seconds = 0.0
        self.load_seconds: float | None = None
        # From the handshake: this build can put a second language tag in the
        # decoder prompt (needs patches/whisper-cpp-second-language.patch).
        self.second_language = False
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        self._lock = threading.Lock()
        self._next_id = 0

    def spawn(self) -> None:
        """Start the process without waiting for the model: the load overlaps
        with whatever the caller does next, and the first request waits for it."""
        self._proc = subprocess.Popen(
            [str(self.binary), "-m", str(self.model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(target=self._pump, args=(self._proc.stdout, self._lines.put), daemon=True).start()
        threading.Thread(target=self._pump, args=(self._proc.stderr, self._stderr_tail.append), daemon=True).start()

    @staticmethod
    def _pump(stream, sink) -> None:
        for line in stream:
            sink(line.rstrip("\n"))
        sink(None)

    def stderr_tail(self) -> str:
        """Last lines the process wrote to stderr. read_audio_data prints one
        per request, so they are only worth showing when something failed."""
        return " | ".join(line for line in list(self._stderr_tail)[-5:] if line)

    def _answer(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise WhisperBatchError(f"no answer within {timeout:.0f}s") from None
        if line is None:
            code = self._proc.poll() if self._proc else None
            raise WhisperBatchError(f"process exited (code {code})")
        try:
            return json.loads(line)
        except ValueError as exc:
            raise WhisperBatchError(f"unreadable answer: {line[:200]!r}") from exc

    def request(self, payload: dict, timeout: float = 600.0) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise WhisperBatchError("not running")
            if self.load_seconds is None:
                ready = self._answer(self.ready_timeout)
                if not ready.get("ready"):
                    raise WhisperBatchError(f"unexpected handshake: {ready}")
                self.load_seconds = float(ready.get("load_seconds") or 0.0)
                self.second_language = bool(ready.get("second_language"))
            self._next_id += 1
            started = time.perf_counter()
            try:
                proc.stdin.write(json.dumps({**payload, "id": self._next_id}) + "\n")
                proc.stdin.flush()
            except OSError as exc:
                raise WhisperBatchError(f"write failed: {exc}") from exc
            answer = self._answer(timeout)
            if answer.get("id") != self._next_id:
                raise WhisperBatchError(f"answer {answer.get('id')} to request {self._next_id}")
            self.busy_seconds += time.perf_counter() - started
            if "error" in answer:
                raise WhisperBatchRequestError(str(answer["error"]))
            self.calls += 1
            return answer

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait()
