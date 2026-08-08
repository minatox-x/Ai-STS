"""
Threaded downloader supporting pause/resume/cancel/retry/checksum verification,
used by the Model Manager UI. Uses HTTP Range requests for resume.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests


@dataclass
class DownloadProgress:
    downloaded_bytes: int
    total_bytes: int
    status: str  # "downloading" | "paused" | "completed" | "cancelled" | "error" | "verifying"
    error_message: Optional[str] = None

    @property
    def fraction(self) -> float:
        return (self.downloaded_bytes / self.total_bytes) if self.total_bytes else 0.0


class DownloadTask:
    """
    One file download. Not started until .start() is called.
    Progress is reported via the `on_progress` callback (called from the worker thread --
    callers updating a Qt UI must marshal back to the main thread, e.g. via a Signal).
    """

    CHUNK_SIZE = 1024 * 512

    def __init__(
        self,
        url: str,
        dest_path: Path,
        expected_sha256: Optional[str] = None,
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
        max_retries: int = 3,
    ):
        self.url = url
        self.dest_path = Path(dest_path)
        self.expected_sha256 = expected_sha256
        self.on_progress = on_progress or (lambda p: None)
        self.max_retries = max_retries

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._partial_path = self.dest_path.with_suffix(self.dest_path.suffix + ".part")

    def start(self) -> None:
        self.dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _run(self) -> None:
        attempt = 0
        while attempt <= self.max_retries:
            try:
                self._download_once()
                self._verify_and_finalize()
                return
            except _Cancelled:
                self.on_progress(DownloadProgress(0, 0, "cancelled"))
                return
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    self.on_progress(DownloadProgress(0, 0, "error", error_message=str(e)))
                    return
                time.sleep(min(2 ** attempt, 30))

    def _download_once(self) -> None:
        resume_from = self._partial_path.stat().st_size if self._partial_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

        with requests.get(self.url, headers=headers, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) + resume_from
            mode = "ab" if resume_from else "wb"
            downloaded = resume_from
            with open(self._partial_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=self.CHUNK_SIZE):
                    if self._cancel_event.is_set():
                        raise _Cancelled()
                    while self._pause_event.is_set():
                        self.on_progress(DownloadProgress(downloaded, total, "paused"))
                        if self._cancel_event.is_set():
                            raise _Cancelled()
                        time.sleep(0.25)
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.on_progress(DownloadProgress(downloaded, total, "downloading"))

    def _verify_and_finalize(self) -> None:
        if self.expected_sha256:
            self.on_progress(DownloadProgress(0, 0, "verifying"))
            h = hashlib.sha256()
            with open(self._partial_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest().lower() != self.expected_sha256.lower():
                self._partial_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Checksum mismatch after download -- the file is likely corrupted "
                    "or the source changed. Deleted the partial file; please retry."
                )
        self._partial_path.replace(self.dest_path)
        self.on_progress(DownloadProgress(self.dest_path.stat().st_size, self.dest_path.stat().st_size, "completed"))


class _Cancelled(Exception):
    pass
