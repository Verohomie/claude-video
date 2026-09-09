"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. We only support English, so the request must stay
bounded to the English-only pattern.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch, *, output: str = "") -> list[list[str]]:
    """Stub every subprocess entry point inside download.py and record each argv.

    Both ``run`` (fetch_captions) and ``Popen`` (download_url, which tees yt-dlp's
    output so it can diagnose a failure) must be stubbed — leaving either live
    turns these into real network calls against YouTube.
    """
    # download.py resolves the best yt-dlp before building argv, which shells
    # out to probe each candidate and caches the result under ~/.config/watch.
    # Stub it: these tests are about the argv, and must neither record the
    # probe's own calls nor touch the developer's real config dir.
    monkeypatch.setattr(download, "resolve_ytdlp", lambda: "yt-dlp")
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    class _Popen:
        def __init__(self, cmd, *args, **kwargs):
            calls.append(list(cmd))
            self.stdout = io.StringIO(output)

        def wait(self):
            return 0

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.subprocess, "Popen", _Popen)
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_english_only(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert all(t.startswith("en") for t in tokens), f"sub-langs must be English-only, got {langs!r}"


def test_fetch_captions_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))


def test_download_url_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))
