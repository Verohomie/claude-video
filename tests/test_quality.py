"""Download quality ceiling: the --quality option and WATCH_QUALITY setting.

The skill used to hard-code height<=720 with no way to raise it, so a 1080p
source was always fetched at 720p and a screen-recorded tutorial arrived
unreadable before frame extraction even started.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402
import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    class _Popen:
        def __init__(self, cmd, *args, **kwargs):
            calls.append(list(cmd))
            self.stdout = io.StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(download.subprocess, "Popen", _Popen)
    return calls


def _fmt(argv: list[str]) -> str:
    return argv[argv.index("-f") + 1]


def test_default_quality_is_1080_not_720():
    """The regression this whole option exists for."""
    assert config.DEFAULT_QUALITY == "1080"
    assert "height<=1080" in download.format_selector()
    assert "height<=720" not in download.format_selector()


def test_format_selector_honours_a_ceiling():
    assert download.format_selector("720") == (
        "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    )


def test_format_selector_best_has_no_ceiling():
    selector = download.format_selector("best")
    assert "height" not in selector


def test_format_selector_falls_back_above_the_ceiling():
    """A ceiling is a preference: a source that exists ONLY above it must still
    download rather than failing outright."""
    assert download.format_selector("720").endswith("/bv+ba/b")


def test_audio_only_ignores_quality():
    assert download.format_selector("2160", audio_only=True) == "ba/bestaudio"


def test_download_url_passes_quality_through(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):  # no real file lands, which is fine
        download.download_url(URL, tmp_path / "d", quality="1440")
    assert "height<=1440" in _fmt(calls[0])


def test_config_reads_watch_quality(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_QUALITY", "1440")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["quality"] == "1440"


def test_config_accepts_a_p_suffix(monkeypatch, tmp_path):
    """`WATCH_QUALITY=1080p` is the obvious thing to type."""
    monkeypatch.setenv("WATCH_QUALITY", "1080p")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["quality"] == "1080"


def test_config_rejects_nonsense_quality(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_QUALITY", "enormous")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["quality"] == config.DEFAULT_QUALITY
