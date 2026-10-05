"""The Gemini engine: which videos may go to Google, and what comes back.

The routing tests are the ones that matter most. A wrong answer there is not a
crash — it is a private screen recording quietly uploaded to a third party, so
every source kind is pinned explicitly, in both directions.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import config
import gemini

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"
KEY = "AIzaTESTKEY0123456789"


# --- which videos may go to Google -------------------------------------------


@pytest.mark.parametrize("source", [
    "https://youtu.be/qSuCPooR3E4",
    "https://www.youtube.com/watch?v=qSuCPooR3E4",
    "https://youtube.com/watch?v=qSuCPooR3E4&t=30s",
    "https://m.youtube.com/watch?v=qSuCPooR3E4",
    "https://www.youtube.com/shorts/qSuCPooR3E4",
    "https://www.youtube.com/live/qSuCPooR3E4",
])
def test_auto_sends_youtube_to_gemini_when_a_key_exists(source):
    assert config.resolve_engine("auto", has_key=True, youtube=gemini.youtube_id(source) is not None) == "gemini"


@pytest.mark.parametrize("source", [
    "/Users/someone/Movies/obs-recording.mp4",
    "clip.mov",
    "https://vimeo.com/123456",
    "https://www.instagram.com/reel/abc/",
    "https://notyoutube.com/watch?v=qSuCPooR3E4",
    "https://youtube.com.evil.example/watch?v=qSuCPooR3E4",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/playlist?list=PL123",
])
def test_auto_keeps_everything_else_local_even_with_a_key(source):
    """Own recordings and other sites never leave the Mac without an explicit ask."""
    assert config.resolve_engine("auto", has_key=True, youtube=gemini.youtube_id(source) is not None) == "local"


def test_auto_without_a_key_is_local():
    assert config.resolve_engine("auto", has_key=False, youtube=True) == "local"


def test_explicit_local_wins_over_a_key():
    assert config.resolve_engine("local", has_key=True, youtube=True) == "local"


def test_explicit_gemini_sends_anything():
    assert config.resolve_engine("gemini", has_key=True, youtube=False) == "gemini"


def test_explicit_gemini_without_a_key_says_why():
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        config.resolve_engine("gemini", has_key=False, youtube=True)


def test_youtube_links_are_normalised_to_the_watch_form():
    assert gemini.youtube_uri("https://youtu.be/qSuCPooR3E4?si=x") == "https://www.youtube.com/watch?v=qSuCPooR3E4"
    assert gemini.youtube_uri("/tmp/a.mp4") is None


# --- settings -----------------------------------------------------------------


@pytest.fixture
def no_config_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "absent.env")
    for name in ("WATCH_ENGINE", "WATCH_GEMINI_MODEL", "WATCH_GEMINI_TIMEOUT", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_engine_defaults(no_config_file):
    cfg = config.get_config()
    assert cfg["engine"] == "auto"
    assert cfg["gemini_model"] == "gemini-3.7-flash"
    assert cfg["gemini_timeout"] == 600.0
    assert config.load_gemini_key() is None


def test_bad_engine_and_timeout_fall_back(no_config_file, monkeypatch):
    monkeypatch.setenv("WATCH_ENGINE", "telepathy")
    monkeypatch.setenv("WATCH_GEMINI_TIMEOUT", "inf")
    cfg = config.get_config()
    assert cfg["engine"] == "auto"
    assert cfg["gemini_timeout"] == 600.0


def test_key_is_read_from_the_config_file(no_config_file, monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"GEMINI_API_KEY={KEY}\nWATCH_ENGINE=local\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    assert config.load_gemini_key() == KEY
    assert config.get_config()["engine"] == "local"


# --- the API exchange ---------------------------------------------------------


class _Response:
    def __init__(self, body: dict | bytes, status: int = 200, headers: dict | None = None):
        self.status = status
        self.headers = headers or {}
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answer(text="At 01:02 he says hello.", tokens=1234):
    return {"status": "completed", "usage": {"total_tokens": tokens},
            "steps": [{"type": "thought"},
                      {"type": "model_output", "content": [{"type": "text", "text": text}]}]}


def test_ask_sends_question_and_reads_the_answer(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["key"] = request.get_header("X-goog-api-key")
        sent["body"] = json.loads(request.data)
        return _Response(_answer())

    monkeypatch.setattr(gemini, "urlopen", fake_urlopen)
    out = gemini.ask({"uri": "https://www.youtube.com/watch?v=qSuCPooR3E4"}, "who is speaking?",
                     model="gemini-3.7-flash", key=KEY)
    assert out["text"] == "At 01:02 he says hello."
    assert out["total_tokens"] == 1234
    assert out["processing"] == "agentic, whole video"
    assert sent["key"] == KEY and KEY not in sent["url"], "key goes in a header, never the URL"
    video, prompt = sent["body"]["input"]
    assert video["processing"] == "agentic"
    assert prompt["text"].startswith("who is speaking?")


def test_a_range_becomes_a_static_clip(monkeypatch):
    sent = {}
    monkeypatch.setattr(gemini, "urlopen",
                        lambda request, timeout: sent.setdefault("b", json.loads(request.data)) and _Response(_answer()))
    out = gemini.ask({"uri": "u"}, None, model="m", key=KEY, start=65.4, end=90.2)
    video = sent["b"]["input"][0]
    assert video["processing"] == {"type": "static", "start_offset": "65s", "end_offset": "91s"}
    assert out["processing"] == "static clip 01:05–01:31"


def test_no_question_asks_for_a_summary(monkeypatch):
    sent = {}
    monkeypatch.setattr(gemini, "urlopen",
                        lambda request, timeout: sent.setdefault("b", json.loads(request.data)) and _Response(_answer()))
    gemini.ask({"uri": "u"}, "  ", model="m", key=KEY)
    assert sent["b"]["input"][1]["text"].startswith("Give a thorough summary")


@pytest.mark.parametrize("code,message,category", [
    (400, "API key not valid. Please pass a valid API key.", "auth"),
    (403, "permission denied", "auth"),
    (429, "Resource has been exhausted", "quota"),
    (503, "overloaded", "service"),
    (400, "video too long", "rejected"),
])
def test_http_errors_are_categorised_and_never_show_the_key(monkeypatch, code, message, category):
    def fake_urlopen(request, timeout):
        body = json.dumps({"error": {"message": f"{message} key={KEY}"}}).encode()
        raise HTTPError(request.full_url, code, "x", {}, io.BytesIO(body))

    monkeypatch.setattr(gemini, "urlopen", fake_urlopen)
    with pytest.raises(gemini.GeminiError) as info:
        gemini.ask({"uri": "u"}, None, model="m", key=KEY)
    assert info.value.category == category
    assert KEY not in str(info.value)
    assert "[redacted]" in str(info.value)


def test_network_failure_is_reported_as_network(monkeypatch):
    def fake_urlopen(request, timeout):
        raise URLError("nodename nor servname provided")

    monkeypatch.setattr(gemini, "urlopen", fake_urlopen)
    with pytest.raises(gemini.GeminiError) as info:
        gemini.ask({"uri": "u"}, None, model="m", key=KEY)
    assert info.value.category == "network"


def test_an_answer_with_no_text_is_an_error(monkeypatch):
    monkeypatch.setattr(gemini, "urlopen",
                        lambda request, timeout: _Response({"status": "failed", "steps": []}))
    with pytest.raises(gemini.GeminiError, match="no answer text"):
        gemini.ask({"uri": "u"}, None, model="m", key=KEY)


def test_upload_streams_waits_and_returns_the_file(monkeypatch, tmp_path):
    clip = tmp_path / "rec.mp4"
    clip.write_bytes(b"\0" * 2048)
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.get_method() + " " + request.full_url)
        if request.full_url.endswith("/upload/v1beta/files"):
            return _Response(b"{}", headers={"X-Goog-Upload-URL": "https://upload.example/session"})
        if request.full_url == "https://upload.example/session":
            return _Response({"file": {"name": "files/abc", "uri": "https://g/files/abc", "state": "PROCESSING"}})
        return _Response({"name": "files/abc", "uri": "https://g/files/abc", "state": "ACTIVE",
                          "mimeType": "video/mp4"})

    monkeypatch.setattr(gemini, "urlopen", fake_urlopen)
    out = gemini.upload_file(clip, KEY, sleep=lambda s: None)
    assert out == {"name": "files/abc", "uri": "https://g/files/abc", "mime_type": "video/mp4"}
    assert calls[-1].startswith("GET ") and calls[-1].endswith("/v1beta/files/abc")


def test_a_failed_delete_is_a_warning_not_an_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(gemini, "urlopen", fake_urlopen)
    warning = gemini.delete_file("files/abc", KEY)
    assert "48 hours" in warning and KEY not in warning


# --- end to end: a local file never reaches Gemini on auto --------------------


def test_watch_on_a_local_file_stays_local_with_a_key_present(cut_clip: Path, tmp_path):
    """Even with a key set, auto must not touch the network for a local file.

    A bogus proxy makes any attempt to reach Google fail loudly, so a pass
    proves the local engine ran rather than Gemini failing quietly.
    """
    env = dict(os.environ, GEMINI_API_KEY=KEY, WATCH_ENGINE="auto",
               HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")
    env.pop("WATCH_DETAIL", None)
    proc = subprocess.run(
        [sys.executable, str(WATCH), str(cut_clip), "--no-whisper", "--detail", "efficient"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Gemini" not in proc.stdout
    assert "**Frames:**" in proc.stdout


# --- the report a Gemini run prints -------------------------------------------


def _args(**over):
    import argparse
    base = dict(source="https://youtu.be/qSuCPooR3E4", question="what tool is shown?",
                max_frames=None, resolution=None, fps=None, detail=None, timestamps=None,
                whisper=None, no_whisper=False, no_dedup=False, quality=None,
                confirm_cost=False)
    base.update(over)
    return argparse.Namespace(**base)


CFG = {"gemini_model": "gemini-3.7-flash", "gemini_timeout": 600.0, "quality": "1080"}


@pytest.fixture(autouse=True)
def no_metadata_lookups(monkeypatch):
    """The cost check reads a video's length first; keep that off the network."""
    import watch
    monkeypatch.setattr(watch, "fetch_duration", lambda url: 510.0)


def test_report_labels_the_answer_as_geminis(monkeypatch, capsys):
    import watch
    seen = {}

    def fake_ask(video, question, **kw):
        seen.update(video=video, question=question)
        return {"text": "At 12:34 the tool is Tiktokenizer.", "model": "m",
                "processing": "agentic, whole video", "total_tokens": 900}

    monkeypatch.setattr(watch.gemini, "ask", fake_ask)
    monkeypatch.setattr(watch.gemini, "upload_file", lambda *a, **k: pytest.fail("YouTube must not upload"))
    assert watch.run_gemini(_args(detail="balanced"), CFG, KEY, None, None) == 0
    out = capsys.readouterr().out
    assert seen["video"] == {"uri": "https://www.youtube.com/watch?v=qSuCPooR3E4"}
    assert seen["question"] == "what tool is shown?"
    assert "Answer (from Gemini)" in out and "not frames you viewed yourself" in out
    assert "nothing downloaded or uploaded" in out
    assert "--detail" in out  # listed as ignored, not silently dropped
    assert KEY not in out


def test_a_failure_reports_and_does_not_fall_back(monkeypatch, capsys):
    import watch

    def fake_ask(*a, **k):
        raise gemini.GeminiError("quota", "HTTP 429")

    monkeypatch.setattr(watch.gemini, "ask", fake_ask)
    monkeypatch.setattr(watch, "download", lambda *a, **k: pytest.fail("no fallback download"))
    assert watch.run_gemini(_args(), CFG, KEY, None, None) == 1
    out = capsys.readouterr().out
    assert "Gemini failed" in out and "--engine local" in out


def test_an_explicit_local_file_is_uploaded_then_deleted(monkeypatch, capsys, tmp_path):
    import watch
    clip = tmp_path / "rec.mp4"
    clip.write_bytes(b"\0" * 10)
    events = []
    monkeypatch.setattr(watch, "download", lambda src, out, **k: {"video_path": str(clip)})
    monkeypatch.setattr(watch.gemini, "upload_file",
                        lambda p, k: events.append("upload") or {"name": "files/x", "uri": "u", "mime_type": "video/mp4"})
    monkeypatch.setattr(watch.gemini, "ask", lambda *a, **k: events.append("ask") or
                        {"text": "ok", "model": "m", "processing": "agentic, whole video", "total_tokens": None})
    monkeypatch.setattr(watch.gemini, "delete_file", lambda name, key: events.append("delete") or None)
    assert watch.run_gemini(_args(source=str(clip)), CFG, KEY, None, None) == 0
    assert events == ["upload", "ask", "delete"]
    assert "uploaded to Google, and deleted" in capsys.readouterr().out


# --- the cost check -----------------------------------------------------------


def test_a_long_video_is_priced_and_held(monkeypatch, capsys):
    import watch
    monkeypatch.setattr(watch, "fetch_duration", lambda url: 3.5 * 3600)
    monkeypatch.setattr(watch.gemini, "ask", lambda *a, **k: pytest.fail("must not be sent"))
    assert watch.run_gemini(_args(), {**CFG, "gemini_confirm_minutes": 30.0}, KEY, None, None) == 2
    out = capsys.readouterr().out
    assert "Cost check — not sent yet" in out
    assert "3:30:00" in out and "--confirm-cost" in out
    assert "$" in out


def test_confirm_cost_sends_it(monkeypatch, capsys):
    import watch
    monkeypatch.setattr(watch, "fetch_duration", lambda url: 3.5 * 3600)
    monkeypatch.setattr(watch.gemini, "ask", lambda *a, **k: {
        "text": "ok", "model": "m", "processing": "agentic, whole video", "total_tokens": 2_000_000})
    assert watch.run_gemini(_args(confirm_cost=True), {**CFG, "gemini_confirm_minutes": 30.0},
                            KEY, None, None) == 0
    out = capsys.readouterr().out
    assert "Estimated cost:" in out and "2,000,000" in out


def test_a_short_range_of_a_long_video_goes_straight_through(monkeypatch, capsys):
    """--start/--end price the clip, not the whole film, and need no lookup."""
    import watch
    monkeypatch.setattr(watch, "fetch_duration", lambda url: pytest.fail("range is already known"))
    monkeypatch.setattr(watch.gemini, "ask", lambda *a, **k: {
        "text": "ok", "model": "m", "processing": "static clip", "total_tokens": None})
    assert watch.run_gemini(_args(), {**CFG, "gemini_confirm_minutes": 30.0}, KEY, 600.0, 900.0) == 0
    assert "for 05:00 of video" in capsys.readouterr().out


def test_an_unknown_length_is_sent_and_says_so(monkeypatch, capsys):
    """Fail open: a metadata refusal must not block the engine built for refusals."""
    import watch
    monkeypatch.setattr(watch, "fetch_duration", lambda url: None)
    monkeypatch.setattr(watch.gemini, "ask", lambda *a, **k: {
        "text": "ok", "model": "m", "processing": "agentic, whole video", "total_tokens": 5000})
    assert watch.run_gemini(_args(), CFG, KEY, None, None) == 0
    assert "length could not be read" in capsys.readouterr().out


def test_estimate_matches_the_measured_rate():
    # 8.5 minutes measured at 113,794 tokens; the estimate should be in that region.
    assert 100_000 < gemini.estimate_tokens(510) < 125_000


def test_price_doubles_on_the_first_of_2027():
    import datetime
    assert gemini.price_per_million(datetime.date(2026, 12, 31)) == 0.75
    assert gemini.price_per_million(datetime.date(2027, 1, 1)) == 1.50


def test_money_reads_plainly():
    assert gemini.dollars(0.085) == "8.5¢"
    assert gemini.dollars(2.05) == "$2.05"
    assert gemini.dollars(0.0) == "0.1¢"


def test_confirm_minutes_setting(no_config_file, monkeypatch):
    assert config.get_config()["gemini_confirm_minutes"] == 30.0
    monkeypatch.setenv("WATCH_GEMINI_CONFIRM_MINUTES", "10")
    assert config.get_config()["gemini_confirm_minutes"] == 10.0
    monkeypatch.setenv("WATCH_GEMINI_CONFIRM_MINUTES", "nan")
    assert config.get_config()["gemini_confirm_minutes"] == 30.0
