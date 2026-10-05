#!/usr/bin/env python3
"""Gemini engine: Google's model watches the video and answers the question.

The agent then relays a written answer instead of reading frames itself — far
cheaper in its own tokens, and for YouTube nothing is downloaded at all, so the
host's download blocks never come into it. The price is that the agent works
from Google's description rather than its own eyes, which the report says.

Standard library only. The key travels in the x-goog-api-key header, never in a
URL, and is scrubbed from every message this module produces.
"""
from __future__ import annotations

import json
import math
import mimetypes
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

API = "https://generativelanguage.googleapis.com"
INTERACTIONS = f"{API}/v1beta/interactions"

_HINTS = {
    "auth": "check GEMINI_API_KEY (create one at https://aistudio.google.com/apikey)",
    "quota": "the key hit its rate limit; wait a minute and retry",
    "rejected": "Gemini refused this request; the video may be private, age-restricted, or too long",
    "service": ("Gemini had a server error; retry shortly. If 'high demand' never clears, check "
                "the key's project has billing on: free-tier keys are refused all video"),
    "network": "could not reach generativelanguage.googleapis.com; check the network or proxy",
    "response": "Gemini returned something this version cannot read",
    "upload": "the video could not be uploaded to or processed by Google",
}

DEFAULT_QUESTION = (
    "Give a thorough summary of this video: what is shown, any on-screen text or code, "
    "and what is said, in chronological order."
)
PROMPT_SUFFIX = (
    "\n\nCite an MM:SS (or H:MM:SS) timestamp for every claim about a specific moment. "
    "Separate what is seen from what is heard when it matters. Spell names exactly as "
    "they appear on screen, and say when a name was only heard. If the video does not "
    "show or say something needed to answer, say so plainly instead of guessing."
)

_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# What a run costs, so a long video is never sent by accident. Measured, not
# taken from Google's benchmark slide: that slide implied ~700 tokens per
# minute of video, and the first real runs used ~12,000-13,400 (113,794 tokens
# for an 8.5-minute interview, 3,917 for 19 seconds). Re-measure if the model
# or Google's processing changes; the report prints the real count every run.
TOKENS_PER_VIDEO_MINUTE = 13_000
# Paid-tier price per million tokens (text, image, video and audio input are
# priced alike). Google doubles it on 2027-01-01. Output is a few thousand
# tokens at most, so input dominates and the estimate uses it alone.
_PRICE_CHANGE = (2027, 1, 1)
_PRICE_BEFORE, _PRICE_AFTER = 0.75, 1.50


def price_per_million(today=None) -> float:
    import datetime
    today = today or datetime.date.today()
    return _PRICE_AFTER if today >= datetime.date(*_PRICE_CHANGE) else _PRICE_BEFORE


def estimate_tokens(seconds: float) -> int:
    return int(round(seconds / 60.0 * TOKENS_PER_VIDEO_MINUTE))


def cost_of(tokens: int, today=None) -> float:
    return tokens / 1_000_000 * price_per_million(today)


def dollars(amount: float) -> str:
    return f"${amount:.2f}" if amount >= 0.995 else f"{max(amount * 100, 0.1):.1f}¢"


def youtube_id(source: str) -> str | None:
    """The 11-character video id of a YouTube link, or None if it isn't one."""
    try:
        parsed = urlparse((source or "").strip())
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = parsed.netloc.lower().split(":")[0]
    path = parsed.path.strip("/").split("/")
    candidate = None
    if host in ("youtu.be", "www.youtu.be"):
        candidate = path[0] if path else None
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        if path and path[0] == "watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        elif len(path) >= 2 and path[0] in ("shorts", "live", "embed"):
            candidate = path[1]
    return candidate if candidate and _YOUTUBE_ID.match(candidate) else None


def youtube_uri(source: str) -> str | None:
    """Canonical watch URL for a YouTube link — the form the API documents."""
    vid = youtube_id(source)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None


def build_prompt(question: str | None) -> str:
    return ((question or "").strip() or DEFAULT_QUESTION) + PROMPT_SUFFIX


class GeminiError(RuntimeError):
    """A categorized, key-free failure. str() is safe to print."""

    def __init__(self, category: str, detail: str):
        self.category = category
        super().__init__(f"Gemini {category}: {_HINTS[category]}. Detail: {detail or 'none'}")


def _scrub(text: str, key: str | None) -> str:
    text = " ".join(str(text).split())[:600]
    return text.replace(key, "[redacted]") if key else text


def _error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        if isinstance(data, list) and data:  # Some errors arrive wrapped in a one-item list.
            data = data[0]
        return str(data["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return text


def _call(method: str, url: str, key: str, *, data=None, headers=None, timeout: float = 60.0):
    request = Request(url, data=data, method=method,
                      headers={"x-goog-api-key": key, **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        message = _error_message(exc.read())
        # A malformed key comes back as 400 "API key not valid", not 401.
        if exc.code in (401, 403) or "api key" in message.lower():
            category = "auth"
        elif exc.code == 429:
            category = "quota"
        elif exc.code >= 500:
            category = "service"
        else:
            category = "rejected"
        raise GeminiError(category, _scrub(f"HTTP {exc.code}: {message}", key)) from None
    except (URLError, TimeoutError, OSError) as exc:
        detail = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
        raise GeminiError("network", _scrub(detail, key)) from None


def _clock(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def processing_for(start: float | None, end: float | None):
    """(processing config, plain label). Agentic for the whole video.

    The API takes clip offsets only in static mode, so a --start/--end run
    trades the agentic search for a fixed window. The label says which.
    """
    if start is None and end is None:
        return "agentic", "agentic, whole video"
    config: dict = {"type": "static"}
    if start is not None:
        config["start_offset"] = f"{int(math.floor(start))}s"
    if end is not None:
        config["end_offset"] = f"{int(math.ceil(end))}s"
    tail = _clock(int(math.ceil(end))) if end is not None else "end"
    return config, f"static clip {_clock(int(math.floor(start or 0)))}–{tail}"


def ask(video: dict, question: str | None, *, model: str, key: str,
        start: float | None = None, end: float | None = None,
        timeout: float = 600.0) -> dict:
    """Send one video reference and the question; return the answer text."""
    processing, label = processing_for(start, end)
    payload = {
        "model": model,
        "input": [
            {"type": "video", **video, "processing": processing},
            {"type": "text", "text": build_prompt(question)},
        ],
    }
    _, _, body = _call("POST", INTERACTIONS, key, data=json.dumps(payload).encode("utf-8"),
                       headers={"Content-Type": "application/json"}, timeout=timeout)
    try:
        data = json.loads(body)
        parts = [
            part["text"]
            for step in data["steps"] if step.get("type") == "model_output"
            for part in step.get("content", [])
            if part.get("type") == "text" and part.get("text")
        ]
        tokens = (data.get("usage") or {}).get("total_tokens")
    except (ValueError, KeyError, TypeError, AttributeError):
        raise GeminiError("response", _scrub(body[:300].decode("utf-8", errors="replace"), key)) from None
    if not parts:
        raise GeminiError("response", f"no answer text (status {data.get('status')})")
    return {
        "text": "\n".join(parts).strip(),
        "model": model,
        "processing": label,
        "total_tokens": tokens if isinstance(tokens, int) else None,
    }


def upload_file(path: Path, key: str, *, poll_seconds: float = 2.0,
                max_wait: float = 900.0, sleep=time.sleep) -> dict:
    """Stream a local video to the Files API and wait until it can be used."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise GeminiError("upload", f"{path.name} is empty")
    guessed = mimetypes.guess_type(path.name)[0] or ""
    mime = guessed if guessed.startswith("video/") else "video/mp4"
    _, headers, _ = _call(
        "POST", f"{API}/upload/v1beta/files", key,
        data=json.dumps({"file": {"display_name": "watch-upload"}}).encode("utf-8"),
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        },
    )
    session = {k.lower(): v for k, v in headers.items()}.get("x-goog-upload-url")
    if not session:
        raise GeminiError("upload", "the Files API did not return an upload session URL")
    with path.open("rb") as stream:  # Streamed; never held in memory whole.
        _, _, body = _call("POST", session, key, data=stream, timeout=3600.0, headers={
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        })
    try:
        info = json.loads(body)["file"]
        waited = 0.0
        while info.get("state") == "PROCESSING":
            if waited >= max_wait:
                delete_file(info["name"], key)
                raise GeminiError("upload", f"still processing after {int(max_wait)}s")
            sleep(poll_seconds)
            waited += poll_seconds
            info = json.loads(_call("GET", f"{API}/v1beta/{info['name']}", key)[2])
        if info.get("state") != "ACTIVE":
            delete_file(info["name"], key)
            raise GeminiError("upload", f"file state {info.get('state')}")
        return {"name": info["name"], "uri": info["uri"], "mime_type": info.get("mimeType") or mime}
    except (ValueError, KeyError, TypeError):
        raise GeminiError("upload", "unreadable Files API response") from None


def delete_file(name: str, key: str) -> str | None:
    """Delete an uploaded file. Returns a warning instead of raising."""
    try:
        _call("DELETE", f"{API}/v1beta/{name}", key)
        return None
    except Exception as exc:  # Cleanup must never mask the real result or error.
        return (f"Could not delete the uploaded copy {name} from Google "
                f"({str(exc).split('. Detail:')[0]}); Google removes it on its own within 48 hours.")
