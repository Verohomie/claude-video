#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import math
import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}

# Download quality ceiling, as a max height in pixels ("best" = no ceiling).
# 1080 rather than 720: a screen-recorded tutorial at 720p cannot be read no
# matter how the frames are extracted, and the extra bytes are cheap next to a
# wasted watch. Raise to 1440/2160 for dense UI; drop to 720 on a slow link.
DEFAULT_QUALITY = "1080"

QUALITIES = {"360", "480", "720", "1080", "1440", "2160", "best"}

# Ceiling on the sampling rate, in frames per second of video. 2.0 is a sane
# default, not a law: auto-fps targets a frame BUDGET, so this only ever binds an
# explicit --fps. It exists for sub-second and fast-action clips, where a budget
# spread over the whole video yields one or two frames of the moment that
# mattered. Raise it for a short focused range, not for a whole film — cost is
# the real ceiling, at roughly 200 tokens per 512px frame.
#   WATCH_MAX_FPS=24 with --start/--end and --max-frames on a few seconds.
DEFAULT_MAX_FPS = 2.0

# Which engine watches the video. "local" is frames + transcript read by the
# agent itself. "gemini" hands the video to Google's model and relays its
# answer. "auto" uses Gemini only for YouTube links, and only when a key is set:
# a YouTube video is already public, while a local file or another site's video
# may be private — those leave the machine only on an explicit --engine gemini.
DEFAULT_ENGINE = "auto"
ENGINES = {"auto", "gemini", "local"}
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_GEMINI_TIMEOUT = 600.0
# Gemini runs longer than this many minutes of video are priced and held for
# the user's go-ahead (--confirm-cost) instead of being sent straight away.
DEFAULT_GEMINI_CONFIRM_MINUTES = 30.0


def read_env_file(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = CONFIG_FILE
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        else:
            # Strip an inline comment (a '#' preceded by whitespace) from an
            # unquoted value. Without this, `WATCH_DETAIL=balanced  # note`
            # parses as "balanced  # note", fails validation, and silently
            # falls back to the default. Keeps '#' inside quotes / API keys.
            for i, ch in enumerate(value):
                if ch == "#" and i > 0 and value[i - 1] in " \t":
                    value = value[:i].rstrip()
                    break
        values[key.strip()] = value
    return values


def get_config() -> dict[str, object]:
    file_values = read_env_file()

    detail = (
        os.environ.get("WATCH_DETAIL")
        or file_values.get("WATCH_DETAIL")
        or DEFAULT_DETAIL
    )
    if detail not in DETAILS:
        detail = DEFAULT_DETAIL

    quality = (
        os.environ.get("WATCH_QUALITY")
        or file_values.get("WATCH_QUALITY")
        or DEFAULT_QUALITY
    )
    quality = str(quality).strip().lower().rstrip("p")
    if quality not in QUALITIES:
        quality = DEFAULT_QUALITY

    raw_max_fps = (
        os.environ.get("WATCH_MAX_FPS")
        or file_values.get("WATCH_MAX_FPS")
        or DEFAULT_MAX_FPS
    )
    # Same contract as detail and quality: an unusable value falls back to the
    # default rather than raising, so a typo in .env cannot take down a run that
    # has already paid for a download.
    try:
        max_fps = float(raw_max_fps)
    except (TypeError, ValueError):
        max_fps = DEFAULT_MAX_FPS
    # isfinite as well as > 0: "1e400" and "inf" both parse, and both compare
    # greater than zero, which would leave --fps effectively unclamped.
    if not (math.isfinite(max_fps) and max_fps > 0):
        max_fps = DEFAULT_MAX_FPS

    engine = str(
        os.environ.get("WATCH_ENGINE")
        or file_values.get("WATCH_ENGINE")
        or DEFAULT_ENGINE
    ).strip().lower()
    if engine not in ENGINES:
        engine = DEFAULT_ENGINE

    gemini_model = (
        os.environ.get("WATCH_GEMINI_MODEL")
        or file_values.get("WATCH_GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip() or DEFAULT_GEMINI_MODEL

    try:
        gemini_timeout = float(
            os.environ.get("WATCH_GEMINI_TIMEOUT")
            or file_values.get("WATCH_GEMINI_TIMEOUT")
            or DEFAULT_GEMINI_TIMEOUT
        )
    except (TypeError, ValueError):
        gemini_timeout = DEFAULT_GEMINI_TIMEOUT
    if not (math.isfinite(gemini_timeout) and gemini_timeout > 0):
        gemini_timeout = DEFAULT_GEMINI_TIMEOUT

    try:
        confirm_minutes = float(
            os.environ.get("WATCH_GEMINI_CONFIRM_MINUTES")
            or file_values.get("WATCH_GEMINI_CONFIRM_MINUTES")
            or DEFAULT_GEMINI_CONFIRM_MINUTES
        )
    except (TypeError, ValueError):
        confirm_minutes = DEFAULT_GEMINI_CONFIRM_MINUTES
    if not (math.isfinite(confirm_minutes) and confirm_minutes >= 0):
        confirm_minutes = DEFAULT_GEMINI_CONFIRM_MINUTES

    return {
        "detail": detail,
        "quality": quality,
        "max_fps": max_fps,
        "engine": engine,
        "gemini_model": gemini_model,
        "gemini_timeout": gemini_timeout,
        "gemini_confirm_minutes": confirm_minutes,
        "config_file": str(CONFIG_FILE),
    }


def load_gemini_key() -> str | None:
    """GEMINI_API_KEY from the environment, else the config file; None if unset."""
    key = os.environ.get("GEMINI_API_KEY") or read_env_file().get("GEMINI_API_KEY")
    key = (key or "").strip()
    return key or None


def resolve_engine(requested: str, *, has_key: bool, youtube: bool) -> str:
    """Decide which engine runs: "gemini" or "local".

    An explicit "gemini" without a key is an error, not a quiet switch to local:
    the user asked for something specific and should hear why it can't happen.
    """
    if requested == "local":
        return "local"
    if requested == "gemini":
        if not has_key:
            raise SystemExit(
                "--engine gemini needs GEMINI_API_KEY in ~/.config/watch/.env "
                "(create one at https://aistudio.google.com/apikey), "
                "or rerun with --engine local."
            )
        return "gemini"
    return "gemini" if (has_key and youtube) else "local"


def frame_cap(detail: str) -> int | None:
    if detail == "efficient":
        return 50
    if detail == "balanced":
        return 100
    if detail == "token-burner":
        return None
    if detail == "transcript":
        return None
    return 100
