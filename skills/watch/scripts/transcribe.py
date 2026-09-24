#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path


# WebVTT makes the hours field OPTIONAL: "01:02.500" is as legal as
# "00:01:02.500", and the spec allows more than two hour digits for very long
# media. Requiring HH: parsed such a file to zero cues and the run reported "no
# transcript" -- silently, since nothing distinguishes "no captions" from "a
# caption file we could not read". YouTube always writes HH:MM:SS, so this was
# latent rather than biting.
_STAMP = r"(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3}"
TS_RE = re.compile(rf"({_STAMP})\s*-->\s*({_STAMP})")
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(stamp: str) -> float:
    """Seconds from a WebVTT/SRT timestamp, with or without an hours field."""
    parts = stamp.replace(",", ".").split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    segments: list[dict] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = _to_seconds(match.group(1))
        end = _to_seconds(match.group(2))
        i += 1

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            # Strip cue tags FIRST, then unescape -- in that order an escaped
            # "&lt;b&gt;" survives as the literal text "<b>" the caption author
            # wrote, instead of turning into a tag and being deleted. Without the
            # unescape, ">>" speaker markers arrive as "&gt;&gt;" and "&amp;" /
            # "&#39;" reach the model raw.
            cleaned = html.unescape(TAG_RE.sub("", lines[i])).strip()
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        cue_text = " ".join(cue_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        out.append(seg)
    return out


def filter_range(
    segments: list[dict],
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict]:
    """Return segments whose time range overlaps [start, end]."""
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [seg for seg in segments if seg["end"] >= lo and seg["start"] <= hi]


def format_transcript(segments: list[dict]) -> str:
    """Render segments as "[MM:SS] text", rolling over to "[H:MM:SS]".

    Without the hours rollover, minutes just kept counting: 1h10m05s rendered as
    "[70:05]", so every citation in a long video was unreadable and could not be
    matched against a frame timestamp. Same shape as frames.format_time, so the
    two halves of the report agree.
    """
    lines = []
    for seg in segments:
        total = int(seg["start"])
        hours, rem = divmod(total, 3600)
        minutes, sec = divmod(rem, 60)
        stamp = f"[{hours}:{minutes:02d}:{sec:02d}]" if hours else f"[{minutes:02d}:{sec:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
