#!/usr/bin/env python3
"""Probe video metadata and extract frames at an auto-scaled fps.

Auto-fps targets a frame budget, not a fixed rate. Token cost scales with frame
count, so budget-by-duration keeps short videos dense and long videos capped.
When a user-specified range is passed, focused-mode budgets denser (they are
zooming in for detail).
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


MAX_FPS = 2.0
SCENE_THRESHOLD = 0.20
# Keep scene-detection results once we have at least this many distinct shots.
# Below this the video is effectively static (screen recording, talking head),
# so we fall back to uniform sampling. Matching the reference fork's behaviour,
# this is a low floor — NOT the frame budget — so normal videos with cuts use
# the (single-pass) scene engine instead of paying for a wasted second decode.
SCENE_MIN_FRAMES = 8
# Below this many decoded keyframes a clip is too sparse for keyframe coverage
# (very short or oddly encoded), so the cheap tier falls back to uniform.
KEYFRAME_MIN = 4
MAX_READ_DIMENSION = 1998
# Frame-delta dedup: downscale each frame to a DEDUP_THUMB x DEDUP_THUMB
# grayscale thumbnail and treat two frames as near-identical when their mean
# per-pixel difference (0-255) is at or below DEDUP_THRESHOLD. Conservative on
# purpose: only collapses frames that are visually the same shot, so a code diff
# / scrolling terminal / slide-gaining-a-bullet survives. Unlike a within-frame
# perceptual hash, this distinguishes flat frames (solid slides, fades) by luma.
DEDUP_THUMB = 16
DEDUP_THRESHOLD = 2.0
SHOWINFO_TS_RE = re.compile(r"pts_time:([0-9.]+)")

# Frame width used when the caller doesn't pick one and the source is not a
# screen recording. Cheap, and plenty for "what happens in this video".
DEFAULT_RESOLUTION = 512
# Frame width for a detected screen recording. A 1920-wide UI squeezed to 512
# renders menu text about three pixels tall — unreadable no matter how good the
# download was. 1536 is comfortably inside MAX_READ_DIMENSION and, measured on a
# Google Cloud console capture, is where menu items and small body text become
# legible. Going higher buys little and costs tokens quadratically.
SCREEN_RESOLUTION = 1536
# A 1536px frame carries ~9x the pixels of a 512px one, so the frame cap comes
# down with it to keep the image-token bill roughly where it was. Only applied
# when the user did not set --max-frames themselves.
SCREEN_FRAME_CAP = 40
# Below this the source has no detail to preserve — _scale_filter never upscales
# (it takes min(resolution, iw)), so raising the width would change nothing.
SCREEN_MIN_WIDTH = 1280
# Screen-recording probe. Two axes, both needed; a frame counts as screen-like
# only if it passes both, and a clip is a screen recording if most sampled
# frames do.
#
#   FLATNESS, per TILE — the share of a region's pixels at one luma value. A UI
#   is mostly uniform chrome and background; camera footage is gradients. This
#   is measured on a 3x3 grid rather than the whole frame because the common
#   tutorial layout is picture-in-picture: a large flat screen capture with a
#   small busy webcam inset. Averaged whole-frame, that composite lands near the
#   boundary (measured 0.41-0.49 on the n8n masterclass) and reads as camera
#   footage — a false negative on precisely the video this exists for. Per tile
#   it is unambiguous: the screen region scores 0.72+, the inset 0.13, and 4+
#   tiles clear the bar. Measured median flat tiles out of 9:
#
#       screen : 4-7   (worst case 4, the picture-in-picture sections)
#       camera : 0-1   (TED talk 1, everything else 0)
#
#   DETAIL — mean absolute difference between horizontally adjacent pixels.
#   This is a guard, not a discriminator: a near-black frame is flat everywhere
#   and passes the tile test with 9/9, but has nothing to resolve, so raising
#   the frame width would spend tokens for no gain. A night-sky timelapse scores
#   0.23 here against 2.26+ for every screen recording measured.
#
# Motion is deliberately NOT used, though it is the intuitive choice. Adjacent
# frames are near-identical in BOTH classes once YouTube's encoder has denoised
# them (measured: 0.0-0.14 mean delta for a talking head as well as a
# screencast), so a motion test reports every video as a screen recording.
SCREEN_PROBE_WIDTH = 192
SCREEN_PROBE_HEIGHT = 108
# 32 buckets of 8 luma levels, scored over a sliding pair of buckets so a
# uniform background straddling a bucket edge still counts as one region.
SCREEN_FLAT_BUCKETS = 32
SCREEN_TILE_GRID = 3
# A tile this uniform is UI chrome rather than photographic content.
SCREEN_TILE_FLATNESS = 0.6
# Of 9 tiles. Screen recordings measured 4-7, camera footage 0-1, so 3 sits
# below every true positive and above every true negative.
SCREEN_MIN_FLAT_TILES = 3
# Below this there is no fine structure for a wider frame to preserve.
SCREEN_DETAIL_FLOOR = 1.0
# Sample points spread across the range; a MAJORITY decides. Voting per frame
# rather than averaging a score keeps a title card or a fade to black from
# swinging the result — those fail the detail guard individually and are
# outvoted anyway.
SCREEN_PROBE_POINTS = 6
SCREEN_VOTE_RATIO = 0.5


def _scale_filter(resolution: int) -> str:
    return (
        f"scale=w='min({resolution},iw)':h='min({MAX_READ_DIMENSION},ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


def _clamp_fps(fps: float, duration_seconds: float, max_frames: int) -> tuple[float, int]:
    fps = min(fps, MAX_FPS)
    target = min(max_frames, max(1, int(round(fps * duration_seconds))))
    return fps, target


def parse_time(value: str | float | int | None) -> float | None:
    """Parse SS, MM:SS, or HH:MM:SS (with optional .ms) into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise SystemExit(f"Cannot parse time value: {value!r} (expected SS, MM:SS, or HH:MM:SS)")


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def get_metadata(video_path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(Path(video_path).resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
        "size_bytes": int(fmt.get("size") or 0),
        "has_audio": audio_stream is not None,
    }


def auto_fps(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Pick fps that targets a sensible frame budget for full-video scans."""
    if duration_seconds <= 0:
        return 1.0, 1

    if duration_seconds <= 30:
        target = min(max_frames, max(12, int(round(duration_seconds))))
    elif duration_seconds <= 60:
        target = min(max_frames, 40)
    elif duration_seconds <= 180:  # 3 min
        target = min(max_frames, 60)
    elif duration_seconds <= 600:  # 10 min
        target = min(max_frames, 80)
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def auto_fps_focus(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Denser budget for user-specified ranges — they are zooming in for detail."""
    if duration_seconds <= 0:
        return min(MAX_FPS, 2.0), 2

    if duration_seconds <= 5:
        target = min(max_frames, max(10, int(round(duration_seconds * 6))))
    elif duration_seconds <= 15:
        target = min(max_frames, max(30, int(round(duration_seconds * 4))))
    elif duration_seconds <= 30:
        target = min(max_frames, 60)
    elif duration_seconds <= 60:
        target = min(max_frames, 80)
    elif duration_seconds <= 180:
        target = max_frames
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def _frame_flatness(frame: bytes) -> float:
    """Share of pixels in the most populated pair of adjacent luma buckets.

    1.0 means a single flat colour fills the frame; a photographic image spreads
    across the histogram and lands near zero.
    """
    if not frame:
        return 0.0
    hist = [0] * SCREEN_FLAT_BUCKETS
    shift = 8 - SCREEN_FLAT_BUCKETS.bit_length() + 1
    for pixel in frame:
        hist[pixel >> shift] += 1
    best = max(hist[i] + hist[i + 1] for i in range(SCREEN_FLAT_BUCKETS - 1))
    return best / len(frame)


def _frame_detail(frame: bytes) -> float:
    """Mean absolute difference between horizontally adjacent pixels.

    A proxy for how much fine structure a frame carries. Near zero for a frame
    with nothing in it (night sky, fade to black), which is the case the flatness
    test alone cannot tell apart from a blank UI.
    """
    total = 0
    count = 0
    for y in range(SCREEN_PROBE_HEIGHT):
        row = frame[y * SCREEN_PROBE_WIDTH:(y + 1) * SCREEN_PROBE_WIDTH]
        total += sum(abs(a - b) for a, b in zip(row, row[1:]))
        count += len(row) - 1
    return total / count if count else 0.0


def _flat_tile_count(frame: bytes) -> int:
    """How many of the SCREEN_TILE_GRID^2 tiles are uniform enough to be UI."""
    tile_w = SCREEN_PROBE_WIDTH // SCREEN_TILE_GRID
    tile_h = SCREEN_PROBE_HEIGHT // SCREEN_TILE_GRID
    count = 0
    for gy in range(SCREEN_TILE_GRID):
        for gx in range(SCREEN_TILE_GRID):
            tile = bytearray()
            for y in range(gy * tile_h, (gy + 1) * tile_h):
                start = y * SCREEN_PROBE_WIDTH + gx * tile_w
                tile += frame[start:start + tile_w]
            if _frame_flatness(bytes(tile)) >= SCREEN_TILE_FLATNESS:
                count += 1
    return count


def _frame_is_screenlike(frame: bytes) -> tuple[bool, float, int]:
    """Judge one frame. Returns ``(verdict, detail, flat_tiles)``."""
    detail = _frame_detail(frame)
    tiles = _flat_tile_count(frame)
    return (detail >= SCREEN_DETAIL_FLOOR and tiles >= SCREEN_MIN_FLAT_TILES), detail, tiles


def _probe_frame(video_path: str, at_seconds: float) -> bytes | None:
    """Decode one frame at ``at_seconds`` as raw grayscale, or None on failure.

    Fast-seek single-frame decode, downscaled by ffmpeg so we stay pure-stdlib.
    The scale is deliberately not aspect-preserving: neither a luma histogram nor
    a neighbour-difference cares about geometry, and a fixed size keeps the byte
    count predictable.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(0.0, at_seconds):.3f}",
        "-i", str(Path(video_path).resolve()),
        "-vf", f"scale={SCREEN_PROBE_WIDTH}:{SCREEN_PROBE_HEIGHT},format=gray",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    expected = SCREEN_PROBE_WIDTH * SCREEN_PROBE_HEIGHT
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return result.stdout[:expected]


def detect_screen_recording(
    video_path: str,
    meta: dict,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[bool, dict]:
    """Decide whether the source is a screen capture rather than camera footage.

    Measured, not guessed: sample frames across the range, judge each on tiled
    flatness plus a detail guard, and take a majority vote (see the SCREEN_*
    constants for the numbers this is calibrated against). Aspect ratio is
    deliberately not part of the test — a 4:3 screencast is still a screencast,
    and 16:9 is just as true of every talking head, so it would only add false
    negatives.

    Fail-open: any probe trouble returns ``(False, ...)`` and the caller keeps
    the ordinary default.
    """
    width = meta.get("width") or 0
    evidence: dict = {
        "width": width,
        "height": meta.get("height"),
        "samples": 0,
        "screenlike": 0,
        "flat_tiles": None,
        "detail": None,
        "reason": "",
    }
    if width < SCREEN_MIN_WIDTH:
        evidence["reason"] = f"source is {width}px wide, below {SCREEN_MIN_WIDTH}"
        return False, evidence

    lo = start_seconds or 0.0
    hi = end_seconds if end_seconds is not None else float(meta.get("duration_seconds") or 0.0)
    span = max(0.0, hi - lo)
    if span <= 0:
        evidence["reason"] = "no measurable duration"
        return False, evidence

    # Sample strictly inside the range: the first and last moments are often a
    # title card or a fade, which are flat in every kind of video.
    points = [lo + span * (i + 1) / (SCREEN_PROBE_POINTS + 1) for i in range(SCREEN_PROBE_POINTS)]
    judged = [
        _frame_is_screenlike(frame)
        for frame in (_probe_frame(video_path, p) for p in points)
        if frame is not None
    ]
    if not judged:
        evidence["reason"] = "frame probe returned nothing"
        return False, evidence

    votes = sum(1 for verdict, _, _ in judged if verdict)
    evidence["samples"] = len(judged)
    evidence["screenlike"] = votes
    evidence["flat_tiles"] = statistics.median([tiles for _, _, tiles in judged])
    evidence["detail"] = round(statistics.median([d for _, d, _ in judged]), 2)
    detected = votes / len(judged) >= SCREEN_VOTE_RATIO
    evidence["reason"] = (
        f"{votes}/{len(judged)} sampled frames look like a screen — median "
        f"{evidence['flat_tiles']:g} of {SCREEN_TILE_GRID ** 2} tiles flat, "
        f"detail {evidence['detail']:g}"
    )
    return detected, evidence


def extract(
    video_path: str,
    out_dir: Path,
    fps: float,
    resolution: int = DEFAULT_RESOLUTION,
    max_frames: int = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[dict]:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
    ]

    # -ss before -i = fast seek (keyframe-snap, good enough for preview frames).
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]

    cmd += [
        "-i", str(Path(video_path).resolve()),
        "-vf", f"fps={fps},{_scale_filter(resolution)}",
        "-frames:v", str(max_frames),
        "-q:v", "4",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    offset = start_seconds or 0.0
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [
        {
            "index": i,
            "timestamp_seconds": round(offset + (i / fps if fps > 0 else 0.0), 2),
            "path": str(p),
            "reason": "uniform",
        }
        for i, p in enumerate(frames)
    ]


def extract_scene_candidates(
    video_path: str,
    out_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    max_frames: int | None = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    threshold: float = SCENE_THRESHOLD,
) -> list[dict]:
    """Extract first frame plus ffmpeg scene-change frames.

    When ``max_frames`` is set, ``-frames:v`` lets ffmpeg stop decoding once it
    has emitted that many frames (early exit) and avoids writing extras that we
    would only delete afterwards. ``None`` (uncapped "complete" detail) keeps
    every detected shot, as the user explicitly opted in.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-y",
    ]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]

    vf = f"select='eq(n\\,0)+gt(scene\\,{threshold})',{_scale_filter(resolution)},showinfo"
    cmd += [
        "-i", str(Path(video_path).resolve()),
        "-vf", vf,
        "-vsync", "vfr",
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [
        "-q:v", "4",
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg scene extraction failed: {result.stderr.strip()}")

    offset = start_seconds or 0.0
    timestamps = [round(offset + float(match.group(1)), 2) for match in SHOWINFO_TS_RE.finditer(result.stderr)]
    frames = sorted(out_dir.glob("frame_*.jpg"))
    out: list[dict] = []
    for i, path in enumerate(frames):
        ts = timestamps[i] if i < len(timestamps) else offset
        out.append({
            "index": i,
            "timestamp_seconds": ts,
            "path": str(path),
            "reason": "first-frame" if i == 0 else "scene-change",
        })
    return out


def _even_indices(count: int, n: int) -> list[int]:
    """Indices of ``n`` evenly-spaced items out of ``count`` (first + last kept).

    ``n >= count`` returns every index; ``n == 1`` returns just the first.
    """
    if n >= count:
        return list(range(count))
    if n <= 1:
        return [0]
    return [round(i * (count - 1) / (n - 1)) for i in range(n)]


def parse_timestamps(value: str | None) -> list[float]:
    """Parse a comma-separated list of times (SS, MM:SS, HH:MM:SS) into a
    sorted, de-duplicated list of seconds. Empty/blank tokens are skipped;
    an unparseable token raises (via :func:`parse_time`)."""
    if not value:
        return []
    out: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seconds = parse_time(token)
        if seconds is not None:
            out.append(float(seconds))
    return sorted(set(out))


def merge_frames(primary: list[dict], pinned: list[dict]) -> list[dict]:
    """Combine two frame lists into one chronological list and reindex 0..n-1.

    ``pinned`` frames (transcript cues) are never dropped — this is a plain
    union, so the cap is enforced upstream by reserving budget for the cues.
    """
    merged = sorted([*primary, *pinned], key=lambda f: f["timestamp_seconds"])
    for i, frame in enumerate(merged):
        frame["index"] = i
    return merged


def extract_at_timestamps(
    video_path: str,
    out_dir: Path,
    timestamps: list[float],
    resolution: int = DEFAULT_RESOLUTION,
    max_frames: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[list[dict], dict]:
    """Grab exactly one frame at each requested timestamp (transcript cues).

    Timestamps are absolute source seconds. Any falling outside an active
    ``[start, end]`` focus window are dropped. Files use a ``cue_*.jpg`` prefix
    so they sit alongside detail-engine ``frame_*.jpg`` output without either
    clobbering the other. When more cues than ``max_frames`` survive, they are
    even-sampled (first + last kept) before extraction.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("cue_*.jpg"):
        existing.unlink()

    lo = start_seconds or 0.0
    hi = end_seconds if end_seconds is not None else float("inf")
    requested = sorted(set(round(float(t), 2) for t in timestamps))
    in_window = [t for t in requested if lo <= t <= hi]
    dropped = len(requested) - len(in_window)

    if max_frames is not None and len(in_window) > max_frames:
        points = [in_window[i] for i in _even_indices(len(in_window), max_frames)]
    else:
        points = in_window

    out: list[dict] = []
    for t in points:
        path = out_dir / f"cue_{len(out):04d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{t:.3f}",
            "-i", str(Path(video_path).resolve()),
            "-frames:v", "1",
            "-vf", _scale_filter(resolution),
            "-q:v", "4",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and path.exists():
            out.append({
                "index": len(out),
                "timestamp_seconds": t,
                "path": str(path),
                "reason": "transcript-cue",
            })

    meta = {
        "engine": "timestamps",
        "candidate_count": len(requested),
        "selected_count": len(out),
        "dropped_out_of_window": dropped,
        "fallback": False,
    }
    return out, meta


def _even_sample(candidates: list[dict], n: int) -> list[dict]:
    """Pick ``n`` evenly-spaced candidates (always including first and last),
    delete the JPEGs we drop, and reindex the survivors 0..len-1.

    Shared by every capped engine so all detail modes sample the same way:
    detect all candidates across the full range, then thin down to the cap.
    ``n >= len(candidates)`` keeps everything (the uncapped / under-cap case).
    """
    selected = [candidates[i] for i in _even_indices(len(candidates), n)]

    keep_paths = {sel["path"] for sel in selected}
    for cand in candidates:
        if cand["path"] not in keep_paths:
            try:
                Path(cand["path"]).unlink()
            except OSError:
                pass
    for i, frame in enumerate(selected):
        frame["index"] = i
    return selected


def _frame_delta(a: bytes, b: bytes) -> float:
    """Mean absolute per-pixel difference (0-255) between two grayscale
    thumbnails. Mismatched lengths are treated as maximally different so a
    decode hiccup never collapses distinct frames."""
    if not a or len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _thumb_frames(paths: list[Path]) -> list[bytes]:
    """Decode every frame in ``paths`` to a small grayscale thumbnail via one
    ffmpeg pass over the JPEG sequence.

    ffmpeg does the pixel decode (keeps us pure-stdlib); we slice the raw
    grayscale stream into one ``DEDUP_THUMB``-square thumbnail per frame.
    Fail-open: any ffmpeg error, an unrecognized name, or a byte-count mismatch
    returns ``[]`` so the caller skips dedup rather than breaking extraction.
    """
    if not paths:
        return []
    paths = [Path(p) for p in paths]
    m = re.match(r"(.*?)(\d+)(\.[A-Za-z0-9]+)$", paths[0].name)
    if m is None:
        return []
    prefix, digits, ext = m.group(1), m.group(2), m.group(3)
    pattern = str(paths[0].parent / f"{prefix}%0{len(digits)}d{ext}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-start_number", str(int(digits)),
        "-i", pattern,
        "-vf", f"scale={DEDUP_THUMB}:{DEDUP_THUMB},format=gray",
        "-f", "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return []

    chunk = DEDUP_THUMB * DEDUP_THUMB
    data = result.stdout
    if len(data) != chunk * len(paths):
        return []
    return [data[i * chunk:(i + 1) * chunk] for i in range(len(paths))]


def dedupe_perceptual(
    candidates: list[dict], threshold: float = DEDUP_THRESHOLD
) -> tuple[list[dict], int]:
    """Drop near-identical frames from a chronological candidate list.

    Thumbnails the extracted JPEGs and greedily removes frames whose mean
    per-pixel difference from the last kept one is within ``threshold``. Returns
    ``(survivors, dropped_count)``; a no-op (unchanged list) when thumbnails are
    unavailable or there are fewer than two candidates.
    """
    if len(candidates) <= 1:
        return candidates, 0
    thumbs = _thumb_frames([Path(c["path"]) for c in candidates])
    return _dedupe_by_deltas(candidates, thumbs, threshold)


def _dedupe_by_deltas(
    candidates: list[dict], thumbs: list[bytes], threshold: float = DEDUP_THRESHOLD
) -> tuple[list[dict], int]:
    """Greedily drop frames within ``threshold`` mean per-pixel difference of the
    last *kept* frame. Deletes dropped JPEGs and reindexes survivors 0..n-1 (same
    cleanup contract as :func:`_even_sample`). Fail-open: if ``thumbs`` does not
    line up 1:1 with ``candidates``, return them unchanged.
    """
    if len(thumbs) != len(candidates) or len(candidates) <= 1:
        return candidates, 0

    kept = [candidates[0]]
    last = thumbs[0]
    dropped: list[dict] = []
    for cand, thumb in zip(candidates[1:], thumbs[1:]):
        if _frame_delta(thumb, last) <= threshold:
            dropped.append(cand)
        else:
            kept.append(cand)
            last = thumb

    for cand in dropped:
        try:
            Path(cand["path"]).unlink()
        except OSError:
            pass
    for i, frame in enumerate(kept):
        frame["index"] = i
    return kept, len(dropped)


def extract_scene_or_uniform(
    video_path: str,
    out_dir: Path,
    fps: float,
    target_frames: int,
    resolution: int = DEFAULT_RESOLUTION,
    max_frames: int | None = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    dedup: bool = True,
) -> tuple[list[dict], dict]:
    """Prefer scene selection, falling back to uniform only when the video is
    effectively static (fewer than ``SCENE_MIN_FRAMES`` detected shots).

    Scene cuts are detected across the *whole* range (uncapped), near-identical
    frames are dropped (:func:`dedupe_perceptual`, unless ``dedup`` is False),
    and the survivors are even-sampled down to ``max_frames`` via
    :func:`_even_sample`, exactly like the keyframe engine. This costs a full
    decode, but it guarantees coverage spans the entire clip — capping detection
    with ``-frames:v`` instead would keep only the first ``max_frames`` cuts and
    drop the tail of long videos (and could even fall below ``SCENE_MIN_FRAMES``
    and misfire the uniform fallback on a cut-heavy clip).
    """
    scene_frames = extract_scene_candidates(
        video_path,
        out_dir,
        resolution=resolution,
        max_frames=None,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    scene_count = len(scene_frames)
    if scene_count >= SCENE_MIN_FRAMES:
        deduped, n_dropped = dedupe_perceptual(scene_frames) if dedup else (scene_frames, 0)
        cap = len(deduped) if max_frames is None else max_frames
        selected = _even_sample(deduped, cap)
        return selected, {
            "engine": "scene",
            "candidate_count": scene_count,
            "deduped_count": n_dropped,
            "selected_count": len(selected),
            "fallback": False,
        }

    fallback_cap = target_frames if max_frames is None else min(max_frames, target_frames)
    frames = extract(
        video_path,
        out_dir,
        fps=fps,
        resolution=resolution,
        max_frames=fallback_cap,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    n_dropped = 0
    if dedup:
        frames, n_dropped = dedupe_perceptual(frames)
    return frames, {
        "engine": "uniform",
        "candidate_count": scene_count,
        "deduped_count": n_dropped,
        "selected_count": len(frames),
        "fallback": True,
    }


def extract_keyframes(
    video_path: str,
    out_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    max_frames: int | None = 50,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    dedup: bool = True,
) -> tuple[list[dict], dict]:
    """Decode only keyframes (I-frames) — the cheap, near-instant tier.

    ``-skip_frame nokey`` makes ffmpeg reconstruct only keyframes, skipping all
    P/B frames. Encoders emit keyframes at scene cuts, so these already
    approximate "distinct moments". Near-identical frames are dropped
    (:func:`dedupe_perceptual`, unless ``dedup`` is False); over-cap →
    even-sample first→last; too few keyframes → uniform fallback.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-y",
    ]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]
    cmd += [
        "-skip_frame", "nokey",
        "-i", str(Path(video_path).resolve()),
        "-vf", f"{_scale_filter(resolution)},showinfo",
        "-vsync", "vfr",
        "-q:v", "4",
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg keyframe extraction failed: {result.stderr.strip()}")

    offset = start_seconds or 0.0
    timestamps = [round(offset + float(m.group(1)), 2) for m in SHOWINFO_TS_RE.finditer(result.stderr)]
    files = sorted(out_dir.glob("frame_*.jpg"))
    candidates: list[dict] = []
    for i, path in enumerate(files):
        ts = timestamps[i] if i < len(timestamps) else offset
        candidates.append({
            "index": i,
            "timestamp_seconds": ts,
            "path": str(path),
            "reason": "keyframe",
        })

    # Too few keyframes → uniform fallback over the same range.
    if len(candidates) < KEYFRAME_MIN:
        for cand in candidates:
            try:
                Path(cand["path"]).unlink()
            except OSError:
                pass
        meta = get_metadata(video_path)
        full_duration = meta["duration_seconds"]
        eff_start = start_seconds or 0.0
        eff_end = end_seconds if end_seconds is not None else full_duration
        eff_duration = max(0.0, eff_end - eff_start)
        budget = max_frames if max_frames is not None else 100
        fps, _ = auto_fps(eff_duration, max_frames=budget)
        frames_out = extract(
            video_path,
            out_dir,
            fps=fps,
            resolution=resolution,
            max_frames=budget,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        n_dropped = 0
        if dedup:
            frames_out, n_dropped = dedupe_perceptual(frames_out)
        return frames_out, {
            "engine": "uniform",
            "candidate_count": len(candidates),
            "deduped_count": n_dropped,
            "selected_count": len(frames_out),
            "fallback": True,
        }

    # Detect-all, drop near-duplicates, then even-sample down to the cap (first +
    # last always kept). ``max_frames is None`` (uncapped) keeps every keyframe.
    candidate_count = len(candidates)
    deduped, n_dropped = dedupe_perceptual(candidates) if dedup else (candidates, 0)
    cap = len(deduped) if max_frames is None else max_frames
    selected = _even_sample(deduped, cap)
    return selected, {
        "engine": "keyframe",
        "candidate_count": candidate_count,
        "deduped_count": n_dropped,
        "selected_count": len(selected),
        "fallback": False,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: frames.py <video-path> <out-dir> [--fps F] [--resolution W] "
            "[--max-frames N] [--start T] [--end T] [--no-dedup]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    out = Path(sys.argv[2])
    args = sys.argv[3:]

    fps_override = None
    resolution = DEFAULT_RESOLUTION
    max_frames = 100
    start_arg = None
    end_arg = None
    dedup = True
    i = 0
    while i < len(args):
        if args[i] == "--fps":
            fps_override = float(args[i + 1]); i += 2
        elif args[i] == "--resolution":
            resolution = int(args[i + 1]); i += 2
        elif args[i] == "--max-frames":
            max_frames = int(args[i + 1]); i += 2
        elif args[i] == "--start":
            start_arg = args[i + 1]; i += 2
        elif args[i] == "--end":
            end_arg = args[i + 1]; i += 2
        elif args[i] == "--no-dedup":
            dedup = False; i += 1
        else:
            i += 1

    meta = get_metadata(video)
    start_sec = parse_time(start_arg)
    end_sec = parse_time(end_arg)
    full_duration = meta["duration_seconds"]

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)

    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if fps_override is not None:
        fps = fps_override
        target = max(1, int(round(fps * effective_duration)))

    frames = extract(
        video, out,
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )
    deduped_count = 0
    if dedup:
        frames, deduped_count = dedupe_perceptual(frames)
    print(json.dumps(
        {
            "meta": meta, "fps": fps, "target": target, "focused": focused,
            "deduped_count": deduped_count, "frames": frames,
        },
        indent=2,
    ))
